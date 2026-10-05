"""Run discovery, the shared HPOTree, and the two-level cache.

**Level 1, the RunReport, on disk.** A small JSON blob keyed by ``(run_id, newest artifact mtime,
schema version)``: cohort size, model roster, the k-sweep, the gate results. Restarting the app
costs nothing, and a rerun of the experiment invalidates its own cache.

**Level 2, the bundle, in process.** The detection tensor, every raw generation, the PhenoBERT
table, the vote masks. An LRU of a few cohorts, because building one takes seconds and switching
between hcy and gsc must not take twice that.

The ``HPOTree`` is built once per process and shared with ``app/tree_ui``, its constructor
parses a 15.5 MB JSON and normalises every phrase in the ontology, which is not something to do per
callback, per app, or per cohort.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict

from . import detections as det
from . import frame as frame_mod
from . import loaders, memo, verify, votes
from hpo_extraction.curation import phenobert_output as pbstandalone

logger = logging.getLogger(__name__)

#: Bump when the RunReport shape changes so stale disk caches are ignored rather than misread.
SCHEMA_VERSION = 3

_BUNDLE_LRU_SIZE = 4

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "phenorag_exp13_06_ui")


def get_tree():
    """The shared ontology. Delegates to ``app/tree_ui`` so both apps pay the parse once."""
    from apps.ui_common.registry import get_tree as _get_tree

    return _get_tree()


def _jsonable(obj):
    """Coerce numpy/pandas scalars so a report survives a JSON round-trip."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int, float)):
        return obj
    for attr in ("item", "tolist"):
        if hasattr(obj, attr):
            try:
                return _jsonable(getattr(obj, attr)())
            except Exception:  # noqa: BLE001 - fall through to str()
                break
    return str(obj)


class Registry:
    """Everything the views ask questions of. One instance, swapped when the paths change."""

    def __init__(self, output_base: str, cache_dir: str | None = None,
                 pb_base: str | None = None, frame_base: str | None = None,
                 annotations_dir: str | None = None):
        self.output_base = output_base
        #: Where to find the PhenoBERT baseline PhenoBERT baseline. Blank means "derive it from each cohort's
        #: run directory", which is what the cluster layout makes possible and what almost every
        #: session wants. A typed path is honoured, see ``pbstandalone.find_run``.
        self.pb_base = (pb_base or "").strip()
        #: Where to find the deep-dive sampling frame, on the same "derive, or honour" rule.
        self.frame_base = (frame_base or "").strip()
        #: Where manual SLM annotations are recorded. Blank means "beside each run", which keeps an
        #: annotation pass with the generations it was made against.
        self.annotations_dir = (annotations_dir or "").strip()
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._bundles: OrderedDict[str, dict] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._lock_guard = threading.Lock()
        self._runs = loaders.find_runs(output_base)
        # Registry-level, not per-bundle on purpose: the frame is one file describing a draw,
        # The bundle is disk-cached on the *run directory's* mtime, and pointing at a new frame must
        # not have to invalidate a cache that is still perfectly valid.
        self.frame_path = frame_mod.find_frame(output_base, self.frame_base or None)
        self.frame = frame_mod.load(self.frame_path)
        self._search = None

    # ── discovery ────────────────────────────────────────────────────────────
    def list_runs(self) -> list[dict]:
        """The discovered runs, one dict per run."""
        return self._runs

    def run_ids(self) -> list[str]:
        """Identifiers of the discovered runs."""
        return [r["run_id"] for r in self._runs]

    def run_dir(self, run_id: str) -> str | None:
        """Folder of the run *run_id*, or None."""
        for run in self._runs:
            if run["run_id"] == run_id:
                return run["run_dir"]
        return None

    def validate(self) -> list[str]:
        """Human-readable problems with the configured paths, shown, not raised."""
        problems: list[str] = []
        if not self.output_base:
            problems.append("No output directory given.")
        elif not os.path.isdir(self.output_base):
            problems.append(f"Output directory does not exist: {self.output_base}")
        elif not self._runs:
            problems.append(
                f"No Free Listing generation run directories under {self.output_base}. Expected "
                f"{loaders.EXP_ID}/<cohort>/ holding llm_extractions_*.jsonl or detections_*.jsonl."
            )
        return problems

    # ── the disk-cached report ───────────────────────────────────────────────
    def _cache_path(self, run_id: str) -> str:
        run_dir = self.run_dir(run_id) or ""
        pb_dir = self.pb_run_dir(run_id) or ""
        key = (f"{run_dir}|{loaders.artifacts_mtime(run_dir)}|"
               f"{pb_dir}|{loaders.artifacts_mtime(pb_dir) if pb_dir else 0}|{SCHEMA_VERSION}")
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        safe = run_id.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}.{digest}.json")

    def get_report(self, run_id: str) -> dict:
        """The compact summary the sidebar and scorecard header read. Cached on disk."""
        path = self._cache_path(run_id)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("ignoring unreadable cache %s", path)

        report = build_report(self.get_bundle(run_id))
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_jsonable(report), f)
        except OSError as exc:
            logger.warning("could not write cache %s: %s", path, exc)
        return report

    # ── the in-process bundle ────────────────────────────────────────────────
    def get_bundle(self, run_id: str) -> dict:
        """The loaded data of run *run_id*, cached. Concurrent callers share one load."""
        with self._lock_guard:
            lock = self._locks.setdefault(run_id, threading.Lock())
        with lock:
            if run_id in self._bundles:
                self._bundles.move_to_end(run_id)
                return self._bundles[run_id]
            bundle = self._build_bundle(run_id)
            self._bundles[run_id] = bundle
            while len(self._bundles) > _BUNDLE_LRU_SIZE:
                self._bundles.popitem(last=False)
            return bundle

    def _build_bundle(self, run_id: str) -> dict:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            raise KeyError(f"unknown run {run_id!r}")
        logger.info("Building bundle for %s (%s)", run_id, run_dir)
        raw = loaders.load_raw(run_dir)
        run = self._run(run_id) or {}
        return build_bundle(run_id, run_dir, raw, self.get_tree_safe(),
                            pb_dir=self.pb_run_dir(run_id),
                            pb_searched=self.pb_searched(run_id),
                            cohort=run.get("cohort"),
                            exp_id=run.get("exp_id", ""),
                            prompt_key=run.get("prompt_key", ""))

    # ── the PhenoBERT baseline baseline ────────────────────────────────────────────────
    def _run(self, run_id: str) -> dict | None:
        for run in self._runs:
            if run["run_id"] == run_id:
                return run
        return None

    def _cohort(self, run_id: str) -> str | None:
        run = self._run(run_id)
        return run["cohort"] if run else None

    def pb_run_dir(self, run_id: str) -> str | None:
        """The PhenoBERT baseline run directory matching *run_id*, or ``None`` if there is none."""
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return None
        return pbstandalone.find_run(run_dir, self._cohort(run_id), self.pb_base or None)

    def pb_searched(self, run_id: str) -> list[str]:
        """Where the baseline was looked for, so a missing one names paths, not a shrug."""
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            return []
        return pbstandalone.searched_paths(run_dir, self._cohort(run_id), self.pb_base or None)

    # ── the deep-dive frame and the ontology index ───────────────────────────
    def frame_searched(self) -> list[str]:
        """Where the frame was looked for, so a missing one names paths, not a shrug."""
        return frame_mod.searched_paths(self.output_base, self.frame_base or None)

    @property
    def search(self):
        """The ontology search index, built at most once per process.

        Imported from ``app/hcy_curation_ui``, not reimplemented: that app already solved
        this, a dropdown that starts empty and is filled by a callback on ``search_value``,
        so the index stays in this process and at most 50 options cross the tunnel. Shipping all
        ~19 000 terms into a browser store is the single reason ``app/annotation_ui`` is unpleasant
        to use on the cluster, and it is not a mistake worth making twice.

        ``get_search`` keys on ``id(tree)`` and the ``HPOTree`` is the shared singleton, so this is
        one index over one ontology parse for every app in the process.

        ``None`` when the ontology could not be parsed, the term picker then says so, and every
        other screen is unaffected.
        """
        if self._search is None:
            tree = self.get_tree_safe()
            if tree is None:
                return None
            from apps.curation_ui.search import get_search

            self._search = get_search(tree)
        return self._search

    def get_tree_safe(self):
        """The ontology, or ``None`` if it cannot be parsed, the app still opens without it.

        Only the FP taxonomy and the term labels need it. A missing ``resources/util/hpo.json``
        should cost those two things, not the whole app.
        """
        try:
            return get_tree()
        except Exception as exc:  # noqa: BLE001 - degraded mode is better than no app
            logger.warning("HPOTree unavailable (%s), labels and FP taxonomy are disabled", exc)
            return None


# ──────────────────────────────────────────────────────────────────────────────
# bundle construction
# ──────────────────────────────────────────────────────────────────────────────
def build_bundle(run_id: str, run_dir: str, raw: dict, tree, pb_dir: str | None = None,
                 pb_searched: list[str] | None = None, cohort: str | None = None,
                 exp_id: str = "", prompt_key: str = "") -> dict:
    """Assemble everything the views need from one cohort's raw artifacts.

    Split out of :class:`Registry` so the selftest and the unit tests can build a bundle from a
    fixture directory without a cache directory, a Dash app or an ontology.

    *pb_dir* is the PhenoBERT baseline PhenoBERT-standalone run for the same cohort, if one was found. It is
    a second experiment's output and every panel that uses it degrades without it, so it is passed
    in, not discovered here, the caller already knows where it looked.

    *cohort* is the **dataset** (``hcy`` / ``gsc``), which is not always the run directory's name:
    an earlier prompt cell lives at ``<cohort>/<prompt_key>/``, so the basename there is the prompt.
    ``loaders.find_runs`` knows the difference and passes it. Falling back to the basename keeps the
    hand-built bundles in the tests working, where the two coincide.
    """
    models_info = raw["models"]
    # Only models with a non-empty detections slice vote, the driver's own membership rule
    # (slm_ensemble_experiment.py:576-592). Models that generated but failed to ground are kept
    # in `all_models` so the app can say *why* they are not voting.
    voters = models_info["active"]
    all_models = loaders._ordered(set(models_info["extracted"]) | set(models_info["active"]))

    sent_hpos = det.sent_hpos_from_detections(raw["detections"])
    report_hpos = det.report_hpo_counts(sent_hpos)
    built = det.build_replies(raw["records"])
    pb_rows = det.build_pb_table(raw["phenobert"], built)
    pb_index = det.index_pb(pb_rows)

    shipped_sets = {rule: data["sets"] for rule, data in raw["predictions"].items()
                    if data["sets"]}
    report_ids = _report_ids(shipped_sets, raw["records"], sent_hpos)
    gold = {rid: set(v) for rid, v in loaders.load_gold(run_dir, raw["rules"]).items()}

    wrote_any: dict[str, bool] = {rid: False for rid in report_ids}
    for (_model, report_id, _sent), reply in built["replies"].items():
        if reply["wrote"] and report_id in wrote_any:
            wrote_any[report_id] = True

    mask_cache: dict[int, dict] = {}
    sent_mask_cache: dict[int, dict] = {}

    def masks(min_count: int) -> dict:
        if min_count not in mask_cache:
            mask_cache[min_count] = det.build_masks(report_hpos, voters, report_ids, min_count)
        return mask_cache[min_count]

    def sentence_masks(min_count: int) -> dict:
        if min_count not in sent_mask_cache:
            sent_mask_cache[min_count] = det.sentence_masks(
                sent_hpos, voters, report_ids, min_count)
        return sent_mask_cache[min_count]

    bundle: dict = {
        "run_id": run_id,
        "run_dir": run_dir,
        "cohort": cohort or os.path.basename(os.path.normpath(run_dir)),
        # Which experiment produced this, and, for the earlier runs, which prompt. Empty for the Free Listing generation run, and
        # every reader treats empty as "the ensemble's own prompt".
        "exp_id": exp_id,
        "prompt_key": prompt_key,
        "models": voters,
        "all_models": all_models,
        "models_info": models_info,
        "report_ids": report_ids,
        "gold": gold,
        "records": raw["records"],
        "sent_hpos": sent_hpos,
        "report_hpos": report_hpos,
        "built": built,
        "pb_rows": pb_rows,
        "pb_index": pb_index,
        "has_phenobert": bool(pb_rows),
        "wrote_any_by_report": wrote_any,
        "sentences": det.sentences_from_replies(built, report_ids),
        "masks": masks,
        "sentence_masks": sentence_masks,
        "mask_of": lambda subset: det.mask_of(subset, voters),
        "rules": raw["rules"],
        "shipped_sets": shipped_sets,
        "agg_summary": raw["agg_summary"],
        "slm_metrics": raw["slm_metrics"],
        "timing": raw["timing"],
        # earlier only, ``None`` otherwise. ``.get`` because a hand-built ``raw`` in a test predates
        # these keys and must keep working.
        "prompt_diagnostics": raw.get("prompt_diagnostics"),
        "prompt_ranking": raw.get("prompt_ranking"),
        "tree": tree,
        # The PhenoBERT baseline baseline, or None. Read by views/compare_pb.py and the deep dive's annotated
        # report. Every other screen is unaffected by its absence.
        "pb_standalone": pbstandalone.load(pb_dir) if pb_dir else None,
        "pb_searched": list(pb_searched or ()),
        "min_count": 1,
        # Derived tables the views share, keyed by configuration. Scoped to the bundle so it is
        # evicted with the cohort and cannot outlive a rerun of the experiment. See :mod:`memo`.
        "memo": memo.new_store(),
    }

    min_count, confirmed = verify.infer_min_count(bundle) if voters else (1, False)
    bundle["min_count"] = min_count
    bundle["min_count_confirmed"] = confirmed
    bundle["default_config"] = default_config(bundle)
    bundle["gates"] = verify.run_gates(bundle) if voters else []
    bundle["gate_summary"] = verify.summarise(bundle["gates"])
    return bundle


def _report_ids(shipped_sets: dict, records: dict, sent_hpos: dict) -> list[str]:
    """The cohort, in the driver's own order where it is recoverable.

    The predictions files are written in ``report_ids`` order, so the first rule directory defines
    it. Without one (a cohort still extracting) the ids are collected from the extractions and
    sorted, which is what ``_report_ids`` in the driver does anyway.
    """
    for _rule, sets in sorted(shipped_sets.items()):
        if sets:
            return list(sets.keys())
    ids: set[str] = set()
    for rows in records.values():
        ids |= {str(r.get("patient_id", r.get("report_id", ""))) for r in rows}
    for per_report in sent_hpos.values():
        ids |= set(per_report)
    return sorted(i for i in ids if i)


def default_config(bundle: dict) -> dict:
    """The configuration a cohort opens on: the shipped configuration.

    That is every voting model, the run's own ``min_detection_count``, and the k with the best
    shipped micro F1, the number the findings quote. Opening on k=1 would show a screen nobody
    reported. Opening on the best k shows the one they did.
    """
    models = list(bundle["models"])
    n = len(models)
    best_k = 1
    rule = "vote_k"
    summary = bundle.get("agg_summary")
    if summary is not None and not summary.empty and "micro_f1" in summary:
        best = summary.loc[summary["micro_f1"].idxmax()]
        name = str(best["rule"])
        if name == "agg_plurality":
            rule = "plurality"
        elif name.startswith("vote_k"):
            best_k = int(name.removeprefix("vote_k"))
    return {
        "models": models,
        "k": max(1, min(best_k, n or 1)),
        "rule": rule,
        "min_count": bundle["min_count"],
    }


def _pb_report(bundle: dict) -> dict:
    """The JSON-serialisable summary of the PhenoBERT baseline baseline, for the sidebar."""
    pb = bundle.get("pb_standalone")
    if not pb:
        return {"loaded": False, "searched": bundle.get("pb_searched", [])}
    return {
        "loaded": True,
        "run_dir": pb["run_dir"],
        "n_reports": len(pb["report_ids"]),
        "n_detections": len(pb["detections"]),
        "n_negated": sum(1 for d in pb["detections"] if d["negated"]),
        "n_unresolved": sum(1 for d in pb["detections"] if not d["resolved"]),
        "n_texts": len(pb["texts"]),
        "n_span_bad": len(pb["span_bad"]),
    }


def build_report(bundle: dict) -> dict:
    """The compact, JSON-serialisable summary. Small enough to cache and to ship to the browser."""
    config = bundle["default_config"]
    sweep = votes.sweep_k(bundle, bundle["gold"], bundle["models"], bundle["min_count"]) \
        if bundle["models"] else []
    return {
        "run_id": bundle["run_id"],
        "cohort": bundle["cohort"],
        "run_dir": bundle["run_dir"],
        "n_reports": len(bundle["report_ids"]),
        "n_gold_terms": sum(len(v) for v in bundle["gold"].values()),
        "n_sentences": sum(len(v) for v in bundle["sentences"].values()),
        "models": bundle["models"],
        "all_models": bundle["all_models"],
        "models_info": bundle["models_info"],
        "has_phenobert": bundle["has_phenobert"],
        "pb_standalone": _pb_report(bundle),
        "min_count": bundle["min_count"],
        "min_count_confirmed": bundle["min_count_confirmed"],
        "rules": bundle["rules"],
        "default_config": config,
        "sweep": sweep,
        "gates": bundle["gates"],
        "gate_summary": bundle["gate_summary"],
        "timing": bundle["timing"].get("aggregate"),
    }
