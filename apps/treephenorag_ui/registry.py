"""What loads what, and what is cached where.

One ``Registry`` per process. It owns the three things that are expensive to build and must exist
once, the ``HPOTree``, the ``OntologyView`` over it, and the traversal graph, and it
owns two caches:

**A disk cache of finished reports**, keyed by a hash of the cell, the configuration, the ground truth
file and the artifacts' mtimes. A report is a few hundred kilobytes of JSON. Recomputing one on a
real cell means re-reading ``*_nodes.jsonl`` and re-walking the ontology for every false positive,
which is seconds to a minute. The mtime in the key is what makes the cache safe while a SLURM
array is still writing: a new shard changes the mtime, which changes the key.

**An in-process LRU of the heavy intermediates**, the node table. These are DataFrames, they can
be gigabytes, and they never cross a callback boundary. The raw ``*_nodes.jsonl`` frame and the
per-report state mapping are build-time inputs and are *not* in the bundle, so they
are released the moment ``_build_bundle`` returns rather than fixed for the bundle's lifetime.

**A byte-offset index per artifact**, for the two files whose size is unbounded. ``*_calls.jsonl``
is one line per retrieved sentence per visited node per report; ``*_nodes.jsonl`` is one per
visited node, which at the loosest swept tau_prune is ~200x the tightest. Indexing them means the
report deep-dive costs one report's bytes whatever the configuration, see
:meth:`Registry.report_bundle`, which is what makes a loose-tau cell openable at all.

The ``*_calls.jsonl`` scan is cached separately and *not* part of the bundle. It is
the one artifact whose size is unbounded (one line per retrieved sentence per visited node per
report) and the majority of the views do not read it, so paying for it on every cell open would
make the whole UI feel broken on a run where it happens to be 6 GB.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import pickle
from collections import OrderedDict

import pandas as pd

from . import (
    curated as curated_mod,
    discovery,
    evidence as evidence_mod,
    loaders,
    nodes as nodes_mod,
    phenobert,
    pruning,
    report as report_mod,
    sample as sample_mod,
    scoring,
    theme,
)

log = logging.getLogger(__name__)

#: Bumped whenever the shape of a cached report or bundle changes. A stale cache that still
#: deserialises is worse than one that fails to: the views would read plausible numbers computed
#: by code that no longer exists.
SCHEMA_VERSION = 2

#: How many built bundles stay resident. Two, not four: a bundle holds the node table for one
#: (cell, configuration), and at the loosest swept tau_prune that table is a million rows. The
#: light/heavy split means a single cell can already occupy both slots, so four was in practice a
#: cap of two cells' worth of the largest structure in the app.
_LRU_SIZE = 2
_CALLS_LRU_SIZE = 2
#: One report's node table is a few hundred rows, so a stepper walking a cell keeps its recent
#: reports without any of them mattering.
_REPORT_LRU_SIZE = 12
#: The nodes index is offsets only, a few hundred entries per cell, so it is cheap to keep more
#: of, and keeping it is what lets the deep-dive step between configurations without rescanning.
_NODES_LRU_SIZE = 8

# Module-level so a Registry rebuilt from new sidebar paths does not re-parse 15 MB of JSON and
# re-walk the ontology. The tree does not depend on any path the sidebar can change.
#: Distinguishes "not looked for yet" from "looked for, not there", the second is a normal state
#: that must not be retried on every callback.
_UNSET = object()

_TREE = None
_VIEW = None
_GRAPH: tuple[dict, list, dict] | None = None


def get_tree():
    """The shared ``HPOTree``, loaded once per process."""
    global _TREE
    if _TREE is None:
        from hpo_extraction.ontology.hpo_tree import HPOTree

        tree = HPOTree()
        tree.buildHPOTree()   # populates depth_dict / depth_long; __init__ does not
        _TREE = tree
    return _TREE


def get_view(tree=None):
    """The shared ``OntologyView``. Its lookups are memoised per instance, so there is one."""
    global _VIEW
    if _VIEW is None:
        from hpo_extraction.evaluation.metrics import OntologyView

        _VIEW = OntologyView(tree if tree is not None else get_tree())
    return _VIEW


def get_graph(view=None):
    """``(children_map, roots, depths)`` for the traversal graph, built once."""
    global _GRAPH
    if _GRAPH is None:
        view = view if view is not None else get_view()
        children_map, roots = pruning.children_map_from_tree(view.tree)
        _GRAPH = (children_map, roots, pruning.bfs_depths(children_map, roots))
    return _GRAPH


def set_shared(view=None, graph=None) -> None:
    """Inject a prebuilt view/graph, used by the tests, which cannot build a real ``HPOTree``.

    ``hpo_extraction.ontology.hpo_tree`` imports nltk and stanza at module scope. Where those are absent the
    whole UI is still exercisable against the ten-node toy ontology, and this is the seam that
    lets it be.
    """
    global _VIEW, _GRAPH, _TREE
    if view is not None:
        _VIEW = view
        _TREE = view.tree
    if graph is not None:
        _GRAPH = graph


class Registry:
    """Paths in, reports out."""

    def __init__(
        self,
        output_base: str,
        *,
        hcy_gt_path: str = "",
        hcy_input_dir: str = "",
        gsc_dir: str = "",
        stanza_dir: str = "",
        cache_dir: str | None = None,
        thesis_tables_dir: str = "",
        hcy_frame_dir: str = "",
        hcy_segments_path: str = "",
    ):
        self.output_base = output_base
        self.hcy_gt_path = hcy_gt_path
        self.hcy_input_dir = hcy_input_dir
        self._hcy_segments_path = hcy_segments_path
        self.gsc_dir = gsc_dir
        self.stanza_dir = stanza_dir
        self.thesis_tables_dir = thesis_tables_dir
        self.hcy_frame_dir = hcy_frame_dir
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/phenorag_exp13_ui")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._cells: dict[str, discovery.Cell] | None = None
        self._gold: dict[str, dict[str, set[str]]] = {}
        self._texts: dict[str, dict[str, str]] = {}
        self._pb: dict[str, dict | None] = {}
        self._bundles: "OrderedDict[str, dict]" = OrderedDict()
        self._calls: "OrderedDict[str, dict]" = OrderedDict()
        self._nodes_scans: "OrderedDict[str, dict]" = OrderedDict()
        self._report_bundles: "OrderedDict[str, dict]" = OrderedDict()
        self._report_ids: dict[str, list[str]] = {}
        self._sample: object = _UNSET
        self._curated: object = _UNSET
        self._evidence = None

    # ── shared heavy objects ────────────────────────────────────────────────

    @property
    def hcy_segments_path(self) -> str:
        """``segmented_reports.csv`` for the HCY cohort, explicit if given, else derived.

        Derived from the ground truth path, not configured, because it is the same cohort directory
        the curated dataset is found in and asking for it twice is asking for the two to drift: the
        curated ground truth's ``segment_idx`` is an index into *this* file, so a build's annotations and
        the segmentation they are located in must come from one place. See
        ``evidence.Evidence._from_segments_csv``.
        """
        if self._hcy_segments_path:
            return self._hcy_segments_path
        hcy_dir = curated_mod.hcy_dir_of(self.hcy_gt_path)
        return os.path.join(hcy_dir, evidence_mod.SEGMENTS_FILE) if hcy_dir else ""

    @property
    def view(self):
        """The shared ``OntologyView``."""
        return get_view()

    @property
    def graph(self):
        """The ontology graph of :attr:`view`."""
        return get_graph(self.view)

    # ── discovery ───────────────────────────────────────────────────────────

    def rescan(self) -> None:
        """Forget every discovered run and cached file, so the next access scans again."""
        self._cells = None
        self._bundles.clear()
        self._calls.clear()
        self._nodes_scans.clear()
        self._report_bundles.clear()
        self._report_ids.clear()
        self._sample = _UNSET
        self._curated = _UNSET
        self._pb.clear()
        # The segmentation too. It is keyed by cohort, not by path, so a rescan that repoints the
        # HCY ground truth at another cohort directory would otherwise keep serving the old
        # ``segmented_reports.csv``, sentences from one build under another build's ground truth.
        self._evidence = None

    @property
    def cells(self) -> dict[str, discovery.Cell]:
        """``{cell id: Cell}`` of every discovered run."""
        if self._cells is None:
            self._cells = {c.cell_id: c for c in discovery.discover(self.output_base)}
        return self._cells

    def cell(self, cell_id: str) -> discovery.Cell | None:
        """The cell *cell_id*, or None."""
        return self.cells.get(cell_id)

    def cell_options(self, *, tree_only: bool = False) -> list[dict]:
        """Dropdown options. Cells that discovered nothing usable are labelled, not hidden."""
        options = []
        for cell in self.cells.values():
            if tree_only and not cell.spec.is_tree:
                continue
            suffix = "" if cell.status == "ok" else f"  ({cell.status})"
            options.append({"label": cell.label + suffix, "value": cell.cell_id})
        return options

    def operating_points(self, cell_id: str) -> list[str]:
        """Configurations declared by the run *cell_id*."""
        cell = self.cell(cell_id)
        return list(cell.operating_points) if cell else []

    def written_operating_points(self, cell_id: str) -> list[str]:
        """Configurations whose predictions artifact holds something."""
        cell = self.cell(cell_id)
        return list(cell.written_operating_points) if cell else []

    def resolve_op(self, cell_id: str, operating_point: str | None) -> str | None:
        """Coerce a requested configuration to one this cell actually has, preferring a written one.

        Two situations, both real on the cluster. Switching methods carries a τ the new method
        never swept (``tau_0.7`` exists for noisyOR, not for the LR gate). And several τ
        directories exist but were never written, because the array was stopped before reaching
        them, on ``an earlier exploratory run``/hcy that is ``tau_0.02`` and ``tau_0.05``, which sort *first*. So
        the fallback is the first **written** point: falling back to the first point outright
        would open every one of those cells on a blank scorecard.
        """
        cell = self.cell(cell_id)
        if cell is None or not cell.operating_points:
            return None
        if operating_point in cell.operating_points:
            return operating_point
        return cell.written_operating_points[0]

    # ── ground truth and reports ────────────────────────────────────────────

    def gold(self, cohort: str) -> dict[str, set[str]]:
        """``{report_id: gold set}`` for a cohort, from the repo's own dataset loaders.

        This is the authority for every label in the UI. The ``is_gold`` flag on the node
        artifacts is *not*: it records whichever ground truth file was configured at inference time, and on
        HCY the inference jobs used ``hcy_ground_truth.csv`` while the thesis tables were computed
        against ``hcy_ground_truth_raw.csv``. Which of those is loaded here is a sidebar choice,
        it is named in every report, and the disagreement count is shown.
        """
        if cohort in self._gold:
            return self._gold[cohort]
        if cohort == "hcy":
            if not self.hcy_gt_path or not os.path.isfile(self.hcy_gt_path):
                raise FileNotFoundError(f"HCY ground truth not found: {self.hcy_gt_path!r}")
            from hpo_extraction.evaluation.datasets.hcy import HCYDataset

            raw = HCYDataset(self.hcy_gt_path, "").load_ground_truth()
        elif cohort == "gsc":
            if not self.gsc_dir or not os.path.isdir(self.gsc_dir):
                raise FileNotFoundError(f"GSC+ directory not found: {self.gsc_dir!r}")
            from hpo_extraction.evaluation.datasets.gsc import load_gsc_ground_truth

            raw = load_gsc_ground_truth(self.gsc_dir)
        else:
            raise ValueError(f"unknown cohort: {cohort!r}")

        self._gold[cohort] = {
            str(k): {str(c).strip() for c in v if c and str(c).strip()} for k, v in raw.items()
        }
        return self._gold[cohort]

    def gold_source(self, cohort: str) -> str:
        """Ground-truth location of *cohort*."""
        return self.hcy_gt_path if cohort == "hcy" else self.gsc_dir

    @property
    def curated(self):
        """The curated HCY dataset behind the loaded ground truth file, or ``None``. Memoised.

        Selected by the ground truth file itself: a ``hcy_gt_path`` inside a ``curated_ground_truth_<date>/``
        directory brings its annotation table, its report table and its manifest with it, and any
        other path brings none. There is no second control, because the failure mode
        it would create, the annotations of one dataset displayed beside the ground truth of another, is
        both easy to reach and impossible to see.

        Absence is memoised too: the raw ground truth file is the normal case on GSC+-only output folders
        and on any checkout that predates the curation pass, and re-walking the filesystem on
        every callback to rediscover that would cost the deep dive a directory scan per keystroke.
        """
        if self._curated is _UNSET:
            self._curated = curated_mod.load(curated_mod.dataset_dir_of(self.hcy_gt_path))
        return self._curated

    def curated_for(self, cohort: str):
        """:attr:`curated`, but only for the cohort it describes. ``None`` on GSC+."""
        dataset = self.curated
        return dataset if dataset is not None and dataset.applies_to(cohort) else None

    def report_texts(self, cohort: str) -> dict[str, str]:
        """``{report_id: raw text}``. Empty, not raising, the deep-dive degrades."""
        if cohort in self._texts:
            return self._texts[cohort]
        try:
            if cohort == "hcy":
                texts = loaders.load_report_texts(self.hcy_input_dir)
            elif cohort == "gsc":
                from hpo_extraction.evaluation.datasets.gsc import load_gsc_reports

                texts = {str(k): v for k, v in load_gsc_reports(self.gsc_dir).items()}
            else:
                texts = {}
        except Exception as exc:  # a missing cohort must not take the UI down
            log.warning("report texts unavailable for %s: %s", cohort, exc)
            texts = {}
        self._texts[cohort] = texts
        return texts

    def phenobert(self, cohort: str) -> dict | None:
        """The PhenoBERT baseline baseline's detections for *cohort*, for the deep dive's annotation layer.

        ``None``, not raising, and memoised including the misses: a cohort with no PhenoBERT
        run is an ordinary state, and re-walking the filesystem on every callback to rediscover
        that would cost the deep dive a directory scan per keystroke.
        """
        if cohort in self._pb:
            return self._pb[cohort]
        self._pb[cohort] = phenobert.load(phenobert.run_dir(self, cohort))
        return self._pb[cohort]

    def phenobert_source(self, cohort: str) -> str:
        """The run directory the annotations came from, or the paths that were tried."""
        bundle = self.phenobert(cohort)
        if bundle:
            return bundle["run_dir"]
        return ", ".join(phenobert.searched(self, cohort)) or "(no candidate paths)"

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Human-readable path problems. Never raises. The sidebar prints whatever comes back."""
        problems: list[str] = []
        if not os.path.isdir(self.output_base):
            problems.append(f"Output folder not found: {self.output_base}")
        elif not self.cells:
            problems.append(
                f"No TreePhenoRAG run directories under {self.output_base}, expected "
                f"<exp_id>/<cohort>/, e.g. An earlier exploratory run/hcy"
            )
        cohorts = {c.cohort for c in self.cells.values()}
        if "hcy" in cohorts and not os.path.isfile(self.hcy_gt_path):
            problems.append(f"HCY ground truth not found: {self.hcy_gt_path}")
        if "gsc" in cohorts and not os.path.isdir(self.gsc_dir):
            problems.append(f"GSC+ directory not found: {self.gsc_dir}")
        if "hcy" in cohorts and self.hcy_input_dir and not os.path.isdir(self.hcy_input_dir):
            problems.append(f"HCY report folder not found: {self.hcy_input_dir} "
                            f"(the deep-dive will show sentence indices, not text)")
        # Without it the deep-dive falls back to re-segmenting with Stanza, which is the route that
        # dies on a login node, so a missing file here is the difference between a report panel
        # and a wall of indices, and it is worth naming before the user opens one.
        segments_path = self.hcy_segments_path
        if "hcy" in cohorts and segments_path and not os.path.isfile(segments_path):
            problems.append(f"HCY segmentation not found: {segments_path}, the deep-dive will "
                            f"fall back to the Free Listing run's artifacts, then to Stanza. Write it with "
                            f"experiments/03_setup/segment_reports.py")
        # A ground truth file that looks curated but whose sidecars did not load is worth saying out loud:
        # every curated feature silently reverts to the plain-ground truth rendering, which is the same
        # screen a non-curated ground truth produces and therefore indistinguishable from working.
        directory = curated_mod.dataset_dir_of(self.hcy_gt_path)
        if "hcy" in cohorts and directory and self.curated is None:
            problems.append(f"Curated dataset at {directory} could not be read, the ground truth still "
                            f"loads, but the annotation evidence and the cohort panel are off")
        return problems

    # ── caching ─────────────────────────────────────────────────────────────

    def _cache_key(self, cell: discovery.Cell, op: str | None, extra: str = "") -> str:
        parts = [cell.cell_id, op or "-", str(SCHEMA_VERSION), self.gold_source(cell.cohort), extra]
        for kind in ("nodes", "predictions", "traversal", "node_metadata", "calls"):
            try:
                parts.append(f"{kind}:{loaders.fileset_mtime(cell.files_for(kind, op)):.0f}")
            except Exception:
                parts.append(f"{kind}:?")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    def _cache_path(self, cell: discovery.Cell, key: str, suffix: str) -> str:
        stem = cell.cell_id.replace("/", "_")
        return os.path.join(self.cache_dir, f"{stem}.{key}.{suffix}")

    def _cached_frame(self, path: str, build):
        """A DataFrame memoised to disk as a pickle, since re-deriving it means re-parsing JSONL.

        Parsing ``*_nodes.jsonl`` for a full HCY tree run is 540k lines and ten seconds, most of
        the cost of opening a cell, and the result is a function of bytes that do not change.
        The pickle is the same eight columns in binary and reloads in a fraction of a second.
        A cache that will not load is discarded and rebuilt, never trusted half-read.
        """
        if os.path.isfile(path):
            try:
                return pd.read_pickle(path)
            except Exception as exc:
                log.warning("discarding unreadable frame cache %s: %s", path, exc)
        frame = build()
        try:
            frame.to_pickle(path)
        except Exception as exc:
            log.warning("could not write frame cache %s: %s", path, exc)
        return frame

    # ── the bundle ──────────────────────────────────────────────────────────

    def bundle(self, cell_id: str, operating_point: str | None = None,
               *, heavy: bool = False, report_filter=None) -> dict | None:
        """Everything one (cell, configuration) needs, minus the calls scan. LRU-cached.

        ``heavy=True`` additionally classifies every false positive against the ontology and
        computes the calibration block. Both are seconds of work read by one tab each, so the
        default leaves them out. A heavy bundle satisfies a light request, never the reverse.

        ``report_filter`` restricts every number in the bundle to those report ids. It is how the
        deep-dive sample's opt-in "also restrict the metrics" mode works, and it is a *different*
        bundle, not a view over the full one, the ledger, the taxonomy and the calibration
        are all cohort aggregates, so there is nothing to subset after the fact. Restricted and
        unrestricted cache under different keys and neither can be served from the other;
        :data:`~apps.treephenorag_ui.sample.WARNING` is what must accompany the result on screen.

        Returns ``None`` for an unknown cell. A cell whose required artifacts are missing comes
        back with empty frames and a populated ``problems`` list, not an exception: the
        views render the problems, which is the diagnosis the user came for.
        """
        cell = self.cell(cell_id)
        if cell is None:
            return None
        op = self.resolve_op(cell_id, operating_point)
        base = f"{cell_id}|{op}|{self._cache_key(cell, op, extra=_filter_key(report_filter))}"
        heavy_key, light_key = f"{base}|heavy", f"{base}|light"
        for key in ((heavy_key,) if heavy else (heavy_key, light_key)):
            if key in self._bundles:
                self._bundles.move_to_end(key)
                return self._bundles[key]

        bundle = self._build_bundle(cell, op, heavy=heavy, report_filter=report_filter)
        self._bundles[heavy_key if heavy else light_key] = bundle
        while len(self._bundles) > _LRU_SIZE:
            # Named and deleted, not dropped on the floor: the evicted bundle holds the
            # largest frames in the process, and the node table participates in reference cycles
            # through pandas' internals, so without the collect it survives until the next
            # generational pass, which on a machine that is already near its ceiling is too late.
            _, evicted = self._bundles.popitem(last=False)
            del evicted
            gc.collect()
        return bundle

    def gold_available(self, cohort: str) -> bool:
        """Whether this cohort's ground truth source is configured and readable at all.

        Distinguishes "you did not point at GSC+" from "GSC+ is there and matches nothing", the first is a configuration gap the sidebar can fix, the second is a real disagreement
        between the artifacts and the ground truth file. The selfcheck reports them differently.
        """
        if cohort == "hcy":
            return bool(self.hcy_gt_path) and os.path.isfile(self.hcy_gt_path)
        if cohort == "gsc":
            return bool(self.gsc_dir) and os.path.isdir(self.gsc_dir)
        return False

    def _build_bundle(self, cell: discovery.Cell, op: str | None,
                      *, heavy: bool = False, report_filter=None) -> dict:
        view = self.view
        children_map, roots, depths = self.graph
        # A cohort whose ground truth file is not configured must degrade to an empty, clearly-labelled
        # bundle, not taking the view down: an analyst looking at HCY should not get an
        # exception page because GSC+ happens to be unreachable from this machine.
        extra_problems: list[str] = []
        try:
            gold_all = self.gold(cell.cohort)
        except (FileNotFoundError, ValueError) as exc:
            gold_all = {}
            extra_problems.append(str(exc))

        predictions = loaders.load_predictions(cell.files_for("predictions", op))
        # The restriction is applied here and nowhere else. Every stage below is driven off
        # ``predictions``, ``gold`` and ``nodes_df``, so narrowing those three narrows the ledger,
        # The node table, the error taxonomy, the calibration and the scorecard together, and
        # keeps them consistent with one another, which a per-view filter could not.
        if report_filter is not None:
            keep = {str(r) for r in report_filter}
            predictions = dict(predictions)
            predictions["predicted"] = {r: v for r, v in predictions["predicted"].items()
                                        if str(r) in keep}
            predictions["gold"] = {r: v for r, v in predictions["gold"].items() if str(r) in keep}
            predictions["report_ids"] = sorted(predictions["predicted"])
        run_reports = set(predictions["predicted"])
        # Ground truth is restricted to the reports this run actually processed. Scoring a shard that
        # covers 1/8 of the cohort against all of it would report a recall of ~1/8 of the truth
        # and blame the method for the array still being in the queue.
        gold = {rid: terms for rid, terms in gold_all.items() if rid in run_reports}

        # The three JSONL reads below are the bulk of a cold cell open, and each is a pure
        # function of bytes keyed by mtime, so each is memoised to disk as a pickle.
        key = self._cache_key(cell, op)
        nodes_fs = cell.files_for("nodes", op) if cell.spec.is_tree else discovery.FileSet("nodes")
        nodes_df = self._cached_frame(
            self._cache_path(cell, key, "nodes.pkl"), lambda: loaders.load_nodes(nodes_fs))
        metadata = self._cached_frame(
            self._cache_path(cell, key, "meta.pkl"),
            lambda: loaders.load_node_metadata(cell.files_for("node_metadata")))
        traversal_df = loaders.load_traversal(cell.files_for("traversal", op))
        timing_df = loaders.load_timing(cell.files_for("timing", op))

        if report_filter is not None:
            nodes_df = _keep_reports(nodes_df, run_reports)
            traversal_df = _keep_reports(traversal_df, run_reports)
            timing_df = _keep_reports(timing_df, run_reports)

        state = nodes_mod.state_by_report(nodes_df)
        if cell.spec.is_tree:
            ledger = pruning.prune_ledger(state, gold, children_map, roots, depths)
        else:
            ledger = _flat_ledger(gold, predictions["predicted"], view)

        node_table = nodes_mod.build_node_table(
            nodes_df, gold, view, ledger, metadata=metadata, classify_errors=heavy,
        )
        if not cell.spec.is_tree:
            node_table = _flat_node_table(node_table, predictions["predicted"], gold, view,
                                          metadata)

        if not gold and predictions["predicted"] and not extra_problems:
            extra_problems.append(
                f"none of this run's {len(predictions['predicted'])} report ids appears in "
                f"{os.path.basename(self.gold_source(cell.cohort) or '?')}, the artifacts and "
                f"the ground truth file describe different cohorts")

        # De-duplicated: a 2-D sweep repeats each τ_prune once per accept threshold, and the
        # leaderboard's question ("which swept τ would have opened this node") is about the
        # distinct prune thresholds, not about how many directories carry each one.
        tau_sweep = sorted({v for v in (discovery.op_value(p) for p in cell.operating_points)
                            if v is not None})
        run_report = report_mod.build_report(
            cell, op, gold=gold, predicted=predictions["predicted"],
            node_table=node_table, ledger=ledger, traversal_df=traversal_df,
            nodes_df=nodes_df, view=view, gold_source=self.gold_source(cell.cohort),
            curated=self.curated_for(cell.cohort), tau_sweep=tau_sweep, heavy=heavy,
        )
        # Stamped onto the report, not just the bundle, so the copy-Markdown provenance line
        # carries it too: a restricted table pasted into the thesis with no trace of the
        # restriction is the one failure mode this feature could actually cause.
        run_report["restricted_n"] = len(run_reports) if report_filter is not None else None
        run_report["problems"] = list(run_report.get("problems") or []) + extra_problems
        if extra_problems and run_report.get("status") == "ok":
            run_report["status"] = "partial"
        if cell.spec.is_tree:
            run_report["fp_factory"] = report_mod._clean(pruning.fp_factory(
                state, nodes_mod.false_positives_by_report(node_table), children_map,
                labels=nodes_mod.LabelLookup(node_table, view),
            ))

        # nodes_df and state are NOT returned. Both are inputs to what is returned, the node
        # table and the ledger, and keeping them would hold three representations of the same
        # artifact for as long as the bundle is cached. Nothing reads them. The check is a grep
        # for bundle["nodes_df"] / bundle["state"].
        return {
            "cell_id": cell.cell_id,
            "operating_point": op,
            "heavy": heavy,
            "report_filter": sorted(str(r) for r in report_filter) if report_filter else None,
            "report": run_report,
            "node_table": node_table,
            "metadata": metadata,
            "traversal_df": traversal_df,
            "timing_df": timing_df,
            "predictions": predictions,
            "gold": gold,
            "ledger": ledger,
        }

    def get_report(self, cell_id: str, operating_point: str | None = None,
                   *, heavy: bool = False, report_filter=None) -> dict | None:
        """The RunReport, from the disk cache when the artifacts have not moved.

        The light and heavy reports are cached separately, so opening the Error anatomy tab once
        does not make every later scorecard pay for the taxonomy.
        """
        cell = self.cell(cell_id)
        if cell is None:
            return None
        op = self.resolve_op(cell_id, operating_point)
        key = self._cache_key(cell, op, extra=_filter_key(report_filter))
        path = self._cache_path(cell, key, "report.heavy.json" if heavy else "report.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                log.warning("discarding unreadable cache %s: %s", path, exc)

        bundle = self.bundle(cell_id, op, heavy=heavy, report_filter=report_filter)
        if bundle is None:
            return None
        run_report = bundle["report"]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(run_report, f)
        except Exception as exc:
            log.warning("could not write cache %s: %s", path, exc)
        return run_report

    # ── the τ frontier: every configuration of one cell ───────────────────

    def tau_frontier(self, cell_id: str, operating_point: str | None = None) -> list[dict]:
        """Cost against quality across the cell's τ_prune sweep, one row per configuration.

        Free in τ_prune because earlier wrote the sweep to disk. A re-run would have to
        score the nodes each τ never visited, and the run holds no score for them, which is the
        reason ``app/tree_ui`` locks its τ_prune slider and this UI does not need to.

        On a 2-D sweep (an earlier exploratory run: 5 τ_prune × 5 τ_accept) the rows are the accept slice through
        *operating_point*, not the whole grid, see :func:`discovery.accept_slice`. Every row here
        is a full scoring pass, so the difference is 5 of them against 25.
        """
        cell = self.cell(cell_id)
        if cell is None or not cell.operating_points:
            return []
        points = discovery.accept_slice(cell.operating_points,
                                        self.resolve_op(cell_id, operating_point))
        rows = []
        for op in points:
            run_report = self.get_report(cell_id, op)
            if not run_report:
                continue
            flat = run_report.get("flat") or {}
            ledger = run_report.get("ledger") or {}
            traversal = run_report.get("traversal") or {}
            rows.append({
                "operating_point": op,
                "op_value": discovery.op_value(op),
                "op_accept": discovery.op_accept(op),
                "micro_precision": flat.get("micro_precision"),
                "micro_recall": flat.get("micro_recall"),
                "micro_f1": flat.get("micro_f1"),
                "tp": flat.get("tp"), "fp": flat.get("fp"), "fn": flat.get("fn"),
                "recall_ceiling": ledger.get("recall_ceiling"),
                "recall_lost_to_pruning": ledger.get("recall_lost_to_pruning"),
                "n_blocked": (ledger.get("fate_counts") or {}).get("blocked"),
                "mean_slm_calls": traversal.get("mean_slm_calls"),
                "mean_nodes_visited": traversal.get("mean_nodes_visited"),
            })
        return report_mod.tau_frontier(rows)

    # ── the HCY deep-dive sampling frame ────────────────────────────────────

    @property
    def sample(self):
        """The loaded :class:`~apps.treephenorag_ui.sample.Sample`, or ``None``. Memoised, absence included.

        Additive, like the PhenoBERT underlay: no frame configured, or nothing at the path, leaves
        every view as it was and disables the subset control.
        """
        if self._sample is _UNSET:
            self._sample = sample_mod.load(self.hcy_frame_dir)
        return self._sample

    def subset_ids(self, cell_id: str, mode: str | None,
                   cell: str | None = None) -> list[str] | None:
        """The report ids a subset choice means *for this cell*, or ``None`` for unrestricted.

        Refuses on the cohort, not filtering to nothing: the frame is an HCY draw, and
        intersecting it with a GSC+ run would silently produce an empty page whose cause is not
        visible anywhere on it.
        """
        frame = self.sample
        run = self.cell(cell_id)
        if frame is None or run is None or not frame.applies_to(run.cohort):
            return None
        return frame.ids(mode, cell)

    def aggregate_filter(self, cell_id: str, subset) -> list[str] | None:
        """The report ids the *metrics* should be restricted to, or ``None``.

        Two conditions, both required: a subset is chosen, and the user has explicitly ticked
        "also restrict the metrics". Paging through a subset changes no number on screen and needs
        no opt-in. Recomputing every rate over a purposive draw does, which is why the two are
        separate controls and why this is the only function that can turn the second one on.
        """
        subset = subset or {}
        if not subset.get("aggregates"):
            return None
        return self.subset_ids(cell_id, subset.get("mode"), subset.get("cell"))

    def subset_note(self, cell_id: str, subset):
        """The banner an aggregate view must show while its numbers are restricted, or ``None``.

        Rendered by every view that reports a cohort aggregate. The wording is
        :data:`~apps.treephenorag_ui.sample.WARNING` verbatim, because the frame's own FRAME.md says it in
        those terms and a paraphrase on screen would be a weaker claim than the one on file.
        """
        subset = subset or {}
        ids = self.aggregate_filter(cell_id, subset)
        if ids is None:
            return None
        frame = self.sample
        where = frame.describe(subset.get("mode"), subset.get("cell")) if frame else "a subset"
        return theme.note(
            f"Every number on this page is computed over {where}, not the full cohort. "
            f"{sample_mod.WARNING}", tone="warn")

    # ── the nodes index, and the one-report path built on it ────────────────

    def report_ids(self, cell_id: str, operating_point: str | None = None) -> list[str]:
        """Every report this configuration processed, without building anything.

        The deep-dive's report dropdown used to call :meth:`bundle` for this, so merely opening
        the tab built the whole cohort node table, a million rows at the loosest swept tau_prune,
        to fill a list of ids. ``load_predictions`` streams and never builds a frame.
        """
        cell = self.cell(cell_id)
        if cell is None:
            return []
        op = self.resolve_op(cell_id, operating_point)
        key = f"{cell_id}|{op}|{self._cache_key(cell, op, extra='ids')}"
        if key not in self._report_ids:
            predictions = loaders.load_predictions(cell.files_for("predictions", op))
            self._report_ids[key] = list(predictions["report_ids"])
        return self._report_ids[key]

    def nodes_scan(self, cell_id: str, operating_point: str | None = None) -> dict | None:
        """The ``*_nodes.jsonl`` byte-offset index. Disk-cached and LRU'd, like the calls scan.

        Only the offsets are kept, a few hundred (report, span) entries, so this is cheap to
        hold and cheap to pickle whatever the artifact weighs.
        """
        cell = self.cell(cell_id)
        if cell is None or not cell.spec.is_tree:
            return None
        op = self.resolve_op(cell_id, operating_point)
        fs = cell.files_for("nodes", op)
        if not fs.present:
            return None

        key = self._cache_key(cell, op, extra="nodes-index")
        if key in self._nodes_scans:
            self._nodes_scans.move_to_end(key)
            return self._nodes_scans[key]

        path = self._cache_path(cell, key, "nodeidx.pkl")
        scan = None
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    scan = pickle.load(f)
            except Exception as exc:
                log.warning("discarding unreadable nodes index %s: %s", path, exc)
        if scan is None:
            n_bytes = sum(os.path.getsize(p) for p in fs.paths if os.path.isfile(p))
            log.info("indexing %s (%.1f MB), one pass, then cached",
                     ", ".join(os.path.basename(p) for p in fs.paths), n_bytes / 1e6)
            scan = loaders.scan_nodes(fs)
            try:
                with open(path, "wb") as f:
                    pickle.dump(scan, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                log.warning("could not write nodes index %s: %s", path, exc)

        self._nodes_scans[key] = scan
        while len(self._nodes_scans) > _NODES_LRU_SIZE:
            self._nodes_scans.popitem(last=False)
        return scan

    def report_bundle(self, cell_id: str, operating_point: str | None,
                      report_id: str) -> dict | None:
        """One report's node table, ledger and ground truth, without the cohort's.

        This is what makes the loosest swept tau_prune usable. The deep-dive asks about one report
        at a time, but got there by masking a cohort table that had to be built first. On a
        1 051 846-row artifact that is the whole memory problem for a page that displays a few
        hundred rows. Here the node records come from the byte index, and the ledger is
        :func:`pruning.prune_ledger` over a one-key ground truth mapping, the same function on the same
        arithmetic, which is why the deep-dive's fates still agree with the scorecard's.

        Falls back to slicing the cohort table for a non-tree cell, which has no nodes artifact
        to index and whose tables are small by design.

        Returns ``{"frame", "gold", "ledger", "predicted", "metadata"}``, or ``None`` for an
        unknown cell.
        """
        cell = self.cell(cell_id)
        if cell is None:
            return None
        op = self.resolve_op(cell_id, operating_point)
        report_id = str(report_id)

        scan = self.nodes_scan(cell_id, op)
        if scan is None:
            bundle = self.bundle(cell_id, op)
            if bundle is None:
                return None
            frame = bundle["node_table"]
            return {
                "frame": frame.loc[frame["report_id"].astype(str) == report_id],
                "gold": bundle["gold"].get(report_id, set()),
                "ledger": bundle["ledger"],
                "predicted": bundle["predictions"]["predicted"].get(report_id, set()),
                "metadata": bundle["metadata"],
            }

        key = f"{cell_id}|{op}|{report_id}|{self._cache_key(cell, op, extra='one')}"
        if key in self._report_bundles:
            self._report_bundles.move_to_end(key)
            return self._report_bundles[key]

        view = self.view
        children_map, roots, depths = self.graph
        try:
            gold_all = self.gold(cell.cohort)
        except (FileNotFoundError, ValueError):
            gold_all = {}
        gold_terms = set(gold_all.get(report_id, ()))
        gold = {report_id: gold_terms}

        nodes_df = loaders.nodes_for_report(scan, report_id)
        metadata = self._cached_frame(
            self._cache_path(cell, self._cache_key(cell, op), "meta.pkl"),
            lambda: loaders.load_node_metadata(cell.files_for("node_metadata")))

        state = nodes_mod.state_by_report(nodes_df)
        ledger = pruning.prune_ledger(state, gold, children_map, roots, depths)
        # classify_errors stays on: the taxonomy is a bounded ontology walk per false positive, and
        # one report's false positives are a bounded number. It is only the cohort-wide version
        # that has to be deferred behind ``heavy``.
        frame = nodes_mod.build_node_table(nodes_df, gold, view, ledger, metadata=metadata,
                                           classify_errors=True)

        result = {"frame": frame, "gold": gold_terms, "ledger": ledger,
                  "predicted": set(frame.loc[frame["accepted"].fillna(False).astype(bool),
                                             "hpo_id"].astype(str)),
                  "metadata": metadata}
        self._report_bundles[key] = result
        while len(self._report_bundles) > _REPORT_LRU_SIZE:
            self._report_bundles.popitem(last=False)
        return result

    # ── the calls scan ──────────────────────────────────────────────────────

    def calls_bytes(self, cell_id: str) -> int:
        """Size in bytes of the verifier call files of *cell_id*."""
        cell = self.cell(cell_id)
        if cell is None:
            return 0
        fs = cell.files_for("calls")
        return sum(os.path.getsize(p) for p in fs.paths if os.path.isfile(p))

    def calls_scan(self, cell_id: str) -> dict | None:
        """The ``*_calls.jsonl`` summary + byte-offset index. Disk-cached and LRU'd.

        Kept out of :meth:`bundle` on purpose: it is the only unbounded artifact, most views do
        not read it, and a cell open should not block on gigabytes nobody asked for.
        """
        cell = self.cell(cell_id)
        if cell is None:
            return None
        fs = cell.files_for("calls")
        if not fs.present:
            return None

        key = self._cache_key(cell, None, extra="calls")
        if key in self._calls:
            self._calls.move_to_end(key)
            return self._calls[key]

        path = self._cache_path(cell, key, "calls.pkl")
        scan = None
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    scan = pickle.load(f)
            except Exception as exc:
                log.warning("discarding unreadable calls cache %s: %s", path, exc)
        if scan is None:
            log.info("scanning %s (%.1f MB), one pass, then cached",
                     ", ".join(os.path.basename(p) for p in fs.paths),
                     self.calls_bytes(cell_id) / 1e6)
            scan = loaders.scan_calls(fs)
            try:
                with open(path, "wb") as f:
                    pickle.dump(scan, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                log.warning("could not write calls cache %s: %s", path, exc)

        self._calls[key] = scan
        while len(self._calls) > _CALLS_LRU_SIZE:
            self._calls.popitem(last=False)
        return scan

    # ── evidence (Stanza) ───────────────────────────────────────────────────

    def evidence(self):
        """The lazily-built :class:`~apps.treephenorag_ui.evidence.Evidence` helper."""
        if self._evidence is None:
            self._evidence = evidence_mod.Evidence(self)
        return self._evidence


def _filter_key(report_filter) -> str:
    """A short, order-independent digest of a report filter, for the cache key.

    Empty string when unrestricted, so an unrestricted bundle keeps the key it has always had and
    every cache written before this feature existed stays valid.
    """
    if not report_filter:
        return ""
    ids = "|".join(sorted(str(r) for r in report_filter))
    return "sub:" + hashlib.sha1(ids.encode()).hexdigest()[:12]


def _keep_reports(frame: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    """Rows whose ``report_id`` is in *keep*, with unused categories dropped.

    The category cleanup is not cosmetic: ``report_id`` is categorical, and a slice keeps every
    category it no longer has rows for. Downstream ``groupby`` calls pass ``observed=True`` for
    that reason, but ``cat.categories`` is also what :class:`ReportStates` and
    ``_depth_cost`` index against, so it is cheaper to drop them once here.
    """
    if frame is None or frame.empty or "report_id" not in frame.columns:
        return frame
    subset = frame.loc[frame["report_id"].astype(str).isin(keep)].reset_index(drop=True)
    if isinstance(subset["report_id"].dtype, pd.CategoricalDtype):
        subset["report_id"] = subset["report_id"].cat.remove_unused_categories()
    return subset


def _flat_ledger(gold: dict[str, set[str]], predicted: dict[str, set[str]], view) -> dict:
    """The ledger for a non-traversal method: every annotated term is either found or rejected.

    RAG-HPO, flat top-M and the ensemble have no traversal, so ``blocked`` and ``unreached`` are
    not outcomes they can produce. Giving them a ledger anyway is what lets the comparison view
    put their recall decomposition beside the tree's, with the pruning buckets structurally zero,
    not absent, which is the honest way to show that they cannot lose recall that way.
    """
    rows = []
    counts = {fate: 0 for fate in pruning.FATES}
    for report_id in sorted(gold):
        # Raw equality, as everywhere a label is decided, see ``scoring.align``.
        pred = set(predicted.get(report_id, ()))
        for term in sorted(gold[report_id]):
            resolved = view.resolve(term)
            fate = "found" if term in pred else "rejected"
            counts[fate] += 1
            rows.append({"report_id": report_id, "hpo_id": term, "fate": fate,
                         "depth": view.depth(resolved) if resolved else None,
                         "culprit": None, "culprit_depth": None, "culprit_prune_score": None})
    n_gold = sum(counts.values())
    return {
        "rows": rows,
        "fate_counts": counts,
        "n_gold": n_gold,
        "recall_actual": counts["found"] / n_gold if n_gold else 0.0,
        "recall_ceiling": 1.0 if n_gold else 0.0,
        "recall_lost_to_pruning": 0.0,
    }


def _flat_node_table(frame, predicted: dict[str, set[str]], gold: dict[str, set[str]], view,
                     metadata):
    """Node table for a method with no ``*_nodes.jsonl``: one row per predicted or annotated term.

    The tree builder starts from visited nodes, of which these methods have none. Their rows come
    from the two sets they do produce, so the FP/FN anatomy views work on them unchanged, with
    ``prune_score``/``accept_score`` empty, which is what makes the calibration panels correctly
    say "not measurable here" instead of plotting zeros.
    """
    import numpy as np
    import pandas as pd

    rows = []
    for report_id in sorted(set(predicted) | set(gold)):
        gold_res = set(gold.get(report_id, ()))
        pred_terms = sorted(predicted.get(report_id, ()))
        pred_res = set(pred_terms)
        for term in pred_terms:
            resolved = view.resolve(term)
            in_gold = term in gold_res
            rows.append({"report_id": report_id, "hpo_id": term, "state": "predicted",
                         "depth": view.depth(resolved) if resolved else None,
                         "prune_score": np.nan, "accept_score": np.nan,
                         "expanded": False, "accepted": True, "in_gold": in_gold,
                         "is_gold_artifact": None, "gold_disagrees": False,
                         "outcome": "TP" if in_gold else "FP", "subtree_has_gold": in_gold})
        for term in sorted(gold.get(report_id, ())):
            resolved = view.resolve(term)
            if term in pred_res:
                continue
            rows.append({"report_id": report_id, "hpo_id": term, "state": "not_predicted",
                         "depth": view.depth(resolved) if resolved else None,
                         "prune_score": np.nan, "accept_score": np.nan,
                         "expanded": False, "accepted": False, "in_gold": True,
                         "is_gold_artifact": None, "gold_disagrees": False,
                         "outcome": "FN", "subtree_has_gold": True})

    if not rows:
        return frame
    built = pd.DataFrame(rows)
    built["fate"] = pd.NA
    built["culprit"] = pd.NA
    built["culprit_prune_score"] = np.nan
    built = nodes_mod._attach_error_classes(built, gold, view)
    built = nodes_mod._attach_metadata(built, metadata, view)
    built = nodes_mod.attach_calls(built, None)
    return built.reindex(columns=nodes_mod.COLUMNS)
