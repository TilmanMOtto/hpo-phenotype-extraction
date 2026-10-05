"""earlier artifacts on disk → the plain dicts and lists ``thesis_metrics`` takes.

Strictly a translation layer: no metric is computed here, and nothing is silently repaired. Two
kinds of work happen:

**Reading.** Each earlier driver writes a slightly different JSONL schema (see the module docstrings
of ``src/hpo_extraction/treephenorag/score_store.py`` et al.). These readers normalise them onto one vocabulary, ``report_id`` as the key everywhere, ground truth and predicted sets as ``set[str]``, per-node scores as
parallel arrays, so ``sections.py`` never branches on which driver produced a file.

**Reconstructing the two confidence scores that were not persisted.** ``flat_topm`` writes
per-sentence margins but no aggregated score, and the ensemble writes per-model detections but no
vote fraction. Both are recomputed here, by the same rule the driver used at inference time
(``max σ(margin)``; ``votes / n_active_models``), so §Calibration and §Risk--coverage cover four
method families instead of two.

**Ground truth is joined, never read off the artifact.** The tree driver writes an ``is_gold`` flag onto
every node line, which would let its calibration sample skip the join, but that flag records
whichever ground-truth set was configured at *inference* time, so re-scoring against a corrected ground truth file
would move the derived prune label while leaving the accept label fixed, mixing two ground truth
definitions inside one table. Every label here therefore comes from the ``gold`` argument;
``is_gold`` is retained only as a cross-check (see
:func:`tree_calibration_sample`). The other families never had the flag and were always joined
from the predictions summary line's ``gold_set``. Both routes are exercised by the tests.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file, skipping blank lines.

    A truncated final line, the normal shape of a wall-clock kill, raises
    ``json.JSONDecodeError`` naming the file and line, rather than being dropped. A partial
    cohort must be visible, not quietly averaged.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{path}:{lineno}: {exc.msg}", exc.doc, exc.pos) from None


# ── Ground truth ─────────────────────────────────────────────────────────────

def load_gold(cohort: str, *, hcy_gt_path: str | None = None, gsc_dir: str | None = None,
              raghpo_dir: str | None = None,
              hcy_curated_gt_path: str | None = None) -> dict[str, set[str]]:
    """``{report_id: gold HPO set}`` for a cohort, via the repo's own dataset loaders.

    Every loader returns the **annotated** terms with no ancestor closure, the ground-truth set earlier scored against, and therefore the one the results must be computed on.

    The two ``gsc_raghpo*`` cohorts are views of the same GSC+ artifacts under the corpus RAG-HPO
    actually published on: their 114 documents, and for ``gsc_raghpo_ann`` their own re-annotation
    of them. See ``resources/data/GSC_RAGHPO/PROVENANCE.md``. ``gsc_2024_eval_206`` is the third
    such view -- AutoPCR's evaluation split of the same corpus, corpus ground truth unchanged. See
    ``resources/data/GSC_2024/PROVENANCE.md``.
    """
    if cohort == "hcy":
        if not hcy_gt_path:
            raise ValueError("cohort 'hcy' requires hcy_gt_path")
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset
        raw = HCYDataset(hcy_gt_path, "").load_ground_truth()
    elif cohort == "hcy_curated":
        # The curation UI's export, in the same two-column shape HCYDataset already reads, so the
        # only difference from `hcy` is which file the ground truth comes out of.
        if not hcy_curated_gt_path:
            raise ValueError("cohort 'hcy_curated' requires hcy_curated_gt_path")
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset
        raw = HCYDataset(hcy_curated_gt_path, "").load_ground_truth()
    elif cohort == "gsc":
        if not gsc_dir:
            raise ValueError("cohort 'gsc' requires gsc_dir")
        from hpo_extraction.evaluation.datasets.gsc import load_gsc_ground_truth
        raw = load_gsc_ground_truth(gsc_dir)
    elif cohort == "gsc_2024_eval_206":
        # AutoPCR's own evaluation frame: the corpus ground truth, restricted to the 206 documents Tao
        # et al. score on. The ground truth is unchanged -- only the document set narrows -- which is what
        # makes their document-level F1 readable against ours.
        if not gsc_dir:
            raise ValueError("cohort 'gsc_2024_eval_206' requires gsc_dir")
        from hpo_extraction.evaluation.datasets.gsc import (load_gsc2024_eval_ids, load_gsc_ground_truth,
                                             restrict_to_ids)
        raw = restrict_to_ids(load_gsc_ground_truth(gsc_dir), load_gsc2024_eval_ids(gsc_dir))
    elif cohort == "gsc_raghpo":
        if not (gsc_dir and raghpo_dir):
            raise ValueError("cohort 'gsc_raghpo' requires gsc_dir and raghpo_dir")
        from hpo_extraction.evaluation.datasets.gsc import (load_gsc_ground_truth, load_raghpo_ids,
                                             restrict_to_ids)
        raw = restrict_to_ids(load_gsc_ground_truth(gsc_dir), load_raghpo_ids(raghpo_dir))
    elif cohort == "gsc_raghpo_ann":
        if not raghpo_dir:
            raise ValueError("cohort 'gsc_raghpo_ann' requires raghpo_dir")
        from hpo_extraction.evaluation.datasets.gsc import load_raghpo_ground_truth
        raw = load_raghpo_ground_truth(raghpo_dir)
    else:
        raise ValueError(f"unknown cohort: {cohort!r}")

    return {str(k): {c.strip() for c in v if c and str(c).strip()} for k, v in raw.items()}


def load_reports(cohort: str, *, input_dir: str | None = None, gsc_dir: str | None = None,
                 raghpo_dir: str | None = None) -> dict[str, str]:
    """``{report_id: raw report text}``, needed only to re-derive the segmentation for eq. (2)."""
    if cohort in ("hcy", "hcy_curated"):
        if not input_dir:
            raise ValueError(f"cohort {cohort!r} requires input_dir")
        from hpo_extraction.data.loading import load_txt
        return {str(k): v for k, v in load_txt(input_dir).items()}
    if cohort in ("gsc", "gsc_2024_eval_206", "gsc_raghpo", "gsc_raghpo_ann"):
        if not gsc_dir:
            raise ValueError(f"cohort {cohort!r} requires gsc_dir")
        from hpo_extraction.evaluation.datasets.gsc import (load_gsc2024_eval_ids, load_gsc_reports,
                                             load_raghpo_ids, restrict_to_ids)
        reports = load_gsc_reports(gsc_dir)
        if cohort == "gsc":
            return reports
        if cohort == "gsc_2024_eval_206":
            return restrict_to_ids(reports, load_gsc2024_eval_ids(gsc_dir))
        if not raghpo_dir:
            raise ValueError(f"cohort {cohort!r} requires raghpo_dir")
        return restrict_to_ids(reports, load_raghpo_ids(raghpo_dir))
    raise ValueError(f"unknown cohort: {cohort!r}")


# ── Predictions (one contract, shared by all four drivers) ───────────────────

def load_predictions(path: str | Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``(predicted_by_report, gold_by_report)`` from a ``*_predictions.jsonl``.

    Every driver writes per-term lines followed by one ``summary: true`` line carrying
    ``predicted_set`` and ``gold_set``. The summary is authoritative: it is the only place an
    **empty** prediction set is recorded, and dropping those reports would inflate precision by
    the reports the system said nothing about.
    """
    predicted: dict[str, set[str]] = {}
    gold: dict[str, set[str]] = {}
    for rec in read_jsonl(path):
        rid = str(rec["report_id"])
        if rec.get("summary"):
            predicted[rid] = set(rec.get("predicted_set") or [])
            gold[rid] = set(rec.get("gold_set") or [])
    return predicted, gold


def load_prediction_candidates(path: str | Path) -> dict[str, list[str]]:
    """``{report_id: ranked candidate terms}``, ``flat_topm`` only, for term-level R@M.

    The list is the M retrieved HPO ids in descending retrieval score, index 0 = rank 1.
    """
    return {
        str(rec["report_id"]): list(rec["candidates"])
        for rec in read_jsonl(path)
        if rec.get("summary") and rec.get("candidates") is not None
    }


# ── Tree artifacts ───────────────────────────────────────────────────────────

def load_nodes(path: str | Path) -> dict[str, list[dict]]:
    """``{report_id: [node record, ...]}`` for every **visited** node at one tau.

    Records carry ``hpo_id``, ``depth``, ``prune_score``, ``accept_score``, ``expanded``,
    ``accepted`` and ``is_gold`` as ``tree_experiment`` wrote them.
    """
    by_report: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(path):
        by_report[str(rec["report_id"])].append(rec)
    return dict(by_report)


def expanded_sets(nodes_by_report: dict[str, list[dict]]) -> dict[str, set[str]]:
    """``{report_id: {hpo_id whose children were revealed}}``, the input to ``reachable_set``."""
    return {
        rid: {r["hpo_id"] for r in recs if r.get("expanded")}
        for rid, recs in nodes_by_report.items()
    }


def visited_sets(nodes_by_report: dict[str, list[dict]]) -> dict[str, set[str]]:
    """``{report_id: {hpo_id evaluated}}``, the traversal's unranked candidate set."""
    return {rid: {r["hpo_id"] for r in recs} for rid, recs in nodes_by_report.items()}


def load_traversal(path: str | Path) -> list[dict]:
    """Per-report traversal records: ``n_slm_calls``, ``n_unique_nodes_visited``,
    ``total_frontier_insertions`` and the nested ``depth_table``.

    Returned unchanged, ``cost.slm_call_stats`` reads these key names.
    """
    return list(read_jsonl(path))


def depth_visits(
    traversal_records: Iterable[dict],
    nodes_by_report: dict[str, list[dict]] | None = None,
) -> list[dict[str, dict]]:
    """Traversal records → the ``visits_by_report`` shape ``depth_cost_table`` expects.

    ``depth_cost_table`` wants ``{hpo_id: {"depth", "expanded", "accepted", "n_slm_calls"}}``.
    The traversal file stores per-depth *aggregates*, not per-node rows, so it alone cannot
    supply the keys.

    Pass ``nodes_by_report`` (from :func:`load_nodes`, the same tau directory) to get **real**
    ``hpo_id`` keys. That is required whenever ground-truth sets are handed to ``depth_cost_table``: its
    ``n_gold_visited`` column tests ``hpo_id in gold``, which can never fire against synthetic
    keys, and the column would silently read zero, not being absent. ``_write_tau`` emits
    the nodes and traversal lines for a report together, so the two files always agree on
    membership and on the per-depth counts (both derive from ``TraversalResult.visits``).

    Without ``nodes_by_report`` each depth row is expanded into ``n_visited`` synthetic entries,
    which still reproduces the per-depth frontier sizes, expansion counts and call totals, the
    only quantities the cost columns read. **Do not pass ground-truth sets in that case.**

    Per-node ``n_slm_calls`` is not persisted, so in both modes the depth's call total is spread
    evenly across the nodes at that depth. Only the per-depth total is ever read.
    """
    out = []
    for rec in traversal_records:
        rid = str(rec.get("report_id", ""))
        calls_by_depth = {
            int(row["depth"]): (int(row.get("n_slm_calls", 0)), int(row.get("n_visited", 0)))
            for row in rec.get("depth_table") or []
        }
        node_recs = (nodes_by_report or {}).get(rid)

        visits: dict[str, dict] = {}
        if node_recs:
            for node in node_recs:
                depth = int(node["depth"])
                n_calls, n_visited = calls_by_depth.get(depth, (0, 0))
                visits[node["hpo_id"]] = {
                    "depth": depth,
                    "expanded": bool(node.get("expanded")),
                    "accepted": bool(node.get("accepted")),
                    "n_slm_calls": n_calls / n_visited if n_visited else 0,
                }
        else:
            for depth, (n_calls, n_visited) in sorted(calls_by_depth.items()):
                row = next(r for r in rec["depth_table"] if int(r["depth"]) == depth)
                n_expanded = int(row.get("n_expanded", 0))
                n_accepted = int(row.get("n_accepted", 0))
                for i in range(n_visited):
                    visits[f"d{depth}_{i}"] = {
                        "depth": depth,
                        "expanded": i < n_expanded,
                        "accepted": i < n_accepted,
                        "n_slm_calls": n_calls / n_visited if n_visited else 0,
                    }
        out.append(visits)
    return out


# ── Calls: per-sentence SLM records (tree and flat_topm) ─────────────────────

def _sigmoid(x: float) -> float:
    # Guard the overflow at |x| ≳ 700 that a confident margin can reach.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
    e = math.exp(max(x, -700.0))
    return e / (1.0 + e)


def load_calls_ranked_segments(
    path: str | Path, ctx_type: str | None = None,
) -> dict[tuple[str, str], list[int]]:
    """``{(report_id, hpo_id): [sent_index by ascending rank]}``, the retrieved list for eq. (2).

    ``ctx_type`` filters the tree's dual-context runs (``"union"`` / ``"own"``). Pass ``None`` for
    ``flat_topm``, whose calls file has no such column.
    """
    ranked: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for rec in read_jsonl(path):
        if ctx_type is not None and rec.get("ctx_type") != ctx_type:
            continue
        key = (str(rec["report_id"]), rec["hpo_id"])
        ranked[key].append((int(rec["rank"]), int(rec["sent_index"])))
    return {k: [s for _, s in sorted(v)] for k, v in ranked.items()}


def accept_scores_from_calls(
    path: str | Path, ctx_type: str | None = None,
) -> dict[tuple[str, str], float]:
    """``{(report_id, hpo_id): max σ(margin)}``, the accept score, recomputed.

    This is the AnyYes aggregation ``flat_topm_experiment`` applies at inference time
    (``accept iff max sigmoid(margin) >= accept_threshold``), so re-running a threshold over these
    values reproduces that run's decisions. The tree driver already persists its
    ``accept_score``. This path exists for the drivers that do not.
    """
    best: dict[tuple[str, str], float] = {}
    for rec in read_jsonl(path):
        if ctx_type is not None and rec.get("ctx_type") != ctx_type:
            continue
        key = (str(rec["report_id"]), rec["hpo_id"])
        score = _sigmoid(float(rec["margin"]))
        if score > best.get(key, -1.0):
            best[key] = score
    return best


def available_ctx_types(path: str | Path) -> set[str]:
    """Which retrieval contexts a calls file actually contains.

    ``retrieval_ctx=union_only`` writes ``"union"`` alone; ``union_prune_own_accept`` writes both.
    Callers use this to pick the right filter, not assuming the config value that ran.
    """
    return {rec.get("ctx_type") for rec in read_jsonl(path) if rec.get("ctx_type") is not None}


# ── Ensemble ─────────────────────────────────────────────────────────────────

def load_detections(path: str | Path) -> tuple[dict[tuple[str, str], set[str]], list[str]]:
    """``({(report_id, hpo_id): {models that detected it}}, sorted active model list)``.

    The merged ``slm_ensemble_detections.jsonl`` is a concatenation of the per-model dumps for
    the models that produced output, so the model set is read from the data, not from the
    config, a run where a model died must not be scored as if eight models had voted.
    """
    votes: dict[tuple[str, str], set[str]] = defaultdict(set)
    models: set[str] = set()
    for rec in read_jsonl(path):
        models.add(rec["model"])
        votes[(str(rec["report_id"]), rec["hpo_id"])].add(rec["model"])
    return dict(votes), sorted(models)


def ensemble_confidences(
    votes: dict[tuple[str, str], set[str]], n_models: int,
) -> dict[tuple[str, str], float]:
    """``{(report_id, hpo_id): votes / n_models}``, the ensemble's only confidence signal.

    Its support has at most ``n_models + 1`` atoms, so ``discrete_score_table`` is the honest
    reliability view for it and equal-width ECE is close to meaningless, the same argument the
    skeleton makes for the tree's ``S+1``-atom score, only sharper.
    """
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    return {k: len(v) / n_models for k, v in votes.items()}


def load_agg_summary(path: str | Path) -> list[dict]:
    """``slm_ensemble_agg_summary.csv``, the Free Listing generation run's own micro/macro numbers.

    Read only for the independent-implementation cross-check in the tests: the result-table library recomputes
    these through ``thesis_metrics`` and the two must agree.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── Calibration sample assembly ──────────────────────────────────────────────

def tree_calibration_sample(
    nodes_by_report: dict[str, list[dict]],
    gold: dict[str, set[str]],
    view,
    score_key: str = "accept_score",
) -> dict[str, list]:
    """Flatten visited tree nodes into the aligned arrays the calibration functions take.

    The label depends on which score is being assessed, and the two are not interchangeable:

    ``accept_score`` predicts **node presence**: 1 iff the node is in that report's ground-truth set.

    ``prune_score`` predicts **subtree presence**, the gate's job is to decide whether anything
    worth finding lies below, not whether this node is itself present. Its label is 1 iff any ground truth
    term of that report lies in the subtree rooted at the node. Using node presence for it would
    score the gate against a question it was never asked, and would make a well-behaved gate look
    badly miscalibrated at every internal node.

    **Both labels are derived here, from the ``gold`` argument.** The tree driver also writes an
    ``is_gold`` flag onto every node line, and reading the accept label off that flag was the
    obvious shortcut, but it silently pins the accept label to whichever ground-truth set was configured
    *at inference time*. Re-scoring against a corrected or extended ground truth file then moves the prune
    label while leaving the accept label behind, so one table mixes two ground truth definitions with
    nothing to show for it. The ground truth file passed to this run is the single authority; ``is_gold``
    is used only as a cross-check, and any disagreement is returned in ``n_is_gold_disagreements``
    for the caller to report.

    Returns aligned lists plus the grouping keys §Signed and conditional calibration needs.
    """
    if score_key not in ("accept_score", "prune_score"):
        raise ValueError(f"score_key must be accept_score or prune_score, got {score_key!r}")

    confidences: list[float] = []
    labels: list[int] = []
    hpo_ids: list[str] = []
    report_ids: list[str] = []
    depths: list[int | None] = []
    n_disagreements = 0

    for rid, recs in nodes_by_report.items():
        gold_set = gold.get(rid, set())
        subtree_cache: dict[str, int] = {}
        for rec in recs:
            hpo_id = rec["hpo_id"]
            node_present = int(hpo_id in gold_set)
            if "is_gold" in rec and int(rec["is_gold"]) != node_present:
                n_disagreements += 1
            if score_key == "accept_score":
                label = node_present
            else:
                if hpo_id not in subtree_cache:
                    subtree = view.descendants_or_self(hpo_id)
                    subtree_cache[hpo_id] = int(bool(subtree & gold_set))
                label = subtree_cache[hpo_id]
            confidences.append(float(rec[score_key]))
            labels.append(label)
            hpo_ids.append(hpo_id)
            report_ids.append(rid)
            depths.append(rec.get("depth"))

    return {
        "confidences": confidences,
        "labels": labels,
        "hpo_ids": hpo_ids,
        "report_ids": report_ids,
        "depths": depths,
        "n_is_gold_disagreements": n_disagreements,
    }


def pair_calibration_sample(
    scores: dict[tuple[str, str], float],
    gold: dict[str, set[str]],
) -> dict[str, list]:
    """The same aligned-array shape, for methods whose scores live in a ``(report, hpo)`` dict.

    Used for ``flat_topm`` (reconstructed ``max σ(margin)``) and the ensemble (vote fraction).
    The label is node presence, these methods make no subtree claim.
    """
    confidences: list[float] = []
    labels: list[int] = []
    hpo_ids: list[str] = []
    report_ids: list[str] = []

    for (rid, hpo_id), score in scores.items():
        confidences.append(float(score))
        labels.append(int(hpo_id in gold.get(rid, set())))
        hpo_ids.append(hpo_id)
        report_ids.append(rid)

    return {
        "confidences": confidences,
        "labels": labels,
        "hpo_ids": hpo_ids,
        "report_ids": report_ids,
        "depths": [None] * len(confidences),
    }


def aligned_sets(
    predicted: dict[str, set[str]],
    gold: dict[str, set[str]],
) -> tuple[list[str], list[set[str]], list[set[str]]]:
    """``(report_ids, gold_sets, pred_sets)`` over the reports **both** dicts cover.

    Every cohort metric in ``thesis_metrics`` zips its two iterables, so alignment has to be
    established once, explicitly, not relying on dict ordering. Reports scored but absent
    from the ground truth file (or vice versa) are excluded here and counted by the caller.
    """
    rids = sorted(set(predicted) & set(gold))
    return rids, [gold[r] for r in rids], [predicted[r] for r in rids]
