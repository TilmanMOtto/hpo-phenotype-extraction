"""The ``RunReport``, the compact, JSON-serialisable dict every view reads.

One report describes one **cell at one configuration**: a (method, cohort, τ) triple. It is
built once per cell, cached to disk keyed by the artifacts' mtimes, and read by every view. The
heavy intermediates it was computed from, the node table, the call-summary DataFrame, the
``OntologyView``, never leave the registry, so nothing here has to be small for the browser's
sake, only serialisable.

Two things are *not* in it:

**The τ frontier** is not per-operating-point. It is the cell's whole sweep, and computing it
means loading every ``tau_*`` directory. :func:`tau_frontier` builds it separately so that
opening the scorecard on one τ does not pay for all of them.

**Per-node rows.** The error explorer pages them server-side out of the node table. A cell with
2 million visited nodes has a report of a few hundred kilobytes and a node table of a gigabyte,
and only one of those is allowed near a callback.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import calib, nodes as nodes_mod, pruning, scoring


def _clean(value):
    """Coerce numpy/pandas scalars and non-finite floats into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    return value


def build_report(
    cell,
    operating_point: str | None,
    *,
    gold: dict[str, set[str]],
    predicted: dict[str, set[str]],
    node_table: pd.DataFrame,
    ledger: dict,
    traversal_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    view,
    gold_source: str,
    curated=None,
    tau_sweep: list[float] | None = None,
    heavy: bool = False,
) -> dict:
    """Assemble the report for one (cell, configuration).

    ``heavy=False`` omits the two sections that dominate the build and are each read by
    one tab. Measured on a 60-report × 800-node run: everything else totals ~1.5 s, the error
    taxonomy costs ~7.6 s (an ontology walk per false positive) and the calibration block ~11 s
    (the risk--coverage pass over every scored node). Computing all three whenever a cell is
    opened made the scorecard, which reads none of them, take twenty seconds to appear.
    """
    gold_sets, pred_sets, report_ids = scoring.align(gold, predicted, view)
    scored = scoring.score(gold, predicted, view)
    # LabelLookup, not labels_map: a culprit is by design a node whose children were not
    # visited, so it is frequently absent from the node table and would print as a bare HPO id.
    labels = nodes_mod.LabelLookup(node_table, view)

    report = {
        "cell_id": cell.cell_id,
        "method": cell.spec.key,
        "label": cell.label,
        "method_label": cell.spec.label,
        "experiment": cell.spec.exp_id,
        "cohort": cell.cohort,
        "kind": cell.spec.kind,
        "is_tree": cell.spec.is_tree,
        "variant": cell.spec.variant,
        "run_dir": cell.run_dir,
        "operating_point": operating_point,
        "operating_points": list(cell.operating_points),
        "sweep_axis": cell.spec.sweep_axis,
        "status": cell.status,
        "problems": list(cell.problems),
        "gold_source": gold_source,
        "curated": _curated_block(curated, predicted, report_ids),
        "counts": {
            "n_reports_scored": len(report_ids),
            "n_reports_in_run": len(predicted),
            "n_gold_terms": int(sum(len(g) for g in gold_sets)),
            "n_predicted_terms": int(sum(len(p) for p in pred_sets)),
            "n_visited_nodes": int((node_table["state"] == "visited").sum())
            if not node_table.empty else 0,
            "n_unreached_gold": int((node_table["state"] != "visited").sum())
            if not node_table.empty else 0,
            "gold_disagreements": nodes_mod.gold_disagreements(node_table),
        },
        "flat": {k: v for k, v in scored["flat"].items() if k != "per_report"},
        "hierarchy": scored["hierarchy"],
        "ledger": {k: v for k, v in ledger.items() if k != "rows"},
        "per_report": _per_report(report_ids, gold_sets, pred_sets, scored, ledger),
    }

    report["heavy"] = heavy
    report["geometry"] = _geometry(node_table)
    report["traversal"] = _traversal(traversal_df, nodes_df, gold_sets, report_ids)
    report["errors"] = _error_report(gold_sets, pred_sets, view) if heavy else None

    if cell.spec.is_tree:
        report["blocking"] = pruning.blocking_depth_histogram(ledger)
        report["culprits"] = pruning.culprit_leaderboard(
            ledger, tau_sweep=tau_sweep or [], labels=labels)
        report["fp_factory"] = []  # filled by the caller, which holds the children map
        # Separation and the accept curve are two numpy passes and stay in the core report. Only
        # The reliability/AURC block is expensive enough to defer.
        report["separation"] = {
            name: calib.score_separation(node_table, name) for name in calib.SCORES
        }
        report["accept_curve"] = calib.threshold_curve(node_table, "accept")
        report["calibration"] = (
            {name: calib.summary(node_table, name) for name in calib.SCORES} if heavy else None
        )
    else:
        report["blocking"] = {}
        report["culprits"] = []
        report["fp_factory"] = []
        report["calibration"] = {} if heavy else None
        report["separation"] = {}
        report["accept_curve"] = []

    return _clean(report)


#: How many out-of-cohort report ids the block names before it stops counting them out loud. The
#: number is always exact. The list is a sample, because on a run against an early curated dataset
#: it can be most of the cohort and the scorecard is not a place to print two hundred ids.
MAX_OUTSIDE_NAMED = 12


def _curated_block(curated, predicted, report_ids) -> dict | None:
    """What the curated ground truth adds to this cell's report, or ``None`` when it is not that ground truth.

    Three facts, none of which the metrics can show on their own:

    **Coverage.** The curated ground truth covers the reports curation reached, and ``scoring.align``
    silently drops the rest. A scorecard that says "48 reports" where the run wrote 60 must also
    say that the other twelve are outside the cohort, or it reads as a run that lost reports.

    **Excluded false positives.** A predicted term that a curator *saw* and the policy dropped, a family-history finding, an unsure row, a term nobody could evidence location, is a different fact
    about a method than an invented term, and it is the single biggest reason a precision number
    moves when the ground truth is swapped. Counted here rather than in the error taxonomy because it is a
    property of the ground truth, not of the ontology.

    **Provenance.** The dataset's date and its curation log's digest, so a table pasted out of
    this app names the log behind it.
    """
    if curated is None:
        return None
    scored_ids = set(report_ids)
    outside = curated.outside(predicted)
    excluded_fp: dict[str, int] = {}
    n_excluded_fp = 0
    for report_id in sorted(scored_ids):
        gold_here = curated.gold.get(str(report_id), set())
        for code in predicted.get(report_id, ()) or ():
            if str(code) in gold_here:
                continue
            row = curated.excluded(report_id, code)
            if row is None:
                continue
            n_excluded_fp += 1
            reason = row.exclude_reason or "unspecified"
            excluded_fp[reason] = excluded_fp.get(reason, 0) + 1
    return {
        "name": curated.name,
        "path": curated.path,
        "date": curated.date,
        "describe": curated.describe(),
        "policy": curated.policy,
        "curation_log": curated.curation_log_digest,
        "n_reports_in_cohort": len(curated.gold),
        "n_gold_pairs": curated.n_gold_pairs,
        "n_annotations_excluded": curated.n_excluded,
        "n_reports_outside": len(outside),
        "reports_outside": outside[:MAX_OUTSIDE_NAMED],
        "n_excluded_fp": n_excluded_fp,
        "excluded_fp_by_reason": dict(sorted(excluded_fp.items(), key=lambda kv: -kv[1])),
    }


def _per_report(report_ids, gold_sets, pred_sets, scored, ledger) -> list[dict]:
    """One row per report: its P/R/F1 and where its annotated terms went.

    The join between ``scored["flat"]["per_report"]`` and ``report_ids`` is positional, which is
    the contract ``scoring.align`` establishes and the only thing holding the two together.
    """
    fates_by_report: dict[str, Counter] = defaultdict(Counter)
    for row in ledger.get("rows", ()):
        fates_by_report[row["report_id"]][row["fate"]] += 1

    per_report = scored["flat"].get("per_report") or []
    rows = []
    for index, report_id in enumerate(report_ids):
        prf = per_report[index] if index < len(per_report) else None
        fates = fates_by_report.get(report_id, Counter())
        rows.append({
            "report_id": report_id,
            "precision": prf[0] if prf else None,
            "recall": prf[1] if prf else None,
            "f1": prf[2] if prf else None,
            "n_gold": len(gold_sets[index]),
            "n_predicted": len(pred_sets[index]),
            **{f"n_{fate}": fates.get(fate, 0) for fate in pruning.FATES},
        })
    return rows


def _error_report(gold_sets, pred_sets, view) -> dict:
    """The FP taxonomy, the FN taxonomy and the near-miss distance distribution."""
    from hpo_extraction.evaluation.metrics import errors

    return {
        "fp_taxonomy": errors.error_taxonomy(gold_sets, pred_sets, view),
        "fn_taxonomy": errors.false_negative_taxonomy(gold_sets, pred_sets, view),
        "near_miss": errors.near_miss_distribution(gold_sets, pred_sets, view),
    }


def _geometry(frame: pd.DataFrame) -> dict:
    """Where in the ontology the errors sit, by depth, and by layer-1 organ system."""
    if frame is None or frame.empty:
        return {"by_depth": [], "by_layer1": [], "fp_by_accept_score": {}}

    by_depth = []
    depths = frame["depth"].dropna()
    for depth in sorted({int(d) for d in depths}):
        subset = frame[frame["depth"] == depth]
        outcomes = subset["outcome"].value_counts()
        n_gold = int(subset["in_gold"].fillna(False).astype(bool).sum())
        by_depth.append({
            "depth": depth,
            "n_nodes": int(len(subset)),
            "TP": int(outcomes.get("TP", 0)), "FP": int(outcomes.get("FP", 0)),
            "FN": int(outcomes.get("FN", 0)), "TN": int(outcomes.get("TN", 0)),
            "n_gold": n_gold,
            "recall": (int(outcomes.get("TP", 0)) / n_gold) if n_gold else None,
        })

    by_layer1 = []
    if "layer1_organ" in frame.columns:
        for organ, subset in frame.groupby(frame["layer1_organ"].fillna("(unknown)")):
            outcomes = subset["outcome"].value_counts()
            n_gold = int(subset["in_gold"].fillna(False).astype(bool).sum())
            by_layer1.append({
                "layer1_organ": str(organ),
                "TP": int(outcomes.get("TP", 0)), "FP": int(outcomes.get("FP", 0)),
                "FN": int(outcomes.get("FN", 0)),
                "n_gold": n_gold,
                "recall": (int(outcomes.get("TP", 0)) / n_gold) if n_gold else None,
            })
        by_layer1.sort(key=lambda r: (-r["FP"], -r["n_gold"], r["layer1_organ"]))

    # Are the false positives confident, or are they sitting just above τ_accept? The answer
    # decides whether raising τ_accept is a fix or a trade.
    fp_scores = frame.loc[frame["outcome"] == "FP", "accept_score"].dropna()
    tp_scores = frame.loc[frame["outcome"] == "TP", "accept_score"].dropna()
    edges = np.linspace(0.0, 1.0, 21)
    return {
        "by_depth": by_depth,
        "by_layer1": by_layer1[:30],
        "fp_by_accept_score": {
            "edges": [float(e) for e in edges],
            "FP": [int(v) for v in np.histogram(fp_scores, bins=edges)[0]],
            "TP": [int(v) for v in np.histogram(tp_scores, bins=edges)[0]],
        },
    }


def _traversal(traversal_df: pd.DataFrame, nodes_df: pd.DataFrame,
               gold_sets, report_ids) -> dict:
    """Inference cost at this configuration, and the per-depth cost/recall exchange.

    ``n_slm_calls`` comes from ``*_traversal.jsonl``, not from the calls artifact: the driver
    memoises node evaluation across the whole τ sweep and writes the calls once, so counting call
    records would charge every τ the union of the sweep's work.
    """
    out: dict = {"n_reports": 0, "total_slm_calls": None, "mean_slm_calls": None,
                 "mean_nodes_visited": None, "mean_frontier_insertions": None,
                 "depth_cost": []}
    if traversal_df is not None and not traversal_df.empty:
        out["n_reports"] = int(len(traversal_df))
        if "n_slm_calls" in traversal_df:
            calls = pd.to_numeric(traversal_df["n_slm_calls"], errors="coerce")
            out["total_slm_calls"] = int(calls.sum())
            out["mean_slm_calls"] = float(calls.mean())
        if "n_unique_nodes_visited" in traversal_df:
            visited = pd.to_numeric(traversal_df["n_unique_nodes_visited"], errors="coerce")
            out["mean_nodes_visited"] = float(visited.mean())
        if "total_frontier_insertions" in traversal_df:
            inserted = pd.to_numeric(traversal_df["total_frontier_insertions"], errors="coerce")
            out["mean_frontier_insertions"] = float(inserted.mean())

    out["depth_cost"] = _depth_cost(nodes_df, gold_sets, report_ids)
    return out


def _depth_cost(nodes_df: pd.DataFrame, gold_sets, report_ids) -> list[dict]:
    """``thesis_metrics.traversal.depth_cost_table`` over this configuration's node records."""
    from hpo_extraction.evaluation.metrics import traversal as traversal_metrics

    if nodes_df is None or nodes_df.empty or not report_ids:
        return []

    # observed=True: report_id is categorical (loaders._nodes_frame), and a groupby over a
    # categorical otherwise yields an empty group for every unused category.
    positions = {str(rid): idx
                 for rid, idx in nodes_df.groupby("report_id", observed=True,
                                                  sort=True).indices.items()}
    # Codes + categories, not an object array: materialising ``hpo_id`` as strings for the
    # whole cohort is the very allocation this function is being taken off. ``numpy.str_``
    # subclasses ``str``, so the ``hpo_id in gold`` test downstream is unaffected.
    ids = nodes_df["hpo_id"]
    if not isinstance(ids.dtype, pd.CategoricalDtype):
        ids = ids.astype(str).astype("category")
    codes = ids.cat.codes.to_numpy()
    categories = ids.cat.categories.astype(str).to_numpy()
    depth = nodes_df["depth"].to_numpy()
    expanded = nodes_df["expanded"].to_numpy(dtype=bool)
    accepted = nodes_df["accepted"].to_numpy(dtype=bool)

    def visits():
        """One report's ``{hpo_id: record}`` at a time, so the cohort's never exists at once.

        ``depth_cost_table`` accumulates per report and independently, so it consumes this
        lazily. Materialising it was one small dict per visited node across the whole cohort, a million of them at the loosest swept tau_prune, for a table of ~15 rows.
        """
        for rid in report_ids:
            idx = positions.get(str(rid))
            if idx is None:
                yield {}
                continue
            yield {categories[codes[i]]: {"depth": int(depth[i]),
                                          "expanded": bool(expanded[i]),
                                          "accepted": bool(accepted[i]), "n_slm_calls": 0}
                   for i in idx}

    return traversal_metrics.depth_cost_table(visits(), gold_sets)


def tau_frontier(rows: list[dict]) -> list[dict]:
    """The cell's sweep as one table: cost against quality, one row per configuration.

    This is the only view of a tree run in which τ_prune is a *free* variable, and it is free
    because earlier materialised the sweep on disk, not leaving it to be re-run. A re-run
    would have to invent scores for the nodes each τ never visited. Reading the directories
    invents nothing. Rows are whatever the caller managed to load, in sweep order, a τ whose
    directory is missing or unreadable is simply absent, never interpolated.
    """
    ordered = sorted(rows, key=lambda r: (r.get("op_value") is None, r.get("op_value") or 0.0))
    for row in ordered:
        recall = row.get("micro_recall")
        ceiling = row.get("recall_ceiling")
        row["identification_gap"] = (
            ceiling - recall if recall is not None and ceiling is not None else None
        )
    return ordered
