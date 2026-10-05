"""The RunReport, one compact, JSON-serialisable dict per run.

This is the only thing the scorecard and geometry views ever read, and the only thing that
crosses the wire. Everything expensive (the node table, the confidence table, the raw
segments) stays behind it in :mod:`registry`'s in-process cache.

Graded and geometry numbers follow ``scripts/analysis/tree_diagnostics.py``, same raw
node sets, same patient intersection, same ``filter_unannotated`` semantics, so the UI and the
CLI are two views of one computation, and a discrepancy is a bug rather than a convention.
"""

from __future__ import annotations

import pandas as pd

from hpo_extraction.evaluation.tree_error_analysis import (
    ERROR_LABELS,
    depth_stratified_recall,
    granularity_vector,
    layer1_performance,
    miss_similarity_histogram,
)
from hpo_extraction.evaluation.tree_metrics import evaluate_graded, normalize_set

from . import calib, nodes as nodes_mod, prune_audit, scoring

# granularity_vector emits a Counter, so zero-count labels are simply absent. Zero-filling here
# means a chart never silently drops a category, an empty "unrelated" bar is information.
GRANULARITY_LABELS = tuple(ERROR_LABELS) + ("no_gt",)

# What an earlier exploratory run / an earlier exploratory run write into the prompt field instead of the prompt.
PLACEHOLDER_PROMPT = "(calibrated node verdict)"


def align_sets(gt_dict, pred_dict, filter_unannotated: bool = True):
    """(gt_sets, pred_sets, patient_ids) over the patients both sides have.

    Mirrors ``tree_diagnostics.align_sets`` / ``_patient_ids``.
    """
    patients = set(gt_dict) & set(pred_dict)
    if filter_unannotated:
        patients = {p for p in patients if gt_dict.get(p)}
    patient_ids = sorted(patients)
    gt_sets = [set(gt_dict.get(p, [])) for p in patient_ids]
    pred_sets = [set(pred_dict.get(p, set())) for p in patient_ids]
    return gt_sets, pred_sets, patient_ids


def _has_real_prompts(response_dict: dict | None) -> bool:
    """Whether the pickle carries actual prompt text.

    An earlier exploratory run and an earlier exploratory run write the placeholder ``"(calibrated node verdict)"`` into the
    prompt field, so a non-empty string is not enough to go on, the patient view must fall
    back to reconstructing the prompt for those runs, and must say so.
    """
    for hpo_map in (response_dict or {}).values():
        for entry in hpo_map.values():
            if not isinstance(entry, dict):
                continue
            for sub in entry.values():
                if not isinstance(sub, dict):
                    continue
                prompt = sub.get("prompt", "")
                if prompt and prompt != PLACEHOLDER_PROMPT:
                    return True
    return False


def _capabilities(raw: dict, nodes: pd.DataFrame) -> dict:
    segments = raw.get("segments") or []
    has_margins = any(r.get("logit_margin") is not None for r in segments) or (
        not nodes.empty and nodes["confidence"].notna().any()
    )
    has_prompts = _has_real_prompts(raw.get("response_dict"))

    return {
        "has_margins": bool(has_margins),
        "has_pruning": bool(not nodes.empty and (nodes["state"] == "pruned").any()),
        "has_skipping": bool(not nodes.empty and (nodes["state"] == "skipped").any()),
        "has_segments": bool(segments),
        "has_response_dict": bool(raw.get("response_dict")),
        "has_prompts": has_prompts,
        "has_tau_sweep": bool(raw.get("tau_sweep")),
        "has_timing": bool(raw.get("timing")),
        "conf_table_unpruned": bool(has_margins) and calib.conf_table_is_unpruned(raw, nodes),
    }


def _traversal(nodes: pd.DataFrame, raw: dict) -> dict:
    if nodes.empty:
        return {}
    counts = nodes["state"].value_counts().to_dict()
    n_evaluated = int(counts.get("evaluated", 0))
    n_pruned = int(counts.get("pruned", 0))
    n_skipped = int(counts.get("skipped", 0))
    n_asked = n_evaluated + n_pruned
    return {
        "n_evaluated": n_evaluated,
        "n_pruned": n_pruned,
        "n_skipped": n_skipped,
        "n_assumed_yes": int(counts.get("assumed_yes", 0)),
        "n_nodes": int(len(nodes)),
        # Pruned nodes are the LLM calls the tree walk avoided.
        "llm_call_savings_pct": (100.0 * n_pruned / n_asked) if n_asked else 0.0,
    }


def _timing(raw: dict) -> dict:
    records = raw.get("timing") or []
    latencies = [
        float(r["judgment_latency_s"]) for r in records if r.get("judgment_latency_s") is not None
    ]
    if not latencies:
        return {}
    return {
        "mean_time_per_report_s": sum(latencies) / len(latencies),
        "total_time_s": sum(latencies),
        "per_patient": [
            {"patient_id": r.get("patient_id"), "latency_s": r.get("judgment_latency_s"),
             "n_evaluated": r.get("n_evaluated"), "n_pruned": r.get("n_pruned")}
            for r in records
        ],
    }


def _per_patient(graded: dict, patient_ids: list[str], gt_sets, pred_sets, nodes: pd.DataFrame) -> list[dict]:
    """Zip evaluate_graded's positional per_sample lists back onto patient ids.

    ``evaluate_graded`` returns three parallel F1 lists with no ids attached, the alignment is
    purely positional against the list order it was handed. tree_diagnostics discards this. Keeping it is what makes "show me the worst patients" a click instead of a rerun.
    """
    per_sample = graded.get("per_sample", {})
    pruned_by_patient = (
        nodes[nodes["state"] == "pruned"].groupby("patient_id").size().to_dict()
        if not nodes.empty else {}
    )
    out: list[dict] = []
    for i, pid in enumerate(patient_ids):
        out.append({
            "patient_id": pid,
            "exact_f1": per_sample.get("exact_f1", [])[i] if i < len(per_sample.get("exact_f1", [])) else None,
            "soft_f1": per_sample.get("soft_f1", [])[i] if i < len(per_sample.get("soft_f1", [])) else None,
            "closure_f1": per_sample.get("closure_f1", [])[i] if i < len(per_sample.get("closure_f1", [])) else None,
            "n_gt": len(gt_sets[i]),
            "n_pred": len(pred_sets[i]),
            "n_pruned": int(pruned_by_patient.get(pid, 0)),
        })
    return out


def build_report(
    run_id: str,
    raw: dict,
    nodes: pd.DataFrame,
    gt_dict: dict[str, list[str]],
    target_symptoms: list[str],
    tree,
) -> dict:
    """Everything the scorecard, geometry and calibration views need, and nothing more."""
    preds = nodes_mod.predicted_sets(raw)
    gt_sets, pred_sets, patient_ids = align_sets(gt_dict, preds, filter_unannotated=True)

    graded = evaluate_graded(gt_sets, pred_sets, tree)

    gv = granularity_vector(gt_sets, pred_sets, tree)
    gv["counts"] = {label: gv["counts"].get(label, 0) for label in GRANULARITY_LABELS}
    gv["fractions"] = {label: gv["fractions"].get(label, 0.0) for label in GRANULARITY_LABELS}

    # Depth keys are ints, and would come back from the JSON disk cache as strings. Stringify
    # once, here, so every consumer sees one type.
    depth_recall = {
        str(d): v for d, v in depth_stratified_recall(gt_sets, pred_sets, tree).items()
    }

    # The main number the experiment itself reported. Recomputed (rather than read from
    # {variant}.csv) when the run never wrote one.
    target_metrics = raw.get("headline")
    if target_metrics is None and raw.get("response_dict"):
        target_metrics = scoring.evaluate_targets(
            raw["response_dict"], target_symptoms, gt_dict, tree,
        )

    return {
        "run_id": run_id,
        "run_dir": raw["run_dir"],
        "variant": raw["variant"],
        "capabilities": _capabilities(raw, nodes),
        "counts": {
            "n_patients": len(patient_ids),
            "n_gt_terms": sum(len(g) for g in gt_sets),
            "n_pred_terms": sum(len(p) for p in pred_sets),
            "unscorable_gt": sum(normalize_set(g, tree)[1] for g in gt_sets),
            "unscorable_pred": sum(normalize_set(p, tree)[1] for p in pred_sets),
        },
        "graded": graded,
        "target_metrics": target_metrics,
        "geometry": {
            "granularity_vector": gv,
            "depth_stratified_recall": depth_recall,
            "miss_similarity_histogram": miss_similarity_histogram(gt_sets, pred_sets, tree),
            "layer1_performance": layer1_performance(gt_sets, pred_sets, tree),
            "depth_confidence": _depth_confidence(nodes),
        },
        "traversal": _traversal(nodes, raw),
        "prune_audit": prune_audit.audit(nodes, gt_dict, tree),
        "calibration": calib.calibration_summary(nodes),
        "timing": _timing(raw),
        "per_patient": _per_patient(graded, patient_ids, gt_sets, pred_sets, nodes),
        "config": raw.get("config"),
    }


def _depth_confidence(nodes: pd.DataFrame) -> list[dict]:
    """Mean confidence by tree depth, does the model get less certain the deeper it goes?"""
    if nodes.empty or nodes["confidence"].isna().all():
        return []
    scored = nodes[(nodes["state"] == "evaluated") & nodes["confidence"].notna() & nodes["depth"].notna()]
    if scored.empty:
        return []
    grouped = scored.groupby("depth").agg(
        mean_confidence=("confidence", "mean"),
        median_confidence=("confidence", "median"),
        n=("confidence", "size"),
        positive_rate=("in_gt", "mean"),
    )
    return [
        {"depth": int(depth), **{k: float(v) for k, v in row.items()}}
        for depth, row in grouped.iterrows()
    ]
