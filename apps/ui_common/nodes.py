"""The per-(patient, HPO) node table, the fat intermediate every drill-down view reads.

One row per node the run touched, plus one row per ground-truth term the traversal never
reached. This is *not* the ``*_predictions.jsonl`` view of the world: that
file is collapsed onto the 9 HCY target symptoms, whereas tree geometry only becomes
visible on the **raw** node set (every HPO id the traversal judged). Same convention as
``scripts/analysis/tree_diagnostics.py``.

Stays server-side. Views receive figures, or a page of rows, never the frame.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from hpo_extraction.evaluation.tree_error_analysis import classify_error, nearest_gt, primary_layer1
from hpo_extraction.evaluation.tree_metrics import resolve_code

POSITIVE_TOKEN = "Yes"  # matches HCYDataset.get_predictions' substring test

# Verdict/response sentinels → the node's traversal state.
_SENTINEL_STATE = {
    "PRUNED": "pruned",
    "SKIPPED": "skipped",
    "ASSUMED_YES": "assumed_yes",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


# ── raw node sets (the tree_diagnostics contract) ────────────────────────────

def predicted_sets_from_response_dict(response_dict: dict) -> dict[str, set[str]]:
    """Raw predicted-positive HPO set per patient: every node whose response says Yes.

    Mirrors ``tree_diagnostics.predicted_sets_from_response_dict``, including the
    substring test, which means the ``ASSUMED_YES`` sentinel counts as positive, as it does
    in ``HCYDataset.get_predictions``.
    """
    out: dict[str, set[str]] = {}
    for patient, hpo_map in response_dict.items():
        positives: set[str] = set()
        for hpo_id, resp in hpo_map.items():
            text = resp.get(0, {}).get("response", "") if isinstance(resp, dict) else ""
            if POSITIVE_TOKEN in text:
                positives.add(hpo_id)
        out[patient] = positives
    return out


def predicted_sets_from_segments(segments: list[dict]) -> dict[str, set[str]]:
    """Fallback for runs with no response pickle (an earlier exploratory run): any 'Yes' segment ⇒ positive."""
    out: dict[str, set[str]] = {}
    for row in segments:
        out.setdefault(row["patient_id"], set())
        if POSITIVE_TOKEN in str(row.get("slm_verdict", "")):
            out[row["patient_id"]].add(row["hpo_id"])
    return out


def predicted_sets(raw: dict) -> dict[str, set[str]]:
    """``{report: set of HPO identifiers}`` predicted positive, from whichever record the run kept."""
    if raw.get("response_dict"):
        return predicted_sets_from_response_dict(raw["response_dict"])
    if raw.get("segments"):
        return predicted_sets_from_segments(raw["segments"])
    return {}


# ── per-node evidence, assembled from whichever artifacts the run has ────────

def _segment_index(segments: list[dict] | None) -> dict[tuple[str, str], list[dict]]:
    """Group segment rows by (patient, hpo), dropping the rank-0 state stubs."""
    index: dict[tuple[str, str], list[dict]] = {}
    for row in segments or []:
        if str(row.get("slm_verdict", "")) in _SENTINEL_STATE:
            continue  # a stub row, carrying no retrieved text
        index.setdefault((row["patient_id"], row["hpo_id"]), []).append(row)
    for rows in index.values():
        rows.sort(key=lambda r: (r.get("cosine_rank") or r.get("rank") or 0))
    return index


def _node_state(entry: dict) -> str:
    """Traversal state of one node, from its response-dict entry."""
    response = str(entry.get(0, {}).get("response", "")) if isinstance(entry, dict) else ""
    for sentinel, state in _SENTINEL_STATE.items():
        if response == sentinel:
            return state
    return "evaluated"


def _node_confidence(entry: dict | None, seg_rows: list[dict]) -> tuple[float | None, float | None]:
    """(confidence, max_margin) for a node, preferring what the run persisted.

    An earlier exploratory run/01 store ``confidence`` + ``max_margin`` on the node. An earlier exploratory run stores
    ``confidence`` + ``logit_margin``. An earlier exploratory run/an earlier exploratory run store only per-sentence margins.
    Where only margins exist we recompute the AnyYes aggregation the pipeline itself uses:
    ``conf = max(sigmoid(margin))``, the same rule as ``scripts/sweep_tau.py``.
    """
    if isinstance(entry, dict):
        for sub in entry.values():
            if not isinstance(sub, dict):
                continue
            if sub.get("confidence") is not None:
                margin = sub.get("max_margin", sub.get("logit_margin"))
                return float(sub["confidence"]), (float(margin) if margin is not None else None)

    margins = [float(r["logit_margin"]) for r in seg_rows if r.get("logit_margin") is not None]
    if not margins:
        return None, None
    return max(_sigmoid(m) for m in margins), max(margins)


def _label(tree, hpo_id: str, fallback: str = "") -> str:
    try:
        return tree.getNameByHPO(hpo_id)
    except (KeyError, IndexError):
        return fallback or hpo_id


# ── the table ────────────────────────────────────────────────────────────────

def build_node_table(raw: dict, gt_dict: dict[str, list[str]], tree) -> pd.DataFrame:
    """One row per (patient, HPO) the run touched, plus never-traversed GT terms.

    Columns:
        patient_id, hpo_id, hpo_label, depth, layer1, layer1_name
        state        evaluated | pruned | skipped | assumed_yes | not_traversed
        predicted    bool  (the raw node-level Yes)
        in_gt        bool
        outcome      TP | FP | FN | TN
        confidence, max_margin, n_segments, top_text, top_cosine
        error_class, nearest_code, nearest_label, nearest_sim
            For an FP: how the prediction relates to its nearest GT term.
            For an FN: how the nearest *prediction* relates to the missed truth.
            Both answer the same question, "how did the closest guess relate to the truth", so the column is comparable across error kinds.
    """
    response_dict: dict = raw.get("response_dict") or {}
    seg_index = _segment_index(raw.get("segments"))
    preds = predicted_sets(raw)

    # Patients the run actually produced output for, intersected with GT (the scoring set).
    patients = sorted(set(preds) & set(gt_dict))

    rows: list[dict[str, Any]] = []
    for patient in patients:
        gt_raw = gt_dict.get(patient, [])
        gt_norm = {resolve_code(tree, g) for g in gt_raw} - {None}
        pred_raw = preds.get(patient, set())
        pred_norm = {resolve_code(tree, p) for p in pred_raw} - {None}

        # Every node the traversal touched (from the pickle, else from the segments).
        touched = set(response_dict.get(patient, {}))
        if not touched:
            touched = {hpo for (pid, hpo) in seg_index if pid == patient}

        for hpo_id in sorted(touched):
            entry = response_dict.get(patient, {}).get(hpo_id)
            seg_rows = seg_index.get((patient, hpo_id), [])
            state = _node_state(entry) if entry is not None else "evaluated"
            confidence, max_margin = _node_confidence(entry, seg_rows)

            canonical = resolve_code(tree, hpo_id)
            predicted = hpo_id in pred_raw
            in_gt = canonical is not None and canonical in gt_norm
            top = seg_rows[0] if seg_rows else {}

            rows.append({
                "patient_id": patient,
                "hpo_id": hpo_id,
                "hpo_label": _label(tree, hpo_id, str(top.get("hpo_label", ""))),
                "state": state,
                "predicted": predicted,
                "in_gt": in_gt,
                "outcome": _outcome(predicted, in_gt),
                "confidence": confidence,
                "max_margin": max_margin,
                "n_segments": len(seg_rows),
                "top_text": str(top.get("text", "")),
                "top_cosine": top.get("cosine_sim"),
                **_tree_position(tree, canonical),
            })

        # GT terms the traversal never visited: real misses that would otherwise be invisible,
        # since they appear in no artifact the run wrote.
        for missed in sorted(gt_norm - {resolve_code(tree, h) for h in touched}):
            rows.append({
                "patient_id": patient,
                "hpo_id": missed,
                "hpo_label": _label(tree, missed),
                "state": "not_traversed",
                "predicted": False,
                "in_gt": True,
                "outcome": "FN",
                "confidence": None,
                "max_margin": None,
                "n_segments": 0,
                "top_text": "",
                "top_cosine": None,
                **_tree_position(tree, missed),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _annotate_errors(df, gt_dict, preds, tree)


def _outcome(predicted: bool, in_gt: bool) -> str:
    if predicted and in_gt:
        return "TP"
    if predicted:
        return "FP"
    if in_gt:
        return "FN"
    return "TN"


def _tree_position(tree, canonical: str | None) -> dict:
    if canonical is None:
        return {"depth": None, "layer1": "None", "layer1_name": "None"}
    layer1 = primary_layer1(tree, canonical)
    return {
        "depth": tree.depth_dict.get(canonical),
        "layer1": layer1,
        "layer1_name": _label(tree, layer1) if layer1 != "None" else "None",
    }


def _annotate_errors(df: pd.DataFrame, gt_dict, preds, tree) -> pd.DataFrame:
    """Attach the tree relationship between each error and the closest thing on the other side.

    Only errors get annotated, Wu-Palmer over every TN would dominate the load time for
    nothing, since a TN has no error geometry to describe.
    """
    error_class: list[Any] = [None] * len(df)
    nearest_code: list[Any] = [None] * len(df)
    nearest_label: list[Any] = [None] * len(df)
    nearest_sim: list[Any] = [None] * len(df)

    by_patient: dict[str, tuple[set, set]] = {}
    for patient in df["patient_id"].unique():
        gt_norm = {resolve_code(tree, g) for g in gt_dict.get(patient, [])} - {None}
        pred_norm = {resolve_code(tree, p) for p in preds.get(patient, set())} - {None}
        by_patient[patient] = (gt_norm, pred_norm)

    for i, row in enumerate(df.itertuples(index=False)):
        if row.outcome not in ("FP", "FN"):
            continue
        gt_norm, pred_norm = by_patient[row.patient_id]
        # An FP is judged against the truth. An FN against whatever we did predict.
        other = gt_norm if row.outcome == "FP" else pred_norm
        if not other:
            error_class[i] = "no_gt" if row.outcome == "FP" else "none_predicted"
            nearest_sim[i] = 0.0
            continue
        code, sim = nearest_gt(row.hpo_id, other, tree)
        if code is None:
            error_class[i] = "unrelated"
            nearest_sim[i] = 0.0
            continue
        pred_side, truth_side = (row.hpo_id, code) if row.outcome == "FP" else (code, row.hpo_id)
        error_class[i] = classify_error(pred_side, truth_side, tree)
        nearest_code[i] = code
        nearest_label[i] = _label(tree, code)
        nearest_sim[i] = sim

    df["error_class"] = error_class
    df["nearest_code"] = nearest_code
    df["nearest_label"] = nearest_label
    df["nearest_sim"] = nearest_sim
    return df
