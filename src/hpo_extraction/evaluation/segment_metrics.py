"""Segment-level retrieval and classification metrics for PhenoRAG.

Pure functions with no IO, importable from both run.py and the dashboard.
All functions operate on *joined segment records* (list[dict]) where each dict
contains at minimum:

    patient_id   str
    hpo_id       str
    rank         int   (1-based)
    cosine_sim   float
    slm_verdict  str   ("Yes" | "No")
    is_gt_relevant bool | None   (None = GT not available)
"""

from __future__ import annotations

from collections import defaultdict


def compute_segment_recall_at_k(
    records: list[dict],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Fraction of (patient, HPO) pairs where ≥1 GT-relevant segment is in top-K.

    Only considers pairs where at least one record has is_gt_relevant set (not None).
    Returns empty dict if no GT-relevant information is present.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Group by (patient, hpo)
    pairs: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("is_gt_relevant") is None:
            continue
        pairs[(r["patient_id"], r["hpo_id"])].append(r)

    if not pairs:
        return {}

    result = {}
    for k in k_values:
        n_found = 0
        for segs in pairs.values():
            top_k = [s for s in segs if s["rank"] <= k]
            if any(s["is_gt_relevant"] for s in top_k):
                n_found += 1
        result[k] = n_found / len(pairs)

    return result


def compute_mrr(records: list[dict]) -> float:
    """Mean Reciprocal Rank of the first GT-relevant segment per (patient, HPO) pair.

    Pairs with no GT-relevant segment in any rank contribute 0 to the mean.
    Returns 0.0 if no GT information is available.
    """
    pairs: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("is_gt_relevant") is None:
            continue
        pairs[(r["patient_id"], r["hpo_id"])].append(r)

    if not pairs:
        return 0.0

    rr_sum = 0.0
    for segs in pairs.values():
        rel_segs = sorted(
            [s for s in segs if s["is_gt_relevant"]],
            key=lambda s: s["rank"],
        )
        if rel_segs:
            rr_sum += 1.0 / rel_segs[0]["rank"]

    return rr_sum / len(pairs)


def compute_error_decomposition(
    records: list[dict],
    predictions: list[dict],
) -> dict[str, int]:
    """Decompose prediction errors into retrieval and classification failures.

    For GT-positive pairs (ground_truth == 1):
      retrieval_failure     GT+ segment never in top-K
      classification_failure GT+ segment retrieved but LLM voted No for all of them
      true_positive         GT+ segment retrieved, LLM voted Yes, prediction == 1
      false_negative_other  GT+ segment retrieved, LLM voted Yes, but prediction == 0
                            (can happen with non-AnyYes aggregation)

    For GT-negative pairs (ground_truth == 0):
      false_positive        prediction == 1 (LLM wrongly confirmed negative HPO)
      true_negative         prediction == 0

    Pairs where ground_truth is None are skipped.
    """
    pred_map = {
        (r["patient_id"], r["hpo_id"]): r
        for r in predictions
    }

    seg_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        seg_groups[(r["patient_id"], r["hpo_id"])].append(r)

    counts: dict[str, int] = {
        "retrieval_failure": 0,
        "classification_failure": 0,
        "true_positive": 0,
        "false_negative_other": 0,
        "false_positive": 0,
        "true_negative": 0,
    }

    for key, segs in seg_groups.items():
        pred_rec = pred_map.get(key)
        if pred_rec is None:
            continue
        gt = pred_rec.get("ground_truth")
        if gt is None:
            continue
        gt = int(gt)
        pred = int(pred_rec.get("prediction", 0))

        if gt == 1:
            # Only count if we have GT relevance information
            if all(s.get("is_gt_relevant") is None for s in segs):
                continue
            gt_pos_segs = [s for s in segs if s.get("is_gt_relevant")]
            if not gt_pos_segs:
                counts["retrieval_failure"] += 1
            else:
                gt_pos_yes = [s for s in gt_pos_segs if s.get("slm_verdict") == "Yes"]
                if not gt_pos_yes:
                    counts["classification_failure"] += 1
                elif pred == 1:
                    counts["true_positive"] += 1
                else:
                    counts["false_negative_other"] += 1
        else:
            if pred == 1:
                counts["false_positive"] += 1
            else:
                counts["true_negative"] += 1

    return counts
