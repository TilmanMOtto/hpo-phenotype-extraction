"""Unit tests for hpo_extraction.evaluation.segment_metrics.

All expected values are hand-computed. See inline comments for derivations.
"""

import pytest

from hpo_extraction.evaluation.segment_metrics import (
    compute_error_decomposition,
    compute_mrr,
    compute_segment_recall_at_k,
)


# ---------------------------------------------------------------------------
# Record factory (inline. No fixture needed since these are tiny dicts)
# ---------------------------------------------------------------------------

def rec(patient_id, hpo_id, rank, is_gt_relevant=None, slm_verdict="No"):
    return {
        "patient_id": patient_id,
        "hpo_id": hpo_id,
        "rank": rank,
        "cosine_sim": 0.5,
        "slm_verdict": slm_verdict,
        "is_gt_relevant": is_gt_relevant,
    }


def pred(patient_id, hpo_id, ground_truth, prediction):
    return {
        "patient_id": patient_id,
        "hpo_id": hpo_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# compute_segment_recall_at_k
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_recall_at_k_all_gt_relevant_none_returns_empty_dict():
    records = [rec("p1", "HP:1", 1, None)]
    assert compute_segment_recall_at_k(records) == {}


@pytest.mark.unit
def test_recall_at_k_empty_records_returns_empty_dict():
    assert compute_segment_recall_at_k([]) == {}


@pytest.mark.unit
def test_recall_at_k_single_hit_at_rank_1():
    # 1 pair, GT-relevant at rank 1 → recall@1 = 1/1 = 1.0
    records = [rec("p1", "HP:1", 1, True)]
    result = compute_segment_recall_at_k(records, k_values=[1])
    assert result[1] == pytest.approx(1.0)


@pytest.mark.unit
def test_recall_at_k_miss_at_rank_1_hit_at_rank_2():
    # 1 pair, GT-relevant at rank 2 → recall@1=0.0, recall@2=1.0
    records = [rec("p1", "HP:1", 1, False), rec("p1", "HP:1", 2, True)]
    result = compute_segment_recall_at_k(records, k_values=[1, 2])
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


@pytest.mark.unit
def test_recall_at_k_two_pairs_one_hit():
    # pair1: GT-relevant at rank 1 → hit
    # pair2: GT-relevant not in top-1 → miss
    # recall@1 = 1/2 = 0.5
    records = [
        rec("p1", "HP:1", 1, True),
        rec("p2", "HP:2", 1, False),
    ]
    result = compute_segment_recall_at_k(records, k_values=[1])
    assert result[1] == pytest.approx(0.5)


@pytest.mark.unit
def test_recall_at_k_default_k_values_present():
    records = [rec("p1", "HP:1", 1, True)]
    result = compute_segment_recall_at_k(records)
    assert 1 in result
    assert 3 in result
    assert 5 in result
    assert 10 in result


@pytest.mark.unit
def test_recall_at_k_mixed_none_and_valid_skips_none():
    # Only records with is_gt_relevant != None count
    records = [
        rec("p1", "HP:1", 1, None),   # skipped
        rec("p2", "HP:2", 1, True),   # counts
    ]
    result = compute_segment_recall_at_k(records, k_values=[1])
    # Only 1 valid pair, hit → 1.0
    assert result[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_mrr
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mrr_no_gt_info_returns_zero():
    records = [rec("p1", "HP:1", 1, None)]
    assert compute_mrr(records) == pytest.approx(0.0)


@pytest.mark.unit
def test_mrr_empty_records_returns_zero():
    assert compute_mrr([]) == pytest.approx(0.0)


@pytest.mark.unit
def test_mrr_first_relevant_at_rank_1_returns_one():
    # RR = 1/1 = 1.0; MRR = 1.0
    records = [rec("p1", "HP:1", 1, True)]
    assert compute_mrr(records) == pytest.approx(1.0)


@pytest.mark.unit
def test_mrr_first_relevant_at_rank_2_returns_half():
    # RR = 1/2 = 0.5; MRR = 0.5
    records = [
        rec("p1", "HP:1", 1, False),
        rec("p1", "HP:1", 2, True),
    ]
    assert compute_mrr(records) == pytest.approx(0.5)


@pytest.mark.unit
def test_mrr_first_relevant_at_rank_5():
    # RR = 1/5 = 0.2
    records = [rec("p1", "HP:1", k, k == 5) for k in range(1, 6)]
    assert compute_mrr(records) == pytest.approx(0.2)


@pytest.mark.unit
def test_mrr_two_pairs_averages_reciprocal_ranks():
    # pair1 (p1,HP:1): first relevant at rank 1 → RR=1.0
    # pair2 (p2,HP:2): first relevant at rank 2 → RR=0.5
    # MRR = (1.0 + 0.5) / 2 = 0.75
    records = [
        rec("p1", "HP:1", 1, True),
        rec("p2", "HP:2", 1, False),
        rec("p2", "HP:2", 2, True),
    ]
    assert compute_mrr(records) == pytest.approx(0.75)


@pytest.mark.unit
def test_mrr_no_relevant_segment_in_any_rank_contributes_zero():
    # Pair with no relevant segment → RR=0
    records = [rec("p1", "HP:1", 1, False), rec("p1", "HP:1", 2, False)]
    assert compute_mrr(records) == pytest.approx(0.0)


@pytest.mark.unit
def test_mrr_returns_float():
    assert isinstance(compute_mrr([]), float)


# ---------------------------------------------------------------------------
# compute_error_decomposition
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_error_decomp_retrieval_failure():
    # GT=1, no gt_relevant segments in retrieved → retrieval_failure
    records = [rec("p1", "HP:1", 1, False)]
    preds = [pred("p1", "HP:1", 1, 0)]
    counts = compute_error_decomposition(records, preds)
    assert counts["retrieval_failure"] == 1


@pytest.mark.unit
def test_error_decomp_classification_failure():
    # GT=1, gt_relevant retrieved but LLM voted No → classification_failure
    records = [rec("p1", "HP:1", 1, True, "No")]
    preds = [pred("p1", "HP:1", 1, 0)]
    counts = compute_error_decomposition(records, preds)
    assert counts["classification_failure"] == 1


@pytest.mark.unit
def test_error_decomp_true_positive():
    # GT=1, gt_relevant retrieved, LLM voted Yes, prediction=1 → true_positive
    records = [rec("p1", "HP:1", 1, True, "Yes")]
    preds = [pred("p1", "HP:1", 1, 1)]
    counts = compute_error_decomposition(records, preds)
    assert counts["true_positive"] == 1


@pytest.mark.unit
def test_error_decomp_false_negative_other():
    # GT=1, gt_relevant retrieved, LLM voted Yes, but prediction=0 → false_negative_other
    records = [rec("p1", "HP:1", 1, True, "Yes")]
    preds = [pred("p1", "HP:1", 1, 0)]
    counts = compute_error_decomposition(records, preds)
    assert counts["false_negative_other"] == 1


@pytest.mark.unit
def test_error_decomp_false_positive():
    # GT=0, prediction=1 → false_positive
    records = [rec("p1", "HP:1", 1, False, "Yes")]
    preds = [pred("p1", "HP:1", 0, 1)]
    counts = compute_error_decomposition(records, preds)
    assert counts["false_positive"] == 1


@pytest.mark.unit
def test_error_decomp_true_negative():
    # GT=0, prediction=0 → true_negative
    records = [rec("p1", "HP:1", 1, False, "No")]
    preds = [pred("p1", "HP:1", 0, 0)]
    counts = compute_error_decomposition(records, preds)
    assert counts["true_negative"] == 1


@pytest.mark.unit
def test_error_decomp_all_buckets_present_in_output():
    counts = compute_error_decomposition([], [])
    expected_keys = {
        "retrieval_failure", "classification_failure", "true_positive",
        "false_negative_other", "false_positive", "true_negative",
    }
    assert set(counts.keys()) == expected_keys


@pytest.mark.unit
def test_error_decomp_skips_pair_missing_from_predictions():
    records = [rec("p1", "HP:1", 1, True, "Yes")]
    preds = []  # no matching prediction
    counts = compute_error_decomposition(records, preds)
    assert sum(counts.values()) == 0


@pytest.mark.unit
def test_error_decomp_skips_pair_with_none_ground_truth():
    records = [rec("p1", "HP:1", 1, True, "Yes")]
    preds = [{"patient_id": "p1", "hpo_id": "HP:1", "ground_truth": None, "prediction": 1}]
    counts = compute_error_decomposition(records, preds)
    assert sum(counts.values()) == 0


@pytest.mark.unit
def test_error_decomp_skips_gt_positive_pair_with_all_none_is_gt_relevant():
    # GT=1 but all segs have is_gt_relevant=None → skipped
    records = [rec("p1", "HP:1", 1, None)]
    preds = [pred("p1", "HP:1", 1, 0)]
    counts = compute_error_decomposition(records, preds)
    assert sum(counts.values()) == 0
