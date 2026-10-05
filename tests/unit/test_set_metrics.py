"""Unit tests for hpo_extraction.evaluation.set_metrics, calc_metric, evaluate_micro_macro, evaluate_UTI.

All expected values are hand-computed. See inline comments for derivations.
"""

import pytest

from hpo_extraction.evaluation.set_metrics import calc_metric, evaluate_micro_macro, evaluate_UTI


# ---------------------------------------------------------------------------
# calc_metric, happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_calc_metric_both_empty_returns_perfect_score():
    # Documented: both empty → (1, 1, 1)
    assert calc_metric(set(), set()) == (1.0, 1.0, 1.0)


@pytest.mark.unit
def test_calc_metric_empty_ground_truth_returns_zero():
    assert calc_metric(set(), {"HP:1"}) == (0.0, 0.0, 0.0)


@pytest.mark.unit
def test_calc_metric_empty_prediction_returns_zero():
    assert calc_metric({"HP:1"}, set()) == (0.0, 0.0, 0.0)


@pytest.mark.unit
def test_calc_metric_perfect_match_single_element():
    pr, re, f1 = calc_metric({"A"}, {"A"})
    assert pr == pytest.approx(1.0)
    assert re == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


@pytest.mark.unit
def test_calc_metric_perfect_match_multiple_elements():
    pr, re, f1 = calc_metric({"A", "B", "C"}, {"A", "B", "C"})
    assert pr == pytest.approx(1.0)
    assert re == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


@pytest.mark.unit
def test_calc_metric_partial_overlap_two_of_three():
    # true={"A","B","C"}, pred={"A","B","D"}
    # TP = |{"A","B","C"} & {"A","B","D"}| = 2
    # pr  = TP / |pred| = 2/3
    # re  = TP / |true| = 2/3
    # f1  = 2*(2/3)*(2/3) / ((2/3)+(2/3)) = (8/9)/(4/3) = 2/3
    pr, re, f1 = calc_metric({"A", "B", "C"}, {"A", "B", "D"})
    assert pr == pytest.approx(2 / 3, abs=1e-6)
    assert re == pytest.approx(2 / 3, abs=1e-6)
    assert f1 == pytest.approx(2 / 3, abs=1e-6)


@pytest.mark.unit
def test_calc_metric_no_overlap_returns_zero_f1():
    pr, re, f1 = calc_metric({"A"}, {"B"})
    assert pr == pytest.approx(0.0)
    assert re == pytest.approx(0.0)
    assert f1 == pytest.approx(0.0)


@pytest.mark.unit
def test_calc_metric_high_precision_low_recall():
    # true={"A","B","C"}, pred={"A"}
    # TP=1, pr=1/1=1.0, re=1/3, f1=2*1.0*(1/3)/(1.0+1/3)=0.5
    pr, re, f1 = calc_metric({"A", "B", "C"}, {"A"})
    assert pr == pytest.approx(1.0)
    assert re == pytest.approx(1 / 3, abs=1e-6)
    assert f1 == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_calc_metric_returns_three_floats():
    result = calc_metric({"A"}, {"A"})
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# calc_metric, parametrize across documented boundary values
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("true_hpos,pred_hpos,expected", [
    (set(), set(), (1.0, 1.0, 1.0)),
    (set(), {"A"}, (0.0, 0.0, 0.0)),
    ({"A"}, set(), (0.0, 0.0, 0.0)),
    ({"A"}, {"A"}, (1.0, 1.0, 1.0)),
    ({"A"}, {"B"}, (0.0, 0.0, 0.0)),
])
def test_calc_metric_parametrized(true_hpos, pred_hpos, expected):
    assert calc_metric(true_hpos, pred_hpos) == expected


# ---------------------------------------------------------------------------
# evaluate_micro_macro, happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_evaluate_micro_macro_two_patients_mixed():
    # Patient 1: gt={"A","B"}, pred={"A","B"} → pr=1.0, re=1.0, TP=2, |pred|=2, |gt|=2
    # Patient 2: gt={"A"},     pred={}         → pr=0.0, re=0.0, TP=0, |pred|=0, |gt|=1
    # tp_count=2, pred_count=2, actual_count=3
    # micro_precision = 2/2 = 1.0
    # micro_recall    = 2/3
    # micro_f1        = 2*1.0*(2/3) / (1.0 + 2/3) = (4/3)/(5/3) = 4/5 = 0.8
    # macro_precision = (1.0 + 0.0)/2 = 0.5
    # macro_recall    = (1.0 + 0.0)/2 = 0.5
    # macro_f1        = 2*0.5*0.5 / (0.5+0.5) = 0.5
    gt = [{"A", "B"}, {"A"}]
    pred = [{"A", "B"}, set()]
    metrics, indiv = evaluate_micro_macro(gt, pred)

    assert metrics["micro_precision"] == pytest.approx(1.0)
    assert metrics["micro_recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["micro_f1"] == pytest.approx(0.8, abs=1e-6)
    assert metrics["macro_precision"] == pytest.approx(0.5)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)


@pytest.mark.unit
def test_evaluate_micro_macro_perfect_predictions():
    # Single patient, perfect prediction
    # TP=2, |pred|=2, |gt|=2 → all metrics = 1.0
    gt = [{"A", "B"}]
    pred = [{"A", "B"}]
    metrics, _ = evaluate_micro_macro(gt, pred)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


@pytest.mark.unit
def test_evaluate_micro_macro_individual_results_structure():
    gt = [{"A"}]
    pred = [{"A"}]
    _, indiv = evaluate_micro_macro(gt, pred)
    assert set(indiv.keys()) == {"precision", "recall", "F1"}
    assert len(indiv["precision"]) == 1
    assert len(indiv["recall"]) == 1
    assert len(indiv["F1"]) == 1


@pytest.mark.unit
def test_evaluate_micro_macro_three_samples_count():
    gt = [{"A"}, {"B"}, {"C"}]
    pred = [{"A"}, {"B"}, {"C"}]
    _, indiv = evaluate_micro_macro(gt, pred)
    assert len(indiv["F1"]) == 3


# ---------------------------------------------------------------------------
# evaluate_micro_macro, known bugs (ZeroDivisionError)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_evaluate_micro_macro_raises_when_all_predictions_empty():
    """Bug: pred_count=0 causes ZeroDivisionError in micro_precision = tp/pred_count.
    This test documents current behaviour. Fix should change this to pytest.approx(0.0)."""
    with pytest.raises(ZeroDivisionError):
        evaluate_micro_macro([{"A"}], [set()])


# ---------------------------------------------------------------------------
# evaluate_UTI, happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_evaluate_uti_returns_correct_precision_recall_f1():
    # 3 patients: p1=UTI, p2=CAUTI, p3=AB
    # pred_pos={"p1"} → TP=1 (p1), FN=1 (p2 missed), TN=1 (p3), FP=0
    # precision = TP/|pred_pos| = 1/1 = 1.0
    # recall    = TP/|true_pos| = 1/2 = 0.5
    # f1        = 2*1.0*0.5 / (1.0+0.5) = 1.0/1.5 = 2/3
    # accuracy  = (TP+TN)/total = (1+1)/3 = 2/3
    patient_data = [("p1", "UTI"), ("p2", "CAUTI"), ("p3", "AB")]
    wrong, results = evaluate_UTI(["p1"], patient_data)

    assert results["Precision"] == pytest.approx(1.0)
    assert results["Recall"] == pytest.approx(0.5)
    assert results["F1"] == pytest.approx(2 / 3, abs=1e-6)
    assert results["Accuracy"] == pytest.approx(2 / 3, abs=1e-6)


@pytest.mark.unit
def test_evaluate_uti_false_negative_in_wrong_dict():
    patient_data = [("p1", "UTI"), ("p2", "CAUTI"), ("p3", "AB")]
    wrong, _ = evaluate_UTI(["p1"], patient_data)
    assert "p2" in wrong["false neg"]


@pytest.mark.unit
def test_evaluate_uti_false_positive_in_wrong_dict():
    # 3 patients: p1=UTI, p2=UTI, p3=AB
    # Predict p1 (correct) and p3 (wrong → false positive)
    # precision=1/2=0.5, recall=1/2=0.5 → no zero division
    # true_neg={"p3"}, pred_neg={"p2"} → false_pos=["p3"]
    patient_data = [("p1", "UTI"), ("p2", "UTI"), ("p3", "AB")]
    wrong, _ = evaluate_UTI(["p1", "p3"], patient_data)
    assert "p3" in wrong["false pos"]


@pytest.mark.unit
def test_evaluate_uti_perfect_prediction():
    patient_data = [("p1", "UTI"), ("p2", "AB")]
    _, results = evaluate_UTI(["p1"], patient_data)
    assert results["Precision"] == pytest.approx(1.0)
    assert results["Recall"] == pytest.approx(1.0)
    assert results["F1"] == pytest.approx(1.0)
    assert results["Accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_UTI, known bugs (ZeroDivisionError)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_evaluate_uti_raises_when_no_positive_predictions():
    """Bug: len(pred_pos)=0 causes ZeroDivisionError in precision = TP/|pred_pos|."""
    patient_data = [("p1", "UTI"), ("p2", "AB")]
    with pytest.raises(ZeroDivisionError):
        evaluate_UTI([], patient_data)
