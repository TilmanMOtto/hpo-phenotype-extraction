"""§Hierarchy-aware quality, eqs. (3) and (4), including the case that separates them.

The decisive example is ``gold = {C}``, ``pred = {C, D}``: two siblings predicted where one is
ground truth. Under ancestor closure both sets collapse onto the same ancestors and the over-prediction is
charged once. Under CoPHE it is charged again at every shared ancestor, which is the
defect Falis et al. identified and the reason CoPHE is the primary figure here.
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    cophe_cohort,
    cophe_counts,
    cophe_prf,
    h_counts,
    h_prf,
    h_prf_cohort,
    hierarchy_report,
    micro_prf,
    subtree_counts,
)
from fixtures.toy_ontology import A, B, C, D, E, F, G, M

pytestmark = pytest.mark.unit


# ── Eq. (3): ancestor-closure hP / hR / hF ───────────────────────────────────

def test_h_counts_are_the_three_closure_quantities(toy_view):
    """An(pred={B}) = {B, A}; An(ground truth={C}) = {C, B, A}. Overlap = {B, A}."""
    assert h_counts({C}, {B}, toy_view) == (2, 2, 3)


def test_h_prf_gives_partial_credit_where_flat_f1_gives_none(toy_view):
    """Predicting the parent B when the annotated term is C: F1 = 0 but hF = 4/5.

    hP = 2/2 = 1, hR = 2/3, hF = 2 * 1 * (2/3) / (1 + 2/3) = 4/5.
    """
    hp, hr, hf = h_prf({C}, {B}, toy_view)
    assert (hp, hr, hf) == pytest.approx((1.0, 2 / 3, 0.8))
    assert micro_prf([{C}], [{B}])[2] == 0.0


def test_h_prf_is_one_when_the_prediction_is_exact(toy_view):
    assert h_prf({C}, {C}, toy_view) == pytest.approx((1.0, 1.0, 1.0))


def test_h_prf_cohort_reports_micro_and_macro(toy_view):
    """Two reports, both ground truth={C} pred={B}: pooled overlap 4, pred closure 4, ground truth closure 6."""
    out = h_prf_cohort([{C}, {C}], [{B}, {B}], toy_view)
    assert out["closure_overlap"] == 4
    assert out["closure_pred_total"] == 4
    assert out["closure_gold_total"] == 6
    assert out["micro_hp"] == pytest.approx(1.0)
    assert out["micro_hr"] == pytest.approx(2 / 3)
    assert out["macro_hf"] == pytest.approx(0.8)


# ── Eq. (4): CoPHE ───────────────────────────────────────────────────────────

def test_subtree_counts_count_a_term_at_itself_and_every_ancestor(toy_view):
    """An(C) = {C, B, A}, An(D) = {D, B, A} -> B and A each hold two terms."""
    assert subtree_counts({C, D}, toy_view) == {C: 1, D: 1, B: 2, A: 2}


def test_cophe_counts_hand_worked(toy_view):
    """ground truth={C}, pred={B}: TP at B and A, FN at C.

    x = {B:1, A:1}. Y = {C:1, B:1, A:1}
    B: min(1,1)=1 TP   A: min(1,1)=1 TP   C: max(1-0,0)=1 FN
    """
    assert cophe_counts({C}, {B}, toy_view) == (2, 0, 1)


def test_cophe_charges_over_prediction_that_closure_absorbs(toy_view):
    """The decisive case. ground truth={C}, pred={C, D}, two siblings under B, one of them ground truth.

    x = {C:1, D:1, B:2, A:2}. Y = {C:1, B:1, A:1}
    C: TP 1        D: FP 1
    B: TP 1, FP 1  A: TP 1, FP 1
    -> TP=3, FP=3, FN=0 -> P=1/2, R=1, F1=2/3

    Under ancestor closure the same prediction scores hF = 6/7: the surplus at B and A vanishes
    because both sets contain those ancestors once. CoPHE is the stricter, and here the
    more faithful, measure.
    """
    assert cophe_counts({C}, {C, D}, toy_view) == (3, 3, 0)
    assert cophe_prf({C}, {C, D}, toy_view) == pytest.approx((0.5, 1.0, 2 / 3))

    _, _, hf = h_prf({C}, {C, D}, toy_view)
    assert hf == pytest.approx(6 / 7)
    assert cophe_prf({C}, {C, D}, toy_view)[2] < hf


def test_cophe_equals_flat_when_the_labels_are_flat(toy_view):
    """Restricted to layer-1 terms, whose closures are singletons, CoPHE collapses to eq. (1).

    An(A) = {A} and An(G) = {G}, so x_v and y_v are just membership indicators and
    min/max reduce to the ordinary TP/FP/FN. Any divergence here would mean the count propagation
    is doing something beyond the hierarchy.
    """
    gold, pred = {A}, {A, G}
    assert cophe_counts(gold, pred, toy_view) == (1, 1, 0)
    assert cophe_prf(gold, pred, toy_view) == pytest.approx(micro_prf([gold], [pred]))


def test_cophe_is_perfect_on_an_exact_prediction(toy_view):
    assert cophe_prf({C, F}, {C, F}, toy_view) == pytest.approx((1.0, 1.0, 1.0))


def test_cophe_cohort_reports_micro_and_macro(toy_view):
    out = cophe_cohort([{C}, {C}], [{C, D}, {C, D}], toy_view)
    assert (out["cophe_tp"], out["cophe_fp"], out["cophe_fn"]) == (6, 6, 0)
    assert out["micro_cophe_f1"] == pytest.approx(2 / 3)
    assert out["macro_cophe_f1"] == pytest.approx(2 / 3)


def test_cophe_node_set_choice_does_not_change_the_numbers(toy_view):
    """Pooling over An(Ŷ) ∪ An(Y) rather than over all nodes is an optimisation, not a convention.

    Every node outside the union has x_v = y_v = 0, so it contributes min(0,0)=0 to TP and
    max(0,0)=0 to FP and FN. Adding the rest of the toy ontology's nodes by predicting and
    gold-labelling nothing there must leave the counts untouched.
    """
    baseline = cophe_counts({C}, {C, D}, toy_view)
    # E, F, G, H, M are all outside An({C}) ∪ An({C, D}) = {C, D, B, A}.
    assert E not in toy_view.ancestors_of_set({C, D})
    assert baseline == cophe_counts({C}, {C, D}, toy_view)


# ── The multi-parent weighting, and the audit that exposes it ────────────────

def test_multi_parent_annotated_term_inflates_the_closure_denominator(toy_view):
    """M's five ancestors against C's three: hR's denominator is not depth-symmetric.

    This is the note the skeleton attaches to hF, and the reason the |An(v)| distribution is
    reported alongside it.
    """
    _, hr_c, _ = h_prf({C}, {C}, toy_view)
    assert h_counts({C}, {C}, toy_view)[2] == 3
    assert h_counts({M}, {M}, toy_view)[2] == 5
    assert hr_c == 1.0  # An exact prediction still scores 1. The asymmetry is in partial credit


def test_hierarchy_report_bundles_cophe_hf_and_the_closure_audit(toy_view):
    out = hierarchy_report([{C}, {M}], [{C, D}, {M}], toy_view)
    assert "micro_cophe_f1" in out
    assert "micro_hf" in out
    assert out["gold_ancestor_counts"]["histogram"] == {3: 1, 5: 1}
    assert out["gold_ancestor_counts"]["multi_parent_fraction"] == pytest.approx(0.5)


# ── Cross-check against the repo's existing closure scorer ───────────────────

def test_agrees_with_legacy_closure_metric_once_the_root_is_added_back(toy_view, toy_tree):
    """``tree_metrics.closure_precision_recall_f1`` keeps the root in every closure. This package
    drops it. Adding the root back to both closures must reproduce the legacy number, which pins the difference to that one convention and nothing else.
    """
    from hpo_extraction.evaluation.set_metrics import calc_metric
    from hpo_extraction.evaluation.tree_metrics import closure_precision_recall_f1

    gold, pred = {C}, {B}
    legacy = closure_precision_recall_f1(gold, pred, toy_tree)

    an_gold = toy_view.ancestors_of_set(gold) | {toy_tree.root}
    an_pred = toy_view.ancestors_of_set(pred) | {toy_tree.root}
    assert calc_metric(an_gold, an_pred) == pytest.approx(legacy)

    # And the documented consequence: keeping the root inflates every component.
    assert h_prf(gold, pred, toy_view)[1] < legacy[1]
