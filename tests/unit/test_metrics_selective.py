"""§Risk--coverage analysis, eq. (8), with the unit and loss this thesis fixed.

The hand example, four evaluated (report, term) cells::

    conf   = [0.9, 0.8, 0.3, 0.1]
    labels = [ 1,   0,   1,   0 ]
    loss on a covered cell = 1[label == 0]  ->  [0, 1, 0, 1]

    tau = 0.8   covered = 2   phi = 1/2   R = 1/2   generalized R = 2/4 = 1/2 * 1/2
    AURC  = mean over k of (cumsum/k)  = mean(0, 1/2, 1/3, 1/2)   = 1/3
    AUGRC = mean over k of (cumsum/n)  = mean(0, 1/4, 1/4, 1/2)   = 1/4
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpo_extraction.evaluation.metrics import selective
from hpo_extraction.evaluation.metrics import (
    augrc,
    aurc,
    coverage,
    generalized_risk,
    risk_coverage_curve,
    selective_report,
    selective_risk,
)

pytestmark = pytest.mark.unit

CONF = [0.9, 0.8, 0.3, 0.1]
LABELS = [1, 0, 1, 0]


# ── The two quantities of eq. (8) ────────────────────────────────────────────

def test_coverage_is_the_covered_fraction():
    assert coverage(CONF, 0.8) == pytest.approx(0.5)
    assert coverage(CONF, 0.0) == pytest.approx(1.0)
    assert coverage(CONF, 1.0) == pytest.approx(0.0)


def test_selective_risk_hand_worked():
    """At tau = 0.8 the two covered cells are one true and one false positive."""
    assert selective_risk(CONF, LABELS, 0.8) == pytest.approx(0.5)


def test_selective_risk_is_nan_when_nothing_is_covered():
    """Substituting zero would let a method look perfect by abstaining everywhere."""
    assert math.isnan(selective_risk(CONF, LABELS, 1.01))


def test_generalized_risk_is_not_renormalised_by_coverage():
    """One failure among four cells at tau = 0.8, so the generalised risk is 1/4."""
    assert generalized_risk(CONF, LABELS, 0.8) == pytest.approx(0.25)


def test_generalized_risk_goes_to_zero_as_coverage_does():
    """Unlike R(tau) it stays finite and does not reward abstention."""
    assert generalized_risk(CONF, LABELS, 1.01) == pytest.approx(0.0)


# ── The identity this unit/loss choice implies ───────────────────────────────

def test_selective_risk_equals_one_minus_precision_at_every_threshold():
    """The property the module docstring states, fixed so it cannot silently break.

    A covered cell is by design reported, so the loss reduces to 1[label == 0] and
    R(tau) is the false-positive rate among the reported cells, i.e. 1 - micro-precision(tau).
    """
    conf = np.array([0.95, 0.9, 0.7, 0.7, 0.4, 0.2, 0.05])
    labels = np.array([1, 0, 1, 1, 0, 1, 0])
    for tau in np.unique(conf):
        covered = conf >= tau
        precision = labels[covered].sum() / covered.sum()
        assert selective_risk(conf, labels, float(tau)) == pytest.approx(1.0 - precision)


# ── The curve ────────────────────────────────────────────────────────────────

def test_risk_coverage_curve_has_one_row_per_distinct_threshold_plus_the_empty_point():
    rows = risk_coverage_curve(CONF, LABELS)
    assert len(rows) == 5  # The zero-coverage point plus four distinct confidences
    assert rows[0]["coverage"] == pytest.approx(0.0)
    assert rows[-1]["coverage"] == pytest.approx(1.0)


def test_curve_runs_from_high_tau_to_low_so_coverage_increases_down_the_rows():
    """The order a risk--coverage plot reads in: tau non-increasing, coverage non-decreasing."""
    rows = risk_coverage_curve(CONF, LABELS)
    taus = [r["tau"] for r in rows]
    coverages = [r["coverage"] for r in rows]
    assert taus == sorted(taus, reverse=True)
    assert coverages == sorted(coverages)


def test_curve_carries_precision_so_the_identity_is_visible_in_the_table():
    rows = risk_coverage_curve(CONF, LABELS)
    for row in rows[1:]:  # The leading zero-coverage row has no defined risk
        assert row["precision"] == pytest.approx(1.0 - row["selective_risk"])


def test_curve_of_an_empty_sample_is_empty():
    assert risk_coverage_curve([], []) == []


# ── Areas ────────────────────────────────────────────────────────────────────

def test_aurc_hand_worked():
    """r(k) over k = 1..4 is (0, 1/2, 1/3, 1/2). The mean is 1/3."""
    assert aurc(CONF, LABELS) == pytest.approx(1 / 3)


def test_augrc_hand_worked():
    """The same sweep without the division by coverage: mean of (0, 1/4, 1/4, 1/2) = 1/4."""
    assert augrc(CONF, LABELS) == pytest.approx(0.25)


def test_augrc_never_exceeds_aurc():
    """AUGRC weights each point by its coverage k/n <= 1, so it is the smaller of the two."""
    rng = np.random.default_rng(5)
    for _ in range(5):
        conf = rng.uniform(0.0, 1.0, 200)
        labels = (rng.uniform(size=conf.size) < conf).astype(int)
        assert augrc(conf, labels) <= aurc(conf, labels) + 1e-12


def test_a_perfect_ranker_scores_zero_on_both_areas():
    """Every positive above every negative, and no negatives at all: no failure is ever covered."""
    assert aurc([0.9, 0.8, 0.7], [1, 1, 1]) == pytest.approx(0.0)
    assert augrc([0.9, 0.8, 0.7], [1, 1, 1]) == pytest.approx(0.0)


def test_a_ranker_that_orders_failures_first_scores_worse_than_one_that_does_not():
    """The discrimination AURC is meant to reward: same labels, opposite confidence ordering.

    good: positives carry the high confidences -> losses (0,0,1,1) -> AURC = 5/24
    bad:  positives carry the low  confidences -> losses (1,1,0,0) -> AURC = 19/24
    """
    labels = [1, 1, 0, 0]
    good = aurc([0.9, 0.8, 0.2, 0.1], labels)
    bad = aurc([0.1, 0.2, 0.8, 0.9], labels)
    assert good == pytest.approx(5 / 24)
    assert bad == pytest.approx(19 / 24)
    assert good < bad


def test_areas_of_an_empty_sample_are_nan():
    assert math.isnan(aurc([], []))
    assert math.isnan(augrc([], []))


# ── Ties, which are the normal case for a score with S+1 atoms ───────────────

def test_expected_tie_handling_is_permutation_invariant():
    """With at most S+1 score atoms, tie groups are enormous. A top-k area must not depend on the
    caller's row order. The default replaces each cell's loss by its tie-group mean, which is the
    exact expectation over orderings within the group."""
    conf = [0.5, 0.5, 0.5, 0.5]
    assert aurc(conf, [1, 0, 1, 0]) == pytest.approx(aurc(conf, [0, 1, 0, 1]))
    assert augrc(conf, [1, 0, 1, 0]) == pytest.approx(augrc(conf, [0, 1, 0, 1]))


def test_ordered_tie_handling_does_depend_on_order_which_is_why_it_is_not_the_default():
    conf = [0.5, 0.5]
    assert aurc(conf, [1, 0], "ordered") != pytest.approx(aurc(conf, [0, 1], "ordered"))


def test_expected_tie_handling_hand_worked():
    """Two tied cells, one positive: each carries the group-mean loss 1/2.

    cumsum/k = (1/2, 1/2), so AURC = 1/2 regardless of which cell is listed first.
    """
    assert aurc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_expected_and_ordered_agree_when_there_are_no_ties():
    conf = [0.9, 0.6, 0.3]
    labels = [1, 0, 1]
    assert aurc(conf, labels, "expected") == pytest.approx(aurc(conf, labels, "ordered"))


def test_unknown_tie_handling_is_rejected():
    with pytest.raises(ValueError, match="tie_handling"):
        aurc(CONF, LABELS, "whatever")


# ── The report bundle ────────────────────────────────────────────────────────

def test_selective_report_fixes_the_areas_at_full_coverage():
    """``risk_at_full_coverage`` is the aggregate error rate, the evidence location that makes the
    discrimination/calibration confound in AURC readable rather than hidden."""
    out = selective_report(CONF, LABELS)
    assert out["n"] == 4
    assert out["prevalence"] == pytest.approx(0.5)
    assert out["aurc"] == pytest.approx(1 / 3)
    assert out["augrc"] == pytest.approx(0.25)
    assert out["risk_at_full_coverage"] == pytest.approx(0.5)
    assert out["coverage_at_lowest_tau"] == pytest.approx(1.0)


def test_selective_report_of_an_empty_sample():
    out = selective_report([], [])
    assert out["n"] == 0
    assert math.isnan(out["aurc"])


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError, match="align"):
        selective_risk([0.5, 0.5], [1], 0.5)


def test_non_binary_labels_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        aurc([0.5, 0.5], [0, 2])


def test_coverage_and_generalized_risk_are_empty_safe():
    assert coverage([], 0.5) == 0.0
    assert math.isnan(generalized_risk([], [], 0.5))


class TestTieGroupMeansAreUnchangedButFaster:
    """The tie-group mean was rewritten from O(U·n) to O(n log n). It must be the same number.

    On a continuous score U ≈ n, so the original loop over ``np.unique`` was quadratic: a
    calibration sample of 540k tree nodes made one AURC take hours, which is why the earlier deep-dive UI could not open its Calibration tab. Nothing about the definition changed, and
    these tests exist to keep it that way.
    """

    @staticmethod
    def _reference(conf, loss):
        """The original implementation, kept here as the thing the fast path must equal."""
        conf = np.asarray(conf, dtype=float)
        loss = np.asarray(loss, dtype=float).copy()
        for value in np.unique(conf):
            group = conf == value
            loss = np.where(group, loss[group].mean(), loss)
        return loss

    @pytest.mark.parametrize("n,decimals", [(1, 6), (2, 0), (50, 0), (200, 1), (500, 6)])
    def test_matches_the_reference_on_random_input(self, n, decimals):
        rng = np.random.default_rng(n * 7 + decimals)
        conf = np.round(rng.random(n), decimals)          # low decimals ⇒ heavy ties
        loss = (rng.random(n) < 0.3).astype(float)
        assert selective._tie_group_means(conf, loss) == pytest.approx(
            self._reference(conf, loss), abs=1e-12)

    def test_all_tied_gives_every_cell_the_group_mean(self):
        conf = np.full(4, 0.5)
        loss = np.array([1.0, 0.0, 0.0, 0.0])
        assert selective._tie_group_means(conf, loss) == pytest.approx([0.25] * 4)

    def test_no_ties_leaves_every_loss_alone(self):
        conf = np.array([0.9, 0.1, 0.5])
        loss = np.array([1.0, 0.0, 1.0])
        assert selective._tie_group_means(conf, loss) == pytest.approx(loss)

    def test_the_result_does_not_depend_on_input_order(self):
        """The whole point of the tie handling, and the place a sort-based rewrite could slip."""
        rng = np.random.default_rng(11)
        conf = np.round(rng.random(120), 1)
        loss = (rng.random(120) < 0.4).astype(float)
        shuffle = rng.permutation(conf.size)
        assert selective.aurc(conf, loss.astype(int)) == pytest.approx(
            selective.aurc(conf[shuffle], loss[shuffle].astype(int)))

    def test_an_empty_sample_is_returned_unchanged(self):
        empty = np.array([], dtype=float)
        assert selective._tie_group_means(empty, empty).size == 0
