"""Conformal risk control for the pruning threshold.

The toy cohort throughout, three reports across three thresholds::

              tau=0.1 (permissive)   tau=0.5        tau=0.9 (strict)
    r1 (4 ground truth)   loses 0/4 = 0.00    1/4 = 0.25     3/4 = 0.75
    r2 (2 ground truth)   loses 0/2 = 0.00    1/2 = 0.50     2/2 = 1.00
    r3 (0 ground truth)              0.00           0.00           0.00

    empirical risk        0.0000         0.2500         0.5833
    bound, n = 3          0.2500         0.4375         0.6875     ( 3/4 * R + 1/4 )

The finite-sample term ``1/(n+1)`` is 0.25 here, which is why nothing below alpha = 0.25 is ever
reachable on three reports. At the ~94 calibration reports of a real outer fold it is about 0.011, the same slack the thesis quotes.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpo_extraction.evaluation.stats import crc_lambda, held_out_risk, loss_curve, miss_rate


@pytest.fixture
def gold():
    return {"r1": {"a", "b", "c", "d"}, "r2": {"x", "y"}, "r3": set()}


@pytest.fixture
def scored():
    return {
        0.1: {"r1": {"a", "b", "c", "d"}, "r2": {"x", "y"}, "r3": set()},
        0.5: {"r1": {"a", "b", "c"}, "r2": {"x"}, "r3": set()},
        0.9: {"r1": {"a"}, "r2": set(), "r3": set()},
    }


class TestMissRate:
    def test_nothing_lost(self):
        assert miss_rate({"a", "b"}, {"a", "b", "c"}) == 0.0

    def test_everything_lost(self):
        assert miss_rate({"a", "b"}, set()) == 1.0

    def test_partial(self):
        assert miss_rate({"a", "b", "c", "d"}, {"a", "b", "c"}) == pytest.approx(0.25)

    def test_empty_gold_contributes_zero(self):
        """It cannot lose what it does not have, and dropping it would change n."""
        assert miss_rate(set(), set()) == 0.0
        assert miss_rate(set(), {"a"}) == 0.0


class TestLossCurve:
    def test_matches_the_hand_computed_table(self, gold, scored):
        taus, losses = loss_curve(gold, scored, report_ids=["r1", "r2", "r3"])
        assert taus == [0.1, 0.5, 0.9]
        assert losses == pytest.approx(np.array([
            [0.00, 0.25, 0.75],
            [0.00, 0.50, 1.00],
            [0.00, 0.00, 0.00],
        ]))

    def test_thresholds_come_back_ascending(self, gold, scored):
        taus, _ = loss_curve(gold, scored)
        assert taus == sorted(taus)

    def test_loss_is_non_decreasing_in_tau(self, gold, scored):
        """The monotonicity the bound is inverted against. A node budget would break it."""
        _, losses = loss_curve(gold, scored)
        assert (np.diff(losses, axis=1) >= -1e-12).all()


class TestCrcLambda:
    def test_picks_the_cheapest_feasible_threshold(self, gold, scored):
        """alpha = 0.5 admits tau 0.1 and 0.5. The stricter one is cheaper, so it wins."""
        taus, losses = loss_curve(gold, scored)
        got = crc_lambda(taus, losses, alpha=0.5)
        assert got["feasible"] is True
        assert got["tau"] == 0.5
        assert got["bound"] == pytest.approx(0.4375)
        assert got["empirical_risk"] == pytest.approx(0.25)

    def test_a_tighter_alpha_forces_a_more_permissive_threshold(self, gold, scored):
        taus, losses = loss_curve(gold, scored)
        got = crc_lambda(taus, losses, alpha=0.4)
        assert got["tau"] == 0.1
        assert got["bound"] == pytest.approx(0.25)

    def test_unreachable_alpha_reports_the_ceiling_rather_than_a_threshold(self, gold, scored):
        """The case the thesis has to print: the bound is set by retrieval, not by pruning."""
        taus, losses = loss_curve(gold, scored)
        got = crc_lambda(taus, losses, alpha=0.05)
        assert got["feasible"] is False
        assert got["tau"] is None
        assert got["attained_ceiling"] == pytest.approx(0.25)
        assert got["attained_ceiling_tau"] == 0.1

    def test_the_ceiling_is_reported_even_when_feasible(self, gold, scored):
        taus, losses = loss_curve(gold, scored)
        got = crc_lambda(taus, losses, alpha=0.5)
        assert got["attained_ceiling"] == pytest.approx(0.25)

    def test_finite_sample_term_is_one_over_n_plus_one(self):
        """With zero empirical risk the bound is 1/(n+1), the slack, alone."""
        for n in (3, 94, 999):
            losses = np.zeros((n, 1))
            got = crc_lambda([0.1], losses, alpha=0.9)
            assert got["bound"] == pytest.approx(1 / (n + 1))

    def test_ninety_four_reports_give_the_slack_the_thesis_quotes(self):
        got = crc_lambda([0.1], np.zeros((94, 1)), alpha=0.9)
        assert got["bound"] == pytest.approx(0.0105, abs=5e-4)

    def test_chosen_threshold_always_satisfies_its_own_bound(self, gold, scored):
        taus, losses = loss_curve(gold, scored)
        for alpha in (0.3, 0.45, 0.5, 0.7, 0.95):
            got = crc_lambda(taus, losses, alpha=alpha)
            if got["feasible"]:
                assert got["bound"] <= alpha + 1e-12

    def test_rejects_a_degenerate_alpha(self, gold, scored):
        taus, losses = loss_curve(gold, scored)
        for bad in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="alpha"):
                crc_lambda(taus, losses, alpha=bad)

    def test_rejects_an_empty_calibration_set(self):
        with pytest.raises(ValueError, match="no calibration reports"):
            crc_lambda([0.1], np.zeros((0, 1)), alpha=0.5)


class TestHeldOutRisk:
    def test_measures_what_actually_happened(self, gold, scored):
        """The number that makes the guarantee checkable rather than merely asserted."""
        got = held_out_risk(gold, scored[0.5], ["r1", "r2", "r3"])
        assert got["mean_loss"] == pytest.approx((0.25 + 0.50 + 0.0) / 3)
        assert got["n_reports"] == 3
        assert got["n_reports_with_gold"] == 2

    def test_empty_report_list(self, gold, scored):
        got = held_out_risk(gold, scored[0.5], [])
        assert got["n_reports"] == 0

    def test_a_report_absent_from_the_scored_map_loses_everything(self, gold, scored):
        got = held_out_risk(gold, {}, ["r1"])
        assert got["mean_loss"] == pytest.approx(1.0)
