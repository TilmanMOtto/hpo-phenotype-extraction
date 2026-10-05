"""§Calibration, eq. (7) and the four estimators that answer the skeleton's objections to it.

The recurring hand example::

    conf   = [0.1, 0.1, 0.9, 0.9]
    labels = [ 0,   0,   1,   1 ]

    Two occupied equal-width bins. Bin at 0.1: conf=0.1, acc=0.0, |gap|=0.1, weight=1/2.
    Bin at 0.9: conf=0.9, acc=1.0, |gap|=0.1, weight=1/2.  ECE = 0.5*0.1 + 0.5*0.1 = 0.1.

The score is *under*-confident in both bins in opposite directions, which is the
sign-cancellation eq. (7) hides and :func:`signed_gap_bins` exposes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpo_extraction.evaluation.metrics import (
    bin_summary,
    brier,
    calibration_report,
    cohort_transfer,
    compare_raw_vs_calibrated,
    conditional_calibration,
    depth_groups,
    discrete_score_table,
    ece_equal_mass,
    ece_equal_width,
    fit_thresholds,
    frequency_buckets,
    kfold_report_split,
    layer1_groups,
    roelofs_monotonic_sweep,
    signed_gap_bins,
    tail_calibration_error,
    transfer_gap,
    wilson_interval,
)
from fixtures.toy_ontology import A, B, C, D, F, G, M

pytestmark = pytest.mark.unit

CONF = [0.1, 0.1, 0.9, 0.9]
LABELS = [0, 0, 1, 1]


# ── Eq. (7) ──────────────────────────────────────────────────────────────────

def test_ece_equal_width_hand_worked():
    assert ece_equal_width(CONF, LABELS, n_bins=10) == pytest.approx(0.1)


def test_ece_is_zero_for_a_perfectly_calibrated_score():
    """Half of the 0.5-confidence predictions are positive, so acc == conf and the gap vanishes."""
    assert ece_equal_width([0.5] * 4, [0, 1, 0, 1]) == pytest.approx(0.0)


def test_ece_is_one_for_a_maximally_wrong_confident_score():
    assert ece_equal_width([1.0, 1.0], [0, 0]) == pytest.approx(1.0)


def test_confidence_of_one_is_not_lost_by_the_final_bin():
    """Half-open bins would drop c = 1.0 entirely. The last bin is closed to prevent that."""
    bins = bin_summary([1.0, 1.0], [1, 1], n_bins=10, binning="equal_width")
    assert sum(b["count"] for b in bins) == 2


def test_ece_rejects_a_logit_margin_passed_as_a_confidence():
    """The single most likely misuse: passing ``margin`` where ``sigmoid(margin)`` was meant."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece_equal_width([-3.2, 4.1], [0, 1])


def test_ece_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        ece_equal_width([0.5, 0.5], [0, 2])


def test_ece_of_an_empty_sample_is_nan():
    assert math.isnan(ece_equal_width([], []))


def test_ece_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="align"):
        ece_equal_width([0.5, 0.5], [1])


def test_a_nonpositive_bin_count_is_rejected():
    with pytest.raises(ValueError, match="n_bins"):
        ece_equal_mass([0.5, 0.5], [0, 1], n_bins=0)


def test_empty_sample_estimators_are_all_nan_safe():
    """Every scalar in the calibration report must survive a run that produced no scored cells."""
    assert math.isnan(brier([], []))
    sweep = roelofs_monotonic_sweep([], [])
    assert sweep["n_bins"] == 0 and math.isnan(sweep["ece"])
    assert bin_summary([], []) == []
    assert discrete_score_table([], []) == []


def test_equal_mass_and_equal_width_disagree_on_a_skewed_score():
    """The bias the skeleton objects to: with the mass piled near zero, equal-width bins leave the
    upper range empty and unweighted while equal-mass bins resolve the crowded region."""
    rng = np.random.default_rng(0)
    conf = np.concatenate([rng.uniform(0.0, 0.05, 400), rng.uniform(0.9, 1.0, 20)])
    labels = (rng.uniform(size=conf.size) < conf).astype(int)
    assert ece_equal_width(conf, labels, 10) != pytest.approx(ece_equal_mass(conf, labels, 10))


# ── Bins, signed and otherwise ───────────────────────────────────────────────

def test_bin_summary_reports_signed_gaps_and_populations():
    bins = bin_summary(CONF, LABELS, n_bins=10, binning="equal_width")
    assert len(bins) == 2
    low, high = bins
    assert low["confidence"] == pytest.approx(0.1)
    assert low["accuracy"] == pytest.approx(0.0)
    assert low["gap"] == pytest.approx(-0.1)   # over-confident
    assert high["gap"] == pytest.approx(0.1)   # under-confident
    assert low["count"] == high["count"] == 2
    assert low["weight"] == pytest.approx(0.5)


def test_signed_gaps_cancel_where_eq_7_does_not():
    """The whole objection in one assertion: the signed gaps sum to zero while ECE reports 0.1."""
    bins = signed_gap_bins(CONF, LABELS, n_bins=10, binning="equal_width")
    assert sum(b["weight"] * b["gap"] for b in bins) == pytest.approx(0.0)
    assert ece_equal_width(CONF, LABELS) == pytest.approx(0.1)


def test_empty_bins_are_omitted_rather_than_reported_as_zero_gap():
    bins = bin_summary(CONF, LABELS, n_bins=10, binning="equal_width")
    assert all(b["count"] > 0 for b in bins)


# ── The Roelofs bin sweep ────────────────────────────────────────────────────

def test_roelofs_sweep_keeps_refining_while_accuracy_stays_monotone():
    """A monotone, well-behaved score: the sweep should not stop at the very first split."""
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.0, 1.0, 2000)
    labels = (rng.uniform(size=conf.size) < conf).astype(int)
    out = roelofs_monotonic_sweep(conf, labels)
    assert out["n_bins"] >= 2
    assert 0.0 <= out["ece"] < 0.1


def test_roelofs_sweep_stops_at_one_bin_when_accuracy_is_not_monotone():
    """An inverted score, high confidence, low presence, is non-monotone at any split."""
    out = roelofs_monotonic_sweep([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert out["n_bins"] == 1
    # With one bin the ECE degenerates to |mean(label) - mean(confidence)| = |0.5 - 0.5| = 0.
    assert out["ece"] == pytest.approx(0.0)


def test_roelofs_sweep_reports_the_score_support():
    """With at most S+1 atoms the support is the natural cap on how fine binning can be."""
    out = roelofs_monotonic_sweep([0.2, 0.2, 0.4, 0.4, 0.6, 0.6], [0, 0, 0, 1, 1, 1])
    assert out["n_distinct_scores"] == 3
    assert out["max_bins_tried"] <= 3


# ── The binning-free limit ───────────────────────────────────────────────────

def test_wilson_interval_matches_the_published_value():
    """5 successes in 10 trials: the 95 % Wilson interval is (0.2366, 0.7634)."""
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_wilson_interval_stays_inside_the_unit_interval_at_zero_successes():
    """Where the normal approximation would leave [0, 1] outright."""
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.2775, abs=1e-4)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_of_an_empty_sample_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_discrete_score_table_is_one_row_per_atom():
    """conf = [0.2, 0.2, 0.8] with labels [0, 1, 1]: the 0.2 atom is half positive."""
    rows = discrete_score_table([0.2, 0.2, 0.8], [0, 1, 1])
    assert [r["score"] for r in rows] == [0.2, 0.8]
    assert rows[0]["count"] == 2
    assert rows[0]["frequency"] == pytest.approx(0.5)
    assert rows[0]["gap"] == pytest.approx(0.3)
    assert rows[1]["frequency"] == pytest.approx(1.0)
    assert rows[0]["wilson_lo"] <= rows[0]["frequency"] <= rows[0]["wilson_hi"]


def test_discrete_score_table_rounds_away_floating_point_noise():
    """Two values that differ in the sixteenth decimal are one atom, not two."""
    rows = discrete_score_table([0.1, 0.1 + 1e-15], [0, 1])
    assert len(rows) == 1
    assert rows[0]["count"] == 2


# ── The low-confidence tail ──────────────────────────────────────────────────

def test_tail_calibration_error_looks_only_below_the_cutoff():
    """The two 0.1 predictions fall in the tail. The two 0.9 predictions do not."""
    out = tail_calibration_error(CONF, LABELS, upper=0.3)
    assert out["n"] == 2
    assert out["coverage"] == pytest.approx(0.5)
    assert out["prevalence"] == pytest.approx(0.0)
    assert out["ece"] == pytest.approx(0.1)


def test_tail_calibration_error_reports_coverage_so_the_aggregate_can_be_judged():
    """"ECE is dominated by the densely populated high-confidence region", coverage says by how
    much."""
    conf = [0.05] * 5 + [0.95] * 95
    labels = [0] * 5 + [1] * 95
    out = tail_calibration_error(conf, labels, upper=0.1)
    assert out["coverage"] == pytest.approx(0.05)


def test_tail_calibration_error_on_an_empty_tail():
    out = tail_calibration_error([0.9, 0.9], [1, 1], upper=0.1)
    assert out["n"] == 0
    assert math.isnan(out["ece"])


def test_tail_calibration_error_rejects_a_cutoff_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        tail_calibration_error(CONF, LABELS, upper=0.0)


# ── Group-conditional calibration ────────────────────────────────────────────

def test_conditional_calibration_finds_miscalibration_the_aggregate_hides():
    """Two groups miscalibrated in opposite directions cancel in aggregate but not per group.

    Group "a": confidence 0.9, never present. Group "b": confidence 0.1, always present. The pooled
    mean confidence and mean accuracy are both 0.5, so a single-bin aggregate sees nothing.
    """
    conf = [0.9] * 50 + [0.1] * 50
    labels = [0] * 50 + [1] * 50
    groups = ["a"] * 50 + ["b"] * 50
    out = conditional_calibration(conf, labels, groups, n_bins=1)

    assert out["aggregate_ece"] == pytest.approx(0.0)
    assert out["groups"]["a"]["ece"] == pytest.approx(0.9)
    assert out["groups"]["b"]["ece"] == pytest.approx(0.9)
    assert out["worst_ece"] == pytest.approx(0.9)
    assert out["max_minus_aggregate_ece"] == pytest.approx(0.9)


def test_small_groups_are_computed_but_flagged_insufficient():
    conf = [0.9] * 50 + [0.1] * 5
    labels = [1] * 50 + [0] * 5
    groups = ["big"] * 50 + ["tiny"] * 5
    out = conditional_calibration(conf, labels, groups, min_count=30)
    assert out["groups"]["big"]["sufficient"] is True
    assert out["groups"]["tiny"]["sufficient"] is False
    assert out["n_groups_sufficient"] == 1
    assert out["worst_group"] == "big"


def test_conditional_calibration_rejects_misaligned_groups():
    with pytest.raises(ValueError, match="align"):
        conditional_calibration([0.5, 0.5], [0, 1], ["a"])


def test_depth_groups_key_by_ontology_depth(toy_view):
    assert depth_groups([A, B, C], toy_view) == [1, 2, 3]


def test_layer1_groups_key_by_organ_system(toy_view):
    assert layer1_groups([C, F], toy_view) == [A, A]
    assert layer1_groups([M], toy_view) == [min(A, G)]


def test_frequency_buckets_rank_terms_by_how_often_they_are_gold():
    """C appears in three ground-truth sets, D in one, F in none."""
    gold_sets = [{C, D}, {C}, {C}]
    buckets = frequency_buckets([C, D, F], gold_sets, n_buckets=2)
    assert buckets[2] == "unseen"          # F is never ground truth, its own category
    assert buckets[0] != buckets[1]        # C is the frequent one, D the rare one
    assert buckets[1] == "q1"              # rarest bucket first


def test_frequency_buckets_with_no_gold_at_all():
    assert frequency_buckets([C, D], [], n_buckets=2) == ["unseen", "unseen"]


# ── The report bundle ────────────────────────────────────────────────────────

def test_calibration_report_collects_every_estimator():
    out = calibration_report(CONF, LABELS, tail_upper=0.3)
    assert out["n"] == 4
    assert out["prevalence"] == pytest.approx(0.5)
    assert out["ece_equal_width"] == pytest.approx(0.1)
    assert out["brier"] == pytest.approx(0.01)
    assert out["tail"]["n"] == 2
    assert "smooth_ece" in out          # None when relplot is not installed


def test_brier_is_the_mean_squared_error():
    """Each of the four predictions is 0.1 away from its label: 0.1^2 = 0.01."""
    assert brier(CONF, LABELS) == pytest.approx(0.01)


@pytest.mark.integration
def test_smooth_ece_agrees_with_binned_ece_on_a_calibrated_sample():
    """Skipped unless ``relplot`` is installed. A well-calibrated score has near-zero smoothECE."""
    relplot = pytest.importorskip("relplot")
    assert relplot is not None
    from hpo_extraction.evaluation.metrics import smooth_ece

    rng = np.random.default_rng(7)
    conf = rng.uniform(0.0, 1.0, 5000)
    labels = (rng.uniform(size=conf.size) < conf).astype(int)
    value = smooth_ece(conf, labels)
    assert 0.0 <= value < 0.1
    assert value == pytest.approx(ece_equal_mass(conf, labels), abs=0.05)


# ── §Threshold transferability ───────────────────────────────────────────────

def _synthetic_evaluate(peak_prune: float, peak_accept: float):
    """An ``evaluate_fn`` whose objective peaks at a known threshold pair.

    Stands in for the per-run re-run the experiments supply, so the *protocol* can be tested without
    a GPU or a run artifact.
    """
    def evaluate(tau_prune: float, tau_accept: float) -> dict:
        loss = abs(tau_prune - peak_prune) + abs(tau_accept - peak_accept)
        return {"micro_f1": 1.0 - loss, "micro_precision": 0.9}
    return evaluate


GRID = [0.1, 0.3, 0.5, 0.7]


def test_fit_thresholds_finds_the_grid_optimum():
    best = fit_thresholds(_synthetic_evaluate(0.3, 0.7), GRID, GRID)
    assert (best["tau_prune"], best["tau_accept"]) == (0.3, 0.7)
    assert best["value"] == pytest.approx(1.0)


def test_fit_thresholds_reports_a_missing_objective_clearly():
    with pytest.raises(KeyError, match="macro_f1"):
        fit_thresholds(_synthetic_evaluate(0.3, 0.7), GRID, GRID, objective="macro_f1")


def test_fit_thresholds_rejects_an_empty_grid():
    with pytest.raises(ValueError, match="empty"):
        fit_thresholds(_synthetic_evaluate(0.3, 0.7), [], [])


def test_transfer_gap_is_zero_when_the_two_splits_agree():
    """The portability claim's success case: thresholds fitted on one split are optimal on the other."""
    same = _synthetic_evaluate(0.3, 0.7)
    out = transfer_gap(same, same, GRID, GRID)
    assert out["gap"] == pytest.approx(0.0)
    assert out["fitted_tau_prune"] == out["oracle_tau_prune"] == 0.3


def test_transfer_gap_prices_a_shifted_optimum():
    """Fit peaks at (0.3, 0.7), evaluation split peaks at (0.7, 0.3): the transfer costs 0.8.

    The oracle on the held-out split scores 1.0. The transferred pair scores
    1 - |0.3-0.7| - |0.7-0.3| = 0.2.
    """
    out = transfer_gap(
        _synthetic_evaluate(0.3, 0.7), _synthetic_evaluate(0.7, 0.3), GRID, GRID
    )
    assert out["transferred_value"] == pytest.approx(0.2)
    assert out["oracle_value"] == pytest.approx(1.0)
    assert out["gap"] == pytest.approx(0.8)


def test_cohort_transfer_covers_both_directions():
    out = cohort_transfer(
        {"hcy": _synthetic_evaluate(0.3, 0.7), "gsc": _synthetic_evaluate(0.5, 0.5)},
        GRID, GRID,
    )
    assert set(out["pairs"]) == {"hcy->gsc", "gsc->hcy"}
    assert out["max_gap"] == pytest.approx(0.4)
    assert out["mean_gap"] == pytest.approx(0.4)


def test_kfold_split_partitions_the_reports_once():
    ids = [f"r{i}" for i in range(10)]
    splits = kfold_report_split(ids, k=5, seed=0)
    assert len(splits) == 5
    held_out = [rid for _, ev in splits for rid in ev]
    assert sorted(held_out) == sorted(ids)
    for fit, ev in splits:
        assert not set(fit) & set(ev)
        assert sorted(fit + ev) == sorted(ids)


def test_kfold_split_is_deterministic_given_a_seed():
    ids = [f"r{i}" for i in range(10)]
    assert kfold_report_split(ids, 5, seed=3) == kfold_report_split(ids, 5, seed=3)
    assert kfold_report_split(ids, 5, seed=3) != kfold_report_split(ids, 5, seed=4)


def test_kfold_split_rejects_impossible_requests():
    with pytest.raises(ValueError, match="at least 2"):
        kfold_report_split(["r0", "r1"], k=1)
    with pytest.raises(ValueError, match="cannot make"):
        kfold_report_split(["r0"], k=5)


def test_compare_raw_vs_calibrated_reports_the_gap_reduction():
    calibrated = transfer_gap(
        _synthetic_evaluate(0.3, 0.7), _synthetic_evaluate(0.3, 0.5), GRID, GRID
    )
    raw = transfer_gap(
        _synthetic_evaluate(0.3, 0.7), _synthetic_evaluate(0.7, 0.3), GRID, GRID
    )
    out = compare_raw_vs_calibrated(calibrated, raw)
    assert out["calibrated_gap"] == pytest.approx(0.2)
    assert out["raw_gap"] == pytest.approx(0.8)
    assert out["gap_reduction"] == pytest.approx(0.6)


# ── Cross-checks against the repo's existing ECE implementations ─────────────

def test_agrees_with_the_dashboard_equal_width_ece():
    """``apps.ui_common.calib.compute_ece`` is the number the tree dashboard has been showing."""
    from apps.ui_common.calib import compute_ece

    rng = np.random.default_rng(11)
    conf = rng.uniform(0.0, 1.0, 500)
    labels = (rng.uniform(size=conf.size) < conf).astype(int)
    assert ece_equal_width(conf, labels, 10) == pytest.approx(
        compute_ece(conf, labels.astype(float), 10)
    )




def test_f1_optimal_threshold_never_splits_a_run_of_ties():
    """``conf >= t`` admits every tied score, so F1 is scored only at the end of a tied block.

    Hand example: conf = [0.9, 0.5, 0.5, 0.5], labels = [1, 1, 0, 0]. At t = 0.9: TP 1, predicted
    1, F1 = 2/3. At t = 0.5: TP 2, predicted 4, F1 = 4/6 = 2/3. Scoring inside the 0.5 block would
    have found TP 2 of 2 predicted (F1 = 1.0) -- a threshold no rule can realise.
    """
    from hpo_extraction.evaluation.metrics.calibration import f1_optimal_threshold_diagnostic

    got = f1_optimal_threshold_diagnostic([0.9, 0.5, 0.5, 0.5], [1, 1, 0, 0])
    assert got["best_f1"] == pytest.approx(2 / 3)
    assert got["best_threshold"] == pytest.approx(0.9)


def test_f1_optimal_threshold_does_not_depend_on_input_order():
    """An isotonic map is all plateaus. The diagnostic once changed between two identical runs."""
    from hpo_extraction.evaluation.metrics.calibration import f1_optimal_threshold_diagnostic

    rng = np.random.default_rng(0)
    conf = np.repeat([0.9, 0.6, 0.3, 0.05], [50, 200, 400, 5000]).astype(float)
    labels = (rng.uniform(size=conf.size) < np.repeat([0.8, 0.45, 0.2, 0.01],
                                                      [50, 200, 400, 5000])).astype(int)
    gaps = {f1_optimal_threshold_diagnostic(conf[p], labels[p])["gap"]
            for p in (np.random.default_rng(s).permutation(conf.size) for s in range(10))}
    assert len(gaps) == 1
