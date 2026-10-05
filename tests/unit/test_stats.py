"""Intervals, significance tests and folds, the machinery the results chapters rest on.

These tests fix the properties that make the numbers in the thesis mean what they say:

* an interval computed twice from one seed is byte-identical, and resampling **reports** gives a
  wider interval than resampling cells would;
* a system tested against itself is not significant, and one that is uniformly better is;
* Holm reproduces a hand-computed example;
* every report lands in one outer fold per repetition, strata stay proportional, and the
  fold file round-trips.
"""

from __future__ import annotations


import pytest

from hpo_extraction.evaluation.stats import (
    ReportResampler,
    bootstrap_reports,
    flatten_ci,
    holm,
    nested_folds,
    paired_randomisation,
    read_folds,
    size_tertiles,
    stratified_report_folds,
    write_folds,
)
from hpo_extraction.evaluation.metrics import micro_prf


def micro(units):
    """Metric function over ``(gold, pred)`` units, in the shape the bootstrap expects."""
    gold = [g for g, _ in units]
    pred = [p for _, p in units]
    p, r, f = micro_prf(gold, pred)
    return {"micro_precision": p, "micro_recall": r, "micro_f1": f}


def micro_pair(gold_sets, pred_sets):
    """Metric function over aligned ground truth/pred sequences, the shape the test expects."""
    p, r, f = micro_prf(gold_sets, pred_sets)
    return {"micro_precision": p, "micro_recall": r, "micro_f1": f}


@pytest.fixture
def units():
    """Twenty reports, five annotated terms each, with a system that gets 3/5 on every one."""
    out = []
    for i in range(20):
        gold = {f"HP:{i}{j}" for j in range(5)}
        pred = {f"HP:{i}{j}" for j in range(3)}
        out.append((gold, pred))
    return out


# ── Bootstrap ────────────────────────────────────────────────────────────────

class TestResampler:
    def test_same_seed_gives_an_identical_matrix(self):
        a = ReportResampler(50, n_resamples=100, seed=7)
        b = ReportResampler(50, n_resamples=100, seed=7)
        assert (a.indices == b.indices).all()

    def test_different_seeds_differ(self):
        a = ReportResampler(50, n_resamples=100, seed=7)
        b = ReportResampler(50, n_resamples=100, seed=8)
        assert not (a.indices == b.indices).all()

    def test_draws_are_with_replacement_and_full_length(self):
        sampler = ReportResampler(10, n_resamples=20, seed=0)
        draws = list(sampler.resample(list(range(10))))
        assert all(len(d) == 10 for d in draws)
        assert any(len(set(d)) < 10 for d in draws), "with replacement, some draw must repeat"

    def test_length_mismatch_is_caught(self):
        sampler = ReportResampler(10, n_resamples=5, seed=0)
        with pytest.raises(ValueError, match="built for 10 reports"):
            list(sampler.resample(list(range(9))))


class TestBootstrap:
    def test_reproducible_from_the_seed(self, units):
        a = bootstrap_reports(micro, units, n_resamples=200, seed=3)
        b = bootstrap_reports(micro, units, n_resamples=200, seed=3)
        assert a == b

    def test_point_estimate_is_the_unresampled_value(self, units):
        got = bootstrap_reports(micro, units, n_resamples=100, seed=0)
        assert got["micro_f1"]["point"] == pytest.approx(micro(units)["micro_f1"])

    def test_interval_brackets_the_point(self, units):
        got = bootstrap_reports(micro, units, n_resamples=500, seed=0)
        for stats in got.values():
            assert stats["lo"] <= stats["point"] <= stats["hi"]

    def test_a_constant_metric_has_a_degenerate_interval(self, units):
        """Every report scores 3/5 identically, so no resample can move micro precision."""
        got = bootstrap_reports(micro, units, n_resamples=200, seed=0)
        assert got["micro_precision"]["lo"] == pytest.approx(got["micro_precision"]["hi"])

    def test_report_level_is_wider_than_cell_level(self):
        """The reason the unit is the report: cells within one are correlated.

        Two cohorts with identical pooled counts, one where success is clustered by report, one
        where it is spread evenly. Resampling reports must show the clustered cohort as the more
        uncertain. A cell-level bootstrap could not tell them apart at all.
        """
        clustered, spread = [], []
        for i in range(20):
            gold = {f"HP:{i}{j}" for j in range(10)}
            # clustered: ten reports perfect, ten reports empty
            clustered.append((gold, set(gold) if i < 10 else set()))
            # spread: every report gets half right
            spread.append((gold, {f"HP:{i}{j}" for j in range(5)}))

        assert micro(clustered)["micro_recall"] == pytest.approx(micro(spread)["micro_recall"])

        c = bootstrap_reports(micro, clustered, n_resamples=500, seed=0)["micro_recall"]
        s = bootstrap_reports(micro, spread, n_resamples=500, seed=0)["micro_recall"]
        assert (c["hi"] - c["lo"]) > (s["hi"] - s["lo"])

    def test_shared_resampler_gives_both_systems_the_same_draws(self, units):
        sampler = ReportResampler(len(units), n_resamples=100, seed=0)
        worse = [(g, set()) for g, _ in units]
        a = bootstrap_reports(micro, units, resampler=sampler)
        b = bootstrap_reports(micro, worse, resampler=sampler)
        assert a["micro_f1"]["point"] > b["micro_f1"]["point"]
        assert sampler.n_resamples == 100

    def test_flatten_ci_produces_table_columns(self, units):
        flat = flatten_ci(bootstrap_reports(micro, units, n_resamples=50, seed=0))
        assert {"micro_f1", "micro_f1_lo", "micro_f1_hi"} <= set(flat)


# ── Randomisation ────────────────────────────────────────────────────────────

class TestPairedRandomisation:
    def test_a_system_against_itself_is_not_significant(self, units):
        gold = [g for g, _ in units]
        pred = [p for _, p in units]
        got = paired_randomisation(gold, pred, pred, micro_pair, n_permutations=200, seed=0)
        assert got["delta"] == 0.0
        assert got["p_value"] == pytest.approx(1.0), "every swap is a no-op, so all are as extreme"

    def test_a_uniformly_better_system_hits_the_floor(self, units):
        """B predicts nothing anywhere, so no swap pattern reproduces the observed gap."""
        gold = [g for g, _ in units]
        pred_a = [p for _, p in units]
        pred_b = [set() for _ in units]
        got = paired_randomisation(gold, pred_a, pred_b, micro_pair, n_permutations=200, seed=0)
        assert got["delta"] > 0
        assert got["p_value"] == pytest.approx(1 / 201)

    def test_p_value_is_strictly_positive(self, units):
        gold = [g for g, _ in units]
        pred_a = [p for _, p in units]
        pred_b = [set() for _ in units]
        got = paired_randomisation(gold, pred_a, pred_b, micro_pair, n_permutations=50, seed=0)
        assert got["p_value"] > 0, "the +1 correction must keep the tail estimate off zero"

    def test_misalignment_raises_rather_than_returning_a_number(self, units):
        gold = [g for g, _ in units]
        pred = [p for _, p in units]
        with pytest.raises(ValueError, match="not aligned"):
            paired_randomisation(gold, pred, pred[:-1], micro_pair, n_permutations=10)

    def test_result_is_reproducible(self, units):
        gold = [g for g, _ in units]
        a = [p for _, p in units]
        b = [set(list(p)[:1]) for _, p in units]
        first = paired_randomisation(gold, a, b, micro_pair, n_permutations=100, seed=5)
        second = paired_randomisation(gold, a, b, micro_pair, n_permutations=100, seed=5)
        assert first == second


# ── Holm ─────────────────────────────────────────────────────────────────────

class TestHolm:
    def test_hand_computed_four_hypothesis_example(self):
        """p = .01, .02, .03, .04 with m = 4.

        Multipliers 4, 3, 2, 1 give .04, .06, .06, .04. The running maximum then makes the sequence
        monotone: .04, .06, .06, .06. At alpha = .05 only the first is rejected, and the fourth is
        a good illustration of why the running maximum is needed, since its raw multiplication
        (.04) would otherwise let the *largest* raw p-value through.
        """
        got = holm({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}, alpha=0.05)
        assert got["a"]["p_holm"] == pytest.approx(0.04)
        assert got["b"]["p_holm"] == pytest.approx(0.06)
        assert got["c"]["p_holm"] == pytest.approx(0.06)
        assert got["d"]["p_holm"] == pytest.approx(0.06)
        assert [got[k]["reject"] for k in "abcd"] == [True, False, False, False]

    def test_adjusted_values_are_monotone_in_the_raw_ones(self):
        got = holm({f"h{i}": p for i, p in enumerate([0.001, 0.2, 0.04, 0.9, 0.03])})
        ordered = sorted(got.values(), key=lambda v: v["p_raw"])
        adjusted = [v["p_holm"] for v in ordered]
        assert adjusted == sorted(adjusted)

    def test_never_exceeds_one(self):
        got = holm({"a": 0.8, "b": 0.9, "c": 0.95})
        assert all(v["p_holm"] <= 1.0 for v in got.values())

    def test_single_hypothesis_is_unadjusted(self):
        got = holm({"only": 0.03})
        assert got["only"]["p_holm"] == pytest.approx(0.03)
        assert got["only"]["reject"] is True

    def test_is_less_conservative_than_bonferroni(self):
        """Holm's advantage, stated as a test: same guarantee, never larger adjusted values."""
        raw = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
        got = holm(raw)
        for key, p in raw.items():
            assert got[key]["p_holm"] <= min(1.0, len(raw) * p) + 1e-12


# ── Folds ────────────────────────────────────────────────────────────────────

@pytest.fixture
def cohort():
    """118 reports with gold-set sizes spanning the real cohort's range."""
    ids = [f"HCY{i:03d}" for i in range(118)]
    sizes = {rid: (i % 17) for i, rid in enumerate(ids)}
    return ids, sizes


class TestTertiles:
    def test_three_roughly_equal_strata(self, cohort):
        _, sizes = cohort
        strata = size_tertiles(sizes)
        counts = [sum(1 for v in strata.values() if v == s) for s in (0, 1, 2)]
        assert sum(counts) == 118
        assert max(counts) - min(counts) <= 1

    def test_ordering_is_by_size(self, cohort):
        _, sizes = cohort
        strata = size_tertiles(sizes)
        # no report in a lower stratum may be larger than one in a higher stratum
        for low in (0, 1):
            hi_min = min(sizes[r] for r, s in strata.items() if s == low + 1)
            lo_max = max(sizes[r] for r, s in strata.items() if s == low)
            assert lo_max <= hi_min

    def test_ties_do_not_depend_on_input_order(self):
        flat = {f"r{i}": 5 for i in range(9)}
        assert size_tertiles(flat) == size_tertiles(dict(reversed(list(flat.items()))))

    def test_empty_input(self):
        assert size_tertiles({}) == {}


class TestStratifiedFolds:
    def test_partitions_the_cohort(self, cohort):
        ids, sizes = cohort
        folds = stratified_report_folds(ids, size_tertiles(sizes), k=5, seed=0)
        assert len(folds) == 5
        pooled = [r for f in folds for r in f]
        assert sorted(pooled) == sorted(ids)
        assert len(pooled) == len(set(pooled)), "no report may appear twice"

    def test_fold_sizes_are_balanced(self, cohort):
        ids, sizes = cohort
        folds = stratified_report_folds(ids, size_tertiles(sizes), k=5, seed=0)
        lengths = [len(f) for f in folds]
        assert max(lengths) - min(lengths) <= 1

    def test_strata_stay_proportional(self, cohort):
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        folds = stratified_report_folds(ids, strata, k=5, seed=0)
        for stratum in (0, 1, 2):
            per_fold = [sum(1 for r in f if strata[r] == stratum) for f in folds]
            assert max(per_fold) - min(per_fold) <= 1

    def test_reproducible_from_the_seed(self, cohort):
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        a = stratified_report_folds(ids, strata, k=5, seed=11)
        b = stratified_report_folds(ids, strata, k=5, seed=11)
        assert a == b

    def test_different_seeds_reshuffle(self, cohort):
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        assert (stratified_report_folds(ids, strata, seed=1)
                != stratified_report_folds(ids, strata, seed=2))

    def test_unstratified_still_works(self, cohort):
        ids, _ = cohort
        folds = stratified_report_folds(ids, None, k=5, seed=0)
        assert sorted(r for f in folds for r in f) == sorted(ids)

    def test_missing_stratum_is_an_error(self, cohort):
        ids, sizes = cohort
        partial = size_tertiles(sizes)
        partial.pop(ids[0])
        with pytest.raises(ValueError, match="no stratum"):
            stratified_report_folds(ids, partial, k=5, seed=0)

    def test_too_few_reports_is_an_error(self):
        with pytest.raises(ValueError, match="cannot make"):
            stratified_report_folds(["a", "b"], None, k=5)


class TestNestedFolds:
    def test_shape(self, cohort):
        ids, sizes = cohort
        rows = nested_folds(ids, size_tertiles(sizes), k_outer=5, k_inner=5, repetitions=3, seed=0)
        assert len(rows) == 15

    def test_each_report_is_evaluated_once_per_repetition(self, cohort):
        ids, sizes = cohort
        rows = nested_folds(ids, size_tertiles(sizes), k_outer=5, k_inner=5, repetitions=3, seed=0)
        for rep in range(3):
            evaluated = [r for row in rows if row["repetition"] == rep for r in row["eval_ids"]]
            assert sorted(evaluated) == sorted(ids)

    def test_train_and_eval_never_overlap(self, cohort):
        ids, sizes = cohort
        rows = nested_folds(ids, size_tertiles(sizes), repetitions=2, seed=0)
        for row in rows:
            assert not (set(row["eval_ids"]) & set(row["train_ids"]))
            assert set(row["eval_ids"]) | set(row["train_ids"]) == set(ids)

    def test_inner_folds_partition_the_training_split(self, cohort):
        ids, sizes = cohort
        rows = nested_folds(ids, size_tertiles(sizes), repetitions=2, seed=0)
        for row in rows:
            pooled = [r for f in row["inner_folds"] for r in f]
            assert sorted(pooled) == sorted(row["train_ids"])
            assert len(pooled) == len(set(pooled))

    def test_no_evaluation_report_leaks_into_selection(self, cohort):
        """The property nested CV exists for."""
        ids, sizes = cohort
        rows = nested_folds(ids, size_tertiles(sizes), repetitions=2, seed=0)
        for row in rows:
            inner = {r for f in row["inner_folds"] for r in f}
            assert not (inner & set(row["eval_ids"]))

    def test_repetitions_differ(self, cohort):
        ids, sizes = cohort
        rows = nested_folds(ids, size_tertiles(sizes), repetitions=2, seed=0)
        first = [r for r in rows if r["repetition"] == 0][0]["eval_ids"]
        second = [r for r in rows if r["repetition"] == 1][0]["eval_ids"]
        assert first != second


class TestFoldFile:
    def test_round_trips(self, cohort, tmp_path):
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        rows = nested_folds(ids, strata, k_outer=5, k_inner=5, repetitions=2, seed=0)
        path = write_folds(tmp_path / "folds.csv", rows, strata)
        back = read_folds(path)

        assert len(back) == len(rows)
        for original, restored in zip(rows, back):
            assert restored["repetition"] == original["repetition"]
            assert restored["outer_fold"] == original["outer_fold"]
            assert restored["eval_ids"] == original["eval_ids"]
            assert restored["train_ids"] == original["train_ids"]
            assert restored["inner_folds"] == original["inner_folds"]

    def test_file_is_byte_identical_for_one_seed(self, cohort, tmp_path):
        """What makes the committed fold file a reproducibility claim rather than an artifact."""
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        first = write_folds(tmp_path / "a.csv", nested_folds(ids, strata, repetitions=2, seed=0),
                            strata).read_bytes()
        second = write_folds(tmp_path / "b.csv", nested_folds(ids, strata, repetitions=2, seed=0),
                             strata).read_bytes()
        assert first == second

    def test_one_line_per_report_per_outer_fold_per_repetition(self, cohort, tmp_path):
        """Every report has a role in every outer fold: eval in one, train in the other four."""
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        rows = nested_folds(ids, strata, k_outer=5, k_inner=5, repetitions=3, seed=0)
        path = write_folds(tmp_path / "folds.csv", rows, strata)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) - 1 == len(ids) * 5 * 3

    def test_each_report_is_eval_once_per_repetition_in_the_file(self, cohort, tmp_path):
        import csv as _csv
        ids, sizes = cohort
        strata = size_tertiles(sizes)
        rows = nested_folds(ids, strata, k_outer=5, k_inner=5, repetitions=3, seed=0)
        path = write_folds(tmp_path / "folds.csv", rows, strata)
        with path.open(encoding="utf-8") as fh:
            evals = [(r["repetition"], r["report_id"])
                     for r in _csv.DictReader(fh) if r["role"] == "eval"]
        assert len(evals) == len(ids) * 3
        assert len(set(evals)) == len(evals)
