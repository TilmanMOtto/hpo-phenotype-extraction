"""The retrieval-curve analysis, the four query-set conditions, the two gates, and the report-clustered bootstrap.

Four things in this experiment are easy to get subtly wrong and impossible to spot in a finished
figure, so each gets a test that pins it to something independently computable:

* **R5's closed form.** The whole point of the control condition is that it carries no Monte-Carlo
  noise. If the hypergeometric expression is wrong, every other condition is measured against a
  mis-drawn floor. Checked against an actual simulation.
* **The tau grid.** Placed *through the cost axis*, the tau giving mean bag ``b`` is the
  ``round(b * n_pairs)``-th largest score, so the grid is only even in the plotted quantity if
  that identity holds. Checked by recomputing the mean bag size the long way.
* **The survivor matrix.** One ``searchsorted`` and a reverse cumulative sum stand in for 120
  boolean masks. Checked against the masks.
* **The bootstrap count matrix.** ``ReportResampler`` hands out indices. The curve maths needs
  counts, and a mis-built count matrix silently reweights the cohort. Checked against the indices,
  and the whole curve checked against a plain per-draw loop.

The gate arithmetic itself is hand-worked on a three-patient toy cohort where every rank and
cosine is written out, so a mean bag size and a hit rate can be read off the fixture by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EXP = Path(__file__).resolve().parents[2] / "experiments" / "04_treephenorag" / "retrieval_curves"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import gate_curves  # noqa: E402
import query_sets  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# A toy cohort: 3 patients, 2 query terms each, small reports with written-out scores.
# ---------------------------------------------------------------------------

def _toy_scores() -> pd.DataFrame:
    """One condition's worth of rows for 3 patients x 2 queries, hand-built.

    Patient p1 has a 4-sentence report, p2 a 3-sentence one, p3 a 6-sentence one, so ``min(k, n)``
    actually bites at k=5 and the mean bag size is below 5.
    """
    spec = [
        # (patient, query, depth, n, [cosines], {relevant segment indices})
        ("p1", "HP:0000002", 2, 4, [0.90, 0.40, 0.20, 0.10], {0}),
        ("p1", "HP:0000003", 5, 4, [0.10, 0.20, 0.30, 0.85], {3}),
        ("p2", "HP:0000002", 2, 3, [0.15, 0.75, 0.05], {1, 2}),
        ("p2", "HP:0000003", 5, 3, [0.05, 0.10, 0.12], {0}),
        ("p3", "HP:0000002", 2, 6, [0.5, 0.4, 0.3, 0.2, 0.1, 0.05], {5}),
        ("p3", "HP:0000003", 5, 6, [0.95, 0.1, 0.1, 0.1, 0.1, 0.1], {0}),
    ]
    rows = []
    for pid, q, depth, n, cos, rel in spec:
        order = np.argsort(-np.asarray(cos), kind="stable")
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(1, n + 1)
        for idx in range(n):
            rows.append({
                "arm": "R1", "patient_id": pid, "query_hpo": q,
                "hop": 0 if depth == 5 else 1, "depth": depth,
                "depth_band": query_sets.depth_band(depth),
                "segment_idx": idx, "rank": int(rank[idx]), "n_segments": n,
                "cosine_sim": cos[idx], "is_gt_relevant": idx in rel,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Depth bands
# ---------------------------------------------------------------------------

class TestDepthBands:
    def test_bands_cover_every_depth_once(self):
        seen = [query_sets.depth_band(d) for d in range(0, 12)]
        assert seen[:7] == ["1", "1", "2", "3", "4-5", "4-5", "6+"]
        assert set(seen) <= set(query_sets.DEPTH_BANDS)

    def test_band_1_absorbs_the_root_side(self):
        # HP:0000118 itself is outside the query set, but depth 0 must not create a 6th band.
        assert query_sets.depth_band(0) == "1"


# ---------------------------------------------------------------------------
# R5, the closed form is the whole value of the control condition
# ---------------------------------------------------------------------------

class TestR5ClosedForm:
    def test_topk_matches_simulation(self):
        rng = np.random.default_rng(11)
        n = np.array([4, 3, 6, 20, 7], dtype=float)
        g = np.array([1, 2, 1, 3, 7], dtype=float)
        for k in (1, 2, 5, 9):
            exact = query_sets.r5_topk_hit(n, g, k)
            trials = 40_000
            sim = []
            for ni, gi in zip(n.astype(int), g.astype(int)):
                keep = min(k, ni)
                hits = 0
                for _ in range(trials):
                    draw = rng.choice(ni, size=keep, replace=False)
                    hits += int((draw < gi).any())   # relevance labels are exchangeable
                sim.append(hits / trials)
            assert exact == pytest.approx(np.array(sim), abs=0.01)

    def test_topk_saturates_when_everything_is_relevant(self):
        n = np.array([5.0])
        assert query_sets.r5_topk_hit(n, np.array([5.0]), 1) == pytest.approx([1.0])

    def test_topk_at_k_equal_n_is_certain(self):
        n = np.array([4.0, 3.0])
        assert query_sets.r5_topk_hit(n, np.array([1.0, 1.0]), 9) == pytest.approx([1.0, 1.0])

    def test_topk_at_k1_is_the_relevant_fraction(self):
        n = np.array([4.0, 10.0])
        g = np.array([1.0, 5.0])
        assert query_sets.r5_topk_hit(n, g, 1) == pytest.approx([0.25, 0.5])

    def test_bernoulli_matches_simulation(self):
        rng = np.random.default_rng(3)
        g = np.array([1, 2, 5], dtype=float)
        for p in (0.1, 0.35, 0.8):
            exact = query_sets.r5_bernoulli_hit(g, p)
            sim = [(rng.random((40_000, int(gi))) < p).any(axis=1).mean() for gi in g]
            assert exact == pytest.approx(np.array(sim), abs=0.01)

    def test_hit_is_monotone_in_k(self):
        n = np.array([12.0]), np.array([2.0])
        prev = 0.0
        for k in range(1, 13):
            cur = float(query_sets.r5_topk_hit(n[0], n[1], k)[0])
            assert cur >= prev - 1e-12
            prev = cur


# ---------------------------------------------------------------------------
# The tau grid and the survivor matrix
# ---------------------------------------------------------------------------

class TestTauGrid:
    def test_grid_lands_on_the_requested_mean_bag_sizes(self):
        rng = np.random.default_rng(5)
        n_pairs = 200
        cos = rng.normal(size=n_pairs * 30)
        taus = gate_curves.tau_grid(cos, n_pairs, x_max=10.0, n_points=20)
        # mean_bag(tau) = #{cos >= tau} / n_pairs, by design of the grid.
        got = np.array([(cos >= t).sum() / n_pairs for t in taus])
        want = np.linspace(10.0 / 20, 10.0, 20)
        assert got == pytest.approx(want, abs=0.02)

    def test_grid_is_descending_in_tau(self):
        cos = np.linspace(0, 1, 5000)
        taus = gate_curves.tau_grid(cos, 100, x_max=20.0, n_points=30)
        assert np.all(np.diff(taus) <= 1e-12)

    def test_survivor_counts_match_explicit_masks(self):
        scores = _toy_scores()
        pairs = gate_curves.ConditionPairs(scores)
        taus = gate_curves.tau_grid(pairs.cos, pairs.n_pairs, x_max=4.0, n_points=7)
        surv = gate_curves.survivor_counts(pairs.codes, pairs.cos, taus, pairs.n_pairs)
        for j, t in enumerate(taus):
            want = np.bincount(pairs.codes[pairs.cos >= t], minlength=pairs.n_pairs)
            assert np.array_equal(surv[:, j], want)

    def test_survivor_counts_never_exceed_the_report_length(self):
        scores = _toy_scores()
        pairs = gate_curves.ConditionPairs(scores)
        taus = gate_curves.tau_grid(pairs.cos, pairs.n_pairs, x_max=6.0, n_points=12)
        surv = gate_curves.survivor_counts(pairs.codes, pairs.cos, taus, pairs.n_pairs)
        assert np.all(surv <= pairs.n_segments[:, None])


# ---------------------------------------------------------------------------
# The per-condition reduction
# ---------------------------------------------------------------------------

class TestConditionPairs:
    def test_reduction_reads_the_fixture_back(self):
        pairs = gate_curves.ConditionPairs(_toy_scores())
        assert pairs.n_pairs == 6
        order = np.lexsort((pairs.query, pairs.patient))
        assert list(pairs.patient[order]) == ["p1", "p1", "p2", "p2", "p3", "p3"]
        # p2 / HP:0000002 has two relevant sentences (idx 1 and 2). The best-ranked is idx 1,
        # cosine 0.75, which is rank 1 in that report.
        sel = np.flatnonzero((pairs.patient == "p2") & (pairs.query == "HP:0000002"))[0]
        assert pairs.n_relevant[sel] == 2
        assert pairs.best_gold_rank[sel] == 1
        assert pairs.best_gold_cos[sel] == pytest.approx(0.75)

    def test_every_pair_has_a_best_gold_sentence(self):
        pairs = gate_curves.ConditionPairs(_toy_scores())
        assert np.all(pairs.n_relevant > 0)
        assert np.all(np.isfinite(pairs.best_gold_cos))


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------

class TestPatientBootstrap:
    def test_count_matrix_matches_the_resampler_indices(self):
        boot = gate_curves.PatientBootstrap(["p1", "p2", "p3"], n_resamples=64, seed=7)
        idx = boot.resampler.indices
        want = np.stack([np.bincount(row, minlength=3) for row in idx]).astype(float)
        assert np.array_equal(boot.counts, want)
        # Every draw resamples the full cohort size.
        assert np.all(boot.counts.sum(axis=1) == 3)

    def test_curve_matches_a_plain_per_draw_loop(self):
        scores = _toy_scores()
        pairs = gate_curves.ConditionPairs(scores)
        patients = sorted(set(pairs.patient))
        boot = gate_curves.PatientBootstrap(patients, n_resamples=200, seed=1)
        rows = boot.rows(pairs.patient)

        ks = np.array([1.0, 2.0, 5.0])
        hit = (pairs.best_gold_rank[:, None] <= ks[None, :]).astype(float)
        bag = np.minimum(ks[None, :], pairs.n_segments[:, None].astype(float))
        out = boot.curve(rows, hit, bag)

        # The slow, obviously-correct version: rebuild each draw's pair set by hand.
        by_patient = {p: np.flatnonzero(pairs.patient == p) for p in patients}
        draws = []
        for row in boot.resampler.indices:
            sel = np.concatenate([by_patient[patients[i]] for i in row])
            draws.append(hit[sel].mean(axis=0))
        draws = np.asarray(draws)
        for j in range(len(ks)):
            lo, hi = np.percentile(draws[:, j], [2.5, 97.5])
            assert out["hit_lo"][j] == pytest.approx(lo, abs=1e-9)
            assert out["hit_hi"][j] == pytest.approx(hi, abs=1e-9)
        assert out["hit_rate"] == pytest.approx(hit.mean(axis=0))
        assert out["mean_bag"] == pytest.approx(bag.mean(axis=0))

    def test_same_seed_is_reproducible_and_different_seed_is_not(self):
        a = gate_curves.PatientBootstrap(list("abcde"), n_resamples=50, seed=0)
        b = gate_curves.PatientBootstrap(list("abcde"), n_resamples=50, seed=0)
        c = gate_curves.PatientBootstrap(list("abcde"), n_resamples=50, seed=1)
        assert np.array_equal(a.counts, b.counts)
        assert not np.array_equal(a.counts, c.counts)


# ---------------------------------------------------------------------------
# The gate arithmetic, hand-worked
# ---------------------------------------------------------------------------

class TestGateArithmetic:
    def test_mean_bag_at_k5_is_below_5_because_reports_run_out(self):
        pairs = gate_curves.ConditionPairs(_toy_scores())
        bag = np.minimum(5.0, pairs.n_segments.astype(float))
        # reports of 4, 4, 3, 3, 6, 6 sentences -> bags of 4, 4, 3, 3, 5, 5
        assert sorted(bag) == [3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
        assert bag.mean() == pytest.approx(4.0)

    def test_topk_hit_is_monotone_and_saturates(self):
        scores = _toy_scores()
        pairs = gate_curves.ConditionPairs(scores)
        patients = sorted(set(pairs.patient))
        boot = gate_curves.PatientBootstrap(patients, n_resamples=20, seed=0)
        mats = gate_curves._gate_matrices(
            "R1", pairs, np.arange(pairs.n_pairs),
            gate_curves.survivor_counts(
                pairs.codes, pairs.cos,
                gate_curves.tau_grid(pairs.cos, pairs.n_pairs, 6.0, 8), pairs.n_pairs),
            gate_curves.tau_grid(pairs.cos, pairs.n_pairs, 6.0, 8), np.linspace(0.1, 1.0, 8),
        )
        _, hit_k, bag_k = mats["topk"]
        rates = hit_k.mean(axis=0)
        assert np.all(np.diff(rates) >= -1e-12)
        assert rates[-1] == pytest.approx(1.0)          # k=60 forwards the whole report
        assert np.all(np.diff(bag_k.mean(axis=0)) >= -1e-12)

    def test_r5_uses_the_reference_conditions_pair_set(self):
        """R5 owns no index, so it must be scored on R1's pairs, same n, same g, same patients."""
        scores = _toy_scores()
        pairs = gate_curves.ConditionPairs(scores)
        mats = gate_curves._gate_matrices("R5", pairs, np.arange(pairs.n_pairs),
                                          None, None, np.linspace(0.1, 1.0, 5))
        _, hit_k, bag_k = mats["topk"]
        assert hit_k.shape == (pairs.n_pairs, len(query_sets.K_GRID))
        assert np.all((hit_k >= 0) & (hit_k <= 1))
        # At k=1 chance is g/n for every pair.
        assert hit_k[:, 0] == pytest.approx(pairs.n_relevant / pairs.n_segments)


# ---------------------------------------------------------------------------
# End to end: the report stage on the toy cohort
# ---------------------------------------------------------------------------

class TestReportStage:
    def test_report_runs_and_writes_both_figures(self, tmp_path):
        scores = pd.concat(
            [_toy_scores().assign(arm=arm) for arm in query_sets.SCORED_CONDITIONS],
            ignore_index=True,
        )
        scores.to_csv(tmp_path / query_sets.SCORES_FILE, index=False, compression="gzip")

        metrics = gate_curves.run_report(tmp_path, n_resamples=200, seed=0, x_max=6.0)

        points = pd.read_csv(tmp_path / query_sets.OPERATING_FILE)
        assert set(points["arm"]) == set(query_sets.CONDITIONS)
        assert set(points["gate"]) == {"topk", "tau"}
        assert set(points["figure"]) == {"A", "B", "C"}

        # Figure C is the union of A's facet and B's five, so its pair count must be the total and
        # it must never sit above A, a bug that merged the facets would show up here first.
        n_c = points[points["figure"] == "C"]["n_pairs"].max()
        n_a = points[points["figure"] == "A"]["n_pairs"].max()
        n_b = points[points["figure"] == "B"].groupby("facet")["n_pairs"].max().sum()
        assert n_c == n_b, "Figure C must pool exactly the depth bands of Figure B"
        assert n_c >= n_a, "Figure C covers Figure A's pairs and more"
        # Intervals must bracket their point estimate.
        assert (points["hit_lo"] <= points["hit_rate"] + 1e-9).all()
        assert (points["hit_hi"] >= points["hit_rate"] - 1e-9).all()

        for name in (query_sets.PAIRS_FILE, query_sets.SUMMARY_FILE, query_sets.DELTAS_FILE,
                     f"{query_sets.FIG_A}.png", f"{query_sets.FIG_A}.pdf",
                     f"{query_sets.FIG_B}.png", f"{query_sets.FIG_B}.pdf",
                     f"{query_sets.FIG_C}.png", f"{query_sets.FIG_C}.pdf"):
            assert (tmp_path / name).is_file(), name

        # Conditions fed identical scores must differ from R1 by zero, and the paired interval
        # must say so, a delta CI that excluded zero here would mean the pairing is broken.
        deltas = pd.read_csv(tmp_path / query_sets.DELTAS_FILE)
        assert set(deltas["arm"]) == set(query_sets.CONDITIONS) - {gate_curves.BASELINE_CONDITION}
        assert (deltas["gate"] == "topk").all()
        # Every closure condition must also be differenced against ITS OWN base, not only against the
        # shipped condition, that contrast is what separates the pooling from the representation.
        for closed, base in query_sets.CLOSURE_OF.items():
            got = deltas[(deltas["arm"] == closed) & (deltas["vs"] == base)]
            assert not got.empty, f"missing paired contrast {closed} - {base}"
        same = deltas[deltas["arm"].isin(set(query_sets.SCORED_CONDITIONS)
                                         - {gate_curves.BASELINE_CONDITION})]
        assert same["delta"].abs().max() == pytest.approx(0.0, abs=1e-12)
        assert (same["delta_lo"] <= 0).all() and (same["delta_hi"] >= 0).all()
        # R5 is chance on the same pairs, so it must be strictly worse at small k.
        r5_k1 = deltas[(deltas["arm"] == "R5") & (deltas["param"] == 1.0)]
        assert (r5_k1["delta"] < 0).all()

        assert "figA_hit_at_k5_R1" in metrics
        assert 0.0 <= metrics["figA_hit_at_k5_R1"] <= 1.0

    def test_identical_conditions_give_identical_curves(self, tmp_path):
        """The three scored conditions are fed the same scores here, so nothing may separate them.

        This is the guard against the reduction picking up a condition-dependent ordering somewhere, a real difference between conditions has to come from the scores, never from the plumbing.
        """
        scores = pd.concat(
            [_toy_scores().assign(arm=arm) for arm in query_sets.SCORED_CONDITIONS],
            ignore_index=True,
        )
        scores.to_csv(tmp_path / query_sets.SCORES_FILE, index=False, compression="gzip")
        gate_curves.run_report(tmp_path, n_resamples=50, seed=0, x_max=6.0)

        points = pd.read_csv(tmp_path / query_sets.OPERATING_FILE)
        # R5 is excluded on purpose: it owns no index, so its continuous gate is swept over a
        # keep-probability rather than a cosine and its parameter grid is a different axis.
        scored = points[(points["figure"] == "A") & points["arm"].isin(query_sets.SCORED_CONDITIONS)]
        wide = scored.pivot_table(index=["gate", "param"], columns="arm", values="hit_rate")
        assert not wide.isna().any().any()
        for arm in set(query_sets.SCORED_CONDITIONS) - {"R1"}:
            assert wide[arm].to_numpy() == pytest.approx(wide["R1"].to_numpy())

    def test_mismatched_pair_sets_are_refused(self, tmp_path):
        """Conditions not covering the same pairs are not paired, and a figure would lie about it."""
        short = _toy_scores()
        short = short[short["patient_id"] != "p3"]
        odd = query_sets.SCORED_CONDITIONS[-1]
        scores = pd.concat(
            [_toy_scores().assign(arm=arm) for arm in query_sets.SCORED_CONDITIONS if arm != odd]
            + [short.assign(arm=odd)],
            ignore_index=True,
        )
        scores.to_csv(tmp_path / query_sets.SCORES_FILE, index=False, compression="gzip")
        with pytest.raises(AssertionError, match="same pairs"):
            gate_curves.run_report(tmp_path, n_resamples=10, seed=0, x_max=6.0)
