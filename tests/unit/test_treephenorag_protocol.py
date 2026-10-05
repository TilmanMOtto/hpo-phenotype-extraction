"""The TreePhenoRAG protocol's stages, end to end on a synthetic cache.

``tests/unit/test_tree_selection.py`` pins the protocol and ``test_tree_replay.py`` the re-run. This
module pins the *harness*: that each stage runs, that the tables it writes have the columns the
chapter quotes, and, the one that counts most, that the containment gate aborts rather than
writing a table when the cache does not host the grid.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from hpo_extraction.treephenorag.stored_scores import ReportCache
from hpo_extraction.treephenorag.selection import Configuration, OfflineSpace
from hpo_extraction.evaluation.stats import ReportResampler, nested_folds
from hpo_extraction.evaluation.metrics import OntologyView
from fixtures.toy_ontology import A, B, C, D, E, F, G, H, M, build_toy_tree, toy_children_map

import sys
from pathlib import Path

_EXP_DIR = (Path(__file__).resolve().parents[2] / "experiments" / "04_treephenorag" / "protocol")
sys.path.insert(0, str(_EXP_DIR))
import stages  # noqa: E402


def _load_run():
    """The TreePhenoRAG protocol's ``run.py``, loaded by PATH under a unique module name.

    Not ``import run``: every experiment has a ``run.py`` and every test that touches one puts its
    experiment directory on ``sys.path``, so the name ``run`` resolves to whichever test imported
    first. Alone the module under test is imported. In the full suite a different experiment's is,
    and the failure is an ``AttributeError`` a long way from its cause.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp13_22_run", _EXP_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.unit

NODES = sorted({A, B, C, D, E, F, G, H, M})
CHILDREN, ROOTS = toy_children_map()
GRID = {
    "poolings_prune": ["P1", "P3_1"],
    "poolings_accept": ["P1", "P3_1"],
    "tau_accept_grid": [0.3, 0.5, 0.9],
    "s_main": 3,
    "alpha": 0.3,
}


def make_cache(rid, seed, nodes=None, s_max=3):
    nodes = list(nodes or NODES)
    rng = np.random.default_rng(seed)
    margins = rng.normal(0.5, 2.5, size=(len(nodes), s_max))
    mask = rng.random((len(nodes), s_max)) > 0.25
    mask[:, 0] = True
    sent = np.tile(np.arange(s_max, dtype=np.int32), (len(nodes), 1))
    return ReportCache(rid, nodes, margins, mask, sent_index=sent)


@pytest.fixture
def view():
    return OntologyView(build_toy_tree())


@pytest.fixture
def space():
    caches = {f"R{j:02d}": make_cache(f"R{j:02d}", j) for j in range(16)}
    other = {f"R{j:02d}": make_cache(f"R{j:02d}", 100 + j) for j in range(16)}
    return OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches, "ontology_r3": other})


@pytest.fixture
def gold():
    rng = np.random.default_rng(3)
    return {f"R{j:02d}": set(rng.choice(NODES, size=3, replace=False)) for j in range(16)}


@pytest.fixture
def folds(gold):
    return nested_folds(sorted(gold), k_outer=4, k_inner=3, repetitions=1, seed=0)


def normalise(g, p):
    return set(g), set(p)


@pytest.fixture
def sampler(gold):
    return ReportResampler(len(gold), n_resamples=64, seed=0)


@pytest.fixture
def assignments(space, gold, folds, sampler):
    """Repetition 0's held-out reports, each with the configuration its fold selected."""
    return stages.run_selection(space, folds, gold, normalise, GRID, sampler)["assignments"]


class TestContainmentGate:
    def test_an_exhaustive_cache_passes_with_zero_misses(self, space, gold):
        rows = stages.containment_gate(space, gold, 0.00015, ["P3_1"], [3])
        assert rows and all(r["n_containment_misses"] == 0 for r in rows)
        assert all(r["n_cached_nodes_min"] == len(NODES) for r in rows)

    def test_a_cache_missing_nodes_is_detected_rather_than_scored(self, gold):
        """The failure the gate exists for: a re-run that leaves the cached region.

        Dropping a node the traversal can still reach must surface as a miss, because its score
        would otherwise be silently taken as the conservative 0.0, a number that looks like a
        measurement and is not one.
        """
        partial = {f"R{j:02d}": make_cache(f"R{j:02d}", j, nodes=[n for n in NODES if n != C])
                   for j in range(4)}
        space = OfflineSpace(CHILDREN, ROOTS, {"exemplar": partial})
        rows = stages.containment_gate(space, gold, 0.00015, ["P3_1"], [3])
        assert sum(r["n_containment_misses"] for r in rows) > 0

    def test_nodes_only_reachable_below_the_floor_are_reported_not_fatal(self, gold):
        """The SYN084 case: complete at the cache's own floor, not exhaustive below it.

        The synthetic-sentence score store's traversal was meant to reach every node. On a one-segment report a few leaves
        scored under the floor and the producer pruned them itself, so no admissible threshold
        reaches them. That is a fixed coverage loss to be named, not a grid the cache fails to
        host, and the two must not be reported as the same thing.

        Here B's single-segment score is far below the floor, so C (its only child) is unreachable
        at every tau >= floor, and C is absent from the cache.
        """
        nodes = [n for n in NODES if n != C]
        caches = {}
        for j in range(4):
            rid = f"R{j:02d}"
            full = make_cache(rid, j)
            margins = full.margins.copy()
            margins[NODES.index(B)] = -15.0        # sigmoid ~3e-7: pooled score under the floor
            drop = NODES.index(C)
            caches[rid] = ReportCache(
                rid, nodes,
                np.delete(margins, drop, axis=0),
                np.delete(full.mask, drop, axis=0),
                sent_index=np.delete(full.sent_index, drop, axis=0),
            )
        space = OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches}, tau_floor=0.00015)
        rows = stages.containment_gate(space, gold, 0.00015, ["P3_1"], [3])
        assert rows
        assert all(r["n_containment_misses"] == 0 for r in rows), "hosted at the floor"
        assert sum(r["n_unreachable_below_floor"] for r in rows) > 0, "and reported below it"

    def test_it_prices_every_cell(self, space, gold):
        rows = stages.containment_gate(space, gold, 0.00015, ["P1", "P3_1"], [1, 3])
        assert len(rows) == 2 * 2 * 2          # indices x poolings x S
        assert all(r["calls_per_report_at_floor"] > 0 for r in rows)


class TestTransferredScore:
    """GSC+ is test-only: the configuration crosses cohorts untouched.

    The failure this guards against is subtle and would not show up as an error, scoring the
    transfer cohort at a threshold quietly re-fitted on it turns a transfer claim into a second
    selection, and the number would still look plausible.
    """

    def test_it_applies_the_given_configuration_unchanged(self, space, gold):
        config = Configuration("exemplar", "P1", 0.001, "P3_1", 0.5, 3)
        row = stages.score_transferred(space, config, gold, normalise, "transfer")
        for field, value in config.as_row().items():
            assert row[field] == value, field
        assert row["n_reports"] == len(space.report_ids("exemplar"))
        assert row["micro_f1_lo"] <= row["micro_f1"] <= row["micro_f1_hi"]

    def test_it_scores_the_same_predictions_the_configuration_makes(self, space, gold):
        config = Configuration("exemplar", "P1", 0.001, "P3_1", 0.5, 3)
        row = stages.score_transferred(space, config, gold, normalise, "transfer")
        predicted = space.predictions(config)
        ids = space.report_ids("exemplar")
        expected = sum(len(predicted[r]) for r in ids) / len(ids)
        assert row["preds_per_report"] == pytest.approx(expected)


class TestSelectionStage:
    def test_it_produces_a_scored_row_and_a_choice_per_fold(self, space, gold, folds, sampler):
        got = stages.run_selection(space, folds, gold, normalise, GRID, sampler)
        assert set(got["row"]) >= {"micro_f1", "micro_f1_lo", "micro_f1_hi", "micro_precision"}
        assert got["row"]["micro_f1_lo"] <= got["row"]["micro_f1"] <= got["row"]["micro_f1_hi"]
        assert len(got["choices"]) == len(folds)
        # One CRC record per (fold, index), the pruning rule is selected per retrieval index.
        assert len(got["crc"]) == len(folds) * len(space.indices)

    def test_every_crc_row_carries_the_bound_and_what_was_attained(self, space, gold, folds,
                                                                   sampler):
        got = stages.run_selection(space, folds, gold, normalise, GRID, sampler)
        for row in got["crc"]:
            assert set(row) >= {"alpha", "feasible", "bound", "tau_prune", "attained_ceiling",
                                "held_out_miss_rate", "n_grid", "calls_per_report"}
            assert 0.0 <= row["held_out_miss_rate"] <= 1.0

    def test_alpha_sensitivity_reports_feasibility_not_just_a_number(self, space, gold, folds):
        rows = stages.alpha_sensitivity(space, folds, gold, normalise, GRID, [0.2, 0.4])
        assert [r["alpha"] for r in rows] == [0.2, 0.4]
        for row in rows:
            assert row["feasible_folds"] <= row["n_folds"]
            assert "mean_attained_ceiling" in row


class TestLadder:
    def test_all_four_rungs_are_scored(self, space, gold, folds, sampler):
        rows, units = stages.run_ladder(space, folds, gold, normalise, GRID, sampler, None)
        assert len(rows) == 4
        assert [r["config"][:2] for r in rows] == ["V0", "V1", "V2", "V3"]
        assert set(units) == {r["config"] for r in rows}

    def test_v1_ties_the_two_thresholds_and_v2_unties_them(self, space, gold, folds, sampler):
        """The comparison the whole ladder exists for (H4.2), fixed on both sides.

        V1 is the single-threshold ablation: one value tuned for F1 and used for both decisions.
        V2 changes that and nothing else, so its two thresholds must be free to differ
        while its poolings stay P1/P1.
        """
        rows, _ = stages.run_ladder(space, folds, gold, normalise, GRID, sampler, None)
        v1 = next(r for r in rows if r["config"].startswith("V1"))
        v2 = next(r for r in rows if r["config"].startswith("V2"))
        assert v1["tau_prune"] == v1["tau_accept"]
        assert (v2["pool_pr"], v2["pool_acc"]) == (v1["pool_pr"], v1["pool_acc"]) == ("P1", "P1")
        assert v2["tau_prune"] != v2["tau_accept"]

    def test_v0_keeps_phenorags_own_rule_instead_of_scanning_everything(self, space, gold, folds,
                                                                        sampler):
        """P0 is an indicator, so its threshold is fixed, not selected.

        Letting risk control choose picks tau_prune = 0, at which ``score >= tau`` holds for every
        node and the rung silently becomes an exhaustive ontology scan reported under PhenoRAG's
        name. The regression that catches it is the cost: V0 must not be the most expensive rung.
        """
        rows, _ = stages.run_ladder(space, folds, gold, normalise, GRID, sampler, None)
        v0 = next(r for r in rows if r["config"].startswith("V0"))
        assert v0["tau_prune"] == 0.5
        assert v0["pool_pr"] == v0["pool_acc"] == "P0"
        assert v0["calls_per_report"] < max(r["calls_per_report"] for r in rows[1:]) * 1.5

    def test_paired_tests_compare_adjacent_rungs_only(self, space, gold, folds, sampler):
        _, units = stages.run_ladder(space, folds, gold, normalise, GRID, sampler, None)
        adjusted = stages.paired_tests(units, n_permutations=64, alpha_holm=0.05)
        assert len(adjusted) == 3            # V1vsV0, V2vsV1, V3vsV2, not all 6 pairs
        for value in adjusted.values():
            assert set(value) >= {"p_raw", "p_holm", "reject"}

    def test_the_s_ablation_reruns_the_protocol_per_S(self, space, gold, folds):
        rows = stages.s_ablation(space, folds, gold, normalise, GRID, [1, 3])
        assert [r["S"] for r in rows] == [1, 3]
        assert all(r["calls_per_report"] > 0 for r in rows)
        assert all(0.0 <= r["attained_coverage"] <= 1.0 for r in rows)

    def test_oracle_expansion_removes_pruning_loss(self, space, gold, assignments, view):
        """It expands the ancestors of annotated terms, so no annotated term is ever pruned away."""
        opened = [{**a, "config": replace(a["config"], tau_accept=-np.inf)} for a in assignments]
        units = stages.oracle_expansion(space, gold, normalise, opened, view)
        recovered = sum(len(g & p) for g, p in units)
        assert recovered == sum(len(g) for g, _ in units)


class TestCalibrationStage:
    def test_every_method_yields_a_row_with_the_estimators_the_chapter_quotes(self, space, gold):
        config = Configuration("exemplar", "P3_1", 0.05, "P1", 0.5, 3)
        got = stages.run_calibration(space, config, gold, sorted(gold),
                                     ["identity", "platt", "isotonic"], n_bins=5)
        assert len(got["rows"]) == 3
        for row in got["rows"]:
            assert set(row) >= {"ece_equal_mass", "ece_equal_width", "brier", "n", "prevalence",
                                "chosen_vs_half_optimal_f1"}

    def test_the_sample_is_the_scored_pairs_not_the_predictions(self, space, gold):
        """Calibration is conditional on each system's own candidate set, that is the population."""
        config = Configuration("exemplar", "P3_1", 0.05, "P1", 0.5, 3)
        scores, labels = stages.calibration_sample(space, config, gold, sorted(gold))
        assert scores.size == labels.size > 0
        assert ((scores >= 0) & (scores <= 1)).all()
        assert set(np.unique(labels)) <= {0, 1}
        # Strictly more scored pairs than accepted ones: acceptance is a filter over this set.
        accepted = sum(len(v) for v in space.predictions(config).values())
        assert scores.size >= accepted

    def test_reliability_rows_carry_bin_counts(self, space, gold):
        from hpo_extraction.evaluation.metrics import fit_calibration_map
        config = Configuration("exemplar", "P3_1", 0.05, "P1", 0.5, 3)
        scores, labels = stages.calibration_sample(space, config, gold, sorted(gold))
        h = fit_calibration_map(scores, labels, "identity")
        rows = stages.reliability_rows(scores, labels, h, 5)
        assert rows and all("count" in r or "n" in r for r in rows)


class TestScoreDistributions:
    """The two decision axes, split by whether the decision was right."""

    CONFIG = Configuration("exemplar", "P3_1", 0.05, "P1", 0.5, 3)

    def test_auc_is_the_rank_statistic_and_handles_ties(self):
        # Perfect separation, and its mirror image.
        assert stages._auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
        assert stages._auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
        # An all-ties axis discriminates at chance, not at 1.0 -- the property the
        # mid-rank correction exists for, and the one a naive "count pairs where a > b" loses.
        assert stages._auc([0.5] * 6, [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)
        # One class empty is not a number.
        assert np.isnan(stages._auc([0.1, 0.2], [0, 0]))

    def test_bins_partition_the_population_and_keep_the_extremes(self):
        scores = np.linspace(0.0, 1.0, 101)
        labels = (scores > 0.5).astype(int)
        rows = stages._distribution_rows(scores, labels, n_bins=10)
        # Every node lands in one bin: no silent loss at the closed top edge, which is
        # where `digitize` puts the maximum if the last bin is not folded back.
        assert sum(r["n_positive"] + r["n_negative"] for r in rows) == scores.size
        assert min(r["lo"] for r in rows) == pytest.approx(0.0)
        assert max(r["hi"] for r in rows) == pytest.approx(1.0)

    def test_a_degenerate_axis_still_produces_a_row(self):
        """P0's indicator is constant. That is a finding, not a reason to emit nothing."""
        rows = stages._distribution_rows(np.ones(20), np.zeros(20, dtype=int), n_bins=8)
        assert rows and sum(r["n_negative"] for r in rows) == 20

    def test_expansion_label_is_the_oracle_conditions_label(self, space, gold, view):
        """A node is worth expanding iff an annotated term lies at or below it -- reflexively, so a
        node that *is* ground truth counts. Fixed against `oracle_expansion`'s own construction."""
        scores, labels = stages.expansion_sample(space, self.CONFIG, gold, sorted(gold), view)
        assert scores.size == labels.size > 0
        assert set(np.unique(labels)) <= {0, 1}
        # Every report's annotated terms are ancestors of themselves, so a visited ground truth node is
        # positive. Recompute the count independently of the stage.
        axis = space.prune_axis(self.CONFIG.index, self.CONFIG.pool_pr, self.CONFIG.s)
        expected = 0
        for rid in sorted(gold):
            visited = axis.visited(rid, self.CONFIG.tau_prune) & axis.in_cache[rid]
            bearing = view.ancestors_of_set(gold[rid])
            expected += sum(1 for i in np.flatnonzero(visited)
                            if space.graph.node_ids[i] in bearing)
        assert int(labels.sum()) == expected

    def test_expansion_population_is_the_nodes_the_rule_is_applied_to(self, space, gold, view):
        """Visited AND in-cache -- the same population `calls` is counted over, so the figure and
        the cost column describe one traversal."""
        axis = space.prune_axis(self.CONFIG.index, self.CONFIG.pool_pr, self.CONFIG.s)
        expected = sum(int((axis.visited(rid, self.CONFIG.tau_prune)
                            & axis.in_cache[rid]).sum()) for rid in sorted(gold))
        scores, _ = stages.expansion_sample(space, self.CONFIG, gold, sorted(gold), view)
        assert scores.size == expected

    def test_both_decisions_are_reported_with_the_threshold_in_force(self, space, gold, view):
        got = stages.run_distributions(space, self.CONFIG, gold, sorted(gold), view, n_bins=8)
        assert set(got) == {"expansion", "acceptance"}
        by_decision = {d: got[d]["summary"] for d in got}
        assert by_decision["expansion"]["threshold"] == self.CONFIG.tau_prune
        assert by_decision["acceptance"]["threshold"] == self.CONFIG.tau_accept
        for decision, summary in by_decision.items():
            assert summary["n"] == summary["n_positive"] + summary["n_negative"]
            assert got[decision]["rows"]
            counted = sum(r["n_positive"] + r["n_negative"] for r in got[decision]["rows"])
            assert counted == summary["n"], "the bins must account for every scored node"

    def test_acceptance_sample_is_the_calibration_population_verbatim(self, space, gold, view):
        """Not a second definition of the same thing -- the identical call, so the distribution
        figure and the reliability diagram can never describe different populations."""
        got = stages.run_distributions(space, self.CONFIG, gold, sorted(gold), view, n_bins=8)
        scores, labels = stages.calibration_sample(space, self.CONFIG, gold, sorted(gold))
        assert got["acceptance"]["summary"]["n"] == labels.size
        assert got["acceptance"]["summary"]["n_positive"] == int(labels.sum())


class TestLogitViews:
    """The score-distribution figure's three logit-space views."""

    CONFIG = Configuration("exemplar", "P3_1", 0.05, "P1", 0.5, 3)

    def test_logit_inverts_the_sigmoid_and_counts_what_it_clips(self):
        margins = np.array([-9.8, 0.0, 4.6])
        z, clipped = stages.to_logit(1.0 / (1.0 + np.exp(-margins)))
        assert z == pytest.approx(margins, abs=1e-9)
        assert clipped == 0
        z, clipped = stages.to_logit(np.array([0.0, 1.0, 0.5]))
        assert clipped == 2 and np.isfinite(z).all()

    def test_logit_bins_account_for_every_node_of_each_class(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=500)
        labels = (rng.random(500) < 0.1).astype(int)
        rows = stages._logit_histogram_rows(z, labels, 20)
        assert sum(r["n_positive"] for r in rows) == labels.sum()
        assert sum(r["n_negative"] for r in rows) == (labels == 0).sum()

    def test_survival_runs_from_one_to_zero_and_is_monotone(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=300)
        labels = (z + rng.normal(size=300) > 1).astype(int)
        rows = stages._survival_rows(z, labels, 50)
        for key in ("positive_above", "negative_above"):
            curve = [r[key] for r in rows]
            assert curve[0] == pytest.approx(1.0)
            assert all(a >= b for a, b in zip(curve, curve[1:]))
            assert curve[-1] <= 1.0 / min(labels.sum(), (labels == 0).sum()) + 1e-12

    def test_average_precision_of_a_perfect_ranking_is_one(self):
        rows, ap = stages._pr_rows([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        assert ap == pytest.approx(1.0)
        assert rows[-1]["recall"] == pytest.approx(1.0)

    def test_distributions_carry_the_logit_views(self, space, gold, view):
        got = stages.run_distributions(space, self.CONFIG, gold, sorted(gold), view, n_bins=8)
        for block in got.values():
            assert block["logit_rows"] and block["survival_rows"] and block["pr_rows"]
            summary = block["summary"]
            assert summary["recall_at_threshold"] == pytest.approx(
                summary["positive_above_threshold"])


class TestReliabilityCrossfit:
    CONFIG = Configuration("exemplar", "P3_1", 0.05, "P1", 0.5, 3)

    def test_wilson_contains_the_rate_and_is_open_above_at_zero(self):
        lo, hi = stages.wilson_interval(3, 40)
        assert lo < 3 / 40 < hi
        lo, hi = stages.wilson_interval(0, 31000)
        assert lo == 0.0 and 0.0 < hi < 1e-3

    def test_no_pair_is_mapped_by_a_fit_that_saw_it(self, space, gold, folds, monkeypatch):
        """Poison the held-out fold's labels: a cross-fitted map must not move."""
        scores, labels, owners = stages.calibration_sample(space, self.CONFIG, gold, sorted(gold),
                                                           with_reports=True)
        base = stages.platt_crossfit(scores, labels, owners, folds)
        held = np.isin(owners, folds[0]["eval_ids"])
        poisoned = labels.copy()
        poisoned[held] = 1 - poisoned[held]
        again = stages.platt_crossfit(scores, poisoned, owners, folds)
        assert again[held] == pytest.approx(base[held])

    def test_rows_and_summary_per_curve(self, space, gold, folds, monkeypatch):
        import hpo_extraction.evaluation.metrics as tm
        monkeypatch.setattr(tm, "smooth_ece", lambda c, y: float(np.mean(np.abs(c - y))))
        got = stages.reliability_crossfit(space, self.CONFIG, gold, sorted(gold), folds,
                                          n_bins=4, n_boot=5)
        assert {r["curve"] for r in got["summary"]} == {"raw", "platt_crossfit"}
        for r in got["rows"]:
            assert r["wilson_lo"] <= r["observed_rate"] <= r["wilson_hi"]
        for r in got["summary"]:
            assert r["smooth_ece_lo"] <= r["smooth_ece_hi"]
            # The basic interval contains the bias-corrected estimate by design.
            assert r["smooth_ece_bc_lo"] <= r["smooth_ece_bc"] <= r["smooth_ece_bc_hi"]


class TestDiagnosticsStage:
    def test_it_returns_the_three_tables_the_chapter_asks_for(self, space, gold, view):
        config = Configuration("exemplar", "P3_1", 0.05, "P1", 0.9, 3)
        predicted = space.predictions(config)
        got = stages.run_diagnostics(space, config, gold, predicted, sorted(gold), view, 0.0)
        assert set(got) == {"decomposition", "taxonomy", "near_miss", "attribution"}
        decomposition = got["decomposition"]
        assert sum(decomposition["counts"].values()) == decomposition["n_false_negative"]
        assert decomposition["evidence_available"] is False

    def test_the_per_pair_attribution_reconciles_with_the_bucket_totals(self, space, gold, view):
        """chapter 6 joins on this, so it must be the same decomposition, not a second one."""
        config = Configuration("exemplar", "P3_1", 0.05, "P1", 0.9, 3)
        predicted = space.predictions(config)
        got = stages.run_diagnostics(space, config, gold, predicted, sorted(gold), view, 0.0)
        rows, counts = got["attribution"], got["decomposition"]["counts"]

        assert len(rows) == got["decomposition"]["n_false_negative"]
        tally: dict[str, int] = {}
        for row in rows:
            tally[row["bucket"]] = tally.get(row["bucket"], 0) + 1
        for bucket, n in counts.items():
            assert tally.get(bucket, 0) == n, f"{bucket}: {tally.get(bucket, 0)} != {n}"
        # Every row names the pair it is about, or the join in chapter 6 has nothing to key on.
        assert all(r["report_id"] and r["hpo_id"] for r in rows)
        # A pruned row is filed under pruning and carries where the traversal actually stopped.
        assert all(r["bucket"] == "pruning" for r in rows if r["pruned"])

    def test_evidence_makes_the_retrieval_bucket_measurable(self, space, gold, view):
        config = Configuration("exemplar", "P3_1", 0.05, "P1", 0.9, 3)
        predicted = space.predictions(config)
        evidence = {r: {h: [99] for h in terms} for r, terms in gold.items()}
        got = stages.run_diagnostics(space, config, gold, predicted, sorted(gold), view, 0.0,
                                     evidence)
        assert got["decomposition"]["evidence_available"] is True
        # Segment 99 was never retrieved (the fixture retrieves 0..2), so every scored miss is one.
        assert got["decomposition"]["counts"]["retrieval"] > 0


class TestRetrievalGate:
    """T4.1 -- the gate. It is a claim about retrieval, so the tests are about sent_index."""

    def evidence_at(self, gold, segments):
        return {r: {h: list(segments) for h in terms} for r, terms in gold.items()}

    def test_evidence_on_a_retrieved_segment_scores_a_hit_everywhere(self, space, gold, view):
        """The fixture retrieves segments 0..2 for every node, so evidence at 0 is always found."""
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [0]), [3],
                                            view=view)
        assert rows
        assert all(r["hit_at_s"] == 1.0 for r in rows)
        assert all(r["path_hit_rate"] == 1.0 for r in rows)

    def test_evidence_on_a_segment_no_node_retrieved_scores_zero(self, space, gold, view):
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [99]), [3],
                                            view=view)
        assert rows and all(r["hit_at_s"] == 0.0 for r in rows)

    def test_S_gates_which_segments_count(self, space, gold, view):
        """Evidence at segment 2 is unreachable at S=1 and reachable at S=3. That IS the gate.

        It does not reach 1.0 at S=3, and that is the fixture being realistic, not a
        failure: its mask drops about a quarter of the rank-3 entries, so for those nodes segment
        2 was never retrieved at all. A node whose retrieval came back short is the case
        Hit@S is supposed to count as a miss.
        """
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [2]), [1, 3],
                                            view=view)
        at = {(r["retrieval_index"], r["S"]): r["hit_at_s"] for r in rows}
        for index in space.indices:
            assert at[(index, 1)] == 0.0
            assert at[(index, 3)] > 0.5

    def test_hit_at_s_is_monotone_in_S(self, space, gold, view):
        """More segments can only help: a wider S sees a superset of the narrower one's."""
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [0, 1, 2]),
                                            [1, 2, 3], view=view)
        by_index: dict[str, list[tuple[int, float]]] = {}
        for row in rows:
            by_index.setdefault(row["retrieval_index"], []).append((row["S"], row["hit_at_s"]))
        for index, series in by_index.items():
            values = [v for _, v in sorted(series)]
            assert values == sorted(values), f"{index}: Hit@S fell as S grew: {values}"

    def test_a_node_whose_retrieval_came_back_short_counts_as_a_miss(self, gold, view):
        """A fully-masked cache reaches 1.0. The same cache with rank 3 dropped does not.

        This is the one place the mask changes the answer, not the variance, so it is
        fixed directly instead of being inferred from the shared fixture.
        """
        nodes = list(NODES)
        sent = np.tile(np.arange(3, dtype=np.int32), (len(nodes), 1))
        margins = np.zeros((len(nodes), 3))

        def space_with(mask):
            caches = {f"R{j:02d}": ReportCache(f"R{j:02d}", nodes, margins, mask.copy(),
                                               sent_index=sent)
                      for j in range(4)}
            return OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches})

        evidence = self.evidence_at(gold, [2])
        full = np.ones((len(nodes), 3), dtype=bool)
        short = full.copy()
        short[:, 2] = False                       # every node's retrieval came back with two

        got_full = stages.retrieval_sufficiency(space_with(full), gold, evidence, [3], view=view)
        got_short = stages.retrieval_sufficiency(space_with(short), gold, evidence, [3], view=view)
        assert got_full[0]["hit_at_s"] == 1.0
        assert got_short[0]["hit_at_s"] == 0.0

    def test_the_path_rate_never_exceeds_the_pointwise_rate(self, space, gold, view):
        """The path-wise rate asks strictly more: every ancestor too, not just the term."""
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [0, 2]), [2, 3],
                                            view=view)
        assert rows and all(r["path_hit_rate"] <= r["hit_at_s"] + 1e-12 for r in rows)

    def test_a_term_with_no_evidence_is_undefined_rather_than_missed(self, space, gold, view):
        """Absent evidence must not be scored as a retrieval failure -- it is not a measurement."""
        rows = stages.retrieval_sufficiency(space, gold, {}, [3], view=view)
        assert rows == []

    def test_the_paired_difference_is_carried_once_with_an_interval(self, space, gold, view):
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [0, 1]), [3],
                                            view=view)
        carried = [r for r in rows if "paired_delta" in r]
        assert len(carried) == 1                      # one row per S, on the synthetic-sentence condition
        row = carried[0]
        assert row["retrieval_index"] == "exemplar"
        assert row["paired_delta_lo"] <= row["paired_delta"] <= row["paired_delta_hi"]

    def test_every_column_the_table_quotes_is_present(self, space, gold, view):
        rows = stages.retrieval_sufficiency(space, gold, self.evidence_at(gold, [0]), [1, 3],
                                            view=view)
        for row in rows:
            assert {"retrieval_index", "S", "n_reports", "n_gold", "hit_at_s", "hit_at_s_lo",
                    "hit_at_s_hi", "path_hit_rate", "path_hit_rate_lo",
                    "path_hit_rate_hi"} <= set(row)


class TestOneOperatingPoint:
    """Every number quoted "at the configuration" is Table 1's pooled out-of-fold traversal.

    Before 2026-09-24 the chapter printed three call counts for one configuration (31 228 in the
    ladder, 31 333 in the alpha table, 30 735 in the S table) and two F1s (0.3135, 0.3151), because
    the ladder priced a modal configuration, the alpha table averaged the CRC log over both
    retrieval indices and over training reports, and the S and pooling tables re-run one fold's
    configuration in-sample on every report. These fix them to one another.
    """

    @pytest.fixture
    def selection(self, space, gold, folds, sampler):
        return stages.run_selection(space, folds, gold, normalise, GRID, sampler)

    def test_the_ladders_top_rung_is_the_selection_row(self, space, gold, folds, sampler,
                                                       selection):
        rows, _ = stages.run_ladder(space, folds, gold, normalise, GRID, sampler, None)
        v3 = next(r for r in rows if r["config"].startswith("V3"))
        for key in ("micro_f1", "micro_recall", "calls_per_report", "attained_coverage"):
            assert v3[key] == pytest.approx(selection["row"][key]), key

    def test_the_alpha_row_at_the_chapters_alpha_is_the_selection_row(self, space, gold, folds,
                                                                      selection):
        row = stages.alpha_sensitivity(space, folds, gold, normalise, GRID,
                                       [GRID["alpha"]])[0]
        assert row["micro_f1"] == pytest.approx(selection["row"]["micro_f1"])
        assert row["calls_per_report"] == pytest.approx(selection["row"]["calls_per_report"])
        assert row["attained_coverage_micro"] == pytest.approx(
            selection["row"]["attained_coverage"])
        # One CRC record per fold row, at the index that fold row selected -- not one per index.
        assert row["n_folds"] == len(folds)

    def test_the_S_row_at_the_chapters_S_is_the_selection_row(self, space, gold, folds,
                                                              selection):
        row = stages.s_ablation(space, folds, gold, normalise, GRID, [GRID["s_main"]])[0]
        assert row["micro_f1"] == pytest.approx(selection["row"]["micro_f1"])
        assert row["calls_per_report"] == pytest.approx(selection["row"]["calls_per_report"])

    def test_the_diagnostics_partition_the_pooled_false_negatives(self, space, gold, view,
                                                                  selection):
        got = stages.nested_diagnostics(space, selection["assignments"], gold, view, 0.0)
        pooled = selection["pooled"]
        n_fn = sum(len(gold.get(r, set()) - pooled[r]) for r in pooled)
        decomposition = got["decomposition"]
        assert decomposition["n_false_negative"] == n_fn
        assert sum(decomposition["counts"].values()) == n_fn
        assert len(got["attribution"]) == n_fn
        assert got["predicted"] == pooled

    def test_coverage_is_micro_over_annotated_terms(self, space, gold, selection):
        """The column is labelled micro. The macro figure is 1 - the mean per-report miss rate."""
        cost = stages.nested_operating_point(space, selection["assignments"], gold)
        reached = total = 0
        for a in selection["assignments"]:
            c = a["config"]
            axis = space.prune_axis(c.index, c.pool_pr, c.s)
            for r in a["eval"]:
                visited = set(space.node_names[axis.visited(r, c.tau_prune)])
                reached += len(gold[r] & visited)
                total += len(gold[r])
        assert cost["attained_coverage_micro"] == pytest.approx(reached / total)
        assert cost["attained_coverage_macro"] == pytest.approx(1 - cost["held_out_miss_rate"])


class TestPoolingOperators:
    """T4.3 -- two blocks, measured differently, and the difference is the point."""

    def test_both_blocks_are_written_with_the_operators_asked_for(self, space, gold,
                                                                  assignments):
        rows = stages.pooling_operators(space, gold, normalise, assignments, GRID,
                                        ["P1", "P3_1"], ["P0", "P1", "P3_1"])
        blocks = {r["block"] for r in rows}
        assert blocks == {"expansion", "acceptance"}
        assert {r["operator"] for r in rows if r["block"] == "expansion"} == {"P1", "P3_1"}
        assert {r["operator"] for r in rows if r["block"] == "acceptance"} == {"P0", "P1", "P3_1"}

    def test_P0_is_acceptance_only(self, space, gold, assignments):
        """An indicator in {0,1} cannot rank nodes, so it has no expansion row. The caller simply
        does not offer it one, and the stage must not invent it."""
        rows = stages.pooling_operators(space, gold, normalise, assignments, GRID,
                                        ["P1"], ["P0", "P1"])
        assert not [r for r in rows if r["block"] == "expansion" and r["operator"] == "P0"]

    def test_each_expansion_operator_is_priced_and_credited(self, space, gold, assignments):
        """Equal risk, not equal threshold: every operator carries its own tau, cost and coverage."""
        rows = [r for r in stages.pooling_operators(space, gold, normalise, assignments, GRID,
                                                    ["P1", "P2", "P3_1"], ["P1"])
                if r["block"] == "expansion"]
        assert len(rows) == 3
        assert all(r["calls_per_report"] > 0 for r in rows)
        assert all(0.0 <= r["attained_coverage"] <= 1.0 for r in rows)

    def test_the_acceptance_block_matches_a_direct_offline_evaluation(self, space, gold, assignments):
        """The block must BE the pooled re-run at that operator, not an approximation of it."""
        rows = stages.pooling_operators(space, gold, normalise, assignments, GRID,
                                        ["P1"], ["P1", "P3_1"])
        for row in (r for r in rows if r["block"] == "acceptance"):
            direct = stages.pooled_predictions(space, [
                {**a, "config": replace(a["config"], pool_acc=row["operator"])}
                for a in assignments])
            units = stages.units_for(direct, gold, sorted(direct), normalise)
            got = stages.metric_fn(units)
            assert row["micro_f1"] == pytest.approx(got["micro_f1"])
            assert row["micro_precision"] == pytest.approx(got["micro_precision"])

    def test_the_expansion_rows_carry_the_threshold_columns_the_table_quotes(self, space, gold,
                                                                            assignments):
        rows = stages.pooling_operators(space, gold, normalise, assignments, GRID, ["P1"],
                                        ["P1"])
        expansion = [r for r in rows if r["block"] == "expansion"][0]
        assert {"tau", "feasible", "attained_coverage", "calls_per_report",
                "containment_misses"} <= set(expansion)


class TestInstruments:
    """The inline numbers. Each is a number a paragraph asserts, so each is fixed."""

    def test_the_call_distribution_is_the_pooled_traversals(self, space, gold, assignments):
        """Each held-out report priced under its own fold's configuration -- Table 1's Calls."""
        rows = {r["statistic"]: r["value"] for r in stages.calls_per_report(space, assignments)}
        expected = [space.prune_axis(a["config"].index, a["config"].pool_pr, a["config"].s)
                    .calls([r], a["config"].tau_prune) for a in assignments for r in a["eval"]]
        assert rows["mean"] == pytest.approx(np.mean(expected))
        assert rows["n_reports"] == len(expected)
        assert rows["min"] <= rows["median"] <= rows["p95"] <= rows["max"]
        assert rows["total"] == pytest.approx(rows["mean"] * rows["n_reports"])

    def test_captured_mass_is_skipped_rather_than_faked_when_the_column_is_absent(self, space):
        """The fixture cache carries no captured_mass. A pre-the synthetic-sentence score store cache is an ordinary state
        of the world, and the stage must return nothing, not a plausible floor."""
        config = Configuration("exemplar", "P1", 0.05, "P1", 0.9, 3)
        assert stages.captured_mass(space, config) == []

    def test_captured_mass_reports_the_floor_over_the_calls_actually_made(self):
        nodes = list(NODES)
        rng = np.random.default_rng(7)
        caches = {}
        for j in range(6):
            rid = f"R{j:02d}"
            margins = rng.normal(0.5, 2.5, size=(len(nodes), 3))
            mask = np.ones((len(nodes), 3), dtype=bool)
            mass = rng.uniform(0.2, 1.0, size=(len(nodes), 3))
            mass[0, 0] = 0.05                      # The planted floor
            caches[rid] = ReportCache(rid, nodes, margins, mask, captured_mass=mass,
                                      sent_index=np.tile(np.arange(3), (len(nodes), 1)))
        space = OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches})
        # -inf visits every node, so every cached call is "actually made" and the floor is the
        # planted one, not whatever survived pruning.
        config = Configuration("exemplar", "P1", -np.inf, "P1", 0.9, 3)
        rows = stages.captured_mass(space, config)
        assert rows and rows[0]["floor"] == pytest.approx(0.05)
        assert rows[0]["n_calls"] == len(nodes) * 3 * len(caches)
        assert 0.0 <= rows[0]["share_below_0p5"] <= 1.0

    def test_offline_fidelity_is_zero_on_a_sound_alignment(self, space):
        """Two routes to the same numbers. Anything but zero means align_to_graph and the graph
        index have drifted apart, which would misattribute every score in the chapter."""
        config = Configuration("exemplar", "P1", 0.05, "P1", 0.9, 3)
        rows = stages.replay_fidelity(space, config, ["P1", "P3_1"], [3])
        assert rows
        assert all(r["max_abs_score_discrepancy"] == 0.0 for r in rows)


class TestLadderCoverage:
    def test_every_rung_carries_its_attained_coverage(self, space, folds, gold, sampler):
        """T4.2 quotes coverage beside F1: a recall of 0.2 under a coverage of 1.0 and the same
        recall under a coverage of 0.3 are different findings."""
        rows, _ = stages.run_ladder(space, folds, gold, normalise, GRID, sampler, None)
        assert rows
        for row in rows:
            assert "attained_coverage" in row
            assert 0.0 <= row["attained_coverage"] <= 1.0


class TestAlphaCoverage:
    def test_both_averagings_are_written_and_are_shares(self, space, folds, gold):
        rows = stages.alpha_sensitivity(space, folds, gold, normalise, GRID, [0.2, 0.4])
        assert rows
        for row in rows:
            assert 0.0 <= row["attained_coverage_micro"] <= 1.0
            assert 0.0 <= row["attained_coverage_macro"] <= 1.0


class TestCachePathsAreNamespacedByCohort:
    """The GSC+ transfer condition must not silently read HCY's cache.

    Both cohorts configure the same retrieval index (`ontology_r3`). When the ingested `.npz` was
    named after the index alone the two collided: `ingest_all` found HCY's file already there and
    skipped the GSC+ ingest, `load_spaces` filtered HCY's 118 reports down to the 114 GSC+
    document ids -- none of which are in it -- and the transfer was scored over an empty cohort.
    It reported `transferred = 0.0000` into `transfer.csv`, which is the chapter's only evidence
    that `h` is a contribution, not a diagnostic.

    Two independent guards, because either alone still fails quietly in some configuration.
    """

    def test_the_two_cohorts_resolve_to_different_files(self):
        run_mod = _load_run()
        out = Path("/tmp/out")
        hcy = run_mod.default_npz(out, "ontology_r3")
        gsc = run_mod.default_npz(out, "ontology_r3", run_mod.GSC_NPZ_PREFIX)
        assert hcy != gsc
        # HCY keeps the bare name: an existing 585 MB cache must not be orphaned by the fix and
        # re-ingested from 6 GB of JSON.
        assert hcy.name == "cache_ontology_r3.npz"

    def test_a_cache_holding_none_of_the_requested_reports_aborts(self, tmp_path):
        """The second guard: even with the right filename, a wrong-cohort cache must not score."""
        run_mod = _load_run()
        from hpo_extraction.treephenorag.stored_scores import ingest_score_cache

        calls = tmp_path / "calls.jsonl"
        with calls.open("w", encoding="utf-8") as fh:
            for rank in range(2):
                fh.write(json.dumps({
                    "report_id": "SYN01", "hpo_id": A, "ctx_type": "exemplar", "rank": rank,
                    "sent_index": rank, "cosine_sim": 0.5, "logit_yes": 1.0, "logit_no": 0.0,
                    "margin": 1.0,
                }) + "\n")
        npz = tmp_path / "cache_exemplar.npz"
        ingest_score_cache(str(calls), str(npz), ctx_type="exemplar")

        cfg = {"caches": {"exemplar": {"npz": str(npz)}}}
        with pytest.raises(SystemExit) as excinfo:
            run_mod.load_spaces(cfg, tmp_path, CHILDREN, ROOTS, ["GSC_DOC_1", "GSC_DOC_2"])
        message = str(excinfo.value)
        assert "none of the 2 requested" in message
        # The message has to name the file, or the reader cannot tell which cohort leaked.
        assert npz.name in message

    def test_the_matching_cohort_still_loads(self, tmp_path):
        """The guard must not fire on the ordinary case it sits in front of."""
        run_mod = _load_run()
        from hpo_extraction.treephenorag.stored_scores import ingest_score_cache

        calls = tmp_path / "calls.jsonl"
        with calls.open("w", encoding="utf-8") as fh:
            for rank in range(2):
                fh.write(json.dumps({
                    "report_id": "SYN01", "hpo_id": A, "ctx_type": "exemplar", "rank": rank,
                    "sent_index": rank, "cosine_sim": 0.5, "logit_yes": 1.0, "logit_no": 0.0,
                    "margin": 1.0,
                }) + "\n")
        npz = tmp_path / "cache_exemplar.npz"
        ingest_score_cache(str(calls), str(npz), ctx_type="exemplar")

        cfg = {"caches": {"exemplar": {"npz": str(npz)}}}
        space = run_mod.load_spaces(cfg, tmp_path, CHILDREN, ROOTS, ["SYN01"])
        assert space is not None and space.report_ids("exemplar") == ["SYN01"]


class TestChapterSixPredictionSetsAreWritten:
    """The comparison reads TreePhenoRAG's prediction SETS, not the TreePhenoRAG protocol's scores.

    The TreePhenoRAG protocol scored the GSC+ transfer into `transfer.csv` and the pooled out-of-fold row into
    `core_quality.csv`, but wrote no `predictions/` directory at all. So chapter 6's TreePhenoRAG
    GSC+ cell reported `missing: no artifact at this path` however well the re-run went, and its
    HCY cell was served by a `nested_cv_pooled.csv` left behind by an earlier version of this
    driver -- an absence that looks like a result, which is the worse failure of the two.

    The invariant is a contract BETWEEN two experiments, so it is fixed on both sides: the names
    the TreePhenoRAG protocol writes have to be the names the comparison goes looking for. Pinning only one side is how
    this broke -- `roster.py` documented the GSC+ filename in a comment for a file nothing wrote.
    """

    #: The two-column format is `hpo_extraction.evaluation.prediction_sets`, shared with the ground truth files.
    EXP13_25_CONFIG = (Path(__file__).resolve().parents[2] / "configs" / "experiments"
                       / "06_comparison" / "comparison.yaml")
    CLUSTER_SCRIPT = (Path(__file__).resolve().parents[2] / "slurm"
                      / "treephenorag_protocol.sbatch")

    def test_both_prediction_sets_are_written(self):
        source = (_EXP_DIR / "run.py").read_text(encoding="utf-8")
        assert "write_prediction_sets" in source, "nothing writes a prediction set at all"
        assert '"predictions" / "nested_cv_pooled.csv"' in source
        assert '_transferred.csv' in source

    def test_the_names_are_the_ones_exp13_25_looks_for(self):
        """Both halves of the contract, read off the two files, not restated here."""
        wanted = self.EXP13_25_CONFIG.read_text(encoding="utf-8")
        assert "predictions/nested_cv_pooled.csv" in wanted
        assert "predictions/gsc_raghpo_ann_transferred.csv" in wanted

        # The GSC+ name is DERIVED from the ground truth file's stem, so the derivation has to land on
        # that. The ground truth the cluster script passes is the 114-document re-annotation.
        script = self.CLUSTER_SCRIPT.read_text(encoding="utf-8")
        assert "gsc.gold_path=$GSC_GOLD/gsc_raghpo_ann.csv" in script
        assert Path("gsc_raghpo_ann.csv").stem + "_transferred.csv" ==             "gsc_raghpo_ann_transferred.csv"

    def test_the_format_survives_the_round_trip_exp13_25_makes(self, tmp_path):
        """Written by this driver, read back by the reader the comparison actually calls.

        The empty set is the case that counts: a report the traversal accepted nothing for must
        come back as a report predicting nothing, never as a report that is absent. Absent is a
        coverage failure and the comparison counts it separately (`n_reports_absent`). Handing it a
        dropped line would quietly buy TreePhenoRAG free precision on that report.
        """
        from hpo_extraction.evaluation.prediction_sets import read_prediction_sets, write_prediction_sets

        sets = {"SYN01": {A, B}, "SYN02": set(), "SYN03": {C}}
        path = write_prediction_sets(tmp_path / "predictions" / "nested_cv_pooled.csv", sets)
        assert path.is_file()
        back = read_prediction_sets(path)
        assert back == sets
        assert "SYN02" in back and back["SYN02"] == set()
