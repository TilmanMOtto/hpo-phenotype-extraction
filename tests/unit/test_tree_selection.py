"""Selecting TreePhenoRAG's configuration without looking at the answer.

The properties fixed here are the ones that decide whether the chapter's numbers are honest:

* an evaluation report is **never** used to select anything that is then applied to it;
* the pooled prediction set covers every report once, from the first repetition only;
* conformal risk control runs on the exact breakpoint set, and reports the attained ceiling rather
  than a threshold that does not honour the bound;
* the expansion pooling is chosen by **cost**, not by F1;
* the memo that makes the search affordable returns the same numbers as recomputing.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpo_extraction.treephenorag.stored_scores import ReportCache
from hpo_extraction.treephenorag.selection import (
    Configuration,
    OfflineSpace,
    crc_candidates,
    nested_evaluation,
    select_acceptance,
    select_prune_pooling,
    select_tau_prune,
    selection_stability,
)
from hpo_extraction.evaluation.stats import nested_folds
from fixtures.toy_ontology import A, B, C, D, E, F, G, H, M, toy_children_map

pytestmark = pytest.mark.unit

NODES = sorted({A, B, C, D, E, F, G, H, M})
CHILDREN, ROOTS = toy_children_map()


def make_cache(rid, seed, s_max=3):
    rng = np.random.default_rng(seed)
    margins = rng.normal(0.5, 2.5, size=(len(NODES), s_max))
    mask = rng.random((len(NODES), s_max)) > 0.25
    mask[:, 0] = True
    sent = np.tile(np.arange(s_max, dtype=np.int32), (len(NODES), 1))
    return ReportCache(rid, list(NODES), margins, mask, sent_index=sent)


@pytest.fixture
def space():
    caches = {rid: make_cache(rid, i) for i, rid in enumerate(f"R{j:02d}" for j in range(20))}
    other = {rid: make_cache(rid, 100 + i)
             for i, rid in enumerate(f"R{j:02d}" for j in range(20))}
    return OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches, "ontology_r3": other})


@pytest.fixture
def gold():
    rng = np.random.default_rng(7)
    return {f"R{j:02d}": set(rng.choice(NODES, size=3, replace=False)) for j in range(20)}


def normalise(g, p):
    """A stand-in for the ontology reduction: identity, so the arithmetic stays checkable."""
    return set(g), set(p)


class TestPruneAxis:
    def test_visited_shrinks_as_the_threshold_rises(self, space):
        axis = space.prune_axis("exemplar", "P3_1", 3)
        previous = None
        for tau in (0.0, 0.2, 0.5, 0.8, 0.99):
            current = set(np.flatnonzero(axis.visited("R00", tau)))
            if previous is not None:
                assert current <= previous
            previous = current

    def test_calls_are_counted_on_segments_actually_scored(self, space):
        """A node retrieved on two of three segments cost two calls, not three."""
        axis = space.prune_axis("exemplar", "P3_1", 3)
        cache = space.caches_by_index["exemplar"]["R00"]
        # At a threshold that visits everything, the cost is the whole mask.
        assert axis.calls(["R00"], -np.inf) == pytest.approx(cache.mask[:, :3].sum())

    def test_an_exhaustive_cache_has_no_containment_misses(self, space):
        axis = space.prune_axis("exemplar", "P3_1", 3)
        assert axis.containment(space.report_ids("exemplar"), 0.0) == 0


class TestGoldMissRates:
    """The vectorised loss must equal the set algebra it replaced, cell for cell.

    This is the one change that could move every CRC number in chapter 4 without failing anything
    else, so it is fixed against `miss_rate` over real `scored_sets`, not against itself.
    """

    def test_it_agrees_with_miss_rate_over_scored_sets(self, space, gold):
        from hpo_extraction.evaluation.stats import miss_rate

        axis = space.prune_axis("exemplar", "P3_1", 3)
        ids = space.report_ids("exemplar")
        taus = crc_candidates(space, axis, gold, ids)
        assert taus.size > 3

        fast = space.gold_miss_rates(axis, gold, ids, taus)
        for t, tau in enumerate(taus):
            scored = space.scored_sets("exemplar", "P3_1", 3, float(tau))
            for i, rid in enumerate(ids):
                assert fast[i, t] == pytest.approx(
                    miss_rate(gold.get(rid, set()), scored[rid])), f"{rid} at tau={tau}"

    def test_a_report_with_no_gold_contributes_nothing(self, space, gold):
        axis = space.prune_axis("exemplar", "P3_1", 3)
        ids = space.report_ids("exemplar")
        empty = dict(gold)
        empty[ids[0]] = set()
        out = space.gold_miss_rates(axis, empty, ids, np.array([0.0, 0.5, 1.0]))
        assert np.all(out[0] == 0.0)

    def test_gold_outside_the_graph_is_missed_at_every_threshold(self, space, gold):
        axis = space.prune_axis("exemplar", "P3_1", 3)
        ids = space.report_ids("exemplar")
        off = dict(gold)
        off[ids[0]] = {"HP:9999999"}
        out = space.gold_miss_rates(axis, off, ids, np.array([-np.inf, 0.0, 1.0]))
        assert np.all(out[0] == 1.0)


class TestConformalRiskControl:
    def test_the_candidate_grid_is_the_annotated_terms_own_breakpoints(self, space, gold):
        axis = space.prune_axis("exemplar", "P3_1", 3)
        ids = space.report_ids("exemplar")
        grid = crc_candidates(space, axis, gold, ids)
        assert grid.ndim == 1 and grid.size > 1
        assert np.all(np.diff(grid) > 0), "must be sorted and deduplicated"
        # Every breakpoint is some annotated term's own r value, and nothing else is.
        allowed = set()
        for rid in ids:
            for t in gold[rid]:
                v = axis.r[rid][space.graph.index[t]]
                if np.isfinite(v):
                    allowed.add(round(float(v), 12))
        assert {round(float(t), 12) for t in grid} <= allowed

    def test_the_grid_never_goes_below_the_cache_floor(self, gold):
        """A threshold under the tau the cache was built at is not a configuration the stored scores can evaluate.

        The producer pruned below its own floor, so the nodes a lower threshold would visit were
        never scored. Annotated terms whose bottleneck sits down there are unreachable at every
        admissible threshold. The grid must therefore start at the floor, and the resulting
        irreducible miss rate must show up as risk, not be bought back with a threshold
        this cache cannot host.
        """
        caches = {rid: make_cache(rid, i) for i, rid in enumerate(f"R{j:02d}" for j in range(20))}
        floor = 0.05
        space = OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches}, tau_floor=floor)
        axis = space.prune_axis("exemplar", "P3_1", 3)
        ids = space.report_ids("exemplar")

        unclipped = crc_candidates(
            OfflineSpace(CHILDREN, ROOTS, {"exemplar": caches}), axis, gold, ids)
        grid = crc_candidates(space, axis, gold, ids)

        assert (unclipped < floor).any(), "fixture must exercise the clip"
        assert grid.min() == pytest.approx(floor)
        assert (grid >= floor).all()
        # Nothing at or above the floor is silently dropped along the way.
        assert set(np.round(unclipped[unclipped >= floor], 12)) <= set(np.round(grid, 12))

    def test_the_bound_is_honoured_on_the_calibration_split_when_feasible(self, space, gold):
        axis = space.prune_axis("exemplar", "P3_1", 3)
        ids = space.report_ids("exemplar")
        got = select_tau_prune(space, axis, gold, ids, alpha=0.5)
        if got["feasible"]:
            assert got["bound"] <= 0.5 + 1e-12

    def test_an_unreachable_alpha_reports_the_ceiling_rather_than_a_threshold(self, space, gold):
        """alpha below the traversal's own coverage ceiling cannot be honoured at any threshold.

        The bound carries a finite-sample slack of 1/(n+1), so on 20 calibration reports nothing
        below ~0.048 is reachable even at a threshold that misses no annotated term at all.
        """
        axis = space.prune_axis("exemplar", "P3_1", 3)
        got = select_tau_prune(space, axis, gold, space.report_ids("exemplar"), alpha=1e-6)
        assert got["feasible"] is False
        assert "attained_ceiling" in got and got["tau_prune"] == got["attained_ceiling_tau"]


class TestPoolingSelection:
    def test_the_expansion_pooling_is_chosen_by_cost_not_by_f1(self, space, gold):
        """Every candidate is priced. The winner is the cheapest feasible one."""
        got = select_prune_pooling(space, "exemplar", ["P1", "P3_1", "P4"], 3, gold,
                                   space.report_ids("exemplar"), alpha=0.5)
        assert set(got["chosen"]) >= {"pooling", "tau_prune", "calls_per_report"}
        assert "micro_f1" not in got["chosen"]
        feasible = [r for r in got["candidates"] if r["feasible"]]
        if feasible:
            assert got["chosen"]["calls_per_report"] == pytest.approx(
                min(r["calls_per_report"] for r in feasible))

    def test_a_cell_that_left_the_cache_is_not_selectable(self, space, gold):
        """Containment is a gate on selection, not a footnote under the table."""
        got = select_prune_pooling(space, "exemplar", ["P3_1"], 3, gold,
                                   space.report_ids("exemplar"), alpha=0.5)
        assert got["chosen"]["containment_misses"] == 0


class TestAcceptanceSelection:
    def test_it_searches_both_retrieval_indices(self, space, gold):
        base = Configuration("exemplar", "P3_1", 0.1, "P1", 0.5, 3)
        folds = [space.report_ids("exemplar")[:10], space.report_ids("exemplar")[10:]]
        got = select_acceptance(space, base, folds, gold, normalise, ["P1", "P2"],
                                [0.3, 0.5, 0.9], indices=["exemplar", "ontology_r3"])
        seen = {(r["retrieval_index"], r["pool_acc"], r["tau_accept"])
                for r in got["candidates"]}
        assert len(seen) == 2 * 2 * 3
        assert got["config"].index in {"exemplar", "ontology_r3"}

    def test_each_index_keeps_its_own_expansion_configuration(self, space, gold):
        """A cache is re-run whole or not at all. Margins from two retrievals never mix."""
        base = Configuration("exemplar", "P3_1", 0.1, "P1", 0.5, 3)
        folds = [space.report_ids("exemplar")]
        got = select_acceptance(
            space, base, folds, gold, normalise, ["P1"], [0.5],
            indices=["exemplar", "ontology_r3"],
            prune_by_index={"exemplar": ("P3_1", 0.1), "ontology_r3": ("P1", 0.4)},
        )
        by_index = {r["retrieval_index"]: r for r in got["candidates"]}
        assert by_index["exemplar"]["pool_pr"] == "P3_1"
        assert by_index["ontology_r3"]["pool_pr"] == "P1"
        assert by_index["ontology_r3"]["tau_prune"] == 0.4


class TestScoringMemo:
    def test_the_memo_agrees_with_recomputing(self, space, gold):
        config = Configuration("exemplar", "P3_1", 0.1, "P1", 0.5, 3)
        ids = space.report_ids("exemplar")
        cached = space.micro_f1(config, ids, gold, normalise)

        predictions = space.predictions(config)
        tp = n_pred = n_gold = 0
        for rid in ids:
            g, p = normalise(gold[rid], predictions[rid])
            tp += len(g & p)
            n_pred += len(p)
            n_gold += len(g)
        precision, recall = tp / n_pred, tp / n_gold
        expected = 2 * precision * recall / (precision + recall) if tp else 0.0
        assert cached == pytest.approx(expected)

    def test_predictions_never_include_a_node_the_cache_lacks(self, space, gold):
        config = Configuration("exemplar", "P3_1", -np.inf, "P1", -np.inf, 3)
        cached_nodes = set(space.caches_by_index["exemplar"]["R00"].node_ids)
        assert space.predictions(config)["R00"] <= cached_nodes


class TestNestedEvaluation:
    def test_no_evaluation_report_selects_its_own_configuration(self, space, gold):
        """The property the whole protocol exists for, checked directly on the fold assignment."""
        folds = nested_folds(space.report_ids("exemplar"), k_outer=4, k_inner=3, repetitions=1,
                             seed=0)
        for row in folds:
            assert not (set(row["eval_ids"]) & set(row["train_ids"]))
            for inner in row["inner_folds"]:
                assert not (set(inner) & set(row["eval_ids"]))

    def test_pooled_predictions_cover_every_report_once(self, space, gold):
        ids = space.report_ids("exemplar")
        folds = nested_folds(ids, k_outer=4, k_inner=3, repetitions=2, seed=0)
        got = nested_evaluation(space, folds, gold, normalise, ["P3_1"], ["P1"], [0.5, 0.9],
                                s=3, alpha=0.5, indices=["exemplar"])
        assert set(got["pooled"]) == set(ids)
        # Two repetitions, four outer folds each.
        assert len(got["choices"]) == 8
        assert {c["repetition"] for c in got["choices"]} == {0, 1}

    def test_every_outer_fold_records_a_held_out_miss_rate(self, space, gold):
        folds = nested_folds(space.report_ids("exemplar"), k_outer=4, k_inner=3, repetitions=1,
                             seed=0)
        got = nested_evaluation(space, folds, gold, normalise, ["P3_1", "P1"], ["P1"], [0.5],
                                s=3, alpha=0.5, indices=["exemplar"])
        assert len(got["crc"]) == 4
        for row in got["crc"]:
            assert 0.0 <= row["held_out_miss_rate"] <= 1.0
            assert row["n_grid"] >= 1

    def test_stability_reports_the_modal_choice(self, space, gold):
        folds = nested_folds(space.report_ids("exemplar"), k_outer=4, k_inner=3, repetitions=3,
                             seed=0)
        got = nested_evaluation(space, folds, gold, normalise, ["P3_1"], ["P1", "P2"], [0.5, 0.9],
                                s=3, alpha=0.5, indices=["exemplar", "ontology_r3"])
        stability = selection_stability(got["choices"])
        assert set(stability) >= {"retrieval_index", "pool_acc", "tau_accept"}
        for field in stability.values():
            assert 0.0 < field["modal_share"] <= 1.0
            assert sum(field["counts"].values()) == len(got["choices"])
