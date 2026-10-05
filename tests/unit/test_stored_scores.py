"""Offline re-run of the TreePhenoRAG traversal from a cached score dump.

Three things have to hold before any number in the tree chapter is trustworthy:

1. **Fidelity**, re-running a run's own configuration reproduces that run's scores and prediction
   set. Fixed here against the driver-schema fixture. The cluster equivalent, against an earlier exploratory run's
   real ``_calls.jsonl``, is the gate the analysis stage runs before it writes a table.
2. **Agreement with the definitions**, the vectorised poolings equal the scalar reference
   implementations in ``hpo_extraction.treephenorag.pooling``, which are what ``an earlier exploratory run``'s leaderboard describes.
3. **Containment**, a configuration that leaves the cached region is *detected*, not silently
   scored as if the missing nodes were absent.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from hpo_extraction.treephenorag import pooling as pg
from hpo_extraction.treephenorag.stored_scores import (
    DOMINATING_POOLING,
    P0_MAX_CACHE_THRESHOLD,
    POOLINGS,
    ReportCache,
    align_to_graph,
    bottleneck_scores,
    build_traversal_graph,
    cache_supports,
    calls_at,
    containment_misses,
    containment_threshold,
    coverage_breakpoints,
    ingest_score_cache,
    load_cache_npz,
    read_cache_provenance,
    expanded_at,
    in_cache_mask,
    load_score_cache,
    replay,
    evaluate_offline_cohort,
    visited_at,
)


def make_cache(bags: list[list[float]], prefix: str = "n") -> ReportCache:
    """A :class:`ReportCache` over ragged margin bags, padded and masked."""
    s_max = max((len(b) for b in bags), default=0)
    margins = np.zeros((len(bags), s_max), dtype=np.float64)
    mask = np.zeros((len(bags), s_max), dtype=bool)
    for i, bag in enumerate(bags):
        margins[i, :len(bag)] = bag
        mask[i, :len(bag)] = True
    return ReportCache("r1", [f"{prefix}{i}" for i in range(len(bags))], margins, mask)


@pytest.fixture
def ragged():
    """400 bags of 1-5 margins, spanning confident-yes to confident-no."""
    rng = np.random.default_rng(0)
    return [list(rng.normal(0, 5, size=int(rng.integers(1, 6)))) for _ in range(400)]


# ── Vectorised poolings against the scalar definitions ───────────────────────

SCALAR = {
    "P0": lambda b: float(pg.accept_confidence(b) > 0.5),
    # kappa is THIS bag's segment count, which is why the vectorised form needs a per-node kappa
    # rather than the report's S_max.
    "P3_S": lambda b: pg.noisy_or_kappa(b, max(1, len(b))),
    "P1": pg.accept_confidence,
    "P2": pg.second_largest_confidence,
    "P3_1": lambda b: pg.noisy_or_kappa(b, 1.0),
    "P3_2": lambda b: pg.noisy_or_kappa(b, 2.0),
    "P4": pg.mean_confidence,
    "lse_beta1": pg.lse_beta_prob,
}


class TestPoolingsMatchTheDefinitions:
    @pytest.mark.parametrize("name", sorted(SCALAR))
    def test_vectorised_equals_scalar(self, ragged, name):
        cache = make_cache(ragged)
        vectorised = POOLINGS[name](cache, None)
        scalar = np.array([SCALAR[name](b) for b in ragged])
        assert np.abs(vectorised - scalar).max() < 1e-12

    def test_p3_1_is_the_plain_noisy_or(self, ragged):
        cache = make_cache(ragged)
        assert np.abs(
            POOLINGS["P3_1"](cache, None) - np.array([pg.noisy_or(b) for b in ragged])
        ).max() < 1e-12

    def test_short_bags_are_not_padded_into_the_score(self):
        """A report with two segments must score as two, not as five with three phantom zeros."""
        cache = make_cache([[3.0, 3.0], [3.0, 3.0, 0.0, 0.0, 0.0]])
        p4 = POOLINGS["P4"](cache, None)
        assert p4[0] == pytest.approx(pg.mean_confidence([3.0, 3.0]))
        assert p4[0] > p4[1], "the padded row has three genuine 0.5s dragging its mean down"

    def test_p2_is_zero_when_only_one_segment_was_retrieved(self):
        cache = make_cache([[4.0], [4.0, 4.0]])
        p2 = POOLINGS["P2"](cache, None)
        assert p2[0] == 0.0
        assert p2[1] > 0.0


class TestDominance:
    """P3₁ is the right operator to build a cache with, and P0 is the documented exception."""

    @pytest.mark.parametrize("name", ["P1", "P2", "P4", "P3_2", "P3_S", "lse_beta1"])
    def test_p3_1_dominates_every_graded_operator(self, ragged, name):
        cache = make_cache(ragged)
        assert (POOLINGS[DOMINATING_POOLING](cache, None) >= POOLINGS[name](cache, None) - 1e-12).all()

    def test_p0_is_not_pointwise_dominated(self, ragged):
        """It is an indicator, so at a node with P1 just over a half it exceeds P3₁ itself."""
        cache = make_cache(ragged)
        assert not (POOLINGS["P3_1"](cache, None) >= POOLINGS["P0"](cache, None) - 1e-12).all()

    def test_but_p0_only_fires_where_p3_1_exceeds_a_half(self, ragged):
        """The argument that rescues containment, and the reason for the tau_min <= 1/2 rule."""
        cache = make_cache(ragged)
        fires = POOLINGS["P0"](cache, None) > 0
        assert (POOLINGS["P3_1"](cache, None)[fires] > P0_MAX_CACHE_THRESHOLD).all()

    def test_p3_1_is_monotone_in_the_segment_prefix(self, ragged):
        """More segments can only add evidence, so a cache at S_max contains every smaller S."""
        cache = make_cache(ragged)
        scores = [POOLINGS["P3_1"](cache, s) for s in range(1, cache.s_max + 1)]
        for smaller, larger in zip(scores, scores[1:]):
            assert (larger >= smaller - 1e-12).all()


class TestContainment:
    def test_g_is_below_the_identity_and_converges_to_it(self):
        for t in (1e-6, 1.5e-4, 1.9e-3, 0.23, 0.5, 0.93, 0.9997):
            assert containment_threshold(t) < t
        assert containment_threshold(1e-8) == pytest.approx(1e-8, rel=1e-6)

    def test_the_shipped_cache_threshold_supports_the_shipped_grid(self):
        """An earlier exploratory run cached at 1.5e-4. Every threshold above the grid floor re-runs from it."""
        for tau in (1.9e-3, 0.23, 0.93, 0.9997):
            assert cache_supports("P3_1", tau, 1.5e-4)

    def test_p0_needs_a_cache_at_or_below_one_half(self):
        assert cache_supports("P0", 0.9, 0.4)
        assert not cache_supports("P0", 0.9, 0.6)
        assert cache_supports("P1", 0.9, 0.6), "a graded operator has no such restriction"

    def test_lrgate_is_never_supported_from_a_margin_cache(self):
        assert not cache_supports("LRGate", 0.23, 1e-9)

    def test_unknown_pooling_is_not_supported(self):
        assert not cache_supports("nonsense", 0.23, 1e-9)

    def test_a_strict_cache_does_not_support_a_loose_offline_evaluation(self):
        assert not cache_supports("P3_1", 1.5e-4, 0.3)


# ── Traversal re-run ─────────────────────────────────────────────────────────

@pytest.fixture
def toy_dag():
    """A -> B, C;  B -> D. Roots: [A]."""
    return {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}, ["A"]


class TestOffline:
    def test_expands_only_above_tau_prune(self, toy_dag):
        children, roots = toy_dag
        cache = ReportCache("r", ["A", "B", "C", "D"],
                            np.zeros((4, 1)), np.ones((4, 1), bool))
        prune = np.array([0.9, 0.1, 0.1, 0.1])   # only A expands
        accept = np.zeros(4)
        got = replay(children, roots, cache, prune, accept, tau_prune=0.5, tau_accept=0.5)
        assert set(got.result.visits) == {"A", "B", "C"}, "D is behind unexpanded B"
        assert got.n_containment_misses == 0

    def test_accepts_only_above_tau_accept(self, toy_dag):
        children, roots = toy_dag
        cache = ReportCache("r", ["A", "B", "C", "D"],
                            np.zeros((4, 1)), np.ones((4, 1), bool))
        prune = np.ones(4)
        accept = np.array([0.1, 0.9, 0.2, 0.95])
        got = replay(children, roots, cache, prune, accept, tau_prune=0.5, tau_accept=0.5)
        assert got.accepted == {"B", "D"}

    def test_at_accept_is_free_and_does_not_retraverse(self, toy_dag):
        """The structural fact that makes the acceptance axis cost nothing."""
        children, roots = toy_dag
        cache = ReportCache("r", ["A", "B", "C", "D"],
                            np.zeros((4, 1)), np.ones((4, 1), bool))
        prune = np.ones(4)
        accept = np.array([0.1, 0.6, 0.2, 0.95])
        got = replay(children, roots, cache, prune, accept, tau_prune=0.5, tau_accept=0.5)
        assert got.at_accept(0.9).accepted == {"D"}
        assert got.at_accept(0.05).accepted == {"A", "B", "C", "D"}
        assert set(got.at_accept(0.9).visits) == set(got.result.visits), "visits are invariant"

    def test_a_missing_node_is_recorded_not_swallowed(self, toy_dag):
        children, roots = toy_dag
        cache = ReportCache("r", ["A", "B"], np.zeros((2, 1)), np.ones((2, 1), bool))
        got = replay(children, roots, cache, np.ones(2), np.ones(2),
                     tau_prune=0.5, tau_accept=0.5)
        assert got.n_containment_misses > 0
        assert "C" in got.missed_nodes

    def test_a_missing_node_neither_expands_nor_is_accepted(self, toy_dag):
        """The conservative choice: an unknown node cannot invent descendants or predictions."""
        children, roots = toy_dag
        cache = ReportCache("r", ["A"], np.zeros((1, 1)), np.ones((1, 1), bool))
        got = replay(children, roots, cache, np.ones(1), np.ones(1),
                     tau_prune=0.5, tau_accept=0.5)
        assert "D" not in got.result.visits, "B was missing, so it cannot have expanded"
        assert got.accepted == {"A"}


class TestOfflineCohort:
    def test_returns_both_the_predicted_and_the_scored_sets(self, toy_dag):
        """Coverage and conformal risk control are defined on `scored`, not on `predicted`."""
        children, roots = toy_dag
        caches = {
            "r1": make_cache([[3.0], [3.0], [-3.0], [3.0]], prefix="")
        }
        caches["r1"].node_ids = ["A", "B", "C", "D"]
        caches["r1"].index = {h: i for i, h in enumerate(["A", "B", "C", "D"])}
        got = evaluate_offline_cohort(children, roots, caches, "P3_1", "P1", 0.5, 0.5)
        assert got["predicted"]["r1"] <= got["scored"]["r1"]
        assert "C" in got["scored"]["r1"] and "C" not in got["predicted"]["r1"]
        assert got["n_containment_misses"] == 0

    def test_reports_cost_per_report(self, toy_dag):
        children, roots = toy_dag
        caches = {"r1": make_cache([[1.0, 1.0]] * 4, prefix="")}
        caches["r1"].node_ids = ["A", "B", "C", "D"]
        caches["r1"].index = {h: i for i, h in enumerate(["A", "B", "C", "D"])}
        got = evaluate_offline_cohort(children, roots, caches, "P3_1", "P1", 0.5, 0.5)
        assert got["calls_per_report"] == got["n_slm_calls"]
        assert got["n_reports"] == 1


# ── Loading, and fidelity against the driver's own schema ────────────────────

class TestLoadScoreCache:
    def test_reads_the_driver_schema(self, tmp_path):
        from fixtures.exp13_output import write_calls
        path = tmp_path / "tree_lse_beta1_calls.jsonl"
        write_calls(path, "tree_lse_beta1")
        caches = load_score_cache(path, ctx_type="union")
        assert caches
        for cache in caches.values():
            assert cache.n_nodes > 0
            assert cache.mask.any()

    def test_ranks_land_in_column_order(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        rows = [
            {"report_id": "r1", "hpo_id": "HP:1", "ctx_type": "union", "rank": 3, "margin": -3.0},
            {"report_id": "r1", "hpo_id": "HP:1", "ctx_type": "union", "rank": 1, "margin": 1.0},
            {"report_id": "r1", "hpo_id": "HP:1", "ctx_type": "union", "rank": 2, "margin": -1.0},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        cache = load_score_cache(path)["r1"]
        assert list(cache.margins[0]) == [1.0, -1.0, -3.0], "sorted by rank, not by file order"

    def test_other_context_types_are_filtered_out(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        rows = [
            {"report_id": "r1", "hpo_id": "HP:1", "ctx_type": "union", "rank": 1, "margin": 1.0},
            {"report_id": "r1", "hpo_id": "HP:2", "ctx_type": "own", "rank": 1, "margin": 9.0},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert load_score_cache(path, ctx_type="union")["r1"].node_ids == ["HP:1"]
        assert load_score_cache(path, ctx_type="own")["r1"].node_ids == ["HP:2"]

    def test_a_truncated_tail_does_not_abort_the_load(self, tmp_path):
        """The driver appends per report, so a hard kill can leave half a line."""
        path = tmp_path / "calls.jsonl"
        good = json.dumps(
            {"report_id": "r1", "hpo_id": "HP:1", "ctx_type": "union", "rank": 1, "margin": 1.0})
        path.write_text(good + "\n{\"report_id\": \"r1\", \"hpo", encoding="utf-8")
        assert load_score_cache(path)["r1"].node_ids == ["HP:1"]

    def test_report_filter(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        rows = [
            {"report_id": "r1", "hpo_id": "HP:1", "ctx_type": "union", "rank": 1, "margin": 1.0},
            {"report_id": "r2", "hpo_id": "HP:1", "ctx_type": "union", "rank": 1, "margin": 1.0},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert set(load_score_cache(path, report_ids=["r1"])) == {"r1"}


class TestFidelity:
    """Re-running a run's own configuration must reproduce that run's scores.

    The fixture writes rank-1 margins of +2.0 and the rest -2.0, so the accept score every driver
    would have recorded is ``sigmoid(2.0)`` for every (report, term) pair. If the re-run does not
    return that, the cache is not being read the way the driver wrote it, and every number
    the tree chapter derives from a re-run is void.
    """

    def test_accept_scores_match_the_driver(self, tmp_path):
        from fixtures.exp13_output import write_calls
        path = tmp_path / "tree_lse_beta1_calls.jsonl"
        write_calls(path, "tree_lse_beta1")
        expected = 1.0 / (1.0 + np.exp(-2.0))
        for cache in load_score_cache(path).values():
            assert POOLINGS["P1"](cache, None) == pytest.approx(expected)

    def test_lse_beta1_matches_its_own_definition_on_the_fixture(self, tmp_path):
        from fixtures.exp13_output import write_calls
        path = tmp_path / "tree_lse_beta1_calls.jsonl"
        write_calls(path, "tree_lse_beta1")
        for cache in load_score_cache(path).values():
            for i in range(cache.n_nodes):
                bag = cache.margins[i][cache.mask[i]]
                assert POOLINGS["lse_beta1"](cache, None)[i] == pytest.approx(
                    pg.lse_beta_prob(bag))


class TestCostAccounting:
    """Verifier calls must be counted over the S prefix, not over everything the cache holds.

    The bug this pins: ``replay`` originally counted ``cache.mask[i].sum()``, the full row, so a
    re-run at S = 1 reported the cost of a re-run at S_max. Cost per report is one of the numbers
    the thesis quotes, and the error was invisible in the predictions, only the cost column moved.
    """

    @pytest.fixture
    def three_nodes(self):
        children = {"A": ["B"], "B": ["C"], "C": []}
        cache = make_cache([[3.0, 3.0, 3.0]] * 3)
        cache.node_ids = ["A", "B", "C"]
        cache.index = {h: i for i, h in enumerate(cache.node_ids)}
        return children, ["A"], cache

    @pytest.mark.parametrize("s,expected", [(1, 3), (2, 6), (3, 9)])
    def test_calls_scale_with_s(self, three_nodes, s, expected):
        children, roots, cache = three_nodes
        got = replay(children, roots, cache,
                     POOLINGS["P1"](cache, s), POOLINGS["P1"](cache, s),
                     tau_prune=0.5, tau_accept=0.5, s=s)
        assert len(got.result.visits) == 3
        assert got.result.n_slm_calls == expected

    def test_omitting_s_counts_the_whole_cache(self, three_nodes):
        children, roots, cache = three_nodes
        got = replay(children, roots, cache,
                     POOLINGS["P1"](cache, None), POOLINGS["P1"](cache, None),
                     tau_prune=0.5, tau_accept=0.5)
        assert got.result.n_slm_calls == 9

    def test_s_above_the_cache_is_clamped(self, three_nodes):
        children, roots, cache = three_nodes
        got = replay(children, roots, cache,
                     POOLINGS["P1"](cache, 99), POOLINGS["P1"](cache, 99),
                     tau_prune=0.5, tau_accept=0.5, s=99)
        assert got.result.n_slm_calls == 9, "cannot cost more segments than were retrieved"

    def test_short_rows_are_not_overcounted(self):
        """A node whose retrieval returned two segments costs two calls even at S = 5."""
        children = {"A": []}
        cache = make_cache([[3.0, 3.0]])
        cache.node_ids = ["A"]
        cache.index = {"A": 0}
        got = replay(children, ["A"], cache,
                     POOLINGS["P1"](cache, 5), POOLINGS["P1"](cache, 5),
                     tau_prune=0.5, tau_accept=0.5, s=5)
        assert got.result.n_slm_calls == 2

    def test_offline_cohort_threads_s_through(self):
        children = {"A": ["B"], "B": []}
        cache = make_cache([[2.0, 2.0, 2.0]] * 2)
        cache.node_ids = ["A", "B"]
        cache.index = {"A": 0, "B": 1}
        one = evaluate_offline_cohort(children, ["A"], {"r": cache}, "P1", "P1", 0.5, 0.5, s=1)
        three = evaluate_offline_cohort(children, ["A"], {"r": cache}, "P1", "P1", 0.5, 0.5, s=3)
        assert one["n_slm_calls"] == 2 and three["n_slm_calls"] == 6


class TestP3SUsesAPerNodeKappa:
    """kappa for P3_S is the segments THIS term was scored on, not the report's S_max.

    Retrieval returns fewer than S whenever a report is short, so nodes within one report differ.
    A report-wide scalar charges a node scored on two segments as though it had five, which mixes
    one node's evidence count into another's and is wrong in a way no aggregate would reveal.
    """

    def test_ragged_rows_match_their_own_scalar_reference(self):
        bags = [[2.0, 1.0], [2.0, 1.0, 0.5, -1.0], [3.0]]
        cache = make_cache(bags)
        got = POOLINGS["P3_S"](cache, None)
        want = np.array([pg.noisy_or_kappa(b, len(b)) for b in bags])
        assert np.abs(got - want).max() < 1e-12

    def test_a_short_row_is_not_charged_for_absent_segments(self):
        """Two identical rows, one padded out by the presence of a longer sibling."""
        alone = make_cache([[2.0, 1.0]])
        beside_longer = make_cache([[2.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        assert POOLINGS["P3_S"](alone, None)[0] == pytest.approx(
            POOLINGS["P3_S"](beside_longer, None)[0])

    def test_equals_one_minus_the_geometric_mean_of_one_minus_sigma(self):
        """The clean reading the per-node kappa buys."""
        bag = [2.0, 1.0, -0.5]
        cache = make_cache([bag])
        sigma = 1.0 / (1.0 + np.exp(-np.array(bag)))
        assert POOLINGS["P3_S"](cache, None)[0] == pytest.approx(
            1.0 - np.prod(1.0 - sigma) ** (1.0 / len(bag)))

    def test_still_dominated_by_p3_1(self):
        bags = [[2.0, 1.0], [2.0, 1.0, 0.5, -1.0], [3.0]]
        cache = make_cache(bags)
        assert (POOLINGS["P3_1"](cache, None) >= POOLINGS["P3_S"](cache, None) - 1e-12).all()

    def test_an_empty_row_scores_zero(self):
        cache = ReportCache("r", ["a"], np.zeros((1, 3)), np.zeros((1, 3), bool))
        assert POOLINGS["P3_S"](cache, None)[0] == 0.0


# ── The bottleneck sweep against the frontier walk ───────────────────────────
#
# This is the gate that counts most in this file. ``bottleneck_scores`` replaces one traversal per
# threshold with one pass per (report, pooling, S), and it is what makes the conformal risk control
# in the TreePhenoRAG protocol computable at all. If it disagrees with ``replay`` anywhere, every number in the tree
# chapter moves silently. So it is tested the same way the poolings are: against the reference, on
# adversarial input, not on a happy path.


def random_dag(n_nodes: int, n_roots: int, seed: int):
    """A random DAG with parents drawn only from strictly earlier nodes, so it cannot cycle."""
    rng = random.Random(seed)
    ids = [f"HP:{i:07d}" for i in range(n_nodes)]
    roots = ids[:n_roots]
    children: dict[str, list[str]] = {h: [] for h in ids}
    for i in range(n_roots, n_nodes):
        for p in rng.sample(ids[:i], min(rng.randint(1, 3), i)):
            children[p].append(ids[i])
    return ids, {h: sorted(set(v)) for h, v in children.items()}, roots


class TestBottleneckSweepEqualsTheFrontierWalk:
    """``r(v) >= tau`` must reproduce ``traverse``, at every threshold."""

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("drop_frac", [0.0, 0.15])
    def test_visited_expanded_accepted_calls_and_misses_all_agree(self, seed, drop_frac):
        rng = np.random.default_rng(seed)
        ids, children, roots = random_dag(180, 5, seed)

        # A cache that may be missing nodes, so containment accounting is exercised too.
        keep = [h for h in ids if rng.random() >= drop_frac]
        s_max = 4
        margins = rng.normal(0, 3, size=(len(keep), s_max))
        mask = rng.random((len(keep), s_max)) > 0.2
        mask[:, 0] = True
        cache = ReportCache(f"R{seed}", list(keep), margins, mask)

        graph = build_traversal_graph(children, roots)
        prune = rng.random(len(keep))
        accept = rng.random(len(keep))

        g_prune = align_to_graph(graph, keep, prune)
        g_accept = align_to_graph(graph, keep, accept)
        g_valid = align_to_graph(graph, keep, mask.sum(axis=1).astype(np.float64))
        in_cache = in_cache_mask(graph, keep)
        r = bottleneck_scores(graph, g_prune)

        for tau in (0.0, 1e-6, 0.05, 0.2, 0.5, 0.75, 0.9, 0.999, 1.0):
            for acc in (0.0, 0.3, 0.5, 0.9, 1.0):
                ref = replay(children, roots, cache, prune, accept, tau, acc)
                visited = visited_at(r, tau)
                names = np.array(graph.node_ids)

                assert set(names[visited]) == set(ref.result.visits)
                assert set(names[expanded_at(r, g_prune, tau)]) == {
                    h for h, v in ref.result.visits.items() if v.expanded
                }
                # No in_cache mask here: re-run() fills an absent node with accept score 0.0 and
                # then thresholds it, so at tau_accept = 0 the reference accepts it. Mirror the
                # reference, not improving on it, a cell with misses is invalid anyway.
                assert set(names[visited & (g_accept >= acc)]) == set(ref.result.accepted)
                assert calls_at(r, g_valid, tau) == ref.result.n_slm_calls
                assert containment_misses(r, in_cache, tau) == ref.n_containment_misses

    def test_a_root_that_is_also_a_child_stays_seeded(self):
        """HPO is a DAG: a layer-1 term can also hang off another phenotype.

        ``traverse`` seeds every root unconditionally, so such a node's ``r`` must stay ``+inf``
        however badly its other parent scored. A layering that ranked it below that parent, the
        obvious implementation, would drop it from the visited set.
        """
        children = {"A": ["X"], "X": ["B"], "B": ["Y"], "Y": []}
        roots, ids = ["A", "B"], ["A", "X", "B", "Y"]
        graph = build_traversal_graph(children, roots)

        prune = np.array([0.9, 0.0, 0.9, 0.9])   # X never expands, so B is unreachable by path
        cache = ReportCache("r", list(ids), np.zeros((4, 1)), np.ones((4, 1), dtype=bool))
        r = bottleneck_scores(graph, align_to_graph(graph, ids, prune))

        for tau in (0.1, 0.5, 0.95):
            ref = replay(children, roots, cache, prune, np.zeros(4), tau, 2.0)
            names = np.array(graph.node_ids)
            assert set(names[visited_at(r, tau)]) == set(ref.result.visits)
        assert np.isinf(r[graph.index["B"]])
        # Y is reachable only through B, which is seeded, so it survives X scoring zero.
        assert "Y" in set(np.array(graph.node_ids)[visited_at(r, 0.5)])

    def test_r_is_monotone_so_the_visited_set_only_shrinks(self):
        """The containment argument the cache rests on, stated on r itself."""
        ids, children, roots = random_dag(120, 4, 11)
        graph = build_traversal_graph(children, roots)
        rng = np.random.default_rng(11)
        r = bottleneck_scores(graph, align_to_graph(graph, ids, rng.random(len(ids))))
        previous = None
        for tau in np.linspace(0.0, 1.0, 25):
            current = set(np.flatnonzero(visited_at(r, tau)))
            if previous is not None:
                assert current <= previous
            previous = current


class TestCoverageBreakpoints:
    """The tau grid conformal risk control needs is the annotated terms' own r-values, and nothing else."""

    def test_the_loss_changes_only_at_a_breakpoint(self):
        ids, children, roots = random_dag(150, 4, 3)
        graph = build_traversal_graph(children, roots)
        rng = np.random.default_rng(3)
        r = bottleneck_scores(graph, align_to_graph(graph, ids, rng.random(len(ids))))

        gold = [graph.index[h] for h in ids[10:40]]
        breaks = coverage_breakpoints(r, gold)

        def loss(tau):
            return float(np.mean([r[i] < tau for i in gold]))

        # Between consecutive breakpoints the loss is constant. At each one it steps.
        for lo, hi in zip(breaks, breaks[1:]):
            mid = (lo + hi) / 2.0
            assert loss(mid) == pytest.approx(loss(lo + 1e-12))
        assert len({loss(b + 1e-12) for b in breaks}) > 1

    def test_infinite_values_are_dropped(self):
        """An annotated term sitting on a root is missed at no finite threshold, so it is not a candidate."""
        children = {"A": ["X"], "X": []}
        graph = build_traversal_graph(children, ["A"])
        r = bottleneck_scores(graph, align_to_graph(graph, ["A", "X"], np.array([0.4, 0.4])))
        assert np.isinf(r[graph.index["A"]])
        assert list(coverage_breakpoints(r, [graph.index["A"], graph.index["X"]])) == [
            pytest.approx(0.4)
        ]


# ── The columnar cache ───────────────────────────────────────────────────────


def write_calls_file(path, n_reports=4, n_nodes=25, s_max=5, instrumented=True, seed=0):
    """A ``_calls.jsonl`` in the driver's own schema, including the two things real ones have.

    A row from a second ``ctx_type`` (which must be filtered out) and a truncated final line (which
    a hard kill leaves behind, and which the reader must skip, not die on).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_reports):
        rid = f"R{r:03d}"
        for n in range(n_nodes):
            for rank in range(1, int(rng.integers(1, s_max + 1)) + 1):
                ly, ln = float(rng.normal(2, 3)), float(rng.normal(0, 3))
                rec = {
                    "report_id": rid, "hpo_id": f"HP:{n:07d}", "ctx_type": "union", "rank": rank,
                    "sent_index": int(rng.integers(0, 40)),
                    "cosine_sim": round(float(rng.random()), 6),
                    "logit_yes": round(ly, 6), "logit_no": round(ln, 6),
                    "logsumexp_all": round(float(np.logaddexp(ly, ln) + abs(rng.normal(0, .5))), 6),
                    "top1_token_id": 9642, "margin": round(ly - ln, 6),
                    "verdict": "Yes" if ly > ln else "No",
                }
                if instrumented:
                    rec["wall_clock_s"] = round(float(rng.random()), 6)
                    rec["n_prompt_tokens"] = int(rng.integers(100, 900))
                rows.append(rec)
        rows.append({**rows[-1], "ctx_type": "own"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"report_id": "R00')
    return rows


class TestColumnarCache:
    """``ingest_score_cache`` must be a faster spelling of ``load_score_cache``, not a different one.

    The synthetic-sentence score store's cache is ~21.7 M rows and 6-7 GB of JSON, which is minutes of parsing and gigabytes
    of interpreter objects on every stage invocation. Converting once is the only way the re-run
    stages are runnable, but only if the conversion is exact, so that is what is fixed here.
    """

    def test_round_trip_matches_the_json_loader_on_every_pooling(self, tmp_path):
        calls = tmp_path / "v_calls.jsonl"
        write_calls_file(calls)
        reference = load_score_cache(calls, ctx_type="union")
        ingest_score_cache(calls, tmp_path / "v.npz", ctx_type="union")
        got = load_cache_npz(tmp_path / "v.npz")

        assert set(got) == set(reference)
        for rid, ref in reference.items():
            new = got[rid]
            assert new.node_ids == ref.node_ids
            assert np.array_equal(new.mask, ref.mask)
            assert np.allclose(new.margins, ref.margins, atol=1e-6)
            for name, fn in POOLINGS.items():
                assert np.allclose(fn(new), fn(ref), atol=1e-6), name

    def test_it_keeps_the_columns_the_json_loader_throws_away(self, tmp_path):
        calls = tmp_path / "v_calls.jsonl"
        write_calls_file(calls)
        ingest_score_cache(calls, tmp_path / "v.npz")
        cache = load_cache_npz(tmp_path / "v.npz")["R000"]

        assert np.isfinite(cache.cosine[cache.mask]).all()
        assert np.isfinite(cache.wall_clock_s[cache.mask]).all()
        assert cache.n_prompt_tokens[cache.mask].min() >= 100
        # captured mass is a probability, derived from the logits, not read
        mass = cache.captured_mass[cache.mask]
        assert ((mass > 0) & (mass <= 1.0 + 1e-6)).all()

    def test_padding_is_distinguishable_from_a_measurement(self, tmp_path):
        """A node scored on 2 of 5 segments must not read as 'cosine 0, zero seconds, zero tokens'."""
        calls = tmp_path / "v_calls.jsonl"
        write_calls_file(calls)
        ingest_score_cache(calls, tmp_path / "v.npz")
        cache = load_cache_npz(tmp_path / "v.npz")["R000"]

        assert (~cache.mask).any(), "fixture should be ragged"
        assert np.isnan(cache.cosine[~cache.mask]).all()
        assert np.isnan(cache.wall_clock_s[~cache.mask]).all()
        assert (cache.n_prompt_tokens[~cache.mask] == -1).all()

    def test_an_uninstrumented_cache_says_not_measured_rather_than_zero(self, tmp_path):
        calls = tmp_path / "old_calls.jsonl"
        write_calls_file(calls, instrumented=False)
        provenance = ingest_score_cache(calls, tmp_path / "old.npz")
        cache = load_cache_npz(tmp_path / "old.npz")["R000"]

        assert not provenance["has_wall_clock"] and not provenance["has_prompt_tokens"]
        assert np.isnan(cache.wall_clock_s).all()
        assert (cache.n_prompt_tokens == -1).all()

    def test_provenance_records_what_was_skipped(self, tmp_path):
        calls = tmp_path / "v_calls.jsonl"
        write_calls_file(calls)
        provenance = ingest_score_cache(calls, tmp_path / "v.npz")

        assert provenance["n_reports"] == 4
        assert provenance["n_undecodable_lines"] == 1   # The truncated tail
        assert read_cache_provenance(tmp_path / "v.npz") == provenance

    def test_an_interleaved_file_raises_instead_of_mis_ordering_ranks(self, tmp_path):
        """Rank order *is* retrieval order, so a report split across the file is not recoverable.

        Concatenated shards are fine (each holds whole reports). An interleaved file is not, and
        quietly merging it would corrupt the ``top-S`` prefix every re-run depends on.
        """
        calls = tmp_path / "v_calls.jsonl"
        write_calls_file(calls)
        rows = [ln for ln in calls.read_text(encoding="utf-8").splitlines()
                if ln.startswith("{") and '"ctx_type": "union"' in ln]
        first = next(ln for ln in rows if '"report_id": "R000"' in ln)
        other = next(ln for ln in rows if '"report_id": "R001"' in ln)
        bad = tmp_path / "bad_calls.jsonl"
        bad.write_text("\n".join([first, other, first]) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="merge_shards"):
            ingest_score_cache(bad, tmp_path / "bad.npz")
