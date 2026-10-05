"""§Reachability recall (eq. 6), §Blocking depth, §Traversal cost by depth.

The traversal graph is the toy DAG with the root removed, so the search starts at the two layer-1
organ systems A and G, the same shape ``hpo_extraction.treephenorag.traversal.build_children_map`` produces::

    A (d1) -> B (d2) -> {C, D, M} (d3)
           -> E (d2) -> F (d3)
    G (d1) -> H (d2) -> M (d3)          # M is reachable by two routes

The decisive case for the blocking-depth rule is ``expanded = {A}``: M's route through B is cut at
depth 2 and its route through H is cut at depth 1. The rule chosen for this thesis records the
**deepest** cut, so M is blocked at 2, not 1.
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    bfs_depths,
    blocking_depth_distribution,
    blocking_depths,
    depth_cost_table,
    parents_from_children,
    reachability_recall,
    reachability_recall_cohort,
    reachable_set,
)
from fixtures.toy_ontology import A, B, C, D, E, F, G, H, M, toy_children_map

pytestmark = pytest.mark.unit

INF = float("inf")


@pytest.fixture
def graph():
    return toy_children_map()


# ── Graph preparation ────────────────────────────────────────────────────────

def test_bfs_depths_put_the_layer1_roots_at_one(graph):
    """Matches ``hpo_extraction.treephenorag.traversal.traverse``, which seeds ``depth_of[root] = 1``."""
    children, roots = graph
    depths = bfs_depths(children, roots)
    assert depths[A] == depths[G] == 1
    assert depths[B] == depths[E] == depths[H] == 2
    assert depths[C] == depths[D] == depths[F] == depths[M] == 3


def test_parents_from_children_recovers_both_parents_of_m(graph):
    children, _ = graph
    parents = parents_from_children(children)
    assert sorted(parents[M]) == sorted([B, H])
    assert parents[C] == [B]


# ── Eq. (6): reachability ────────────────────────────────────────────────────

def test_reachable_set_stops_at_unexpanded_nodes(graph):
    """expanded = {A, B}: G is visited but never opened, so H and F are never reached."""
    children, roots = graph
    assert reachable_set(children, roots, {A, B}) == {A, G, B, E, C, D, M}


def test_roots_are_always_reached_even_when_they_do_not_expand(graph):
    children, roots = graph
    assert reachable_set(children, roots, set()) == {A, G}


def test_reachability_recall_is_the_gold_fraction_still_reachable(graph):
    """ground truth = {C, F}: C survives, F does not."""
    children, roots = graph
    reachable = reachable_set(children, roots, {A, B})
    assert reachability_recall({C, F}, reachable) == pytest.approx(0.5)


def test_reachability_recall_is_none_without_gold(graph):
    children, roots = graph
    assert reachability_recall(set(), reachable_set(children, roots, {A})) is None


def test_reachability_recall_bounds_end_to_end_recall(graph):
    """The property the metric exists for: a term the search never visits cannot be accepted.

    Whatever the accept threshold does, the accepted set is a subset of the reachable set, so
    end-to-end recall can never exceed reachability recall.
    """
    children, roots = graph
    reachable = reachable_set(children, roots, {A, B})
    gold = {C, D, F}
    accepted = reachable & {C, D}  # any accept rule can only pick from what was reached
    end_to_end = len(gold & accepted) / len(gold)
    assert end_to_end <= reachability_recall(gold, reachable)


def test_reachability_recall_cohort_micro_and_macro(graph):
    """Report 1 ground truth={C, F} -> 1/2. Report 2 ground truth={C} -> 1/1.

    micro = 2/3 (pooled terms), macro = 3/4 (mean over reports), they differ because the reports
    carry different numbers of annotated terms.
    """
    children, roots = graph
    reachable = reachable_set(children, roots, {A, B})
    out = reachability_recall_cohort([{C, F}, {C}], [reachable, reachable])
    assert out["micro_reachability_recall"] == pytest.approx(2 / 3)
    assert out["macro_reachability_recall"] == pytest.approx(0.75)
    assert out["n_gold_terms"] == 3
    assert out["n_gold_reachable"] == 2


def test_reachability_recall_cohort_skips_reports_without_gold(graph):
    children, roots = graph
    reachable = reachable_set(children, roots, {A, B})
    out = reachability_recall_cohort([{C}, set()], [reachable, reachable])
    assert out["n_reports_with_gold"] == 1


# ── Blocking depth ───────────────────────────────────────────────────────────

def test_reachable_nodes_have_infinite_blocking_depth(graph):
    children, roots = graph
    best = blocking_depths(children, roots, {A, B})
    for node in (A, G, B, E, C, D, M):
        assert best[node] == INF


def test_blocked_nodes_record_the_depth_of_the_severing_parent(graph):
    """expanded = {A, B}: G refuses to expand at depth 1, E at depth 2."""
    children, roots = graph
    best = blocking_depths(children, roots, {A, B})
    assert best[H] == 1  # cut at G, depth 1
    assert best[F] == 2  # cut at E, depth 2


def test_deepest_severing_node_wins_over_the_shallower_one(graph):
    """The rule this thesis chose, fixed. expanded = {A}: every route below depth 1 is cut.

    M has two routes. Through B (depth 2) and through H, which is itself cut at G (depth 1). The
    chosen rule takes the **max**, the path the traversal got furthest along, so M is blocked at
    2. Had "shallowest severing node" been chosen instead, this assertion would read 1.
    """
    children, roots = graph
    best = blocking_depths(children, roots, {A})
    assert best[H] == 1
    assert best[B] == INF  # B is reached. It simply does not expand
    assert best[C] == best[D] == 2
    assert best[M] == 2


def test_a_second_parent_can_rescue_a_node_from_pruning(graph):
    """expanded = {A, G, E, H}: B is cut, but M survives through H, "any path" is enough."""
    children, roots = graph
    best = blocking_depths(children, roots, {A, G, E, H})
    assert best[C] == 2   # only route is through B, which did not expand
    assert best[M] == INF  # rescued via H


def test_blocking_depth_is_infinite_for_the_reachable_set(graph):
    """The two functions must agree on which nodes survive, one property, two implementations."""
    children, roots = graph
    for expanded in ({A}, {A, B}, {A, G, E, H}, set(), set(children)):
        reachable = reachable_set(children, roots, expanded)
        best = blocking_depths(children, roots, expanded)
        assert {n for n, v in best.items() if v == INF} == reachable


def test_blocking_depth_distribution_hand_worked(graph):
    """ground truth = {C, F, H} with expanded = {A, B}: C reachable, F cut at 2, H cut at 1."""
    children, roots = graph
    out = blocking_depth_distribution([{C, F, H}], children, roots, [{A, B}])
    assert out["histogram"] == {1: 1, 2: 1}
    assert out["n_blocked"] == 2
    assert out["n_reachable"] == 1
    assert out["mean"] == pytest.approx(1.5)
    assert out["fraction_at_depth_1"] == pytest.approx(0.5)


def test_blocking_depth_distribution_flags_terms_outside_the_graph(graph):
    """A ground truth code the traversal graph does not contain is an ontology hygiene issue, not pruning."""
    children, roots = graph
    out = blocking_depth_distribution([{"HP:9999999"}], children, roots, [{A}])
    assert out["n_outside_graph"] == 1
    assert out["n_blocked"] == 0


def test_cycles_are_rejected_rather_than_silently_looping():
    with pytest.raises(ValueError, match="cycle"):
        blocking_depths({"a": ["b"], "b": ["c"], "c": ["b"]}, ["a"], set())


# ── §Scalability: the per-depth table ────────────────────────────────────────

@pytest.fixture
def visits():
    """One report's traversal trace, in the ``*_nodes.jsonl`` shape."""
    def rec(depth, expanded, accepted):
        return {"depth": depth, "expanded": expanded, "accepted": accepted, "n_slm_calls": 5}

    return {
        A: rec(1, True, False),
        G: rec(1, False, False),
        B: rec(2, True, False),
        E: rec(2, False, False),
        C: rec(3, False, True),
        D: rec(3, False, False),
        M: rec(3, False, False),
    }


def test_depth_cost_table_hand_worked(visits):
    """Two nodes at depth 1 (one expands), two at depth 2 (one expands), three at depth 3."""
    rows = depth_cost_table([visits])
    assert [r["depth"] for r in rows] == [1, 2, 3]
    assert [r["frontier_size"] for r in rows] == [2, 2, 3]
    assert [r["n_expanded"] for r in rows] == [1, 1, 0]
    assert [r["survival_rate"] for r in rows] == pytest.approx([0.5, 0.5, 0.0])
    assert [r["cumulative_slm_calls"] for r in rows] == [10, 20, 35]
    assert [r["cumulative_frontier"] for r in rows] == [2, 4, 7]


def test_depth_cost_table_adds_cumulative_reachability_when_gold_is_given(visits):
    """ground truth = {C, F}: C is visited at depth 3, F never, cumulative reach tops out at 1/2."""
    rows = depth_cost_table([visits], [{C, F}])
    assert [r["cumulative_gold_visited"] for r in rows] == [0, 0, 1]
    assert rows[-1]["cumulative_reachability_recall"] == pytest.approx(0.5)


def test_depth_cost_table_without_gold_omits_the_recall_columns(visits):
    rows = depth_cost_table([visits])
    assert "cumulative_reachability_recall" not in rows[0]


def test_depth_cost_table_rejects_misaligned_gold(visits):
    with pytest.raises(ValueError, match="aligned"):
        depth_cost_table([visits], [{C}, {D}])


def test_depth_cost_table_pools_across_reports(visits):
    rows = depth_cost_table([visits, visits])
    assert [r["frontier_size"] for r in rows] == [4, 4, 6]
    assert rows[-1]["cumulative_slm_calls"] == 70
