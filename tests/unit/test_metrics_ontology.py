"""Graph primitives, the closure convention every other metric takes over.

Reference values are the ones written out in ``tests/fixtures/toy_ontology``'s docstring. The real
ontology is exercised separately at the bottom of this file so the convention is fixed against
``hpo.json`` and not only against a fixture that could share a mistake with the code.
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import UNIVERSAL_NODES, OntologyView, ancestor_count_distribution
from fixtures.toy_ontology import (
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    HALLUCINATION,
    M,
    OBSOLETE_C,
    OUT_OF_SUBTREE,
    ROOT,
)

pytestmark = pytest.mark.unit


# ── The closure convention ───────────────────────────────────────────────────

def test_ancestors_include_self_and_exclude_both_universal_nodes(toy_view):
    """An(C) = {C, B, A}: C itself, its parent, its grandparent, and not the root.

    This is the decision the whole hierarchy section rests on. The root is a member of every term's
    ``Father`` set in the real ``hpo.json``, so without the exclusion each term would hand the
    system a free true positive at both universal nodes.
    """
    assert toy_view.ancestors(C) == frozenset({C, B, A})
    assert ROOT not in toy_view.ancestors(C)
    assert not toy_view.ancestors(C) & UNIVERSAL_NODES


def test_layer1_organ_systems_are_kept_in_the_closure(toy_view):
    """A is a layer-1 organ system and stays in An(C), only the two universal nodes are dropped."""
    assert A in toy_view.ancestors(C)
    assert A in toy_view.scorable


def test_multi_parent_term_has_a_larger_closure_at_equal_depth(toy_view):
    """M and C are both at depth 3, but |An(M)| = 5 against |An(C)| = 3.

    "In a DAG the closure is a union over all parent paths, so multi-parent terms are weighted more
    heavily than single-parent terms at equal depth."
    """
    assert toy_view.depth(M) == toy_view.depth(C) == 3
    assert toy_view.ancestors(M) == frozenset({M, B, A, H, G})
    assert len(toy_view.ancestors(M)) == 5
    assert len(toy_view.ancestors(C)) == 3


def test_ancestors_of_set_is_the_union(toy_view):
    """An({C, F}) = An(C) ∪ An(F) = {C, B, A} ∪ {F, E, A} = {C, B, A, F, E}."""
    assert toy_view.ancestors_of_set({C, F}) == {C, B, A, F, E}


def test_descendants_or_self_is_the_subtree(toy_view):
    """The subtree rooted at B is {B, C, D, M}."""
    assert toy_view.descendants_or_self(B) == frozenset({B, C, D, M})
    assert toy_view.descendants_or_self(C) == frozenset({C})


# ── Resolution, and the three kinds of "not a scorable term" ─────────────────

def test_alt_id_is_remapped(toy_view):
    assert toy_view.resolve(OBSOLETE_C) == C


def test_whitespace_is_stripped(toy_view):
    assert toy_view.resolve(f"  {C}  ") == C


def test_out_of_subtree_term_is_unresolvable_but_in_the_ontology(toy_view):
    """The distinction the false-positive taxonomy needs: a real term in the wrong subtree."""
    assert toy_view.resolve(OUT_OF_SUBTREE) is None
    assert toy_view.in_ontology(OUT_OF_SUBTREE) is True


def test_hallucination_is_neither_resolvable_nor_known(toy_view):
    assert toy_view.resolve(HALLUCINATION) is None
    assert toy_view.in_ontology(HALLUCINATION) is False


def test_universal_nodes_are_not_scorable(toy_view):
    assert toy_view.resolve(ROOT) is None


def test_resolve_set_reports_what_it_dropped(toy_view):
    kept, dropped = toy_view.resolve_set([C, OBSOLETE_C, HALLUCINATION, OUT_OF_SUBTREE])
    assert kept == {C}  # C and its alt id collapse to one term
    assert sorted(dropped) == sorted([HALLUCINATION, OUT_OF_SUBTREE])


# ── Depth, organ system ──────────────────────────────────────────────────────

def test_depths_follow_hpotree_convention_with_layer1_at_one(toy_view):
    assert [toy_view.depth(n) for n in (A, G)] == [1, 1]
    assert [toy_view.depth(n) for n in (B, E, H)] == [2, 2, 2]
    assert [toy_view.depth(n) for n in (C, D, F, M)] == [3, 3, 3, 3]


def test_layer1_of_a_multi_parent_term_is_deterministic(toy_view):
    """M sits under both A and G. The lexicographically smaller is taken, every time."""
    assert toy_view.layer1(M) == min(A, G)
    assert toy_view.layer1(C) == A
    assert toy_view.layer1(A) == A


def test_layer1_ancestors_keeps_every_organ_system_the_canonical_pick_discards(toy_view):
    """``layer1`` partitions; ``layer1_ancestors`` does not, and the taxonomy needs the latter.

    M hangs off B (nervous system) and H (skeletal system), so it belongs to *both*. ``layer1``
    must return one so that per-organ grouping stays a partition. The ``same_branch`` bucket of
    :mod:`~hpo_extraction.evaluation.metrics.errors_existential` asks whether two terms share an ancestor
    of depth one, which is a question about the whole set: comparing canonical picks would call a
    genuinely shared branch ``unrelated`` whenever the two picks happened to differ.
    """
    assert toy_view.layer1_ancestors(M) == frozenset({A, G})
    assert toy_view.layer1(M) == A            # The partition still picks one
    assert toy_view.layer1_ancestors(C) == frozenset({A})
    assert toy_view.layer1_ancestors(A) == frozenset({A})
    # M and H share the skeletal system even though M's canonical organ system is the nervous one.
    assert toy_view.layer1_ancestors(M) & toy_view.layer1_ancestors(H)
    assert toy_view.layer1(M) != toy_view.layer1(H)


def test_layer1_ancestors_is_empty_for_an_unknown_code(toy_view):
    assert toy_view.layer1_ancestors(HALLUCINATION) == frozenset()


# ── Eq. (5)'s distance ───────────────────────────────────────────────────────

def test_undirected_distance_parent_child_is_one(toy_view):
    assert toy_view.undirected_distance(B, C) == 1
    assert toy_view.undirected_distance(C, B) == 1


def test_undirected_distance_siblings_is_two(toy_view):
    """C and D share the parent B: C -> B -> D."""
    assert toy_view.undirected_distance(C, D) == 2


def test_undirected_distance_uses_the_dag_shortcut(toy_view):
    """dist(C, H) = 3 via M, not 4 via the tree route C-B-A-G-H.

    This is why eq. (5) cannot be evaluated from depths or from a spanning tree: M's second parent
    creates a shorter route between two otherwise distant branches.
    """
    assert toy_view.undirected_distance(C, H) == 3


def test_undirected_distance_across_branches(toy_view):
    """dist(C, F) = 4: C -> B -> A -> E -> F."""
    assert toy_view.undirected_distance(C, F) == 4


def test_undirected_distance_to_self_is_zero(toy_view):
    assert toy_view.undirected_distance(C, C) == 0


def test_undirected_distance_is_none_for_unresolvable_codes(toy_view):
    assert toy_view.undirected_distance(C, HALLUCINATION) is None
    assert toy_view.undirected_distance(C, OUT_OF_SUBTREE) is None


# ── The |An(v)| audit ────────────────────────────────────────────────────────

def test_ancestor_count_distribution_reports_the_multi_parent_weighting(toy_view):
    """Ground-truth sets {C, M} and {C}: closures of size 3, 5 and 3 → mean 11/3, one multi-parent term."""
    stats = ancestor_count_distribution([{C, M}, {C}], toy_view)
    assert stats["n_terms"] == 3
    assert stats["histogram"] == {3: 2, 5: 1}
    assert stats["mean"] == pytest.approx(11 / 3)
    assert stats["median"] == 3
    assert stats["max"] == 5
    assert stats["multi_parent_fraction"] == pytest.approx(1 / 3)


def test_ancestor_count_distribution_is_empty_safe(toy_view):
    stats = ancestor_count_distribution([], toy_view)
    assert stats["n_terms"] == 0
    assert stats["histogram"] == {}


def test_ancestor_count_distribution_skips_unresolvable_gold_codes(toy_view):
    """An obsolete or hallucinated ground truth code has no closure to measure and must not be counted."""
    stats = ancestor_count_distribution([{C, HALLUCINATION}], toy_view)
    assert stats["n_terms"] == 1
    assert stats["histogram"] == {3: 1}


# ── Construction ─────────────────────────────────────────────────────────────

def test_view_builds_depths_when_the_tree_has_not(toy_tree):
    """``HPOTree.__init__`` does *not* populate ``depth_dict``, the view must do it itself.

    This branch fires on every real run, since production code constructs ``HPOTree()`` and hands it
    straight to the view. Without it, every depth lookup would raise.
    """
    del toy_tree.depth_dict
    assert not hasattr(toy_tree, "depth_dict")

    view = OntologyView(toy_tree)
    assert view.depth(C) == 3


# ── Guards on unresolvable input ─────────────────────────────────────────────

def test_graph_queries_on_an_unresolvable_code_return_empty_not_raise(toy_view):
    """A hallucinated code must degrade to "nothing known", never to an exception mid-cohort."""
    assert toy_view.ancestors(HALLUCINATION) == frozenset()
    assert toy_view.descendants_or_self(HALLUCINATION) == frozenset()
    assert toy_view.direct_parents(HALLUCINATION) == frozenset()
    assert toy_view.neighbours(HALLUCINATION) == frozenset()
    assert toy_view.depth(HALLUCINATION) is None
    assert toy_view.layer1(HALLUCINATION) is None


def test_empty_and_none_codes_resolve_to_none(toy_view):
    assert toy_view.resolve("") is None
    assert toy_view.resolve(None) is None
    assert toy_view.in_ontology("") is False


def test_is_ancestor_is_strict_and_guards_unresolvable_codes(toy_view):
    assert toy_view.is_ancestor(B, C) is True
    assert toy_view.is_ancestor(C, C) is False           # strict: not its own ancestor
    assert toy_view.is_ancestor(C, B) is False
    assert toy_view.is_ancestor(HALLUCINATION, C) is False


def test_distance_beyond_the_hop_cap_is_none(toy_view):
    """``max_hops`` is a safety valve, not a silent truncation to some large number."""
    assert toy_view.undirected_distance(C, F, max_hops=1) is None
    assert toy_view.undirected_distance(C, F, max_hops=64) == 4


# ── The same convention, against the real ontology ───────────────────────────

@pytest.mark.integration
def test_real_ontology_seizure_closure_excludes_both_universal_nodes(real_hpo_tree):
    """HP:0001250 (Seizure), the closure the thesis will quote, computed on the real hpo.json.

    ``hpo.json``'s ``Father`` set for Seizure is ``{HP:0000001, HP:0000118, HP:0000707,
    HP:0012638}``. The two universal nodes must be gone and the organ system must remain, or every
    hF in the results chapter is inflated.
    """
    view = OntologyView(real_hpo_tree)
    seizure = "HP:0001250"
    ancestors = view.ancestors(seizure)

    assert seizure in ancestors
    assert not ancestors & UNIVERSAL_NODES
    assert "HP:0000707" in ancestors  # Abnormality of the nervous system (layer 1)
    assert "HP:0012638" in ancestors  # Abnormal nervous system physiology
    assert view.depth(seizure) == 3
    assert view.layer1(seizure) == "HP:0000707"


@pytest.mark.integration
def test_real_ontology_distance_between_a_term_and_its_child(real_hpo_tree):
    """HP:0002069 is a child of HP:0001250, so the undirected distance is one hop."""
    view = OntologyView(real_hpo_tree)
    assert view.undirected_distance("HP:0001250", "HP:0002069") == 1
