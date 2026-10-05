"""§Severity of near-misses, eq. (5), and the Garcia et al. false-positive taxonomy.

The toy DAG's distances (from ``tests/fixtures/toy_ontology``)::

    dist(B, C) = 1   parent/child
    dist(C, D) = 2   siblings under B
    dist(C, H) = 3   via M's second parent, not 4 via the tree route
    dist(C, F) = 4   C-B-A-E-F
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    FN_BUCKETS,
    GARCIA_BUCKETS,
    classify_false_negative,
    classify_false_positive,
    error_taxonomy,
    false_negative_taxonomy,
    near_miss_distribution,
    nearest_gold,
    nearest_gold_distance,
)
from fixtures.toy_ontology import (
    A,
    B,
    C,
    D,
    F,
    HALLUCINATION,
    H,
    M,
    OUT_OF_SUBTREE,
)

pytestmark = pytest.mark.unit


# ── Eq. (5) ──────────────────────────────────────────────────────────────────

def test_nearest_gold_distance_picks_the_minimum(toy_view):
    """B is one hop from C and three from F, so d(B, {C, F}) = 1."""
    assert nearest_gold_distance(B, {C, F}, toy_view) == 1


def test_nearest_gold_returns_the_reference_term(toy_view):
    assert nearest_gold(B, {C, F}, toy_view) == (C, 1)


def test_nearest_gold_breaks_ties_lexicographically(toy_view):
    """F is four hops from both C and D. The smaller id wins, deterministically."""
    reference, distance = nearest_gold(F, {C, D}, toy_view)
    assert distance == 4
    assert reference == min(C, D)


def test_nearest_gold_distance_is_none_without_a_usable_reference(toy_view):
    assert nearest_gold_distance(HALLUCINATION, {C}, toy_view) is None
    assert nearest_gold_distance(C, set(), toy_view) is None


def test_near_miss_distribution_hand_worked(toy_view):
    """ground truth={C}, pred={B, F}: two false positives at distances 1 and 4."""
    out = near_miss_distribution([{C}], [{B, F}], toy_view)
    assert out["n_fp"] == 2
    assert out["histogram"] == {1: 1, 4: 1}
    assert out["mean"] == pytest.approx(2.5)
    assert out["median"] == pytest.approx(2.5)
    assert out["fraction_within_2"] == pytest.approx(0.5)


def test_near_miss_distribution_excludes_true_positives(toy_view):
    """C is predicted and ground truth, so only B counts as a false positive."""
    out = near_miss_distribution([{C}], [{C, B}], toy_view)
    assert out["n_fp"] == 1
    assert out["histogram"] == {1: 1}


def test_near_miss_distribution_counts_unmeasurable_fps_separately(toy_view):
    """A hallucination has no ontological distance. It is not silently given a large one."""
    out = near_miss_distribution([{C}], [{HALLUCINATION}], toy_view)
    assert out["n_fp"] == 1
    assert out["n_unmeasurable"] == 1
    assert out["n_measurable"] == 0


# ── The taxonomy ─────────────────────────────────────────────────────────────

def test_ancestor_bucket(toy_view):
    """Predicting B when C is ground truth: too general."""
    assert classify_false_positive(B, {C}, toy_view) == ("ancestor", C)


def test_descendant_bucket(toy_view):
    """Predicting C when B is ground truth: too specific."""
    assert classify_false_positive(C, {B}, toy_view) == ("descendant", B)


def test_sibling_bucket_requires_a_shared_direct_parent(toy_view):
    """D and C are both direct children of B."""
    assert classify_false_positive(D, {C}, toy_view) == ("sibling", C)


def test_multi_parent_term_is_a_sibling_via_one_shared_parent(toy_view):
    """M's parents are {B, H}; C's are {B}. One shared direct parent is enough."""
    assert classify_false_positive(M, {C}, toy_view) == ("sibling", C)


def test_unrelated_bucket(toy_view):
    """F and C share only the organ system A, not a direct parent, so, unrelated.

    This is where the literal sibling rule chosen for this thesis differs from the repo's older
    ``tree_error_analysis.classify_error``, which would call this pair "sibling" on the strength of
    a lowest common subsumer at depth 2. The two modules are not interchangeable.
    """
    assert classify_false_positive(F, {C}, toy_view) == ("unrelated", C)


def test_hallucination_bucket(toy_view):
    assert classify_false_positive(HALLUCINATION, {C}, toy_view) == ("hallucination", None)


def test_out_of_subtree_is_not_filed_as_a_hallucination(toy_view):
    """A real HPO term outside the phenotypic-abnormality subtree gets its own hygiene bucket."""
    assert classify_false_positive(OUT_OF_SUBTREE, {C}, toy_view) == ("out_of_subtree", None)


def test_no_gold_bucket_is_not_folded_into_unrelated(toy_view):
    """With no annotated term there is no reference, so "unrelated" would assert more than is known."""
    assert classify_false_positive(C, set(), toy_view) == ("no_gold", None)


def test_precedence_ancestor_beats_sibling(toy_view):
    """B is an ancestor of C and would also share no direct parent, ancestry is checked first."""
    bucket, _ = classify_false_positive(B, {C, D}, toy_view)
    assert bucket == "ancestor"


def test_error_taxonomy_counts_and_renormalises(toy_view):
    """ground truth={C}. Pred={B, D, F, HALLUCINATION} -> ancestor, sibling, unrelated, hallucination."""
    out = error_taxonomy([{C}], [{B, D, F, HALLUCINATION}], toy_view)
    assert out["n_fp"] == 4
    assert out["counts"]["ancestor"] == 1
    assert out["counts"]["sibling"] == 1
    assert out["counts"]["unrelated"] == 1
    assert out["counts"]["hallucination"] == 1
    assert out["fractions"]["ancestor"] == pytest.approx(0.25)
    # All four fall in Garcia buckets here, so the renormalised column matches.
    assert out["garcia_fractions"]["ancestor"] == pytest.approx(0.25)
    assert sum(out["garcia_fractions"].values()) == pytest.approx(1.0)


def test_error_taxonomy_does_not_classify_true_positives(toy_view):
    """C is both predicted and ground truth, so only B is a false positive to be bucketed."""
    out = error_taxonomy([{C}], [{C, B}], toy_view)
    assert out["n_fp"] == 1
    assert out["counts"]["ancestor"] == 1


def test_error_taxonomy_renormalisation_excludes_hygiene_buckets(toy_view):
    """With an out-of-subtree code present, the Garcia column is renormalised over the five."""
    out = error_taxonomy([{C}], [{B, OUT_OF_SUBTREE}], toy_view)
    assert out["n_fp"] == 2
    assert out["n_garcia_classified"] == 1
    assert out["fractions"]["ancestor"] == pytest.approx(0.5)
    assert out["garcia_fractions"]["ancestor"] == pytest.approx(1.0)


def test_every_bucket_key_is_always_present(toy_view):
    """A results table needs a stable set of columns even when a bucket is empty."""
    out = error_taxonomy([{C}], [{B}], toy_view)
    for bucket in GARCIA_BUCKETS:
        assert bucket in out["counts"]
        assert bucket in out["fractions"]


# ── The false-negative mirror ────────────────────────────────────────────────

def test_missed_term_with_its_ancestor_predicted(toy_view):
    assert classify_false_negative(C, {B}, toy_view) == "ancestor_predicted"


def test_missed_term_with_its_descendant_predicted(toy_view):
    assert classify_false_negative(B, {C}, toy_view) == "descendant_predicted"


def test_missed_term_with_a_sibling_predicted(toy_view):
    assert classify_false_negative(C, {D}, toy_view) == "sibling_predicted"


def test_missed_term_with_nothing_near_it(toy_view):
    """A miss with no prediction anywhere near is a detection failure, not a granularity one."""
    assert classify_false_negative(C, {H}, toy_view) == "nothing_near"
    assert classify_false_negative(C, set(), toy_view) == "nothing_near"


def test_false_negative_taxonomy_over_a_cohort(toy_view):
    out = false_negative_taxonomy([{C}, {C}], [{B}, set()], toy_view)
    assert out["n_fn"] == 2
    assert out["counts"]["ancestor_predicted"] == 1
    assert out["counts"]["nothing_near"] == 1
    assert set(out["counts"]) == set(FN_BUCKETS)


def test_false_negative_taxonomy_ignores_recovered_terms(toy_view):
    """An annotated term that was predicted is not a false negative and must not appear."""
    out = false_negative_taxonomy([{C, D}], [{C}], toy_view)
    assert out["n_fn"] == 1
    assert out["counts"]["sibling_predicted"] == 1


def test_an_unresolvable_annotated_term_cannot_be_classified(toy_view):
    """A hallucinated code in the *ground truth* set has no position in the ontology to reason from."""
    assert classify_false_negative(HALLUCINATION, {C}, toy_view) == "nothing_near"


def test_unrelated_organ_system_prediction_is_nothing_near(toy_view):
    """A prediction in a different layer-1 subtree entirely."""
    assert classify_false_negative(C, {A}, toy_view) == "ancestor_predicted"
    assert classify_false_negative(F, {H}, toy_view) == "nothing_near"


# ── nearest_gold: the per-gold-set BFS table against the pairwise definition ──
def _pairwise_nearest(pred, gold, view):
    """The definition nearest_gold used to compute directly: min over ground truth of the undirected
    distance, ties broken by the smallest id."""
    if view.resolve(pred) is None:
        return None, None
    best_code, best_dist = None, None
    for g in sorted({view.resolve(g) for g in gold} - {None}):
        d = view.undirected_distance(pred, g)
        if d is None:
            continue
        if best_dist is None or d < best_dist or (d == best_dist and g < best_code):
            best_code, best_dist = g, d
    return best_code, best_dist


def test_nearest_gold_matches_pairwise_on_every_toy_pair(toy_view):
    nodes = sorted(toy_view.scorable)
    golds = [set(), {A}, {B, C}, {D, H}, {M, F, B}, set(nodes)]
    for gold in golds:
        for pred in nodes + [HALLUCINATION, OUT_OF_SUBTREE]:
            assert nearest_gold(pred, gold, toy_view) == _pairwise_nearest(pred, gold, toy_view), \
                (pred, sorted(gold))


def test_nearest_gold_matches_pairwise_on_the_fixed_ontology(real_hpo_tree):
    """Random (prediction, ground-truth set) draws over the real HPO release, ties included: the fixed
    ontology is what the result-table library runs on, and its multi-parent terms are where a tie-break slips."""
    import random

    from hpo_extraction.evaluation.metrics import OntologyView

    view = OntologyView(real_hpo_tree)
    nodes = sorted(view.scorable)
    rng = random.Random(7)
    for _ in range(60):
        gold = set(rng.sample(nodes, rng.randint(1, 12)))
        # Near predictions (neighbours of annotated terms) exercise ties. Random ones the long paths.
        near = [n for g in gold for n in view.neighbours(g)]
        preds = rng.sample(nodes, 3) + (rng.sample(near, min(3, len(near))) if near else [])
        for pred in preds:
            assert nearest_gold(pred, gold, view) == _pairwise_nearest(pred, gold, view), \
                (pred, sorted(gold))
