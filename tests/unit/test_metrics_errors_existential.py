"""The thesis false-positive taxonomy: existential precedence and the ``same_branch`` bucket.

Run against ``tests/fixtures/toy_ontology``::

                        ROOT
                      /      \\
                     A        G          depth 1, the two organ-system roots
                   /   \\       |
                  B     E      H         depth 2
                 / \\     \\    /
                C   D      F  /          depth 3
                 \\____ M ____/           M has TWO parents, B and H

Layer-1 membership, which is what ``same_branch`` keys on:
``A`` holds B, C, D, E, F and M; ``G`` holds H. (``M`` is reachable from both, and ``layer1``
resolves it to ``A``.)
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    THESIS_BUCKETS,
    classify_false_positive,
    classify_false_positive_existential,
    error_taxonomy_existential,
)
from fixtures.toy_ontology import (
    A, B, C, D, E, F, G, H, M,
    HALLUCINATION, OUT_OF_SUBTREE,
)


class TestBuckets:
    def test_strict_ancestor_of_some_annotated_term(self, toy_view):
        bucket, witness = classify_false_positive_existential(B, {C}, toy_view)
        assert (bucket, witness) == ("ancestor", C)

    def test_strict_descendant_of_some_annotated_term(self, toy_view):
        bucket, witness = classify_false_positive_existential(C, {B}, toy_view)
        assert (bucket, witness) == ("descendant", B)

    def test_sibling_shares_a_direct_parent(self, toy_view):
        bucket, witness = classify_false_positive_existential(D, {C}, toy_view)
        assert (bucket, witness) == ("sibling", C), "C and D are both children of B"

    def test_same_branch_shares_only_an_organ_system_root(self, toy_view):
        """The bucket the nearest-ground truth taxonomy does not have.

        C sits under B, F under E, both under A. They are not ancestors, descendants or siblings of
        each other, but "wrong finding, right organ system" is not the same error as "wrong organ
        system", and this is where that distinction lives.
        """
        bucket, witness = classify_false_positive_existential(C, {F}, toy_view)
        assert (bucket, witness) == ("same_branch", F)

    def test_unrelated_crosses_organ_systems(self, toy_view):
        bucket, witness = classify_false_positive_existential(C, {H}, toy_view)
        assert bucket == "unrelated"
        assert witness is None

    def test_invalid_for_a_code_the_release_does_not_have(self, toy_view):
        assert classify_false_positive_existential(HALLUCINATION, {C}, toy_view) == ("invalid", None)

    def test_no_gold_when_the_report_has_none(self, toy_view):
        assert classify_false_positive_existential(C, set(), toy_view) == ("no_gold", None)

    def test_out_of_subtree_is_invalid_here(self, toy_view):
        """Normalisation discards these upstream. If one arrives it is not a scorable term."""
        bucket, _ = classify_false_positive_existential(OUT_OF_SUBTREE, {C}, toy_view)
        assert bucket == "invalid"

    def test_no_gold_wins_over_invalid_is_not_the_order(self, toy_view):
        """``invalid`` is checked first: a fabricated code is fabricated regardless of the ground truth."""
        assert classify_false_positive_existential(HALLUCINATION, set(), toy_view)[0] == "invalid"

    def test_every_bucket_is_declared(self, toy_view):
        seen = {
            classify_false_positive_existential(p, g, toy_view)[0]
            for p, g in [
                (B, {C}), (C, {B}), (D, {C}), (C, {F}), (C, {H}),
                (HALLUCINATION, {C}), (C, set()),
            ]
        }
        assert seen == set(THESIS_BUCKETS)


class TestPrecedence:
    def test_ancestor_beats_descendant_when_both_hold(self, toy_view):
        """The divergence that counts most, and the reason this module exists.

        Predict B with ground truth ``{A, C}``. B is a *descendant* of A and an *ancestor* of C, both are
        true at once. The nearest-ground truth rule picks A (distance 1, against C's 1, broken by id) and
        files the prediction as "too specific". The existential rule checks ancestry first and files
        it as "too general", which is what a traversal accepting the chain A, B while the ground truth is C
        is actually doing.
        """
        assert classify_false_positive(B, {A, C}, toy_view)[0] == "descendant"
        assert classify_false_positive_existential(B, {A, C}, toy_view) == ("ancestor", C)

    def test_sibling_beats_same_branch(self, toy_view):
        """C is a sibling of D and same-branch with F. The stronger relation wins."""
        bucket, witness = classify_false_positive_existential(C, {D, F}, toy_view)
        assert (bucket, witness) == ("sibling", D)

    def test_same_branch_beats_unrelated(self, toy_view):
        bucket, witness = classify_false_positive_existential(C, {F, H}, toy_view)
        assert (bucket, witness) == ("same_branch", F)

    def test_witness_is_deterministic_under_ties(self, toy_view):
        """Two annotated terms could each earn the bucket. The sorted-first one is always chosen."""
        first = classify_false_positive_existential(F, {C, D}, toy_view)
        for _ in range(5):
            assert classify_false_positive_existential(F, {D, C}, toy_view) == first
        assert first == ("same_branch", min(C, D))


class TestDivergenceFromNearestGold:
    @pytest.mark.parametrize("pred,gold,nearest,existential", [
        (B, {F}, "unrelated", "same_branch"),
        (B, {A, C}, "descendant", "ancestor"),
        (E, {A, F}, "descendant", "ancestor"),
        (H, {G, M}, "descendant", "ancestor"),
        (M, {E}, "unrelated", "same_branch"),
    ])
    def test_the_two_rules_disagree_as_documented(
        self, toy_view, pred, gold, nearest, existential
    ):
        assert classify_false_positive(pred, gold, toy_view)[0] == nearest
        assert classify_false_positive_existential(pred, gold, toy_view)[0] == existential

    def test_the_existential_rule_is_never_less_related(self, toy_view):
        """Its whole point: it cannot file something as further away than the nearest-ground truth rule."""
        rank = {b: i for i, b in enumerate(
            ["ancestor", "descendant", "sibling", "same_branch", "unrelated"])}
        for pred in (A, B, C, D, E, F, G, H, M):
            for gold in ({C}, {F}, {C, F}, {A, C}, {G, M}, {C, H}):
                if pred in gold:
                    continue
                near = classify_false_positive(pred, gold, toy_view)[0]
                exist = classify_false_positive_existential(pred, gold, toy_view)[0]
                if near in rank and exist in rank:
                    assert rank[exist] <= rank[near], (pred, gold, near, exist)


class TestCohortTaxonomy:
    def test_counts_and_fractions_sum(self, toy_view):
        gold_sets = [{C}, {F}, set()]
        pred_sets = [{B, D, H}, {C}, {C}]
        got = error_taxonomy_existential(gold_sets, pred_sets, toy_view)
        assert got["n_fp"] == 5
        assert sum(got["counts"].values()) == 5
        assert sum(got["fractions"].values()) == pytest.approx(1.0)

    def test_true_positives_are_not_counted(self, toy_view):
        got = error_taxonomy_existential([{C}], [{C}], toy_view)
        assert got["n_fp"] == 0
        assert all(v == 0 for v in got["counts"].values())

    def test_buckets_are_as_expected(self, toy_view):
        got = error_taxonomy_existential([{C}], [{B, D, F, H}], toy_view)["counts"]
        assert got["ancestor"] == 1      # B
        assert got["sibling"] == 1       # D
        assert got["same_branch"] == 1   # F
        assert got["unrelated"] == 1     # H

    def test_severity_travels_with_the_decomposition(self, toy_view):
        """Paired here so a caller cannot mix one rule's buckets with another rule's distances."""
        got = error_taxonomy_existential([{C}], [{B, H}], toy_view)
        assert got["n_distance_measurable"] == 2
        assert got["distance_histogram"] == {1: 1, 3: 1}, "dist(B,C)=1, dist(H,C)=3 via M"
        assert got["distance_mean"] == pytest.approx(2.0)

    def test_invalid_codes_are_counted_but_have_no_distance(self, toy_view):
        got = error_taxonomy_existential([{C}], [{HALLUCINATION}], toy_view)
        assert got["counts"]["invalid"] == 1
        assert got["n_distance_measurable"] == 0
        assert got["n_distance_unmeasurable"] == 1

    def test_empty_cohort(self, toy_view):
        got = error_taxonomy_existential([], [], toy_view)
        assert got["n_fp"] == 0
        assert got["fractions"] == {b: 0.0 for b in THESIS_BUCKETS}
