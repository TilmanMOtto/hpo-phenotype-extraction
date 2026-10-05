"""Ontology-aware voting: counting jurors over ancestor closures instead of exact identifiers.

The scenario the rule exists for, in the toy ontology::

                        ROOT
                      /      \\
                     A        G
                   /   \\       |
                  B     E      H
                 / \\     \\    /
                C   D      F  /
                 \\____ M ____/

Two jurors agree a finding is present but disagree on specificity, one names ``C``, the other
``D``. Exact voting gives each one vote, so nothing reaches k = 2 and the finding is dropped
entirely. Closure voting gives their shared ancestor ``B`` two votes, so the jury reports the most
specific term it actually agrees on instead of reporting nothing.
"""

from __future__ import annotations

import pytest

from hpo_extraction.phenojury.ensemble_votes import (
    closure_counts,
    closure_vote_from_masks,
    mask_of,
    vote_k_from_masks,
)
from fixtures.toy_ontology import A, B, C, D, E, F, G, H, M

MODELS = ["m0", "m1", "m2"]
ALL = 0b111


def masks_from(per_model: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, int]]:
    """``{model: {report: [terms]}}`` -> the ``{report: {term: mask}}`` contract."""
    out: dict[str, dict[str, int]] = {}
    for i, model in enumerate(MODELS):
        bit = 1 << i
        for report_id, terms in per_model.get(model, {}).items():
            row = out.setdefault(report_id, {})
            for term in terms:
                row[term] = row.get(term, 0) | bit
    return out


class TestTheMotivatingCase:
    """Two jurors agree on the finding, disagree on the specificity."""

    MASKS = masks_from({"m0": {"r1": [C]}, "m1": {"r1": [D]}})

    def test_exact_voting_drops_it_at_k_2(self, toy_view):
        assert vote_k_from_masks(self.MASKS, ALL, 2)["r1"] == set()

    def test_closure_voting_recovers_the_shared_ancestor(self, toy_view):
        got = closure_vote_from_masks(self.MASKS, ALL, MODELS, toy_view, 2)["r1"]
        assert B in got, "both jurors named a descendant of B, so B has two votes"
        assert A in got, "and of A"
        assert C not in got and D not in got, "neither specific term has two votes"

    def test_the_counts_are_per_juror(self, toy_view):
        counts = closure_counts(self.MASKS, ALL, MODELS, toy_view)["r1"]
        assert counts[C] == 1 and counts[D] == 1
        assert counts[B] == 2 and counts[A] == 2


class TestMonotonicity:
    """n_up(u) >= n_up(v) for every ancestor u of v, what makes the accepted set ancestor-closed."""

    def test_counts_never_fall_toward_the_root(self, toy_view):
        masks = masks_from({"m0": {"r1": [C, F]}, "m1": {"r1": [M]}, "m2": {"r1": [H]}})
        counts = closure_counts(masks, ALL, MODELS, toy_view)["r1"]
        for term, n in counts.items():
            for ancestor in toy_view.ancestors(term):
                assert counts.get(ancestor, 0) >= n, f"{ancestor} above {term}"

    def test_the_accepted_set_is_ancestor_closed(self, toy_view):
        masks = masks_from({"m0": {"r1": [C]}, "m1": {"r1": [D]}, "m2": {"r1": [M]}})
        accepted = closure_vote_from_masks(masks, ALL, MODELS, toy_view, 2)["r1"]
        for term in accepted:
            assert toy_view.ancestors(term) <= accepted | {term}


class TestAgreementWithExactVoting:
    def test_k_1_covers_the_exact_union(self, toy_view):
        """At k = 1 the closure rule adds ancestors but loses nothing the exact rule found."""
        masks = masks_from({"m0": {"r1": [C, F]}, "m1": {"r1": [M]}})
        exact = vote_k_from_masks(masks, ALL, 1)["r1"]
        closure = closure_vote_from_masks(masks, ALL, MODELS, toy_view, 1)["r1"]
        assert exact <= closure

    def test_unanimity_on_one_term_agrees(self, toy_view):
        masks = masks_from({m: {"r1": [C]} for m in MODELS})
        assert vote_k_from_masks(masks, ALL, 3)["r1"] == {C}
        assert C in closure_vote_from_masks(masks, ALL, MODELS, toy_view, 3)["r1"]

    def test_closure_is_never_smaller_at_the_same_k(self, toy_view):
        """A juror naming v supports every ancestor of v, so no term can lose votes."""
        masks = masks_from({"m0": {"r1": [C, F]}, "m1": {"r1": [D, H]}, "m2": {"r1": [M]}})
        for k in (1, 2, 3):
            exact = vote_k_from_masks(masks, ALL, k)["r1"]
            closure = closure_vote_from_masks(masks, ALL, MODELS, toy_view, k)["r1"]
            assert exact <= closure, f"k={k}"


class TestOneVotePerJuror:
    def test_naming_several_descendants_still_counts_once(self, toy_view):
        """Otherwise a verbose juror could carry a term to threshold single-handed."""
        masks = masks_from({"m0": {"r1": [C, D, M]}})
        counts = closure_counts(masks, ALL, MODELS, toy_view)["r1"]
        assert counts[B] == 1, "one juror, one vote for B, despite three descendants named"
        assert counts[A] == 1

    def test_a_lone_verbose_juror_cannot_reach_k_2(self, toy_view):
        masks = masks_from({"m0": {"r1": [C, D, M, F]}})
        assert closure_vote_from_masks(masks, ALL, MODELS, toy_view, 2)["r1"] == set()


class TestSubsetRestriction:
    def test_only_jurors_in_the_subset_vote(self, toy_view):
        masks = masks_from({"m0": {"r1": [C]}, "m1": {"r1": [D]}, "m2": {"r1": [C]}})
        only_m0 = mask_of(["m0"], MODELS)
        counts = closure_counts(masks, only_m0, MODELS, toy_view)["r1"]
        assert counts[C] == 1
        assert counts[B] == 1, "m1's D must not contribute"

    def test_an_empty_subset_yields_no_votes(self, toy_view):
        masks = masks_from({"m0": {"r1": [C]}})
        assert closure_counts(masks, 0, MODELS, toy_view)["r1"] == {}


class TestEdgeCases:
    def test_multi_parent_term_supports_both_branches(self, toy_view):
        """M sits under B and H, so naming it votes for A's branch and G's at once."""
        masks = masks_from({"m0": {"r1": [M]}})
        counts = closure_counts(masks, ALL, MODELS, toy_view)["r1"]
        assert counts[B] == 1 and counts[A] == 1
        assert counts[H] == 1 and counts[G] == 1

    def test_unrelated_findings_do_not_reinforce(self, toy_view):
        """C is under A, H under G. They share only the excluded root, so nothing reaches 2."""
        masks = masks_from({"m0": {"r1": [C]}, "m1": {"r1": [H]}})
        assert closure_vote_from_masks(masks, ALL, MODELS, toy_view, 2)["r1"] == set()

    def test_a_report_nobody_voted_on(self, toy_view):
        masks = {"r1": {}}
        assert closure_counts(masks, ALL, MODELS, toy_view) == {"r1": {}}
        assert closure_vote_from_masks(masks, ALL, MODELS, toy_view, 1) == {"r1": set()}

    def test_unresolvable_ids_contribute_nothing(self, toy_view):
        """A juror naming a code the release does not have cannot vote for anything."""
        masks = masks_from({"m0": {"r1": ["HP:9999999"]}})
        assert closure_counts(masks, ALL, MODELS, toy_view)["r1"] == {}

    def test_every_report_appears_in_the_output(self, toy_view):
        masks = masks_from({"m0": {"r1": [C], "r2": [F]}})
        got = closure_vote_from_masks(masks, ALL, MODELS, toy_view, 1)
        assert set(got) == {"r1", "r2"}
