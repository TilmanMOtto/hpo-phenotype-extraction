"""The pre-scoring normalisation: what is discarded, what is kept, and what that costs.

Run against ``tests/fixtures/toy_ontology``, which carries one example of each case the pipeline
has to tell apart:

    HP:0009999   an alt id remapping to C          -> resolved, scored as C
    HP:0000005   a real term outside the subtree   -> DISCARDED, counted
    HP:9999999   not in the ontology at all        -> KEPT on the prediction side, scored as an FP

The asymmetry between the last two is the whole point of the module: one is a category error the
task's label space excludes, the other is a system inventing a code.
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    ancestor_closure,
    counts,
    normalise_gold,
    normalise_pair,
    normalise_predictions,
    split_unscorable,
)
from fixtures.toy_ontology import (
    A, B, C, D, G, H, M,
    HALLUCINATION, OBSOLETE_C, OUT_OF_SUBTREE,
)


class TestSplitUnscorable:
    def test_three_way_split(self, toy_view):
        keep, oos, nonexistent = split_unscorable(
            [C, D, OUT_OF_SUBTREE, HALLUCINATION], toy_view)
        assert keep == {C, D}
        assert oos == [OUT_OF_SUBTREE]
        assert nonexistent == [HALLUCINATION]

    def test_obsolete_id_resolves_rather_than_dropping(self, toy_view):
        keep, oos, nonexistent = split_unscorable([OBSOLETE_C], toy_view)
        assert keep == {C}, "an alt id must be remapped, not discarded"
        assert oos == [] and nonexistent == []

    def test_obsolete_and_canonical_collapse_to_one_term(self, toy_view):
        keep, _, _ = split_unscorable([C, OBSOLETE_C], toy_view)
        assert keep == {C}, "the same term named twice is one prediction, not two"


class TestGoldSide:
    def test_out_of_subtree_gold_is_discarded_and_counted(self, toy_view):
        keep, n_dropped = normalise_gold([C, OUT_OF_SUBTREE], toy_view)
        assert keep == {C}
        assert n_dropped == 1

    def test_nonexistent_gold_is_also_discarded(self, toy_view):
        """Keeping it would charge every system a false negative for an annotation error."""
        keep, n_dropped = normalise_gold([C, HALLUCINATION], toy_view)
        assert keep == {C}
        assert n_dropped == 1


class TestPredictionSide:
    def test_out_of_subtree_prediction_is_discarded_and_counted(self, toy_view):
        scored, n_oos, n_nonexistent = normalise_predictions([C, OUT_OF_SUBTREE], toy_view)
        assert scored == {C}
        assert (n_oos, n_nonexistent) == (1, 0)

    def test_nonexistent_prediction_is_kept_and_counted(self, toy_view):
        scored, n_oos, n_nonexistent = normalise_predictions([C, HALLUCINATION], toy_view)
        assert scored == {C, HALLUCINATION}
        assert (n_oos, n_nonexistent) == (0, 1)

    def test_the_two_unscorable_kinds_are_counted_apart(self, toy_view):
        scored, n_oos, n_nonexistent = normalise_predictions(
            [C, OUT_OF_SUBTREE, HALLUCINATION], toy_view)
        assert scored == {C, HALLUCINATION}
        assert (n_oos, n_nonexistent) == (1, 1)

    def test_kept_nonexistent_scores_as_one_false_positive(self, toy_view):
        gold, pred, _ = normalise_pair([C], [C, HALLUCINATION], toy_view)
        tp, fp, fn = counts(gold, pred)
        assert (tp, fp, fn) == (1, 1, 0)

    def test_discarded_out_of_subtree_costs_no_false_positive(self, toy_view):
        """The policy's whole effect, in one assertion: it raises precision."""
        gold, pred, bookkeeping = normalise_pair([C], [C, OUT_OF_SUBTREE], toy_view)
        tp, fp, fn = counts(gold, pred)
        assert (tp, fp, fn) == (1, 0, 0)
        assert bookkeeping["n_pred_out_of_subtree"] == 1, "but the count must survive"


class TestPairBookkeeping:
    def test_every_column_the_tables_carry(self, toy_view):
        gold, pred, bookkeeping = normalise_pair(
            [C, OUT_OF_SUBTREE],
            [D, OUT_OF_SUBTREE, HALLUCINATION],
            toy_view,
        )
        assert gold == {C}
        assert pred == {D, HALLUCINATION}
        assert bookkeeping == {
            "n_gold_out_of_subtree": 1,
            "n_gold_nonexistent": 0,
            "n_pred_out_of_subtree": 1,
            "n_pred_nonexistent": 1,
        }


class TestIdempotence:
    @pytest.mark.parametrize("raw", [
        [C, D, M],
        [C, OBSOLETE_C, OUT_OF_SUBTREE, HALLUCINATION],
        [],
    ])
    def test_second_pass_changes_nothing(self, toy_view, raw):
        once, _, _ = normalise_predictions(raw, toy_view)
        twice, n_oos, n_nonexistent = normalise_predictions(once, toy_view)
        assert twice == once
        assert n_oos == 0, "nothing unscorable survives the first pass"


class TestNoReductionHappens:
    """The convention this module does not implement.

    A parent and its child predicted together must BOTH survive: the thesis scores unreduced and
    prices the specificity question through an earlier exploratory run instead. If a reduction is ever reinstated,
    these are the assertions that will fail, and the module docstring says why they should not be
    changed lightly.
    """

    def test_ancestor_and_descendant_both_survive(self, toy_view):
        gold, pred, _ = normalise_pair([C], [B, C], toy_view)
        assert pred == {B, C}
        tp, fp, fn = counts(gold, pred)
        assert (tp, fp, fn) == (1, 1, 0), "B is a false positive, not a free redundancy"

    def test_a_full_chain_is_kept_whole(self, toy_view):
        _, pred, _ = normalise_pair([], [A, B, C], toy_view)
        assert pred == {A, B, C}

    def test_gold_keeps_an_annotated_ancestor_descendant_pair(self, toy_view):
        """The 59 curated pairs whose child sits in another segment depend on this."""
        gold, _, _ = normalise_pair([B, C], [], toy_view)
        assert gold == {B, C}


class TestAncestorClosure:
    def test_matches_the_hand_computed_reference(self, toy_view):
        assert ancestor_closure([C], toy_view) == {C, B, A}

    def test_multi_parent_node_contributes_both_paths(self, toy_view):
        assert ancestor_closure([M], toy_view) == {M, B, A, H, G}

    def test_nonexistent_id_contributes_nothing(self, toy_view):
        """The documented asymmetry: an FP in flat metrics, invisible in hierarchical ones."""
        assert ancestor_closure([HALLUCINATION], toy_view) == set()
        assert ancestor_closure([C, HALLUCINATION], toy_view) == {C, B, A}
