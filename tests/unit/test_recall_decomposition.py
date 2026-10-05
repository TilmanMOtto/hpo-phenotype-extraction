"""§Recall decomposition, attribution of every false negative to the stage that lost it.

Worked by hand on ``tests/fixtures/toy_ontology``, whose shape is::

    A (layer-1) -> B -> {C, D}, B -> M ; A -> E -> F ; G (layer-1) -> H -> M

The properties that matter are the ones a wrong implementation would break quietly: the five
buckets partition the false negatives, the coverage identity of §Diagnostics falls out of
the same rows, a pruned term is attributed to its *deepest* blocker rather than any blocker, and the
retrieval bucket refuses to report a number when it has no evidence segments to decide with.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpo_extraction.treephenorag.stored_scores import ReportCache
from hpo_extraction.evaluation.metrics.recall_decomposition import (
    DECOMPOSITION_BUCKETS,
    attribute_false_negative,
    delta_m_sensitivity,
    recall_decomposition,
)
from fixtures.toy_ontology import A, B, C, D, E, F, H, M

pytestmark = pytest.mark.unit


def cache_for(margins_by_node, sents_by_node=None):
    """A :class:`ReportCache` over ``{hpo: [margins]}``, padded and masked."""
    nodes = sorted(margins_by_node)
    width = max((len(v) for v in margins_by_node.values()), default=1)
    margins = np.zeros((len(nodes), width))
    mask = np.zeros((len(nodes), width), dtype=bool)
    sent = np.full((len(nodes), width), -1, dtype=np.int32)
    for i, h in enumerate(nodes):
        bag = margins_by_node[h]
        margins[i, :len(bag)] = bag
        mask[i, :len(bag)] = True
        if sents_by_node and h in sents_by_node:
            got = sents_by_node[h]
            sent[i, :len(got)] = got
    return ReportCache("r1", nodes, margins, mask, sent_index=sent)


class TestAttribution:
    def test_a_scored_term_is_attributed_to_itself(self, toy_view):
        got = attribute_false_negative(C, scored={A, B, C}, expanded={A, B}, view=toy_view)
        assert got == {"term": C, "lost_at": C, "pruned": False, "blocking_depth": None}

    def test_a_blocked_term_is_attributed_to_its_blocking_ancestor(self, toy_view):
        """C was never scored because B was scored and not expanded."""
        got = attribute_false_negative(C, scored={A, B}, expanded={A}, view=toy_view)
        assert got["pruned"] is True
        assert got["lost_at"] == B
        assert got["blocking_depth"] == 2

    def test_the_deepest_blocker_wins(self, toy_view):
        """A and B both block C; B is deeper, and it is how far the traversal actually got."""
        got = attribute_false_negative(C, scored={A, B}, expanded=set(), view=toy_view)
        assert got["lost_at"] == B
        assert got["blocking_depth"] == 2

    def test_a_term_with_no_blocker_is_not_filed_as_pruned_at_a_node(self, toy_view):
        """Unreachable in the graph is a different failure from blocked at a term."""
        got = attribute_false_negative(C, scored=set(), expanded=set(), view=toy_view)
        assert got["lost_at"] is None


class TestBuckets:
    def test_judgement_miss_when_every_margin_is_negative(self, toy_view):
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [-4.0, -2.0]})}, toy_view,
        )
        assert out["counts"]["judgement"] == 1
        assert out["counts"]["pruning"] == 0

    def test_pooling_miss_when_the_verifier_said_yes_but_the_score_did_not_clear(self, toy_view):
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [3.0, -1.0]})}, toy_view,
            pooled_by_report={"r1": {C: 0.4}}, threshold=0.9,
        )
        assert out["counts"]["pooling"] == 1

    def test_residual_when_the_score_did_clear(self, toy_view):
        """Accepted-by-score but absent from the prediction set: a descendant took its place."""
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [3.0]})}, toy_view,
            pooled_by_report={"r1": {C: 0.99}}, threshold=0.9,
        )
        assert out["counts"]["residual"] == 1

    def test_pruning_reports_the_cause_at_the_blocker_not_at_the_annotated_term(self, toy_view):
        """C was never called. The reason is B's margins, and describing C's would be fiction."""
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B}}, {"r1": {A}},
            {"r1": cache_for({A: [1.0], B: [-5.0]})}, toy_view,
        )
        assert out["counts"]["pruning"] == 1
        assert out["blocking_causes"]["judgement"] == 1
        assert out["blocking_depth_histogram"] == {2: 1}


class TestRetrievalNeedsEvidence:
    def test_with_evidence_segments_a_missed_sentence_is_a_retrieval_miss(self, toy_view):
        """The verifier was confident, but never saw the sentence the curator pointed at."""
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [4.0, 3.0]},
                             sents_by_node={C: [7, 8]})}, toy_view,
            evidence={"r1": {C: [42]}},
        )
        assert out["counts"]["retrieval"] == 1
        assert out["evidence_available"] is True

    def test_the_same_term_is_a_judgement_miss_once_the_sentence_was_retrieved(self, toy_view):
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [-4.0, -3.0]},
                             sents_by_node={C: [42, 8]})}, toy_view,
            evidence={"r1": {C: [42]}},
        )
        assert out["counts"]["retrieval"] == 0
        assert out["counts"]["judgement"] == 1

    def test_without_evidence_the_flag_says_the_bucket_is_not_quotable(self, toy_view):
        out = recall_decomposition(
            {"r1": [C]}, {"r1": []}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [-4.0]})}, toy_view,
        )
        assert out["evidence_available"] is False


class TestPartitionAndIdentities:
    def test_the_five_buckets_partition_every_false_negative(self, toy_view):
        gold = [C, D, F, M]
        out = recall_decomposition(
            {"r1": gold}, {"r1": [D]}, {"r1": {A, B, C, D, E}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [-2.0], D: [2.0], E: [-1.0]})}, toy_view,
        )
        assert sum(out["counts"].values()) == out["n_false_negative"] == 3   # C, F, M missed
        assert set(out["counts"]) == set(DECOMPOSITION_BUCKETS)
        assert sum(out["fractions"].values()) == pytest.approx(1.0)

    def test_the_coverage_identity_falls_out_of_the_same_rows(self, toy_view):
        """1 - Cov is the pruned share and Cov - R_mu the rest, over the same ground truth denominator."""
        gold = [C, D, F]
        scored = {A, B, C, D, E}
        out = recall_decomposition(
            {"r1": gold}, {"r1": [D]}, {"r1": scored}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [-2.0], D: [2.0], E: [-1.0]})}, toy_view,
        )
        coverage = len(set(gold) & scored) / len(gold)
        recall = 1 / len(gold)
        assert out["coverage_loss"] == pytest.approx(1 - coverage)
        assert out["loss_at_scored_terms"] == pytest.approx(coverage - recall)

    def test_a_predicted_annotated_term_is_not_a_false_negative(self, toy_view):
        out = recall_decomposition(
            {"r1": [C]}, {"r1": [C]}, {"r1": {A, B, C}}, {"r1": {A, B}},
            {"r1": cache_for({A: [1.0], B: [1.0], C: [3.0]})}, toy_view,
        )
        assert out["n_false_negative"] == 0
        assert out["n_gold"] == 1


class TestDeltaMSensitivity:
    def test_the_sweep_moves_terms_between_judgement_and_pooling(self, toy_view):
        kwargs = dict(
            gold_by_report={"r1": [C]}, predicted_by_report={"r1": []},
            scored_by_report={"r1": {A, B, C}}, expanded_by_report={"r1": {A, B}},
            caches={"r1": cache_for({A: [1.0], B: [1.0], C: [-1.0]})}, view=toy_view,
            pooled_by_report={"r1": {C: 0.2}}, threshold=0.9,
        )
        rows = delta_m_sensitivity([0.0, 2.0], **kwargs)
        # margin -1.0: at delta_m = 0 it is <= 0 so judgement. At delta_m = 2 it exceeds -2 so the
        # verdict moves to the pooled score, which did not clear.
        assert rows[0]["judgement"] == 1 and rows[0]["pooling"] == 0
        assert rows[1]["judgement"] == 0 and rows[1]["pooling"] == 1
