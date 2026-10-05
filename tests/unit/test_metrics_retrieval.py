"""§Core extraction quality, retrieval stage, eq. (2) at both units.

Segment-level uses a five-segment report. Term-level uses a four-term candidate list. Both are
small enough to count the intersections by eye.
"""

from __future__ import annotations

import pytest

from hpo_extraction.evaluation.metrics import (
    candidate_set_recall,
    combined_recall_bound,
    relevant_segments,
    segment_pr_at_s,
    segment_pr_curve,
    term_pr_at_m,
    term_pr_curve,
)
from fixtures.toy_ontology import A, B, C, D, F, M

pytestmark = pytest.mark.unit


# ── The true-path relevance rule ─────────────────────────────────────────────

def test_relevance_propagates_annotations_up_the_ontology(toy_view):
    """Segments annotated with C or D evidence their ancestor B, that is the true-path rule.

    "Without propagating annotations up the ontology, every ancestor query would look like a miss
    by design."
    """
    annotations = {0: [C], 1: [F], 2: [D]}
    assert relevant_segments(annotations, B, toy_view) == {0, 2}


def test_relevance_at_the_organ_system_covers_its_whole_subtree(toy_view):
    annotations = {0: [C], 1: [F], 2: [D]}
    assert relevant_segments(annotations, A, toy_view) == {0, 1, 2}


def test_relevance_of_a_leaf_is_only_its_own_annotations(toy_view):
    annotations = {0: [C], 1: [F], 2: [D]}
    assert relevant_segments(annotations, C, toy_view) == {0}


def test_relevance_is_empty_for_an_unresolvable_query(toy_view):
    assert relevant_segments({0: [C]}, "HP:9999999", toy_view) == set()


# ── Eq. (2), segment level ───────────────────────────────────────────────────

def test_segment_pr_at_s_hand_worked():
    """Ranked [s0..s4], relevant {s1, s4}. At S=3 one of two relevant segments is retrieved."""
    ranked = ["s0", "s1", "s2", "s3", "s4"]
    relevant = {"s1", "s4"}
    precision, recall, hits = segment_pr_at_s(ranked, relevant, 3)
    assert hits == 1
    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(0.5)


def test_segment_recall_reaches_one_when_s_covers_every_relevant_segment():
    ranked = ["s0", "s1", "s2", "s3", "s4"]
    precision, recall, hits = segment_pr_at_s(ranked, {"s1", "s4"}, 5)
    assert (hits, precision, recall) == (2, pytest.approx(0.4), pytest.approx(1.0))


def test_recall_is_none_rather_than_zero_when_there_is_nothing_to_find():
    """Averaging a zero here would depress the very ceiling this metric measures."""
    _, recall, _ = segment_pr_at_s(["s0"], set(), 3)
    assert recall is None


def test_precision_divides_by_s_even_when_fewer_segments_were_returned():
    """The literal reading of eq. (2), and the conservative one, flagged via ``n_pairs_short``."""
    precision, _, hits = segment_pr_at_s(["s0"], {"s0"}, 5)
    assert hits == 1
    assert precision == pytest.approx(1 / 5)


def test_segment_pr_at_s_rejects_a_nonpositive_depth():
    with pytest.raises(ValueError, match="positive"):
        segment_pr_at_s(["s0"], {"s0"}, 0)


def test_segment_pr_curve_micro_and_macro_hand_worked():
    """Two pairs over the same three segments.

    pair 1  ranked [s0,s1,s2]  relevant {s1}      at S=2: 1 hit, P=1/2, R=1
    pair 2  ranked [s0,s1,s2]  relevant {s0,s2}   at S=2: 1 hit, P=1/2, R=1/2

    micro P = 2 hits / (2 pairs * S=2) = 1/2. Micro R = 2 hits / 3 relevant = 2/3
    macro P = 1/2. Macro R = (1 + 1/2)/2 = 3/4
    """
    pairs = [(["s0", "s1", "s2"], {"s1"}), (["s0", "s1", "s2"], {"s0", "s2"})]
    out = segment_pr_curve(pairs, [2])[2]
    assert out["micro_precision"] == pytest.approx(0.5)
    assert out["micro_recall"] == pytest.approx(2 / 3)
    assert out["macro_precision"] == pytest.approx(0.5)
    assert out["macro_recall"] == pytest.approx(0.75)
    assert out["n_pairs"] == 2
    assert out["n_pairs_with_relevant"] == 2
    assert out["n_pairs_short"] == 0


def test_segment_pr_curve_flags_short_reports():
    out = segment_pr_curve([(["s0"], {"s0"})], [5])[5]
    assert out["n_pairs_short"] == 1


def test_segment_recall_is_monotone_in_s():
    """More retrieved segments can only find more relevant ones."""
    pairs = [(["s0", "s1", "s2", "s3"], {"s2"})]
    curve = segment_pr_curve(pairs, [1, 2, 3, 4])
    recalls = [curve[s]["macro_recall"] for s in (1, 2, 3, 4)]
    assert recalls == sorted(recalls)


# ── Eq. (2), term level ──────────────────────────────────────────────────────

def test_term_pr_at_m_hand_worked():
    """Candidates [C, D, F, M], ground truth {C, M}. At M=2 only C is in the list."""
    precision, recall, hits = term_pr_at_m([C, D, F, M], {C, M}, 2)
    assert hits == 1
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_term_recall_at_full_list_finds_everything():
    _, recall, _ = term_pr_at_m([C, D, F, M], {C, M}, 4)
    assert recall == pytest.approx(1.0)


def test_term_recall_is_none_for_a_report_without_gold():
    _, recall, _ = term_pr_at_m([C, D], set(), 2)
    assert recall is None


def test_term_pr_curve_micro_and_macro():
    """Two reports: [C,D,F,M] ground truth {C,M}; [D,C,F,M] ground truth {C}. At M=2 both find one."""
    reports = [([C, D, F, M], {C, M}), ([D, C, F, M], {C})]
    out = term_pr_curve(reports, [2])[2]
    assert out["micro_precision"] == pytest.approx(2 / 4)
    assert out["micro_recall"] == pytest.approx(2 / 3)
    assert out["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert out["n_reports_with_gold"] == 2


def test_term_pr_at_m_rejects_a_nonpositive_list_size():
    with pytest.raises(ValueError, match="positive"):
        term_pr_at_m([C], {C}, 0)


def test_candidate_set_recall_for_an_unranked_visited_set():
    """The tree traversal has no global ranking, so R@M collapses to plain set recall."""
    assert candidate_set_recall({C, D, B}, {C, M}) == pytest.approx(0.5)
    assert candidate_set_recall({C}, set()) is None


# ── The composed bound ───────────────────────────────────────────────────────

def test_combined_recall_bound_is_the_product():
    assert combined_recall_bound(0.9, 0.95) == pytest.approx(0.855)


def test_combined_recall_bound_is_never_above_either_factor():
    bound = combined_recall_bound(0.9, 0.95)
    assert bound <= 0.9 and bound <= 0.95


def test_combined_recall_bound_rejects_values_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        combined_recall_bound(1.2, 0.9)
    with pytest.raises(ValueError, match="reachability_recall"):
        combined_recall_bound(0.9, -0.1)
