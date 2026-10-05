"""Unit tests for hpo_extraction.evaluation.tree_pruning.

The pure pruning/sweep functions do not need a real HPOTree, they operate on a hand-built
``direct_parents_map`` and per-node confidence dict, so these tests run without the ontology
(nltk) chain. A tiny synthetic tree is used:

    R  (root. Depth-skipped)
    └─ A      (layer 1. Depth-skipped)
        └─ B
            └─ C
                └─ D

``direct_parents_map`` gives each node its single parent (R/A carry no scored parents because
they are depth-skipped and never evaluated).
"""

from __future__ import annotations

from pathlib import Path


from hpo_extraction.evaluation.tree_pruning import (  # noqa: E402
    apply_pruning_verdict,
    should_prune_confident,
    sweep_thresholds,
)

R, A, B, C, D = "R", "A", "B", "C", "D"
PARENTS = {R: [], A: [], B: [A], C: [B], D: [C]}
TOPO = [R, A, B, C, D]
SKIP = {R, A}


# ── should_prune_confident: the an earlier exploratory run cases already hand-validated ──────────

def test_prune_when_two_consecutive_confidently_absent():
    # C's parent B and grandparent A both confidently absent -> prune C.
    node_conf = {B: 0.05, C: 0.9}
    node_pred = {A: "no", B: "no"}  # A treated via conf; B low
    node_conf[A] = 0.05
    assert should_prune_confident(C, PARENTS, node_conf, node_pred, tau_prune=0.1) is True


def test_no_prune_when_parent_only_moderately_absent():
    # B at 0.30 is above tau_prune=0.1 -> NOT confidently absent -> C survives.
    node_conf = {A: 0.05, B: 0.30}
    node_pred = {A: "no", B: "no"}
    assert should_prune_confident(C, PARENTS, node_conf, node_pred, tau_prune=0.1) is False


def test_no_prune_when_grandparent_high():
    # A high, B low -> only one confidently-absent ancestor -> no prune.
    node_conf = {A: 0.8, B: 0.05}
    node_pred = {A: "yes", B: "no"}
    assert should_prune_confident(C, PARENTS, node_conf, node_pred, tau_prune=0.1) is False


def test_prune_propagates_through_pruned_parent():
    node_conf = {B: 0.05}
    node_pred = {A: "no", B: "pruned"}
    assert should_prune_confident(C, PARENTS, node_conf, node_pred, tau_prune=0.1) is True


# ── apply_pruning_verdict == inline an earlier exploratory run walk (correctness backbone) ───────

def _inline_walk(node_conf, tau_accept, tau_prune):
    """Reference implementation mirroring an earlier exploratory run's inline loop: pruned nodes are NOT scored."""
    node_pred: dict[str, str] = {}
    inline_conf: dict[str, float] = {}
    for hpo_id in TOPO:
        if hpo_id in SKIP:
            node_pred[hpo_id] = "skipped"
            continue
        if should_prune_confident(hpo_id, PARENTS, inline_conf, node_pred, tau_prune):
            node_pred[hpo_id] = "pruned"
            continue
        conf = node_conf[hpo_id]
        inline_conf[hpo_id] = conf  # only scored nodes get a conf inline
        node_pred[hpo_id] = "yes" if conf >= tau_accept else "no"
    return node_pred


def test_posthoc_equals_inline_across_grid():
    # Full (unpruned) confidence table for every scored node.
    full_conf = {B: 0.05, C: 0.9, D: 0.95}
    for ta in (0.3, 0.5, 0.7):
        for tp in (0.01, 0.1, 0.2):
            _, posthoc_pred = apply_pruning_verdict(TOPO, PARENTS, SKIP, full_conf, ta, tp)
            inline_pred = _inline_walk(full_conf, ta, tp)
            assert posthoc_pred == inline_pred, f"mismatch at ta={ta} tp={tp}"


def test_posthoc_prunes_subtree_under_confident_absent_chain():
    # B and (skipped) A: B<tp; A is skipped so not "low". Need B + its parent both low.
    # Here make B low and its parent A NOT scored (skipped) -> C should NOT prune on A/B alone.
    full_conf = {B: 0.02, C: 0.9, D: 0.9}
    _, pred = apply_pruning_verdict(TOPO, PARENTS, SKIP, full_conf, tau_accept=0.5, tau_prune=0.1)
    # A is skipped (not low), B low but only one confidently-absent ancestor for C -> C evaluated.
    assert pred[B] == "no"
    assert pred[C] == "yes"


# ── sweep_thresholds monotonicity, with a fake dataset ────────────────────────

class _FakeDataset:
    """Minimal stand-in for HCYDataset.evaluate, returns trivial metrics, records nothing."""

    def evaluate(self, response_dict, targets, output_dir=None, filter_unannotated=False):
        return {
            "micro_precision": 1.0, "micro_recall": 1.0, "micro_f1": 1.0,
            "macro_precision": 1.0, "macro_recall": 1.0, "macro_f1": 1.0,
        }


def test_sweep_monotonicity():
    node_conf_by_patient = {
        "p1": {B: 0.6, C: 0.4, D: 0.2},
        "p2": {B: 0.05, C: 0.7, D: 0.9},
    }
    rows = sweep_thresholds(
        node_conf_by_patient, TOPO, PARENTS, SKIP, _FakeDataset(),
        original_targets=[B, C, D], expanded_targets=None,
        tau_accept_grid=[0.3, 0.5, 0.7], tau_prune_grid=[0.01, 0.1, 0.3],
    )
    by_key = {(r["tau_accept"], r["tau_prune"]): r for r in rows}
    # Raising tau_accept never increases positives (fixed tau_prune).
    for tp in (0.01, 0.1, 0.3):
        pos = [by_key[(ta, tp)]["n_positive"] for ta in (0.3, 0.5, 0.7)]
        assert pos == sorted(pos, reverse=True)
    # Raising tau_prune never decreases pruned count (fixed tau_accept).
    for ta in (0.3, 0.5, 0.7):
        pruned = [by_key[(ta, tp)]["n_pruned"] for tp in (0.01, 0.1, 0.3)]
        assert pruned == sorted(pruned)
