"""Reading the earlier artifacts, and reconstructing the two scores no driver persisted.

Every assertion here is against ``tests/fixtures/exp13_output``, whose writers mirror the drivers'
dict literals line by line. A schema change in ``src/core/*_experiment.py`` should surface as a
failure here rather than as a silently empty column in the results.
"""

from __future__ import annotations

import json
import math

import pytest

from fixtures.exp13_output import ACCEPT_SCORE, GOLD, PRUNE_SCORE
from fixtures.toy_ontology import A, B, C, D, E, F, G

pytestmark = pytest.mark.unit


def _preds(results_dir, exp_id, cohort, *parts):
    return results_dir / exp_id / cohort / "/".join(parts)


# ── Predictions ──────────────────────────────────────────────────────────────

def test_predictions_come_from_the_summary_line_not_the_per_term_lines(
        exp13_modules, exp13_results_dir):
    """The summary is the only record of an *empty* prediction set.

    Reading the per-term lines alone would drop the reports the system said nothing about, which
    inflates precision by those reports.
    """
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_predictions.jsonl")
    predicted, gold = exp13_modules.loaders.load_predictions(path)
    assert predicted == {"r1": {C, D}, "r2": {D}}
    assert gold == {"r1": {C, F}, "r2": {D}}


def test_a_report_with_no_predictions_is_kept_with_an_empty_set(exp13_modules, tmp_path):
    from fixtures.exp13_output import write_predictions

    path = tmp_path / "p.jsonl"
    write_predictions(path, {"r1": [], "r2": [C]}, {"r1": [C], "r2": [C]})
    predicted, gold = exp13_modules.loaders.load_predictions(path)
    assert predicted["r1"] == set()
    assert "r1" in predicted


def test_candidates_are_read_as_a_ranked_list(exp13_modules, exp13_results_dir):
    """``flat_topm``'s candidates are ordered by descending retrieval score. Order is the metric."""
    path = _preds(exp13_results_dir, "exp13_05_topm_retrieval_slm", "hcy",
                  "flat_topm_predictions.jsonl")
    candidates = exp13_modules.loaders.load_prediction_candidates(path)
    assert candidates["r1"] == [C, D, F, E]
    assert candidates["r2"][0] == D


def test_a_truncated_final_line_raises_rather_than_being_dropped(exp13_modules, tmp_path):
    """A wall-clock kill truncates the last line. Skipping it would silently shrink the cohort."""
    path = tmp_path / "broken.jsonl"
    path.write_text('{"report_id": "r1", "summary": true, "predicted_set": [], "gold_set": []}\n'
                    '{"report_id": "r2", "summ')
    with pytest.raises(json.JSONDecodeError, match="broken.jsonl:2"):
        exp13_modules.loaders.load_predictions(path)


# ── Nodes ────────────────────────────────────────────────────────────────────

def test_nodes_carry_both_scores_and_the_label(exp13_modules, exp13_results_dir):
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    seizure = next(r for r in nodes["r1"] if r["hpo_id"] == C)
    assert seizure["prune_score"] == PRUNE_SCORE[C]
    assert seizure["accept_score"] == ACCEPT_SCORE[C]
    assert seizure["is_gold"] == 1


def test_visited_and_expanded_sets_differ(exp13_modules, exp13_results_dir):
    """A node can be visited (scored) without being expanded (children revealed), that is pruning."""
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    visited = exp13_modules.loaders.visited_sets(nodes)
    expanded = exp13_modules.loaders.expanded_sets(nodes)
    assert visited["r1"] == {A, B, C, D, E, F, G}
    assert expanded["r1"] == {A, B, E, G}
    assert C not in expanded["r1"]      # a visited leaf that was never expanded


def test_the_higher_tau_visits_a_subset(exp13_modules, exp13_results_dir):
    """Monotonicity of the sweep, the property that makes the lowest tau the calibration sample."""
    load = exp13_modules.loaders.load_nodes
    visited = exp13_modules.loaders.visited_sets
    low = visited(load(_preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                              "tree_gate_lr_nodes.jsonl")))
    high = visited(load(_preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.5",
                               "tree_gate_lr_nodes.jsonl")))
    assert high["r1"] < low["r1"]


# ── Traversal ────────────────────────────────────────────────────────────────

def test_traversal_records_keep_the_driver_key_names(exp13_modules, exp13_results_dir):
    """``slm_call_stats`` reads these names directly, so they must survive unchanged."""
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_traversal.jsonl")
    records = exp13_modules.loaders.load_traversal(path)
    assert {"n_slm_calls", "n_unique_nodes_visited", "total_frontier_insertions",
            "depth_table"} <= set(records[0])


def test_depth_visits_reproduces_the_per_depth_totals(exp13_modules, exp13_results_dir):
    """r1 at tau_0.1 visits 2 nodes at depth 1 (A, G), 2 at depth 2 (B, E), 3 at depth 3.

    ``depth_cost_table`` reads only depth, expanded, accepted and n_slm_calls, so expanding the
    per-depth aggregates into synthetic entries reproduces what it sums.
    """
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_traversal.jsonl")
    records = exp13_modules.loaders.load_traversal(path)
    r1 = next(r for r in records if r["report_id"] == "r1")
    visits = exp13_modules.loaders.depth_visits([r1])[0]

    by_depth = {}
    for rec in visits.values():
        by_depth[rec["depth"]] = by_depth.get(rec["depth"], 0) + 1
    assert by_depth == {1: 2, 2: 2, 3: 3}
    assert sum(r["n_slm_calls"] for r in visits.values()) == pytest.approx(5 * 7)


def test_depth_visits_keys_on_real_hpo_ids_when_nodes_are_supplied(
        exp13_modules, exp13_results_dir):
    """The ground truth columns of ``depth_cost_table`` test ``hpo_id in gold``, so synthetic keys make
    them read zero. With the nodes file the keys are the real ontology ids, and the per-depth
    totals the cost columns sum are unchanged, both files derive from the same
    ``TraversalResult.visits``."""
    run_dir = exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
    records = exp13_modules.loaders.load_traversal(run_dir / "tree_gate_lr_traversal.jsonl")
    nodes = exp13_modules.loaders.load_nodes(run_dir / "tree_gate_lr_nodes.jsonl")
    r1 = next(r for r in records if r["report_id"] == "r1")

    real = exp13_modules.loaders.depth_visits([r1], nodes)[0]
    synthetic = exp13_modules.loaders.depth_visits([r1])[0]

    assert set(real) == set(exp13_modules.loaders.visited_sets(nodes)["r1"])
    assert all(k.startswith("HP:") for k in real)
    assert not any(k.startswith("HP:") for k in synthetic)

    def totals(visits):
        out: dict[int, tuple[int, int, int, float]] = {}
        for rec in visits.values():
            n, exp, acc, calls = out.get(rec["depth"], (0, 0, 0, 0.0))
            out[rec["depth"]] = (n + 1, exp + int(rec["expanded"]),
                                 acc + int(rec["accepted"]), calls + rec["n_slm_calls"])
        return out

    real_totals, synth_totals = totals(real), totals(synthetic)
    assert real_totals.keys() == synth_totals.keys()
    for depth, (n, exp, acc, calls) in real_totals.items():
        assert (n, exp, acc) == synth_totals[depth][:3]
        assert calls == pytest.approx(synth_totals[depth][3])


# ── The two reconstructed scores ─────────────────────────────────────────────

def test_accept_score_is_reconstructed_as_max_sigmoid_margin(exp13_modules, exp13_results_dir):
    """``flat_topm`` persists per-sentence margins but no aggregated score.

    The fixture gives rank 1 a margin of +2.0 and the rest -2.0, so AnyYes aggregation yields
    sigmoid(2) = 0.8808 for every pair, the same rule the driver applies at inference time, so
    re-running a threshold over these reproduces that run's decisions.
    """
    path = exp13_results_dir / "exp13_05_topm_retrieval_slm" / "hcy" / "flat_topm_calls.jsonl"
    scores = exp13_modules.loaders.accept_scores_from_calls(path)
    assert scores[("r1", C)] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_sigmoid_does_not_overflow_on_a_confident_margin(exp13_modules, tmp_path):
    """A margin of 1000 is reachable and must not raise ``OverflowError`` mid-cohort."""
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"report_id": "r", "hpo_id": C, "rank": 1, "sent_index": 0,
                                "margin": 1000.0}) + "\n"
                    + json.dumps({"report_id": "r", "hpo_id": D, "rank": 1, "sent_index": 0,
                                  "margin": -1000.0}) + "\n")
    scores = exp13_modules.loaders.accept_scores_from_calls(path)
    assert scores[("r", C)] == pytest.approx(1.0)
    assert scores[("r", D)] == pytest.approx(0.0)


def test_ranked_segments_are_ordered_by_rank(exp13_modules, exp13_results_dir):
    """Eq. (2) is defined over the ranked list, so the order is the measurement."""
    path = exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tree_gate_lr_calls.jsonl"
    ranked = exp13_modules.loaders.load_calls_ranked_segments(path, "union")
    assert ranked[("r1", C)] == [0, 1, 2]
    assert ranked[("r1", F)] == [2, 0, 1]


def test_ctx_type_filter_separates_the_two_retrieval_contexts(exp13_modules, tmp_path):
    """``union_prune_own_accept`` writes both. Retrieval must be measured on one of them."""
    from fixtures.exp13_output import write_calls

    path = tmp_path / "calls.jsonl"
    write_calls(path, "tree_gate_lr", ctx_types=("union", "own"))
    assert exp13_modules.loaders.available_ctx_types(path) == {"union", "own"}
    assert exp13_modules.loaders.load_calls_ranked_segments(path, "own")[("r1", C)] == [0, 1, 2]


def test_ensemble_confidence_is_the_vote_fraction(exp13_modules, exp13_results_dir):
    """Four models; C in r1 is found by all four, D by two, E by one."""
    path = (exp13_results_dir / "phenojury_generation_free_listing" / "hcy"
            / "slm_ensemble_detections.jsonl")
    votes, models = exp13_modules.loaders.load_detections(path)
    assert models == ["m1", "m2", "m3", "m4"]

    conf = exp13_modules.loaders.ensemble_confidences(votes, len(models))
    assert conf[("r1", C)] == pytest.approx(1.0)
    assert conf[("r1", D)] == pytest.approx(0.5)
    assert conf[("r1", E)] == pytest.approx(0.25)


def test_the_active_model_count_is_read_from_the_data(exp13_modules, tmp_path):
    """A run where a model died must not be scored as if eight models had voted."""
    from fixtures.exp13_output import write_detections

    path = tmp_path / "det.jsonl"
    write_detections(path, models=("m1", "m2"))
    votes, models = exp13_modules.loaders.load_detections(path)
    assert models == ["m1", "m2"]
    assert exp13_modules.loaders.ensemble_confidences(votes, len(models))[("r1", C)] == 1.0


def test_zero_models_is_rejected(exp13_modules):
    with pytest.raises(ValueError, match="positive"):
        exp13_modules.loaders.ensemble_confidences({}, 0)


# ── Calibration sample assembly ──────────────────────────────────────────────

def test_accept_score_is_labelled_by_node_presence(exp13_modules, exp13_results_dir, toy_view):
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    gold = {k: set(v) for k, v in GOLD.items()}
    sample = exp13_modules.loaders.tree_calibration_sample(nodes, gold, toy_view, "accept_score")

    labels = dict(zip(zip(sample["report_ids"], sample["hpo_ids"]), sample["labels"]))
    assert labels[("r1", C)] == 1        # C is ground truth in r1
    assert labels[("r1", B)] == 0        # B is an ancestor of C, but not itself ground truth
    assert labels[("r1", D)] == 0


def test_prune_score_is_labelled_by_subtree_presence(exp13_modules, exp13_results_dir, toy_view):
    """The gate decides whether anything worth finding lies *below* a node, not whether the node
    is itself present. Scoring it against ``is_gold`` would score it against a question it was
    never asked, and would make a well-behaved gate look badly miscalibrated at every ancestor.

    In r1, ground truth is {C, F}. B is an ancestor of C, so its subtree label is 1 even though B itself
    is not ground truth. G's subtree holds neither, so its label is 0.
    """
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    gold = {k: set(v) for k, v in GOLD.items()}
    sample = exp13_modules.loaders.tree_calibration_sample(nodes, gold, toy_view, "prune_score")

    labels = dict(zip(zip(sample["report_ids"], sample["hpo_ids"]), sample["labels"]))
    assert labels[("r1", B)] == 1        # C is in B's subtree
    assert labels[("r1", A)] == 1        # both C and F are under A
    assert labels[("r1", E)] == 1        # F is under E
    assert labels[("r1", G)] == 0        # nothing ground truth under G
    assert labels[("r1", C)] == 1        # a node is in its own subtree


def test_the_two_labellings_actually_differ(exp13_modules, exp13_results_dir, toy_view):
    """If they did not, the distinction above would be decoration, not a decision."""
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    gold = {k: set(v) for k, v in GOLD.items()}
    node_labels = exp13_modules.loaders.tree_calibration_sample(
        nodes, gold, toy_view, "accept_score")["labels"]
    subtree_labels = exp13_modules.loaders.tree_calibration_sample(
        nodes, gold, toy_view, "prune_score")["labels"]
    assert node_labels != subtree_labels
    assert sum(subtree_labels) > sum(node_labels)


def test_accept_label_follows_the_gold_argument_not_the_drivers_is_gold(
        exp13_modules, exp13_results_dir, toy_view):
    """Regression: the accept label used to be read off the artifact's ``is_gold`` flag.

    That flag records whichever ground-truth set was configured at *inference* time. Re-scoring against a
    corrected ground truth file then moved the derived prune label while the accept label stayed fixed,
    so one calibration table mixed two ground truth definitions, invisible in the output. Here the ground truth
    argument drops C and adds B, and the accept labels must follow it. The count of
    disagreements with ``is_gold`` is returned so the caller can say the ground truth changed.
    """
    path = _preds(exp13_results_dir, "exp13_00_tree_gate_lr", "hcy", "tau_0.1",
                  "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    revised = {k: set(v) for k, v in GOLD.items()}
    revised["r1"] = {B, F}          # C was ground truth at inference time; B was not

    sample = exp13_modules.loaders.tree_calibration_sample(
        nodes, revised, toy_view, "accept_score")
    labels = dict(zip(zip(sample["report_ids"], sample["hpo_ids"]), sample["labels"]))

    assert labels[("r1", B)] == 1        # newly ground truth, though the artifact says is_gold=0
    assert labels[("r1", C)] == 0        # no longer ground truth, though the artifact says is_gold=1
    assert labels[("r1", F)] == 1
    assert sample["n_is_gold_disagreements"] == 2

    unchanged = exp13_modules.loaders.tree_calibration_sample(
        nodes, {k: set(v) for k, v in GOLD.items()}, toy_view, "accept_score")
    assert unchanged["n_is_gold_disagreements"] == 0


def test_an_unknown_score_key_is_rejected(exp13_modules, toy_view):
    with pytest.raises(ValueError, match="accept_score or prune_score"):
        exp13_modules.loaders.tree_calibration_sample({}, {}, toy_view, "whatever")


def test_pair_sample_labels_by_membership_in_the_ground_truth_set(exp13_modules):
    scores = {("r1", C): 0.9, ("r1", D): 0.4}
    sample = exp13_modules.loaders.pair_calibration_sample(scores, {"r1": {C}})
    assert dict(zip(sample["hpo_ids"], sample["labels"])) == {C: 1, D: 0}


# ── Alignment ────────────────────────────────────────────────────────────────

def test_aligned_sets_intersects_and_orders_deterministically(exp13_modules):
    """Every cohort metric zips two iterables, so alignment must be explicit, not incidental."""
    predicted = {"r2": {C}, "r1": {D}, "r9": {C}}
    gold = {"r1": {D}, "r2": {C}}
    rids, gold_sets, pred_sets = exp13_modules.loaders.aligned_sets(predicted, gold)
    assert rids == ["r1", "r2"]          # r9 has no ground truth, so it is excluded
    assert gold_sets == [{D}, {C}]
    assert pred_sets == [{D}, {C}]


def test_gold_loader_normalises_to_sets_of_stripped_codes(exp13_modules, tmp_path):
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text(f"patient_id,hpo_codes\nr1,\"['{C}', ' {D} ']\"\nr2,{F}\n")
    gold = exp13_modules.loaders.load_gold("hcy", hcy_gt_path=str(csv_path))
    assert gold == {"r1": {C, D}, "r2": {F}}


def test_gold_loader_rejects_an_unknown_cohort(exp13_modules):
    with pytest.raises(ValueError, match="unknown cohort"):
        exp13_modules.loaders.load_gold("nope")
