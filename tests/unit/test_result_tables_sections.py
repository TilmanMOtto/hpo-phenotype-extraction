"""The skeleton sections, on a scenario small enough to work out on paper.

The fixture cohort (``tests/fixtures/exp13_output``)::

    r1  ground truth {C, F}   predicted {C, D}   -> TP=1 (C), FP=1 (D), FN=1 (F)
    r2  ground truth {D}      predicted {D}      -> TP=1 (D), FP=0,     FN=0

    pooled: TP=2, FP=1, FN=1  ->  micro P = R = F1 = 2/3

Toy-ontology facts these tests lean on (see ``tests/fixtures/toy_ontology``)::

    An(C) = {C, B, A}    An(D) = {D, B, A}    An(F) = {F, E, A}
    dist(D, C) = 2  (siblings under B)
    depth: A = G = 1, B = E = 2, C = D = F = 3
"""

from __future__ import annotations

import math

import pytest

from fixtures.exp13_output import EXPANDED, GOLD
from fixtures.toy_ontology import A, B, C, D, E, F, toy_children_map

pytestmark = pytest.mark.unit


@pytest.fixture
def runs():
    """Two runs of one method, so main selection has something to choose between."""
    gold_sets = [{C, F}, {D}]
    pred_sets = [{C, D}, {D}]
    return [
        {"method": "tree_gate_lr", "cohort": "hcy", "point": "tau_0.1",
         "gold_sets": gold_sets, "pred_sets": pred_sets},
        {"method": "tree_gate_lr", "cohort": "hcy", "point": "tau_0.5",
         "gold_sets": gold_sets, "pred_sets": [{C}, {D}]},
    ]


# ── §Core extraction quality, eq. (1) ────────────────────────────────────────

def test_micro_prf_is_two_thirds_by_hand(exp13_modules, runs):
    """TP=2, FP=1, FN=1 -> P = 2/3, R = 2/3, F1 = 2/3."""
    row = exp13_modules.sections.core_quality(runs)[0]
    assert (row["tp"], row["fp"], row["fn"]) == (2, 1, 1)
    assert row["micro_precision"] == pytest.approx(2 / 3)
    assert row["micro_recall"] == pytest.approx(2 / 3)
    assert row["micro_f1"] == pytest.approx(2 / 3)


def test_both_macro_conventions_are_reported_and_differ(exp13_modules, runs):
    """r1: P=1/2, R=1/2, F1=1/2.  r2: P=R=F1=1.

    Mean of per-report F1 = 3/4, the skeleton's definition.
    Harmonic mean of the averaged P and R = 2*(3/4)*(3/4)/(3/2) = 3/4 too, here. But the two are
    distinct quantities and both must be present so a table cannot quote one while citing the
    other. The keys existing is the contract. The arithmetic below pins the skeleton's one.
    """
    row = exp13_modules.sections.core_quality(runs)[0]
    assert row["macro_f1"] == pytest.approx(0.75)
    assert "macro_f1_of_means" in row
    assert row["macro_precision"] == pytest.approx(0.75)


def test_the_main_result_is_the_best_micro_f1_and_is_labelled_an_oracle(exp13_modules, runs):
    """tau_0.5 predicts {C} and {D}: TP=2, FP=0, FN=1 -> P=1, R=2/3, F1=0.8 > 2/3."""
    rows = exp13_modules.sections.core_quality(runs)
    headline = [r for r in rows if r["is_headline"]]
    assert len(headline) == 1
    assert headline[0]["operating_point"] == "tau_0.5"
    assert headline[0]["micro_f1"] == pytest.approx(0.8)
    assert "oracle" in headline[0]["headline_selection"]


def test_one_main_result_per_method_and_cohort(exp13_modules, runs):
    extra = dict(runs[0], cohort="gsc")
    rows = exp13_modules.sections.core_quality(runs + [extra])
    per_group = {}
    for row in rows:
        if row["is_headline"]:
            per_group[(row["method"], row["cohort"])] = per_group.get(
                (row["method"], row["cohort"]), 0) + 1
    assert set(per_group.values()) == {1}


def test_main_result_ties_go_to_the_first_operating_point(exp13_modules, runs):
    """Deterministic, and the conservative choice: lower tau_prune visits more nodes."""
    tied = [runs[0], dict(runs[0], point="tau_0.5")]
    rows = exp13_modules.sections.core_quality(tied)
    assert rows[0]["is_headline"] and not rows[1]["is_headline"]


# ── Cohort coverage ──────────────────────────────────────────────────────────

def test_cohort_coverage_counts_the_gold_label_space(exp13_modules, toy_view):
    """ground truth = {r1: {C, F}, r2: {D}} -> 3 doc-term pairs, 3 unique terms, 1.5 per report.

    All three are at depth 3. C and D sit under A; F sits under A as well (via E), so the cohort
    touches one organ system.
    """
    gold = {k: set(v) for k, v in GOLD.items()}
    cov = exp13_modules.sections.cohort_coverage("hcy", gold, toy_view)
    assert cov["n_reports"] == 2
    assert cov["n_doc_term_pairs"] == 3
    assert cov["n_unique_terms"] == 3
    assert cov["terms_per_report_mean"] == pytest.approx(1.5)
    assert cov["depth_mean_cohort"] == pytest.approx(3.0)
    assert cov["n_organ_systems_total"] == 1
    assert cov["ancestor_counts"]["n_terms"] == 3


# ── §Hierarchy-aware quality, eqs. (3) and (4) ───────────────────────────────

def test_closure_hf_exceeds_flat_f1(exp13_modules, runs, toy_view):
    """Ancestor closure introduces true positives at shallow depths that any system recovers.

    r1: An(pred {C,D}) = {C,B,A,D}; An(ground truth {C,F}) = {C,B,A,F,E}. Overlap {C,B,A} = 3.
    So hP = 3/4, hR = 3/5, both far above the flat 1/2, which is the inflation the skeleton
    warns about and the reason hF is never reported without F1 beside it.
    """
    row = exp13_modules.sections.hierarchy_quality(runs, toy_view)[0]
    assert row["micro_hf"] > row["micro_f1_flat"]
    assert row["hf_minus_f1"] == pytest.approx(row["micro_hf"] - row["micro_f1_flat"])


def test_cophe_sits_between_flat_and_closure(exp13_modules, runs, toy_view):
    """CoPHE propagates counts, so over-prediction inside a surviving subtree is not absorbed."""
    row = exp13_modules.sections.hierarchy_quality(runs, toy_view)[0]
    assert row["micro_f1_flat"] <= row["micro_cophe_f1"] <= row["micro_hf"]


def test_hierarchy_rows_carry_flat_f1_so_hf_is_never_alone(exp13_modules, runs, toy_view):
    for row in exp13_modules.sections.hierarchy_quality(runs, toy_view):
        assert row["micro_f1_flat"] is not None


# ── §Severity of near-misses, eq. (5), and the taxonomy ──────────────────────

def test_the_only_false_positive_is_a_sibling_two_hops_from_gold(exp13_modules, runs, toy_view):
    """D is predicted in r1 where ground truth is {C, F}. D and C share the direct parent B, so D is a
    sibling error at undirected distance 2, a near-miss, not an unrelated prediction."""
    tax, near = exp13_modules.sections.error_analysis(runs, toy_view)
    assert tax[0]["n_fp"] == 1
    assert tax[0]["fp_sibling"] == 1
    assert tax[0]["garcia_sibling_frac"] == pytest.approx(1.0)
    assert near[0]["distance_mean"] == pytest.approx(2.0)
    assert near[0]["fraction_within_2"] == pytest.approx(1.0)


def test_the_missed_term_is_classified_by_what_was_predicted_near_it(exp13_modules, runs,
                                                                     toy_view):
    """F is missed in r1. Nothing predicted ({C, D}) is F's ancestor, descendant or sibling, F's parent is E while C and D are under B, so this is a detection failure, not a
    granularity error."""
    tax, _ = exp13_modules.sections.error_analysis(runs, toy_view)
    assert tax[0]["n_fn"] == 1
    assert tax[0]["fn_nothing_near"] == 1


def test_garcia_fractions_sum_to_one_over_the_five_published_buckets(exp13_modules, runs,
                                                                     toy_view):
    """The renormalisation is what makes the profile comparable to RAG-HPO's decomposition."""
    tax, _ = exp13_modules.sections.error_analysis(runs, toy_view)
    total = sum(v for k, v in tax[0].items() if k.startswith("garcia_") and k.endswith("_frac"))
    assert total == pytest.approx(1.0)


# ── §Reachability recall, eq. (6), and blocking depth ────────────────────────

def test_reachability_recall_is_recomputed_from_the_expand_decisions(exp13_modules):
    """At tau_0.1, r1 expands {A, B, E, G}, so C (under B) and F (under E) are both reachable:
    recall 1.0. r2 expands {A, B, G} and its ground truth {D} is under B: also 1.0."""
    children, roots = toy_children_map()
    gold = {k: set(v) for k, v in GOLD.items()}
    expanded = {k: set(v) for k, v in EXPANDED["tau_0.1"].items()}
    reach, _ = exp13_modules.sections.traversal_quality(
        "tree_gate_lr", "hcy", "tau_0.1", gold, expanded, children, roots)
    assert reach["micro_reachability_recall"] == pytest.approx(1.0)
    assert reach["n_gold_terms"] == 3
    assert reach["n_gold_reachable"] == 3


def test_pruning_a_branch_costs_reachability(exp13_modules):
    """At tau_0.5, r1 no longer expands E, so F becomes unreachable: 1 of 2 annotated terms in r1,
    2 of 3 across the cohort. This is the ceiling pruning imposes on end-to-end recall,
    independent of how competent the identification stage is at the leaf."""
    children, roots = toy_children_map()
    gold = {k: set(v) for k, v in GOLD.items()}
    expanded = {k: set(v) for k, v in EXPANDED["tau_0.5"].items()}
    reach, blocking = exp13_modules.sections.traversal_quality(
        "tree_gate_lr", "hcy", "tau_0.5", gold, expanded, children, roots)
    assert reach["micro_reachability_recall"] == pytest.approx(2 / 3)
    assert reach["n_gold_reachable"] == 2

    # F's path was severed at E, which sits at depth 2.
    assert blocking["n_blocked"] == 1
    assert blocking["histogram"] == {2: 1}
    assert blocking["depth_mean"] == pytest.approx(2.0)


def test_blocking_depth_can_be_skipped_for_a_sweep(exp13_modules):
    """It is a DAG-wide dynamic program per report, so the sweep runs it only where asked."""
    children, roots = toy_children_map()
    gold = {k: set(v) for k, v in GOLD.items()}
    expanded = {k: set(v) for k, v in EXPANDED["tau_0.5"].items()}
    _, blocking = exp13_modules.sections.traversal_quality(
        "tree_gate_lr", "hcy", "tau_0.5", gold, expanded, children, roots,
        compute_blocking=False)
    assert blocking is None


def test_visited_set_recall_agrees_with_reachability_recall(exp13_modules, exp13_results_dir):
    """Two independent routes to the same quantity: set membership in the recorded visited set,
    versus a BFS recomputed from the ontology adjacency and the expand decisions. If they
    disagree, either the driver's trace or this analysis is wrong."""
    path = (exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.5"
            / "tree_gate_lr_nodes.jsonl")
    nodes = exp13_modules.loaders.load_nodes(path)
    gold = {k: set(v) for k, v in GOLD.items()}
    children, roots = toy_children_map()

    from_visited = exp13_modules.sections.traversal_candidate_recall(
        "tree_gate_lr", "hcy", "tau_0.5", exp13_modules.loaders.visited_sets(nodes), gold)
    reach, _ = exp13_modules.sections.traversal_quality(
        "tree_gate_lr", "hcy", "tau_0.5", gold,
        exp13_modules.loaders.expanded_sets(nodes), children, roots, compute_blocking=False)

    assert from_visited["recall_macro"] == pytest.approx(reach["macro_reachability_recall"])


# ── §Calibration, eq. (7) ────────────────────────────────────────────────────

@pytest.fixture
def calibration_sample():
    """A well-separated sample: the positives carry the high confidences."""
    return {
        "confidences": [0.9, 0.85, 0.8, 0.2, 0.15, 0.1] * 6,
        "labels": [1, 1, 1, 0, 0, 0] * 6,
        "hpo_ids": [C, D, F, E, B, A] * 6,
        "report_ids": ["r1"] * 36,
        "depths": [3, 3, 3, 2, 2, 1] * 6,
    }


def test_calibration_reports_every_estimator_the_skeleton_names(exp13_modules,
                                                                calibration_sample, toy_view):
    row, gaps, conds = exp13_modules.sections.calibration_quality(
        "tree_gate_lr", "hcy", "tau_0.1", calibration_sample, toy_view,
        {"r1": {C, D, F}}, "accept_score", tau_prune=0.1, delta=0.1)
    for key in ("ece_equal_width", "ece_equal_mass", "ece_roelofs", "brier",
                "n_distinct_scores"):
        assert row[key] is not None
    assert row["n"] == 36
    assert row["prevalence"] == pytest.approx(0.5)
    assert gaps and conds


def test_the_tail_metric_is_restricted_to_the_low_confidence_region(exp13_modules,
                                                                    calibration_sample, toy_view):
    """tau_prune + delta = 0.2, so the tail covers the three cells at 0.1, 0.15 and 0.2, half the sample. This is the region where pruning decisions are made and where an aggregate
    ECE, dominated by the dense high-confidence bins, is blind."""
    row, _, _ = exp13_modules.sections.calibration_quality(
        "tree_gate_lr", "hcy", "tau_0.1", calibration_sample, toy_view,
        {"r1": {C, D, F}}, "accept_score", tau_prune=0.1, delta=0.1)
    assert row["tail_upper"] == pytest.approx(0.2)
    assert row["tail_coverage"] == pytest.approx(0.5)
    assert row["tail_n"] == 18


def test_the_label_semantics_are_recorded_on_the_row(exp13_modules, calibration_sample,
                                                     toy_view):
    """The two scores answer different questions. A table without this column would suggest
    they are the same measurement at two thresholds."""
    accept, _, _ = exp13_modules.sections.calibration_quality(
        "tree_gate_lr", "hcy", "t", calibration_sample, toy_view, {}, "accept_score", None, 0.1)
    prune, _, _ = exp13_modules.sections.calibration_quality(
        "tree_gate_lr", "hcy", "t", calibration_sample, toy_view, {}, "prune_score", None, 0.1)
    assert accept["label_semantics"] == "node present"
    assert "subtree" in prune["label_semantics"]


def test_conditional_calibration_covers_all_three_groupings(exp13_modules, calibration_sample,
                                                            toy_view):
    """Depth, organ system and gold-set term frequency, aggregate ECE cannot detect
    group-conditional miscalibration, and the transferability claim is regional."""
    _, _, conds = exp13_modules.sections.calibration_quality(
        "tree_gate_lr", "hcy", "t", calibration_sample, toy_view, {"r1": {C, D}},
        "accept_score", None, 0.1)
    assert {r["grouping"] for r in conds} == {"depth", "organ_system", "term_frequency"}
    assert all("sufficient" in r for r in conds)


def test_the_discrete_score_table_renames_the_atom_to_avoid_a_column_collision(
        exp13_modules, calibration_sample):
    """``discrete_score_table`` calls the atom ``score``. So does the column naming which score
    this is. The atom is renamed so both survive."""
    rows = exp13_modules.sections.discrete_reliability(
        "tree_gate_lr", "hcy", "t", calibration_sample, "accept_score")
    assert {r["score"] for r in rows} == {"accept_score"}
    assert sorted(r["score_atom"] for r in rows) == [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]
    assert all(r["wilson_lo"] <= r["frequency"] <= r["wilson_hi"] for r in rows)


# ── §Risk--coverage, eq. (8) ─────────────────────────────────────────────────

def test_selective_risk_equals_one_minus_precision_in_the_emitted_curve(exp13_modules,
                                                                        calibration_sample):
    """The identity this unit and loss imply, visible in the table rather than merely asserted."""
    _, curve = exp13_modules.sections.risk_coverage(
        "tree_gate_lr", "hcy", "t", calibration_sample, "accept_score")
    for row in curve:
        if row["n_covered"]:
            assert row["precision"] == pytest.approx(1.0 - row["selective_risk"])


def test_augrc_never_exceeds_aurc(exp13_modules, calibration_sample):
    """AUGRC weights each point by its coverage, so it is the smaller of the two."""
    row, _ = exp13_modules.sections.risk_coverage(
        "tree_gate_lr", "hcy", "t", calibration_sample, "accept_score")
    assert row["augrc"] <= row["aurc"] + 1e-12


def test_risk_at_full_coverage_is_the_reference_for_the_aurc_confound(exp13_modules,
                                                                    calibration_sample):
    """Prevalence is 0.5, so half the covered cells are failures at full coverage.

    AURC conflates discrimination with calibration because methods differ here. Without this
    evidence location two AURCs are not comparable, which is why the row carries it and a note saying so."""
    row, _ = exp13_modules.sections.risk_coverage(
        "tree_gate_lr", "hcy", "t", calibration_sample, "accept_score")
    assert row["risk_at_full_coverage"] == pytest.approx(0.5)
    assert "conflates discrimination with calibration" in row["aurc_confound_note"]


def test_a_perfect_ranker_beats_an_inverted_one_on_aurc(exp13_modules):
    good = {"confidences": [0.9, 0.8, 0.2, 0.1], "labels": [1, 1, 0, 0],
            "hpo_ids": [C] * 4, "report_ids": ["r"] * 4, "depths": [3] * 4}
    bad = dict(good, confidences=[0.1, 0.2, 0.8, 0.9])
    assert exp13_modules.sections.aurc_pair(good)[0] < exp13_modules.sections.aurc_pair(bad)[0]


# ── §Threshold transferability ───────────────────────────────────────────────

def test_transferability_reports_the_gap_not_the_absolute_score(exp13_modules,
                                                                exp13_results_dir):
    """The measured quantity is oracle-on-held-out minus transferred-to-held-out. The absolute
    score mostly reflects how hard the held-out split is. The gap is what portability means."""
    load = exp13_modules.loaders.load_nodes
    base = exp13_results_dir / "exp13_00_tree_gate_lr"
    nodes_by_cohort = {
        cohort: {
            0.1: load(base / cohort / "tau_0.1" / "tree_gate_lr_nodes.jsonl"),
            0.5: load(base / cohort / "tau_0.5" / "tree_gate_lr_nodes.jsonl"),
        }
        for cohort in ("hcy", "gsc")
    }
    gold = {c: {k: set(v) for k, v in GOLD.items()} for c in ("hcy", "gsc")}

    rows = exp13_modules.sections.transferability(
        "tree_gate_lr", nodes_by_cohort, gold, [0.3, 0.5, 0.7], k_folds=2)
    cross = [r for r in rows if r.get("protocol") == "cross-cohort"]
    assert len(cross) == 2                      # hcy->gsc and gsc->hcy
    for row in cross:
        assert row["gap"] == pytest.approx(row["oracle_value"] - row["transferred_value"])
        assert row["gap"] >= -1e-12             # The oracle cannot be beaten on its own split


def test_transferability_placeholders_when_only_one_cohort_is_available(exp13_modules,
                                                                        exp13_results_dir):
    load = exp13_modules.loaders.load_nodes
    nodes = {"hcy": {0.1: load(exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
                               / "tree_gate_lr_nodes.jsonl")}}
    rows = exp13_modules.sections.transferability(
        "tree_gate_lr", nodes, {"hcy": {k: set(v) for k, v in GOLD.items()}}, [0.5], k_folds=2)
    cross = [r for r in rows if r.get("protocol") == "cross-cohort"]
    assert cross[0]["status"] == "placeholder"
    assert "both HCY and GSC+" in cross[0]["reason"]


# ── §Scalability ─────────────────────────────────────────────────────────────

def test_slm_calls_and_the_dedup_ratio(exp13_modules, exp13_results_dir):
    """r1 visits 7 nodes with 8 frontier insertions, r2 visits 4 with 5, a DAG effect the
    skeleton requires be auditable. 5 SLM calls per node, so (35 + 20) / 2 = 27.5 per report."""
    path = (exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
            / "tree_gate_lr_traversal.jsonl")
    row = exp13_modules.sections.slm_cost(
        "tree_gate_lr", "hcy", "tau_0.1", exp13_modules.loaders.load_traversal(path))
    assert row["slm_calls_per_report"] == pytest.approx(27.5)
    assert row["unique_nodes_per_report"] == pytest.approx(5.5)
    assert row["dedup_ratio"] == pytest.approx(13 / 11)
    assert row["calls_per_node"] == pytest.approx(5.0)


def _depth_cost_inputs(exp13_modules, exp13_results_dir):
    run_dir = exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
    records = exp13_modules.loaders.load_traversal(run_dir / "tree_gate_lr_traversal.jsonl")
    nodes = exp13_modules.loaders.load_nodes(run_dir / "tree_gate_lr_nodes.jsonl")
    gold = [set(GOLD[str(r["report_id"])]) for r in records]
    return records, nodes, gold


def test_depth_cost_table_is_ordered_and_cumulative(exp13_modules, exp13_results_dir):
    records, nodes, gold = _depth_cost_inputs(exp13_modules, exp13_results_dir)
    rows = exp13_modules.sections.depth_cost(
        "tree_gate_lr", "hcy", "tau_0.1",
        exp13_modules.loaders.depth_visits(records, nodes), gold)

    assert [r["depth"] for r in rows] == [1, 2, 3]
    cumulative = [r["cumulative_slm_calls"] for r in rows]
    assert cumulative == sorted(cumulative)
    assert all(0.0 <= r["survival_rate"] <= 1.0 for r in rows)


def test_depth_cost_gold_columns_are_populated_when_nodes_supply_real_ids(
        exp13_modules, exp13_results_dir):
    """Regression: the ground truth columns read zero for the whole 2026-08-02 run.

    ``depth_cost_table`` counts a ground truth visit with ``hpo_id in gold``, so it needs real ontology
    ids. Fed only the traversal file, which stores per-depth aggregates, the loader had to
    synthesise keys, and every ``n_gold_visited`` silently came out 0 instead of the column being
    absent. Passing the nodes file fixes it; ``r1`` visits ground truth C and F at depth 3, ``r2`` visits
    ground truth D at depth 3, so all four annotated terms of the cohort are visited at depth 3 and nowhere
    else.
    """
    records, nodes, gold = _depth_cost_inputs(exp13_modules, exp13_results_dir)
    rows = exp13_modules.sections.depth_cost(
        "tree_gate_lr", "hcy", "tau_0.1",
        exp13_modules.loaders.depth_visits(records, nodes), gold)

    by_depth = {r["depth"]: r for r in rows}
    n_gold_total = sum(len(g) for g in gold)
    assert sum(r["n_gold_visited"] for r in rows) > 0
    assert by_depth[1]["n_gold_visited"] == 0
    assert by_depth[3]["n_gold_visited"] == sum(
        len(g & set(exp13_modules.loaders.visited_sets(nodes)[rid]))
        for rid, g in zip([str(r["report_id"]) for r in records], gold))
    # Cumulative recall is monotone and ends at (ground truth visited / ground truth total).
    cum = [r["cumulative_reachability_recall"] for r in rows]
    assert cum == sorted(cum)
    assert cum[-1] == pytest.approx(
        sum(r["n_gold_visited"] for r in rows) / n_gold_total)


def test_depth_cost_omits_gold_columns_when_only_aggregates_are_available(
        exp13_modules, exp13_results_dir):
    """Without the nodes file the keys are synthetic, so ground truth must not be passed at all, the
    column has to be *absent*, which reads as `n/a`, not present and zero."""
    records, _nodes, _gold = _depth_cost_inputs(exp13_modules, exp13_results_dir)
    rows = exp13_modules.sections.depth_cost(
        "tree_gate_lr", "hcy", "tau_0.1", exp13_modules.loaders.depth_visits(records), None)
    assert rows and all("n_gold_visited" not in r for r in rows)


def test_recall_cost_pareto_joins_the_slm_calls_table_on_the_operating_point(exp13_modules):
    """Regression: the 2026-08-02 run was handed the `scalability` hardware table instead.

    Its rows are keyed ``operating_point=""``, so every swept method joined against nothing and
    the cost axis of the main efficiency/recall figure came out empty, with no row missing
    to make that visible.
    """
    core = [{"method": "tree_gate_lr", "cohort": "hcy", "operating_point": "tau_0.1",
             "micro_recall": 0.1, "micro_precision": 0.2, "micro_f1": 0.13, "is_headline": True}]
    reach = [{"method": "tree_gate_lr", "cohort": "hcy", "operating_point": "tau_0.1",
              "micro_reachability_recall": 0.11}]
    slm = [{"method": "tree_gate_lr", "cohort": "hcy", "operating_point": "tau_0.1",
            "slm_calls_per_report": 377.6, "unique_nodes_per_report": 86.0}]

    rows = exp13_modules.sections.recall_cost_pareto(core, reach, slm)
    assert len(rows) == 1
    assert rows[0]["slm_calls_per_report"] == pytest.approx(377.6)
    assert rows[0]["reachability_recall"] == pytest.approx(0.11)


def test_recall_cost_pareto_refuses_the_hardware_table(exp13_modules):
    """Passing `scalability` rows must fail loudly, not silently emptying the cost axis."""
    core = [{"method": "tree_gate_lr", "cohort": "hcy", "operating_point": "tau_0.1",
             "micro_recall": 0.1, "micro_precision": 0.2, "micro_f1": 0.13}]
    hardware = [{"method": "tree_gate_lr", "cohort": "hcy", "operating_point": "",
                 "slm_calls_per_report": 377.6, "seconds_per_report": 17.6}]
    with pytest.raises(ValueError, match="slm_calls table"):
        exp13_modules.sections.recall_cost_pareto(core, [], hardware)


def test_scaling_is_a_placeholder_without_enough_points(exp13_modules):
    """No earlier cluster script sweeps restrict_to_subtree, so this is the normal path, and it
    has to name the runs required, not emit a fitted exponent from one point."""
    result = exp13_modules.sections.scaling([])
    assert result["status"] == "placeholder"
    assert "restrict_to_subtree" in result["reason"]


def test_scaling_measures_sublinearity_when_points_exist(exp13_modules):
    """Doubling N while calls grow by 2**0.5 is an exponent of 0.5, sublinear."""
    points = [(100.0, 10.0), (200.0, 10.0 * 2 ** 0.5), (400.0, 20.0)]
    result = exp13_modules.sections.scaling(points)
    assert result["status"] == "ok"
    assert result["exponent"] == pytest.approx(0.5, abs=1e-6)
    assert result["sublinear"] is True


def test_deployment_says_what_break_even_is_missing(exp13_modules):
    """Deployment GPU-hours is a property of the earlier jobs, not of any earlier run."""
    row = exp13_modules.sections.deployment(None, None, None, None, 10.0, 5.0)
    assert row["gpu_hours"] is None
    assert "deployment_gpu_hours" in row["break_even_reason"]


def test_break_even_is_computed_when_every_input_is_supplied(exp13_modules):
    """1 GPU-hour = 3600 s of deployment. Saving 5 s per report pays it back after 720 reports."""
    row = exp13_modules.sections.deployment(1.0, None, None, None, 10.0, 5.0)
    assert row["n_reports"] == pytest.approx(720.0)
    assert row["economical"] is True


def test_a_method_that_is_slower_than_the_baseline_never_breaks_even(exp13_modules):
    row = exp13_modules.sections.deployment(1.0, None, None, None, 5.0, 10.0)
    assert math.isinf(row["n_reports"])
    assert row["economical"] is False


# ── Placeholders ─────────────────────────────────────────────────────────────

def test_placeholder_rows_carry_their_reason(exp13_modules):
    row = exp13_modules.sections.placeholder_row("raghpo_8b", "gsc", "the run did not happen")
    assert row["status"] == "placeholder"
    assert row["reason"] == "the run did not happen"
    assert row["label"] == "RAG-HPO (LLaMA-8B)"


def test_the_negation_placeholder_says_what_is_covered_instead(exp13_modules):
    """Close-neighbour behaviour *is* measured. Only negation attribution is not, and the
    difference has to be stated or a reader will read the gap as a blanket omission."""
    row = exp13_modules.sections.negation_placeholder()
    assert row["status"] == "placeholder"
    assert "eq. (5)" in row["covered_instead"]
    assert "unmeasured error rate" in row["reason"]


def test_summarise_counts_reports_how_much_of_a_table_is_real(exp13_modules, runs):
    rows = exp13_modules.sections.core_quality(runs)
    rows.append(exp13_modules.sections.placeholder_row("raghpo_70b", "hcy", "did not run"))
    counts = exp13_modules.sections.summarise_counts(rows)
    assert counts["ok"] == 2
    assert counts["placeholder"] == 1
