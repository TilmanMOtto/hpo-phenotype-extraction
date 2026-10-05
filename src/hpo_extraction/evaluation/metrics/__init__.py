"""Every measured metric referenced in ``thesis/sections/sceleton.tex``, implemented from its formula.

This package exists to be *checked*, not just called. Each function implements one equation or one
named quantity from the skeleton, its docstring quotes the sentence it answers to, and
:data:`THESIS_METRIC_INDEX` maps every section and equation number to the function that computes it, so a number in the results chapter can be traced to a formula in one step.

Deliberate properties, all of them consequences of that goal:

* **Self-contained.** Nothing here imports the repo's older metric modules
  (``hpo_extraction.evaluation.set_metrics``, ``hpo_extraction.evaluation.tree_metrics``, ``hpo_extraction.evaluation.tree_error_analysis``,
  ``apps.ui_common.calib``, ``an earlier exploratory run.../proxy_metrics``), which are left untouched so dashboards and
  past experiments keep reproducing their published numbers. The only dependencies are ``numpy``,
  ``scipy``, ``hpo_extraction.ontology.hpo_tree.HPOTree`` and, for :func:`~.calibration.smooth_ece` alone, ``relplot``. Where the two overlap, the test suite asserts they agree.
* **No IO.** Inputs are plain sets, dicts and arrays. Nothing reads a run artifact, a config or a
  file, so every function is testable on hand-worked examples and none of them can be wrong about a
  file format.
* **No silent conventions.** Every place the skeleton left a choice open, the ancestor-closure
  root exclusion, the CoPHE node set, the blocking-depth rule in a DAG, the risk--coverage unit and
  loss, the sibling definition, the empty-report convention, tie handling in AURC, is documented at
  the point of implementation and, where it is genuinely a parameter, exposed as one.

Entry points at a glance::

    from hpo_extraction.evaluation.metrics import OntologyView, flat_report, hierarchy_report

    view = OntologyView(HPOTree())
    flat = flat_report(gold_sets, pred_sets, view)          # §Core, eq. (1)
    hier = hierarchy_report(gold_sets, pred_sets, view)     # §Hierarchy-aware, eqs. (3)-(4)
    calib = calibration_report(conf, labels, tail_upper=0.4)  # §Calibration, eq. (7)
    sel = selective_report(conf, labels)                    # §Risk--coverage, eq. (8)

Scope note: the skeleton's mention-level granularity axis is **not** implemented. No system under
comparison emits mention spans, so it would be unmeasurable for every method. All extraction quality
here is document-level, with both micro and macro averaging.
"""

from __future__ import annotations

from .calibration import (
    CALIBRATION_METHODS,
    CalibrationMap,
    bin_edges,
    bin_summary,
    brier,
    calibration_report,
    cohort_transfer,
    compare_raw_vs_calibrated,
    conditional_calibration,
    depth_groups,
    discrete_score_table,
    ece_equal_mass,
    ece_equal_width,
    expected_calibration_error,
    f1_optimal_threshold_diagnostic,
    fit_calibration_map,
    fit_thresholds,
    frequency_buckets,
    kfold_report_split,
    layer1_groups,
    roelofs_monotonic_sweep,
    signed_gap_bins,
    smooth_ece,
    tail_calibration_error,
    transfer_gap,
    wilson_interval,
)
from .cost import (
    break_even_corpus_size,
    deployment_cost,
    hardware_profile,
    scaling_curve,
    slm_call_stats,
)
from .errors import (
    FN_BUCKETS,
    GARCIA_BUCKETS,
    HYGIENE_BUCKETS,
    classify_false_negative,
    classify_false_positive,
    error_taxonomy,
    false_negative_taxonomy,
    near_miss_distribution,
    nearest_gold,
    nearest_gold_distance,
)
from .flat import (
    counts,
    flat_report,
    macro_prf_by_report,
    macro_prf_by_term,
    macro_prf_by_term_domains,
    micro_prf,
    prf,
    report_prf,
)
from .hierarchy import (
    cophe_cohort,
    cophe_counts,
    cophe_prf,
    h_counts,
    h_prf,
    h_prf_cohort,
    hierarchy_report,
    subtree_counts,
)
from .ontology import (
    UNIVERSAL_NODES,
    OntologyView,
    ancestor_count_distribution,
)
from .errors_existential import (
    THESIS_BUCKETS,
    classify_false_positive_existential,
    error_taxonomy_existential,
)
from .normalise import (
    ancestor_closure,
    normalise_gold,
    normalise_pair,
    normalise_predictions,
    split_unscorable,
)
from .retrieval import (
    candidate_set_recall,
    combined_recall_bound,
    relevant_segments,
    segment_pr_at_s,
    segment_pr_curve,
    term_pr_at_m,
    term_pr_curve,
)
from .selective import (
    augrc,
    aurc,
    coverage,
    generalized_risk,
    risk_coverage_curve,
    selective_report,
    selective_risk,
)
from .recall_decomposition import (
    BLOCKING_CAUSES,
    DECOMPOSITION_BUCKETS,
    attribute_false_negative,
    delta_m_sensitivity,
    recall_decomposition,
)
from .traversal import (
    blocking_depth_distribution,
    blocking_depths,
    bfs_depths,
    depth_cost_table,
    parents_from_children,
    reachability_recall,
    reachability_recall_cohort,
    reachable_set,
    topological_order,
)

#: Maps each measured quantity in ``thesis/sections/sceleton.tex`` to the function that computes it.
#: Keys name the skeleton section and, where the text numbers one, the equation. Use this to check a
#: results-chapter number against the formula it came from.
THESIS_METRIC_INDEX: dict[str, dict[str, object]] = {
    "§Core extraction quality — eq. (1) P, R, F1": {
        "per report": report_prf,
        "micro": micro_prf,
        "macro over reports": macro_prf_by_report,
        "macro over terms": macro_prf_by_term,
        "all of the above": flat_report,
    },
    "§Core extraction quality — eq. (2) P@S, R@S": {
        "one (report, phenotype) pair": segment_pr_at_s,
        "across a range of S": segment_pr_curve,
        "relevance under the true-path rule": relevant_segments,
        "term-level P@M / R@M": term_pr_curve,
        "unranked candidate set (tree traversal)": candidate_set_recall,
    },
    "§Hierarchical Candidate Traversal — the combined recall bound": {
        "retrieval recall × reachability recall": combined_recall_bound,
    },
    "§Hierarchy-aware quality — eq. (3) hP, hR, hF": {
        "per report": h_prf,
        "cohort, micro and macro": h_prf_cohort,
        "the |An(v)| audit the text promises alongside": ancestor_count_distribution,
    },
    "§Hierarchy-aware quality — eq. (4) CoPHE": {
        "per report": cophe_prf,
        "cohort, micro and macro": cophe_cohort,
        "subtree counts x_v, y_v": subtree_counts,
        "eqs. (3) and (4) together": hierarchy_report,
    },
    "§Severity of near-misses — eq. (5) d(ŷ, Y_r)": {
        "one false positive": nearest_gold_distance,
        "distribution over a cohort": near_miss_distribution,
        "Garcia et al. taxonomy, one FP": classify_false_positive,
        "Garcia et al. taxonomy, cohort": error_taxonomy,
        "the false-negative mirror (§Results)": false_negative_taxonomy,
    },
    "§Reachability recall — eq. (6)": {
        "which nodes survive pruning": reachable_set,
        "per report": reachability_recall,
        "cohort, micro and macro": reachability_recall_cohort,
    },
    "§Blocking depth distribution": {
        "per node, deepest severing depth": blocking_depths,
        "distribution over missed gold terms": blocking_depth_distribution,
        "where each missed gold term was lost": recall_decomposition,
    },
    "§Calibration — eq. (7) ECE": {
        "the field default (equal-width)": ece_equal_width,
        "equal-mass": ece_equal_mass,
        "bin count by the Roelofs monotonic sweep": roelofs_monotonic_sweep,
        "bin-free SmoothECE (relplot)": smooth_ece,
        "binning-free limit for a discrete score": discrete_score_table,
        "all of the above": calibration_report,
    },
    "§Signed and conditional calibration": {
        "signed gap with bin populations": signed_gap_bins,
        "by group (depth / subtree / frequency)": conditional_calibration,
        "group builder — ontology depth": depth_groups,
        "group builder — top-level subtree": layer1_groups,
        "group builder — gold-set term frequency": frequency_buckets,
        "restricted to [0, τ_prune + δ]": tail_calibration_error,
    },
    "§Threshold transferability": {
        "grid-search a threshold pair on one split": fit_thresholds,
        "fit here, evaluate there": transfer_gap,
        "cross-cohort HCY ↔ GSC+": cohort_transfer,
        "K-fold over reports": kfold_report_split,
        "calibrated vs raw under one protocol": compare_raw_vs_calibrated,
    },
    "§Risk--coverage analysis — eq. (8)": {
        "coverage φ(τ)": coverage,
        "selective risk R(τ)": selective_risk,
        "the curve": risk_coverage_curve,
        "AURC": aurc,
        "AUGRC": augrc,
        "all of the above": selective_report,
    },
    "§Scalability — per-report inference cost": {
        "SLM calls, unique nodes, frontier insertions": slm_call_stats,
        "wall-clock, memory, model size, tokens": hardware_profile,
    },
    "§Scalability — deployment cost": {
        "GPU-hours and index footprint": deployment_cost,
        "break-even corpus size": break_even_corpus_size,
    },
    "§Scalability — traversal cost by depth": {
        "the per-depth table": depth_cost_table,
    },
    "§Scalability — scaling in ontology size": {
        "log-log slope vs a linear reference": scaling_curve,
    },
}

__all__ = [
    "THESIS_METRIC_INDEX",
    "UNIVERSAL_NODES",
    "GARCIA_BUCKETS",
    "HYGIENE_BUCKETS",
    "FN_BUCKETS",
    "OntologyView",
    "ancestor_count_distribution",
    # normalise
    "ancestor_closure", "split_unscorable",
    "normalise_gold", "normalise_predictions", "normalise_pair",
    # flat
    "counts", "prf", "report_prf", "micro_prf",
    "macro_prf_by_report", "macro_prf_by_term", "macro_prf_by_term_domains", "flat_report",
    # retrieval
    "relevant_segments", "segment_pr_at_s", "segment_pr_curve",
    "term_pr_at_m", "term_pr_curve", "candidate_set_recall", "combined_recall_bound",
    # hierarchy
    "h_counts", "h_prf", "h_prf_cohort", "subtree_counts",
    "cophe_counts", "cophe_prf", "cophe_cohort", "hierarchy_report",
    # errors
    "nearest_gold", "nearest_gold_distance", "near_miss_distribution",
    "classify_false_positive", "error_taxonomy",
    "THESIS_BUCKETS",
    "classify_false_positive_existential", "error_taxonomy_existential",
    "classify_false_negative", "false_negative_taxonomy",
    # traversal
    "parents_from_children", "bfs_depths", "reachable_set", "topological_order",
    "DECOMPOSITION_BUCKETS", "BLOCKING_CAUSES", "attribute_false_negative",
    "recall_decomposition", "delta_m_sensitivity",
    "reachability_recall", "reachability_recall_cohort",
    "blocking_depths", "blocking_depth_distribution", "depth_cost_table",
    # calibration
    "bin_edges", "bin_summary", "expected_calibration_error",
    "ece_equal_width", "ece_equal_mass", "roelofs_monotonic_sweep", "smooth_ece",
    "brier", "wilson_interval", "discrete_score_table", "signed_gap_bins",
    "conditional_calibration", "depth_groups", "layer1_groups", "frequency_buckets",
    "tail_calibration_error", "calibration_report",
    "fit_thresholds", "transfer_gap", "cohort_transfer", "kfold_report_split",
    "CalibrationMap", "fit_calibration_map", "CALIBRATION_METHODS",
    "f1_optimal_threshold_diagnostic",
    "compare_raw_vs_calibrated",
    # selective
    "coverage", "selective_risk", "generalized_risk", "risk_coverage_curve",
    "aurc", "augrc", "selective_report",
    # cost
    "slm_call_stats", "hardware_profile", "deployment_cost",
    "break_even_corpus_size", "scaling_curve",
]
