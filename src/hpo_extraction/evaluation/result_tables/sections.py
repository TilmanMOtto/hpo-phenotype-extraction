"""One function per ``thesis/sections/sceleton.tex`` subsection, in the skeleton's own order.

This module contains **no formulas**. Every number it returns is produced by
:mod:`hpo_extraction.evaluation.metrics`, which implements each skeleton equation from its statement and is
unit-tested against hand-worked examples. What lives here is the wiring: which artifact feeds which
equation, at which configuration, for which method, and the decisions the skeleton leaves open,
each recorded at the point it is made.

The four subsections map to the four builders below:

===============================  ============================================
skeleton subsection              builder
===============================  ============================================
§Core extraction quality         :func:`core_quality`, :func:`retrieval_quality`
§Hierarchy-aware quality         :func:`hierarchy_quality`, :func:`traversal_quality`
§Calibration                     :func:`calibration_quality`, :func:`transferability`
§Scalability                     :func:`scalability`
===============================  ============================================

Every builder returns lists of flat dicts, one dict per table row, with the configuration and
the method carried in the row rather than in the structure. ``report.py`` turns those into CSV,
LaTeX and markdown without knowing anything about metrics.

**The main configuration.** Methods with a sweep (τ_prune for the tree, vote-k for the
ensemble) get every point in the sweep table, and the point with the best micro-F1 is flagged
``is_headline``. That selection is an oracle: it reads the evaluation cohort's labels. Every table
that shows a main row also carries ``headline_selection = "oracle: best micro_f1 on this
cohort"``, and :mod:`report` prints it as a footnote, because a number chosen this way is not a
held-out number and the results chapter must not present it as one. All methods are treated
identically, so the comparison between them stays fair even though each row is optimistic.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from hpo_extraction.evaluation.metrics import (
    OntologyView,
    ancestor_count_distribution,
    aurc,
    augrc,
    blocking_depth_distribution,
    calibration_report,
    candidate_set_recall,
    combined_recall_bound,
    conditional_calibration,
    cophe_cohort,
    depth_cost_table,
    depth_groups,
    error_taxonomy,
    false_negative_taxonomy,
    flat_report,
    frequency_buckets,
    h_prf_cohort,
    layer1_groups,
    micro_prf,
    near_miss_distribution,
    reachability_recall_cohort,
    reachable_set,
    risk_coverage_curve,
    segment_pr_curve,
    selective_report,
    signed_gap_bins,
    slm_call_stats,
    term_pr_curve,
    transfer_gap,
)

from hpo_extraction.evaluation.result_tables.discovery import DERIVED_COHORTS, METHODS_BY_KEY

logger = logging.getLogger(__name__)

MAIN_RESULT_SELECTION = "oracle: best micro_f1 on this cohort"
MAIN_RESULT_OBJECTIVE = "micro_f1"


def _mark_main_result(rows: list[dict]) -> list[dict]:
    """Flag the best-micro-F1 row within each (method, cohort) group.

    Ties go to the first row in sweep order, the more conservative configuration for the tree
    (lower τ_prune visits more nodes) and the lower k for the ensemble. Deterministic either way.
    """
    best: dict[tuple[str, str], tuple[float, int]] = {}
    for i, row in enumerate(rows):
        key = (row["method"], row["cohort"])
        value = row.get(MAIN_RESULT_OBJECTIVE)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if key not in best or value > best[key][0]:
            best[key] = (value, i)

    for row in rows:
        row["is_headline"] = False
        row["headline_selection"] = MAIN_RESULT_SELECTION
    for _, i in best.values():
        rows[i]["is_headline"] = True
    return rows


#: Pseudo-keys a placeholder may carry when it is about a whole table, not one method.
_TABLE_WIDE = "all"


def _base_row(method: str, cohort: str, point: str) -> dict:
    """The identifying columns every row shares.

    strict: an unregistered method key raises, not producing a row labelled
    with a typo, which would show up in the thesis as a method nobody can find.
    """
    spec = METHODS_BY_KEY[method]
    return {
        "method": method,
        "label": spec.label,
        "experiment": spec.exp_id,
        "cohort": cohort,
        "operating_point": point or "-",
        "sweep_axis": spec.sweep_axis or "-",
    }


def placeholder_row(method: str, cohort: str, reason: str, **extra) -> dict:
    """A row that says *why* it is empty, so no blank in the results is unexplained.

    *method* may be the pseudo-key ``"all"`` when the gap is table-wide, a missing segment
    annotation file affects every method at once, and inventing seven identical rows would say
    less, not more.
    """
    if method in METHODS_BY_KEY:
        row = _base_row(method, cohort, "")
    else:
        row = {"method": method, "label": method, "experiment": "-",
               "cohort": cohort, "operating_point": "-", "sweep_axis": "-"}
    row.update({"status": "placeholder", "reason": reason, **extra})
    return row


# ── §Core extraction quality, eq. (1) ───────────────────────────────────────

def core_quality(runs: list[dict]) -> list[dict]:
    """Flat P/R/F1 at every configuration of every available (method, cohort) cell.

    Each *run* is ``{"method", "cohort", "point", "gold_sets", "pred_sets"}``, already aligned by
    :func:`loaders.aligned_sets`.

    Both macro conventions are carried. The skeleton defines macro as "evaluating per report and
    taking the unweighted mean", which is the mean of per-report F1 (``macro_f1``). The repo's
    legacy scorer instead reports the harmonic mean of the averaged P and R
    (``macro_f1_of_means``). They differ, sometimes materially, and printing one while citing the
    other is the sort of error a results chapter cannot recover from, so both are in the
    table.

    ``empty_convention="perfect"`` matches ``hpo_extraction.evaluation.set_metrics.calc_metric``, the convention every
    prior experiment in this repo scored under. Changing it here would silently break comparability
    with the earlier runs-earlier findings.
    """
    rows = []
    for run in runs:
        report = flat_report(run["gold_sets"], run["pred_sets"], empty_convention="perfect")
        row = _base_row(run["method"], run["cohort"], run["point"])
        row.update({
            "status": "ok",
            "n_reports": report["n_reports"],
            "tp": report["tp"], "fp": report["fp"], "fn": report["fn"],
            "micro_precision": report["micro_precision"],
            "micro_recall": report["micro_recall"],
            "micro_f1": report["micro_f1"],
            "macro_precision": report["macro_precision"],
            "macro_recall": report["macro_recall"],
            "macro_f1": report["macro_f1"],
            "macro_f1_of_means": report["macro_f1_of_means"],
            "macro_term_precision": report["macro_term_precision"],
            "macro_term_recall": report["macro_term_recall"],
            "macro_term_f1": report["macro_term_f1"],
            "macro_term_n_units": report["macro_term_n_units"],
        })
        rows.append(row)
    return _mark_main_result(rows)


#: The averaging RAG-HPO's published tables use. They compute precision, recall and F1 per case and
#: report the average of each (their Methods. Their released workbook reproduces 0.71 for
#: LLaMA-3 70B as the mean per-case F1, and 0.73 as the harmonic mean of the means). So their F1 is
#: ``macro_f1``, NOT ``macro_f1_of_means``. They also print pooled TP/FP/FN, from which
#: ``micro_f1`` follows -- the like-for-like column for our micro tables.
RAGHPO_COMPARISON_COLUMNS = ("macro_precision", "macro_recall", "macro_f1")


def _micro_f1_from_counts(entry: dict):
    """Micro F1 from a published row's pooled TP/FP/FN, or None when it prints no counts."""
    try:
        tp, fp, fn = (float(entry[k]) for k in ("tp", "fp", "fn"))
    except (KeyError, TypeError, ValueError):
        return None
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def raghpo_replication(core_rows: list[dict], published: list[dict]) -> list[dict]:
    """Our numbers on RAG-HPO's own corpus, beside the numbers they published on it.

    Takes the main row of every method on the two derived cohorts (see
    ``resources/data/GSC_RAGHPO/PROVENANCE.md``) and appends whatever reference rows the config
    supplies. *published* rows are **config data the author transcribed**, never anything this
    harness derived, the experiment carries no published comparison numbers of its own, and the
    ``source`` column says which is which on every row.

    Each published row is ``{label, tp, fp, fn, precision, recall, f1}``. Only ``label`` is
    required. A row missing a field leaves that cell blank, not guessing at it.
    """
    rows = []
    for row in core_rows:
        if row.get("cohort") not in DERIVED_COHORTS or not row.get("is_headline"):
            continue
        out = dict(row)
        out["source"] = "this work"
        rows.append(out)

    for entry in published:
        entry = dict(entry)
        row = {
            "method": "published", "label": entry.get("label", "?"), "experiment": "-",
            "cohort": entry.get("cohort", "gsc_raghpo_ann"),
            "operating_point": "-", "sweep_axis": "-", "status": "ok",
            "source": entry.get("source", "published"),
            "n_reports": entry.get("n_reports"),
            "tp": entry.get("tp"), "fp": entry.get("fp"), "fn": entry.get("fn"),
            "macro_precision": entry.get("precision"),
            "macro_recall": entry.get("recall"),
            "macro_f1": entry.get("f1"),
            "micro_f1": _micro_f1_from_counts(entry),
        }
        rows.append(row)
    return rows


def cohort_coverage(cohort: str, gold: dict[str, set[str]], view: OntologyView) -> dict:
    """The dataset-description numbers §Results asks for, computed from the ground-truth sets in hand.

    "How many HPOs (total + unique + avg. per report) + nodal depth (mean cohort and mean per
    case) + organ systems total + mean per report". Computed here, not cross-referenced
    from an earlier exploratory run so the results section is self-contained and so the coverage table describes
    the same reports the metrics above were computed on.
    """
    n_pairs = sum(len(v) for v in gold.values())
    unique = set().union(*gold.values()) if gold else set()

    depths = [d for terms in gold.values() for t in terms if (d := view.depth(t)) is not None]
    per_report_depth = [
        sum(ds) / len(ds)
        for terms in gold.values()
        if (ds := [d for t in terms if (d := view.depth(t)) is not None])
    ]
    organ_by_report = [
        {o for t in terms if (o := view.layer1(t)) is not None} for terms in gold.values()
    ]
    all_organs = set().union(*organ_by_report) if organ_by_report else set()

    resolved = {t for t in unique if view.resolve(t) is not None}
    return {
        "cohort": cohort,
        "n_reports": len(gold),
        "n_reports_annotated": sum(1 for v in gold.values() if v),
        "n_doc_term_pairs": n_pairs,
        "n_unique_terms": len(unique),
        "terms_per_report_mean": n_pairs / len(gold) if gold else 0.0,
        "depth_mean_cohort": sum(depths) / len(depths) if depths else None,
        "depth_mean_per_report": (sum(per_report_depth) / len(per_report_depth)
                                  if per_report_depth else None),
        "n_organ_systems_total": len(all_organs),
        "organ_systems_per_report_mean": (sum(len(o) for o in organ_by_report)
                                          / len(organ_by_report) if organ_by_report else 0.0),
        "n_terms_unresolvable": len(unique) - len(resolved),
        "ancestor_counts": ancestor_count_distribution(list(gold.values()), view),
    }


# ── §Core extraction quality, eq. (2), the retrieval stage ──────────────────

def retrieval_quality(
    method: str, cohort: str, pairs: list, s_values: list[int], pair_stats: dict,
) -> list[dict]:
    """Segment-level P@S / R@S across a range of S, per skeleton eq. (2).

    ``R@S`` is the quantity of interest, it upper-bounds the recall the identification stage can
    reach at any configuration, and it is one of the two factors in the combined recall bound.
    ``P@S`` divides by S literally, as the equation is written, which penalises reports with fewer
    than S segments; ``n_pairs_short`` counts those so the effect is visible, not baked in.

    The pair count is carried on every row because for HCY it is small (only the manually
    annotated patients have segment-level ground truth) and a P@S quoted without its n is not
    interpretable.
    """
    rows = []
    for s, stats in segment_pr_curve(pairs, s_values).items():
        row = _base_row(method, cohort, "")
        row.update({
            "status": "ok",
            "unit": "segment",
            "S": s,
            "precision_micro": stats["micro_precision"],
            "recall_micro": stats["micro_recall"],
            "precision_macro": stats["macro_precision"],
            "recall_macro": stats["macro_recall"],
            "n_pairs": stats["n_pairs"],
            "n_pairs_with_relevant": stats["n_pairs_with_relevant"],
            "n_pairs_short": stats["n_pairs_short"],
            "n_reports_annotated": pair_stats.get("n_reports_annotated"),
        })
        rows.append(row)
    return rows


def term_retrieval_quality(
    method: str, cohort: str, reports: list[tuple[list[str], set[str]]], m_values: list[int],
) -> list[dict]:
    """Term-level P@M / R@M, eq. (2) at the ontology-scale retrieval unit.

    This is the ceiling on the flat top-M method: an annotated term that never enters the top-M
    candidate list cannot be recovered by the SLM, whatever the accept threshold. Unlike the
    segment unit it needs only the ground-truth set, so it covers both cohorts in full.
    """
    rows = []
    for m, stats in term_pr_curve(reports, m_values).items():
        row = _base_row(method, cohort, "")
        row.update({
            "status": "ok",
            "unit": "term",
            "M": m,
            "precision_micro": stats["micro_precision"],
            "recall_micro": stats["micro_recall"],
            "precision_macro": stats["macro_precision"],
            "recall_macro": stats["macro_recall"],
            "n_reports": stats["n_reports"],
            "n_reports_with_gold": stats["n_reports_with_gold"],
        })
        rows.append(row)
    return rows


def traversal_candidate_recall(
    method: str, cohort: str, point: str,
    visited: dict[str, set[str]], gold: dict[str, set[str]],
) -> dict:
    """R@M for the traversal, whose candidate set is unranked.

    The BFS visits a set, not a ranked list, so P@M/R@M at a cutoff is undefined and the metric
    collapses to plain set recall over the visited nodes, which is the same quantity as
    reachability recall computed a different way, and the two are cross-checked in the tests.
    """
    recalls = [r for rid in visited if (r := candidate_set_recall(visited[rid],
                                                                 gold.get(rid, set()))) is not None]
    row = _base_row(method, cohort, point)
    row.update({
        "status": "ok",
        "unit": "term (unranked visited set)",
        "recall_macro": sum(recalls) / len(recalls) if recalls else None,
        "n_reports_with_gold": len(recalls),
        "mean_candidates_per_report": (sum(len(v) for v in visited.values()) / len(visited)
                                       if visited else 0.0),
    })
    return row


# ── §Hierarchy-aware quality, eqs. (3), (4), (5) ────────────────────────────

def hierarchy_quality(runs: list[dict], view: OntologyView) -> list[dict]:
    """CoPHE (primary) and ancestor-closure hP/hR/hF (for comparability), per eqs. (3) and (4).

    Both are reported because they fail differently. Closure ``hF`` is systematically optimistic:
    the closure introduces true positives at shallow depths that any system recovers. CoPHE
    propagates counts instead of set membership, so over-prediction inside a surviving subtree, the failure mode a traversal that accepts several siblings is most prone to, survives as
    ``FP_v`` at the shared ancestor, not being absorbed by set collapse.

    ``hF`` is never emitted without flat ``F_1`` beside it, per the skeleton's own note, and the
    ``|An(v)|`` audit that makes the DAG's multi-parent weighting visible is emitted alongside in
    :func:`cohort_coverage`.
    """
    rows = []
    for run in runs:
        gold_sets, pred_sets = run["gold_sets"], run["pred_sets"]
        cophe = cophe_cohort(gold_sets, pred_sets, view)
        closure = h_prf_cohort(gold_sets, pred_sets, view)
        flat = micro_prf(gold_sets, pred_sets)

        row = _base_row(run["method"], run["cohort"], run["point"])
        row.update({
            "status": "ok",
            "n_reports": cophe["n_reports"],
            "micro_f1_flat": flat[2],
            "micro_cophe_precision": cophe["micro_cophe_precision"],
            "micro_cophe_recall": cophe["micro_cophe_recall"],
            "micro_cophe_f1": cophe["micro_cophe_f1"],
            "macro_cophe_f1": cophe["macro_cophe_f1"],
            "micro_hp": closure["micro_hp"],
            "micro_hr": closure["micro_hr"],
            "micro_hf": closure["micro_hf"],
            "macro_hf": closure["macro_hf"],
            "hf_minus_f1": closure["micro_hf"] - flat[2],
        })
        rows.append(row)
    return _mark_main_result(rows)


def error_analysis(runs: list[dict], view: OntologyView) -> tuple[list[dict], list[dict]]:
    """``(taxonomy_rows, near_miss_rows)``, eq. (5) and the Garcia et al. decomposition.

    The five Garcia buckets (ancestor, descendant, sibling, unrelated, hallucination) are what
    make this error profile directly comparable to RAG-HPO's published decomposition, so a
    renormalised column over those five is emitted next to the raw fractions. Two extra
    hygiene buckets, ``out_of_subtree`` (a real HPO term outside the phenotypic-abnormality
    subtree) and ``no_gold`` (a false positive on a report with no annotated terms at all), are
    reported separately, not folded into "unrelated", which would assert more than is known.

    The false-negative mirror answers the question the FP taxonomy cannot: when an annotated term was
    missed, did the system predict something near it (a granularity error) or nothing at all (a
    detection failure)?
    """
    taxonomy_rows, near_miss_rows = [], []
    for run in runs:
        gold_sets, pred_sets = run["gold_sets"], run["pred_sets"]
        fp = error_taxonomy(gold_sets, pred_sets, view)
        fn = false_negative_taxonomy(gold_sets, pred_sets, view)
        near = near_miss_distribution(gold_sets, pred_sets, view)

        row = _base_row(run["method"], run["cohort"], run["point"])
        row.update({"status": "ok", "n_fp": fp["n_fp"], "n_fn": fn["n_fn"],
                    "n_garcia_classified": fp["n_garcia_classified"]})
        for bucket, count in fp["counts"].items():
            row[f"fp_{bucket}"] = count
            row[f"fp_{bucket}_frac"] = fp["fractions"][bucket]
        for bucket, frac in fp["garcia_fractions"].items():
            row[f"garcia_{bucket}_frac"] = frac
        for bucket, count in fn["counts"].items():
            row[f"fn_{bucket}"] = count
            row[f"fn_{bucket}_frac"] = fn["fractions"][bucket]
        taxonomy_rows.append(row)

        nm = _base_row(run["method"], run["cohort"], run["point"])
        nm.update({
            "status": "ok",
            "n_fp": near["n_fp"],
            "n_measurable": near["n_measurable"],
            "n_unmeasurable": near["n_unmeasurable"],
            "distance_mean": near["mean"],
            "distance_median": near["median"],
            "fraction_within_2": near["fraction_within_2"],
            "histogram": dict(sorted(near["histogram"].items())),
        })
        near_miss_rows.append(nm)

    return taxonomy_rows, near_miss_rows


# ── §Reachability recall, §Blocking depth, §Traversal cost by depth ──────────

def traversal_quality(
    method: str, cohort: str, point: str,
    gold: dict[str, set[str]],
    expanded: dict[str, set[str]],
    children_map: dict[str, list[str]],
    roots: list[str],
    compute_blocking: bool = True,
) -> tuple[dict, dict | None]:
    """``(reachability_row, blocking_row)``, skeleton eq. (6) and the blocking-depth diagnostic.

    Reachability recall is the ceiling on end-to-end recall: a subtree pruned at an internal node
    is unreachable no matter how competent the identification stage is at the leaf. Recomputing it
    from ``children_map`` + the recorded expand decisions, not reading the visited set
    straight off ``nodes.jsonl``, makes it an independent check on the traversal itself. The
    tests assert the two agree.

    Blocking depth localises the loss. Concentration at depth 1-2 means τ_prune is too aggressive
    near the roots, where a broad term's surface expression in a clinical narrative is weakest. A
    flat distribution means diffuse identification error instead. This is the diagnostic that ties
    the efficiency gains of §Scalability, which accrue almost entirely at shallow depths, to the
    recall they cost.

    Blocking depth is a DAG-wide dynamic program per report, so it is gated by *compute_blocking*
    for sweeps where only the main point is wanted.
    """
    rids = sorted(set(gold) & set(expanded))
    gold_sets = [gold[r] for r in rids]
    expanded_sets = [expanded[r] for r in rids]
    reachable_sets = [reachable_set(children_map, roots, e) for e in expanded_sets]

    reach = reachability_recall_cohort(gold_sets, reachable_sets)
    row = _base_row(method, cohort, point)
    row.update({
        "status": "ok",
        "n_reports": len(rids),
        "micro_reachability_recall": reach["micro_reachability_recall"],
        "macro_reachability_recall": reach["macro_reachability_recall"],
        "n_gold_terms": reach["n_gold_terms"],
        "n_gold_reachable": reach["n_gold_reachable"],
        "mean_expanded_per_report": (sum(len(e) for e in expanded_sets) / len(expanded_sets)
                                     if expanded_sets else 0.0),
    })

    if not compute_blocking:
        return row, None

    dist = blocking_depth_distribution(gold_sets, children_map, roots, expanded_sets)
    blocking = _base_row(method, cohort, point)
    blocking.update({
        "status": "ok",
        "n_blocked": dist["n_blocked"],
        "n_reachable": dist["n_reachable"],
        "n_outside_graph": dist["n_outside_graph"],
        "depth_mean": dist["mean"],
        "depth_median": dist["median"],
        "fraction_at_depth_1": dist["fraction_at_depth_1"],
        "histogram": dict(sorted(dist["histogram"].items())),
    })
    return row, blocking


def depth_cost(method: str, cohort: str, point: str, visits_by_report: list[dict],
               gold_sets: list[set[str]] | None = None) -> list[dict]:
    """The per-depth frontier / survival / cumulative-cost table of §Traversal cost by depth."""
    rows = []
    for entry in depth_cost_table(visits_by_report, gold_sets):
        row = _base_row(method, cohort, point)
        row.update({"status": "ok", **entry})
        rows.append(row)
    return rows


def recall_cost_pareto(core_rows: list[dict], reach_rows: list[dict],
                       slm_call_rows: list[dict]) -> list[dict]:
    """Recall against SLM calls per report, the efficiency/recall exchange, one row per τ.

    The skeleton's cost axis is SLM invocations, not wall-clock: it is hardware-independent and
    reproducible, which is what lets this curve be compared against a future run on
    different hardware. The combined recall bound (retrieval recall × reachability recall) is
    included where both factors are available, since it is the ceiling the measured recall should
    be read against.

    Args:
        slm_call_rows: the **``slm_calls``** table, one row per (method, cohort, operating
            point). Not the ``scalability`` hardware table, whose rows carry
            ``operating_point = ""`` and would therefore join against nothing for every swept
            method, silently emptying the cost axis while every row still appears.
    """
    if any("seconds_per_report" in r for r in slm_call_rows):
        raise ValueError(
            "recall_cost_pareto expects the slm_calls table (one row per operating point); it "
            "was given the scalability hardware table, whose rows carry operating_point='' and "
            "so join against nothing for every swept method — the cost axis would come out "
            "empty with no row missing to show it")

    by_key = {(r["method"], r["cohort"], r["operating_point"]): r for r in reach_rows}
    cost_by_key = {(r["method"], r["cohort"], r["operating_point"]): r for r in slm_call_rows}

    rows = []
    for core in core_rows:
        key = (core["method"], core["cohort"], core["operating_point"])
        reach = by_key.get(key)
        cost = cost_by_key.get(key)
        if reach is None and cost is None:
            continue
        row = _base_row(core["method"], core["cohort"], core["operating_point"])
        row.update({
            "status": "ok",
            "micro_recall": core.get("micro_recall"),
            "micro_precision": core.get("micro_precision"),
            "micro_f1": core.get("micro_f1"),
            "is_headline": core.get("is_headline", False),
            "reachability_recall": (reach or {}).get("micro_reachability_recall"),
            "slm_calls_per_report": (cost or {}).get("slm_calls_per_report"),
            "unique_nodes_per_report": (cost or {}).get("unique_nodes_per_report"),
        })
        rows.append(row)
    return rows


def combined_bound(retrieval_recall: float | None,
                   reachability: float | None) -> float | None:
    """The product bound §Hierarchical Candidate Traversal refers to, when both factors exist."""
    if retrieval_recall is None or reachability is None:
        return None
    return combined_recall_bound(retrieval_recall, reachability)


# ── §Calibration, eq. (7), signed/conditional, tail ─────────────────────────

def calibration_quality(
    method: str, cohort: str, point: str, sample: dict, view: OntologyView,
    gold: dict[str, set[str]], score_name: str, tau_prune: float | None, delta: float,
) -> tuple[dict, list[dict], list[dict]]:
    """``(summary_row, signed_gap_rows, conditional_rows)`` for one score of one run.

    The skeleton reports plain ECE "because it is the field's default" while stating three reasons
    it is inadequate here, so all of the alternatives it names are computed together:
    equal-mass binning with the bin count chosen by the Roelofs monotonicity-preserving sweep,
    bin-free SmoothECE, and, because ``c(p_i)`` aggregates S binary responses and therefore has at
    most ``S+1`` atoms, the discrete score table with Wilson intervals, which is the binning-free
    limit of a reliability diagram for a score like this.

    The **tail** restriction is the one that counts for the pruning argument. τ_prune operates in
    the low-confidence tail while ECE is dominated by the densely populated high-confidence region,
    so an aggregate ECE can look excellent while the region the traversal actually depends on is
    badly calibrated. ``tail_upper = tau_prune + delta``.

    Group-conditional calibration is reported by ontology depth, top-level organ system and gold-set
    term frequency, because "aggregate ECE cannot detect group-conditional miscalibration" and this
    system's transferability claim is regional by design.
    """
    conf, labels = sample["confidences"], sample["labels"]
    hpo_ids = sample["hpo_ids"]

    tail_upper = None
    if tau_prune is not None:
        tail_upper = min(1.0, tau_prune + delta)

    report = calibration_report(conf, labels, tail_upper=tail_upper)
    row = _base_row(method, cohort, point)
    row.update({
        "status": "ok",
        "score": score_name,
        "label_semantics": ("node present" if score_name == "accept_score"
                            else "subtree contains a gold term"),
        "n": report["n"],
        "prevalence": report["prevalence"],
        "mean_confidence": report["mean_confidence"],
        "ece_equal_width": report["ece_equal_width"],
        "ece_equal_mass": report["ece_equal_mass"],
        "ece_roelofs": report["ece_roelofs"],
        "roelofs_n_bins": report["roelofs_n_bins"],
        "smooth_ece": report["smooth_ece"],
        "brier": report["brier"],
        "n_distinct_scores": report["n_distinct_scores"],
    })
    if "tail" in report:
        row.update({
            "tail_upper": report["tail"]["upper"],
            "tail_coverage": report["tail"]["coverage"],
            "tail_n": report["tail"]["n"],
            "tail_ece": report["tail"]["ece"],
            "tail_prevalence": report["tail"]["prevalence"],
            "tail_mean_confidence": report["tail"]["mean_confidence"],
        })

    gap_rows = []
    for entry in signed_gap_bins(conf, labels, binning="equal_mass"):
        gap = _base_row(method, cohort, point)
        gap.update({"status": "ok", "score": score_name, **entry})
        gap_rows.append(gap)

    cond_rows = []
    groupings = {
        "depth": depth_groups(hpo_ids, view),
        "organ_system": layer1_groups(hpo_ids, view),
        "term_frequency": frequency_buckets(hpo_ids, list(gold.values())),
    }
    for grouping, keys in groupings.items():
        result = conditional_calibration(conf, labels, keys)
        for group_key, stats in sorted(result["groups"].items(), key=lambda kv: str(kv[0])):
            cond = _base_row(method, cohort, point)
            cond.update({
                "status": "ok", "score": score_name,
                "grouping": grouping, "group": str(group_key),
                "aggregate_ece": result["aggregate_ece"],
                "worst_group": str(result["worst_group"]),
                "max_minus_aggregate_ece": result["max_minus_aggregate_ece"],
                **stats,
            })
            cond_rows.append(cond)

    return row, gap_rows, cond_rows


def discrete_reliability(method: str, cohort: str, point: str, sample: dict,
                         score_name: str) -> list[dict]:
    """One row per attainable score atom, with its Wilson 95% interval.

    "Where S is small enough that the score support is enumerable, we additionally tabulate the
    empirical presence frequency and Wilson interval at each attainable score." For the ensemble
    (9 atoms) and the AnyYes accept score (S+1 atoms) this *is* the reliability diagram. Binning
    it would only blur it.
    """
    from hpo_extraction.evaluation.metrics import discrete_score_table

    rows = []
    for entry in discrete_score_table(sample["confidences"], sample["labels"]):
        row = _base_row(method, cohort, point)
        # ``discrete_score_table`` names the atom value ``score``, which would collide with the
        # column naming *which* score this is. The atom is renamed, not the column, so
        # every calibration table keys on ``score`` consistently.
        entry = dict(entry)
        row.update({"status": "ok", "score": score_name, "score_atom": entry.pop("score"),
                    **entry})
        rows.append(row)
    return rows


# ── §Risk--coverage analysis, eq. (8) ───────────────────────────────────────

def risk_coverage(method: str, cohort: str, point: str,
                  sample: dict, score_name: str) -> tuple[dict, list[dict]]:
    """``(summary_row, curve_rows)``, coverage, selective risk, AURC and AUGRC.

    The unit is one evaluated (report, term) cell, and the loss on a covered cell is
    ``1[term not in Y_r]``, a covered cell is by design reported, so ``R(τ)`` reduces to
    ``1 − micro-precision(τ)``. That identity is fixed by a test in the metrics package and the
    curve carries ``precision`` per row so it is visible in the table, not merely asserted.

    AUGRC is reported next to AURC because AURC over-weights high-confidence failures in a way
    that can invert method rankings, and because it conflates discrimination with calibration:
    methods differ in accuracy at full coverage. ``risk_at_full_coverage`` is emitted as the
    evidence location that makes the confound readable, an AURC cannot be compared across two methods whose
    full-coverage risks differ without it.
    """
    conf, labels = sample["confidences"], sample["labels"]
    report = selective_report(conf, labels)

    row = _base_row(method, cohort, point)
    row.update({
        "status": "ok",
        "score": score_name,
        "n": report["n"],
        "prevalence": report["prevalence"],
        "aurc": report["aurc"],
        "augrc": report["augrc"],
        "risk_at_full_coverage": report["risk_at_full_coverage"],
        "coverage_at_lowest_tau": report.get("coverage_at_lowest_tau"),
        "aurc_confound_note": (
            "AURC conflates discrimination with calibration; compare only against "
            "risk_at_full_coverage"),
    })

    curve_rows = []
    for entry in risk_coverage_curve(conf, labels):
        curve = _base_row(method, cohort, point)
        curve.update({"status": "ok", "score": score_name, **entry})
        curve_rows.append(curve)

    return row, curve_rows


def aurc_pair(sample: dict) -> tuple[float, float]:
    """``(AURC, AUGRC)`` for a sample, used by the transferability comparison."""
    return aurc(sample["confidences"], sample["labels"]), augrc(
        sample["confidences"], sample["labels"])


# ── §Threshold transferability ───────────────────────────────────────────────

def _offline_evaluator(nodes_by_tau: dict[float, dict[str, list[dict]]],
                      gold: dict[str, set[str]], report_ids: set[str] | None = None):
    """Build an ``evaluate(tau_prune, tau_accept) -> {"micro_f1": ...}`` closure.

    Re-run is **exact**, not approximate: ``tau_*/{variant}_nodes.jsonl`` holds every visited
    node's ``accept_score``, so applying a new τ_accept reproduces the prediction set that run
    would have produced. τ_prune, by contrast, can only take the values that were actually swept,
    because it changed which nodes were visited at all, the grid is therefore the sweep, and
    :func:`fit_thresholds` is handed those values.
    """
    def evaluate(tau_prune: float, tau_accept: float) -> dict[str, float]:
        nodes = nodes_by_tau[tau_prune]
        rids = sorted(set(nodes) & set(gold))
        if report_ids is not None:
            rids = [r for r in rids if r in report_ids]
        gold_sets = [gold[r] for r in rids]
        pred_sets = [
            {rec["hpo_id"] for rec in nodes[r] if float(rec["accept_score"]) >= tau_accept}
            for r in rids
        ]
        p, r, f1 = micro_prf(gold_sets, pred_sets)
        return {"micro_precision": p, "micro_recall": r, "micro_f1": f1, "n_reports": len(rids)}

    return evaluate


def transferability(
    method: str,
    nodes_by_cohort: dict[str, dict[float, dict[str, list[dict]]]],
    gold_by_cohort: dict[str, dict[str, set[str]]],
    tau_accept_grid: list[float],
    k_folds: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Fit a threshold pair on one split, evaluate on a held-out one, cross-cohort and K-fold.

    The claim being tested is narrow and worth restating: monotone recalibration does not reorder
    scores, so it cannot improve discrimination. What it can do is make a *single threshold pair
    portable*. The measured quantity is therefore the **gap** between the transferred pair's score
    on the held-out split and the oracle pair's score on that same split, not the absolute score,
    which would mostly reflect how hard the held-out data is.

    Two protocols, per the plan: HCY ↔ GSC+ (the harder test, different institution, different
    annotation convention) and K-fold over reports within a cohort (the easier one, which isolates
    sampling noise from distribution shift).
    """
    from hpo_extraction.evaluation.metrics import cohort_transfer, kfold_report_split

    rows: list[dict] = []
    cohorts = [c for c in nodes_by_cohort if nodes_by_cohort[c] and gold_by_cohort.get(c)]

    # ── Cross-cohort ──
    if len(cohorts) >= 2:
        shared_taus = sorted(set.intersection(*(set(nodes_by_cohort[c]) for c in cohorts)))
        if shared_taus:
            evaluators = {
                c: _offline_evaluator(nodes_by_cohort[c], gold_by_cohort[c]) for c in cohorts
            }
            result = cohort_transfer(evaluators, shared_taus, tau_accept_grid)
            for pair, gap in result["pairs"].items():
                fit_c, eval_c = pair.split("->")
                row = _base_row(method, f"{fit_c}->{eval_c}", "")
                row.update({
                    "status": "ok",
                    "protocol": "cross-cohort",
                    "fit_on": fit_c, "evaluated_on": eval_c,
                    "fitted_tau_prune": gap["fitted_tau_prune"],
                    "fitted_tau_accept": gap["fitted_tau_accept"],
                    "fit_value": gap["fit_value"],
                    "transferred_value": gap["transferred_value"],
                    "oracle_tau_prune": gap["oracle_tau_prune"],
                    "oracle_tau_accept": gap["oracle_tau_accept"],
                    "oracle_value": gap["oracle_value"],
                    "gap": gap["gap"],
                })
                rows.append(row)
        else:
            rows.append(placeholder_row(
                method, "cross-cohort",
                "the two cohorts share no swept tau_prune value, so no threshold pair can be "
                "transferred between them",
                protocol="cross-cohort"))
    else:
        rows.append(placeholder_row(
            method, "cross-cohort",
            "cross-cohort transfer needs both HCY and GSC+ node artifacts; "
            f"available: {', '.join(cohorts) or 'none'}",
            protocol="cross-cohort"))

    # ── K-fold over reports, within each cohort ──
    for cohort in cohorts:
        taus = sorted(nodes_by_cohort[cohort])
        report_ids = sorted(set(gold_by_cohort[cohort]) & {
            r for tau in taus for r in nodes_by_cohort[cohort][tau]
        })
        if len(report_ids) < k_folds:
            rows.append(placeholder_row(
                method, cohort,
                f"{len(report_ids)} reports is fewer than k={k_folds}; K-fold transfer skipped",
                protocol=f"{k_folds}-fold"))
            continue

        gaps = []
        for fold, (fit_ids, eval_ids) in enumerate(
                kfold_report_split(report_ids, k=k_folds, seed=seed)):
            gap = transfer_gap(
                _offline_evaluator(nodes_by_cohort[cohort], gold_by_cohort[cohort], set(fit_ids)),
                _offline_evaluator(nodes_by_cohort[cohort], gold_by_cohort[cohort], set(eval_ids)),
                taus, tau_accept_grid)
            gaps.append(gap)
            row = _base_row(method, cohort, f"fold{fold}")
            row.update({
                "status": "ok",
                "protocol": f"{k_folds}-fold",
                "fit_on": f"{cohort} fit fold {fold}", "evaluated_on": f"{cohort} held-out {fold}",
                "fitted_tau_prune": gap["fitted_tau_prune"],
                "fitted_tau_accept": gap["fitted_tau_accept"],
                "fit_value": gap["fit_value"],
                "transferred_value": gap["transferred_value"],
                "oracle_tau_prune": gap["oracle_tau_prune"],
                "oracle_tau_accept": gap["oracle_tau_accept"],
                "oracle_value": gap["oracle_value"],
                "gap": gap["gap"],
            })
            rows.append(row)

        if gaps:
            summary = _base_row(method, cohort, "mean")
            summary.update({
                "status": "ok",
                "protocol": f"{k_folds}-fold (mean)",
                "fit_on": cohort, "evaluated_on": cohort,
                "transferred_value": sum(g["transferred_value"] for g in gaps) / len(gaps),
                "oracle_value": sum(g["oracle_value"] for g in gaps) / len(gaps),
                "gap": sum(g["gap"] for g in gaps) / len(gaps),
                "gap_max": max(g["gap"] for g in gaps),
            })
            rows.append(summary)

    return rows


def calibrated_vs_raw(calibrated_rows: list[dict], raw_rows: list[dict],
                      calibrated_method: str, raw_method: str) -> list[dict]:
    """Compare the transfer gap of a calibrated score against an uncalibrated one.

    The skeleton requires the raw aggregated score be run through the *identical* protocol, which
    earlier supplies as a natural experiment: an earlier exploratory run's LR gate is calibrated, an earlier exploratory run's noisyOR is
    the same traversal with an uncalibrated prune score. A smaller gap for the calibrated score is
    the evidence for portability. Anything else falsifies the claim, and either way the number
    belongs in the results.
    """
    from hpo_extraction.evaluation.metrics import compare_raw_vs_calibrated

    by_key = {(r["cohort"], r.get("protocol"), r["operating_point"]): r for r in raw_rows}
    rows = []
    for cal in calibrated_rows:
        if cal.get("status") != "ok" or "gap" not in cal:
            continue
        raw = by_key.get((cal["cohort"], cal.get("protocol"), cal["operating_point"]))
        if raw is None or "gap" not in raw:
            continue
        result = compare_raw_vs_calibrated(cal, raw)
        row = _base_row(calibrated_method, cal["cohort"], cal["operating_point"])
        row.update({
            "status": "ok",
            "protocol": cal.get("protocol"),
            "calibrated_method": calibrated_method,
            "raw_method": raw_method,
            **result,
        })
        rows.append(row)
    return rows


# ── §Scalability ─────────────────────────────────────────────────────────────

def slm_cost(method: str, cohort: str, point: str,
             traversal_records: list[dict]) -> dict[str, Any]:
    """SLM calls per report, with the DAG deduplication the skeleton requires be auditable.

    "Because H is a DAG, a term may be inserted into the frontier along several paths, and we
    report unique nodes evaluated and total frontier insertions separately so that deduplication
    is auditable." ``dedup_ratio`` is insertions ÷ unique nodes. A ratio near 1 means the DAG
    behaves like a tree for these reports, and a large one means memoisation is doing real work.
    """
    stats = slm_call_stats(traversal_records)
    row = _base_row(method, cohort, point)
    row.update({
        "status": "ok",
        "n_reports": stats["n_reports"],
        "slm_calls_per_report": stats["slm_calls_per_report"]["mean"],
        "slm_calls_median": stats["slm_calls_per_report"]["median"],
        "slm_calls_max": stats["slm_calls_per_report"]["max"],
        "unique_nodes_per_report": stats["unique_nodes_per_report"]["mean"],
        "frontier_insertions_per_report": stats["frontier_insertions_per_report"]["mean"],
        "dedup_ratio": stats["dedup_ratio"],
        "calls_per_node": stats["calls_per_node"],
        "_raw": stats,
    })
    return row


def deployment(gpu_hours: float | None, index_bytes: int | None,
               n_terms_indexed: int | None, n_context_sentences: int | None,
               baseline_seconds_per_report: float | None,
               method_seconds_per_report: float | None) -> dict:
    """Deployment cost and the break-even corpus size.

    "Contextual database generation over the reachable ontology is a substantial one-time expense
    incurred per ontology version, and reporting inference cost alone would misstate the system's
    economics." The break-even corpus size is the quantity that actually decides whether the
    approach is economical for a given archive.

    ``gpu_hours`` has no derivable source in the earlier runs, it is a property of the earlier context-generation jobs, so it comes from the config or comes back ``None``.
    """
    from hpo_extraction.evaluation.metrics import break_even_corpus_size, deployment_cost

    row: dict = {"status": "ok", **deployment_cost(
        gpu_hours, index_bytes, n_terms_indexed, n_context_sentences)}

    if (gpu_hours is not None and baseline_seconds_per_report is not None
            and method_seconds_per_report is not None):
        row.update(break_even_corpus_size(
            gpu_hours, baseline_seconds_per_report, method_seconds_per_report))
    else:
        missing = [n for n, v in (
            ("deployment_gpu_hours", gpu_hours),
            ("baseline_seconds_per_report", baseline_seconds_per_report),
            ("method_seconds_per_report", method_seconds_per_report)) if v is None]
        row["break_even_reason"] = (
            "break-even needs " + ", ".join(missing) + "; not recorded by any exp13 run")
    return row


def scaling(points: list[tuple[float, float]]) -> dict:
    """SLM calls per report against label-space size N, sublinearity, measured not asserted.

    "This is the measurement that distinguishes a scalability claim from a runtime report."
    Producing the points needs traversal runs at increasing ``restrict_to_subtree`` sizes. The
    config key exists but no earlier cluster script sweeps it, so this is normally a placeholder.
    """
    from hpo_extraction.evaluation.metrics import scaling_curve

    if len(points) < 2:
        return {
            "status": "placeholder",
            "reason": (
                "needs >=2 traversal runs at different label-space sizes. No exp13 run varies "
                "restrict_to_subtree; produce them with e.g. "
                "`python experiments/exp13_00_tree_gate_lr/run.py restrict_to_subtree=HP:0000707 "
                "...` for several layer-1 subtrees of increasing size."),
            "n_points": len(points),
        }
    return {"status": "ok", **scaling_curve(points)}


def negation_placeholder() -> dict:
    """The §Results item this experiment does not measure.

    "Item strength analysis → when it comes to negations, family history, close HPO neighbours."
    Close-neighbour behaviour *is* measured, by eq. (5) and the Garcia taxonomy. Negation and
    family-history attribution are not: they require linguistic analysis of the segment that
    triggered each false positive, and while the segment is recoverable (``sent_index`` in
    ``*_calls.jsonl`` indexes the Stanza segmentation this experiment re-derives), any bucketing
    would rest on a cue-list heuristic whose own error rate is unmeasured. Reporting a heuristic's
    output as a measurement is the thing this experiment exists to avoid.
    """
    return {
        "status": "placeholder",
        "item": "negation / family-history attribution of false positives",
        "covered_instead": (
            "close-neighbour errors are measured in full by eq. (5) near-miss distance and the "
            "Garcia ancestor/descendant/sibling/unrelated/hallucination taxonomy"),
        "reason": (
            "requires linguistic analysis of the triggering segment. The join is feasible — "
            "sent_index in *_calls.jsonl indexes the same Stanza segmentation segments.py "
            "re-derives — but any cue-list bucketing would have an unmeasured error rate."),
        "to_produce": (
            "a separate experiment with an annotated negation/family-history subset, so the "
            "heuristic's own precision and recall can be reported alongside its output"),
    }


def summarise_counts(rows: list[dict]) -> Counter:
    """``{status: n}`` over a table, how much of it is real measurement."""
    return Counter(r.get("status", "unknown") for r in rows)
