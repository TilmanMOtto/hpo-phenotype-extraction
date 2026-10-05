"""The chapter-4 stages, as functions over a :class:`~hpo_extraction.treephenorag.selection.ReplaySpace`.

Kept out of ``run.py`` so each stage is callable from a test without Hydra, an output directory or
a 6 GB cache. ``run.py`` owns configuration, IO and ordering. This module owns what each stage
actually computes.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import replace

import numpy as np

from hpo_extraction.treephenorag.stored_scores import POOLINGS
from hpo_extraction.treephenorag.selection import (
    Configuration,
    nested_evaluation,
    select_acceptance,
    select_prune_pooling,
    select_tau_prune,
    selection_stability,
)
from hpo_extraction.evaluation.stats import (ReportResampler, bootstrap_reports, holm,
                              paired_randomisation)
from hpo_extraction.evaluation.metrics import (
    calibration_report,
    error_taxonomy_existential,
    f1_optimal_threshold_diagnostic,
    fit_calibration_map,
    macro_prf_by_report,
    micro_prf,
    near_miss_distribution,
    recall_decomposition,
    transfer_gap,
)

logger = logging.getLogger("exp13_22")


# ── Scoring helpers ──────────────────────────────────────────────────────────

def units_for(predicted, gold, report_ids, normalise):
    """``[(gold_set, pred_set)]`` per report, after the ontology reduction of §Metrics."""
    out = []
    for rid in report_ids:
        g, p = normalise(gold.get(rid, set()), predicted.get(rid, set()))
        out.append((g, p))
    return out


def metric_fn(units):
    """Micro and macro P/R/F1 over the same units.

    Both averagings are computed here rather than in two places because ``bootstrap_reports``
    resamples whatever this returns: adding a key gives it a point estimate and an interval for
    free, and, more importantly, guarantees the macro figure is over the *same* units as the
    micro one. The macro convention is the mean of the per-report triples (eq. 2's
    ``macro over reports``), which is the convention ``comparison``'s ``t1_overall.csv`` reports
    every other system under. Anything else would put two different quantities in one table.
    """
    gold = [g for g, _ in units]
    pred = [p for _, p in units]
    p, r, f = micro_prf(gold, pred)
    ma_p, ma_r, ma_f, _ = macro_prf_by_report(gold, pred)
    return {"micro_precision": p, "micro_recall": r, "micro_f1": f,
            "macro_precision": ma_p, "macro_recall": ma_r, "macro_f1": ma_f}


def score_units(label, units, sampler, extra=None):
    """One row of ``core_quality.csv``, with a report-level bootstrap interval.

    Only micro-$F_1$ carries an interval in the table: it is the column every comparison in the
    chapter is made on, and five more interval pairs would double the table's width to support no
    claim. The macro columns are point estimates, for comparability with published figures.
    """
    ci = bootstrap_reports(metric_fn, units, resampler=sampler)
    row = {
        "config": label,
        "micro_precision": ci["micro_precision"]["point"],
        "micro_recall": ci["micro_recall"]["point"],
        "micro_f1": ci["micro_f1"]["point"],
        "micro_f1_lo": ci["micro_f1"]["lo"],
        "micro_f1_hi": ci["micro_f1"]["hi"],
        "macro_precision": ci["macro_precision"]["point"],
        "macro_recall": ci["macro_recall"]["point"],
        "macro_f1": ci["macro_f1"]["point"],
        "preds_per_report": sum(len(p) for _, p in units) / max(1, len(units)),
        **(extra or {}),
    }
    logger.info("%-46s F1=%.4f [%.4f, %.4f]  P=%.4f R=%.4f  macro F1=%.4f", label,
                row["micro_f1"], row["micro_f1_lo"], row["micro_f1_hi"],
                row["micro_precision"], row["micro_recall"], row["macro_f1"])
    return row


# ── The pooled out-of-fold configuration ───────────────────────────────────
#
# Table 1's main is the pooled out-of-fold prediction set of repetition 0: every report is
# predicted once, by the configuration ITS outer fold selected. Every other number the
# chapter quotes "at the configuration" -- calls, coverage, the S and pooling ablations,
# The recall decomposition, the false-positive taxonomy -- has to be measured on that same set, or
# The chapter reports three call counts and two F1s for one method (it did: 31 228, 31 333 and
# 30 735 calls; 0.3135 and 0.3151). So these helpers carry the per-report configuration rather
# than one "selected" configuration applied in-sample to every report.

def fold_assignments(folds, configs, repetition=0):
    """``[{"train", "eval", "config"}]`` for one repetition's outer folds, in fold order.

    ``configs`` is aligned with ``folds`` (one selected :class:`Configuration` per fold row), which
    is how both :func:`run_ladder` and :func:`rep0_assignments` produce it.
    """
    return [{"train": list(row["train_ids"]), "eval": list(row["eval_ids"]), "config": config}
            for row, config in zip(folds, configs) if row["repetition"] == repetition]


def rep0_assignments(folds, choices):
    """:func:`fold_assignments` from ``nested_evaluation``'s ``choices`` records."""
    by_key = {(c["repetition"], c["outer_fold"]): Configuration(
        c["retrieval_index"], c["pool_pr"], float(c["tau_prune"]), c["pool_acc"],
        float(c["tau_accept"]), int(c["S"])) for c in choices}
    return fold_assignments(folds, [by_key[(row["repetition"], row["outer_fold"])]
                                    for row in folds])


def pooled_predictions(space, assignments):
    """``{report: accepted terms}``, each held-out report under its own fold's configuration."""
    out = {}
    for a in assignments:
        predicted = space.predictions(a["config"])
        out.update({r: predicted.get(r, set()) for r in a["eval"]})
    return out


def nested_operating_point(space, assignments, gold):
    """Verifier calls and reachability of the pooled out-of-fold traversal.

    ``calls`` is one count per held-out report under its own fold's ``(index, pool_pr, tau_prune,
    S)``. Coverage is reported both ways: **micro** pools annotated terms over reports (the
    share of all annotated terms the traversal reaches), **macro** is ``1 -`` the mean per-report miss
    rate ``L_i``, which is the quantity conformal risk control bounds. A report with an empty ground truth
    set has ``L_i = 0`` by the loss's own definition, as in the calibration step.
    """
    from hpo_extraction.treephenorag.stored_scores import calls_at

    calls, losses, weights = [], [], []
    for a in assignments:
        config = a["config"]
        axis = space.prune_axis(config.index, config.pool_pr, config.s)
        ids = [r for r in a["eval"] if r in axis.r]
        if not ids:
            continue
        calls.extend(calls_at(axis.r[r], axis.n_valid[r], config.tau_prune) for r in ids)
        losses.extend(space.gold_miss_rates(axis, gold, ids,
                                            np.array([config.tau_prune]))[:, 0].tolist())
        weights.extend(len(gold.get(r, ())) for r in ids)
    calls = np.asarray(calls, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return {
        "calls": calls,
        "calls_per_report": float(calls.mean()) if calls.size else float("nan"),
        "attained_coverage_micro": (1.0 - float((losses * weights).sum() / weights.sum()))
        if weights.sum() else float("nan"),
        "attained_coverage_macro": 1.0 - float(losses.mean()) if losses.size else float("nan"),
        "held_out_miss_rate": float(losses.mean()) if losses.size else float("nan"),
        "n_reports": int(calls.size),
    }


# ── T4.1 the gate: does retrieval reach the annotated term at all ─────────────────

def _hit_at_s(cache, node, evidence_segments, s):
    """Did any of ``node``'s top-``s`` retrieved segments carry the curator's own evidence?

    ``sent_index[i, j]`` is *which sentence of the report* was retrieved at rank ``j+1`` for node
    ``i``. That column is the whole reason this measurement is decidable: without it a miss at a
    annotated term is indistinguishable from the verifier having read the wrong sentence, and the split
    the chapter rests on cannot be made. It is one of the four columns ``ingest_score_cache``
    keeps and ``load_score_cache`` discards.

    Returns ``None`` when the quantity is undefined for this (report, term), the node was never
    retrieved for, or the cache carries no ``sent_index``, so the caller can keep undefined and
    "defined and missed" apart instead of scoring them together.
    """
    if cache is None or cache.sent_index is None or not evidence_segments:
        return None
    position = cache.index.get(node)
    if position is None:
        return None
    width = min(int(s), cache.s_max)
    valid = cache.mask[position, :width]
    if not valid.any():
        return None
    retrieved = cache.sent_index[position, :width][valid.astype(bool)]
    return bool(set(int(x) for x in retrieved if x >= 0) & evidence_segments)


def retrieval_sufficiency(space, gold, evidence, s_values, sampler=None, view=None):
    """T4.1, the ceiling every recall number in the chapter sits under.

    Two rates per (index, S), and the difference between them is the point:

    ``hit_at_s``        the annotated term's own retrieval reached the curator's evidence segment;
    ``path_hit_rate``   *every* term on some root-to-ground truth path did.

    The second is what the traversal actually depends on. An annotated term whose own retrieval is
    perfect is still unreachable if an ancestor's retrieval failed, because the ancestor is where
    the traversal would have had to expand, so the path-wise rate, not the pointwise one, is the
    real bound. Reporting only the first would overstate the ceiling by the amount the
    pruning rule cannot recover.

    Both carry a **cluster bootstrap over reports**, not over pairs: two annotated terms in one report
    share a document, an author and a segmentation, so treating them as independent would shrink
    the intervals by roughly the square root of the mean pairs per report. ``sampler`` is the
    shared :class:`ReportResampler` so these intervals are comparable with every other table's.

    The paired difference ``I_ex - I_ont`` is computed report-by-report on the reports both
    indices cover, never by subtracting two marginals: the indices are evaluated on slightly
    different report sets when one cache is short, and a difference of marginals would silently
    mix that in.
    """
    if not evidence:
        logger.warning("T4.1 retrieval sufficiency needs evidence segments (evidence_path); "
                       "without them Hit@S is not measurable and the stage is skipped")
        return []

    # Per-report, per-S hit vectors, so the bootstrap resamples reports and the paired difference
    # is taken inside a report.
    def rates_for(index, s):
        """``{report: (n_hit, n_gold, n_path_hit)}`` over the reports this index covers."""
        caches = space.caches_by_index[index]
        out = {}
        for rid in space.report_ids(index):
            cache = caches.get(rid)
            ev = evidence.get(rid) or {}
            n_hit = n_path = n_terms = 0
            for raw in gold.get(rid, ()):
                term = (view.resolve(raw) or raw) if view is not None else raw
                segments = {int(x) for x in ev.get(term, ev.get(raw, ()))}
                if not segments:
                    continue                      # undefined, not missed
                own = _hit_at_s(cache, term, segments, s)
                if own is None:
                    continue
                n_terms += 1
                n_hit += int(own)
                # Path-wise: every ancestor on the annotated term's own path must also have been
                # retrieved onto the evidence. `ancestors_of_set` is the traversal's own notion
                # of reachability, so this asks what the traversal would need.
                path = (view.ancestors_of_set({term}) if view is not None else {term})
                on_path = [h for h in path if cache is not None and h in cache.index]
                n_path += int(bool(on_path) and all(
                    _hit_at_s(cache, h, segments, s) for h in on_path))
            if n_terms:
                out[rid] = (n_hit, n_terms, n_path)
        return out

    rows = []
    shared = None
    for s in s_values:
        per_index = {index: rates_for(index, int(s)) for index in space.indices}
        # The reports BOTH indices resolved, which is the only set on which a paired difference
        # is defined. Computed per S because a short cache can drop a report at one S and not
        # another.
        common = sorted(set.intersection(*(set(v) for v in per_index.values()))
                        if per_index else set())
        if shared is None:
            shared = common

        paired = {}
        if len(per_index) == 2 and common:
            ex, ont = "exemplar", "ontology_r3"
            if ex in per_index and ont in per_index:
                units = [(per_index[ex][r], per_index[ont][r]) for r in common]

                def delta(us):
                    a = sum(u[0][0] for u in us) / max(1, sum(u[0][1] for u in us))
                    b = sum(u[1][0] for u in us) / max(1, sum(u[1][1] for u in us))
                    return {"delta": a - b}

                got = bootstrap_reports(delta, units,
                                        resampler=ReportResampler(len(units), n_resamples=10000,
                                                                  seed=0))
                paired = {"paired_delta": got["delta"]["point"],
                          "paired_delta_lo": got["delta"]["lo"],
                          "paired_delta_hi": got["delta"]["hi"]}

        for index in space.indices:
            per_report = per_index[index]
            ids = sorted(per_report)
            units = [per_report[r] for r in ids]

            def rate(us):
                total = max(1, sum(u[1] for u in us))
                return {"hit_at_s": sum(u[0] for u in us) / total,
                        "path_hit_rate": sum(u[2] for u in us) / total}

            ci = bootstrap_reports(rate, units,
                                   resampler=ReportResampler(len(units), n_resamples=10000,
                                                             seed=0)) if units else {}
            row = {
                "retrieval_index": index, "S": int(s),
                "n_reports": len(ids),
                "n_gold": sum(u[1] for u in units),
                "hit_at_s": ci.get("hit_at_s", {}).get("point", float("nan")),
                "hit_at_s_lo": ci.get("hit_at_s", {}).get("lo", float("nan")),
                "hit_at_s_hi": ci.get("hit_at_s", {}).get("hi", float("nan")),
                "path_hit_rate": ci.get("path_hit_rate", {}).get("point", float("nan")),
                "path_hit_rate_lo": ci.get("path_hit_rate", {}).get("lo", float("nan")),
                "path_hit_rate_hi": ci.get("path_hit_rate", {}).get("hi", float("nan")),
            }
            if index == "exemplar":
                row.update(paired)
            rows.append(row)
            logger.info("T4.1 %-12s S=%-3d Hit@S=%.4f path=%.4f over %d gold in %d reports",
                        index, int(s), row["hit_at_s"], row["path_hit_rate"],
                        row["n_gold"], row["n_reports"])
    return rows


# ── T4.3 both pooling axes at the selected configuration ─────────────────────

def pooling_operators(space, gold, normalise, assignments, cfg_grid, poolings_pr, poolings_acc):
    """T4.3 -- every operator on both axes, on the pooled out-of-fold traversal.

    The two blocks answer two different questions and are measured differently on purpose.

    **Expansion.** Each operator gets its own threshold from conformal risk control, fitted on each
    outer fold's TRAINING reports as the protocol fits it, so the operators are compared at
    equal *risk*, not at equal threshold. Each is then charged its verifier calls and
    credited its coverage on that fold's HELD-OUT reports, pooled over the folds of
    repetition 0 -- the same reports, the same estimator, as Table 1. Comparing operators at a
    shared threshold would be meaningless: they are different functions of the same margins, so an
    operator whose scores simply run higher would look cheap purely by not having to clear the same
    bar. ``tau`` is the median of the per-fold thresholds. The threshold is refitted per fold, so
    no single value was used everywhere.

    **Acceptance.** Every held-out report keeps its fold's selected expansion and ``tau_accept`` and
    only ``pool_acc`` varies, so the operator the protocol selected reproduces Table 1's V3 row.
    ``P0`` appears here and not above: it is an indicator in {0, 1}, so it cannot rank nodes for
    expansion, but it can accept them.

    Every row is an exact re-run of the same cached margins. No operator here costs a model call.
    """
    alpha = float(cfg_grid["alpha"])
    rows = []

    for pooling in poolings_pr:
        per_fold, taus = [], []
        feasible = True
        for a in assignments:
            config = a["config"]
            axis = space.prune_axis(config.index, pooling, config.s)
            got = select_tau_prune(space, axis, gold, a["train"], alpha)
            feasible &= bool(got["feasible"])
            taus.append(got["tau_prune"])
            per_fold.append({**a, "config": replace(config, pool_pr=pooling,
                                                    tau_prune=got["tau_prune"])})
        cost = nested_operating_point(space, per_fold, gold)
        containment = sum(space.prune_axis(a["config"].index, pooling, a["config"].s)
                          .containment(a["eval"], a["config"].tau_prune) for a in per_fold)
        rows.append({
            "block": "expansion", "operator": pooling,
            "tau": float(np.median(taus)), "tau_min": float(min(taus)),
            "tau_max": float(max(taus)), "feasible": feasible,
            "calls_per_report": cost["calls_per_report"],
            "attained_coverage": cost["attained_coverage_micro"],
            "attained_coverage_macro": cost["attained_coverage_macro"],
            "empirical_risk": cost["held_out_miss_rate"],
            "containment_misses": containment,
            "micro_precision": "", "micro_recall": "", "micro_f1": "",
        })
        logger.info("T4.3 expansion %-10s tau~%.6g calls=%.0f coverage=%.4f feasible=%s",
                    pooling, rows[-1]["tau"], rows[-1]["calls_per_report"],
                    rows[-1]["attained_coverage"], feasible)

    taus_accept = sorted({a["config"].tau_accept for a in assignments})
    for pooling in poolings_acc:
        predicted = pooled_predictions(space, [
            {**a, "config": replace(a["config"], pool_acc=pooling)} for a in assignments])
        ids = sorted(predicted)
        units = units_for(predicted, gold, ids, normalise)
        p, r, f = micro_prf([g for g, _ in units], [pp for _, pp in units])
        rows.append({
            "block": "acceptance", "operator": pooling,
            "tau": taus_accept[0] if len(taus_accept) == 1 else "",
            "tau_min": "", "tau_max": "", "feasible": "",
            "calls_per_report": "", "attained_coverage": "", "attained_coverage_macro": "",
            "empirical_risk": "", "containment_misses": "",
            "micro_precision": p, "micro_recall": r, "micro_f1": f,
        })
        logger.info("T4.3 acceptance %-10s P=%.4f R=%.4f F1=%.4f", pooling, p, r, f)
    return rows


# ── the inline numbers ───────────────────────────────────────────────────────

def calls_per_report(space, assignments):
    """The call-count distribution of the pooled out-of-fold traversal, not just its mean.

    ``PruneAxis.calls`` averages over reports, and an average is the one summary a cost claim must
    not rest on: report length varies by an order of magnitude here, so the mean sits well above
    the median and a deployment sized on it is sized for a report that mostly does not occur. The
    prose quotes median, mean and p95, so all three are written. Each report is priced under its
    own fold's configuration, so the mean is Table 1's Calls column.
    """
    counts = nested_operating_point(space, assignments, {})["calls"]
    if counts.size == 0:
        return []
    stats = {
        "median": float(np.median(counts)),
        "mean": float(counts.mean()),
        "p95": float(np.percentile(counts, 95)),
        "min": float(counts.min()),
        "max": float(counts.max()),
        "total": float(counts.sum()),
        "n_reports": float(counts.size),
    }
    logger.info("calls/report: median=%.0f mean=%.0f p95=%.0f (min %.0f, max %.0f, n=%d)",
                stats["median"], stats["mean"], stats["p95"], stats["min"], stats["max"],
                int(stats["n_reports"]))
    return [{"statistic": k, "value": v} for k, v in stats.items()]


def cache_calls(space, cohort):
    """Verifier calls the GENERATING run made per report: the rows of the exhaustive score cache.

    The comparison's cost table charges a method the run that produced its predictions, and for the tree
    that is the cache build (the term-information score store), not the re-run. Its seconds come from that run's logs, and
    its call count has to come from the same run or the row prices one run's clock against another
    run's calls (it did: 22 938 s beside the configuration's 31 228 calls). Every cached
    ``(node, segment)`` row is one verifier call, so the count is the mask's sum.
    """
    rows = []
    for index in space.indices:
        counts = np.array([float(c.mask.sum()) for c in space.caches_by_index[index].values()],
                          dtype=np.float64)
        if counts.size:
            rows.append({"cohort": cohort, "retrieval_index": index,
                         "calls_per_report": float(counts.mean()),
                         "calls_median": float(np.median(counts)),
                         "n_reports": int(counts.size)})
    return rows


def captured_mass(space, config):
    """The captured-mass floor, and the share of verifier calls that fall below it.

    ``captured_mass`` is the probability the two answer tokens carry between them:
    ``exp(logaddexp(logit_yes, logit_no) - logsumexp_all)``. When it is low the model spent its
    mass somewhere else entirely and the Yes/No margin is not a judgement about the phenotype --
    it is the residue of a question the model did not answer in the requested form. The floor is
    therefore a validity bound on the margins the whole chapter re-runs, and the share below it is
    how much of the evidence base sits under that bound.

    Reported over the calls the selected configuration actually makes, not over the whole cache:
    a call the traversal never issues costs nothing and says nothing about this configuration.
    """
    caches = space.caches_by_index[config.index]
    axis = space.prune_axis(config.index, config.pool_pr, config.s)
    if not any(c.captured_mass is not None for c in caches.values()):
        logger.warning("no captured_mass column in this cache (pre-exp13_21 ingest) — the "
                       "captured-mass floor is not measurable and the stage is skipped")
        return []

    width = int(config.s)
    values = []
    for rid in space.report_ids(config.index):
        cache = caches.get(rid)
        if cache is None or cache.captured_mass is None or rid not in axis.r:
            continue
        visited = axis.visited(rid, config.tau_prune) & axis.in_cache[rid]
        # Map graph positions back to cache rows: the graph is the union over reports, the cache
        # is one report, and the two orders are not the same.
        wanted = [cache.index[h] for h in space.node_names[visited].tolist() if h in cache.index]
        if not wanted:
            continue
        take = min(width, cache.s_max)
        block = cache.captured_mass[np.array(wanted), :take]
        valid = cache.mask[np.array(wanted), :take].astype(bool)
        values.append(block[valid])
    if not values:
        return []
    mass = np.concatenate(values)
    mass = mass[np.isfinite(mass)]
    if mass.size == 0:
        return []
    floor = float(np.min(mass))
    rows = [{
        "floor": floor,
        "share_calls_below_floor": 0.0,      # by design: the floor IS the minimum
        "n_calls": int(mass.size),
        "median": float(np.median(mass)),
        "p05": float(np.percentile(mass, 5)),
        "mean": float(mass.mean()),
        **{f"share_below_{t:g}".replace(".", "p"): float((mass < t).mean())
           for t in (0.5, 0.9, 0.95, 0.99)},
    }]
    logger.info("captured mass: min=%.4f p05=%.4f median=%.4f | below 0.9: %.2f%% of %d calls",
                floor, rows[0]["p05"], rows[0]["median"],
                rows[0]["share_below_0p9"] * 100, mass.size)
    return rows


def replay_fidelity(space, config, poolings, s_values):
    """Is the columnar re-run bit-identical to pooling the cache directly?

    ``tests/unit/test_tree_replay.py`` pins ``bottleneck_scores`` against the reference frontier
    walk on random DAGs. This is the same check at cohort scale and against a different reference:
    the aligned score vector the re-run uses, versus the pooling applied to the cache rows in the
    cache's own order. They are the same numbers reached two ways, so the maximum absolute
    discrepancy should be zero, and a non-zero value means ``align_to_graph`` and the
    graph index have drifted apart -- which would silently misattribute every score in the
    chapter to the wrong term.
    """
    from hpo_extraction.treephenorag.stored_scores import POOLINGS as _POOLINGS

    rows = []
    for index in space.indices:
        caches = space.caches_by_index[index]
        for pooling in poolings:
            fn = _POOLINGS[pooling]
            for s in s_values:
                axis = space.prune_axis(index, pooling, int(s))
                worst, worst_at = 0.0, ""
                for rid, cache in caches.items():
                    direct = fn(cache, int(s))
                    aligned = axis.scores[rid]
                    positions = [space.graph.index[h] for h in cache.node_ids
                                 if h in space.graph.index]
                    keep = [j for j, h in enumerate(cache.node_ids) if h in space.graph.index]
                    if not keep:
                        continue
                    diff = np.abs(aligned[np.array(positions)] - direct[np.array(keep)])
                    diff = diff[np.isfinite(diff)]
                    if diff.size and float(diff.max()) > worst:
                        worst, worst_at = float(diff.max()), rid
                rows.append({
                    "retrieval_index": index, "pooling": pooling, "S": int(s),
                    "max_abs_score_discrepancy": worst,
                    "worst_report": worst_at,
                    "n_reports": len(caches),
                })
                logger.info("replay fidelity %-12s %-10s S=%-3d max|delta|=%.3e",
                            index, pooling, int(s), worst)
    return rows


# ── containment ──────────────────────────────────────────────────────────────

def containment_gate(space, gold, tau_cache, poolings, s_values):
    """Does the cache host the grid at all? Returns rows. The caller aborts on a non-zero count.

    The gate is evaluated at ``tau_cache``, the most permissive threshold the cache can
    honestly host, since below it the producer itself pruned.
    ``n_unreachable_below_floor`` is the separate, reported diagnostic: how much further a
    fully exhaustive cache would have reached. The synthetic-sentence score store's traversal was *meant* to be
    exhaustive, all 18 354 phenotypic-abnormality nodes at every report, so a non-zero
    value there is worth naming, not assuming away. On HCY it is a handful of leaves
    under one-segment reports, whose single verifier call scored below the floor. A cache
    non-exhaustive in bulk would instead put whole subtrees outside the cached region while
    still producing plausible-looking numbers, and then the gate fires.
    """
    rows = []
    for index in space.indices:
        n_nodes = {c.n_nodes for c in space.caches_by_index[index].values()}
        for pooling in poolings:
            for s in s_values:
                axis = space.prune_axis(index, pooling, s)
                ids = space.report_ids(index)
                # The gate: at the cache's own floor, is every node the re-run reaches scored?
                misses = axis.containment(ids, tau_cache)
                # The diagnostic: -inf is the most permissive threshold expressible, so the
                # gap between the two is what the producer pruned below its own floor.
                unreachable = axis.containment(ids, -np.inf) - misses
                rows.append({
                    "retrieval_index": index, "pooling": pooling, "S": s,
                    "tau_cache": tau_cache,
                    "n_cached_nodes_min": min(n_nodes), "n_cached_nodes_max": max(n_nodes),
                    "n_containment_misses": misses,
                    "n_unreachable_below_floor": unreachable,
                    "calls_per_report_at_floor": axis.calls(ids, tau_cache),
                })
    return rows


# ── select ───────────────────────────────────────────────────────────────────

def _fold_progress(label, n_folds):
    """A heartbeat for the one stage that runs for hours and says nothing until it ends.

    Without it a nested evaluation is indistinguishable from a hang, which on a 24 h wall is
    the difference between waiting and resubmitting.
    """
    state = {"n": 0, "t0": time.monotonic()}

    def report(record):
        state["n"] += 1
        elapsed = time.monotonic() - state["t0"]
        per = elapsed / state["n"]
        logger.info("%s | fold %d/%d (rep %s, outer %s) -> %s tau_p=%.6g %s tau_a=%.3g "
                    "inner_f1=%.4f | %.1f s/fold, ~%.1f min left",
                    label, state["n"], n_folds, record["repetition"], record["outer_fold"],
                    record["retrieval_index"], record["tau_prune"], record["pool_acc"],
                    record["tau_accept"], record["inner_f1"], per,
                    per * (n_folds - state["n"]) / 60.0)

    return report


def run_selection(space, folds, gold, normalise, cfg_grid, sampler, label="select"):
    """E4.3, the configuration chosen by rule. Returns rows, choices, CRC records, stability."""
    got = nested_evaluation(
        space, folds, gold, normalise,
        poolings_pr=cfg_grid["poolings_prune"],
        poolings_acc=cfg_grid["poolings_accept"],
        tau_accepts=cfg_grid["tau_accept_grid"],
        s=cfg_grid["s_main"],
        alpha=cfg_grid["alpha"],
        indices=space.indices,
        on_fold=_fold_progress(label, len(folds)),
    )
    stability = selection_stability(got["choices"])
    for field, stat in stability.items():
        logger.info("selection stability | %-16s mode=%-12s modal share=%.2f over %d distinct",
                    field, str(stat["mode"]), stat["modal_share"], stat["n_distinct"])

    report_ids = sorted(got["pooled"])
    units = units_for(got["pooled"], gold, report_ids, normalise)
    assignments = rep0_assignments(folds, got["choices"])
    cost = nested_operating_point(space, assignments, gold)
    row = score_units("nested CV (pooled out-of-fold)", units, sampler,
                      {"n_reports": len(report_ids),
                       "attained_coverage": cost["attained_coverage_micro"],
                       "calls_per_report": cost["calls_per_report"]})
    return {"row": row, "units": units, **got, "stability": stability,
            "assignments": assignments}


def alpha_sensitivity(space, folds, gold, normalise, cfg_grid, alphas):
    """What the risk tolerance buys, and where it stops being reachable.

    Each alpha reruns the whole protocol, and every column is read off that run's own **pooled
    out-of-fold** traversal -- each held-out report under the configuration its fold selected --
    so the alpha = 0.10 row is Table 1's V3 row, not a neighbour of it. Two columns summarise the
    fold rows instead, over all repetitions, and only at the retrieval index the fold row actually
    selected (conformal risk control is run for every candidate index before the index is chosen,
    so the raw CRC log carries one record per index and averaging it would mix in an index no fold
    used): ``feasible_folds`` -- below the traversal's own coverage ceiling no threshold honours the
    bound, and the run should say so, not quote a number -- and ``mean_attained_ceiling``.
    """
    rows = []
    for alpha in alphas:
        got = nested_evaluation(
            space, folds, gold, normalise,
            poolings_pr=cfg_grid["poolings_prune"],
            poolings_acc=cfg_grid["poolings_accept"],
            tau_accepts=cfg_grid["tau_accept_grid"],
            s=cfg_grid["s_main"], alpha=float(alpha), indices=space.indices,
            on_fold=_fold_progress(f"alpha={alpha}", len(folds)),
        )
        chosen = {(c["repetition"], c["outer_fold"]): c["retrieval_index"]
                  for c in got["choices"]}
        crc = [c for c in got["crc"]
               if chosen.get((c["repetition"], c["outer_fold"])) == c["retrieval_index"]]
        ids = sorted(got["pooled"])
        units = units_for(got["pooled"], gold, ids, normalise)
        p, r, f = micro_prf([g for g, _ in units], [p for _, p in units])
        cost = nested_operating_point(space, rep0_assignments(folds, got["choices"]), gold)
        rows.append({
            "alpha": float(alpha),
            "feasible_folds": sum(1 for c in crc if c["feasible"]),
            "n_folds": len(crc),
            "mean_attained_ceiling": float(np.mean([c["attained_ceiling"] for c in crc])),
            "held_out_miss_rate": cost["held_out_miss_rate"],
            "attained_coverage_micro": cost["attained_coverage_micro"],
            "attained_coverage_macro": cost["attained_coverage_macro"],
            "calls_per_report": cost["calls_per_report"],
            "calls_per_report_median": float(np.median(cost["calls"])),
            "n_reports": cost["n_reports"],
            "micro_precision": p, "micro_recall": r, "micro_f1": f,
        })
        logger.info("alpha=%.3f | %d/%d fold-rows feasible | held-out miss %.4f | coverage %.4f "
                    "| %.0f calls/report | F1 %.4f", alpha, rows[-1]["feasible_folds"],
                    rows[-1]["n_folds"], rows[-1]["held_out_miss_rate"],
                    rows[-1]["attained_coverage_micro"], rows[-1]["calls_per_report"], f)
    return rows


# ── ablations ────────────────────────────────────────────────────────────────

#: ``tab:tpr-ladder``. Each rung changes ONE element of the previous one, and every rung is tuned by
#: The same nested-CV protocol, a ladder whose rungs are oracles compares two oracles and isolates
#: nothing. Fields: ``(label, pool_pr, pool_acc, shared_threshold, fixed_tau_prune)``.
#:
#: **V0's threshold is fixed at 0.5 and that is not a shortcut.** P0 is an indicator in {0, 1}, so
#: every threshold in (0, 1] induces the identical node set and the rung has nothing to tune,
#: which is itself the sharpest available statement of what the two-threshold design buys. Letting
#: risk control choose here picks 0.0, at which ``score >= tau`` is vacuously true and the
#: "traversal" expands the entire ontology: not PhenoRAG's any-yes rule but an exhaustive scan,
#: scored under its name. 0.5 is the canonical reading, expand iff some segment answered Yes with
#: confidence above a half.
LADDER = (
    ("V0 traversal, P0 both decisions", "P0", "P0", False, 0.5),
    ("V1 P1 both, one shared threshold", "P1", "P1", True, None),
    ("V2 P1, tau_prune by risk control", "P1", "P1", False, None),
    ("V3 selected poolings", None, None, False, None),
)


def run_ladder(space, folds, gold, normalise, cfg_grid, sampler, selected):
    """E4.4, V0 to V3, plus the paired tests between adjacent rungs.

    ``shared_threshold`` is what makes V1 the single-threshold ablation: one value is tuned for F1
    and used for BOTH decisions, which is the design the two-threshold split is being compared
    against. V2 unties them; V3 additionally lets the pooling pair be selected.
    """
    rows, units_by_label = [], {}
    for label, pool_pr, pool_acc, shared, fixed_tau in LADDER:
        prune_candidates = [pool_pr] if pool_pr else cfg_grid["poolings_prune"]
        accept_candidates = [pool_acc] if pool_acc else cfg_grid["poolings_accept"]

        pooled: dict[str, set[str]] = {}
        choices = []
        for row in folds:
            train, evaluate = list(row["train_ids"]), list(row["eval_ids"])
            inner = [list(f) for f in row["inner_folds"]]

            if shared:
                # One threshold, both decisions, tuned for F1 on the inner folds.
                config, best = None, -1.0
                for index in space.indices:
                    for tau in cfg_grid["tau_accept_grid"]:
                        candidate = Configuration(index, pool_pr, float(tau), pool_acc,
                                                  float(tau), cfg_grid["s_main"])
                        score = float(np.mean([space.micro_f1(candidate, f, gold, normalise)
                                               for f in inner if f]))
                        if score > best:
                            config, best = candidate, score
            else:
                prune_by_index = {}
                for index in space.indices:
                    if fixed_tau is not None:
                        # Nothing to select: see LADDER's note on P0 having no operating curve.
                        prune_by_index[index] = (prune_candidates[0], float(fixed_tau))
                        continue
                    picked = select_prune_pooling(space, index, prune_candidates,
                                                  cfg_grid["s_main"], gold, train,
                                                  cfg_grid["alpha"])["chosen"]
                    prune_by_index[index] = (picked["pooling"], picked["tau_prune"])
                first = space.indices[0]
                base = Configuration(first, *prune_by_index[first], accept_candidates[0],
                                     float(cfg_grid["tau_accept_grid"][0]),
                                     cfg_grid["s_main"])
                config = select_acceptance(space, base, inner, gold, normalise,
                                           accept_candidates, cfg_grid["tau_accept_grid"],
                                           space.indices, prune_by_index)["config"]

            predictions = space.predictions(config)
            if row["repetition"] == 0:
                pooled.update({r: predictions.get(r, set()) for r in evaluate})
            choices.append(config)

        ids = sorted(pooled)
        units = units_by_label[label] = units_for(pooled, gold, ids, normalise)
        # Calls and attained coverage of the SAME pooled traversal the F1 is scored on: each
        # held-out report under its own fold's configuration. A rung's F1 is bounded by its
        # coverage, so the two belong on one row -- V0's recall of 0.225 against a coverage near
        # 1.0 says something quite different from the same recall against a coverage of 0.3.
        # (This used to price the rung at its modal configuration over all 50 fold rows instead,
        # which is neither the traversal that was scored nor deterministic: `max(set(...))`
        # broke a 25/25 tie by hash order.)
        assignments = fold_assignments(folds, choices)
        cost = nested_operating_point(space, assignments, gold)
        # The modal configuration of the scored repetition, named for the reader. Counter breaks a
        # tie by first occurrence, i.e. by fold order, so the label cannot change between runs.
        modal = Counter(a["config"] for a in assignments).most_common(1)[0][0]
        rows.append(score_units(label, units, sampler, {
            **modal.as_row(),
            "attained_coverage": cost["attained_coverage_micro"],
            "calls_per_report": cost["calls_per_report"],
        }))
    return rows, units_by_label


def paired_tests(units_by_label, n_permutations, alpha_holm):
    """Adjacent rungs only. Testing every pair would inflate the family and answer nothing asked."""
    labels = list(units_by_label)
    tests, deltas = {}, {}
    for lower, higher in zip(labels, labels[1:]):
        # `delta` is A minus B, so the HIGHER rung is A: a rung that improves on the one below it
        # then reads as a positive delta, matching the "higher vs lower" label. Passing them the
        # other way round reports every improvement as a negative number.
        got = paired_randomisation(
            [u[0] for u in units_by_label[higher]],
            [u[1] for u in units_by_label[higher]], [u[1] for u in units_by_label[lower]],
            lambda g, p: dict(zip(("micro_precision", "micro_recall", "micro_f1"),
                                  micro_prf(g, p))),
            n_permutations=n_permutations, seed=0)
        name = f"{higher} vs {lower}"
        tests[name] = got["p_value"]
        deltas[name] = got["delta"]
        logger.info("%-58s delta=%+.4f p=%.4f", name, got["delta"], got["p_value"])
    if not tests:
        return {}
    return {k: {**v, "delta": deltas[k]} for k, v in holm(tests, alpha=alpha_holm).items()}


def s_ablation(space, folds, gold, normalise, cfg_grid, s_values):
    """Vary S by re-running cache prefixes -- the whole protocol rerun at each S.

    Not the selected configuration with S swapped: tau_prune is fitted by conformal risk control
    ON the S-segment scores, so a threshold fitted at S = 10 is not a valid configuration at
    S = 3. Each S therefore gets its own nested evaluation, and its row is that run's pooled
    out-of-fold set -- so the S = ``s_headline`` row IS Table 1's V3 row. Only repetition 0 is
    run: it is the only one that contributes predictions, so the others would cost 9x the time and
    change no number in the table.
    """
    first = [row for row in folds if row["repetition"] == 0]
    rows = []
    for s in s_values:
        got = nested_evaluation(
            space, first, gold, normalise,
            poolings_pr=cfg_grid["poolings_prune"],
            poolings_acc=cfg_grid["poolings_accept"],
            tau_accepts=cfg_grid["tau_accept_grid"],
            s=int(s), alpha=cfg_grid["alpha"], indices=space.indices,
            on_fold=_fold_progress(f"S={s}", len(first)),
        )
        ids = sorted(got["pooled"])
        units = units_for(got["pooled"], gold, ids, normalise)
        # Micro AND macro, from the same units -- the table beside the ladder reports both.
        scores = metric_fn(units)
        cost = nested_operating_point(space, rep0_assignments(first, got["choices"]), gold)
        rows.append({"S": int(s), **scores,
                     "calls_per_report": cost["calls_per_report"],
                     "attained_coverage": cost["attained_coverage_micro"],
                     "attained_coverage_macro": cost["attained_coverage_macro"]})
        logger.info("S=%-3d F1=%.4f macro F1=%.4f calls/report=%.0f coverage=%.4f", int(s),
                    scores["micro_f1"], scores["macro_f1"], cost["calls_per_report"],
                    cost["attained_coverage_micro"])
    return rows


def oracle_expansion(space, gold, normalise, assignments, view):
    """The condition that removes all pruning loss: expand the terms with an annotated term below them.

    It isolates retrieval and verification from the pruning rule, and its gap to V3 is what the
    expansion decision costs. Not an configuration -- it reads the answer -- and labelled as
    such. Each held-out report keeps its own fold's acceptance rule, so the gap to V3 is the
    expansion decision and nothing else.
    """
    predictions = {}
    for a in assignments:
        config = a["config"]
        accept = space.accept_scores(config.index, config.pool_acc, config.s)
        axis = space.prune_axis(config.index, config.pool_pr, config.s)
        for rid in a["eval"]:
            if rid not in axis.r:
                continue
            reachable = view.ancestors_of_set(gold.get(rid, set()))
            keep = np.array([h in reachable for h in space.graph.node_ids])
            predictions[rid] = set(space.node_names[
                keep & (accept[rid] >= config.tau_accept) & axis.in_cache[rid]])
    ids = sorted(predictions)
    return units_for(predictions, gold, ids, normalise)


# ── calibration ──────────────────────────────────────────────────────────────

def calibration_sample(space, config, gold, report_ids, with_reports=False):
    """``(scores, labels)`` over every **scored** pair, the population §Calibration is defined on.

    The target is ``z = 1[v in Y_i]``, and the conditioning is deliberate: each system scores its
    own candidate set, so calibration is conditional on that selection and comparable only between
    configurations of the same architecture.

    ``with_reports=True`` adds a third array naming each pair's report -- what a report-level
    bootstrap and a cross-fit over report folds need, and nothing else does.
    """
    axis = space.prune_axis(config.index, config.pool_pr, config.s)
    accept = space.accept_scores(config.index, config.pool_acc, config.s)
    scores, labels, owners = [], [], []
    for rid in report_ids:
        if rid not in axis.r:
            continue
        visited = axis.visited(rid, config.tau_prune) & axis.in_cache[rid]
        gold_set = gold.get(rid, set())
        for i in np.flatnonzero(visited):
            scores.append(float(accept[rid][i]))
            labels.append(int(space.graph.node_ids[i] in gold_set))
            owners.append(rid)
    if with_reports:
        return np.array(scores), np.array(labels, dtype=int), np.array(owners, dtype=object)
    return np.array(scores), np.array(labels, dtype=int)


def run_calibration(space, config, gold, report_ids, methods, n_bins):
    """E4.6, fit h, and report what it does and does not change."""
    scores, labels = calibration_sample(space, config, gold, report_ids)
    rows, maps = [], {}
    for method in methods:
        h = fit_calibration_map(scores, labels, method=method)
        maps[method] = h
        got = calibration_report(h(scores), labels, n_bins=n_bins)
        rows.append({
            "calibration": method, "fitted": h.method,
            **{k: v for k, v in got.items() if not isinstance(v, dict)},
            "chosen_vs_half_optimal_f1":
                f1_optimal_threshold_diagnostic(scores, labels, h)["gap"],
        })
        logger.info("h=%-9s (fitted %-9s) ECE=%.4f SmoothECE=%s Brier=%.4f n=%d prevalence=%.4f",
                    method, h.method, got["ece_equal_mass"],
                    "n/a" if got["smooth_ece"] is None else f"{got['smooth_ece']:.4f}",
                    got["brier"], got["n"], got["prevalence"])
    return {"rows": rows, "maps": maps, "scores": scores, "labels": labels}


# ── E4.6b the two score distributions, split by whether the decision was right ───

def _auc(scores, labels):
    """Mann-Whitney AUC: P(score of a positive > score of a negative), ties counted as half.

    Rank-based and therefore exact, which counts because the positive class is ~0.2% of the
    population and a threshold sweep over so few positives is noisy where the rank statistic is
    not. NaN when either class is empty -- a separation statistic over one class is not a number.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average the ranks inside each tie group, or a score used by both classes would be scored as
    # if it discriminated. Verifier margins tie constantly at the pooled extremes.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, sorted_scores.size + 1):
        if i == sorted_scores.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _distribution_rows(scores, labels, n_bins):
    """One row per bin: ``lo, hi, n_positive, n_negative`` over a shared equal-width grid.

    **Binned, never per-node.** The population here is every scored (report, term) pair -- upwards
    of half a million rows carrying a report id -- and the figure needs only the shape. One shared
    grid for both classes is what makes the two curves comparable at a glance. Separate per-class
    grids would put the same score in different bins.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0:
        return []
    lo, hi = float(scores.min()), float(scores.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # A degenerate axis is a finding (P0's indicator is one), not a reason to emit nothing.
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    # `digitize` puts the maximum in a bin of its own past the last edge. Fold it back so the top
    # bin is closed and the highest-scoring node is not silently dropped.
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        sel = idx == i
        if not sel.any():
            continue
        rows.append({
            "bin": i,
            "lo": float(edges[i]),
            "hi": float(edges[i + 1]),
            "n_positive": int(labels[sel].sum()),
            "n_negative": int(sel.sum() - labels[sel].sum()),
        })
    return rows


def expansion_sample(space, config, gold, report_ids, view):
    """``(prune_score, 1[subtree of v contains a gold term])`` over the nodes the rule is applied to.

    The population is the visited, in-cache nodes at the selected ``tau_prune`` -- the set
    the expansion decision is taken over, which is what makes this the mirror image of
    :func:`calibration_sample`, not a different question asked on a different population.

    The label is the one :func:`oracle_expansion` uses: ``v`` is worth expanding iff an annotated term
    lies at or below it, i.e. ``v`` is a reflexive ancestor of the report's ground-truth set. A node that
    *is* ground truth counts as worth expanding -- the traversal has to reach it to accept it.
    """
    axis = space.prune_axis(config.index, config.pool_pr, config.s)
    scores, labels = [], []
    for rid in report_ids:
        if rid not in axis.r:
            continue
        visited = axis.visited(rid, config.tau_prune) & axis.in_cache[rid]
        if not visited.any():
            continue
        bearing = view.ancestors_of_set(gold.get(rid, set()))
        node_scores = axis.scores[rid]
        for i in np.flatnonzero(visited):
            scores.append(float(node_scores[i]))
            labels.append(int(space.graph.node_ids[i] in bearing))
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=int)


def run_distributions(space, config, gold, report_ids, view, n_bins):
    """Both decision axes as binned distributions, split by whether the decision was right.

    The chapter reports two thresholds and an $F_1$. Neither says whether the verifier's score
    *separates* the cases it is thresholding. A threshold that cannot be placed well is a different
    problem from one that was placed badly, and only the distributions distinguish them.

    Returns ``{"expansion": {...}, "acceptance": {...}}``, each with ``rows`` (the bins) and
    ``summary`` (one row: counts, class medians, AUC and the threshold in force).
    """
    out = {}
    for name, (scores, labels), threshold, positive in (
        ("expansion", expansion_sample(space, config, gold, report_ids, view),
         config.tau_prune, "subtree contains a gold term"),
        ("acceptance", calibration_sample(space, config, gold, report_ids),
         config.tau_accept, "term is gold"),
    ):
        n_pos = int(labels.sum())
        summary = {
            "decision": name,
            "positive_class": positive,
            "pooling": config.pool_pr if name == "expansion" else config.pool_acc,
            "threshold": float(threshold),
            "n": int(labels.size),
            "n_positive": n_pos,
            "n_negative": int(labels.size - n_pos),
            "prevalence": float(n_pos / labels.size) if labels.size else float("nan"),
            "median_positive": float(np.median(scores[labels == 1])) if n_pos else float("nan"),
            "median_negative": (float(np.median(scores[labels == 0]))
                                if n_pos < labels.size else float("nan")),
            "auc": _auc(scores, labels),
            # What the threshold in force actually keeps of each class. This is the trade-off the
            # figure exists to show, and quoting it saves the reader measuring a shaded area.
            "positive_above_threshold": (float((scores[labels == 1] >= threshold).mean())
                                         if n_pos else float("nan")),
            "negative_above_threshold": (float((scores[labels == 0] >= threshold).mean())
                                         if n_pos < labels.size else float("nan")),
        }
        logger.info("%-10s n=%d prevalence=%.5f AUC=%.4f  kept: %.3f of positives, "
                    "%.3f of negatives", name, summary["n"], summary["prevalence"],
                    summary["auc"], summary["positive_above_threshold"],
                    summary["negative_above_threshold"])
        z, n_clipped = to_logit(scores)
        pr, average_precision = _pr_rows(scores, labels)
        tp = int(((scores >= threshold) & (labels == 1)).sum())
        fp = int(((scores >= threshold) & (labels == 0)).sum())
        summary.update({
            "threshold_logit": float(to_logit(np.array([threshold]))[0][0]),
            "n_clipped_to_logit_range": n_clipped,
            "average_precision": average_precision,
            "precision_at_threshold": tp / (tp + fp) if tp + fp else float("nan"),
            "recall_at_threshold": tp / n_pos if n_pos else float("nan"),
        })
        if n_clipped:
            logger.warning("%s: %d score(s) saturate the float64 sigmoid and sit at the logit "
                           "clip (+/-%.1f)", name, n_clipped, float(to_logit(np.array([1.0]))[0][0]))
        out[name] = {"rows": _distribution_rows(scores, labels, n_bins),
                     "logit_rows": _logit_histogram_rows(z, labels, LOGIT_BINS),
                     "survival_rows": _survival_rows(z, labels, SURVIVAL_POINTS),
                     "pr_rows": pr,
                     "summary": summary}
    return out


# ── The same two axes in logit space ─────────────────────────────────────────
# Both pooled scores live in (0, 1) but pile up against its ends: the negative class of either
# decision puts most of its mass below 1e-3, and the positive acceptance class above 0.99. On the
# logit of the score -- which for lse_beta1 IS the pooled margin, since that score is
# sigmoid(logsumexp(margins)) -- both classes become readable humps and a threshold is a line.

#: Shared equal-width bins for the logit histogram.
LOGIT_BINS = 60
#: Grid points for the survival curves.
SURVIVAL_POINTS = 240
#: The PR curve is downsampled to at most this many points. The configuration and AP are exact.
PR_POINTS = 500
#: float64's sigmoid saturates to 1.0 for margins above ~37. Clipping here keeps those
#: scores finite, at +/-27.6 logits, and the number clipped is reported, not hidden.
LOGIT_EPS = 1e-12


def to_logit(scores, eps=LOGIT_EPS):
    """``(log(p / (1 - p)), n_clipped)`` with ``p`` clipped into ``[eps, 1 - eps]``."""
    p = np.asarray(scores, dtype=np.float64)
    clipped = np.clip(p, eps, 1.0 - eps)
    n_clipped = int((clipped != p).sum())
    return np.log(clipped) - np.log1p(-clipped), n_clipped


def _logit_histogram_rows(z, labels, n_bins):
    """Per-class counts over ONE shared equal-width grid in logit space."""
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if z.size == 0:
        return []
    lo, hi = float(z.min()), float(z.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pos, _ = np.histogram(z[labels == 1], bins=edges)
    neg, _ = np.histogram(z[labels == 0], bins=edges)
    return [{"bin": i, "lo": float(edges[i]), "hi": float(edges[i + 1]),
             "n_positive": int(pos[i]), "n_negative": int(neg[i])} for i in range(n_bins)]


def _survival_rows(z, labels, n_points):
    """Per class, the share scoring at or above ``t``, on a grid spanning the observed range.

    Each curve runs from 1 to 0 whatever the class size, so the 1 : 400 imbalance does not flatten
    one of them. The threshold's two crossings are the class-wise keep rates.
    """
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if z.size == 0:
        return []
    pos = np.sort(z[labels == 1])
    neg = np.sort(z[labels == 0])
    grid = np.linspace(float(z.min()), float(z.max()), n_points)

    def above(sorted_values, t):
        if sorted_values.size == 0:
            return float("nan")
        return float((sorted_values.size - np.searchsorted(sorted_values, t, side="left"))
                     / sorted_values.size)

    return [{"t_logit": float(t), "positive_above": above(pos, t),
             "negative_above": above(neg, t)} for t in grid]


def _pr_rows(scores, labels, max_points=PR_POINTS):
    """``(rows, average_precision)`` -- precision against recall over every distinct threshold.

    AP is the step-wise sum ``sum_i (R_i - R_{i-1}) P_i`` over ALL thresholds. Only the returned
    curve is thinned, evenly in recall, which is the axis the eye reads it along.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    if scores.size == 0 or n_pos == 0:
        return [], float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # One point per distinct score: the last index of each tie group.
    last = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), s_sorted.size - 1]
    precision = tp[last] / (tp[last] + fp[last])
    recall = tp[last] / n_pos
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    keep = np.unique(np.searchsorted(recall, np.linspace(0.0, 1.0, max_points), side="left")
                     .clip(0, recall.size - 1))
    rows = [{"threshold": float(s_sorted[last][i]), "precision": float(precision[i]),
             "recall": float(recall[i])} for i in keep]
    return rows, average_precision


def acceptance_ranking(space, config, gold, report_ids, poolings):
    """Every acceptance operator's ranking of the same scored pairs: AUC and average precision.

    tab:tpr-pooling compares the acceptance operators at ONE threshold, the one selected for the
    chapter's operator, so an operator on a different scale looks useless there without ranking any
    worse (check Q4). AUC and AP need no threshold. The population is :func:`calibration_sample`'s
    -- the visited pairs at the selected ``tau_prune`` -- which ``pool_acc`` does not change, so
    every operator is ranked over identical pairs and labels.
    """
    rows, n_ref = [], None
    for pooling in poolings:
        scores, labels = calibration_sample(space, replace(config, pool_acc=pooling), gold,
                                            report_ids)
        if n_ref is None:
            n_ref = (labels.size, int(labels.sum()))
        elif (labels.size, int(labels.sum())) != n_ref:
            raise RuntimeError(f"acceptance_ranking: {pooling} was sampled over "
                               f"{labels.size} pairs, not {n_ref[0]} -- the population moved")
        _, average_precision = _pr_rows(scores, labels)
        rows.append({"operator": pooling, "n": int(labels.size), "n_positive": int(labels.sum()),
                     "prevalence": float(labels.mean()) if labels.size else float("nan"),
                     "auc": _auc(scores, labels), "average_precision": average_precision})
        logger.info("acceptance %-10s AUC=%.4f AP=%.4f", pooling, rows[-1]["auc"],
                    average_precision)
    return rows


def reliability_rows(scores, labels, h, n_bins):
    """Bin-level signed gaps with counts, the reliability diagram, as a table."""
    from hpo_extraction.evaluation.metrics import bin_summary

    return [{"bin": i, **b} for i, b in enumerate(bin_summary(h(scores), labels, n_bins,
                                                              "equal_mass"))]


def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score interval for ``k`` successes in ``n`` trials; ``(nan, nan)`` when ``n = 0``.

    Unlike the Wald interval it is not degenerate at ``k = 0``: its upper end stays positive, which
    is what a reliability bin with no observed positive needs to be drawn honestly.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def platt_crossfit(scores, labels, owners, folds):
    """Platt scaling fitted on the other outer folds and applied to each held-out fold.

    Fitted on the **logit** of the score -- Platt's own parametrisation, ``sigmoid(a*logit(p)+b)``.
    Two parameters per fit, and no pair is ever mapped by a fit that saw it. Uses the first
    repetition only, the one every other stage describes. Pairs whose report falls in no eval fold
    come back NaN and are dropped by the caller, with a count.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    z, _ = to_logit(scores)
    out = np.full(z.size, np.nan)
    first = [f for f in folds if int(f.get("repetition", 0)) == 0]
    for fold in first:
        held = np.isin(owners, list(fold["eval_ids"]))
        train = np.isin(owners, list(fold["train_ids"]))
        if not held.any():
            continue
        if labels[train].sum() == 0 or labels[train].sum() == train.sum():
            raise ValueError(f"outer fold {fold.get('outer_fold')}: one class absent in training")
        model = LogisticRegression(max_iter=5000)
        model.fit(z[train].reshape(-1, 1), labels[train])
        out[held] = model.predict_proba(z[held].reshape(-1, 1))[:, 1]
    return out


def reliability_crossfit(space, config, gold, report_ids, folds, n_bins, n_boot, seed=0):
    """The reliability diagram's rows and its SmoothECE, for the raw and cross-fitted Platt score.

    Returns ``{"rows": [...], "summary": [...]}``: ``rows`` one per (curve, equal-mass bin) with
    the Wilson 95% interval on the observed rate; ``summary`` one per curve with SmoothECE and a
    95% bootstrap interval resampling REPORTS -- two pairs from one report are not independent.
    """
    from hpo_extraction.evaluation.metrics import bin_summary, smooth_ece  # noqa: PLC0415

    scores, labels, owners = calibration_sample(space, config, gold, report_ids,
                                                with_reports=True)
    mapped = platt_crossfit(scores, labels, owners, folds)
    ok = np.isfinite(mapped)
    if not ok.all():
        logger.warning("cross-fit: %d pair(s) in no eval fold are dropped", int((~ok).sum()))
    scores, labels, owners, mapped = scores[ok], labels[ok], owners[ok], mapped[ok]

    by_report = {}
    for i, rid in enumerate(owners):
        by_report.setdefault(rid, []).append(i)
    groups = [np.array(v) for v in by_report.values()]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
             for _ in range(int(n_boot))]

    rows, summary = [], []
    for curve, conf in (("raw", scores), ("platt_crossfit", mapped)):
        for i, b in enumerate(bin_summary(conf, labels, n_bins, "equal_mass")):
            positives = int(round(b["accuracy"] * b["count"]))
            lo, hi = wilson_interval(positives, b["count"])
            rows.append({"curve": curve, "bin": i, "lo": b["lo"], "hi": b["hi"],
                         "count": b["count"], "positives": positives,
                         "mean_predicted": b["confidence"], "observed_rate": b["accuracy"],
                         "wilson_lo": lo, "wilson_hi": hi})
        point = smooth_ece(conf, labels)
        boot = np.array([smooth_ece(conf[d], labels[d]) for d in draws])
        q_lo = float(np.quantile(boot, 0.025)) if boot.size else float("nan")
        q_hi = float(np.quantile(boot, 0.975)) if boot.size else float("nan")
        boot_mean = float(boot.mean()) if boot.size else float("nan")
        # SmoothECE is biased upward under resampling (a resample is noisier than the sample), so
        # The percentile interval can sit entirely above the point estimate -- it did, for the
        # cross-fitted Platt curve. The bias-corrected estimate 2*theta - mean(theta*) and the
        # basic interval [2*theta - q_hi, 2*theta - q_lo] remove that bias. The interval contains
        # The corrected estimate by design. A calibration error is non-negative, so both
        # are clipped at 0. The percentile columns stay for reference.
        summary.append({"curve": curve, "smooth_ece": point,
                        "smooth_ece_boot_mean": boot_mean,
                        "smooth_ece_bc": max(0.0, 2 * point - boot_mean),
                        "smooth_ece_bc_lo": max(0.0, 2 * point - q_hi),
                        "smooth_ece_bc_hi": max(0.0, 2 * point - q_lo),
                        "smooth_ece_lo": q_lo,
                        "smooth_ece_hi": q_hi,
                        "n_bootstrap": int(n_boot), "n": int(labels.size),
                        "n_positive": int(labels.sum()), "n_reports": len(groups),
                        "n_bins": int(n_bins)})
        logger.info("reliability %-15s SmoothECE=%.5f  bias-corrected %.5f [%.5f, %.5f]", curve,
                    point, summary[-1]["smooth_ece_bc"], summary[-1]["smooth_ece_bc_lo"],
                    summary[-1]["smooth_ece_bc_hi"])
    return {"rows": rows, "summary": summary}


def score_transferred(space_gsc, config, gold, normalise, label):
    """The selected configuration applied to GSC+ **unchanged**, and what it scores there.

    GSC+ is test-only: no folds, no risk control, nothing re-tuned. Every axis -- retrieval
    index, both poolings, both thresholds, S -- is the one HCY's nested CV chose, so this row
    is a transfer result and not a second selection. It is reported with its own report-level
    bootstrap over the 114 documents.
    """
    ids = space_gsc.report_ids(config.index)
    units = units_for(space_gsc.predictions(config), gold, ids, normalise)
    sampler = ReportResampler(len(ids), n_resamples=10000, seed=0)
    axis = space_gsc.prune_axis(config.index, config.pool_pr, config.s)
    return score_units(label, units, sampler, {
        **config.as_row(),
        "n_reports": len(ids),
        "calls_per_report": axis.calls(ids, config.tau_prune),
    })


def transfer_to_gsc(space_hcy, space_gsc, config, gold_hcy, gold_gsc, normalise, tau_grid):
    """Does the HCY-selected acceptance threshold survive a genre change?

    ``tau_accept`` is fitted on HCY over the protocol's grid and applied to GSC+ unchanged. The gap
    to the best grid threshold on GSC+ itself is what the change of genre costs.

    **There is no calibrated condition, and there cannot be one.** ``h`` is fitted on HCY only and is
    monotone, so ``h(a) >= t`` is ``a >= h^-1(t)`` on GSC+ as on HCY: for every threshold
    on the calibrated scale there is a raw threshold accepting the identical set of terms in both
    cohorts. A calibrated condition can therefore only differ from the raw one through the grid it is
    searched over -- which is what the earlier version measured: it searched the raw grid
    (0.3 ... 0.995) on the Platt scale, where nearly every score sits below 0.3, and reported the
    grid's misfit (0.24 against 0.32 on HCY) as a calibration effect. Removed on 2026-09-24.
    """
    def evaluate_on(space, gold):
        ids = space.report_ids(config.index)

        def run(tau_prune, tau_accept):
            candidate = replace(config, tau_prune=float(tau_prune), tau_accept=float(tau_accept))
            units = units_for(space.predictions(candidate), gold, ids, normalise)
            p, r, f = micro_prf([g for g, _ in units], [pp for _, pp in units])
            return {"micro_precision": p, "micro_recall": r, "micro_f1": f}
        return run

    out = {"raw": transfer_gap(evaluate_on(space_hcy, gold_hcy), evaluate_on(space_gsc, gold_gsc),
                               [config.tau_prune], list(tau_grid))}
    logger.info("transfer HCY->GSC+: fit %.4f -> transferred %.4f vs oracle %.4f (gap %.4f)",
                out["raw"]["fit_value"], out["raw"]["transferred_value"],
                out["raw"]["oracle_value"], out["raw"]["gap"])
    return out


# ── diagnostics ──────────────────────────────────────────────────────────────

def run_diagnostics(space, config, gold, predicted, report_ids, view, delta_m, evidence=None):
    """E4.7, where recall was lost, and how far wrong the false positives are."""
    axis = space.prune_axis(config.index, config.pool_pr, config.s)
    accept = space.accept_scores(config.index, config.pool_acc, config.s)
    ids = [r for r in report_ids if r in axis.r]

    scored = {r: set(space.node_names[axis.visited(r, config.tau_prune)]) for r in ids}
    expanded = {r: set(space.node_names[axis.expanded(r, config.tau_prune)]) for r in ids}
    pooled = {r: {space.graph.node_ids[i]: float(accept[r][i])
                  for i in np.flatnonzero(axis.visited(r, config.tau_prune))} for r in ids}

    decomposition = recall_decomposition(
        {r: gold.get(r, set()) for r in ids}, predicted, scored, expanded,
        space.caches_by_index[config.index], view,
        pooled_by_report=pooled, threshold=config.tau_accept,
        evidence=evidence, delta_m=delta_m,
    )
    gold_sets = [gold.get(r, set()) for r in ids]
    pred_sets = [predicted.get(r, set()) for r in ids]
    return {
        "decomposition": decomposition,
        "taxonomy": error_taxonomy_existential(gold_sets, pred_sets, view),
        "near_miss": near_miss_distribution(gold_sets, pred_sets, view),
        "attribution": attribution_rows(decomposition),
    }


def nested_diagnostics(space, assignments, gold, view, delta_m, evidence=None):
    """:func:`run_diagnostics` on the pooled out-of-fold predictions -- Figures 3 and 5's population.

    Where an annotated term was lost depends on the configuration that lost it (what was scored, what
    was expanded, which threshold it missed), so the decomposition is run once per outer fold under
    that fold's configuration on its held-out reports, and the pieces are summed. The counts are
    additive and the fractions are recomputed from the sums, so the buckets still partition every
    false negative of the pooled set -- the 1 - Cov identity holds on the whole as it does per
    fold. The false-positive taxonomy and the near-miss histogram depend only on the ground truth and the
    predictions, so they are computed once over the pooled set.
    """
    predicted = pooled_predictions(space, assignments)
    parts = [run_diagnostics(space, a["config"], gold, predicted, a["eval"], view, delta_m,
                             evidence)["decomposition"] for a in assignments]
    counts = Counter()
    causes, depths = Counter(), Counter()
    n_gold = n_fn = n_pruned = 0
    per_term = []
    for d in parts:
        counts.update(d["counts"])
        causes.update(d["blocking_causes"])
        depths.update(d["blocking_depth_histogram"])
        n_gold += d["n_gold"]
        n_fn += d["n_false_negative"]
        n_pruned += round(d["coverage_loss"] * d["n_gold"])
        per_term.extend(d["per_term"])
    buckets = list(parts[0]["counts"]) if parts else []
    decomposition = {
        "n_gold": n_gold,
        "n_false_negative": n_fn,
        "counts": {b: counts.get(b, 0) for b in buckets},
        "fractions": {b: (counts.get(b, 0) / n_fn if n_fn else 0.0) for b in buckets},
        "coverage_loss": n_pruned / n_gold if n_gold else 0.0,
        "loss_at_scored_terms": (n_fn - n_pruned) / n_gold if n_gold else 0.0,
        "blocking_causes": {c: causes.get(c, 0) for c in (parts[0]["blocking_causes"]
                                                          if parts else {})},
        "blocking_depth_histogram": dict(sorted(depths.items())),
        "evidence_available": all(d["evidence_available"] for d in parts) if parts else False,
        "delta_m": float(delta_m),
        "per_term": per_term,
    }
    ids = sorted(predicted)
    gold_sets = [gold.get(r, set()) for r in ids]
    pred_sets = [predicted.get(r, set()) for r in ids]
    return {
        "decomposition": decomposition,
        "taxonomy": error_taxonomy_existential(gold_sets, pred_sets, view),
        "near_miss": near_miss_distribution(gold_sets, pred_sets, view),
        "attribution": attribution_rows(decomposition),
        "predicted": predicted,
    }


def attribution_rows(decomposition):
    """One row per missed annotated pair: which bucket lost it, and where.

    ``recall_decomposition`` already computes this per term and keeps it as ``per_term``. Only the
    bucket totals were ever written out. The totals are enough for chapter 4's own table and not
    enough for anything that has to join on a *pair* -- which is what chapter 6's
    complementarity table does when it asks which of TreePhenoRAG's losses PhenoJury recovers.
    Answering that from the totals alone would require assuming the two methods' misses are
    independent, which is the one thing the question is about.

    The bucket is normalised to the five names of the decomposition: a row the module filed under
    a blocking cause (``pruning``) keeps that, and ``lost_at`` names the ancestor the traversal
    actually stopped at, not the annotated term, since that is where the failure happened.
    """
    rows = []
    for record in decomposition.get("per_term", ()):
        bucket = "pruning" if record.get("pruned") else record.get("cause", "residual")
        rows.append({
            "report_id": record.get("report_id", ""),
            "hpo_id": record.get("term", ""),
            "bucket": bucket,
            "cause": record.get("cause", ""),
            "lost_at": record.get("lost_at") or "",
            "blocking_depth": record.get("blocking_depth")
            if record.get("blocking_depth") is not None else "",
            "pruned": bool(record.get("pruned")),
        })
    return rows
