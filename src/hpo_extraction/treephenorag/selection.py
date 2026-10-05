r"""Choosing TreePhenoRAG's configuration by a rule instead of by looking at the answer.

``04_TreePhenoRAG.tex``, §Evaluation protocol: the 118 patients are split into five outer folds. Within each outer training split :math:`\tau_{\mathrm{prune}}` is set by conformal risk control and
the poolings, :math:`h` and :math:`\tau_{\mathrm{accept}}` are selected by five-fold inner
cross-validation. Outer-fold predictions are pooled into one prediction per patient. The whole thing
is repeated over ten fold assignments. This module is that procedure, with no IO and no Hydra, so it
can be tested on a toy ontology.

The selection order is the thesis's and is not arbitrary
-------------------------------------------------------

1. :math:`\tau_{\mathrm{prune}}` **by conformal risk control**, on the outer-training reports.
   Pruning is a *constraint* (cover the ground truth), not an :math:`F_1` optimum: a false prune removes
   every annotated term below a node irreversibly, while a false expansion costs verifier calls and no
   error at all. Because coverage does not depend on the acceptance decision, this is fixed first,
   which avoids a two-dimensional search on very few reports.
2. :math:`\mathrm{pool}_{\mathrm{pr}}` **by fewest verifier calls per report at its own**
   :math:`\hat\lambda`, explicitly *not* by the discrimination of expansion scores, because the
   scored terms on which that could be computed are themselves selected by the traversal.
3. :math:`\mathrm{pool}_{\mathrm{acc}}`, :math:`\tau_{\mathrm{accept}}` **and the retrieval index**
   by inner-fold micro :math:`F_1`.
4. :math:`h` **fitted on the outer-training scored pairs**, then refitted on the full training
   split.

What :math:`h` does, stated once
--------------------------------

:math:`h` is non-decreasing, so ``h(a) >= t`` is ``a >= h^{-1}(t)``. Within one cohort it is a
*reparametrisation* of the acceptance axis: it cannot change which terms are accepted at a
threshold tuned on the same data, and it cannot change discrimination either. So the inner folds
select :math:`\tau_{\mathrm{accept}}` on **raw** pooled scores, searching calibrated space as well
would be searching the same grid twice, and :math:`h` is fitted alongside so the selected
threshold can be *expressed* in calibrated space and carried to GSC+ unchanged. That transfer is
the only place :math:`h` can earn anything, which is why §Calibration's cross-cohort gap is the
evidence for it and the within-cohort reliability diagram is a diagnostic.

Cost
----

The naive shape re-scores every configuration on every fold. With ~160 acceptance configurations per
retrieval index, five inner folds, five outer folds and ten repetitions that is millions of ontology
reductions. Two facts collapse it:

* a configuration's **predictions do not depend on the fold**, folds only decide which reports are
  averaged, so per-report contributions are computed once per configuration and reused;
* micro :math:`F_1` over any set of reports is three sums, so once a configuration has cached
  ``(|g ∩ p|, |p|, |g|)`` per report, scoring a fold is integer addition.

:class:`ReplaySpace` owns that memo. The expensive axis remains expansion, as in
:mod:`hpo_extraction.treephenorag.stored_scores`: acceptance never touches the frontier.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from hpo_extraction.treephenorag.stored_scores import (
    POOLINGS,
    align_to_graph,
    bottleneck_scores,
    build_traversal_graph,
    calls_at,
    containment_misses,
    coverage_breakpoints,
    expanded_at,
    in_cache_mask,
    visited_at,
)
from hpo_extraction.evaluation.stats import crc_lambda, held_out_risk, miss_rate


@dataclass(frozen=True)
class Configuration:
    """One fully-specified configuration. Hashable, so it keys the memo."""

    index: str
    pool_pr: str
    tau_prune: float
    pool_acc: str
    tau_accept: float
    s: int

    def as_row(self) -> dict:
        """The configuration as a table row, with the column names of the stored tables."""
        return {"retrieval_index": self.index, "pool_pr": self.pool_pr,
                "tau_prune": self.tau_prune, "pool_acc": self.pool_acc,
                "tau_accept": self.tau_accept, "S": self.s}


@dataclass
class PruneAxis:
    """Everything the expansion decision needs for one ``(index, pool_pr, S)``, per report.

    ``r`` is the bottleneck value of :func:`hpo_extraction.treephenorag.stored_scores.bottleneck_scores`: the largest
    ``tau_prune`` at which a node is still visited. Storing it means the whole threshold axis is a
    comparison rather than a traversal.
    """

    index: str
    pooling: str
    s: int
    r: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    n_valid: dict[str, np.ndarray]
    in_cache: dict[str, np.ndarray]

    def visited(self, report_id: str, tau: float) -> np.ndarray:
        """Boolean mask of the terms the traversal visits on one report at expansion threshold *tau*."""
        return visited_at(self.r[report_id], tau)

    def expanded(self, report_id: str, tau: float) -> np.ndarray:
        """Boolean mask of the terms the traversal expands on one report at expansion threshold *tau*."""
        return expanded_at(self.r[report_id], self.scores[report_id], tau)

    def calls(self, report_ids: Iterable[str], tau: float) -> float:
        """Mean verifier calls per report at expansion threshold *tau*."""
        ids = list(report_ids)
        if not ids:
            return 0.0
        return sum(calls_at(self.r[r], self.n_valid[r], tau) for r in ids) / len(ids)

    def containment(self, report_ids: Iterable[str], tau: float) -> int:
        """Number of visited terms the stored scores do not contain, summed over the reports (must be 0)."""
        return sum(containment_misses(self.r[r], self.in_cache[r], tau) for r in report_ids)


class OfflineSpace:
    """The re-run grid for one cohort, over one or more retrieval indices.

    ``caches_by_index`` maps a retrieval-index name (``exemplar`` / ``ontology_r3``) to that index's
    ``{report_id: ReportCache}``. The two are *never* mixed inside a traversal, a re-run whose
    margins came from two retrievals is not a configuration the pipeline could be run at, but the
    *choice between* them is a legitimate axis, and it is selected here like any other.
    """

    def __init__(self, children_map, roots, caches_by_index: Mapping[str, Mapping[str, object]],
                 tau_floor: float = -np.inf):
        self.graph = build_traversal_graph(children_map, roots)
        self.caches_by_index = {k: dict(v) for k, v in caches_by_index.items()}
        # The tau the cache was PRODUCED at. Below it the producer pruned, so the cache has
        # nothing to say and a re-run there is not hosted -- see `crc_candidates`. The -inf
        # default keeps the unit fixtures, which are exhaustive by design, unchanged.
        self.tau_floor = float(tau_floor)
        self.node_names = np.array(self.graph.node_ids)
        self._prune: dict[tuple[str, str, int], PruneAxis] = {}
        self._accept: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        self._units: dict[tuple[Configuration, str], tuple[int, int, int]] = {}
        # Predictions are bounded, units are not. One configuration's prediction sets are thousands
        # of term strings per report and the selection sweep visits ~7 200 configurations per
        # nested evaluation, so keeping them all is tens of gigabytes. Keeping the three integers
        # per (config, report) that scoring actually needs is megabytes. Recomputing a prediction
        # set is a numpy mask and an index, so a window of a few configurations is enough to cover
        # The one access pattern that counts, `units` walking every report of one configuration.
        self._pred_cache: "OrderedDict[Configuration, dict[str, set[str]]]" = OrderedDict()
        self._pred_cache_size = 8

    # ── Axes ────────────────────────────────────────────────────────────────

    @property
    def indices(self) -> list[str]:
        """Names of the retrieval indices with stored scores."""
        return sorted(self.caches_by_index)

    def report_ids(self, index: str) -> list[str]:
        """Reports with stored scores for one retrieval index."""
        return sorted(self.caches_by_index[index])

    def prune_axis(self, index: str, pooling: str, s: int) -> PruneAxis:
        """Expansion scores, bottleneck values and call counts for one (index, pooling, S), computed once and cached."""
        key = (index, pooling, s)
        if key not in self._prune:
            fn = POOLINGS[pooling]
            r, scores, n_valid, in_cache = {}, {}, {}, {}
            for rid, cache in self.caches_by_index[index].items():
                aligned = align_to_graph(self.graph, cache.node_ids, fn(cache, s))
                scores[rid] = aligned
                r[rid] = bottleneck_scores(self.graph, aligned)
                n_valid[rid] = align_to_graph(self.graph, cache.node_ids, cache.n_valid(s))
                in_cache[rid] = in_cache_mask(self.graph, cache.node_ids)
            self._prune[key] = PruneAxis(index, pooling, s, r, scores, n_valid, in_cache)
        return self._prune[key]

    def accept_scores(self, index: str, pooling: str, s: int) -> dict[str, np.ndarray]:
        """``{report: acceptance score per term}`` for one (index, pooling, S), each between 0 and 1."""
        key = (index, pooling, s)
        if key not in self._accept:
            fn = POOLINGS[pooling]
            self._accept[key] = {
                rid: align_to_graph(self.graph, cache.node_ids, fn(cache, s))
                for rid, cache in self.caches_by_index[index].items()
            }
        return self._accept[key]

    # ── Predictions ─────────────────────────────────────────────────────────

    def predictions(self, config: Configuration) -> dict[str, set[str]]:
        """``{report: accepted terms}``. Memoised: a configuration's predictions are fold-free."""
        if config in self._pred_cache:
            self._pred_cache.move_to_end(config)
            return self._pred_cache[config]
        prune = self.prune_axis(config.index, config.pool_pr, config.s)
        accept = self.accept_scores(config.index, config.pool_acc, config.s)
        out = {}
        for rid in self.caches_by_index[config.index]:
            keep = (visited_at(prune.r[rid], config.tau_prune)
                    & (accept[rid] >= config.tau_accept)
                    & prune.in_cache[rid])
            # `.tolist()` hands back interned Python strings. The numpy `str_` objects the
            # fancy index produces are three times the size and hash slower.
            out[rid] = set(self.node_names[keep].tolist())
        self._pred_cache[config] = out
        while len(self._pred_cache) > self._pred_cache_size:
            self._pred_cache.popitem(last=False)
        return out

    def scored_sets(self, index: str, pool_pr: str, s: int, tau: float) -> dict[str, set[str]]:
        """:math:`\\mathcal{C}_i`, the candidate set conformal risk control is defined on."""
        axis = self.prune_axis(index, pool_pr, s)
        return {rid: set(self.node_names[axis.visited(rid, tau)].tolist()) for rid in axis.r}

    def gold_miss_rates(self, axis: PruneAxis, gold, report_ids: Sequence[str],
                       taus: np.ndarray) -> np.ndarray:
        """``(n_reports, n_taus)`` of :math:`L_i(\\tau)`, without building one scored set.

        Identical by design to
        ``miss_rate(gold_i, scored_sets(..., tau)[i])``, ``visited`` is ``r >= tau``, so a
        annotated term is scored when its own bottleneck clears the threshold. Three
        cases the set form handles implicitly and this one has to name: a report with no
        ground truth contributes 0 (it cannot lose what it has not got). An annotated term outside the
        traversal graph is missed at every threshold. And a report absent from the cache
        loses all of its ground truth.

        This is the whole point of the bottleneck formulation. Materialising the scored set
        per threshold costs ~1e9 string insertions per pooling per fold and made the CRC
        inversion, not the acceptance sweep, the cost of the entire chapter.
        """
        taus = np.asarray(taus, dtype=np.float64)
        out = np.zeros((len(report_ids), taus.size), dtype=np.float64)
        for i, rid in enumerate(report_ids):
            terms = gold.get(rid) or ()
            n_gold = len(terms)
            if n_gold == 0:
                continue
            if rid not in axis.r:
                out[i] = 1.0
                continue
            positions = [self.graph.index[t] for t in terms if t in self.graph.index]
            off_graph = n_gold - len(positions)
            if positions:
                values = axis.r[rid][positions]
                missed = (values[:, None] < taus[None, :]).sum(axis=0)
            else:
                missed = np.zeros(taus.size, dtype=np.int64)
            out[i] = (missed + off_graph) / n_gold
        return out

    def expanded_sets(self, index: str, pool_pr: str, s: int, tau: float) -> dict[str, set[str]]:
        """``{report: set of expanded terms}`` at expansion threshold *tau*."""
        axis = self.prune_axis(index, pool_pr, s)
        return {rid: set(self.node_names[axis.expanded(rid, tau)].tolist()) for rid in axis.r}

    # ── Scoring ─────────────────────────────────────────────────────────────

    def units(self, config: Configuration, report_id: str, gold, normalise) -> tuple[int, int, int]:
        """``(|g ∩ p|, |p|, |g|)`` for one report under one configuration, after normalisation.

        Cached because this is the only expensive step: ``normalise`` performs the ontology
        reduction of §Metrics, and it is invariant to which fold the report lands in.
        """
        key = (config, report_id)
        if key not in self._units:
            g, p = normalise(gold.get(report_id, set()),
                             self.predictions(config).get(report_id, set()))
            self._units[key] = (len(g & p), len(p), len(g))
        return self._units[key]

    def micro_f1(self, config: Configuration, report_ids, gold, normalise) -> float:
        """Micro F1 of one configuration over *report_ids*, between 0 and 1."""
        tp = n_pred = n_gold = 0
        for rid in report_ids:
            a, b, c = self.units(config, rid, gold, normalise)
            tp += a
            n_pred += b
            n_gold += c
        if tp == 0:
            return 0.0
        precision, recall = tp / n_pred, tp / n_gold
        return 2 * precision * recall / (precision + recall)


# ── Step 1: tau_prune by conformal risk control ──────────────────────────────

def crc_candidates(space: OfflineSpace, axis: PruneAxis, gold, report_ids) -> np.ndarray:
    """The exhaustive threshold set: every annotated term's own bottleneck value, and nothing else.

    ``L_i(tau)`` is constant between consecutive ground truth ``r`` values, so this is the finest grid that
    exists *and* the coarsest that is faithful. It is what lets the risk bound be inverted
    instead of over five quantile points, the thing the draft audit lists as needing a rerun.

    The set is **clipped to the cache floor**. A threshold below the tau the cache was
    produced at is not a configuration this cache can be re-run at: the producer pruned
    there, so the nodes a lower threshold would visit were never scored -- which is
    what the containment gate measures. Annotated terms whose bottleneck sits below the floor are
    therefore unreachable at *every* admissible threshold. That is an irreducible coverage
    loss; ``crc_lambda`` sees it as a floor on the empirical risk and reports it as one,
    not buying it back with a threshold this cache cannot host.
    """
    values: list[np.ndarray] = []
    for rid in report_ids:
        positions = [space.graph.index[t] for t in gold.get(rid, ())
                     if t in space.graph.index]
        if positions:
            values.append(coverage_breakpoints(axis.r[rid], positions))
    floor = space.tau_floor
    if not values:
        return np.array([floor if np.isfinite(floor) else 0.0])
    grid = np.unique(np.concatenate(values))
    if np.isfinite(floor):
        # The floor itself is always a candidate: it is the most permissive admissible
        # threshold, and dropping the breakpoints below it without re-adding it would leave
        # The grid starting stricter than the cache actually supports.
        grid = np.unique(np.concatenate([[floor], grid[grid >= floor]]))
    return grid


def select_tau_prune(
    space: OfflineSpace,
    axis: PruneAxis,
    gold,
    calibration_ids: Sequence[str],
    alpha: float,
    taus: np.ndarray | None = None,
) -> dict:
    """Conformal risk control on the calibration reports. Returns the CRC record plus the grid.

    Falls back to the attained ceiling when ``alpha`` is unreachable at every cached threshold, in
    which case the binding constraint is retrieval and the verifier, not the pruning rule, and the
    run should say so, not quote a threshold that does not honour the bound.
    """
    ids = [r for r in calibration_ids if r in axis.r]
    grid = crc_candidates(space, axis, gold, ids) if taus is None else np.asarray(taus)
    losses = space.gold_miss_rates(axis, gold, ids, grid)
    got = crc_lambda([float(t) for t in grid], losses, alpha=alpha)
    tau = got["tau"] if got["feasible"] else got["attained_ceiling_tau"]
    return {**got, "tau_prune": float(tau), "n_grid": int(grid.size),
            "pooling": axis.pooling, "index": axis.index, "S": axis.s}


def select_prune_pooling(
    space: OfflineSpace,
    index: str,
    poolings: Sequence[str],
    s: int,
    gold,
    calibration_ids: Sequence[str],
    alpha: float,
) -> dict:
    """The expansion pooling with the fewest verifier calls per report at its own lambda-hat.

    Feasible rules are always preferred over infeasible ones: a cheaper pooling that cannot
    honour the risk bound has not solved the problem, it has changed it. Among **feasible**
    rules cost decides, the thesis's stated criterion, and the reason this step does not look
    at F1 at all.

    When **nothing** is feasible, cost is the wrong criterion and actively the worst one: the
    cheapest rule is the most aggressive pruner, so ranking by cost picks the pooling that
    loses the most ground truth in the regime where coverage is already unattainable. (At
    alpha = 0.05 on HCY that produced a held-out miss rate of 0.38 against an attainable
    ceiling of ~0.08.) There the rule is the **lowest attained ceiling**, the pooling that
    comes closest to the bound it cannot meet, and the run reports `feasible=False` beside
    it, so the row reads as a limit of retrieval and the verifier, not as a choice.
    """
    rows = []
    for pooling in poolings:
        axis = space.prune_axis(index, pooling, s)
        got = select_tau_prune(space, axis, gold, calibration_ids, alpha)
        rows.append({**got,
                     "calls_per_report": axis.calls(calibration_ids, got["tau_prune"]),
                     "containment_misses": axis.containment(calibration_ids, got["tau_prune"])})
    usable = [r for r in rows if r["containment_misses"] == 0] or rows
    if any(r["feasible"] for r in usable):
        best = min((r for r in usable if r["feasible"]),
                   key=lambda r: r["calls_per_report"])
    else:
        best = min(usable, key=lambda r: (r["attained_ceiling"], r["calls_per_report"]))
    return {"chosen": best, "candidates": rows}


# ── Step 3: acceptance by inner cross-validation ─────────────────────────────

def select_acceptance(
    space: OfflineSpace,
    base: Configuration,
    inner_folds: Sequence[Sequence[str]],
    gold,
    normalise,
    pool_accs: Sequence[str],
    tau_accepts: Sequence[float],
    indices: Sequence[str] | None = None,
    prune_by_index: Mapping[str, tuple[str, float]] | None = None,
) -> dict:
    """Choose ``(retrieval index, pool_acc, tau_accept)`` by mean inner-fold micro F1.

    ``prune_by_index`` carries each index's own expansion configuration, since the pruning rule is
    selected per index, a cache is re-run whole or not at all.

    Ties break towards the earlier index, the earlier pooling and the **higher** threshold, i.e.
    towards the more conservative configuration. Arbitrary, but fixed, and stated here, not
    left to dictionary order.
    """
    candidates = list(indices or [base.index])
    best = None
    rows = []
    for index in candidates:
        pool_pr, tau_prune = (prune_by_index or {}).get(index, (base.pool_pr, base.tau_prune))
        for pool_acc in pool_accs:
            for tau_accept in tau_accepts:
                config = replace(base, index=index, pool_pr=pool_pr, tau_prune=tau_prune,
                                 pool_acc=pool_acc, tau_accept=float(tau_accept))
                scores = [space.micro_f1(config, fold, gold, normalise) for fold in inner_folds
                          if fold]
                mean = float(np.mean(scores)) if scores else 0.0
                rows.append({**config.as_row(), "inner_f1": mean})
                key = (mean, -candidates.index(index), -list(pool_accs).index(pool_acc),
                       float(tau_accept))
                if best is None or key > best[0]:
                    best = (key, config, mean)
    return {"config": best[1], "inner_f1": best[2], "candidates": rows}


# ── The outer loop ───────────────────────────────────────────────────────────

def nested_evaluation(
    space: OfflineSpace,
    folds: Sequence[Mapping],
    gold,
    normalise,
    poolings_pr: Sequence[str],
    poolings_acc: Sequence[str],
    tau_accepts: Sequence[float],
    s: int,
    alpha: float,
    indices: Sequence[str] | None = None,
    on_fold: Callable[[dict], None] | None = None,
) -> dict:
    """Run the whole protocol. Returns pooled out-of-fold predictions and what each fold chose.

    Only the **first repetition** contributes to the pooled prediction set, so every report appears
    once and the resulting numbers are a single honest evaluation. The remaining
    repetitions are not wasted: they measure how often the procedure makes the same choice, which
    is what decides whether "the protocol selected X" is a finding or a coin flip.
    """
    candidates = list(indices or space.indices)
    pooled: dict[str, set[str]] = {}
    choices: list[dict] = []
    crc_rows: list[dict] = []

    for row in folds:
        train = list(row["train_ids"])
        evaluate = list(row["eval_ids"])
        inner = [list(f) for f in row["inner_folds"]]

        prune_by_index: dict[str, tuple[str, float]] = {}
        for index in candidates:
            picked = select_prune_pooling(space, index, poolings_pr, s, gold, train, alpha)
            chosen = picked["chosen"]
            prune_by_index[index] = (chosen["pooling"], chosen["tau_prune"])
            crc_rows.append({
                "repetition": row["repetition"], "outer_fold": row["outer_fold"],
                "retrieval_index": index, "alpha": alpha, **{
                    k: chosen[k] for k in ("pooling", "tau_prune", "feasible", "bound",
                                           "empirical_risk", "attained_ceiling", "n_calibration",
                                           "n_grid", "calls_per_report", "containment_misses")},
                "held_out_miss_rate": float(np.mean(space.gold_miss_rates(
                    space.prune_axis(index, chosen["pooling"], s), gold,
                    [r for r in evaluate if r in space.caches_by_index[index]],
                    np.array([chosen["tau_prune"]]),
                ))) if any(r in space.caches_by_index[index] for r in evaluate)
                else float("nan"),
            })

        first_index = candidates[0]
        base = Configuration(first_index, *prune_by_index[first_index],
                             poolings_acc[0], float(tau_accepts[0]), s)
        picked = select_acceptance(space, base, inner, gold, normalise, poolings_acc,
                                   tau_accepts, candidates, prune_by_index)
        config = picked["config"]

        predictions = space.predictions(config)
        if row["repetition"] == 0:
            pooled.update({r: predictions.get(r, set()) for r in evaluate})
        record = {"repetition": row["repetition"], "outer_fold": row["outer_fold"],
                  **config.as_row(), "inner_f1": picked["inner_f1"],
                  "n_eval": len(evaluate)}
        choices.append(record)
        if on_fold is not None:
            on_fold(record)

    return {"pooled": pooled, "choices": choices, "crc": crc_rows}


def selection_stability(choices: Sequence[Mapping]) -> dict:
    """How often the procedure made the same choice across fold assignments.

    A modal share near 1 means "the protocol selected X" is a fact about the data. Near 1/k it
    means the selection is noise and the chapter should say so instead of naming a winner.
    """
    from collections import Counter

    out = {}
    for field in ("retrieval_index", "pool_pr", "pool_acc", "tau_accept", "tau_prune"):
        tally = Counter(c[field] for c in choices if field in c)
        if not tally:
            continue
        value, n = tally.most_common(1)[0]
        out[field] = {"mode": value, "modal_share": n / sum(tally.values()),
                      "n_distinct": len(tally), "counts": dict(tally)}
    return out
