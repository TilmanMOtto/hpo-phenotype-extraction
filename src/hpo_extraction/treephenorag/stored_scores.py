r"""Re-run the TreePhenoRAG traversal offline from a cached score dump, no GPU, no model.

``{variant}_calls.jsonl`` records one row per (report, term, retrieved segment) with that segment's
verifier margin. Because :func:`hpo_extraction.treephenorag.traversal.traverse` takes its per-node work as an injected
callable, everything downstream of the margins, the pooling rule, both thresholds, and the number
of segments used, can be re-derived from that file at zero GPU cost. This module is the engine that
does it, and it is what turns a threshold sweep from a cluster job into a loop.

Containment
-----------

A re-run is only valid where the cache actually has the nodes. The traversal scores a node only if
some parent was expanded, so the cached node set is ``{v : pool_cache(v) >= tau_cache}`` closed under
reachability, and a re-run at ``(pool, tau)`` is contained in it iff every node that re-run would
expand is already there.

For the margin poolings this is decidable in closed form. Writing ``x_s = exp(m_s)``:

.. math::
    \mathrm{lse}\beta_1 = \frac{\sum_s x_s}{1 + \sum_s x_s}, \qquad
    P3_1 = 1 - \frac{1}{\prod_s (1 + x_s)} .

Since :math:`\prod(1+x) \ge 1 + \sum x`, :math:`P3_1 \ge \mathrm{lse}\beta_1`, P3₁ dominates. And
because :math:`\prod(1+x) \le e^{\sum x}`, the reverse bound holds too:

.. math::
    \{P3_1 \ge t\} \subseteq \{\mathrm{lse}\beta_1 \ge g(t)\}, \qquad
    g(t) = \frac{a}{1+a}, \quad a = -\ln(1-t),

with :math:`g(t) < t` for every :math:`t > 0`, converging to :math:`t` as :math:`t \to 0` (both
operators reduce to :math:`\sum_s x_s` in the small-score limit). So a cache built at ``tau_cache``
hosts **any** operator in the pooling family at any ``tau_replay`` satisfying
``g(tau_replay) >= tau_cache``, see :func:`containment_threshold`.

Two consequences worth stating plainly, because they are counter-intuitive:

* A cache built with the *weaker* ``lse_beta1`` at a low threshold is far more useful than one built
  with the dominating ``P3_1`` at a high threshold. Which operator was used counts much less than
  how permissive the threshold was.
* ``LRGate`` is **not** covered. It mixes in tree geometry (depth, subtree size) and can fire where
  every margin is low, so no bound over margins constrains it. A cache intended to host it has to
  have been built with it, in practice by expanding on ``max(noisy_or, LRGate)``.

Containment is checked at runtime regardless: a re-run that reaches a node the cache does not have
records a **containment miss**, and a caller must treat a non-zero count as invalidating the cell
rather than as a rounding detail.

Cost
----

The naive shape, pooling inside ``evaluate_node``, recomputes the same score once per threshold.
With seven poolings, a threshold grid fine enough to invert a risk bound, and four values of S, that
is thousands of traversals per report over thousands of nodes. Two structural facts collapse it:

* **Acceptance never touches the frontier.** ``traverse`` expands on ``prune_score`` alone;
  ``accept_score`` only sets a flag, and :meth:`~hpo_extraction.treephenorag.traversal.TraversalResult.at_accept`
  re-derives acceptance without re-traversing. One traversal per ``(pool_pr, tau_prune, S)``
  therefore serves every acceptance pooling and every ``tau_accept``.
* **Pooling vectorises.** Margins are padded once into an ``(n_nodes, S_max)`` array, so each
  pooling is a single numpy reduction over all of a report's nodes at once.

Hence :func:`replay` takes a **precomputed score vector**, not a pooling callable. The scalar
reference implementations in :mod:`hpo_extraction.treephenorag.pooling` remain the definition. The vectorised ones
here are the fast path, and ``tests/unit/test_tree_replay.py`` pins them to agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from hpo_extraction.treephenorag.traversal import NodeEval, TraversalResult, traverse


# ── The cache ────────────────────────────────────────────────────────────────

@dataclass
class ReportCache:
    """One report's margins, padded to a rectangle so every pooling is one numpy call.

    ``margins[i, j]`` is the verifier margin for node ``node_ids[i]`` on its rank-``j+1`` retrieved
    segment, and ``mask[i, j]`` says whether that entry is real. Reports with fewer segments than
    ``S_max``, and terms whose retrieval returned short, are the reason the mask exists, not
    a sentinel value: there is no margin that is safely "absent" for every pooling at once.
    """

    report_id: str
    node_ids: list[str]
    margins: np.ndarray
    mask: np.ndarray
    index: dict[str, int] = field(default_factory=dict)

    #: Columns :func:`load_score_cache` does not read and :func:`load_cache_npz` does. All are
    #: ``None`` for a cache built the old way, so a reader must check, not assume: ``cosine``
    #: is what separates a retrieval miss from a judgement miss, and the two instrumentation columns
    #: exist only for the synthetic-sentence score store onward.
    cosine: np.ndarray | None = None
    captured_mass: np.ndarray | None = None
    sent_index: np.ndarray | None = None
    wall_clock_s: np.ndarray | None = None
    n_prompt_tokens: np.ndarray | None = None

    def __post_init__(self):
        if not self.index:
            self.index = {h: i for i, h in enumerate(self.node_ids)}

    @property
    def n_nodes(self) -> int:
        """Number of terms with stored scores for this report."""
        return len(self.node_ids)

    @property
    def s_max(self) -> int:
        """Largest number of stored segments per term (10 in the thesis runs)."""
        return int(self.margins.shape[1]) if self.margins.size else 0

    def n_valid(self, s: int | None = None) -> np.ndarray:
        """Segments actually scored per node at this ``S``, the per-node verifier-call count."""
        width = self.s_max if s is None else min(int(s), self.s_max)
        return self.mask[:, :width].sum(axis=1).astype(np.float64)


def load_score_cache(
    calls_path: str | Path,
    ctx_type: str = "union",
    report_ids: Iterable[str] | None = None,
) -> dict[str, ReportCache]:
    """Read ``{variant}_calls.jsonl`` into one :class:`ReportCache` per report.

    Rows are keyed by ``(report_id, hpo_id)`` and ordered by ``rank``, which is the retrieval order
    the run used, so taking the first ``S`` columns is the prefix a run with ``top_n = S``
    would have seen. ``ctx_type`` selects the retrieval context: ``union_only`` runs write one
    (``union``), ``union_prune_own_accept`` writes both.
    """
    wanted = set(report_ids) if report_ids is not None else None
    per_report: dict[str, dict[str, list[tuple[int, float]]]] = {}

    with Path(calls_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail from a hard kill. The driver appends per report
            if rec.get("ctx_type") != ctx_type:
                continue
            rid = rec["report_id"]
            if wanted is not None and rid not in wanted:
                continue
            per_report.setdefault(rid, {}).setdefault(rec["hpo_id"], []).append(
                (int(rec["rank"]), float(rec["margin"]))
            )

    caches: dict[str, ReportCache] = {}
    for rid, by_node in per_report.items():
        node_ids = sorted(by_node)
        s_max = max((len(v) for v in by_node.values()), default=0)
        margins = np.zeros((len(node_ids), s_max), dtype=np.float64)
        mask = np.zeros((len(node_ids), s_max), dtype=bool)
        for i, h in enumerate(node_ids):
            for j, (_, m) in enumerate(sorted(by_node[h])):
                margins[i, j] = m
                mask[i, j] = True
        caches[rid] = ReportCache(rid, node_ids, margins, mask)
    return caches


# ── Vectorised poolings ──────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prepare(cache: ReportCache, s: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(sigma, mask, n_valid)`` over the first ``s`` ranks, invalid entries zeroed."""
    width = cache.s_max if s is None else min(int(s), cache.s_max)
    margins = cache.margins[:, :width]
    mask = cache.mask[:, :width]
    sigma = np.where(mask, _sigmoid(margins), 0.0)
    return sigma, mask, mask.sum(axis=1)


def pool_p0(cache: ReportCache, s: int | None = None) -> np.ndarray:
    """P0, PhenoRAG's own rule: at least one segment answers Yes with confidence above a half."""
    return (pool_p1(cache, s) > 0.5).astype(np.float64)


def pool_p1(cache: ReportCache, s: int | None = None) -> np.ndarray:
    """P1, the largest segment confidence (``accept_confidence``)."""
    sigma, _, n_valid = _prepare(cache, s)
    out = sigma.max(axis=1) if sigma.size else np.zeros(cache.n_nodes)
    return np.where(n_valid > 0, out, 0.0)


def pool_p2(cache: ReportCache, s: int | None = None) -> np.ndarray:
    """P2, the second largest segment confidence. Zero where only one segment was retrieved."""
    sigma, _, n_valid = _prepare(cache, s)
    if sigma.shape[1] < 2:
        return np.zeros(cache.n_nodes)
    second = np.sort(sigma, axis=1)[:, -2]
    return np.where(n_valid >= 2, second, 0.0)


def pool_p3(cache: ReportCache, s: int | None = None, kappa=1.0) -> np.ndarray:
    """P3_kappa, tempered noisy-OR. ``kappa = 1`` is the maximum of the whole pooling family.

    ``kappa`` may be a scalar or a per-node array. The array form is what ``P3_S`` needs: the
    exponent is meant to be *the number of segments this term was scored on*, and terms differ, retrieval returns fewer than S whenever the report is short. Using the report's S_max for every
    node would charge a node scored on two segments as though it had five, mixing one node's
    evidence count into another's.

    With ``kappa`` equal to the valid count, the rule has a clean reading: ``1 -`` the geometric
    mean of ``1 - sigma``, i.e. The single-observation probability the segments average to.
    """
    sigma, mask, n_valid = _prepare(cache, s)
    if sigma.size == 0:
        return np.zeros(cache.n_nodes)
    k = np.asarray(kappa, dtype=np.float64)
    if k.ndim == 0:
        k = np.full(cache.n_nodes, float(k))
    k = np.maximum(k, 1.0)[:, None]
    # Invalid entries contribute a factor of 1, i.e. no evidence either way.
    factors = np.where(mask, (1.0 - sigma) ** (1.0 / k), 1.0)
    out = 1.0 - factors.prod(axis=1)
    return np.where(n_valid > 0, out, 0.0)


def pool_p4(cache: ReportCache, s: int | None = None) -> np.ndarray:
    """P4, the mean segment confidence, the negative control."""
    sigma, _, n_valid = _prepare(cache, s)
    if sigma.size == 0:
        return np.zeros(cache.n_nodes)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sigma.sum(axis=1) / n_valid
    return np.where(n_valid > 0, out, 0.0)


def pool_lse_beta1(cache: ReportCache, s: int | None = None, beta: float = 1.0) -> np.ndarray:
    """``sigmoid(logsumexp(beta * margins) / beta)``, the rule every shipped tree number uses."""
    width = cache.s_max if s is None else min(int(s), cache.s_max)
    margins = cache.margins[:, :width]
    mask = cache.mask[:, :width]
    if margins.size == 0:
        return np.zeros(cache.n_nodes)
    scaled = np.where(mask, margins * beta, -np.inf)
    peak = np.max(scaled, axis=1, keepdims=True)
    finite = np.isfinite(peak[:, 0])
    total = np.zeros(cache.n_nodes)
    if finite.any():
        shifted = np.exp(scaled[finite] - peak[finite])
        total[finite] = (peak[finite, 0] + np.log(shifted.sum(axis=1))) / beta
    return np.where(finite, _sigmoid(total), 0.0)


#: The thesis's Table 3 family plus the two as-implemented rules, by the ids the tables use.
POOLINGS: dict[str, Callable[..., np.ndarray]] = {
    "P0": pool_p0,
    "P1": pool_p1,
    "P2": pool_p2,
    "P3_1": lambda c, s=None: pool_p3(c, s, kappa=1.0),
    "P3_2": lambda c, s=None: pool_p3(c, s, kappa=2.0),
    # kappa is the per-node count of segments actually scored, not the report's S_max.
    "P3_S": lambda c, s=None: pool_p3(c, s, kappa=_prepare(c, s)[2]),
    "P4": pool_p4,
    "lse_beta1": pool_lse_beta1,
}

#: The operator a cache should be built with: it dominates every other *graded* entry of
#: :data:`POOLINGS`, so its visited set contains theirs at any threshold at or above the cache's own.
DOMINATING_POOLING = "P3_1"

#: P0 is the exception to pointwise dominance, and the reason the thesis requires a permissive
#: cache, not merely a dominating one. It is an indicator in ``{0, 1}``, so at a node where
#: ``P1 > 1/2`` but ``P3_1 < 1`` it *exceeds* P3₁ and no pointwise bound holds. Containment survives
#: by a different argument: P0 fires only when ``P1 > 1/2``, and ``P3_1 >= P1 > 1/2``, so every node
#: P0 would expand has a cached P3₁ above one half. That is safe when the cache threshold
#: is at most one half, hence this constant, and hence the thesis's ``tau_min <= 1/2`` requirement.
P0_MAX_CACHE_THRESHOLD = 0.5


def containment_threshold(tau_replay: float) -> float:
    """``g(t)``, the loosest cache threshold that still contains a re-run at ``tau_replay``.

    A cache built at ``tau_cache <= containment_threshold(tau_replay)`` provably contains every node
    any **graded** margin pooling would expand at ``tau_replay``. See the module docstring for the
    derivation; ``g(t) < t`` always, so the condition is slightly weaker than ``tau_cache <= t``.

    P0 is not covered, see :data:`P0_MAX_CACHE_THRESHOLD`, and ``LRGate`` is not covered at all.
    Use :func:`cache_supports` for the full check.
    """
    if not 0.0 <= tau_replay < 1.0:
        raise ValueError(f"tau_replay must be in [0, 1), got {tau_replay}")
    if tau_replay == 0.0:
        return 0.0
    a = -np.log1p(-tau_replay)
    return float(a / (1.0 + a))


def cache_supports(pooling: str, tau_replay: float, tau_cache: float) -> bool:
    """Whether a cache at ``tau_cache`` provably contains ``pooling``'s expansion at ``tau_replay``.

    The single place the three different containment arguments live, so a caller never has to
    remember which operator obeys which:

    * graded margin poolings, ``tau_cache <= g(tau_replay)``;
    * ``P0``, additionally ``tau_cache <= 1/2``, since it is an indicator and not pointwise bounded;
    * ``LRGate``, never, from a margin cache. It reads tree geometry the margins do not determine.

    Returning ``False`` does not prove a violation will occur, only that none is ruled out. The
    runtime ``n_containment_misses`` count remains the authority.
    """
    if pooling not in POOLINGS:
        return False
    if pooling == "P0" and tau_cache > P0_MAX_CACHE_THRESHOLD:
        return False
    return tau_cache <= containment_threshold(tau_replay) + 1e-15


# ── The re-run ───────────────────────────────────────────────────────────────

@dataclass
class OfflineResult:
    """A traversal re-derived from cache, plus the audit the cache makes necessary."""

    result: TraversalResult
    n_containment_misses: int
    missed_nodes: list[str]

    @property
    def accepted(self) -> set[str]:
        """Terms accepted by the offline traversal."""
        return self.result.accepted

    def at_accept(self, tau_accept: float) -> TraversalResult:
        """Acceptance re-derived at another threshold, free, and never re-traverses."""
        return self.result.at_accept(tau_accept)


def replay(
    children_map: Mapping[str, Sequence[str]],
    roots: Sequence[str],
    cache: ReportCache,
    prune_scores: np.ndarray,
    accept_scores: np.ndarray,
    tau_prune: float,
    tau_accept: float,
    max_nodes: int | None = None,
    s: int | None = None,
) -> OfflineResult:
    """Re-run one report's traversal over precomputed score vectors.

    ``prune_scores`` and ``accept_scores`` are aligned to ``cache.node_ids``, the output of two
    calls into :data:`POOLINGS`. Passing vectors, not a callable is what keeps the cost of a
    large sweep in the frontier walk, not in recomputing the same pooling thousands of times.

    ``s`` must match the prefix the score vectors were computed over. It does not affect any
    decision, the scores already encode it, but it is what the reported **cost** is counted on: a
    node scored over 3 of its 10 cached segments cost three verifier calls, not ten. Leaving it
    ``None`` counts every cached segment, which is right only when the scores used all of them.
    Getting this wrong silently reports the cache's cost for every value of S, which is why
    ``calls_per_report`` is one of the numbers the thesis quotes.

    A node the traversal reaches but the cache lacks scores ``0.0`` on both decisions, the
    conservative choice, since it neither expands nor is accepted, and is recorded in
    ``missed_nodes``. **A non-zero ``n_containment_misses`` invalidates the cell.** It means the
    configuration left the cached region, and the right response is to widen the cache or narrow the
    grid, never to report the number.
    """
    misses: list[str] = []
    width = cache.s_max if s is None else min(int(s), cache.s_max)

    def evaluate_node(h: str) -> NodeEval:
        i = cache.index.get(h)
        if i is None:
            misses.append(h)
            return NodeEval(0.0, 0.0, 0, [])
        return NodeEval(
            float(prune_scores[i]),
            float(accept_scores[i]),
            int(cache.mask[i, :width].sum()),
            [],
        )

    result = traverse(
        children_map, roots, evaluate_node, tau_prune, tau_accept, max_nodes=max_nodes
    )
    return OfflineResult(result, len(misses), misses)


def evaluate_offline_cohort(
    children_map: Mapping[str, Sequence[str]],
    roots: Sequence[str],
    caches: Mapping[str, ReportCache],
    pool_pr: str,
    pool_acc: str,
    tau_prune: float,
    tau_accept: float,
    s: int | None = None,
) -> dict:
    """One configuration over a whole cohort: predictions, visited sets, cost and the audit.

    Returns ``predicted`` and ``scored`` per report, the second is the traversal's *candidate* set,
    which is what the coverage diagnostic and conformal risk control are defined on, and is not
    recoverable from the predictions alone.
    """
    pr_fn, acc_fn = POOLINGS[pool_pr], POOLINGS[pool_acc]
    predicted: dict[str, set[str]] = {}
    scored: dict[str, set[str]] = {}
    n_calls = n_misses = 0

    for rid, cache in caches.items():
        res = replay(
            children_map, roots, cache,
            pr_fn(cache, s), acc_fn(cache, s),
            tau_prune, tau_accept, s=s,
        )
        predicted[rid] = set(res.result.accepted)
        scored[rid] = set(res.result.visits)
        n_calls += res.result.n_slm_calls
        n_misses += res.n_containment_misses

    return {
        "predicted": predicted,
        "scored": scored,
        "n_slm_calls": n_calls,
        "n_reports": len(caches),
        "calls_per_report": n_calls / len(caches) if caches else 0.0,
        "n_containment_misses": n_misses,
        "config": {
            "pool_pr": pool_pr, "pool_acc": pool_acc,
            "tau_prune": tau_prune, "tau_accept": tau_accept, "S": s,
        },
    }


# ── Bottleneck reachability: the whole tau_prune axis in one pass ─────────────
#
# :func:`replay` walks the frontier once per threshold. That is the reference implementation and it
# stays, but it does not scale to the grid the thesis needs. The synthetic-sentence score store's cache is exhaustive, the
# permissive traversal reached all 18 354 phenotypic-abnormality nodes, so a single cell costs
# ~18k node visits per report, and a tau grid fine enough to invert Eq. (10) over eight poolings and
# four values of S runs to billions of visits.
#
# It is also unnecessary. ``traverse`` seeds every root unconditionally, visits a child iff some
# visited node expanded it, and expands ``v`` iff ``prune_score(v) >= tau``. With no ``max_nodes``
# cap that rule is order-independent, so whether ``v`` is visited depends on tau only through the
# **bottleneck** of the best root-to-parent path:
#
#     r(v) = +inf                                        v is a root
#     r(v) = max over parents p of min(r(p), score(p))    otherwise
#
#     visited(v, tau)  <=>  r(v) >= tau
#     expanded(v, tau) <=>  r(v) >= tau and score(v) >= tau
#
# r is a *widest-path* (maximin) value and one topological pass computes it for every tau at once.
# Two things follow that are results, not optimisations:
#
# 1. **The tau_prune grid disappears.** The per-report miss rate of the pruning loss,
#    ``L_i(tau) = |{y in Y_i : r(y) < tau}| / |Y_i|``, is a step function whose only breakpoints are
#    The r-values of that report's annotated terms. Feeding those to ``crc_lambda`` inverts the
#    risk bound without approximation, instead of on a five-point quantile grid.
# 2. **The acceptance axis becomes boolean algebra** over two node vectors, so the inner-fold
#    selection is numpy, not a nest of traversals.
#
# ``max_nodes`` is the one thing this cannot express, a cap makes the visited set depend on frontier
# order, so :func:`bottleneck_scores` is exact only for the uncapped traversal every shipped
# configuration uses. ``tests/unit/test_tree_replay.py`` pins the two against each other, the same
# way the vectorised poolings above are fixed to ``hpo_extraction.treephenorag.pooling``.


@dataclass(frozen=True)
class TraversalGraph:
    """The DAG below the layer-1 roots, laid out so a threshold sweep is a handful of numpy calls.

    ``node_ids`` is in topological order, so every node's parents precede it. The ``layer_*`` arrays
    group nodes into longest-path layers: within a layer no node is another's ancestor, so a whole
    layer's bottleneck values reduce in one ``np.maximum.reduceat``. The graph depends only on the
    ontology, never on a report, so it is built once and reused across every report, pooling and S.
    """

    node_ids: tuple[str, ...]
    index: dict[str, int]
    is_root: np.ndarray          # (n,) bool
    depth: np.ndarray            # (n,) int, BFS depth, roots at 1, independent of tau
    #: Per layer (excluding layer 0, the roots): the graph slots written, the parent slots read,
    #: and the reduceat offsets that group the latter by the former.
    layer_nodes: tuple[np.ndarray, ...]
    layer_parents: tuple[np.ndarray, ...]
    layer_offsets: tuple[np.ndarray, ...]

    @property
    def n_nodes(self) -> int:
        """Number of terms in the traversal graph."""
        return len(self.node_ids)


def build_traversal_graph(
    children_map: Mapping[str, Sequence[str]],
    roots: Sequence[str],
) -> TraversalGraph:
    """Lay out ``children_map`` for :func:`bottleneck_scores`. Ontology-only, so build it once."""
    from hpo_extraction.evaluation.metrics.traversal import bfs_depths, topological_order

    order, parents_within = topological_order(children_map, roots)
    index = {h: i for i, h in enumerate(order)}
    n = len(order)

    root_set = set(roots)
    is_root = np.array([h in root_set for h in order], dtype=bool)

    depths = bfs_depths(children_map, roots)
    depth = np.array([depths.get(h, 0) for h in order], dtype=np.int32)

    # Longest-path layering. Roots are forced to layer 0 whatever their in-degree: ``traverse``
    # seeds them regardless of any parent, so their r is +inf and nothing upstream can matter. Every
    # non-root then sits strictly below all of its parents, which is what makes the sweep valid.
    layer = np.zeros(n, dtype=np.int32)
    for i, h in enumerate(order):
        if is_root[i]:
            continue
        ps = parents_within.get(h, ())
        layer[i] = 1 + max((int(layer[index[p]]) for p in ps), default=-1)

    layer_nodes: list[np.ndarray] = []
    layer_parents: list[np.ndarray] = []
    layer_offsets: list[np.ndarray] = []
    for lv in range(1, (int(layer.max()) + 1) if n else 1):
        slots = np.flatnonzero(layer == lv)
        if slots.size == 0:
            continue
        offsets: list[int] = []
        flat: list[int] = []
        for i in slots:
            offsets.append(len(flat))
            flat.extend(index[p] for p in parents_within[order[i]])
        layer_nodes.append(slots.astype(np.int64))
        layer_parents.append(np.array(flat, dtype=np.int64))
        layer_offsets.append(np.array(offsets, dtype=np.int64))

    return TraversalGraph(
        node_ids=tuple(order), index=index, is_root=is_root, depth=depth,
        layer_nodes=tuple(layer_nodes), layer_parents=tuple(layer_parents),
        layer_offsets=tuple(layer_offsets),
    )


def bottleneck_scores(graph: TraversalGraph, scores: np.ndarray) -> np.ndarray:
    """``r(v)`` for every node, the largest ``tau_prune`` at which ``v`` is still visited.

    ``scores`` is the expansion score aligned to ``graph.node_ids`` (see :func:`align_to_graph`).
    Nodes the cache lacks must already carry the conservative ``0.0`` that :func:`replay` gives
    them, so the two agree about what leaving the cache means.

    Roots get ``+inf`` (seeded unconditionally) and anything unreachable ``-inf``, so ``r >= tau``
    is the visited predicate at every finite threshold with no special case.
    """
    r = np.full(graph.n_nodes, -np.inf, dtype=np.float64)
    r[graph.is_root] = np.inf
    for slots, parents, offsets in zip(
        graph.layer_nodes, graph.layer_parents, graph.layer_offsets
    ):
        if parents.size == 0:
            continue
        through = np.minimum(r[parents], scores[parents])
        reduced = np.maximum.reduceat(through, offsets)
        # A root reached again from below keeps +inf. A path must never lower an existing value.
        np.maximum(r[slots], reduced, out=reduced)
        r[slots] = reduced
    return r


def align_to_graph(
    graph: TraversalGraph,
    node_ids: Sequence[str],
    values: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Scatter a cache-aligned vector onto graph slots, filling absent nodes with ``fill``.

    ``fill=0.0`` is the conservative default and matches :func:`replay`: a node the cache lacks
    neither expands nor is accepted. It does **not** make the cell valid, see
    :func:`containment_misses`.
    """
    out = np.full(graph.n_nodes, float(fill), dtype=np.float64)
    for h, v in zip(node_ids, values):
        i = graph.index.get(h)
        if i is not None:
            out[i] = v
    return out


def in_cache_mask(graph: TraversalGraph, node_ids: Iterable[str]) -> np.ndarray:
    """``(n,)`` bool, which graph slots the cache actually holds a score for."""
    out = np.zeros(graph.n_nodes, dtype=bool)
    for h in node_ids:
        i = graph.index.get(h)
        if i is not None:
            out[i] = True
    return out


def visited_at(r: np.ndarray, tau_prune: float) -> np.ndarray:
    """``(n,)`` bool, the traversal's candidate set ``C_i(tau)``."""
    return r >= tau_prune


def expanded_at(r: np.ndarray, scores: np.ndarray, tau_prune: float) -> np.ndarray:
    """``(n,)`` bool, visited *and* over threshold, i.e. The nodes that reveal their children."""
    return (r >= tau_prune) & (scores >= tau_prune)


def calls_at(r: np.ndarray, n_valid: np.ndarray, tau_prune: float) -> int:
    """Verifier calls for one report: the segments actually scored, summed over visited nodes.

    ``n_valid`` is the per-node count of cached segments **at the S being re-run**, a node
    retrieved on three of ten cached segments cost three calls, not ten.
    """
    return int(n_valid[r >= tau_prune].sum())


def containment_misses(r: np.ndarray, in_cache: np.ndarray, tau_prune: float) -> int:
    """Visited nodes the cache has no score for. Non-zero invalidates the cell. It is not a warning.

    The audit :func:`replay` performs, expressed on the sweep. With an exhaustive cache this is zero
    by design, not by the ``g(tau)`` bound, which is worth asserting, not
    assuming.
    """
    return int((visited_at(r, tau_prune) & ~in_cache).sum())


def coverage_breakpoints(r: np.ndarray, positions: Iterable[int]) -> np.ndarray:
    """The thresholds at which a report's miss rate can change: the ``r`` of its annotated terms.

    ``L_i(tau) = |{y : r(y) < tau}| / |Y_i|`` is constant between consecutive ground truth r-values, so this
    set is *exhaustive*, no finer grid exists and no coarser one is faithful. Feeding it to
    :func:`hpo_extraction.evaluation.stats.crc.crc_lambda` inverts the risk bound.

    ``+inf`` (an annotated term at a root) is dropped: no finite threshold ever misses it.
    """
    vals = np.array([r[i] for i in positions], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return np.unique(vals)


# ── The cache on disk: one streaming pass, then numpy ────────────────────────
#
# :func:`load_score_cache` parses the whole ``_calls.jsonl`` into nested Python containers. That was
# fine for an earlier exploratory run (618 MB, 18 001 calls/report). It is not fine for the synthetic-sentence score store: the permissive
# traversal reaches all 18 354 nodes at S = 10, so a cohort is ~21.7 M rows and ~6-7 GB of JSON,
# which costs minutes of ``json.loads`` and gigabytes of interpreter objects **every time a stage
# runs**. The re-run stages run many times.
#
# So the cache is ingested **once** into a columnar ``.npz`` and read from there afterwards. The
# conversion also keeps three columns ``load_score_cache`` throws away and the analysis needs:
#
# * ``cosine`` and ``sent_index``, the retrieval similarity and *which sentence* was retrieved.
#   Together with the curated ground truth's own evidence segments, ``sent_index`` is what splits a
#   *retrieval miss* ("the sentence carrying the evidence never reached the verifier") from a
#   *judgement miss* ("it did, and the verifier said no") in the recall decomposition. Without it
#   The two are indistinguishable and the decomposition collapses to a guess.
# * ``captured_mass``, derived here as ``exp(logaddexp(logit_yes, logit_no) - logsumexp_all)``
#,   not read, so it is available for caches that were already running when the field was
#   added. Note it is over the **singleton** Yes/No token ids the verifier actually uses, not the
#   token-variant sets the thesis defines. Say so wherever it is reported.
# * ``wall_clock_s`` / ``n_prompt_tokens``, present only for instrumented runs (the synthetic-sentence score store onward),
#   filled with NaN / -1 otherwise so a reader can tell "not measured" from "measured as zero".

#: Written into every ``.npz`` so a stale conversion is detectable, not silently re-run.
CACHE_NPZ_VERSION = 1


def _report_cache_from_columns(rid, node_ids, margins, mask, extras) -> ReportCache:
    return ReportCache(
        rid, list(node_ids), np.asarray(margins, dtype=np.float64),
        np.asarray(mask, dtype=bool),
        **{k: np.asarray(v) for k, v in extras.items()},
    )


def ingest_score_cache(
    calls_path: str | Path,
    out_path: str | Path,
    ctx_type: str = "union",
    report_ids: Iterable[str] | None = None,
) -> dict:
    """Stream ``{variant}_calls.jsonl`` into a columnar ``.npz``. Run once per cache.

    Memory is bounded by **one report**, not by the file: rows arrive grouped by report (the driver
    appends per report, and merged shards concatenate whole shard files), so each report's rows are
    accumulated, converted to arrays and released before the next begins. A report id that reappears
    after its group closed means the file was interleaved, not concatenated, and that raises, silently merging it would produce a cache whose ``rank`` ordering is not retrieval order.

    Returns a provenance dict. It is also stored in the archive.
    """
    calls_path, out_path = Path(calls_path), Path(out_path)
    wanted = set(report_ids) if report_ids is not None else None

    report_order: list[str] = []
    closed: set[str] = set()
    per_node: dict[str, dict[int, dict]] = {}
    current: str | None = None

    node_ids_all: list[str] = []
    starts: list[int] = [0]
    blocks: dict[str, list[np.ndarray]] = {k: [] for k in
                                           ("margin", "mask", "cosine", "captured_mass",
                                            "wall_clock_s", "n_prompt_tokens", "sent_index")}
    s_max_seen = 0
    n_rows = n_bad = 0
    saw_clock = saw_tokens = False

    def flush(rid: str) -> None:
        nonlocal s_max_seen
        if not per_node:
            return
        nodes = sorted(per_node)
        width = max(max(d) for d in per_node.values())
        s_max_seen = max(s_max_seen, width)
        shape = (len(nodes), width)
        margin = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        cosine = np.full(shape, np.nan, dtype=np.float32)
        cmass = np.full(shape, np.nan, dtype=np.float32)
        clock = np.full(shape, np.nan, dtype=np.float32)
        tokens = np.full(shape, -1, dtype=np.int32)
        sent = np.full(shape, -1, dtype=np.int32)
        for i, h in enumerate(nodes):
            for rank, rec in per_node[h].items():
                j = rank - 1
                margin[i, j] = rec["margin"]
                mask[i, j] = True
                cosine[i, j] = rec["cosine"]
                cmass[i, j] = rec["captured_mass"]
                if rec["clock"] is not None:
                    clock[i, j] = rec["clock"]
                if rec["tokens"] is not None:
                    tokens[i, j] = rec["tokens"]
                sent[i, j] = rec["sent_index"]
        node_ids_all.extend(nodes)
        starts.append(starts[-1] + len(nodes))
        for key, arr in (("margin", margin), ("mask", mask), ("cosine", cosine),
                         ("captured_mass", cmass), ("wall_clock_s", clock),
                         ("n_prompt_tokens", tokens), ("sent_index", sent)):
            blocks[key].append(arr)
        report_order.append(rid)
        closed.add(rid)
        per_node.clear()

    with calls_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue  # truncated tail from a hard kill. The driver appends per report
            if rec.get("ctx_type") != ctx_type:
                continue
            rid = rec["report_id"]
            if wanted is not None and rid not in wanted:
                continue
            if rid != current:
                if current is not None:
                    flush(current)
                if rid in closed:
                    raise ValueError(
                        f"report {rid!r} reappears after its rows ended in {calls_path.name}. "
                        "The file is interleaved, not concatenated — rebuild it with "
                        "src/evaluation/merge_shards.py rather than by cat-ing shards together."
                    )
                current = rid
            n_rows += 1
            ly, ln = float(rec["logit_yes"]), float(rec["logit_no"])
            # Prefer the recorded value. Derive it for caches written before the field existed.
            # Identical either way, hpo_extraction.treephenorag.score_store._captured_mass is the same expression,
            # so a mixed cohort of old and new caches stays comparable.
            if rec.get("captured_mass") is not None:
                cmass = float(rec["captured_mass"])
            else:
                lse = rec.get("logsumexp_all")
                cmass = (float(np.exp(np.logaddexp(ly, ln) - float(lse)))
                         if lse is not None else float("nan"))
            clock = rec.get("wall_clock_s")
            tokens = rec.get("n_prompt_tokens")
            saw_clock = saw_clock or clock is not None
            saw_tokens = saw_tokens or tokens is not None
            per_node.setdefault(rec["hpo_id"], {})[int(rec["rank"])] = {
                "margin": float(rec["margin"]),
                "cosine": float(rec.get("cosine_sim", np.nan)),
                "captured_mass": cmass,
                "clock": None if clock is None else float(clock),
                "tokens": None if tokens is None else int(tokens),
                "sent_index": int(rec.get("sent_index", -1)),
            }
    if current is not None:
        flush(current)

    def stack(key: str) -> np.ndarray:
        """Right-pad every report's block to the cohort-wide S_max before concatenating."""
        parts = []
        for arr in blocks[key]:
            if arr.shape[1] < s_max_seen:
                pad = np.zeros((arr.shape[0], s_max_seen - arr.shape[1]), dtype=arr.dtype)
                if arr.dtype == np.float32:
                    pad[:] = np.nan
                elif arr.dtype == np.int32:
                    pad[:] = -1
                arr = np.hstack([arr, pad])
            parts.append(arr)
        return np.concatenate(parts) if parts else np.zeros((0, s_max_seen))

    provenance = {
        "version": CACHE_NPZ_VERSION,
        "source": str(calls_path),
        "ctx_type": ctx_type,
        "n_reports": len(report_order),
        "n_rows": n_rows,
        "n_undecodable_lines": n_bad,
        "s_max": s_max_seen,
        "has_wall_clock": saw_clock,
        "has_prompt_tokens": saw_tokens,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        report_ids=np.array(report_order, dtype=object),
        node_ids=np.array(node_ids_all, dtype=object),
        report_starts=np.array(starts, dtype=np.int64),
        provenance=np.array(json.dumps(provenance), dtype=object),
        **{k: stack(k) for k in blocks},
    )
    return provenance


def load_cache_npz(
    path: str | Path,
    report_ids: Iterable[str] | None = None,
    with_extras: bool = True,
) -> dict[str, ReportCache]:
    """Read an :func:`ingest_score_cache` archive back into one :class:`ReportCache` per report.

    Equivalent to :func:`load_score_cache` on the same file, but seconds, not minutes, and it
    carries ``cosine`` / ``captured_mass`` / ``wall_clock_s`` / ``n_prompt_tokens`` as well.
    """
    path = Path(path)
    wanted = set(report_ids) if report_ids is not None else None
    with np.load(path, allow_pickle=True) as z:
        rids = list(z["report_ids"])
        nodes = list(z["node_ids"])
        starts = z["report_starts"]
        margins, mask = z["margin"], z["mask"]
        extra_arrays = (
            {k: z[k] for k in ("cosine", "captured_mass", "wall_clock_s", "n_prompt_tokens",
                               "sent_index")}
            if with_extras else {}
        )
    caches: dict[str, ReportCache] = {}
    for r, rid in enumerate(rids):
        rid = str(rid)
        if wanted is not None and rid not in wanted:
            continue
        a, b = int(starts[r]), int(starts[r + 1])
        caches[rid] = _report_cache_from_columns(
            rid, [str(h) for h in nodes[a:b]], margins[a:b], mask[a:b],
            {k: v[a:b] for k, v in extra_arrays.items()},
        )
    return caches


def read_cache_provenance(path: str | Path) -> dict:
    """The provenance dict :func:`ingest_score_cache` stored, without loading the arrays."""
    with np.load(Path(path), allow_pickle=True) as z:
        return json.loads(str(z["provenance"]))
