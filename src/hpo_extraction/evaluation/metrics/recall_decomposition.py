r"""§Recall decomposition, where each missed annotated term was actually lost.

``04_TreePhenoRAG.tex``:

    Each false negative :math:`y` is attributed to the term :math:`u` at which it was lost:
    :math:`u = y` if :math:`y` was scored, and otherwise the blocking ancestor of maximal depth.
    The loss at :math:`u` is a *retrieval miss* if no evidence segment of :math:`u` was retrieved,
    a *judgement miss* if evidence was retrieved but its largest margin is at most
    :math:`-\delta_m`, and a *pooling miss* if that margin exceeds :math:`-\delta_m` but the
    relevant pooled score fell below its threshold. Remaining cases, such as an annotated term replaced
    by an accepted descendant, form a residual category. Only pooling misses are addressable by the
    choice of pooling operator.

**The top-level split is the coverage identity, not a fifth bucket.** §Diagnostics defines
:math:`\mathrm{Cov} = \sum_i |Y_i \cap \mathcal{C}_i| / \sum_i |Y_i|` and reads
:math:`1 - \mathrm{Cov}` as "the recall lost to pruning" and :math:`\mathrm{Cov} - R_\mu` as "the
recall lost at scored terms". This module partitions every false negative the same way, so the two
numbers are the same number:

* **pruning**, :math:`y \notin \mathcal{C}_i`. The term was never scored, so nothing downstream
  had a chance. Attributed to its deepest blocking ancestor, and the cause *at that ancestor* is
  reported alongside, because "pruning" names the mechanism, not the reason.
* **retrieval / judgement / pooling / residual**, :math:`y \in \mathcal{C}_i` but not accepted.

Summing the five gives every false negative once.

**Retrieval misses need external evidence, and say so when they do not have it.** "No evidence
segment of :math:`u` was retrieved" is not decidable from the run alone: the run knows which
sentences it retrieved, not which sentence carried the finding. The HCY curated ground truth does know, every annotation names a segment and a trigger word, so when an ``evidence`` map is supplied the
bucket is exact. Without one, the only honest fallback is the degenerate test "the term was scored
on no segment at all", which on an exhaustive cache is almost never true. The result then reports
``evidence_available=False`` and the retrieval bucket must not be quoted. A decomposition that
silently guessed here would put the blame for the tree's recall on whichever stage the fallback
happened to favour.

:math:`\delta_m` is left ``\todo`` in the draft. The default here is ``0.0``, which reads as "the
verifier answered No on every segment it saw", the only value that needs no further justification.
:func:`recall_decomposition` takes it as a parameter so the sensitivity can be swept and reported.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

#: The partition of false negatives. ``pruning`` is the coverage half. The rest are scored terms.
DECOMPOSITION_BUCKETS = ("pruning", "retrieval", "judgement", "pooling", "residual")

#: Why the *blocking ancestor* of a pruned term failed to expand. Same vocabulary, one level up.
BLOCKING_CAUSES = ("retrieval", "judgement", "pooling", "residual")


def _cause_at(
    position: int | None,
    margins_row: np.ndarray | None,
    mask_row: np.ndarray | None,
    sent_row: np.ndarray | None,
    evidence: frozenset[int] | None,
    pooled: float | None,
    threshold: float | None,
    delta_m: float,
) -> str:
    """The three-way test of §Error analysis, applied at one term. See the module docstring."""
    if position is None or mask_row is None or not bool(mask_row.any()):
        # Never scored on any segment at all, nothing was retrieved, by any reading.
        return "retrieval"

    valid = np.asarray(mask_row, dtype=bool)
    if evidence is not None:
        retrieved = set(int(s) for s in np.asarray(sent_row)[valid]) if sent_row is not None \
            else set()
        if not (retrieved & evidence):
            return "retrieval"

    largest = float(np.asarray(margins_row)[valid].max())
    if largest <= -delta_m:
        return "judgement"
    if pooled is not None and threshold is not None and pooled < threshold:
        return "pooling"
    return "residual"


def attribute_false_negative(
    missed: str,
    scored: Iterable[str],
    expanded: Iterable[str],
    view,
    depths: Mapping[str, int] | None = None,
) -> dict:
    """Where a missed annotated term was lost: at itself, or at its deepest blocking ancestor.

    ``scored`` is the traversal's candidate set :math:`\\mathcal{C}_i` and ``expanded`` the subset
    that revealed its children. A term outside ``scored`` was blocked. Its blocking set is
    :math:`\\mathrm{An}(y) \\cap (\\mathcal{C}_i \\setminus \\mathcal{X}_i)` and the attribution
    point is the deepest member, which is how far the traversal actually got towards it.

    Returns ``{"term", "lost_at", "pruned", "blocking_depth"}``. ``lost_at`` is ``None`` only when a
    blocked term has no scored-but-unexpanded ancestor at all, which means it is unreachable in the
    graph rather than pruned, a different failure, and one the caller must not file as pruning.
    """
    scored = set(scored)
    resolved = view.resolve(missed) or missed
    if resolved in scored:
        return {"term": resolved, "lost_at": resolved, "pruned": False, "blocking_depth": None}

    blockers = (set(view.ancestors(resolved)) - {resolved}) & (scored - set(expanded))
    if not blockers:
        return {"term": resolved, "lost_at": None, "pruned": True, "blocking_depth": None}

    def depth_of(node: str) -> int:
        if depths is not None and node in depths:
            return int(depths[node])
        d = view.depth(node)
        return int(d) if d is not None else 0

    deepest = max(sorted(blockers), key=depth_of)
    return {"term": resolved, "lost_at": deepest, "pruned": True,
            "blocking_depth": depth_of(deepest)}


def recall_decomposition(
    gold_by_report: Mapping[str, Iterable[str]],
    predicted_by_report: Mapping[str, Iterable[str]],
    scored_by_report: Mapping[str, Iterable[str]],
    expanded_by_report: Mapping[str, Iterable[str]],
    caches: Mapping[str, object],
    view,
    pooled_by_report: Mapping[str, Mapping[str, float]] | None = None,
    threshold: float | None = None,
    evidence: Mapping[str, Mapping[str, Iterable[int]]] | None = None,
    delta_m: float = 0.0,
    depths: Mapping[str, int] | None = None,
) -> dict:
    """Decompose every false negative over a cohort. See the module docstring for the buckets.

    Args:
        scored_by_report: :math:`\\mathcal{C}_i` per report, the traversal's candidate set. Not
            recoverable from the predictions, which is why it is a separate argument.
        expanded_by_report: :math:`\\mathcal{X}_i` per report.
        caches: ``{report_id: ReportCache}``, for the per-segment margins and ``sent_index``.
        pooled_by_report: ``{report: {hpo: accept score}}``. Needed to separate *pooling* from
            *residual*. Without it the two collapse into ``residual``.
        evidence: ``{report: {hpo: [segment indices]}}`` from the curated ground truth. Without it the
            retrieval bucket is not measurable, see the module docstring.
        delta_m: the judgement margin. ``0.0`` reads as "the verifier said No on every segment".

    Returns counts and fractions over :data:`DECOMPOSITION_BUCKETS`, the blocking-cause breakdown
    for the pruned share, the blocking-depth histogram, and ``per_term`` rows for inspection.
    """
    counts: Counter = Counter()
    blocking_causes: Counter = Counter()
    blocking_depths: Counter = Counter()
    rows: list[dict] = []
    n_gold = 0

    for rid, gold in gold_by_report.items():
        predicted = {view.resolve(p) or p for p in predicted_by_report.get(rid, ())}
        scored = set(scored_by_report.get(rid, ()))
        expanded = set(expanded_by_report.get(rid, ()))
        cache = caches.get(rid)
        pooled = (pooled_by_report or {}).get(rid, {})
        ev_report = (evidence or {}).get(rid, {})

        for raw in gold:
            term = view.resolve(raw) or raw
            n_gold += 1
            if term in predicted:
                continue

            where = attribute_false_negative(term, scored, expanded, view, depths)
            at = where["lost_at"]

            # The cause is always evaluated AT the attribution point, which for a pruned term is
            # The blocking ancestor, not the annotated term itself. Evaluating it at the ground truth
            # term would describe a verifier call that never happened.
            index = getattr(cache, "index", {}) if cache is not None else {}
            position = index.get(at) if at is not None else None
            ev = ev_report.get(at) if at is not None else None
            cause = _cause_at(
                position,
                None if position is None else cache.margins[position],
                None if position is None else cache.mask[position],
                None if position is None or cache.sent_index is None
                else cache.sent_index[position],
                None if ev is None else frozenset(int(s) for s in ev),
                pooled.get(at) if at is not None else None,
                threshold,
                delta_m,
            )

            if where["pruned"]:
                counts["pruning"] += 1
                blocking_causes[cause] += 1
                if where["blocking_depth"] is not None:
                    blocking_depths[int(where["blocking_depth"])] += 1
            else:
                counts[cause] += 1
            rows.append({"report_id": rid, **where, "cause": cause})

    n_fn = sum(counts.values())
    n_pruned = counts.get("pruning", 0)
    return {
        "n_gold": n_gold,
        "n_false_negative": n_fn,
        "counts": {b: counts.get(b, 0) for b in DECOMPOSITION_BUCKETS},
        "fractions": {b: (counts.get(b, 0) / n_fn if n_fn else 0.0)
                      for b in DECOMPOSITION_BUCKETS},
        # The coverage identity, recomputed from the same rows that produced the table above.
        "coverage_loss": n_pruned / n_gold if n_gold else 0.0,
        "loss_at_scored_terms": (n_fn - n_pruned) / n_gold if n_gold else 0.0,
        "blocking_causes": {c: blocking_causes.get(c, 0) for c in BLOCKING_CAUSES},
        "blocking_depth_histogram": dict(sorted(blocking_depths.items())),
        "evidence_available": bool(evidence),
        "delta_m": float(delta_m),
        "per_term": rows,
    }


def delta_m_sensitivity(
    deltas: Sequence[float],
    **kwargs,
) -> list[dict]:
    """:func:`recall_decomposition` over several :math:`\\delta_m`, since the draft leaves it open.

    A judgement/pooling split that moves a lot across this sweep is a split the chapter should not
    lean on. One that barely moves is a result.
    """
    out = []
    for d in deltas:
        got = recall_decomposition(delta_m=float(d), **kwargs)
        out.append({"delta_m": float(d), **got["counts"],
                    "n_false_negative": got["n_false_negative"]})
    return out
