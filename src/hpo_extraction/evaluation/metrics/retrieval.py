"""§Core extraction quality, retrieval stage, :math:`P@S` / :math:`R@S` (skeleton eq. 2).

"The retrieval stage is evaluated separately, since a phenotype that is never retrieved cannot be
recovered by the SLM." Two different things are called retrieval in this pipeline and the skeleton
touches both, so both are implemented here:

**Segment-level** (eq. 2 as written). For a candidate phenotype :math:`p_i`, the report is split
into segments and the :math:`S` most cosine-similar to :math:`p_i`'s synthetic-sentence sentences are kept.
:math:`\\mathrm{Rel}(p_i)` is the set of segments that genuinely evidence :math:`p_i`. This needs
*segment-level* annotations and is therefore only computable on an exhaustively annotated subset.

**Term-level**. At ontology scale the prior question is whether the ground truth *term* survives into the
candidate list at all, the top-:math:`M` terms of the flat retrieve-then-classify baseline, or the
visited set of the tree traversal. No segment annotations needed, so it covers every report.

:math:`R` is the quantity of interest in both cases: it upper-bounds the recall attainable by the
identification stage at any configuration. :func:`combined_recall_bound` composes it with the
traversal's reachability recall into the §traversal bound.

**Relevance follows the true-path rule.** A segment annotated with a *descendant* of :math:`q`
evidences :math:`q`: without propagating annotations up the ontology, every ancestor query would
look like a miss by design.

**Denominator convention.** :math:`P@S` divides by :math:`S` as the equation is written,
not by the number of segments actually returned. A report shorter than :math:`S` segments therefore
dilutes its own precision. This is the literal reading and it is the conservative one. The count of
affected pairs is reported as ``n_pairs_short`` so the effect is auditable.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .ontology import OntologyView


# ── Segment-level relevance (the true-path rule) ─────────────────────────────

def relevant_segments(
    segment_annotations: Mapping[object, Iterable[str]],
    query_hpo: str,
    view: OntologyView,
) -> set:
    """:math:`\\mathrm{Rel}(p_i)`, the segments of one report that evidence ``query_hpo``.

    Args:
        segment_annotations: ``{segment_id: [annotated HPO codes]}`` for a single report.
        query_hpo: the candidate phenotype :math:`p_i` being retrieved for.
        view: ontology view supplying the descendant closure.

    A segment is relevant if it carries ``query_hpo`` itself or any of its descendants (true-path
    rule). Returns an empty set for an unresolvable query, which callers must treat as "recall
    undefined", not "recall zero".
    """
    implied = view.descendants_or_self(query_hpo)
    if not implied:
        return set()
    return {
        seg_id
        for seg_id, codes in segment_annotations.items()
        if any(view.resolve(c) in implied for c in codes)
    }


def segment_pr_at_s(
    ranked_segments: Sequence,
    relevant: set,
    s: int,
) -> tuple[float, float | None, int]:
    """Skeleton eq. (2) for one ``(report, phenotype)`` pair.

    Args:
        ranked_segments: segment ids ordered by descending cosine similarity.
        relevant: :math:`\\mathrm{Rel}(p_i)` for this pair.
        s: the retrieval depth :math:`S`.

    Returns:
        ``(P@S, R@S, n_hits)``. ``R@S`` is ``None`` when :math:`|\\mathrm{Rel}(p_i)| = 0`, the
        recall of a pair with nothing to find is undefined, and averaging a zero in its place would
        depress the very ceiling this metric exists to measure.
    """
    if s <= 0:
        raise ValueError(f"S must be positive, got {s}")
    hits = len(set(ranked_segments[:s]) & relevant)
    precision = hits / s
    recall = hits / len(relevant) if relevant else None
    return precision, recall, hits


def segment_pr_curve(
    pairs: Iterable[tuple[Sequence, set]],
    s_values: Sequence[int],
) -> dict[int, dict]:
    """:math:`P@S` and :math:`R@S` across a range of :math:`S`, micro- and macro-averaged.

    "Following standard practice in retrieval-based label prediction, we report both across a range
    of :math:`S`."

    Args:
        pairs: one ``(ranked_segment_ids, relevant_set)`` per ``(report, phenotype)`` pair.
        s_values: the retrieval depths to evaluate.

    Returns:
        ``{S: {...}}`` with, per depth: ``micro_precision`` / ``micro_recall`` (pooled hits over
        pooled denominators), ``macro_precision`` / ``macro_recall`` (mean of the per-pair ratios),
        ``n_pairs``, ``n_pairs_with_relevant`` (the recall denominator's population) and
        ``n_pairs_short`` (pairs that returned fewer than :math:`S` segments, see the module
        docstring's denominator note).
    """
    materialised = [(list(ranked), set(rel)) for ranked, rel in pairs]
    out: dict[int, dict] = {}

    for s in s_values:
        hits_total = 0
        rel_total = 0
        macro_p_sum = 0.0
        macro_r_sum = 0.0
        n_pairs = 0
        n_with_rel = 0
        n_short = 0

        for ranked, rel in materialised:
            p, r, hits = segment_pr_at_s(ranked, rel, s)
            n_pairs += 1
            if len(ranked) < s:
                n_short += 1
            hits_total += hits
            macro_p_sum += p
            if r is not None:
                n_with_rel += 1
                macro_r_sum += r
                rel_total += len(rel)

        out[s] = {
            "micro_precision": hits_total / (n_pairs * s) if n_pairs else 0.0,
            "micro_recall": hits_total / rel_total if rel_total else None,
            "macro_precision": macro_p_sum / n_pairs if n_pairs else 0.0,
            "macro_recall": macro_r_sum / n_with_rel if n_with_rel else None,
            "n_pairs": n_pairs,
            "n_pairs_with_relevant": n_with_rel,
            "n_pairs_short": n_short,
        }
    return out


# ── Term-level retrieval (the ontology-scale ceiling) ────────────────────────

def term_pr_at_m(
    ranked_candidates: Sequence[str],
    gold: set[str],
    m: int,
) -> tuple[float, float | None, int]:
    """:math:`P@M` / :math:`R@M` over retrieved **terms** for one report.

    The same shape as eq. (2) with the unit changed from segments to ontology terms:
    "did the annotated term survive into the candidate list the SLM is asked about?". ``R@M`` is the
    hard ceiling on the identification stage's recall for that report.

    Returns ``(P@M, R@M, n_hits)``; ``R@M`` is ``None`` for a report with no annotated terms.
    """
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    hits = len(set(ranked_candidates[:m]) & gold)
    return hits / m, (hits / len(gold) if gold else None), hits


def term_pr_curve(
    reports: Iterable[tuple[Sequence[str], set[str]]],
    m_values: Sequence[int],
) -> dict[int, dict]:
    """Term-level :math:`P@M` / :math:`R@M` across a range of :math:`M`, micro and macro.

    Args:
        reports: one ``(ranked_candidate_terms, gold_set)`` per report.
        m_values: the candidate-list sizes to evaluate.
    """
    materialised = [(list(c), set(g)) for c, g in reports]
    out: dict[int, dict] = {}

    for m in m_values:
        hits_total = 0
        gold_total = 0
        macro_p_sum = 0.0
        macro_r_sum = 0.0
        n_reports = 0
        n_with_gold = 0

        for candidates, gold in materialised:
            p, r, hits = term_pr_at_m(candidates, gold, m)
            n_reports += 1
            hits_total += hits
            macro_p_sum += p
            if r is not None:
                n_with_gold += 1
                macro_r_sum += r
                gold_total += len(gold)

        out[m] = {
            "micro_precision": hits_total / (n_reports * m) if n_reports else 0.0,
            "micro_recall": hits_total / gold_total if gold_total else None,
            "macro_precision": macro_p_sum / n_reports if n_reports else 0.0,
            "macro_recall": macro_r_sum / n_with_gold if n_with_gold else None,
            "n_reports": n_reports,
            "n_reports_with_gold": n_with_gold,
        }
    return out


def candidate_set_recall(candidates: Iterable[str], gold: Iterable[str]) -> float | None:
    """Recall of an *unranked* candidate set, the tree traversal's visited-set ceiling.

    The traversal has no global ranking, so :math:`R@M` collapses to plain set recall over whatever
    node set the search actually reached. ``None`` for a report with no annotated terms.
    """
    gold = set(gold)
    if not gold:
        return None
    return len(set(candidates) & gold) / len(gold)


# ── The composed bound of §Hierarchical Candidate Traversal ──────────────────

def combined_recall_bound(retrieval_recall: float, reachability_recall: float) -> float:
    """Upper bound on end-to-end recall: retrieval recall at the leaf × reachability recall.

    "The combined-system recall is therefore bounded above by the product of the retrieval recall at
    the leaf and the probability that every ancestor on the path survives pruning."

    The two factors are treated as independent, which is what the product form asserts. They are not
    guaranteed to be, a term whose surface expression is weak is plausibly both hard to retrieve
    and hard to keep reachable, so this is an *upper bound on the bound*, and any measured recall
    that exceeds it indicates the independence assumption, not the measurement, has failed.
    """
    for name, v in (("retrieval_recall", retrieval_recall),
                    ("reachability_recall", reachability_recall)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {v}")
    return retrieval_recall * reachability_recall
