"""§Severity of near-misses and the false-positive taxonomy (skeleton eq. 5).

"To characterise how far wrong the system is when it is wrong" the skeleton reports, for each false
positive, the shortest **undirected** path length to the nearest annotated term, and partitions false
positives into ancestors, descendants, siblings, unrelated terms and hallucinations following the
taxonomy of Garcia et al., so the error profile is directly comparable to the published RAG-HPO
decomposition.

**Reference term.** Every classification is made against the *nearest* annotated term of the same report,
nearest by :func:`nearest_gold_distance`, with ties broken by lexicographically smallest HPO id.
The tie-break is arbitrary but fixed, which is what makes the taxonomy reproducible. The chosen
reference term is returned alongside the label so any individual decision can be inspected.

**Bucket definitions and their precedence.** Applied in this order, first match wins:

===============  ===========================================================================
``hallucination`` the predicted code names no term the ontology knows (not in ``data``, not an
                  alt id). This is the bucket Garcia et al. use for invented codes.
``out_of_subtree`` a real HPO term that lies outside the phenotypic-abnormality subtree the
                  metrics are defined over (inheritance modes, clinical modifiers, …). *Not* one
                  of Garcia's five, it is a hygiene bucket, reported separately so such codes are
                  neither counted as hallucinations nor silently dropped.
``no_gold``       the report has no annotated terms at all, so there is no reference to be near. Also
                  not a Garcia bucket. Reported separately rather than folded into ``unrelated``,
                  which would assert a closeness judgement the data cannot support.
``ancestor``      a strict ancestor of the reference annotated term, the system was too general.
``descendant``    a strict descendant of the reference annotated term, too specific.
``sibling``       shares at least one direct ``is_a`` parent with the reference annotated term.
``unrelated``     everything else.
===============  ===========================================================================

Note that ``sibling`` is the literal reading, a common *direct* parent, not the
lowest-common-subsumer-depth heuristic used by the repo's older
``hpo_extraction.evaluation.tree_error_analysis.classify_error``. The two modules disagree. This one
follows the skeleton and Garcia et al., the other predates them and is left untouched.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .ontology import OntologyView

#: The five buckets of the Garcia et al. decomposition, in "closeness" order.
GARCIA_BUCKETS = ("ancestor", "descendant", "sibling", "unrelated", "hallucination")

#: Buckets this package adds so that nothing is silently dropped or misfiled.
HYGIENE_BUCKETS = ("out_of_subtree", "no_gold")


# ── Eq. (5): distance to the nearest annotated term ───────────────────────────────

def nearest_gold(
    pred: str,
    gold: Iterable[str],
    view: OntologyView,
) -> tuple[str | None, int | None]:
    """The annotated term nearest to ``pred`` and that undirected distance.

    Implements the :math:`\\arg\\min` of eq. (5). Ties are broken by lexicographically smallest HPO
    id so repeated runs agree. Returns ``(None, None)`` when the prediction is unresolvable or the
    ground-truth set contains no scorable term.
    """
    resolved = view.resolve(pred)
    if resolved is None:
        return None, None
    gold_res = frozenset({view.resolve(g) for g in gold} - {None})
    if not gold_res:
        return None, None
    hit = _nearest_table(view, gold_res).get(resolved)
    return (hit[1], hit[0]) if hit is not None else (None, None)


#: Distinct ground-truth sets whose nearest-ground truth table is kept. One per report per cohort is a few hundred;
#: The bound only protects a caller that streams far more.
_NEAREST_CACHE_SIZE = 4096
_nearest_cache: "dict[tuple[OntologyView, frozenset], dict[str, tuple[int, str]]]" = {}


def _nearest_table(view: OntologyView, gold_res: frozenset) -> "dict[str, tuple[int, str]]":
    """``{node: (distance, nearest gold)}`` for every node reachable from ``gold_res``.

    One multi-source BFS over the undirected graph ``view.neighbours`` defines, run once per
    distinct ground-truth set instead of one bidirectional BFS per (false positive, annotated term) pair. That
    pair loop was the whole cost of the error analysis: with 921 scored configurations it ran
    for 9+ hours (the result-table library job 9612175, 2026-09-27), because every permissive sweep point brings
    thousands of false positives and each paid ~|ground truth| full searches of the ontology.

    what :func:`nearest_gold` computed before. Levels are expanded in order, so a node's
    distance is its shortest path to the set. And the nearest annotated terms of a node at distance d are
    the union of those of its neighbours at distance d-1, so carrying the lexicographic minimum
    reproduces the old tie-break (smallest id among the equally near). Fixed against the
    pairwise definition in tests/unit/test_thesis_metrics_errors.py.
    """
    key = (view, gold_res)
    table = _nearest_cache.get(key)
    if table is not None:
        return table
    table = {g: (0, g) for g in gold_res}
    frontier = sorted(gold_res)
    dist = 0
    while frontier:
        dist += 1
        best: dict[str, str] = {}
        for node in frontier:
            src = table[node][1]
            for nb in view.neighbours(node):
                if nb in table:
                    continue
                if nb not in best or src < best[nb]:
                    best[nb] = src
        for nb, src in best.items():
            table[nb] = (dist, src)
        frontier = list(best)
    if len(_nearest_cache) >= _NEAREST_CACHE_SIZE:
        _nearest_cache.clear()
    _nearest_cache[key] = table
    return table


def nearest_gold_distance(pred: str, gold: Iterable[str], view: OntologyView) -> int | None:
    """:math:`d(\\hat{y}, Y_r) = \\min_{y \\in Y_r} \\mathrm{dist}_{\\mathcal{H}}(\\hat{y}, y)` (eq. 5)."""
    return nearest_gold(pred, gold, view)[1]


def near_miss_distribution(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """Distribution of eq. (5) over every false positive in a cohort.

    "This converts the qualitative observation that most false positives are ontological relatives
    into a measured quantity." A mass concentrated at distance 1–2 says the system is picking the
    wrong granularity. A flat tail says it is genuinely lost.

    Returns:
        ``{"n_fp", "n_measurable", "histogram", "mean", "median", "fraction_within_2",
        "n_unmeasurable"}``. ``n_unmeasurable`` counts false positives with no usable distance
        (hallucinations, out-of-subtree codes, reports with no annotated terms). They are excluded from
        the mean, not assigned an arbitrary large distance.
    """
    distances: list[int] = []
    n_fp = 0
    n_unmeasurable = 0

    for gold, pred in zip(gold_sets, pred_sets):
        gold_res = {view.resolve(g) for g in gold} - {None}
        for p in pred:
            if view.resolve(p) in gold_res:
                continue  # a true positive
            n_fp += 1
            d = nearest_gold_distance(p, gold_res, view)
            if d is None:
                n_unmeasurable += 1
            else:
                distances.append(d)

    if not distances:
        return {"n_fp": n_fp, "n_measurable": 0, "histogram": {}, "mean": float("nan"),
                "median": float("nan"), "fraction_within_2": float("nan"),
                "n_unmeasurable": n_unmeasurable}

    ordered = sorted(distances)
    n = len(ordered)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "n_fp": n_fp,
        "n_measurable": n,
        "histogram": dict(sorted(Counter(ordered).items())),
        "mean": sum(ordered) / n,
        "median": median,
        "fraction_within_2": sum(1 for d in ordered if d <= 2) / n,
        "n_unmeasurable": n_unmeasurable,
    }


# ── The Garcia et al. false-positive taxonomy ────────────────────────────────

def classify_false_positive(
    pred: str,
    gold: Iterable[str],
    view: OntologyView,
) -> tuple[str, str | None]:
    """Classify one false positive. Returns ``(bucket, reference_gold_term)``.

    See the module docstring for the bucket definitions and their precedence. The reference term is
    ``None`` for the three buckets that have no reference (``hallucination``, ``out_of_subtree``,
    ``no_gold``).
    """
    resolved = view.resolve(pred)
    if resolved is None:
        return ("out_of_subtree" if view.in_ontology(pred) else "hallucination"), None

    gold_res = {view.resolve(g) for g in gold} - {None}
    if not gold_res:
        return "no_gold", None

    reference, _ = nearest_gold(resolved, gold_res, view)
    if reference is None:
        return "unrelated", None

    if view.is_ancestor(resolved, reference):
        return "ancestor", reference
    if view.is_ancestor(reference, resolved):
        return "descendant", reference
    if view.direct_parents(resolved) & view.direct_parents(reference):
        return "sibling", reference
    return "unrelated", reference


def error_taxonomy(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """False-positive decomposition over a cohort, in the shape of RAG-HPO's published table.

    Returns:
        ``counts`` and ``fractions`` over all seven buckets, plus ``garcia_fractions``, the five
        comparable buckets renormalised over only those five, which is the column that lines up
        against the published decomposition when the hygiene buckets are non-empty.
    """
    counts: Counter = Counter()
    for gold, pred in zip(gold_sets, pred_sets):
        gold_res = {view.resolve(g) for g in gold} - {None}
        for p in pred:
            if view.resolve(p) in gold_res:
                continue
            bucket, _ = classify_false_positive(p, gold_res, view)
            counts[bucket] += 1

    n_fp = sum(counts.values())
    n_garcia = sum(counts[b] for b in GARCIA_BUCKETS)
    all_buckets = GARCIA_BUCKETS + HYGIENE_BUCKETS
    return {
        "n_fp": n_fp,
        "counts": {b: counts.get(b, 0) for b in all_buckets},
        "fractions": {b: (counts.get(b, 0) / n_fp if n_fp else 0.0) for b in all_buckets},
        "garcia_fractions": {
            b: (counts.get(b, 0) / n_garcia if n_garcia else 0.0) for b in GARCIA_BUCKETS
        },
        "n_garcia_classified": n_garcia,
    }


# ── The mirror image: what happened around each missed annotated term ─────────────

#: Buckets for :func:`false_negative_taxonomy`, in "closeness" order.
FN_BUCKETS = ("ancestor_predicted", "descendant_predicted", "sibling_predicted", "nothing_near")


def classify_false_negative(
    missed: str,
    pred: Iterable[str],
    view: OntologyView,
) -> str:
    """Classify one missed annotated term by what the system predicted *around* it.

    §Results asks for a false-negative analysis alongside the false-positive one. The question a
    recall failure raises is not "how far off was the prediction" but "did the system see this
    region of the ontology at all": a miss with an ancestor predicted is a granularity failure, a
    miss with nothing near it is a detection failure, and they call for different fixes.

    Precedence mirrors :func:`classify_false_positive`: ancestor, then descendant, then sibling.
    """
    resolved = view.resolve(missed)
    if resolved is None:
        return "nothing_near"
    pred_res = {view.resolve(p) for p in pred} - {None}
    if any(view.is_ancestor(p, resolved) for p in pred_res):
        return "ancestor_predicted"
    if any(view.is_ancestor(resolved, p) for p in pred_res):
        return "descendant_predicted"
    parents = view.direct_parents(resolved)
    if parents and any(parents & view.direct_parents(p) for p in pred_res):
        return "sibling_predicted"
    return "nothing_near"


def false_negative_taxonomy(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """Decomposition of every missed annotated term by what was predicted near it."""
    counts: Counter = Counter()
    for gold, pred in zip(gold_sets, pred_sets):
        pred_res = {view.resolve(p) for p in pred} - {None}
        for g in gold:
            resolved = view.resolve(g)
            if resolved is None or resolved in pred_res:
                continue
            counts[classify_false_negative(resolved, pred_res, view)] += 1

    n_fn = sum(counts.values())
    return {
        "n_fn": n_fn,
        "counts": {b: counts.get(b, 0) for b in FN_BUCKETS},
        "fractions": {b: (counts.get(b, 0) / n_fn if n_fn else 0.0) for b in FN_BUCKETS},
    }
