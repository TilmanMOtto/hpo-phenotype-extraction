r"""The thesis's own false-positive taxonomy, existential precedence, and a ``same_branch`` bucket.

:mod:`~hpo_extraction.evaluation.metrics.errors` classifies each false positive against the **nearest** ground truth
term in the report. This module classifies it against **any** annotated term: a prediction is an
*ancestor* error if it is an ancestor of *some* annotated term, a *descendant* error if it is a
descendant of *some* annotated term, and so on, with the checks applied in a fixed order and the first
match winning.

Both modules exist because the two answer different questions, and mixing them silently would be
the worst outcome:

* the **nearest-ground truth** rule asks *"what is the closest thing this prediction could have been aiming
  at, and how did it miss?"*, the right question for the severity distribution, which is why
  ``nearest_gold_distance`` stays the severity measure for both taxonomies;
* the **existential** rule asks *"is this prediction ontologically related to anything the report is
  annotated with?"*, the right question for a decomposition, because a prediction that is an exact
  ancestor of one annotated term and an unrelated distant cousin of the nearest one is, in any useful
  sense, an over-general prediction rather than an unrelated one.

The difference is not cosmetic. A traversal that accepts whole ancestor chains produces predictions
that *are* ancestors of some annotated term while frequently being nearest to a different one. The
nearest-ground truth rule files a share of those as ``unrelated``, which understates how close the method's
errors actually are.

**The bucket order**, first match wins:

===============  ===========================================================================
``no_gold``       the report has no annotated terms, so there is no relation to be had.
``invalid``       the predicted code names no term in the release. The thesis's name for what
                  :mod:`~hpo_extraction.evaluation.metrics.errors` calls ``hallucination``.
``ancestor``      a strict ancestor of some annotated term, the system was too general.
``descendant``    a strict descendant of some annotated term, too specific.
``sibling``       shares a direct ``is_a`` parent with some annotated term.
``same_branch``   shares **any** depth-one ancestor (an organ-system root) with some annotated term,
                  without being a relative of any of them in the three stronger senses. This is the
                  bucket :mod:`~hpo_extraction.evaluation.metrics.errors` does not have: "wrong finding,
                  right organ system" is a materially different error from "wrong organ system",
                  and on a cohort whose precision problem concentrates in two branches out of 23
                  the distinction carries real weight. Note *any*: a DAG term can hang off several
                  organ systems, and comparing one canonical pick per term would file a genuinely
                  shared branch as ``unrelated`` whenever the two picks differed.
``unrelated``     everything else.
===============  ===========================================================================

**Out-of-subtree codes never reach this function.** Under the thesis's normalisation
(:mod:`~hpo_extraction.evaluation.metrics.normalise`) they are discarded before scoring and counted in the
``n_pred_out_of_subtree`` column, so the ``out_of_subtree`` bucket of the other taxonomy has no
counterpart here. If one is passed anyway it is classified ``invalid``, since by that point it is
not a scorable term, but the count, not the bucket, is where such codes are meant to be read.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .errors import nearest_gold_distance
from .ontology import OntologyView

#: Buckets of the thesis taxonomy, in "closeness" order. ``same_branch`` is the one the
#: nearest-ground truth taxonomy in :mod:`~hpo_extraction.evaluation.metrics.errors` does not carry.
THESIS_BUCKETS = (
    "ancestor", "descendant", "sibling", "same_branch", "unrelated", "invalid", "no_gold",
)


def classify_false_positive_existential(
    pred: str,
    gold: Iterable[str],
    view: OntologyView,
) -> tuple[str, str | None]:
    """Classify one false positive against *any* annotated term. Returns ``(bucket, witness)``.

    The witness is the annotated term that earned the bucket, the first one found, in sorted order, so
    the choice is deterministic, or ``None`` for ``no_gold`` and ``invalid``. Returning it keeps
    an individual decision inspectable, which is what makes the table auditable, not merely
    reproducible.
    """
    resolved = view.resolve(pred)
    if resolved is None:
        return "invalid", None

    gold_res = sorted({g for g in (view.resolve(g) for g in gold) if g is not None})
    if not gold_res:
        return "no_gold", None

    # Strict relations: An(v) and De(v) are reflexive here, so the identity case is excluded
    # explicitly. A prediction equal to an annotated term is a true positive and should never arrive.
    for y in gold_res:
        if resolved != y and view.is_ancestor(resolved, y):
            return "ancestor", y
    for y in gold_res:
        if resolved != y and view.is_ancestor(y, resolved):
            return "descendant", y

    pred_parents = view.direct_parents(resolved)
    for y in gold_res:
        if pred_parents & view.direct_parents(y):
            return "sibling", y

    # Every depth-one ancestor, not the canonical one. ``view.layer1`` picks the
    # lexicographically smallest so that grouping is a partition, which is right for grouping and
    # wrong here: the thesis asks whether the two "share an ancestor of depth one", and a DAG term
    # routinely sits under several organ systems. Comparing canonical picks files a genuinely
    # shared branch as ``unrelated`` whenever the two terms' picks happen to differ.
    pred_layer1 = view.layer1_ancestors(resolved)
    if pred_layer1:
        for y in gold_res:
            if pred_layer1 & view.layer1_ancestors(y):
                return "same_branch", y

    return "unrelated", None


def error_taxonomy_existential(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """The cohort-level decomposition, with the severity distribution beside it.

    Returns ``counts`` and ``fractions`` over :data:`THESIS_BUCKETS`, plus ``distance_histogram``
    and ``distance_mean`` over the false positives whose distance to the nearest annotated term is
    measurable, the severity half of the same table, computed here so a caller cannot pair a
    decomposition from one rule with a severity distribution from another.
    """
    counts: Counter = Counter()
    distances: list[int] = []
    n_unmeasurable = 0

    for gold, pred in zip(gold_sets, pred_sets):
        gold_res = {g for g in (view.resolve(g) for g in gold) if g is not None}
        for p in pred:
            if view.resolve(p) in gold_res:
                continue  # a true positive
            bucket, _ = classify_false_positive_existential(p, gold_res, view)
            counts[bucket] += 1
            d = nearest_gold_distance(p, gold_res, view)
            if d is None:
                n_unmeasurable += 1
            else:
                distances.append(d)

    n_fp = sum(counts.values())
    return {
        "n_fp": n_fp,
        "counts": {b: counts.get(b, 0) for b in THESIS_BUCKETS},
        "fractions": {
            b: (counts.get(b, 0) / n_fp if n_fp else 0.0) for b in THESIS_BUCKETS
        },
        "distance_histogram": dict(sorted(Counter(distances).items())),
        "distance_mean": (sum(distances) / len(distances)) if distances else float("nan"),
        "n_distance_measurable": len(distances),
        "n_distance_unmeasurable": n_unmeasurable,
    }
