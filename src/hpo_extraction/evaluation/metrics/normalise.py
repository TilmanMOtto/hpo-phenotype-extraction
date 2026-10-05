r"""§Metrics, the pre-scoring normalisation every system's output passes through.

The thesis fixes one normalisation order and applies it identically to every system, so a
difference between two rows is a difference between the systems rather than between two ways of
reading their output files. Three steps, in this order:

1. **Obsolete identifiers are mapped to their replacement.** Handled by
   :meth:`~hpo_extraction.evaluation.metrics.ontology.OntologyView.resolve` through ``alt_id``.
2. **Valid identifiers outside the subontology below** :math:`v_0` **are discarded**, and their
   number is reported. These are real HPO terms, modes of inheritance, clinical modifiers,
   frequency terms, that name something outside the task's label space. Discarding them is a
   policy choice, not a fact, and it is the reason :func:`normalise_predictions` returns the count,
   not swallowing it: the count belongs in the results table beside the row it changed.
3. **Identifiers that do not exist in the release are kept, and count as false positives.** A
   system that invents ``HP:9999999`` has made an error that the reader must see. This is the one
   place this module departs from
   :meth:`~hpo_extraction.evaluation.metrics.ontology.OntologyView.resolve_set`, which drops them.

**There is no specificity reduction here.** An earlier draft of the thesis scored
:math:`\lfloor Y \rfloor`, the most specific elements of each set, on both the ground truth and the
predictions. That was dropped, for four reasons worth recording at the point someone would
reinstate it:

* The curated HCY ground truth **already carries a specificity policy, and it is not this one.** Curation
  removed a parent only where the more specific child was annotated *in the same segment*, and
  kept the 59 pairs whose child sits in a different segment. A document-scoped
  reduction would overrule those decisions.
* The problem a reduction addresses, a system penalised for emitting *Hypotonia* together with its
  ancestor *Abnormal muscle tone*, is what :mod:`~hpo_extraction.evaluation.metrics.hierarchy` already
  measures, both through :math:`hF` and, for the multi-label failure mode specifically, through
  CoPHE.
* Its effect is a property of the **ground truth's annotation convention**, not of the method under test.
  On identical documents, stripping ancestors is worth roughly +0.08 micro-:math:`F_1` to a
  mention-linking system and −0.015 to every traversal variant, so adopting it silently re-ranks
  method families.
* The prediction-side half of the idea is measured on its own terms by
  ``experiments/an earlier exploratory run``, at **sentence** scope, the same scope the curated
  ground truth uses, and reported as its own column, not folded into the primary metric.

Conventions a caller should know:

* A kept-but-nonexistent identifier is an FP under :mod:`~hpo_extraction.evaluation.metrics.flat`, but it
  is **invisible** to :mod:`~hpo_extraction.evaluation.metrics.hierarchy`, because it has no ancestors and
  therefore contributes nothing to :math:`\uparrow \hat{Y}`. That asymmetry is intended, a
  fabricated code cannot be near-missed, and ``n_nonexistent`` is returned so a hierarchy table
  can say how many predictions it could not place.
* Normalisation is **idempotent**: everything it returns is either already canonical or is a string
  the ontology does not know, and neither changes on a second pass.
"""

from __future__ import annotations

from typing import Iterable

from .ontology import OntologyView


def ancestor_closure(terms: Iterable[str], view: OntologyView) -> set[str]:
    r""":math:`\uparrow Y = \bigcup_{v \in Y} \mathrm{An}(v)`, the reflexive ancestor closure.

    A thin re-export of :meth:`~hpo_extraction.evaluation.metrics.ontology.OntologyView.ancestors_of_set`,
    so that :mod:`~hpo_extraction.evaluation.metrics.hierarchy` and the ensemble's closure-voting rule share
    one definition of the closure, not each rolling their own. The universal nodes are
    excluded, as they are everywhere in this package.
    """
    return view.ancestors_of_set(terms)


def split_unscorable(codes: Iterable[str], view: OntologyView) -> tuple[set[str], list[str], list[str]]:
    """``(scorable, out_of_subtree, nonexistent)`` for one set of raw identifiers.

    The three-way split every other function here is built from.
    :meth:`~hpo_extraction.evaluation.metrics.ontology.OntologyView.resolve_set` already separates the
    scorable codes from the rest. This adds the distinction the taxonomy needs, using
    :meth:`~hpo_extraction.evaluation.metrics.ontology.OntologyView.in_ontology`, a code the ontology knows
    but cannot score is *out of subtree*, one it does not know at all is *nonexistent*.
    """
    keep, dropped = view.resolve_set(codes)
    out_of_subtree = [c for c in dropped if view.in_ontology(c)]
    nonexistent = [c for c in dropped if not view.in_ontology(c)]
    return keep, out_of_subtree, nonexistent


def normalise_gold(codes: Iterable[str], view: OntologyView) -> tuple[set[str], int]:
    """``(scorable, n_out_of_subtree)`` for one report's ground-truth set.

    Unlike the prediction side, a nonexistent ground truth code is **also** dropped: it cannot be recovered
    by any system, so keeping it would charge every method a false negative for an annotation error.
    Both losses are counted together in the returned number only when they are the same thing. Use
    :func:`split_unscorable` directly when the two need to be reported apart.
    """
    keep, out_of_subtree, nonexistent = split_unscorable(codes, view)
    return keep, len(out_of_subtree) + len(nonexistent)


def normalise_predictions(
    codes: Iterable[str],
    view: OntologyView,
) -> tuple[set[str], int, int]:
    """``(scored, n_out_of_subtree, n_nonexistent)`` for one report's predicted set.

    ``scored`` is the scorable codes **union the nonexistent ones**, the latter carried through as
    the raw strings they arrived as. They can never intersect a normalised ground-truth set, a ground-truth set
    contains only scorable ids, so each one lands as one false positive, which is the
    intent.
    """
    keep, out_of_subtree, nonexistent = split_unscorable(codes, view)
    return keep | set(nonexistent), len(out_of_subtree), len(nonexistent)


def normalise_pair(
    gold: Iterable[str],
    pred: Iterable[str],
    view: OntologyView,
) -> tuple[set[str], set[str], dict[str, int]]:
    """One report's ``(gold, pred, counts)`` under the full pipeline.

    The form the scoring harness calls: it applies the same policy to both sides in one place, and
    hands back the bookkeeping as a dict whose keys are the column names the results tables carry.
    """
    gold_keep, gold_oos, gold_nonexistent = split_unscorable(gold, view)
    pred_keep, pred_oos, pred_nonexistent = split_unscorable(pred, view)
    counts = {
        "n_gold_out_of_subtree": len(gold_oos),
        "n_gold_nonexistent": len(gold_nonexistent),
        "n_pred_out_of_subtree": len(pred_oos),
        "n_pred_nonexistent": len(pred_nonexistent),
    }
    return gold_keep, pred_keep | set(pred_nonexistent), counts
