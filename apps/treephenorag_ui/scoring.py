"""Flat and hierarchical scoring, thin wrappers, so the UI and the thesis report one number.

Every metric here is computed by ``hpo_extraction.evaluation.metrics``. This module only assembles the
aligned ``(gold_sets, pred_sets)`` lists those functions take and flattens their output into the
JSON-serialisable shape the views read. Nothing is reimplemented: ``result_tables``
produces the tables in the thesis from the same functions, so a disagreement between the UI and
the thesis is a discovery/loading bug, and ``selfcheck`` is able to say so.

**Cohort membership comes from the summary lines.** A report the driver processed but predicted
nothing for is a real data point, at the higher τ_prune values on the tree runs there are many, and dropping it inflates macro recall. Reports with an empty *ground truth* set are a different matter:
recall is undefined for them and they are excluded from the recall-bearing macro averages by
``thesis_metrics.flat`` itself, which is why that decision is not repeated here.
"""

from __future__ import annotations

from typing import Iterable, Mapping


def align(
    gold: Mapping[str, Iterable[str]],
    predicted: Mapping[str, Iterable[str]],
    view=None,
    *,
    report_ids: Iterable[str] | None = None,
    canonicalise: bool = False,
) -> tuple[list[set[str]], list[set[str]], list[str]]:
    """``(gold_sets, pred_sets, report_ids)``, positionally aligned.

    **Codes are compared as raw strings by default**, which is what
    ``result_tables/loaders.aligned_sets`` does: it takes the intersection of the two dicts and hands
    the sets straight to ``flat_report`` with no ``view``. Resolving here instead would remap alt
    ids and silently drop out-of-subtree codes, which nudges precision up by a fraction of a
    percent, enough that every number in this UI would disagree with the thesis tables, for a
    reason no reader could see. ``canonicalise=True`` opts into resolution. The selfcheck reports
    how much it would move.

    Reports are the intersection of the two mappings unless ``report_ids`` names them explicitly.
    A report the run processed but that has no ground truth entry cannot be scored and is left out, not
    counted as an all-false-positive report.
    """
    if report_ids is None:
        ids = sorted(set(gold) & set(predicted))
    else:
        ids = [r for r in report_ids if r in gold and r in predicted]

    if not canonicalise:
        return [set(gold[r]) for r in ids], [set(predicted[r]) for r in ids], ids

    if view is None:
        raise ValueError("canonicalise=True needs an OntologyView")
    gold_sets, pred_sets = [], []
    for report_id in ids:
        gold_sets.append({view.resolve(g) for g in gold[report_id]} - {None})
        pred_sets.append({view.resolve(p) for p in predicted[report_id]} - {None})
    return gold_sets, pred_sets, ids


def unresolvable_counts(gold, predicted, view) -> dict:
    """How many codes canonicalisation would drop or remap, the cost of the default above.

    Shown by the selfcheck so the raw-string choice is auditable rather than merely stated. A
    large number here would mean the artifacts carry alt ids or out-of-subtree codes in bulk, and
    that the thesis tables are undercounting them too.
    """
    gold_sets, pred_sets, _ = align(gold, predicted)
    def dropped(sets):
        return sum(1 for s in sets for code in s if view.resolve(code) is None)
    def remapped(sets):
        return sum(1 for s in sets for code in s
                   if (r := view.resolve(code)) is not None and r != code)
    return {
        "gold_unresolvable": dropped(gold_sets), "gold_remapped": remapped(gold_sets),
        "pred_unresolvable": dropped(pred_sets), "pred_remapped": remapped(pred_sets),
    }


#: How a report with neither annotated terms nor predictions scores. ``result_tables`` fixes this to
#: ``"perfect"`` (``run.py:291``, ``sections.py:161``) to match the repo's legacy
#: ``hpo_extraction.evaluation.set_metrics.calc_metric``. The UI must use the same convention or its macro averages
#: will not equal the thesis's, and the selfcheck's cross-check would fail for a reason that has
#: nothing to do with the run.
EMPTY_CONVENTION = "perfect"


def flat_metrics(gold_sets, pred_sets) -> dict:
    """Micro/macro precision, recall, F1 and the pooled TP/FP/FN counts, plus per-report P/R/F1.

    ``flat_report`` is called without a ``view``: :func:`align` has already canonicalised both
    sides, and resolving twice would report the dropped-code counts against an already-clean
    input.
    """
    from hpo_extraction.evaluation.metrics import flat

    report = dict(flat.flat_report(gold_sets, pred_sets, empty_convention=EMPTY_CONVENTION))
    report["per_report"] = [
        flat.report_prf(gold, pred, EMPTY_CONVENTION)
        for gold, pred in zip(gold_sets, pred_sets)
    ]
    return report


def hierarchy_metrics(gold_sets, pred_sets, view) -> dict:
    """Ancestor-closure hP/hR/hF and the CoPHE count-preserving variant.

    Flat F1 charges a predicted parent of the right term the same as a predicted term from the
    wrong organ system. On a traversal method that reports internal nodes by design, that single
    number hides most of what the method did, so the scorecard always shows both.
    """
    from hpo_extraction.evaluation.metrics import hierarchy

    report = dict(hierarchy.hierarchy_report(gold_sets, pred_sets, view))
    # The |An(v)| distribution is a nested dict of numpy-ish scalars. The views do not read it and
    # it is the one key that would not survive the JSON cache round-trip unchanged.
    report.pop("gold_ancestor_counts", None)
    return report


def score(gold, predicted, view, *, report_ids=None) -> dict:
    """Everything the scorecard needs from one configuration's predictions."""
    gold_sets, pred_sets, ids = align(gold, predicted, view, report_ids=report_ids)
    if not ids:
        return {"n_reports": 0, "report_ids": [], "flat": {}, "hierarchy": {}}
    return {
        "n_reports": len(ids),
        "report_ids": ids,
        "flat": flat_metrics(gold_sets, pred_sets),
        "hierarchy": hierarchy_metrics(gold_sets, pred_sets, view),
    }


def assert_matches_thesis_metrics(gold, predicted, view) -> None:
    """Fix :func:`score` to ``thesis_metrics.flat.flat_report`` called directly.

    The wrapper's only job is alignment and renaming, and both are the kind of thing that breaks
    without failing: a mis-keyed rename yields ``None`` in a stat tile, an alignment bug yields a
    plausible-looking number computed over the wrong reports. Recomputing the micro triple from
    the raw sets and comparing catches both.
    """
    from hpo_extraction.evaluation.metrics import flat

    gold_sets, pred_sets, _ = align(gold, predicted, view)
    expected = flat.flat_report(gold_sets, pred_sets, empty_convention=EMPTY_CONVENTION)
    actual = score(gold, predicted, view)["flat"]
    for key in ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "tp", "fp", "fn"):
        theirs, mine = expected[key], actual[key]
        assert mine == theirs or abs(mine - theirs) < 1e-12, \
            f"scoring.score drifted from flat_report on {key}: {mine} != {theirs}"
