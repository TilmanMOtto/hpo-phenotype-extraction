"""§Core extraction quality, flat precision, recall and :math:`F_1` (skeleton eq. 1).

These are the metrics that "treat the label space as flat, ignoring the ontology structure
entirely", establishing comparability with prior work and guarding against the score inflation the
hierarchical measures are susceptible to.

**Granularity is document-level.** A term counts as a single prediction per report regardless of how
many textual mentions support it. The skeleton's mention-level axis is not implemented: no system
under comparison (TreePhenoRAG, RAG-HPO, flat top-M) emits mention spans, so it would be
unmeasurable for every method.

**Both averaging schemes, on both units.**

* **micro**, pool :math:`\\mathrm{TP}`, :math:`\\mathrm{FP}`, :math:`\\mathrm{FN}` over all reports
  before applying eq. (1). Weights reports in proportion to their phenotype count and is dominated
  by frequent terms.
* **macro over reports**, evaluate eq. (1) per report, take the unweighted mean. Weights every
  report equally.
* **macro over terms**, evaluate eq. (1) per HPO term (pooling over reports), take the unweighted
  mean. This is the averaging the skeleton describes as "sensitive to rare phenotypes, which
  constitute the majority of the HPO". Per-report macro cannot express that sensitivity.

Inputs everywhere are two aligned iterables of *sets of HPO id strings*, one entry per report:
``gold_sets[i]`` and ``pred_sets[i]`` describe the same report.

.. note::
   ``macro_f1`` here is the **mean of the per-unit** :math:`F_1`, following the skeleton's
   "evaluating per report and taking the unweighted mean". The repo's legacy
   ``hpo_extraction.evaluation.set_metrics.evaluate_micro_macro`` instead reports the harmonic mean of the averaged
   precision and recall, which is a different quantity. Both are returned, ``macro_f1`` and ``macro_f1_of_means``, so the divergence is visible rather than silent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal

from .ontology import OntologyView

EmptyConvention = Literal["perfect", "skip"]


def counts(gold: set[str], pred: set[str]) -> tuple[int, int, int]:
    """``(TP, FP, FN)`` for one report's document-level term sets."""
    tp = len(gold & pred)
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Skeleton eq. (1): :math:`P = TP/(TP+FP)`, :math:`R = TP/(TP+FN)`, :math:`F_1 = 2PR/(P+R)`.

    A zero denominator yields ``0.0``, not raising: no predictions means no precision to
    speak of, and the pooled caller wants a number it can put in a table.
    """
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def report_prf(
    gold: set[str],
    pred: set[str],
    empty_convention: EmptyConvention = "perfect",
) -> tuple[float, float, float] | None:
    """Eq. (1) for a single report, with the empty-set case made explicit.

    The skeleton is silent on what a report with no annotated terms *and* no predictions should score.
    Two defensible conventions, chosen by the caller, not assumed:

    * ``"perfect"``, ``(1.0, 1.0, 1.0)``. The system was right that there was nothing to find.
      Matches the repo's legacy ``hpo_extraction.evaluation.set_metrics.calc_metric``.
    * ``"skip"``, ``None``, excluding the report from the macro mean entirely, so a cohort with
      many empty reports cannot hand the macro average free credit.

    A report with annotated terms but no predictions (or vice versa) scores ``(0, 0, 0)`` under both.
    """
    if not gold and not pred:
        return (1.0, 1.0, 1.0) if empty_convention == "perfect" else None
    return prf(*counts(gold, pred))


def micro_prf(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
) -> tuple[float, float, float]:
    """Micro-averaged eq. (1): sum the three counts over all reports, then divide once."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_sets, pred_sets):
        a, b, c = counts(set(gold), set(pred))
        tp += a
        fp += b
        fn += c
    return prf(tp, fp, fn)


def macro_prf_by_report(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    empty_convention: EmptyConvention = "perfect",
) -> tuple[float, float, float, int]:
    """Macro over reports: mean of the per-report ``(P, R, F1)``.

    Returns ``(P, R, F1, n_units)``, ``n_units`` is the number of reports that actually entered
    the mean, which differs from the cohort size under ``empty_convention="skip"``.
    """
    sums = [0.0, 0.0, 0.0]
    n = 0
    for gold, pred in zip(gold_sets, pred_sets):
        scored = report_prf(set(gold), set(pred), empty_convention)
        if scored is None:
            continue
        n += 1
        for i, v in enumerate(scored):
            sums[i] += v
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    return sums[0] / n, sums[1] / n, sums[2] / n, n


def macro_prf_by_term(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
) -> tuple[float, float, float, int]:
    """Macro over terms: pool each HPO term's counts across reports, then mean over terms.

    The unit set is every term appearing in at least one ground-truth set or one prediction, a term that
    is neither predicted nor ground truth anywhere has no counts and cannot be scored. Because rare terms
    dominate that set, this is the average that is "sensitive to rare phenotypes".

    Returns ``(P, R, F1, n_terms)``.
    """
    per_term: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for gold, pred in zip(gold_sets, pred_sets):
        gold, pred = set(gold), set(pred)
        for t in gold & pred:
            per_term[t][0] += 1
        for t in pred - gold:
            per_term[t][1] += 1
        for t in gold - pred:
            per_term[t][2] += 1

    if not per_term:
        return 0.0, 0.0, 0.0, 0
    sums = [0.0, 0.0, 0.0]
    for tp, fp, fn in per_term.values():
        for i, v in enumerate(prf(tp, fp, fn)):
            sums[i] += v
    n = len(per_term)
    return sums[0] / n, sums[1] / n, sums[2] / n, n


def macro_prf_by_term_domains(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
) -> dict:
    r"""Label-macro with each average taken over the term set that average is defined on.

    :func:`macro_prf_by_term` averages precision, recall and :math:`F_1` over one shared unit set, every term that is ground truth or predicted anywhere. That is defensible but it mixes two different
    quantities into each mean: a term that is ground truth in some report and never predicted contributes
    ``P = 0`` to the precision average, even though the system never made a precision claim about
    it, and symmetrically for recall.

    The thesis therefore reports each average over its own domain:

    * **precision** over terms predicted at least once, :math:`\mathrm{TP}_v + \mathrm{FP}_v > 0`;
    * **recall** over terms ground truth in at least one report, :math:`\mathrm{TP}_v + \mathrm{FN}_v > 0`;
    * :math:`F_1` over the **union** of the two, which is :func:`macro_prf_by_term`'s unit set.

    The three denominators differ, and they differ by a lot, a method that predicts 600 distinct
    terms against a 435-term ground truth averages its precision over a set half again the size of the one
    its recall is averaged over. Printing them is the point: the counts are returned alongside the
    values so a table can say what each mean was taken over.

    Returns:
        ``precision``/``recall``/``f1`` and ``n_precision_terms``/``n_recall_terms``/``n_f1_terms``.
    """
    per_term: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for gold, pred in zip(gold_sets, pred_sets):
        gold, pred = set(gold), set(pred)
        for t in gold & pred:
            per_term[t][0] += 1
        for t in pred - gold:
            per_term[t][1] += 1
        for t in gold - pred:
            per_term[t][2] += 1

    p_sum = r_sum = f_sum = 0.0
    n_p = n_r = n_f = 0
    for tp, fp, fn in per_term.values():
        p, r, f = prf(tp, fp, fn)
        if tp + fp:
            p_sum += p
            n_p += 1
        if tp + fn:
            r_sum += r
            n_r += 1
        n_f += 1
        f_sum += f

    return {
        "precision": p_sum / n_p if n_p else 0.0,
        "recall": r_sum / n_r if n_r else 0.0,
        "f1": f_sum / n_f if n_f else 0.0,
        "n_precision_terms": n_p,
        "n_recall_terms": n_r,
        "n_f1_terms": n_f,
    }


def flat_report(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView | None = None,
    empty_convention: EmptyConvention = "perfect",
    label_macro_domains: bool = False,
) -> dict:
    """Every §Core extraction-quality number for one cohort, in one dict.

    Args:
        gold_sets, pred_sets: aligned per-report iterables of HPO id strings.
        view: if given, both sides are canonicalised through
            :meth:`~hpo_extraction.evaluation.metrics.ontology.OntologyView.resolve_set` first and the
            dropped codes are reported. If ``None``, the raw strings are compared as-is, which is
            what you want when checking a number against a system that was scored without
            canonicalisation.
        empty_convention: see :func:`report_prf`.

    Returns:
        ``micro_*``, ``macro_*`` (by report), ``macro_term_*``, the pooled counts, and, when a
        ``view`` was supplied, ``n_gold_dropped`` / ``n_pred_dropped`` so unresolvable codes are
        accounted for, not silently scored as misses.
    """
    gold_list = [set(g) for g in gold_sets]
    pred_list = [set(p) for p in pred_sets]
    if len(gold_list) != len(pred_list):
        raise ValueError(
            f"gold_sets and pred_sets must be aligned: {len(gold_list)} vs {len(pred_list)}"
        )

    dropped = {}
    if view is not None:
        n_gold_dropped = n_pred_dropped = 0
        resolved_gold, resolved_pred = [], []
        for g, p in zip(gold_list, pred_list):
            gs, gd = view.resolve_set(g)
            ps, pd_ = view.resolve_set(p)
            resolved_gold.append(gs)
            resolved_pred.append(ps)
            n_gold_dropped += len(gd)
            n_pred_dropped += len(pd_)
        gold_list, pred_list = resolved_gold, resolved_pred
        dropped = {"n_gold_dropped": n_gold_dropped, "n_pred_dropped": n_pred_dropped}

    tp = fp = fn = 0
    for g, p in zip(gold_list, pred_list):
        a, b, c = counts(g, p)
        tp += a
        fp += b
        fn += c
    mi_p, mi_r, mi_f = prf(tp, fp, fn)
    ma_p, ma_r, ma_f, n_reports = macro_prf_by_report(gold_list, pred_list, empty_convention)
    mt_p, mt_r, mt_f, n_terms = macro_prf_by_term(gold_list, pred_list)

    domains = {}
    if label_macro_domains:
        d = macro_prf_by_term_domains(gold_list, pred_list)
        domains = {
            "label_macro_precision": d["precision"],
            "label_macro_recall": d["recall"],
            "label_macro_f1": d["f1"],
            "label_macro_n_precision_terms": d["n_precision_terms"],
            "label_macro_n_recall_terms": d["n_recall_terms"],
            "label_macro_n_f1_terms": d["n_f1_terms"],
        }

    return {
        "n_reports": len(gold_list),
        "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": mi_p, "micro_recall": mi_r, "micro_f1": mi_f,
        "macro_precision": ma_p, "macro_recall": ma_r, "macro_f1": ma_f,
        # The legacy repo convention, kept alongside so the two definitions never get confused.
        "macro_f1_of_means": (
            2 * ma_p * ma_r / (ma_p + ma_r) if (ma_p + ma_r) else 0.0
        ),
        "macro_n_units": n_reports,
        "macro_term_precision": mt_p, "macro_term_recall": mt_r, "macro_term_f1": mt_f,
        "macro_term_n_units": n_terms,
        **domains,
        **dropped,
    }
