"""§Hierarchy-aware quality, ancestor-closure :math:`hP/hR/hF` (eq. 3) and CoPHE (eq. 4).

"Flat metrics assign zero credit to a prediction of *Abnormality of the nervous system* when the
annotated term is *Seizure*, despite the two being ontologically adjacent." Under a
non-mandatory-leaf-node prediction regime such near-misses are the dominant error mode.

Two measures, with different failure modes, are implemented so they can check each other:

**Ancestor closure** (Kiritchenko et al.) augments both sets with all ancestors and applies eq. (1).
Simple, comparable to the wider literature, and systematically optimistic, the closure introduces
true positives at shallow depths that any reasonable system recovers. It must always be read next to
flat :math:`F_1`, never alone.

**CoPHE** (Falis et al.) fixes a defect specific to the multi-label setting: if several descendants
of a common ancestor are predicted but fewer are ground truth, both sets collapse to the same ancestor under
set closure and the over-prediction *disappears from the count*. Since TreePhenoRAG may accept
several siblings inside one surviving subtree, that failure mode is directly relevant here. CoPHE
propagates counts rather than set membership, so the surplus survives as :math:`\\mathrm{FP}_v`.

CoPHE-:math:`F` is the primary hierarchy-aware figure; :math:`hF` is reported for comparability.

**The closure convention**, :math:`\\mathrm{An}(v)` includes :math:`v` and excludes ``HP:0000001``
and ``HP:0000118``, lives in :mod:`~hpo_extraction.evaluation.metrics.ontology` and is applied
consistently to both measures. See that module for why the exclusion is essential.

**The node set for eq. (4).** The equation says "pooling over :math:`v`" without naming the range.
This package pools over :math:`\\mathrm{An}(\\hat{Y}_r) \\cup \\mathrm{An}(Y_r)`. Every node outside
that union has :math:`x_v = y_v = 0` and therefore contributes ``0`` to all three counts, so the
result is *numerically identical* to ranging over all ~19k ontology nodes while being a few dozen
iterations instead of ~19k per report.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .flat import prf
from .ontology import OntologyView, ancestor_count_distribution


# ── Ancestor-closure hP / hR / hF (eq. 3) ────────────────────────────────────

def h_counts(gold: Iterable[str], pred: Iterable[str], view: OntologyView) -> tuple[int, int, int]:
    """``(|An(Ŷ) ∩ An(Y)|, |An(Ŷ)|, |An(Y)|)``, the three quantities eq. (3) is built from."""
    an_pred = view.ancestors_of_set(pred)
    an_gold = view.ancestors_of_set(gold)
    return len(an_pred & an_gold), len(an_pred), len(an_gold)


def h_prf(gold: Iterable[str], pred: Iterable[str], view: OntologyView) -> tuple[float, float, float]:
    """Skeleton eq. (3) for one report: :math:`hP`, :math:`hR`, :math:`hF`.

    A prediction that is an ancestor of an annotated term receives partial, not zero credit, in
    proportion to the shared path length. Empty closures yield ``0.0`` in the affected component,
    not raising.
    """
    overlap, n_pred, n_gold = h_counts(gold, pred, view)
    hp = overlap / n_pred if n_pred else 0.0
    hr = overlap / n_gold if n_gold else 0.0
    hf = 2 * hp * hr / (hp + hr) if (hp + hr) else 0.0
    return hp, hr, hf


def h_prf_cohort(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """:math:`hP/hR/hF` for a cohort, micro (pooled closures) and macro (mean over reports).

    Micro pools the closure intersections and closure sizes across reports before dividing. Macro
    averages the per-report triples. As with flat :math:`F_1`, ``macro_hf`` is the mean of the
    per-report :math:`hF`, not the harmonic mean of the averaged :math:`hP` and :math:`hR`.
    """
    overlap_total = pred_total = gold_total = 0
    sums = [0.0, 0.0, 0.0]
    n = 0
    for gold, pred in zip(gold_sets, pred_sets):
        o, np_, ng = h_counts(gold, pred, view)
        overlap_total += o
        pred_total += np_
        gold_total += ng
        for i, v in enumerate(h_prf(gold, pred, view)):
            sums[i] += v
        n += 1

    micro_p = overlap_total / pred_total if pred_total else 0.0
    micro_r = overlap_total / gold_total if gold_total else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    denom = n or 1
    return {
        "n_reports": n,
        "micro_hp": micro_p, "micro_hr": micro_r, "micro_hf": micro_f,
        "macro_hp": sums[0] / denom, "macro_hr": sums[1] / denom, "macro_hf": sums[2] / denom,
        "closure_overlap": overlap_total,
        "closure_pred_total": pred_total,
        "closure_gold_total": gold_total,
    }


# ── CoPHE: count-preserving hierarchical evaluation (eq. 4) ──────────────────

def subtree_counts(terms: Iterable[str], view: OntologyView) -> Counter:
    """``{v: number of terms in the subtree rooted at v}``, the :math:`x_v` / :math:`y_v` of eq. (4).

    A term is counted at every one of its ancestors *and at itself*, since
    :math:`\\mathrm{An}(v) \\ni v`. In a DAG a term with several parents is counted once at each
    ancestor it reaches by any path, which is the same multi-parent weighting the closure metrics
    carry and which :func:`ancestor_count_distribution` quantifies.
    """
    counts: Counter = Counter()
    for t in terms:
        for a in view.ancestors(t):
            counts[a] += 1
    return counts


def cophe_counts(
    gold: Iterable[str],
    pred: Iterable[str],
    view: OntologyView,
) -> tuple[int, int, int]:
    """Skeleton eq. (4) for one report: pooled ``(TP, FP, FN)`` over the closure union.

    For each node :math:`v`, with :math:`x_v` predicted and :math:`y_v` annotated terms in its subtree:

    .. code-block:: text

        TP_v = min(x_v, y_v)
        FP_v = max(x_v - y_v, 0)
        FN_v = max(y_v - x_v, 0)

    Over-prediction inside a surviving subtree survives as :math:`\\mathrm{FP}_v` at the shared
    ancestor instead of being absorbed by set collapse, the whole point of the measure.
    """
    x = subtree_counts(pred, view)
    y = subtree_counts(gold, view)
    tp = fp = fn = 0
    for v in set(x) | set(y):
        xv, yv = x.get(v, 0), y.get(v, 0)
        tp += min(xv, yv)
        fp += max(xv - yv, 0)
        fn += max(yv - xv, 0)
    return tp, fp, fn


def cophe_prf(
    gold: Iterable[str],
    pred: Iterable[str],
    view: OntologyView,
) -> tuple[float, float, float]:
    """CoPHE precision, recall and :math:`F_1` for one report ("by the usual definitions")."""
    return prf(*cophe_counts(gold, pred, view))


def cophe_cohort(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """CoPHE for a cohort, micro (pooled counts) and macro (mean over reports)."""
    tp = fp = fn = 0
    sums = [0.0, 0.0, 0.0]
    n = 0
    for gold, pred in zip(gold_sets, pred_sets):
        a, b, c = cophe_counts(gold, pred, view)
        tp += a
        fp += b
        fn += c
        for i, v in enumerate(prf(a, b, c)):
            sums[i] += v
        n += 1

    mi_p, mi_r, mi_f = prf(tp, fp, fn)
    denom = n or 1
    return {
        "n_reports": n,
        "cophe_tp": tp, "cophe_fp": fp, "cophe_fn": fn,
        "micro_cophe_precision": mi_p, "micro_cophe_recall": mi_r, "micro_cophe_f1": mi_f,
        "macro_cophe_precision": sums[0] / denom,
        "macro_cophe_recall": sums[1] / denom,
        "macro_cophe_f1": sums[2] / denom,
    }


def hierarchy_report(
    gold_sets: Iterable[Iterable[str]],
    pred_sets: Iterable[Iterable[str]],
    view: OntologyView,
) -> dict:
    """Every §Hierarchy-aware number in one dict: CoPHE, :math:`hF`, and the closure audit.

    The :math:`|\\mathrm{An}(v)|` distribution is included because the skeleton commits to
    publishing it whenever :math:`hF` is published, it is the only way a reader can tell how much
    of the closure credit came from multi-parent weighting.
    """
    gold_list = [set(g) for g in gold_sets]
    pred_list = [set(p) for p in pred_sets]
    return {
        **cophe_cohort(gold_list, pred_list, view),
        **h_prf_cohort(gold_list, pred_list, view),
        "gold_ancestor_counts": ancestor_count_distribution(gold_list, view),
    }
