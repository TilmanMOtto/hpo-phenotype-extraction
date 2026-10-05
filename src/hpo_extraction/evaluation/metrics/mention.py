"""Mention-level metrics, the axis the skeleton declares and nothing could measure until now.

``thesis_metrics/__init__.py`` states it plainly: *"the skeleton's mention-level granularity axis is
not implemented. No system under comparison emits mention spans, so it would be unmeasurable for
every method. All extraction quality here is document-level."* earlier is the first system in this
project that emits them, so this module opens that axis.

Why it counts beyond completeness: **pooled F1 treats a fabricated identifier, a wrongly-negated
finding, and an over-specific but related term as equivalent errors.** They are not. The first is a
fabrication, the second inverts the patient record, and the third is a minor loss of precision. This
module reports them separately, and every function here is designed to be read next to the others
rather than averaged with them.

The repository has already demonstrated the general form of that failure at document level:
``the earlier runs`` re-scored a model whose verdicts were **26.9 % wrong** and micro-F1 moved by 0.006, because
precision tripling and recall halving cancel in the harmonic mean. Only macro-F1 registered it. A
single main number cannot detect a broken stage, and this module is built on that assumption.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------
# Span matching
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldSpan:
    """One human-marked mention."""

    report_id: str
    start: int
    end: int
    hpo_id: str = ""
    text: str = ""


@dataclass(frozen=True)
class PredSpan:
    """One span the system produced."""

    report_id: str
    start: int
    end: int
    hpo_id: str = ""
    text: str = ""


def overlaps(a, b) -> bool:
    """Any shared character. The loose criterion."""
    return a.report_id == b.report_id and a.start < b.end and b.start < a.end


def exact(a, b) -> bool:
    """Identical offsets. The strict criterion."""
    return a.report_id == b.report_id and a.start == b.start and a.end == b.end


def jaccard(a, b) -> float:
    """Character-level overlap, for reporting how *close* a near-miss was."""
    if a.report_id != b.report_id:
        return 0.0
    lo, hi = max(a.start, b.start), min(a.end, b.end)
    inter = max(0, hi - lo)
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union else 0.0


def match_spans(gold: list, predicted: list, *, criterion=overlaps) -> tuple[list, list, list]:
    """Greedy one-to-one matching. Returns ``(pairs, unmatched_gold, unmatched_pred)``.

    One-to-one counts. Without it, one enormous predicted span covering a whole sentence would
    "find" every ground truth mention in it, and Stage A, which is over-generative by design, would
    score a recall of 1.0 by emitting the sentence itself. Greedy assignment by descending overlap
    keeps a span accountable for one mention.
    """
    remaining = list(predicted)
    pairs: list[tuple] = []
    unmatched_gold: list = []

    for g in gold:
        candidates = [p for p in remaining if criterion(g, p)]
        if not candidates:
            unmatched_gold.append(g)
            continue
        best = max(candidates, key=lambda p: jaccard(g, p))
        pairs.append((g, best))
        remaining.remove(best)

    return pairs, unmatched_gold, remaining


def span_prf(gold: list, predicted: list, *, criterion=overlaps) -> dict:
    """Precision, recall and F1 over mentions.

    **Precision here is not a quality target for Stage A.** The specification asks for recall above
    0.97 at a precision of 0.5-0.7 and says precision is not optimised at this stage: a span that is
    marked and later discarded costs one Stage D forward pass, while a span never marked can never
    be recovered by anything downstream. Report both. Hold only recall to a threshold.
    """
    pairs, missed, spurious = match_spans(gold, predicted, criterion=criterion)
    tp, fn, fp = len(pairs), len(missed), len(spurious)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "criterion": criterion.__name__,
        "mean_jaccard": (sum(jaccard(g, p) for g, p in pairs) / len(pairs)) if pairs else 0.0,
    }


def span_recall(gold: list, predicted: list, *, criterion=overlaps) -> float:
    """Stage A's calibration target: above 0.97 against a human-marked span set (§6.2)."""
    return span_prf(gold, predicted, criterion=criterion)["recall"]


# ---------------------------------------------------------------------------------------------
# Type and polarity
# ---------------------------------------------------------------------------------------------


def type_confusion(pairs: list[tuple[str, str]], labels: list[str]) -> dict:
    """Full matrix over the span-type vocabulary.

    Reported as a matrix, not an accuracy because the specification names the cells that
    matter: negated/resolved and finding/disease. An aggregate accuracy of 0.9 is compatible with
    every resolved finding being recorded as negated, which inverts those patients' records while
    barely moving the number.
    """
    counts = Counter(pairs)
    matrix = {t: {p: counts.get((t, p), 0) for p in labels} for t in labels}
    total = sum(counts.values())
    correct = sum(counts.get((label, label), 0) for label in labels)
    return {
        "matrix": matrix,
        "labels": list(labels),
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "critical_cells": {
            "negated_as_resolved": counts.get(("NEGATED_FINDING", "RESOLVED_FINDING"), 0),
            "resolved_as_negated": counts.get(("RESOLVED_FINDING", "NEGATED_FINDING"), 0),
            "finding_as_disease": sum(
                n for (t, p), n in counts.items()
                if t.endswith("_FINDING") and p.startswith("DISEASE_")
            ),
            "disease_as_finding": sum(
                n for (t, p), n in counts.items()
                if t.startswith("DISEASE_") and p.endswith("_FINDING")
            ),
        },
    }


def polarity_accuracy(pairs: list[tuple[str, str, str]]) -> dict:
    """Assertion-status accuracy **conditioned on the term being right** (§6.2).

    Conditioning is the whole point. Unconditioned, a wrong status on a wrong term is
    indistinguishable from a wrong status on a right one, and only the second is the failure the
    specification is about, the right term with the wrong assertion status, which is the error
    class that silently inverts a patient record.

    ``pairs`` are ``(term_correct, gold_status, predicted_status)``.
    """
    eligible = [(g, p) for correct, g, p in pairs if correct]
    if not eligible:
        return {"n": 0, "accuracy": 0.0, "by_status": {}}
    by_status: dict[str, dict[str, int]] = {}
    for gold_status, pred_status in eligible:
        bucket = by_status.setdefault(gold_status, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(gold_status == pred_status)
    return {
        "n": len(eligible),
        "accuracy": sum(b["correct"] for b in by_status.values()) / len(eligible),
        "by_status": {
            status: {**b, "accuracy": b["correct"] / b["n"]} for status, b in by_status.items()
        },
    }


# ---------------------------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------------------------


def specificity_error(gold_id: str, pred_id: str, view) -> int | None:
    """Signed ontology depth error when the prediction is in the right subtree.

    Positive means **over-specific** (the prediction is a descendant of ground truth, the system claimed
    detail the text did not give). Negative means **under-specific** (an ancestor was chosen where
    the text supported more). ``None`` means the two are not on one path, which is a different kind
    of error and must not be averaged in as a zero.

    The sign carries the clinical reading. Over-specificity asserts something unstated. Under-
    specificity merely declines to. They should never be summed into one magnitude.
    """
    gold_id = view.resolve(gold_id) or gold_id
    pred_id = view.resolve(pred_id) or pred_id
    if gold_id == pred_id:
        return 0
    gold_depth, pred_depth = view.depth(gold_id), view.depth(pred_id)
    if gold_depth is None or pred_depth is None:
        return None
    if view.is_ancestor(gold_id, pred_id):
        return pred_depth - gold_depth          # prediction below ground truth: over-specific
    if view.is_ancestor(pred_id, gold_id):
        return pred_depth - gold_depth          # prediction above ground truth: under-specific (negative)
    return None


def specificity_distribution(pairs: list[tuple[str, str]], view) -> dict:
    """Reported as a distribution, not a mean (§6.2)."""
    signed = [specificity_error(g, p, view) for g, p in pairs]
    on_path = [s for s in signed if s is not None]
    histogram = Counter(on_path)
    over = [s for s in on_path if s > 0]
    under = [s for s in on_path if s < 0]
    return {
        "n": len(signed),
        "n_on_path": len(on_path),
        "n_off_path": len(signed) - len(on_path),
        "n_exact": sum(1 for s in on_path if s == 0),
        "n_over_specific": len(over),
        "n_under_specific": len(under),
        "mean_over": sum(over) / len(over) if over else 0.0,
        "mean_under": sum(under) / len(under) if under else 0.0,
        "histogram": dict(sorted(histogram.items())),
    }


# ---------------------------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------------------------


def escalation_quality(
    escalated: set, human_ambiguous: set
) -> dict:
    """Escalation precision and recall (§6.2 targets: > 0.7 and > 0.9).

    Precision is the fraction of escalated spans a human agreed genuinely needed review. Recall is
    the fraction of spans a human judged ambiguous that the system escalated. Both are needed: a
    system escalating everything has perfect recall and useless precision, and one escalating
    nothing has the reverse. Neither is minimised, escalation is a success state.
    """
    tp = len(escalated & human_ambiguous)
    precision = tp / len(escalated) if escalated else 0.0
    recall = tp / len(human_ambiguous) if human_ambiguous else 0.0
    return {
        "n_escalated": len(escalated),
        "n_human_ambiguous": len(human_ambiguous),
        "precision": precision,
        "recall": recall,
        "meets_precision_target": precision > 0.70,
        "meets_recall_target": recall > 0.90,
    }


def fabrication_rate(predicted_ids: list[str], view) -> dict:
    """Identifiers absent from the fixed release. Target: zero, **by design**.

    Under the earlier runs'constrained decoding this cannot be non-zero, the model emits an option letter,
    never an identifier, so a non-zero value here does not mean the model hallucinated. It means
    the constraint has been bypassed somewhere, which is a far more serious finding than a
    hallucination and is why the metric is computed even though it is provably zero.
    """
    unknown = [hpo_id for hpo_id in predicted_ids if not view.in_ontology(hpo_id)]
    return {
        "n": len(predicted_ids),
        "n_fabricated": len(unknown),
        "rate": len(unknown) / len(predicted_ids) if predicted_ids else 0.0,
        "examples": sorted(set(unknown))[:10],
    }
