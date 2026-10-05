"""Recall sliced by *why* an annotated term was hard, from the curation pass's own qualifier labels.

"Recall = 0.55" hides every reason a term was missed. ``src/hpo_extraction/curation/labels.py`` records
seven qualifiers on the annotation itself, the curator ticked them while looking at the report, and ``hcy_ground_truth`` exports them to ``hcy_curated_annotations.csv`` as ``q_*`` columns. Joining those
against a method's predictions turns the aggregate into an answer: does this system fail on
measurements, on descriptions, or on attribution?

Three subgroups, and they are **not** three instances of one thing:

``lab_value``   the evidence is a number with a unit, not a phrase. Nothing lexical points at the
                term, so a dictionary or a span tagger has no route to it at all.
``implicit``    described but never named ("could not hold his head up" for hypotonia). A lexical
                route does not exist either, but for a different reason: the reader has to
                understand the sentence rather than resolve a value against a range.
``family``      the finding belongs to a **relative**. This one is scored the other way up.

## Why family is a count and not a recall

The shipped ground truth policy **excludes** family-labelled annotations
(``curated_gold.DEFAULT_EXCLUDE_LABELS``), so those terms are not in the ground truth and there is no
denominator to recall against. What a family label supports is the opposite measurement: a method
that emits a relative's phenotype as the patient's has made an *attribution* error, and that
prediction is already a false positive in the main table. This module names which false
positives those are.

It is reported as a **count over 12 pairs in 5 reports**, with no rate and no interval. A ratio on
twelve pairs would move by 0.08 per pair and invites the over-reading it cannot support.

**One family-labelled pair is legitimately in the ground truth.** ``SYN101 / HP:0100502`` carries a family
label on one annotation row and none on a duplicate row from another source, and the duplicate
survived the policy. Predicting it is *correct*, so it is subtracted: the attribution set is every
labelled pair the final ground truth does not contain. :func:`build` does that subtraction against the ground truth
file itself, not against the ``in_gold`` column, which keeps the recall subgroups provably
subsets, and the attribution set provably non-members, of the denominator the main table uses.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: The qualifier columns ``hcy_ground_truth``'s annotation export writes, keyed by the slice they define.
RECALL_LABELS: tuple[str, ...] = ("lab_value", "implicit")
ATTRIBUTION_LABEL = "family"

#: Every qualifier, so the audit table can report the ones that are too small to slice on rather
#: than leaving the reader to wonder whether they were considered.
ALL_LABELS: tuple[str, ...] = (
    "lab_value", "implicit", "family", "negated", "resolved",
    "unsure_report", "unsure_annotation",
)

ID_COLUMN = "patient_id"
CODE_COLUMN = "hpo_code"


@dataclass
class Subgroup:
    """One slice of the ground truth, as the pairs it contains and the reports they live in."""

    label: str
    kind: str                                   # "recall" | "attribution"
    pairs: set = field(default_factory=set)     # {(report_id, hpo_code)}

    @property
    def report_ids(self) -> list:
        """Reports that have at least one pair in the subgroup."""
        return sorted({r for r, _ in self.pairs})

    @property
    def by_report(self) -> dict:
        """``{report: set of HPO identifiers}`` of the subgroup."""
        out: dict = {}
        for report_id, code in self.pairs:
            out.setdefault(report_id, set()).add(code)
        return out


def load_annotations(path: str) -> list[dict]:
    """The rows of ``hcy_curated_annotations.csv``, unfiltered."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def labelled_pairs(rows: list[dict], label: str) -> set:
    """Every ``(report, code)`` any annotation row flags with *label*.

    Taken over rows, not over annotation files: one phenotype recorded twice, once with the
    label and once without, is one labelled phenotype. That is the reading the curator's tick
    supports, they ticked the term, not the file it happened to arrive in.
    """
    column = "q_" + label
    if rows and column not in rows[0]:
        raise KeyError(
            "the annotation export has no " + repr(column) + " column. It is written by "
            "exp13_18's `dataset` stage from app/hcy_curation_ui's own vocabulary; an older "
            "export predates the qualifier columns and cannot support the subgroup table.")
    return {(str(r[ID_COLUMN]).strip(), str(r[CODE_COLUMN]).strip())
            for r in rows if str(r.get(column, "")).strip() == "1"}


def build(rows: list[dict], gold: dict) -> dict:
    """``{label: Subgroup}`` for the two recall slices and the attribution slice.

    *ground truth* is the scoring ground truth itself, ``{report_id: set of codes}``, so a recall subgroup is a
    subset of the main denominator by design, not by agreement between two files.
    """
    in_gold = {(r, c) for r, codes in gold.items() for c in codes}
    out: dict = {}
    for label in RECALL_LABELS:
        out[label] = Subgroup(label, "recall", labelled_pairs(rows, label) & in_gold)
    out[ATTRIBUTION_LABEL] = Subgroup(
        ATTRIBUTION_LABEL, "attribution", labelled_pairs(rows, ATTRIBUTION_LABEL) - in_gold)
    return out


def audit(rows: list[dict], gold: dict) -> list[dict]:
    """One row per qualifier: how many pairs it marks, how many are in the ground truth, how many reports.

    Written whether or not the label is sliced on. A vocabulary entry with four instances is a real
    fact about the curation pass, and reporting it is how a reader learns that ``negated`` was
    considered and is too small, not that it was forgotten.
    """
    in_gold = {(r, c) for r, codes in gold.items() for c in codes}
    out = []
    for label in ALL_LABELS:
        pairs = labelled_pairs(rows, label)
        hit = pairs & in_gold
        role = ("recall subgroup" if label in RECALL_LABELS
                else "attribution set" if label == ATTRIBUTION_LABEL
                else "not sliced on")
        out.append({
            "label": label, "role": role,
            "n_pairs_labelled": len(pairs),
            "n_pairs_in_gold": len(hit),
            "n_pairs_not_in_gold": len(pairs - in_gold),
            "n_reports_labelled": len({r for r, _ in pairs}),
            "n_reports_in_gold": len({r for r, _ in hit}),
        })

    # The two recall slices are separate qualifier COLUMNS, not a partition: a curator could mark
    # one annotation both laboratory-located and implicit, so their counts double-count any such
    # pair and their sum is an upper bound on the union, not its size. T6.2 quotes "the
    # share of ground truth with no string match", which is the union, so the union has to be written --
    # otherwise the only number available is the sum and the table has to hedge.
    union = set()
    for label in RECALL_LABELS:
        union |= labelled_pairs(rows, label)
    both = set.intersection(*(labelled_pairs(rows, label) for label in RECALL_LABELS)) \
        if len(RECALL_LABELS) > 1 else set()
    out.append({
        "label": "|".join(RECALL_LABELS), "role": "recall subgroup union",
        "n_pairs_labelled": len(union),
        "n_pairs_in_gold": len(union & in_gold),
        "n_pairs_not_in_gold": len(union - in_gold),
        "n_reports_labelled": len({r for r, _ in union}),
        "n_reports_in_gold": len({r for r, _ in union & in_gold}),
        "n_pairs_in_both": len(both & in_gold),
    })
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

def recall_units(subgroup: Subgroup, predicted: dict) -> list:
    """``[(subgroup gold, predicted ∩ subgroup gold), ...]``, one entry per carrying report.

    The prediction is intersected with the subgroup's own ground truth **before** scoring, which is what
    makes this a recall and nothing else: a term the method predicted that is not in this slice is
    neither a hit nor a miss here, and counting it would produce a precision-like number over a
    denominator with no meaning.

    The unit list is the reports carrying at least one subgroup term. A report with none
    contributes 0/0 to every resample and only dilutes the interval.
    """
    units = []
    for report_id, codes in sorted(subgroup.by_report.items()):
        units.append((codes, set(predicted.get(report_id, ())) & codes))
    return units


def recall_metric(units) -> dict:
    """Micro recall over the subgroup, in the shape ``bootstrap_reports`` expects."""
    found = sum(len(p) for _, p in units)
    total = sum(len(g) for g, _ in units)
    return {"recall": found / total if total else 0.0}


def attribution_counts(subgroup: Subgroup, predicted: dict) -> dict:
    """How many of the relative-owned pairs a method emitted, and which ones.

    No rate: see the module docstring. Every hit here is also a false positive in
    ``t1_overall.csv``, which is the point, this names them, not adding to them.
    """
    hits = sorted(
        report_id + "/" + code
        for report_id, codes in subgroup.by_report.items()
        for code in sorted(codes)
        if code in set(predicted.get(report_id, ()))
    )
    return {
        "n_pairs": len(subgroup.pairs),
        "n_reports": len(subgroup.report_ids),
        "n_emitted": len(hits),
        "emitted_pairs": ";".join(hits),
    }
