"""The HCY deep-dive sampling frame, as this app reads it.

``apps/compare_ui/select_hcy_documents.py`` draws 20 HCY reports into seven cells and writes the draw to
``frame.json`` / ``FRAME.md`` / ``pools.csv``. Each cell answers a different question about where a
method's error comes from, the five reports PhenoBERT scores highest on, the five it scores lowest
on, two nobody annotated, and two apiece carrying a ``family`` / ``lab_value`` / ``implicit`` /
``negated`` qualifier. This module turns that directory into the two things the UI needs: which
reports are in the sample, and *why each one was drawn*.

**It is a purposive sample, and that is a constraint on what may be displayed, not a footnote.**
Cells are defined on outcomes, PhenoBERT's own per-report score, and on annotation qualifiers, so
any rate computed over these 20 reports is a rate over the selection rule. ``FRAME.md`` says so in
bold, and :data:`WARNING` carries the same sentence into every panel and every copied Markdown
block that is showing restricted numbers. Restricting the *metrics* is therefore opt-in. Restricting
which reports you can *page through* is free and is the normal use.

A missing or unreadable frame is an ordinary state, not an error: :func:`load` returns ``None`` and
the subset control disables itself. The frame is HCY-only, GSC+ has no equivalent draw, so
:meth:`Sample.applies_to` refuses any other cohort rather than silently filtering nothing.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

FRAME_FILE = "frame.json"
POOLS_FILE = "pools.csv"

#: Display order of the cells. ``frame.json`` is written with ``sort_keys=True``, so the file's own
#: order is alphabetical and says nothing. This is ``select_hcy_deepdive.CELL_SIZES``' order, which
#: reads as an argument, the two outcome cells, then the unannotated control, then the qualifiers.
CELL_ORDER: tuple[str, ...] = ("pb_best", "pb_worst", "no_annotation",
                               "family", "lab_value", "implicit", "negated")

#: One line for each cell, shown beside the report so the reason it was drawn is on screen rather
#: than in a script nobody has open. Wording follows ``select_hcy_deepdive``'s own docstring and its
#: "Notes carried into the analysis" section.
CELL_NOTES: dict[str, str] = {
    "pb_best": "PhenoBERT scores this report among its best, drawn at random from its top 10.",
    "pb_worst": "PhenoBERT scores this report among its worst, drawn at random from its bottom 10.",
    "no_annotation": "No source annotated this report, so its ground-truth set is empty and every "
                     "prediction on it is a false positive by design. Read it as a "
                     "precision probe only.",
    "family": "Carries a family annotation. Family findings are excluded from the curated ground truth, "
              "so a method predicting one scores a false positive it arguably should not, that is "
              "the point of the cell, not a bug in it.",
    "lab_value": "Carries a lab_value annotation, a finding stated as a measurement rather than "
                 "a phenotype name.",
    "implicit": "Carries an implicit annotation, a finding the report describes without naming.",
    "negated": "Carries a negated annotation, a finding the report explicitly rules out.",
}

#: The sentence that must travel with any number computed over the sample.
WARNING = ("Purposive sample, cells are defined on PhenoBERT's own per-report score and on "
           "annotation qualifiers, so any rate over these reports is a rate over the selection "
           "rule. Mechanisms travel; numbers stay on the full cohort.")

#: The four qualifier counts ``pools.csv`` carries per report.
QUALIFIERS: tuple[str, ...] = ("family", "lab_value", "implicit", "negated")

#: The cohort the frame describes. Everything here is HCY; GSC+ has no draw.
COHORT = "hcy"


@dataclass(frozen=True)
class Sample:
    """One loaded sampling frame."""

    path: str
    picks: dict[str, list[str]]
    pools: dict[str, list[str]] = field(default_factory=dict)
    rows: dict[str, dict] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    generated: str = ""

    @property
    def cells(self) -> tuple[str, ...]:
        """Cells that actually drew something, in :data:`CELL_ORDER`, unknown ones last."""
        drawn = [c for c in self.picks if self.picks[c]]
        known = [c for c in CELL_ORDER if c in drawn]
        return tuple(known + sorted(c for c in drawn if c not in CELL_ORDER))

    @property
    def cell_of(self) -> dict[str, str]:
        """``{report_id: cell}``. The draw makes reports distinct, so this is well defined."""
        return {rid: cell for cell, ids in self.picks.items() for rid in ids}

    @property
    def selected(self) -> list[str]:
        """Every selected report, sorted."""
        return sorted(self.cell_of)

    def ids(self, mode: str | None, cell: str | None = None) -> list[str] | None:
        """The report ids a subset choice means, or ``None`` for "do not filter".

        ``None`` and ``"all"`` both mean unrestricted, ``None`` is what an absent or cleared
        control sends, and conflating them here keeps every caller from having to.
        """
        if not mode or mode == "all":
            return None
        if mode == "cell":
            return sorted(self.picks.get(cell or "", ()))
        return self.selected

    def row(self, report_id: str) -> dict | None:
        """This report's ``pools.csv`` row, PhenoBERT's score on it and its qualifier counts."""
        return self.rows.get(str(report_id))

    def label(self, report_id: str) -> str:
        """The cell name, or the empty string for a report outside the draw."""
        return self.cell_of.get(str(report_id), "")

    def note(self, report_id: str) -> str:
        """The selection note of *report_id*, or an empty string."""
        return CELL_NOTES.get(self.label(report_id), "")

    def applies_to(self, cohort: str) -> bool:
        """True when this sampling frame was drawn from *cohort*."""
        return cohort == COHORT

    def describe(self, mode: str | None, cell: str | None = None) -> str:
        """A short human name for a subset choice, for banners and provenance lines."""
        chosen = self.ids(mode, cell)
        if chosen is None:
            return ""
        if mode == "cell":
            return f"deep-dive cell {cell} ({len(chosen)} report(s))"
        return f"the deep-dive sample ({len(chosen)} report(s))"


def _number(value, cast):
    """``cast(value)``, or ``None``, an empty cell means "not applicable", never 0.

    A report with no ground truth has no precision, and writing 0.0 there would put a perfect-precision
    report and an unmeasurable one in the same bucket.
    """
    if value is None or value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _read_pools(path: str) -> dict[str, dict]:
    """``pools.csv`` → ``{report_id: row}``, numbers parsed. Absent is fine. It is additive."""
    if not os.path.isfile(path):
        return {}
    rows: dict[str, dict] = {}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                report_id = str(raw.get("patient_id") or "")
                if not report_id:
                    continue
                row: dict = {"cell": raw.get("cell") or ""}
                for key in ("n_gold", "n_pred", "tp", "fp", "fn", *QUALIFIERS):
                    row[key] = _number(raw.get(key), int)
                for key in ("precision", "recall", "f1"):
                    row[key] = _number(raw.get(key), float)
                rows[report_id] = row
    except (OSError, csv.Error) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}
    return rows


def load(path: str) -> Sample | None:
    """Read a sampling frame from its directory, or straight from a ``frame.json``.

    Returns ``None`` when there is nothing there or what is there does not parse, this is an
    additive feature and an absent frame must leave the app as it was, the way an absent
    the PhenoBERT baseline run leaves the PhenoBERT underlay off.
    """
    if not path:
        return None
    frame_path = path if path.endswith(".json") else os.path.join(path, FRAME_FILE)
    if not os.path.isfile(frame_path):
        return None
    try:
        with open(frame_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read sampling frame %s: %s", frame_path, exc)
        return None

    picks_raw = payload.get("picks")
    if not isinstance(picks_raw, dict):
        log.warning("sampling frame %s has no picks mapping, ignoring it", frame_path)
        return None
    picks = {str(cell): [str(r) for r in (ids or ())] for cell, ids in picks_raw.items()}
    if not any(picks.values()):
        log.warning("sampling frame %s drew no reports, ignoring it", frame_path)
        return None

    pools_raw = payload.get("pools") or {}
    pools = {str(cell): [str(r) for r in (ids or ())] for cell, ids in pools_raw.items()}
    directory = os.path.dirname(frame_path) or "."

    return Sample(
        path=directory,
        picks=picks,
        pools=pools,
        rows=_read_pools(os.path.join(directory, POOLS_FILE)),
        params=payload.get("params") or {},
        generated=str(payload.get("generated") or ""),
    )
