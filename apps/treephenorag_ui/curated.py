"""The curated HCY ground truth, as this app reads it.

``experiments/03_setup/ground_truth`` walked ``app/hcy_curation_ui`` end to end and wrote the
result down as a dated directory beside the cohort it was built from::

    <hcy_dir>/curated_ground_truth_2026-09-08/
        hcy_ground_truth_curated.csv   patient_id,hpo_codes -- the applied policy
        hcy_curated_annotations.csv    every candidate, in the ground truth or excluded, with its evidence
        hcy_curated_reports.csv        one row per report: in the cohort or not, and why
        manifest.json                  the policy, the inputs, the curation log's hash
        README.md

This module reads the last four. **It does not reimplement the ground truth.** The two-column file is a
drop-in for ``hcy_gt_path``, so the ground truth this app scores against still travels the one code path
every other ground truth does -- ``HCYDataset.load_ground_truth`` in :meth:`Registry.gold`. A second
implementation of the inclusion policy here would be free to disagree with the table the ground-truth build
publishes, and then "the UI says recall is R" and "the ground-truth build says recall is R" would be two claims
about two ground-truth sets wearing one name. What is loaded here is the **evidence behind** that ground truth.

Three things it adds that no other ground truth file in this repo can:

**The cohort is a claim, not an accident.** A report outside the curated cohort is *absent* from
the ground truth file rather than present with an empty cell, because an empty cell is a report that
genuinely has no phenotypes and an absent one is a report nobody annotated. ``scoring.align``
already intersects, so those reports drop out of every metric on their own -- but silently. The
reports table says how many, and :meth:`CuratedGold.outside` names them, so a recall computed over
40 of 60 reports says so on screen instead of looking like a run that lost twenty reports.

**An annotated term now has a location.** ``segment_idx`` + ``trigger_word`` is the segment and the words
a curator placed the term on. The original HCY ground truth is a set of codes per report with no offsets,
which is why the deep-dive's text panel could only ever draw PhenoBERT's matches. With this, a
*miss* can be shown on the words it was missed at. The index is the same ``sent_index`` the call
records carry -- ``experiments/03_setup/segment_reports.py`` and ``hpo_extraction.treephenorag.score_store`` both segment with
``split_sents(segment_dict(load_stanza(...)))`` -- but that is **verified per annotation**, not assumed, by checking the shipped ``segment_text`` against the segmentation this app
recovered. An annotation that fails falls back to locating its own segment text, and then to no
placement at all. A subtly wrong placement is worse than none: the index would still resolve, and
every underline would be on the wrong sentence.

**A false positive can be an excluded annotation.** The annotations table carries the candidates
the policy *dropped* -- a ``family`` finding, an ``unsure`` qualifier, a term nobody could evidence location
-- and each one's reason. A method predicting one of those scores a false positive it arguably
should not, and that is a different fact about the method than an invented term.
:meth:`CuratedGold.excluded` returns the dropped row, so the deep-dive can say which.

A missing or unreadable dataset is an ordinary state: :func:`load` returns ``None`` and every
consumer degrades to what it showed before, the same way an absent the PhenoBERT baseline run leaves the
PhenoBERT underlay off. This is HCY-only -- GSC+ has no curation pass -- and
:meth:`CuratedGold.applies_to` refuses any other cohort, not filtering nothing.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: The dated directory ``hcy_ground_truth/dataset.py`` writes, and the four files in it.
#: Spelled here, not imported: this app must open a dataset directory that was written
#: months ago by a version of that module nobody has checked out, and an import would also drag
#: ``apps.curation_ui`` (and its label vocabulary) into a UI that needs none of it.
DIR_PREFIX = "curated_ground_truth_"
GOLD_FILE = "hcy_ground_truth_curated.csv"
ANNOTATIONS_FILE = "hcy_curated_annotations.csv"
REPORTS_FILE = "hcy_curated_reports.csv"
MANIFEST_FILE = "manifest.json"

#: The cohort a curation pass exists for. GSC+ is published abstracts with their own ground truth.
COHORT = "hcy"

#: ``curated_gold_YYYY-MM-DD``. Matched, not globbed on ``*`` so a stray ``curated_gold_old``
#: directory cannot become the newest dataset by sorting after every real date.
_DIR_RE = re.compile(r"^" + re.escape(DIR_PREFIX) + r"(\d{4}-\d{2}-\d{2})$")

#: Why a candidate annotation is not in the ground truth, and what that means for a method that predicted
#: it anyway. Keyed by ``exclude_reason`` as ``curated_gold._decide`` writes it. An unknown reason
#: falls back to the bare string, so a new one in a future dataset shows, not vanishing.
EXCLUDE_HELP: dict[str, str] = {
    "family": "A family-history finding -- the phenotype belongs to a relative, not the patient. "
              "The curated policy drops it, so predicting it is a false positive that the raw HCY "
              "ground truth would have scored as a hit.",
    "unsure": "A curator marked the annotation unsure. It is neither ground truth nor a licensed "
              "prediction; read this false positive as unadjudicated rather than wrong.",
    "removed": "A curator ruled against this annotation. The prediction is a false positive on "
               "evidence, not by omission.",
    "deleted": "Under an open deletion proposal when the dataset was built.",
    "unanchored": "No segment and no trigger word could be located for it in the report, so the "
                  "curated policy has no evidence to stand the term on. It may still be correct.",
}


@dataclass(frozen=True)
class Annotation:
    """One candidate annotation -- in the ground truth or excluded from it, with the evidence either way."""

    patient_id: str
    hpo_code: str
    hpo_name: str = ""
    in_gold: bool = False
    exclude_reason: str = ""
    source: str = ""
    status: str = ""
    segment_idx: int | None = None
    trigger_word: str = ""
    segment_text: str = ""
    anchored: bool = False
    anchor_how: str = ""
    qualifiers: tuple[str, ...] = ()
    note: str = ""

    @property
    def located(self) -> bool:
        """Does this row carry enough to point at words in the report?"""
        return self.segment_idx is not None or bool(self.segment_text)

    @property
    def why(self) -> str:
        """The long form of :attr:`exclude_reason`, or the bare reason when it is a new one."""
        if self.in_gold or not self.exclude_reason:
            return ""
        return EXCLUDE_HELP.get(self.exclude_reason, self.exclude_reason)

    def describe(self) -> str:
        """One line for a table cell: the words it sits on, or that it has none."""
        if self.trigger_word and self.segment_idx is not None:
            return f"“{self.trigger_word}” in sentence {self.segment_idx}"
        if self.trigger_word:
            return f"“{self.trigger_word}”"
        if self.segment_idx is not None:
            return f"sentence {self.segment_idx}"
        return "not located"


@dataclass(frozen=True)
class CuratedGold:
    """One loaded curated dataset directory."""

    path: str
    gold: dict[str, set[str]]
    annotations: dict[str, list[Annotation]] = field(default_factory=dict)
    reports: dict[str, dict] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    # -- identity and provenance ---------------------------------------------

    @property
    def name(self) -> str:
        """Display name of the curated ground truth."""
        return os.path.basename(self.path.rstrip("/\\")) or self.path

    @property
    def date(self) -> str:
        """Build date of the curated ground truth (``YYYY-MM-DD``)."""
        return str(self.manifest.get("date") or "")

    @property
    def gold_path(self) -> str:
        """Path of the ground-truth CSV."""
        return os.path.join(self.path, GOLD_FILE)

    @property
    def policy(self) -> dict:
        """The inclusion policy the ground truth was built with."""
        return dict(self.manifest.get("policy") or {})

    @property
    def counts(self) -> dict:
        """Counts recorded at build time."""
        return dict(self.manifest.get("counts") or {})

    @property
    def curation_log_digest(self) -> str:
        """The SHA-256 prefix of the curation event log, so a number can name the log behind it."""
        return str((self.manifest.get("inputs") or {}).get("curation_log_sha256_16") or "")

    def applies_to(self, cohort: str) -> bool:
        """True for the HCY cohort, the only one this ground truth describes."""
        return cohort == COHORT

    def describe(self) -> str:
        """A short provenance sentence for a banner or a copied Markdown block."""
        parts = [self.name]
        if self.date:
            parts.append(f"built {self.date}")
        parts.append(f"{len(self.gold)} report(s), {self.n_gold_pairs} annotated term(s)")
        if self.curation_log_digest:
            parts.append(f"curation log {self.curation_log_digest}")
        return " · ".join(parts)

    # -- the cohort ----------------------------------------------------------

    @property
    def n_gold_pairs(self) -> int:
        """Number of (report, term) pairs in the ground truth."""
        return sum(len(v) for v in self.gold.values())

    @property
    def n_excluded(self) -> int:
        """Number of annotations kept out of the ground truth."""
        return sum(1 for rows in self.annotations.values() for a in rows if not a.in_gold)

    def in_cohort(self, patient_id: str) -> bool:
        """True when *patient_id* is in the curated cohort."""
        return str(patient_id) in self.gold

    def outside(self, report_ids) -> list[str]:
        """Which of *report_ids* the curated cohort excludes -- the reports no metric here covers.

        ``scoring.align`` drops them on its own, which is correct and invisible. Naming them is
        what stops a cohort that shrank from reading as a run that lost reports.
        """
        return sorted({str(r) for r in report_ids} - set(self.gold))

    def report_row(self, patient_id: str) -> dict:
        """The report-level row of *patient_id*, or an empty dict."""
        return self.reports.get(str(patient_id), {})

    def cohort_reason(self, patient_id: str) -> str:
        """Why this report is in or out of the cohort, in the dataset's own words."""
        return str(self.report_row(patient_id).get("cohort_reason") or "")

    # -- the evidence --------------------------------------------------------

    def for_report(self, patient_id: str) -> list[Annotation]:
        """The annotations of *patient_id*."""
        return self.annotations.get(str(patient_id), [])

    def find(self, patient_id: str, hpo_code: str) -> Annotation | None:
        """The best row for one (report, term), or ``None``.

        "Best" is a real choice: one term can be annotated by several sources, and they need not
        agree on whether it survived the policy or where it sits. In-ground truth rows win over excluded
        ones -- the question a caller asks about a true positive or a miss is *where is the
        evidence*, and an excluded duplicate is not it. Within a group, a located row wins over
        an unanchored one and a located one over a floating one, because those are the rows that
        can actually be drawn on the report.
        """
        code = str(hpo_code).strip()
        rows = [a for a in self.for_report(patient_id) if a.hpo_code == code]
        if not rows:
            return None
        return max(rows, key=lambda a: (a.in_gold, a.anchored, a.located, bool(a.trigger_word)))

    def excluded(self, patient_id: str, hpo_code: str) -> Annotation | None:
        """The row for a term a curator saw and the policy dropped -- else ``None``.

        This is what turns a bare false positive into a described one. It returns
        nothing when the term is in the ground truth: a term that is both is a true positive, and
        reporting its excluded duplicate would be an accusation with no basis.
        """
        code = str(hpo_code).strip()
        if code in self.gold.get(str(patient_id), set()):
            return None
        rows = [a for a in self.for_report(patient_id) if a.hpo_code == code and not a.in_gold]
        if not rows:
            return None
        return max(rows, key=lambda a: (a.anchored, a.located, bool(a.trigger_word)))

    def qualifiers(self, patient_id: str, hpo_code: str) -> tuple[str, ...]:
        """Qualifiers (negated, family, lab value and so on) of one annotation, or an empty tuple."""
        row = self.find(patient_id, hpo_code)
        return row.qualifiers if row else ()

    def placements(self, patient_id: str, sentences) -> dict[int, list[Annotation]]:
        """``{sentence_index: [Annotation, ...]}`` for the annotated terms of one report.

        The join between the curated dataset's ``segment_idx`` and this app's ``sent_index`` is
        checked, not assumed. Both sides run ``split_sents(segment_dict(load_stanza(...)))`` --
        ``experiments/03_setup/segment_reports.py`` and ``hpo_extraction.treephenorag.score_store`` respectively -- so the indices
        *should* agree, and where they do this is a no-op. Where they do not, the shipped
        ``segment_text`` is located in the recovered segmentation instead, and an annotation that
        matches nothing is left unplaced, not drawn somewhere plausible.

        Only rows in the ground truth are placed. An excluded annotation drawn on the report would read as
        something a method was expected to find.
        """
        out: dict[int, list[Annotation]] = {}
        if not sentences:
            return out
        normalised = [_squash(s) for s in sentences]
        by_text: dict[str, int] = {}
        for index, text in enumerate(normalised):
            by_text.setdefault(text, index)   # first occurrence: a repeated sentence is ambiguous

        for row in self.for_report(patient_id):
            if not row.in_gold:
                continue
            index = _place(row, normalised, by_text)
            if index is None:
                continue
            out.setdefault(index, []).append(row)
        return out

    def placement_coverage(self, patient_id: str, sentences, marks=None) -> tuple[int, int]:
        """``(placed, in_gold)`` for one report -- what the text panel could and could not draw.

        Counted in **distinct codes**, not annotation rows. A term two sources annotated is one
        annotated term, and counting the rows made the denominator larger than the number of chips on
        screen, which reads as terms going missing.

        *marks* is :meth:`placements`' own output where the caller already has it. The text panel
        does, and recomputing the whole placement pass for a count ran the ladder twice per render.
        """
        if marks is None:
            marks = self.placements(patient_id, sentences)
        placed = {row.hpo_code for rows in marks.values() for row in rows}
        total = {a.hpo_code for a in self.for_report(patient_id) if a.in_gold}
        return len(placed), len(total)

    def locations(self, patient_id: str, sentences, marks=None) -> dict[int, list[tuple]]:
        """``{sentence_index: [(start, end, Annotation), ...]}`` -- the ground truth, on its own words.

        A curated annotation names a segment *and* the trigger word the curator read it at, and
        ``hcy_ground_truth`` marks it ``anchored`` only when that word really occurs in that segment. So
        the word can be drawn where it sits, the way the curation app draws it: the reader sees
        "missed *Anfaelle*, sentence 4", not "recall lost HP:0001250".

        The search is ``anchors.find_trigger`` -- a word-boundary regex, whitespace-tolerant --
        and it runs **only inside the one segment the curator named**. That is the whole
        difference between evidence and a guess: a trigger located in its own named segment is
        what was recorded, while the same string searched across the report would land on the
        first of three occurrences of a common word and claim a position nobody wrote down. A row
        whose trigger is absent from its segment gets no span and falls back to a chip.
        """
        if marks is None:
            marks = self.placements(patient_id, sentences)
        out: dict[int, list[tuple]] = {}
        for index, rows in marks.items():
            if not 0 <= index < len(sentences):
                continue
            text = sentences[index]
            for row in rows:
                if not row.trigger_word:
                    continue
                span = _find_trigger(text, row.trigger_word)
                if span is None:
                    continue
                out.setdefault(index, []).append((span[0], span[1], row))
        for rows in out.values():
            rows.sort(key=lambda item: (item[0], -item[1]))
        return out


def _find_trigger(text: str, trigger: str):
    """``anchors.find_trigger``, degrading to ``None`` where the curation app cannot be imported.

    The import is lazy and guarded for the same reason every other cross-app import here is: an
    absent or unreadable dependency must leave this app as it was, drawing chips instead
    of underlines, not taking the page down.
    """
    try:
        from hpo_extraction.curation.evidence_location import find_trigger
    except Exception:  # noqa: BLE001 - an optional reading aid, never a hard dependency
        return None
    try:
        return find_trigger(text, trigger)
    except Exception:  # noqa: BLE001
        return None


def _squash(text) -> str:
    """Whitespace-collapsed and casefolded, for comparing two segmentations of one report."""
    return " ".join(str(text or "").split()).casefold()


def _place(row: Annotation, normalised: list[str], by_text: dict[str, int]) -> int | None:
    """Which recovered sentence *row* belongs to -- index first, then its own text, then nothing."""
    wanted = _squash(row.segment_text)
    index = row.segment_idx
    if index is not None and 0 <= index < len(normalised):
        if not wanted or normalised[index] == wanted:
            return index
    if wanted:
        if wanted in by_text:
            return by_text[wanted]
        # A segmentation that split one sentence differently still contains the text. The length
        # guard is what keeps a two-word fragment from matching the first sentence it occurs in.
        for candidate, text in enumerate(normalised):
            if wanted in text or (len(text) > 8 and text in wanted):
                return candidate
    return None


# ------------------------------------------------------------------------------
# finding and reading a dataset
# ------------------------------------------------------------------------------
def hcy_dir_of(gt_path: str) -> str:
    """The cohort directory a ground truth file sits in, whether or not it is inside a dataset directory.

    ``.../hcy/hcy_ground_truth_raw.csv`` and ``.../hcy/curated_ground_truth_2026-09-08/...curated.csv``
    both answer ``.../hcy``, so the sidebar can offer every dataset from whichever file is loaded.
    """
    if not gt_path:
        return ""
    parent = os.path.dirname(os.path.abspath(gt_path))
    if _DIR_RE.match(os.path.basename(parent)):
        return os.path.dirname(parent)
    return parent


def discover(hcy_dir: str) -> list[str]:
    """Every curated dataset directory under *hcy_dir*, **newest first**.

    A directory counts only when its two-column ground truth file is actually there: a half-written
    dataset offered in a dropdown is a path that loads to an empty scorecard.
    """
    if not hcy_dir or not os.path.isdir(hcy_dir):
        return []
    try:
        entries = os.listdir(hcy_dir)
    except OSError as exc:
        log.warning("could not list %s: %s", hcy_dir, exc)
        return []
    found = []
    for name in entries:
        match = _DIR_RE.match(name)
        if not match:
            continue
        path = os.path.join(hcy_dir, name)
        if os.path.isfile(os.path.join(path, GOLD_FILE)):
            found.append((match.group(1), path))
    return [path for _, path in sorted(found, reverse=True)]


def newest(hcy_dir: str) -> str:
    """The newest curated dataset directory under *hcy_dir*, or ``""``."""
    found = discover(hcy_dir)
    return found[0] if found else ""


def gold_path_for(hcy_dir: str) -> str:
    """The newest dataset's two-column ground truth file, or ``""`` -- what the app defaults to."""
    directory = newest(hcy_dir)
    return os.path.join(directory, GOLD_FILE) if directory else ""


def dataset_dir_of(gt_path: str) -> str:
    """The curated dataset directory a loaded ground truth file belongs to, or ``""`` for any other file.

    This is the whole selection mechanism: the curated ground truth is chosen by pointing ``hcy_gt_path``
    at ``hcy_ground_truth_curated.csv``, and the sidecar evidence is found from that path, not configured separately. One control, and no way to load the annotations of one dataset
    beside the ground truth of another.
    """
    if not gt_path:
        return ""
    parent = os.path.dirname(os.path.abspath(gt_path))
    if _DIR_RE.match(os.path.basename(parent)) and os.path.isfile(
            os.path.join(parent, GOLD_FILE)):
        return parent
    return ""


def label_for(path: str) -> str:
    """The sidebar's one-line description of a dataset directory, read from its manifest.

    Cheap on purpose -- the dropdown is built on every Load, and reading four CSVs per candidate
    to fill in a label would make the sidebar the slowest thing in the app.
    """
    name = os.path.basename(str(path).rstrip("/\\"))
    manifest_path = os.path.join(path, MANIFEST_FILE)
    counts: dict = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                counts = (json.load(handle).get("counts") or {})
        except (OSError, json.JSONDecodeError, AttributeError):
            counts = {}
    reports = counts.get("n_reports_in_cohort")
    pairs = counts.get("n_gold_pairs")
    if reports is None or pairs is None:
        return f"{name}, the curated ground truth"
    return (f"{name}, the curated ground truth ({reports} report{'' if reports == 1 else 's'}, "
            f"{pairs} term{'' if pairs == 1 else 's'})")


def _rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        log.warning("could not read %s: %s", path, exc)
        return []


def _flag(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _index(value) -> int | None:
    """``segment_idx`` as an int, or ``None`` -- an empty cell means unplaced, never sentence 0."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _number(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return text


def _annotation(raw: dict) -> Annotation | None:
    patient_id = str(raw.get("patient_id") or "").strip()
    hpo_code = str(raw.get("hpo_code") or "").strip()
    if not patient_id or not hpo_code:
        return None
    qualifiers = tuple(q.strip() for q in str(raw.get("qualifiers") or "").split(";") if q.strip())
    return Annotation(
        patient_id=patient_id,
        hpo_code=hpo_code,
        hpo_name=str(raw.get("hpo_name") or ""),
        in_gold=_flag(raw.get("in_gold")),
        exclude_reason=str(raw.get("exclude_reason") or ""),
        source=str(raw.get("source") or ""),
        status=str(raw.get("status") or ""),
        segment_idx=_index(raw.get("segment_idx")),
        trigger_word=str(raw.get("trigger_word") or ""),
        segment_text=str(raw.get("segment_text") or ""),
        anchored=_flag(raw.get("anchored")),
        anchor_how=str(raw.get("anchor_how") or ""),
        qualifiers=qualifiers,
        note=str(raw.get("note") or ""),
    )


def load(path: str) -> CuratedGold | None:
    """Read a curated dataset directory, or ``None`` when there is not one there.

    Accepts the directory or any file inside it, so a caller holding ``hcy_gt_path`` can pass it
    straight through.

    The ground truth mapping is read from the dataset's own two-column file so this object stands on its
    own in a test or a selfcheck. In the running app that same file has already been read by
    :meth:`Registry.gold` through ``HCYDataset``, and :mod:`apps.treephenorag_ui.selfcheck` asserts the two
    agree -- the check that catches a dataset directory whose ground truth file and annotation table were
    written by different runs.
    """
    if not path:
        return None
    directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    gold_file = os.path.join(directory, GOLD_FILE)
    if not os.path.isfile(gold_file):
        return None

    gold: dict[str, set[str]] = {}
    for raw in _rows(gold_file):
        patient_id = str(raw.get("patient_id") or "").strip()
        if not patient_id:
            continue
        cell = str(raw.get("hpo_codes") or "")
        gold[patient_id] = {c.strip() for c in cell.replace(",", ";").split(";") if c.strip()}
    if not gold:
        log.warning("curated dataset %s has no ground truth rows -- ignoring it", directory)
        return None

    annotations: dict[str, list[Annotation]] = {}
    for raw in _rows(os.path.join(directory, ANNOTATIONS_FILE)):
        row = _annotation(raw)
        if row is not None:
            annotations.setdefault(row.patient_id, []).append(row)

    reports: dict[str, dict] = {}
    for raw in _rows(os.path.join(directory, REPORTS_FILE)):
        patient_id = str(raw.get("patient_id") or "").strip()
        if not patient_id:
            continue
        row = dict(raw)
        for key in ("n_segments", "n_annotations", "n_gold_terms", "n_dropped", "n_unanchored",
                    "n_prior_annotation", "n_daphne", "n_suggestions", "n_adjudicated", "n_comments"):
            row[key] = _number(raw.get(key))
        row["in_cohort"] = _flag(raw.get("in_cohort"))
        row["is_confirmed"] = _flag(raw.get("is_confirmed"))
        reports[patient_id] = row

    manifest: dict = {}
    manifest_path = os.path.join(directory, MANIFEST_FILE)
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", manifest_path, exc)

    return CuratedGold(path=directory, gold=gold, annotations=annotations,
                       reports=reports, manifest=manifest)
