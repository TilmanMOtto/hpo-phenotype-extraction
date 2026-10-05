"""The curated HCY ground truth: every **located** annotation the curation pass left standing.

``app/hcy_curation_ui`` has now been walked end to end. Every report was read, and every annotation
either placed on the words it came from or explicitly left unplaceable. What is *not* finished is
the adjudication: most rows carry no keep/remove verdict, because pressing a button on a row you
have just read and agreed with is the part of curation with the lowest information content.

So the ground truth is defined by **evidence**, not by approval. In one line:

    every phenotype from ``prior_annotation``, ``daphne`` or a curator's own suggestion that sits on a
    segment and a trigger word, minus the ones under an open deletion proposal, minus the ones a
    curator ruled against, and minus three qualifiers.

Two questions are still answered separately here, and neither is allowed to leak into the other:

**Which reports may be scored at all** (:func:`cohort`). A report qualifies when any annotation
source carries something for it, :data:`DEFAULT_CRITERIA` is ``prior_annotation``, ``daphne`` and
``suggestion``. That is a wider net than this module used to cast, and the reason is that the
curation pass is complete: the
old filter existed to keep half-curated reports out of a table, and there are none left. A report
matching nothing is still **out of the cohort entirely** rather than scored with an empty ground-truth set, an empty cell is a report that genuinely has no phenotypes, and a report nobody annotated is not
the same claim. ``segmented`` is available in :data:`CRITERIA` for the opposite reading (score every
report the segmentation covers, empty ground truth included), and is off by default.

**Which terms are in a qualifying report's ground truth** (:func:`build`), the curation log folded over the
annotation files, under an explicit :class:`GoldPolicy`.

The four decisions that policy makes, all of which the caller can see and change:

* an annotation must be **located**: a segment index, and a trigger word that actually occurs in
  that segment. This is the same test ``reader.needs_anchor`` puts on the Evidence location tab's queue, run
  from the same functions (:mod:`hpo_extraction.curation.evidence_location`), so a row this module drops is a row
  the app would have shown as still needing work. It is evaluated **per code, across every row for
  that code**, prior_annotation, daphne and the curator's own, because two files recording one annotation
  badly and well is one annotation that landed. Priced by ``keep_unanchored``.
* an **unruled** annotation is *in*. The annotation files are somebody's judgement already, and the
  curation pass read every one of them. Refusing to score against a term nobody re-confirmed would
  shrink the ground truth to whatever went through Approve mode, which measures the backlog, not the
  data. ``store.export_gold`` takes the opposite view (``IN_GOLD`` only) because it writes the
  *approved* ground truth. This is the evidence-based one.
* a **suggestion** is *in* even before it is approved, the curator's answer to "what is missing
  from this ground truth", and it carries a segment and a trigger word by design.
* three **qualifiers** take an annotation out: the two ``unsure`` labels and ``family``
  (:data:`DEFAULT_EXCLUDE_LABELS`).

Out on lifecycle, in two groups. An open deletion proposal (``delete_suggested``, a term under
active dispute is not something to score against), and the three verdicts in which a curator has
already ruled against the row: ``removed``, ``rejected``, ``needs_work`` (see :data:`OUT_STATUSES`).
The user-facing rule names only the first. The other three are kept because a curator who pressed
*Remove* has said something stronger than silence, and dropping their verdict on the floor would
make Approve mode's buttons decorative.

:data:`VARIANTS` prices every one of those choices, the same cohort, six ground truth definitions, so the
sensitivity of the main number to a curation policy is a table, not an assumption.

Nothing here reads a prediction, and nothing here is HCY-specific beyond the file names: the inputs
are the segmentation, the two annotation CSVs and the append-only log, all read through the curation
app's own loaders (:mod:`hpo_extraction.curation.sources` / ``.store`` / ``.anchors`` / ``.spans``),
not reimplemented. A second parser of the event log, or a second definition of *located*,
would be free to disagree with the screen the curator was looking at.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from hpo_extraction.curation import evidence_location as locations, sources, spans, store

logger = logging.getLogger(__name__)

#: The two doubt qualifiers. ``unsure_report`` says the text does not settle it, ``unsure_annotation``
#: says the annotation is doubtful. Both mean "a model that disagrees here is not demonstrably
#: wrong", which is the only defensible reason to drop a term from a denominator, not count
#: it as a miss.
UNSURE_LABELS: tuple[str, ...] = ("unsure_report", "unsure_annotation")

#: The qualifiers that take an annotation out of the ground truth by default: the two above, plus
#: ``family``.
#:
#: ``family`` is a different kind of exclusion and worth naming as such. The finding is real and
#: The text does support it, it just belongs to a relative. A system that extracts it is doing
#: something defensible, so counting it as a false positive measures the annotation convention
#:, not the system. It is priced on its own by the ``keep_family`` variant, because unlike
#: The doubt labels this one is a genuine editorial choice about what the task *is*.
#:
#: All three stay in the exported dataset, flagged, not deleted, see :mod:`dataset`. An
#: exclusion the reader cannot undo is a decision the file has made on their behalf.
DEFAULT_EXCLUDE_LABELS: tuple[str, ...] = UNSURE_LABELS + ("family",)

#: Statuses that put a row out of the ground truth no matter what the policy says. ``needs_work`` is here
#: because the curator flagged it as neither in nor out (``theme.STATUS_HELP``), and a table has to
#: choose. Scoring against a term somebody has explicitly parked would be choosing for them.
OUT_STATUSES: frozenset = frozenset({"rejected", "removed", "needs_work",
                                     store.DELETE_SUGGESTED})

#: Every report-level inclusion signal this module understands, in the order
#: :attr:`ReportRow.reason` lists them. The **vocabulary**, not the default, see
#: :data:`DEFAULT_CRITERIA`.
CRITERIA: tuple[str, ...] = ("prior_annotation", "daphne", "suggestion", "approved", "segmented",
                             "original_gold")

#: The signals in force by default: any annotation, from any of the three sources that can enter
#: The ground truth.
#:
#: This is the whole annotated cohort in practice, and that is the point. The narrow filter this
#: module shipped with (``daphne`` + ``suggestion``) existed because curation was half-done and
#: mixing a curated report with an uncurated one would attribute a gold-side difference to the
#: pipeline. The pass is finished, so the filter now only removes reports no annotator ever put
#: anything on.
#:
#: ``approved`` adds nothing on its own, every route to an approval passes through a report that
#: already carries an annotation or a suggestion, except via ``confirm_patient``, a whole-report
#: sign-off. ``segmented`` is the real switch: it admits every report the segmentation covers,
#: including those with **no annotation at all**, which are scored against an empty ground truth and so
#: test precision, not recall. That is a defensible cohort now that every report has been
#: read, but it is a different measurement, so it is opt-in.
#:
#: ``original_gold`` is the narrow version of that same switch, and the difference between the two
#: is a claim about who asserted what. The original ``hcy_ground_truth`` lists a report **with an
#: empty code list** for reports it holds to have no phenotype, that is an assertion, and one the
#: curation pass then read the report and let stand. A report absent from that file carries no such
#: assertion, and scoring against an empty ground truth would be inventing one. So this criterion admits
#: The reports the original ground truth enumerates, whatever those rows contain, which is also
#: what makes the curated cohort a **superset** of the original one: without it, ``curated`` minus
#: ``original`` is part gold-side difference and part cohort difference, which is the one thing
#: this experiment exists to keep apart.
#:
#: Read the empty rows for what they are worth: of the 18 the original ground truth carries, curation found
#: real phenotypes in 13. The five that survive are five reports two independent passes agree are
#: empty, not five reports nobody looked at.
DEFAULT_CRITERIA: tuple[str, ...] = ("prior_annotation", "daphne", "suggestion")


@dataclass(frozen=True)
class GoldPolicy:
    """What counts as an annotated term. Every field is a decision the curation log leaves open.

    ``require_evidence`` is the main rule: an annotation enters the ground truth only when it sits on a
    segment and a trigger word that occurs in it. Turn it off and the ground truth becomes the code sets
    the files carry, with the verdicts and the qualifiers still applied.

    ``prior_annotation_fallback`` picks between the two readings of a term the confirmed pass does not
    carry:

    ``"adjudicated"``  the curation app's own rule (``sources.adjudicated``). **At the level of a
                       term set this is the union of the two files**: a code only prior_annotation carries
                       is kept, a code only the confirmed pass carries is kept, and a code both
                       carry is kept once. What adjudication decides is not whether the code is in
                       but *which row carries its verdict*, the row the Approve screen showed the
                       curator, so a verdict lands once and settles the term instead of having to
                       be passed twice.
    ``"daphne_first"`` for a patient the confirmed pass reached, its annotation set is the whole
                       ground truth. The prior_annotation file is treated as superseded, i.e. The drop is read as a
                       decision. A patient it never reached still falls back to prior_annotation in full,
                       or that report would have no ground truth at all while sitting in the cohort on a
                       suggestion.
    """

    prior_annotation_fallback: str = "adjudicated"
    require_evidence: bool = True
    include_undecided: bool = True
    include_suggested: bool = True
    exclude_labels: tuple[str, ...] = DEFAULT_EXCLUDE_LABELS

    def __post_init__(self) -> None:
        if self.prior_annotation_fallback not in ("adjudicated", "daphne_first"):
            raise ValueError(
                "prior_annotation_fallback must be 'adjudicated' or 'daphne_first', got "
                f"{self.prior_annotation_fallback!r}")


#: The default policy, and the five one-change neighbours that price its choices. Each differs from
#: ``default`` in one field, so the sensitivity table reads as an ablation, not as
#: six unrelated ground-truth sets.
VARIANTS: dict[str, GoldPolicy] = {
    "default": GoldPolicy(),
    # What the evidence requirement costs. This row minus `default` is every annotation whose
    # trigger word could not be found in any segment, the locating queue, priced.
    "keep_unanchored": GoldPolicy(require_evidence=False),
    # What the confirmed pass dropping a term is worth, if that drop is read as a decision.
    "daphne_first": GoldPolicy(prior_annotation_fallback="daphne_first"),
    # What excluding a relative's finding is worth. The doubt labels still exclude, so this row
    # minus `default` is the `family` decision alone, the one editorial choice in the policy.
    "keep_family": GoldPolicy(exclude_labels=UNSURE_LABELS),
    # What every label exclusion is worth together.
    "keep_all_labels": GoldPolicy(exclude_labels=()),
    # What ``store.export_gold`` would write today: adjudicated rows only. The floor.
    "approved_only": GoldPolicy(include_undecided=False, include_suggested=False),
}

DEFAULT_VARIANT = "default"

#: How an annotation came to sit where it sits, in the order :func:`anchor_index` prefers them.
#: ``curated`` is a curator's own placement, the Evidence location tab, or a row editor, and outranks
#: anything a file claimed. The rest are ``anchors.locate``'s four strategies, and the distinction
#: between them is not decoration: ``lexical`` means *the app scanned the whole report and found the
#: trigger somewhere*, which is this module's guess, not the annotator's claim. It counts as
#: evidence, the words are in the report and the segment is real, but it travels in the ``how``
#: column so a reader can discount it.
LOCATION_KINDS: tuple[str, ...] = ("curated", "segment", "offset", "context", "lexical")


@dataclass(frozen=True)
class Anchor:
    """Where one annotation sits: a segment index, the trigger word, and how it was placed."""

    segment_idx: int
    trigger_word: str
    how: str

    @property
    def rank(self) -> int:
        """Preference rank of how the evidence was located (lower is more reliable)."""
        return LOCATION_KINDS.index(self.how) if self.how in LOCATION_KINDS else len(LOCATION_KINDS)


@dataclass
class TermRow:
    """One candidate annotated term, with the reason it is in or out written down beside it."""

    patient_id: str
    hpo_code: str
    hpo_name: str
    source: str            # "daphne" | "holistic" | "new"
    status: str            # "" when nobody has ruled on it
    labels: tuple
    in_gold: bool
    reason: str
    key: str
    trigger_word: str = ""
    segment_idx: object = None
    segment_text: str = ""
    anchored: bool = False
    how: str = ""          # which strategy placed it; "" when nothing did
    note: str = ""
    edited: bool = False   # The curation log rewrote this row's code
    original_hpo_code: str = ""


@dataclass
class ReportRow:
    """One report's cohort verdict, with every signal that fed it."""

    patient_id: str
    in_cohort: bool
    n_prior_annotation: int
    n_daphne: int
    n_suggestions: int
    n_adjudicated: int
    is_confirmed: bool
    reason: str
    n_segments: int = 0
    n_gold_terms: int = 0
    n_dropped: int = 0
    n_unanchored: int = 0
    difficulty: str = ""
    report_labels: tuple = ()
    n_comments: int = 0


@dataclass
class CuratedGold:
    """The product: a ground truth mapping over the cohort, plus the audit trail behind it."""

    gold: dict
    reports: list
    terms: list
    policy: GoldPolicy
    criteria: tuple
    locations: dict = field(default_factory=dict)   # {patient_id: {key: Evidence location}}

    @property
    def report_ids(self) -> list:
        """Reports in the curated cohort, sorted."""
        return sorted(self.gold)

    @property
    def n_pairs(self) -> int:
        """Number of (report, term) pairs in the curated ground truth."""
        return sum(len(v) for v in self.gold.values())

    def summary(self) -> dict:
        """Counts that describe the build: cohort size, reports seen, pairs, drops by reason."""
        return {
            "n_reports_in_cohort": len(self.gold),
            "n_reports_seen": len(self.reports),
            "n_gold_pairs": self.n_pairs,
            "n_terms_considered": len(self.terms),
            "n_terms_dropped": sum(1 for t in self.terms if not t.in_gold),
            "n_terms_unanchored": sum(1 for t in self.terms if not t.anchored),
            "n_reports_empty": sum(1 for v in self.gold.values() if not v),
            "criteria": list(self.criteria),
            "prior_annotation_fallback": self.policy.prior_annotation_fallback,
            "require_evidence": self.policy.require_evidence,
            "include_undecided": self.policy.include_undecided,
            "include_suggested": self.policy.include_suggested,
            "exclude_labels": list(self.policy.exclude_labels),
        }


# ──────────────────────────────────────────────────────────────────────────────
# inputs
# ──────────────────────────────────────────────────────────────────────────────
def resolve_paths(hcy_dir: str, prior_annotation_path: str = "", confirmed_path: str = "",
                  curation_dir: str = "", segments_path: str = "",
                  phenobert_dir: str = "") -> dict:
    """The input paths, each overridable, otherwise derived from *hcy_dir*.

    The layout is the curation app's (``registry.CURATION_SUBDIR``), so pointing this at the same
    ``hcy_dir`` the app was run against needs no further configuration.

    *phenobert_dir* is optional and affects only the **verbatim report text** the segments are
    sliced out of, as it does in ``registry._build_patient``: PhenoBERT's staged copy is
    preferred where there is one, the prior_annotation file's own ``report_text`` column otherwise. Leave
    it unset and the prior_annotation text is used throughout, which changes a trigger match only where the
    two copies of a report differ by more than whitespace.
    """
    hcy_dir = str(hcy_dir or "")
    return {
        "hcy_dir": hcy_dir,
        "segments": segments_path or os.path.join(hcy_dir, sources.SEGMENTS_FILE),
        "prior_annotation": prior_annotation_path or os.path.join(hcy_dir, sources.HOLISTIC_FILE),
        "confirmed": confirmed_path or os.path.join(hcy_dir, sources.CONFIRMED_FILE),
        "curation_dir": curation_dir or os.path.join(hcy_dir, "curation"),
        "phenobert": phenobert_dir or "",
    }


def load_annotations(paths: dict) -> dict:
    """``{source: {patient_id: [record, …]}}`` for the two rich sources, keyed as the log keys them.

    ``prior_annotation_2`` is absent. It is a second annotator's code list and carries no evidence,
    so the curation app treats it as a cross-check that can never enter the ground truth
    (``sources.ADJUDICATION_ORDER``). Admitting it here would put terms in the denominator that no
    screen ever asked anybody about, and under the evidence rule it could not be located anyway,
    since a code-only row names no words.
    """
    prior_annotation, _ = sources.load_prior_annotation(paths["prior_annotation"])
    annotations = {"prior_annotation": prior_annotation, "daphne": sources.load_confirmed(paths["confirmed"])}
    for source, table in annotations.items():
        for patient_id, records in table.items():
            for record in records:
                record["key"] = store.target_key(source, patient_id, record["hpo_code"],
                                                 record.get("slot"))
    return annotations


def load_corpus(paths: dict) -> dict:
    """``{"segments": {pid: [sentence, …]}, "texts": {pid: report}, "prior_annotation_texts": {…}}``.

    The reading surface, which is what makes *located* checkable at all: without the segments there
    is no segment for an annotation to sit in, and without the verbatim report the segments cannot
    be sliced back out of the text a ``char_offset`` indexes.

    ``texts`` is what the report is drawn from, PhenoBERT's staged copy where one was configured,
    the prior_annotation file's own column otherwise. ``prior_annotation_texts`` is kept separately because
    ``char_offset`` indexes *that* text and no other, and mapping an offset through the wrong
    alignment is the one error :mod:`hpo_extraction.curation.spans` exists to refuse.
    """
    segments = sources.load_segments(paths["segments"])
    _, prior_annotation_texts = sources.load_prior_annotation(paths["prior_annotation"])

    texts = dict(prior_annotation_texts)
    if paths.get("phenobert"):
        _, pb_texts = sources.load_phenobert(paths["phenobert"], list(segments))
        for patient_id, text in pb_texts.items():
            if text:
                texts[patient_id] = text

    missing = sorted(pid for pid in prior_annotation_texts if pid not in segments)
    if missing:
        logger.warning("%d report(s) carry annotations but have no segmentation (e.g. %s) — every "
                       "annotation on them is unanchorable and the evidence rule will drop it.",
                       len(missing), missing[0])
    logger.info("corpus: %d segmented report(s), %d verbatim report text(s)",
                len(segments), len([t for t in texts.values() if t]))
    return {"segments": segments, "texts": texts, "prior_annotation_texts": prior_annotation_texts}


def load_curation(curation_dir: str) -> dict:
    """The folded event log. A missing log folds to empty, not raising.

    An empty fold is a meaningful state, it says nothing has been curated yet, and it is what
    every ``n_suggestions`` and ``is_confirmed`` below reads as false. Refusing to run without a log
    would make the "how much has curation changed" question unanswerable at its own baseline.
    """
    path = os.path.join(curation_dir, store.EVENTS_FILE)
    if not os.path.isfile(path):
        logger.warning("no curation log at %s — folding to empty state", path)
        return {"rows": {}, "confirmed": set(), "patients": {}, "comments": {}, "n_skipped": 0}
    events = store.read_events(path)
    state = store.fold_events(events)
    logger.info("curation log: %d event(s) → %d row(s), %d confirmed report(s), %d skipped",
                len(events), len(state["rows"]), len(state["confirmed"]), state["n_skipped"])
    return state


# ──────────────────────────────────────────────────────────────────────────────
# evidence, the segment and the trigger word an annotation sits on
# ──────────────────────────────────────────────────────────────────────────────
def display_segments(patient_id: str, corpus: dict) -> list:
    """The segments as they read in the report, sliced from the verbatim text where it aligns.

    Identical to what ``registry._build_patient`` hands the reader, and it has to be: a curator's
    saved ``trigger_word`` is a verbatim slice of one of these strings (``common.snap_trigger``), so
    checking it against the raw tokenized segment instead would reject a trigger over a tab the
    screen never showed anybody.
    """
    segments = corpus.get("segments", {}).get(patient_id, [])
    text = corpus.get("texts", {}).get(patient_id, "")
    if not segments:
        return []
    if not text:
        return list(segments)
    return spans.segment_texts(text, segments, spans.align_segments(text, segments))


def curated_location(row: dict, display: list) -> "Anchor | None":
    """The placement a **curator** recorded for a row, or ``None``.

    Both halves or neither, which is ``reader._curated_anchor``'s rule and for its reason: a segment
    with no trigger word, or a trigger word the named sentence does not contain, is not an evidence location, it is the unplaced state with something typed into it.
    """
    if not row or not str(row.get("trigger_word") or "").strip():
        return None
    try:
        idx = int(row.get("segment_idx"))
    except (TypeError, ValueError):
        return None
    if not 0 <= idx < len(display):
        return None
    trigger = str(row["trigger_word"]).strip()
    span = locations.find_trigger(display[idx], trigger)
    return None if span is None else Anchor(idx, trigger, "curated")


def location_index(patient_id: str, records: list, rows: list, corpus: dict) -> dict:
    """``{key: Anchor}`` for every row of one patient that sits on a segment and a trigger word.

    Two inputs, and the order between them is the point. ``anchors.place_all`` reads the annotation
    **files** and never sees the curation log, so on its own it would report a row the curator
    located by hand as still unplaced, the same round trip ``reader.needs_anchor`` folds the log
    back in for. A curator's placement therefore wins outright. A file's claim only fills in where
    nobody has said otherwise.

    A record whose file named a sentence but whose trigger did not survive into it comes back from
    ``anchors.locate`` with ``start is None``. That is **not** an evidence location: the sentence is a real
    claim worth showing on screen, but this module needs the words, and a term whose trigger is not
    in the report cannot be matched back to it by anything downstream.
    """
    display = display_segments(patient_id, corpus)
    out: dict = {}
    if not display:
        return out

    for row in rows:
        anchor = curated_location(row, display)
        if anchor is not None:
            out[str(row.get("key") or "")] = anchor

    prior_annotation_text = corpus.get("prior_annotation_texts", {}).get(patient_id, "")
    text = corpus.get("texts", {}).get(patient_id, "")
    segments = corpus.get("segments", {}).get(patient_id, [])
    prior_annotation_ranges = None
    if prior_annotation_text:
        prior_annotation_ranges = spans.align_segments(prior_annotation_text, segments)

    placed, _ = locations.place_all(
        records, display,
        ranges_by_source={"prior_annotation": prior_annotation_ranges},
        texts_by_source={"prior_annotation": prior_annotation_text},
    )
    for idx, hits in placed.items():
        for hit in hits:
            if hit.get("start") is None:
                continue
            key = str(hit.get("key") or "")
            if key and key not in out:
                out[key] = Anchor(idx, str(hit.get("trigger_word") or ""),
                                  str(hit.get("how") or "lexical"))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The cohort
# ──────────────────────────────────────────────────────────────────────────────
def _rows_by_patient(fold: dict) -> dict:
    out: dict = {}
    for row in fold["rows"].values():
        out.setdefault(str(row.get("patient_id", "")), []).append(row)
    return out


def cohort(annotations: dict, fold: dict, criteria: tuple = DEFAULT_CRITERIA,
           corpus: dict = None, prior_gold_ids=None) -> list:
    """One :class:`ReportRow` per report any input mentions, ``in_cohort`` set by *criteria*.

    Every report is returned, including the excluded ones. The count of what was left out, and
    which signal each survivor came in on, is the first thing anyone reading these numbers has to
    be able to check, and a function that silently returned only the survivors would make
    "how much of HCY is this" unanswerable from its own output.

    *corpus* is required only by the ``segmented`` criterion, which is the one signal that is a
    property of the report, not of its annotation.

    *prior_gold_ids* is required only by ``original_gold``, and is the **key set** of the original
    ``hcy_ground_truth``, not its terms. A report it lists with an empty code list is a report
    that file holds to have no phenotype, which is a claim worth admitting. A report it omits is
    not.
    """
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"unknown cohort criteria {sorted(unknown)}; "
                         f"expected a subset of {list(CRITERIA)}")
    if "segmented" in criteria and not (corpus or {}).get("segments"):
        raise ValueError("the 'segmented' criterion needs the segmentation to know which reports "
                         "exist; pass corpus=load_corpus(paths)")
    if "original_gold" in criteria and prior_gold_ids is None:
        raise ValueError("the 'original_gold' criterion needs the original gold's report list; "
                         "pass prior_gold_ids=set(loaders.load_gold('hcy', hcy_gt_path=...)), "
                         "i.e. set hcy_gt_path")

    corpus = corpus or {}
    segments_by_patient = corpus.get("segments", {})
    prior_ids = set(prior_gold_ids or ())
    curated_rows = _rows_by_patient(fold)
    # Every report the segmentation knows about is *listed*, whether or not it is admitted. What
    # was left out is the first thing anyone reading a cohort-restricted number has to be able to
    # check, and a report nobody annotated is the interesting exclusion, invisible unless
    # The segmentation is what enumerates the cohort.
    #
    # The original ground truth's key set joins that enumeration for the same reason: a report it lists and
    # The segmentation does not is a gap worth seeing in the manifest, not one to drop silently.
    patient_ids = set(curated_rows) | set(segments_by_patient) | prior_ids
    for table in annotations.values():
        patient_ids |= set(table)
    patient_ids.discard("")

    reports: list = []
    for patient_id in sorted(patient_ids):
        rows = curated_rows.get(patient_id, [])
        n_prior_annotation = len(annotations.get("prior_annotation", {}).get(patient_id, []))
        n_daphne = len(annotations.get("daphne", {}).get(patient_id, []))
        n_suggestions = sum(1 for r in rows if r.get("source") == "new")
        n_adjudicated = sum(1 for r in rows if r.get("status") in store.IN_GOLD)
        is_confirmed = patient_id in fold["confirmed"]
        n_segments = len(segments_by_patient.get(patient_id, []))
        patient_labels = fold.get("patients", {}).get(patient_id, {})

        matched = []
        if "prior_annotation" in criteria and n_prior_annotation:
            matched.append("prior_annotation")
        if "daphne" in criteria and n_daphne:
            matched.append("daphne")
        if "suggestion" in criteria and n_suggestions:
            matched.append("suggestion")
        if "approved" in criteria and (is_confirmed or n_adjudicated):
            matched.append("approved")
        if "segmented" in criteria and n_segments:
            matched.append("segmented")
        if "original_gold" in criteria and patient_id in prior_ids:
            matched.append("original_gold")

        reports.append(ReportRow(
            patient_id=patient_id,
            in_cohort=bool(matched),
            n_prior_annotation=n_prior_annotation,
            n_daphne=n_daphne,
            n_suggestions=n_suggestions,
            n_adjudicated=n_adjudicated,
            is_confirmed=is_confirmed,
            reason="+".join(matched) if matched else "no annotation from any source",
            n_segments=n_segments,
            difficulty=str(patient_labels.get("difficulty") or ""),
            report_labels=tuple(patient_labels.get("labels") or ()),
            n_comments=len(fold.get("comments", {}).get(patient_id, [])),
        ))
    return reports


# ──────────────────────────────────────────────────────────────────────────────
# The ground truth
# ──────────────────────────────────────────────────────────────────────────────
def _all_records(annotations: dict, patient_id: str) -> list:
    """Every annotation both rich sources carry for one patient, whatever it duplicates."""
    return [dict(r) for source in sources.ADJUDICATION_ORDER
            for r in annotations.get(source, {}).get(patient_id, [])]


def _base_records(records: list, policy: GoldPolicy) -> list:
    """The existing annotations that are candidates for this patient's ground truth.

    ``sources.adjudicated`` is passed the sources that carry something **for this patient**, not the
    sources that were loaded. The distinction is the one ``registry.gold_sources_for`` documents: a
    patient with no row in the confirmed file is not a patient the two files disagree about, it is a
    patient that pass never reached, and adjudicating against an absent source would silently drop
    every prior_annotation annotation it has.
    """
    present = {r["source"] for r in records}
    if policy.prior_annotation_fallback == "daphne_first" and "daphne" in present:
        return [r for r in records if r["source"] == "daphne"]
    return sources.adjudicated(records, loaded=present)


def _decide(status: str, labels, anchored: bool, policy: GoldPolicy) -> tuple:
    """``(in_gold, why)`` for one candidate. The reason string is the audit trail.

    The order is deliberate: a qualifier is a statement about the annotation itself, a verdict
    against it is a curator's decision, and both outrank the mechanical question of whether the
    words could be found. An approved term with no findable trigger reports as ``no_evidence``, the honest reading, since the reason it is out has nothing to do with anybody's doubt about the
    phenotype.
    """
    excluded = [v for v in policy.exclude_labels if v in set(labels or ())]
    if excluded:
        return False, "label:" + ",".join(excluded)
    if status in OUT_STATUSES:
        return False, f"status:{status}"
    if policy.require_evidence and not anchored:
        return False, "no_evidence"
    if status in store.IN_GOLD:
        return True, f"status:{status}"
    if status == "suggested":
        return (policy.include_suggested,
                "suggested" if policy.include_suggested else "suggested:excluded")
    if not status:
        return (policy.include_undecided,
                "unruled" if policy.include_undecided else "unruled:excluded")
    # A status this version does not know, written by a newer curation app. Out, and named, so it
    # shows up as a line in the drop table, not as an unexplained shortfall.
    return False, f"status:unknown:{status}"


def _effective_code(row: dict, record: dict) -> str:
    """The code to score: the curation log's, where an ``edit`` rewrote it, otherwise the file's."""
    return sources.fix_hpo_id(str(row.get("hpo_code") or record.get("hpo_code") or "").strip())


def _segment_text(display: list, idx) -> str:
    try:
        return display[int(idx)]
    except (TypeError, ValueError, IndexError):
        return ""


def _best(location_by_code: dict, code: str, anchor) -> None:
    """Keep the highest-ranked evidence location seen for *code*, a curator's placement beats a file's."""
    if anchor is None or not code:
        return
    current = location_by_code.get(code)
    if current is None or anchor.rank < current.rank:
        location_by_code[code] = anchor


def build(annotations: dict, fold: dict, policy: GoldPolicy = GoldPolicy(),
          criteria: tuple = DEFAULT_CRITERIA, reports: list = None,
          corpus: dict = None, prior_gold_ids=None) -> CuratedGold:
    """The curated ground truth over the qualifying reports, plus every decision that produced it.

    *reports* lets a caller reuse one cohort across several policies, which is what
    :func:`variants` does, and what keeps the sensitivity table honest: six ground truth definitions over
    one fixed set of reports, so the only thing moving between its rows is the policy.

    *corpus* is :func:`load_corpus`'s output, and is required unless ``policy.require_evidence`` is
    off. Defaulting it to "no evidence anywhere" would silently empty the ground truth, which is the one
    failure mode a P/R table cannot show you.
    """
    if policy.require_evidence and not (corpus or {}).get("segments"):
        raise ValueError(
            "policy.require_evidence is on, which needs the segmentation to check a trigger word "
            "against its segment. Pass corpus=load_corpus(paths), or set require_evidence=False "
            "(the `keep_unanchored` variant) to score the code sets the files carry.")
    corpus = corpus or {"segments": {}, "texts": {}, "prior_annotation_texts": {}}

    if reports is None:
        reports = cohort(annotations, fold, criteria, corpus, prior_gold_ids=prior_gold_ids)
    included = {r.patient_id for r in reports if r.in_cohort}
    curated_rows = _rows_by_patient(fold)

    gold: dict = {pid: set() for pid in included}
    terms: list = []
    locations_by_patient: dict = {}

    for patient_id in sorted(included):
        rows = curated_rows.get(patient_id, [])
        rows_by_key = {r["key"]: r for r in rows}
        records = _all_records(annotations, patient_id)
        display = display_segments(patient_id, corpus)

        location_by_key = location_index(patient_id, records, rows, corpus)
        locations_by_patient[patient_id] = location_by_key

        # An annotation is one phenotype however many files recorded it, so the evidence question
        # is asked per **code** over every row carrying it. Holistic getting the words right and
        # daphne getting them wrong is one annotation that landed. Dropping the adjudicated row
        # because its own file's claim failed would lose a term the report demonstrably supports.
        location_by_code: dict = {}
        for record in records:
            _best(location_by_code, _effective_code(rows_by_key.get(record["key"], {}), record),
                  location_by_key.get(record["key"]))
        for row in rows:
            if row.get("source") == "new":
                _best(location_by_code, sources.fix_hpo_id(str(row.get("hpo_code") or "").strip()),
                      location_by_key.get(row["key"]))

        candidates: list = []

        # 1. every existing annotation the curation app would put on the Approve screen
        for record in _base_records(records, policy):
            row = rows_by_key.get(record["key"], {})
            # An ``edit`` event may have rewritten the code, a repair, not a verdict. The repaired
            # code is the one the curator means, so it is the one scored.
            code = _effective_code(row, record)
            anchor = location_by_key.get(record["key"]) or location_by_code.get(code)
            in_gold, why = _decide(str(row.get("status") or ""), row.get("labels"),
                                   anchor is not None, policy)
            idx = (anchor.segment_idx if anchor
                   else row.get("segment_idx", record.get("segment_idx")))
            candidates.append(TermRow(
                patient_id=patient_id, hpo_code=code,
                hpo_name=str(row.get("hpo_name") or record.get("hpo_name") or ""),
                source=record["source"], status=str(row.get("status") or ""),
                labels=tuple(row.get("labels") or ()), in_gold=in_gold, reason=why,
                key=record["key"],
                trigger_word=str((anchor.trigger_word if anchor else "")
                                 or row.get("trigger_word") or record.get("trigger_word") or ""),
                segment_idx=idx,
                segment_text=_segment_text(display, idx),
                anchored=anchor is not None,
                how=anchor.how if anchor else "",
                note=str(row.get("note") or ""),
                edited=bool(row.get("hpo_code")) and row["hpo_code"] != record["hpo_code"],
                original_hpo_code=str(record.get("hpo_code") or ""),
            ))

        # 2. every term proposed in this app. Keyed by event id, so it can never collide with (1).
        for row in rows:
            if row.get("source") != "new":
                continue
            code = sources.fix_hpo_id(str(row.get("hpo_code") or "").strip())
            if not code:
                continue
            anchor = location_by_key.get(row["key"]) or location_by_code.get(code)
            in_gold, why = _decide(str(row.get("status") or ""), row.get("labels"),
                                   anchor is not None, policy)
            idx = anchor.segment_idx if anchor else row.get("segment_idx")
            candidates.append(TermRow(
                patient_id=patient_id, hpo_code=code,
                hpo_name=str(row.get("hpo_name") or ""), source="new",
                status=str(row.get("status") or ""), labels=tuple(row.get("labels") or ()),
                in_gold=in_gold, reason=why, key=row["key"],
                trigger_word=str((anchor.trigger_word if anchor else "")
                                 or row.get("trigger_word") or ""),
                segment_idx=idx,
                segment_text=_segment_text(display, idx),
                anchored=anchor is not None,
                how=anchor.how if anchor else "",
                note=str(row.get("note") or ""),
                original_hpo_code="",
            ))

        terms.extend(candidates)
        gold[patient_id] = {t.hpo_code for t in candidates if t.in_gold and t.hpo_code}

    by_patient = {r.patient_id: r for r in reports}
    dropped: dict = {}
    unanchored: dict = {}
    for term in terms:
        if not term.in_gold:
            dropped[term.patient_id] = dropped.get(term.patient_id, 0) + 1
        if not term.anchored:
            unanchored[term.patient_id] = unanchored.get(term.patient_id, 0) + 1
    for patient_id, codes in gold.items():
        by_patient[patient_id].n_gold_terms = len(codes)
        by_patient[patient_id].n_dropped = dropped.get(patient_id, 0)
        by_patient[patient_id].n_unanchored = unanchored.get(patient_id, 0)

    result = CuratedGold(gold=gold, reports=reports, terms=terms, policy=policy,
                         criteria=tuple(criteria), locations=locations_by_patient)
    logger.info("curated gold: %d/%d report(s) in cohort, %d gold pair(s), %d term(s) dropped "
                "(%d of them for want of evidence)",
                len(gold), len(reports), result.n_pairs,
                sum(1 for t in terms if not t.in_gold),
                sum(1 for t in terms if not t.anchored))
    return result


def variants(annotations: dict, fold: dict, criteria: tuple = DEFAULT_CRITERIA,
             names=None, corpus: dict = None, prior_gold_ids=None) -> dict:
    """One :class:`CuratedGold` per entry of :data:`VARIANTS`, over **one** shared cohort."""
    reports = cohort(annotations, fold, criteria, corpus, prior_gold_ids=prior_gold_ids)
    names = list(names) if names is not None else list(VARIANTS)
    return {name: build(annotations, fold, VARIANTS[name], criteria,
                        reports=[replace(r) for r in reports], corpus=corpus)
            for name in names}


# ──────────────────────────────────────────────────────────────────────────────
# export
# ──────────────────────────────────────────────────────────────────────────────
def write_gold_csv(path, gold: dict) -> str:
    """The two-column ``patient_id,hpo_codes`` file every scorer in this repo already reads.

    Semicolon-joined and sorted, matching ``store.export_gold``'s shape, so the output is a drop-in
    for ``hcy_gt_path``, and for the result-table library's ``hcy_curated_gt_path``, which is the route to the
    full thesis metric set over this ground truth.

    Only cohort members get a line. A report that failed the filter is **absent**, not present with
    an empty cell: an empty cell is a real report with no phenotypes, and conflating the two would
    hand every method free precision on the uncurated remainder.

    The converse is deliberate too. A cohort member whose ground truth came out empty **does** get a line,
    with an empty ``hpo_codes`` cell, that is the whole point of the ``original_gold`` and
    ``segmented`` criteria, which admit reports in order to charge a method for every
    term it emits there.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.write_csv_atomic(
        str(path), ["patient_id", "hpo_codes"],
        [{"patient_id": pid, "hpo_codes": ";".join(sorted(gold[pid]))} for pid in sorted(gold)],
    )
    return path.name
