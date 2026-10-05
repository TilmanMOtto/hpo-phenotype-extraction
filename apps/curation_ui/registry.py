"""Everything the screen asks questions of, held in this process.

The app has two kinds of state and they age differently, so the caching is split along that seam:

**Static per patient**, the segments, the verbatim report, the segment↔report alignment, the
PhenoBERT detections placed on segments, every annotation source's records placed on segments, and
the lexical trigger candidates for the codes that still have no evidence. None of it changes while
the app runs, all of it costs real work (two alignments and a handful of regex sweeps per patient),
so it is built on first visit and kept in a small LRU. Sixteen patients is enough that paging back
and forth through a cohort never rebuilds.

**Dynamic**, the folded curation state. It changes on every click, so it is not in that cache. It
is folded once and re-folded when the log grows. Folding a few thousand events is sub-millisecond,
and the alternative, patching the fold in place, is how a derived view and its log drift apart.

**Three annotation sources, one shape.** ``prior_annotation`` and ``daphne`` carry a trigger word and a
position, so their annotations are *drawn on the report*; ``prior_annotation_2`` is code-only, so its terms go to
:mod:`locate` for a candidate origin instead. Which is which is data (``sources.GOLD_SOURCES``),
not a branch in every panel, a fourth source is a row in that tuple.

They are also not equals. ``view["existing"]`` is the subset Approve mode rules on, per code, the
best source in ``sources.ADJUDICATION_ORDER`` that has one, and ``view["records"]`` is everything,
which is what the report and the document panel draw. Keeping both on the view is deliberate: the
screen shows more than it asks about, and a panel that conflated the two would either hide a
cross-check or ask for a verdict on something that can never enter the ground truth.

The ``HPOTree`` is shared with the other analysis apps through ``apps.ui_common.registry.get_tree``:
its constructor parses a 15.5 MB JSON, which is not something to do per app, per patient, or, as ``hpo_extraction.treephenorag.verifier_prompt`` does, per prompt.

Nothing here is JSON-serialisable and nothing here is meant to be. Dash callbacks carry a patient
id and a mode. The objects stay on this side of the tunnel.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

from . import locate
from hpo_extraction.curation import evidence_location as locations
from hpo_extraction.curation import sources
from hpo_extraction.curation import spans
from hpo_extraction.curation import store
from .search import get_search

logger = logging.getLogger(__name__)

_PATIENT_LRU = 16

#: Where the curation log and its derived files live, under the hcy directory.
CURATION_SUBDIR = "curation"


def get_tree():
    """The shared ontology. Delegates to ``app/tree_ui`` so every app pays the parse once."""
    from apps.ui_common.registry import get_tree as _get_tree

    return _get_tree()


class Registry:
    """One instance per path set, swapped when the sidebar is repointed."""

    def __init__(self, paths: dict, author: str = "unknown", curation_dir: str | None = None):
        self.paths = dict(paths)
        self.author = author
        self.curation_dir = curation_dir or os.path.join(
            self.paths.get("hcy_dir", "."), CURATION_SUBDIR
        )

        self.segments: dict[str, list[str]] = sources.load_segments(self.paths["segments"])

        # One loader per source, one shape out of all of them. ``holistic`` also yields the report
        # text its offsets index, which doubles as a verbatim report for a patient PhenoBERT was
        # never run on, so those reports read as prose instead of as tokenized sentences.
        prior_annotation, self.report_texts = sources.load_prior_annotation(self.paths.get("prior_annotation", ""))
        self.annotations: dict[str, dict[str, list[dict]]] = {
            "prior_annotation": prior_annotation,
            "daphne": sources.load_confirmed(self.paths.get("confirmed", "")),
            "prior_annotation_2": sources.load_code_only(self.paths.get("prior_annotation_2", ""), "prior_annotation_2"),
        }
        for source, table in self.annotations.items():
            for patient_id, records in table.items():
                for record in records:
                    record["key"] = store.target_key(source, patient_id, record["hpo_code"],
                                                     record.get("slot"))

        #: The sources that actually carry something. A source whose file is missing is left out
        #: rather than listed as empty: "every source agrees" must not be satisfiable by a file
        #: that was never read.
        self.gold_sources: list[str] = [
            spec["id"] for spec in sources.GOLD_SOURCES if self.annotations.get(spec["id"])
        ]

        # Codes per source, in file order, what the panels that count terms, not read them
        # still want, and what ``agreement`` compares.
        self.gold: dict[str, dict[str, list[str]]] = {
            source: {
                patient_id: list(dict.fromkeys(r["hpo_code"] for r in records))
                for patient_id, records in table.items()
            }
            for source, table in self.annotations.items()
        }

        self.detections, self.texts = sources.load_phenobert(
            self.paths.get("phenobert", ""), self.segments.keys()
        )

        # Every patient any source knows about, so a report present only in one annotation file
        # still gets a row and can be seen to have no segmentation, not vanishing.
        ids = set(self.segments) | set(self.detections) | set(self.report_texts)
        for table in self.annotations.values():
            ids |= set(table)
        self.patient_ids: list[str] = sorted(ids)

        self.tree = get_tree()
        self.search = get_search(self.tree)
        self.log = store.EventLog(self.curation_dir, author=author)

        self._patients: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._state: dict | None = None
        self._state_at = -1

        logger.info(
            "Registry: %d patients | %s | PhenoBERT %d reports | log %d events",
            len(self.patient_ids),
            " / ".join(f"{source} {sum(len(v) for v in table.values())} ann"
                       for source, table in self.annotations.items()),
            len(self.detections), len(self.log.events),
        )

    # ── validation ───────────────────────────────────────────────────────────
    def validate(self) -> list[str]:
        """Problems worth refusing to swap the registry for. Missing PhenoBERT is not one."""
        problems = []
        if not self.patient_ids:
            problems.append("No patients found in any source.")
        if not self.segments:
            problems.append(f"No segments read from {self.paths['segments']}.")
        if not self.gold_sources:
            problems.append("No annotation source could be read.")
        # Checked here, not left to the first click. Every other problem in this list is
        # "there is nothing to show you". This one is "you can read everything and record nothing",
        # which is the failure a curator would otherwise discover after forming a verdict.
        unwritable = store.writability_problem(self.curation_dir)
        if unwritable:
            problems.append(unwritable)
        return problems

    def notes(self) -> list[str]:
        """Things the sidebar should say out loud but that are not failures."""
        notes = []
        missing = [spec["label"] for spec in sources.GOLD_SOURCES
                   if spec["id"] not in self.gold_sources]
        if missing:
            notes.append(f"Not loaded: {', '.join(missing)}.")
        if not self.detections:
            notes.append("No PhenoBERT run loaded, that column will be empty.")
        missing_segments = [p for p in self.patient_ids if not self.segments.get(p)]
        if missing_segments:
            notes.append(f"{len(missing_segments)} patient(s) have no segmented report.")
        return notes

    # ── curation state ───────────────────────────────────────────────────────
    def state(self) -> dict:
        """The folded log. Re-folded only when the log has grown since the last fold.

        ``refresh`` comes first because "grown" has to mean grown *on disk*: several curators share
        one curation directory, and without it this cache would only ever notice our own clicks.
        """
        with self._lock:
            self.log.refresh()
            if self._state is None or self._state_at != len(self.log.events):
                self._state = self.log.fold()
                self._state_at = len(self.log.events)
            return self._state

    def rows_by_patient(self) -> dict[str, list[dict]]:
        """``{patient_id: [row, …]}`` from the folded state, ordered for display and for the CSV."""
        out: dict[str, list[dict]] = {}
        for row in self.state()["rows"].values():
            out.setdefault(row["patient_id"], []).append(row)
        for rows in out.values():
            rows.sort(key=lambda r: (
                str(r.get("source", "")),
                str(r.get("hpo_code", "")),
                _as_int(r.get("segment_idx")),
            ))
        return out

    def rows_for(self, patient_id: str) -> list[dict]:
        """The current annotation rows of one report, after applying the event log."""
        return self.rows_by_patient().get(patient_id, [])

    def gold_sources_for(self, patient_id: str) -> list[str]:
        """The loaded annotation sources that carry something **for this patient**.

        :attr:`gold_sources` answers "which files were read", which is the right set for the
        sidebar and the wrong one for the word *disagree*. A patient with no row in
        ``annotations_confirmed.csv`` is not a patient the files disagree about, it is a patient
        that file never reached, and comparing against a source with nothing to say made every
        term on that screen carry a disagreement chip, put an ``≠`` beside the patient in the
        sidebar, and counted it in the Overview queue. That is the configuration being reported as
        if it were a finding about the annotation.

        Narrowing to the sources actually present makes the comparison the real one: prior_annotation
        against prior_annotation_2, and a row that reads ``prior_annotation + prior_annotation_2`` or ``prior_annotation only``, not a
        blanket ``disagree``. With a single source present there is nothing to disagree about, which
        is the same rule :func:`common.agreement` already applies to a single *loaded* source.
        """
        return [source for source in self.gold_sources
                if self.annotations.get(source, {}).get(patient_id)]

    def is_confirmed(self, patient_id: str) -> bool:
        """True when the curator marked the report as confirmed."""
        return patient_id in self.state()["confirmed"]

    def patient_labels(self, patient_id: str) -> dict:
        """``{"labels": [...], "difficulty": "", "note": ""}`` for a report, empty if unlabelled."""
        entry = self.state()["patients"].get(patient_id) or {}
        return {
            "labels": list(entry.get("labels") or ()),
            "difficulty": entry.get("difficulty", ""),
            "note": entry.get("note", ""),
        }

    def comments_for(self, patient_id: str) -> list[dict]:
        """Every comment on a report, oldest first. Empty list when nobody has said anything."""
        return list(self.state()["comments"].get(patient_id) or ())

    def write_labels(self) -> dict:
        """Rewrite the label and comment files. Called from the Overview tab's button."""
        annotation = store.write_labels(self.log.labels_path, self.rows_by_patient())
        report = store.write_report_labels(self.log.report_labels_path,
                                           self.state()["patients"], self.patient_ids,
                                           self.state()["comments"])
        comments = store.write_comments(self.log.comments_path, self.state()["comments"])
        return {"annotation": annotation, "report": report, "comments": comments}

    # ── mutation ─────────────────────────────────────────────────────────────
    def record(self, action: str, patient_id: str, **fields) -> dict:
        """Append one event, then rewrite the derived CSV. The only path that changes anything.

        The CSV rewrite is not deferred and not batched: it is a few milliseconds, and a curator
        who alt-tabs to check the file must find it agreeing with the screen. Deferring it is how a
        "saved" indicator starts lying.
        """
        event = self.log.append(action, patient_id, **fields)
        self.log.commit(self.rows_by_patient(), self.state()["patients"], self.patient_ids,
                        self.state()["comments"])
        return event

    def export_gold(self) -> dict:
        """Write the two-column curated ground truth. Called from the Overview tab's button."""
        return store.export_gold(self.log.export_path, self.rows_by_patient(), self.patient_ids)

    # ── per-patient view model ───────────────────────────────────────────────
    def patient(self, patient_id: str) -> dict | None:
        """The static half of a patient's screen, LRU-cached. ``None`` for an unknown id."""
        if patient_id not in self.patient_ids:
            return None
        with self._lock:
            cached = self._patients.get(patient_id)
            if cached is not None:
                self._patients.move_to_end(patient_id)
                return cached
        view = self._build_patient(patient_id)
        with self._lock:
            self._patients[patient_id] = view
            self._patients.move_to_end(patient_id)
            while len(self._patients) > _PATIENT_LRU:
                self._patients.popitem(last=False)
        return view

    def records_for(self, patient_id: str) -> list[dict]:
        """Every annotation every source carries for one patient, in source order then file order."""
        out: list[dict] = []
        for spec in sources.GOLD_SOURCES:
            out.extend(self.annotations.get(spec["id"], {}).get(patient_id, []))
        return out

    def _build_patient(self, patient_id: str) -> dict:
        segments = self.segments.get(patient_id, [])

        # The verbatim report, in preference order. PhenoBERT's staged copy first, because its
        # detection offsets index *that* text and nothing else. The holistic file's own copy
        # otherwise, which is what lets a patient the PhenoBERT baseline never saw still read as a document.
        pb_text = self.texts.get(patient_id, "")
        prior_annotation_text = self.report_texts.get(patient_id, "")
        text = pb_text or prior_annotation_text
        detections = self.detections.get(patient_id, [])

        if text:
            ranges = spans.align_segments(text, segments)
            alignment = spans.alignment_stats(ranges)
            display = spans.segment_texts(text, segments, ranges)
        else:
            ranges = [None] * len(segments)
            alignment = {"n_segments": len(segments), "n_aligned": 0,
                         "n_unaligned": len(segments)}
            display = list(segments)

        # Detections are placed only against the text they were produced from. Falling back to the
        # holistic report here would put PhenoBERT's offsets in a coordinate system nobody
        # promised they index, which is the one thing ``spans`` exists to refuse.
        if pb_text:
            placed, unplaced = spans.place_detections(ranges, detections)
        else:
            placed, unplaced = {}, list(detections)

        records = self.records_for(patient_id)
        # What Approve mode rules on: per code, the best source that has one. Everything else stays
        # on the screen as reference, see ``sources.adjudicated``.
        existing = sources.adjudicated(records, self.gold_sources)
        # ``char_offset`` indexes the holistic file's own report text, so it needs that text's own
        # alignment, which is the one already computed when it *is* the text being drawn.
        prior_annotation_ranges = (
            ranges if prior_annotation_text and prior_annotation_text == text
            else (spans.align_segments(prior_annotation_text, segments) if prior_annotation_text else None)
        )
        annotations, unplaced_annotations = locations.place_all(
            records, display,
            ranges_by_source={"prior_annotation": prior_annotation_ranges},
            texts_by_source={"prior_annotation": prior_annotation_text},
        )

        gold_codes = {source: list(table.get(patient_id, []))
                      for source, table in self.gold.items()}
        all_codes: list[str] = []
        for source in sources.GOLD_SOURCE_IDS:
            for code in gold_codes.get(source, []):
                if code not in all_codes:
                    all_codes.append(code)

        # Only the codes nothing could be *drawn* for. A lexical guess beside an annotation already
        # underlined on its own words is noise competing with evidence, but a code whose file named
        # a trigger that occurs nowhere in the report is the case a candidate helps with, so
        # this keys on what was placed, not on what was claimed.
        drawn = {record["hpo_code"]
                 for rows in annotations.values() for record in rows
                 if record.get("start") is not None}
        triggers = locate.find_all(self.tree, [c for c in all_codes if c not in drawn],
                                   display or segments)

        return {
            "patient_id": patient_id,
            "segments": segments,
            "display": display,
            "text": text,
            "text_source": "phenobert" if pb_text else ("prior_annotation" if prior_annotation_text else ""),
            # Where each segment sits in the verbatim report. The reader needs it to put the text
            # *between* segments back, headings, blank lines, the paragraph breaks that make a
            # report readable, which the segmentation drops on the floor.
            "ranges": ranges,
            "alignment": alignment,
            "detections": placed,
            "unplaced": unplaced,
            "n_detections": len(detections),
            "records": records,
            "existing": existing,
            "existing_keys": {record["key"] for record in existing},
            "annotations": annotations,
            "unplaced_annotations": unplaced_annotations,
            "gold": gold_codes,
            "gold_codes": all_codes,
            "triggers": triggers,
        }

    # ── cohort overview ──────────────────────────────────────────────────────
    def overview(self) -> list[dict]:
        """One row per patient: source counts, decision counts, confirmed flag.

        cheap, counts off the code dicts and the folded rows, never a patient build.
        Opening the overview must not warm sixteen patients into the LRU and evict the one being
        worked on.
        """
        rows_by_patient = self.rows_by_patient()
        confirmed = self.state()["confirmed"]
        out = []
        for patient_id in self.patient_ids:
            rows = rows_by_patient.get(patient_id, [])
            # Restricted to the sources that carry something for *this* patient, see
            # ``gold_sources_for``. A file with nothing to say about a report is not a file that
            # disagrees with the ones that do.
            per_source = {source: set(self.gold[source].get(patient_id, []))
                          for source in self.gold_sources_for(patient_id)}
            # Cheap on purpose, off the record lists, with no alignment and no regex, so opening
            # The overview still cannot warm sixteen patients into the LRU.
            n_existing = len(sources.adjudicated(self.records_for(patient_id), self.gold_sources))
            # Kept as ``gold_sources`` and not the per-patient set on purpose: adjudication order
            # is about which file *wins* a code it carries, which the absence of another file does
            # not change.
            union: set[str] = set().union(*per_source.values()) if per_source else set()
            agreed = set.intersection(*per_source.values()) if per_source else set()
            statuses = [r.get("status", "") for r in rows]
            patient_labels = self.patient_labels(patient_id)
            out.append({
                "patient_id": patient_id,
                "n_segments": len(self.segments.get(patient_id, [])),
                **{f"n_{source}": len(codes) for source, codes in per_source.items()},
                "n_terms": len(union),
                "n_existing": n_existing,
                # Terms not every loaded source carries, the queue this pass works through. With
                # one source loaded there is nothing to disagree about, and it reads as zero.
                "n_agree": len(agreed),
                "n_disagree": len(union) - len(agreed) if len(per_source) > 1 else 0,
                "n_phenobert": len(self.detections.get(patient_id, [])),
                "n_suggested": statuses.count("suggested"),
                "n_decided": sum(1 for s in statuses if s and s != "suggested"),
                "n_in_gold": sum(1 for s in statuses if s in store.IN_GOLD),
                "n_labelled": sum(1 for r in rows if r.get("labels")),
                "n_delete_suggested": statuses.count(store.DELETE_SUGGESTED),
                "n_comments": len(self.comments_for(patient_id)),
                "difficulty": patient_labels["difficulty"],
                "n_doc_labels": len(patient_labels["labels"]),
                "confirmed": patient_id in confirmed,
            })
        return out

    def totals(self) -> dict:
        """Cohort-level counts for the sidebar and the overview header."""
        rows = self.overview()
        return {
            "n_patients": len(rows),
            "n_confirmed": sum(1 for r in rows if r["confirmed"]),
            "n_open": sum(r["n_suggested"] for r in rows),
            "n_in_gold": sum(r["n_in_gold"] for r in rows),
            "n_labelled": sum(r["n_labelled"] for r in rows),
            "n_graded_reports": sum(1 for r in rows if r["difficulty"]),
            "n_delete_suggested": sum(r["n_delete_suggested"] for r in rows),
            "n_comments": sum(r["n_comments"] for r in rows),
            "n_events": len(self.log.events),
        }


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
