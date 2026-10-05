"""The ground truth standard, and the one rule both cohorts obey: a drawn span is a recorded span.

Two cohorts, two ground truth standards, and they differ in what they can tell you about *where* a
phenotype was read -- which is the whole reason the reader tab looks the way it does.

**HCY** is a curated dataset. Every annotation carries the segment the curator was reading and the
trigger word inside it, so ``CuratedGold.anchors`` can locate the word *within that one segment*
and the underline the reader sees is a person's own reading. The confinement to one segment is
what makes it evidence rather than a guess, and it is imported from ``apps/treephenorag_ui/curated.py``,
not reimplemented here for that reason.

**GSC+** under RAG-HPO's re-annotation carries ``(doc_id, hpo_id, hpo_description)`` and nothing
else -- no offsets, no sentence, no trigger word (see ``resources/data/GSC_RAGHPO/PROVENANCE.md``).
Placing those terms takes a ladder, and the rungs are **not** merged, because they are different
kinds of claim:

1. The GSC+ corpus's own annotation of the same code on the same document, which is a
   mention-level span a human recorded, checked against the characters it claims to spell;
2. failing that, RAG-HPO's own ``hpo_description`` occurring verbatim in the abstract -- weaker,
   but still their wording, not ours, and badged ``description match``;
3. failing that, **nothing**. The term goes in the grid's footer, where the reader can see it
   exists and that nobody said where.

Roughly 11% of RAG-HPO's pairs are terms the GSC+ corpus does not annotate on that document, so
rung 1 is unavailable for a real and interesting minority. Those carry the ``raghpo-only``
qualifier, which is the same channel HCY's curation qualifiers use.

Both sources return the same two dicts -- ``({resolved: row}, {resolved: row})``, placed and
unplaced -- so :mod:`apps.compare_ui.build` has one code path and cannot grow a second definition
of what an annotated term is.
"""

from __future__ import annotations

import csv
import logging
import os
import re

from apps.compare_ui.methods import base

logger = logging.getLogger(__name__)

_HPO_RE = re.compile(r"HP:\d{7}")

#: Filenames inside a ``resources/data/GSC_RAGHPO`` directory.
RAGHPO_ANNOTATIONS = "annotations.csv"
RAGHPO_IDS = "document_ids.txt"

#: The qualifier a term gets when the GSC+ corpus does not annotate it on this document. It is a
#: property of the ground truth, not of any method, and it is the single most useful thing to know when a
#: term everybody missed turns out to be one only RAG-HPO's annotators saw.
RAGHPO_ONLY = "raghpo-only"


def row(hpo_id: str, gold_code: str, **kwargs) -> dict:
    """One ground truth row in the bundle's shape. Every key is present even when empty.

    Present-but-empty, not absent: the views read these by key, and a schema that is
    sometimes missing a field is one the renderer has to defend against at every use.
    """
    out = {"hpo_id": hpo_id, "gold_code": gold_code, "segment_idx": None, "span": None,
           "trigger": "", "qualifiers": [], "source": "", "note": ""}
    out.update(kwargs)
    return out


# -- HCY: the curated dataset ------------------------------------------------

class CuratedGold:
    """The HCY curated ground truth, as ``app/exp13_ui`` reads it.

    A thin wrapper and so: the placement ladder, the trigger locating and the
    exclusion policy all live in ``exp13_ui.curated`` and took two passes to get right there. What
    this adds is the bundle's row shape and nothing else.
    """

    kind = "curated"

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def name(self) -> str:
        """Display name of this ground truth."""
        return getattr(self.dataset, "name", "")

    @property
    def path(self) -> str:
        """Path of the ground-truth file."""
        return getattr(self.dataset, "gold_path", "")

    def codes(self, report_id: str) -> set:
        """HPO identifiers annotated in *report_id*."""
        return set(self.dataset.gold.get(report_id) or ())

    def rows(self, rv, resolve):
        """``(placed, unplaced)`` for one report.

        ``anchors`` is given the **display** strings, matching ``exp13_ui``'s own deep dive: the
        spans it returns must index the text the reader sees, not the tokenized sentence.
        """
        raw = self.codes(rv.report_id)
        marks = self.dataset.placements(rv.report_id, rv.sentences)
        locations = self.dataset.locations(rv.report_id, rv.display, marks)

        where = {}
        for idx, rows in locations.items():
            for start, end, annotation in rows:
                where.setdefault(annotation.hpo_code, (idx, [start, end]))
        for idx, rows in marks.items():
            for annotation in rows:
                where.setdefault(annotation.hpo_code, (idx, None))

        placed, unplaced = {}, {}
        for code in sorted(raw):
            annotation = self.dataset.find(rv.report_id, code)
            entry = row(
                resolve(code), code,
                trigger=getattr(annotation, "trigger_word", "") or "",
                qualifiers=list(getattr(annotation, "qualifiers", ()) or ()),
                source=getattr(annotation, "source", "") or "",
                note=getattr(annotation, "note", "") or "")
            found = where.get(code)
            if found is not None:
                entry["segment_idx"], entry["span"] = found[0], found[1]
                placed[entry["hpo_id"]] = entry
            else:
                unplaced[entry["hpo_id"]] = entry
        return placed, unplaced


# -- GSC+: RAG-HPO's re-annotation, located against the corpus's own --------

class RagHpoGold:
    """RAG-HPO's 114-document annotation, placed by the ladder in the module docstring.

    The corpus annotation is read for **placement only**. It is never mixed into the ground truth: the
    cohort this app reads is ``gsc_raghpo_ann``, whose membership is RAG-HPO's file and nobody
    else's, and a term that exists in one and not the other is the case worth seeing.
    """

    kind = "raghpo"

    def __init__(self, raghpo_dir: str, gsc_dir: str = ""):
        self.raghpo_dir = raghpo_dir
        self.gsc_dir = gsc_dir
        self.annotations = read_raghpo_annotations(
            os.path.join(raghpo_dir, RAGHPO_ANNOTATIONS))
        self._corpus: dict = {}

    @property
    def name(self) -> str:
        """Display name of this ground truth."""
        return "gsc_raghpo_ann (RAG-HPO's own annotation of their 114 GSC+ documents)"

    @property
    def path(self) -> str:
        """Path of the ground-truth file."""
        return os.path.join(self.raghpo_dir, RAGHPO_ANNOTATIONS)

    @property
    def gold(self) -> dict:
        """``{doc_id: {hpo_id}}`` -- the same attribute name ``CuratedGold``'s dataset exposes."""
        return {doc: set(terms) for doc, terms in self.annotations.items()}

    def document_ids(self) -> list:
        """Documents in the ground truth, sorted."""
        path = os.path.join(self.raghpo_dir, RAGHPO_IDS)
        if not os.path.isfile(path):
            return sorted(self.annotations)
        with open(path, encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def codes(self, report_id: str) -> set:
        """HPO identifiers annotated in *report_id*."""
        return set(self.annotations.get(report_id) or {})

    def corpus_annotation(self, report_id: str) -> dict:
        """``{hpo_id: [(start, end, mention), ...]}`` from ``GSC_2024/Annotations/<doc_id>``.

        Memoised per document. A missing file, an unreadable line and a malformed code all degrade
        to "no annotation for this document", which costs an underline and never a report.
        """
        if report_id in self._corpus:
            return self._corpus[report_id]
        out: dict = {}
        path = os.path.join(self.gsc_dir or "", "Annotations", str(report_id))
        if self.gsc_dir and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError as exc:                            # pragma: no cover - defensive
                logger.warning("could not read %s: %s", path, exc)
                text = ""
            for line in text.splitlines():
                parts = line.split("\t")
                if len(parts) < 2 or not _HPO_RE.fullmatch(parts[1].strip()):
                    continue
                offsets = parts[0].split(":")
                try:
                    start, end = int(offsets[0]), int(offsets[1])
                except (IndexError, ValueError):
                    continue
                mention = parts[2].strip() if len(parts) > 2 else ""
                out.setdefault(parts[1].strip(), []).append((start, end, mention))
        self._corpus[report_id] = out
        return out

    def rows(self, rv, resolve):
        """``(placed, unplaced)`` for one abstract."""
        described = self.annotations.get(rv.report_id) or {}
        corpus = self.corpus_annotation(rv.report_id)

        placed, unplaced = {}, {}
        for code in sorted(described):
            description = described[code]
            where, trigger, source = self._locate(rv, code, description, corpus)
            qualifiers = [] if code in corpus else [RAGHPO_ONLY]
            entry = row(
                resolve(code), code,
                trigger=trigger,
                qualifiers=qualifiers,
                source=source,
                note=("RAG-HPO annotate this term on this document and the GSC+ corpus does not."
                      if qualifiers else ""))
            if where is not None:
                entry["segment_idx"], entry["span"] = where
                placed[entry["hpo_id"]] = entry
            else:
                unplaced[entry["hpo_id"]] = entry
        return placed, unplaced

    def _locate(self, rv, code: str, description: str, corpus: dict):
        """``((segment_idx, span) | None, trigger, source)`` -- the ladder, rung by rung.

        The corpus's offsets are **checked** before they are trusted, the same way
        ``methods/phenobert`` checks a detection's: offsets that do not spell the mention they
        claim were measured against a different staging of the document, and underlining them
        would put an annotator's name on the wrong three words.
        """
        for start, end, mention in corpus.get(code) or ():
            if mention and not base.span_spells(rv.text, start, end, mention):
                idx, span = base.locate_phrase(rv, mention)
                if idx is not None:
                    return (idx, span), mention, "gsc+ annotation (located by mention)"
                continue
            local = rv.local(start, end)
            if local is not None:
                idx, s, e = local
                return (idx, [s, e]), mention or rv.display[idx][s:e], "gsc+ annotation"

        if description:
            idx, span = base.locate_phrase(rv, description)
            if idx is not None:
                return (idx, span), description, "raghpo description match"

        return None, description, "raghpo annotation"


def read_raghpo_annotations(path: str) -> dict:
    """``{doc_id: {hpo_id: hpo_description}}`` from ``annotations.csv``.

    Also the selection script's reader (``apps/compare_ui/select_gsc_documents.py``), so the cells a
    document was drawn into and the ground truth it is then scored against cannot come from two different
    parses of the same file.
    """
    out: dict = {}
    if not os.path.isfile(path):
        logger.warning("no RAG-HPO annotations at %s", path)
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle):
            doc = str(entry.get("doc_id") or "").strip()
            code = str(entry.get("hpo_id") or "").strip()
            if not doc or not _HPO_RE.fullmatch(code):
                continue
            out.setdefault(doc, {}).setdefault(
                code, str(entry.get("hpo_description") or "").strip())
    return out
