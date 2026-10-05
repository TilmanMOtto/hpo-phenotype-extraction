"""What the SLM actually saw at a node, recovered, because earlier did not persist it.

``*_calls.jsonl`` records ``sent_index`` and the logits, and nothing else: no sentence text, no
prompt. That is a reasonable choice for an artifact written once per retrieved sentence per
visited node per report, and it is fatal for a deep-dive, where the only question that counts at
a node the model got wrong is *what did it read*.

Both are recoverable, because the pipeline is deterministic:

**The sentence.** ``tree_experiment.py:186`` computes its segmentation as
``split_sents(segment_dict(reports, load_stanza(stanza_dir, "TOKENIZER")))``. Running the same
three calls over the same report yields the same list, and ``sent_index`` indexes it. The
transcription is one line long and it is fixed by :meth:`Evidence.assert_segmentation_matches`.

**The prompt.** ``hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1`` builds it from the term's name,
definition and synonyms. It is transcribed here only because the original constructs a fresh
``HPOTree`` on every call, 15 MB of JSON parsed per prompt, which a UI cannot pay, and
:meth:`Evidence.assert_prompt_matches` pins the transcription to the original string for string.

Running Stanza is the *last* of five routes to the segmentation, not the first, because it is the
one that fails. It drags in torch, and on a login node or a WSL workstation that import dies with
things like ``libcusparseLt.so.0: failed to map segment from shared object``, leaving a deep-dive
that shows sentence indices and logits and no text, which is the one thing it exists to show. The
order is:

``memory``      already segmented in this process.
``cache``       :data:`SEGMENT_CACHE_VERSION`-tagged JSON in the registry's cache directory, keyed
                by a digest of the report texts themselves. Written by whichever route succeeded,
                so the expensive one is paid at most once per machine per cohort.
``csv``         ``<hcy_dir>/segmented_reports.csv``. ``experiments/03_setup/segment_reports.py`` writes it with
                ``split_sents(segment_dict(load_txt(...), load_stanza(...)))``, the *same three
                calls* ``tree_experiment.py:183-184`` makes, so this file is persisted Stanza
                output and its ``sentence_idx`` **is** ``sent_index``. It is also the coordinate
                system the curated ground truth is located in (``hcy_ground_truth``'s ``segment_idx`` is an index
                into this file), so reading it is what makes a curator's segment and this app's
                sentence the same object. Read through
                ``hpo_extraction.curation.sources.load_segments`` rather than reimplemented, so the
                curation app and this one can never disagree about what the file says.
``artifact``    ``phenojury_generation_free_listing/<cohort>/llm_extractions_*.jsonl``. The ensemble driver
                segments with the *same* call, ``split_sents(segment_dict(...))``,
                ``slm_ensemble_experiment.py:131-132``, and, unlike the tree driver, it writes the
                sentence text beside its ``sentence_number``. So a cohort that has been through
                the Free Listing generation run carries its own segmentation, and no tokenizer is needed to read it.
``stanza``      the original three calls.

The csv and artifact routes are *verified before they are used*, per report, because a segmentation
that is subtly wrong is worse than none: indices would still resolve and every sentence shown would
be the wrong one. :func:`aligns_to_text` is the check, the sentences must occur in the report text
in order, if possible and whitespace-insensitively otherwise, because Stanza tokenization is
not guaranteed to preserve the source bytes (see ``src/hpo_extraction/curation/spans.py``). The artifact
route additionally requires the sentence numbers to be a complete ``0..n-1`` run. A report that
fails is dropped, not shown, and the count of dropped reports is in :meth:`source`.

The one case with no verification is a cohort whose report texts are not reachable at all. The csv
is then still served, a deep-dive with sentences beats one without, but :meth:`source` says
**UNVERIFIED** in those words, and nothing is written to the cache, whose key is a digest of the
texts that do not exist.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from collections import OrderedDict

log = logging.getLogger(__name__)

#: The Free Listing generation run's run directory and its per-model generation dumps, the one earlier artifact family
#: that persists sentence *text* against the sentence numbers the tree runs index by.
ENSEMBLE_EXP_ID = "phenojury_generation_free_listing"
EXTRACTION_GLOB = "llm_extractions_*.jsonl"

#: The persisted Stanza segmentation, named here so the message that asks for it can name it.
#: Fixed to ``hpo_extraction.curation.sources.SEGMENTS_FILE`` by the tests, the two apps must ask for
#: The same file, or a curator's segment 4 and this app's sentence 4 stop being the same sentence.
SEGMENTS_FILE = "segmented_reports.csv"

#: Bumped when the cached shape changes, **or when a route that feeds it changes**. A stale cache
#: that still parses is worse than one that does not: the deep-dive would show sentences from a
#: segmentation nothing on disk agrees with. v2 added the ``segmented_reports.csv`` route.
SEGMENT_CACHE_VERSION = 2

#: How many per-report alignments stay resident. A report stepper walks a cohort one report at a
#: time and comes back to the ones it has just seen. The structure is a list of char ranges, so
#: keeping a dozen costs nothing.
_VIEW_LRU_SIZE = 12


class Evidence:
    """Sentence and prompt recovery for one registry. Built on first use, then cached."""

    def __init__(self, registry):
        self.registry = registry
        self._pipeline = None
        self._segments: dict[str, dict[str, list[str]]] = {}
        self._status: dict[str, str] = {}
        self._source: dict[str, str] = {}
        self._views: "OrderedDict[tuple, dict]" = OrderedDict()

    # ── segmentation ────────────────────────────────────────────────────────

    def _stanza(self):
        """The tokenizer pipeline, or ``None`` with the reason recorded in ``status``."""
        if self._pipeline is not None:
            return self._pipeline
        stanza_dir = self.registry.stanza_dir
        if not stanza_dir:
            self._status["stanza"] = "no stanza_dir configured"
            return None
        try:
            from hpo_extraction.data.segmentation import load_stanza

            self._pipeline = load_stanza(stanza_dir=stanza_dir, mode="TOKENIZER")
        except Exception as exc:
            self._status["stanza"] = f"Stanza unavailable ({exc.__class__.__name__}: {exc})"
            log.warning("evidence: %s", self._status["stanza"])
            return None
        return self._pipeline

    def segments(self, cohort: str) -> dict[str, list[str]]:
        """``{report_id: [sentence]}`` for a cohort, segmented as inference did.

        Computed once per cohort and then kept, both in this process and in the registry's cache
        directory. See the module docstring for the five routes and why Stanza is the last of
        them; :meth:`source` names the one that answered.
        """
        if cohort in self._segments:
            return self._segments[cohort]

        texts = self.registry.report_texts(cohort)

        if texts:
            cached = self._read_cache(cohort, texts)
            if cached is not None:
                self._segments[cohort] = cached
                self._source[cohort] = "cache"
                return cached

        from_csv, csv_note = self._from_segments_csv(cohort, texts)

        if not texts:
            # Nothing to verify against, and nothing for the artifact or Stanza routes to work on
            # either, both need the report text. The csv is the only answer available, and where
            # it has none the panel degrades as it always did.
            self._segments[cohort] = from_csv
            if from_csv:
                self._source[cohort] = csv_note
            else:
                self._status[cohort] = (
                    "report texts not reachable, set the report folder in Data sources, or point "
                    f"the HCY ground truth at a cohort directory holding {SEGMENTS_FILE}")
                self._source[cohort] = "none"
            return from_csv

        recovered = dict(from_csv)
        notes = [csv_note] if from_csv else []

        if len(recovered) < len(texts):
            from_artifacts, dropped = self._from_artifacts(
                cohort, {rid: texts[rid] for rid in texts if rid not in recovered})
            if from_artifacts:
                recovered.update(from_artifacts)
                notes.append(self._artifact_source(cohort, len(from_artifacts), dropped))

        missing = [rid for rid in texts if rid not in recovered]
        if recovered and not missing:
            self._segments[cohort] = recovered
            self._source[cohort] = "; ".join(notes)
            self._write_cache(cohort, texts, recovered)
            return recovered

        by_stanza = self._from_stanza(cohort, {rid: texts[rid] for rid in missing})
        merged = {**recovered, **by_stanza}
        self._segments[cohort] = merged
        if not merged:
            self._source[cohort] = "none"
            return merged

        if by_stanza:
            notes.append(f"{len(by_stanza)} report(s) re-segmented with Stanza")
        elif missing:
            notes.append(f"{len(missing)} report(s) have none, and Stanza could not supply them "
                         f"({self._status.get('stanza', 'not tried')})")
        self._source[cohort] = "; ".join(notes)
        self._write_cache(cohort, texts, merged)
        return merged

    def _segments_csv_path(self, cohort: str) -> str:
        """Where this cohort's ``segmented_reports.csv`` is, or ``""``.

        HCY only. GSC+ has no such file, and asking the registry for one would invite a path that
        happens to exist to be read as a segmentation of a different cohort's documents.
        """
        if cohort != "hcy":
            return ""
        return getattr(self.registry, "hcy_segments_path", "") or ""

    def _from_segments_csv(self, cohort: str, texts: dict[str, str]) -> tuple[dict, str]:
        """``segmented_reports.csv`` for this cohort, verified report by report.

        Returns ``({report_id: [sentence]}, description)``. The description is what :meth:`source`
        prints, and it is where "UNVERIFIED" appears when there was no text to check against.
        """
        path = self._segments_csv_path(cohort)
        if not path or not os.path.isfile(path):
            return {}, ""
        try:
            from hpo_extraction.curation.sources import load_segments

            by_report = {str(k): list(v) for k, v in load_segments(path).items()}
        except Exception as exc:  # noqa: BLE001 - an unreadable csv is a route that did not answer
            self._status["segments_csv"] = f"{os.path.basename(path)} unreadable ({exc})"
            log.warning("evidence: %s", self._status["segments_csv"])
            return {}, ""

        name = os.path.basename(path)
        if not texts:
            return by_report, (
                f"{len(by_report)} report(s) from {name}, UNVERIFIED, there is no report text to "
                f"check the segmentation against")

        recovered: dict[str, list[str]] = {}
        dropped = 0
        for report_id, sentences in by_report.items():
            if report_id not in texts or not sentences:
                continue
            if not aligns_to_text(sentences, texts[report_id]):
                dropped += 1
                continue
            recovered[report_id] = sentences
        if dropped:
            log.warning("evidence: %d report(s) of %s were dropped, %s does not line up with the "
                        "report text", dropped, cohort, name)
        if not recovered:
            return {}, ""
        return recovered, (f"{len(recovered)} report(s) from {name}"
                           + (f"; {dropped} rejected as not matching the report text"
                              if dropped else ""))

    def _from_stanza(self, cohort: str, texts: dict[str, str]) -> dict[str, list[str]]:
        """The original three calls, over whatever reports are still missing."""
        if not texts:
            return {}
        pipeline = self._stanza()
        if pipeline is None:
            return {}
        try:
            from hpo_extraction.data.segmentation import segment_dict, split_sents

            return split_sents(segment_dict(texts, pipeline))
        except Exception as exc:
            self._status[cohort] = f"segmentation failed ({exc.__class__.__name__}: {exc})"
            log.warning("evidence: %s", self._status[cohort])
            return {}

    def _from_artifacts(self, cohort: str, texts: dict[str, str]) -> tuple[dict, int]:
        """The Free Listing generation run's saved segmentation for this cohort, verified report by report.

        Returns ``({report_id: [sentence]}, n_dropped)``. Several models' extraction dumps cover
        the same reports. Whichever is read first wins and the rest are only consulted for reports
        it did not reach, a model whose array died halfway leaves a partial file, and unioning
        them is what makes the cohort whole again.
        """
        recovered: dict[str, list[str]] = {}
        dropped = 0
        for path in self._extraction_paths(cohort):
            if len(recovered) == len(texts):
                break   # The cohort is whole. The other seven models say the same thing
            by_report = _read_extraction_sentences(path)
            for report_id, numbered in by_report.items():
                if report_id in recovered or report_id not in texts:
                    continue
                sentences = _dense(numbered)
                if sentences is None or not aligns_to_text(sentences, texts[report_id]):
                    dropped += 1
                    continue
                recovered[report_id] = sentences
        if dropped:
            log.warning("evidence: %d report(s) of %s were dropped: the Free Listing run's segmentation does "
                        "not line up with the report text", dropped, cohort)
        return recovered, dropped

    def _extraction_paths(self, cohort: str) -> list[str]:
        base = getattr(self.registry, "output_base", "") or ""
        run_dir = os.path.join(base, ENSEMBLE_EXP_ID, cohort)
        return sorted(glob.glob(os.path.join(run_dir, EXTRACTION_GLOB)))

    def _artifact_source(self, cohort: str, n: int, dropped: int) -> str:
        names = ", ".join(os.path.basename(p) for p in self._extraction_paths(cohort)[:3])
        text = f"{n} report(s) from the Free Listing run's saved segmentation ({names})"
        return text + (f"; {dropped} rejected as not matching the report text" if dropped else "")

    # ── the on-disk segmentation cache ──────────────────────────────────────

    def _cache_path(self, cohort: str, texts: dict[str, str]) -> str | None:
        """Where a cohort's segmentation is cached, keyed by a digest of the texts themselves.

        Keyed by content and not by path or mtime: the point of the cache is that a machine which
        *can* segment (or a Free Listing generation run that already did) hands the answer to a machine which
        cannot, and an mtime does not survive an rsync.
        """
        cache_dir = getattr(self.registry, "cache_dir", "")
        if not cache_dir:
            return None
        digest = hashlib.sha1()
        for report_id in sorted(texts):
            digest.update(report_id.encode())
            digest.update(texts[report_id].encode("utf-8", "replace"))
        return os.path.join(
            cache_dir, f"segments.{cohort}.v{SEGMENT_CACHE_VERSION}.{digest.hexdigest()[:16]}.json")

    def _read_cache(self, cohort: str, texts: dict[str, str]) -> dict[str, list[str]] | None:
        path = self._cache_path(cohort, texts)
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.warning("discarding unreadable segmentation cache %s: %s", path, exc)
            return None
        if not isinstance(data, dict) or not data:
            return None
        return {str(k): [str(s) for s in v] for k, v in data.items()}

    def _write_cache(self, cohort: str, texts: dict[str, str], segments: dict) -> None:
        path = self._cache_path(cohort, texts)
        if not path or not segments:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(segments, f)
        except Exception as exc:
            log.warning("could not write segmentation cache %s: %s", path, exc)

    # ── lookups ─────────────────────────────────────────────────────────────

    def sentence(self, cohort: str, report_id: str, sent_index) -> str | None:
        """The sentence a ``sent_index`` refers to, or ``None`` if it cannot be recovered.

        Out-of-range indices return ``None``, not raising or clamping. An index past the
        end of the segmentation means the run and this segmentation disagree, a different Stanza
        version, or a report file that changed, and quietly showing the wrong sentence would be
        the single most misleading thing this module could do.
        """
        sentences = self.segments(cohort).get(str(report_id))
        if not sentences or sent_index is None:
            return None
        try:
            index = int(sent_index)
        except (TypeError, ValueError):
            return None
        return sentences[index] if 0 <= index < len(sentences) else None

    def status(self, cohort: str, report_id: str | None = None) -> str | None:
        """Why the text is missing, if it is. ``None`` when everything resolved.

        *report_id* counts because partial recovery is a real state: a cohort map can be
        non-empty and still not hold *this* report. Without it this returned ``None`` in
        that case, and the two call sites in the deep-dive interpolated the literal string
        ``"None"`` into a note explaining why the text was missing.
        """
        recovered = self.segments(cohort)
        if recovered and (report_id is None or str(report_id) in recovered):
            return None
        if recovered:
            return (f"report {report_id} is not in the recovered segmentation "
                    f"({len(recovered)} of this cohort's reports are)")
        return self._status.get(cohort) or self._status.get("stanza") or "text unavailable"

    def source(self, cohort: str) -> str:
        """Where this cohort's segmentation came from, named on screen, never assumed.

        The five routes are not equally trustworthy (Stanza reproduces inference. The
        others reproduce something checked against it), so which one answered is part of the
        evidence and belongs next to the sentences it produced.
        """
        self.segments(cohort)
        return self._source.get(cohort, "none")

    def coverage(self, cohort: str, report_id: str, indices) -> dict:
        """How many of a report's referenced ``sent_index`` values the segmentation can resolve.

        Surfaced in the deep-dive because partial resolution is the dangerous case: if the
        segmentation drifted by one sentence every displayed sentence would be wrong while most
        indices still resolved. A coverage below 1.0 is shown as a warning, not hidden.

        Unparseable indices are dropped, not counted. ``calls["sent_index"].tolist()`` yields
        ``nan`` for a call record missing the field, and ``int(nan)`` raises, which used to take
        the whole callback down, not degrading the one panel.
        """
        sentences = self.segments(cohort).get(str(report_id)) or []
        wanted = []
        for value in indices:
            if value is None:
                continue
            try:
                wanted.append(int(value))
            except (TypeError, ValueError):
                continue
        resolved = sum(1 for i in wanted if 0 <= i < len(sentences))
        return {
            "n_referenced": len(wanted),
            "n_resolved": resolved,
            "n_sentences": len(sentences),
            "max_index": max(wanted, default=None),
            "complete": bool(wanted) and resolved == len(wanted),
        }

    # ── one report, aligned to its own text ─────────────────────────────────

    def report_view(self, cohort: str, report_id: str, text: str | None = None) -> dict:
        """Everything the report panel needs to draw one report as a document.

        ``{"text", "segments", "display", "ranges", "alignment"}``, mirroring
        ``apps.curation_ui.registry.Registry._build_patient`` so the two apps lay a report out
        the same way.

        ``segments`` is the tokenizer's own strings, what ``sent_index`` indexes and what
        ``phenobert.annotate`` and ``curated.placements`` are given. ``display`` is each segment
        **as the report spells it**, recovered through the alignment, and it is what gets drawn:
        offsets index these, never the tokenized ones. ``ranges`` is what puts the text *between*
        segments back, the headings and blank lines the segmentation drops, and the whole
        difference between a document and a list of sentences.

        *text* lets the caller name the coordinate system. The deep-dive passes the PhenoBERT
        baseline's staged copy when that layer is usable, so the prose, the inter-segment gaps and
        the detection offsets are all indices into one string, not three.
        """
        report_id = str(report_id)
        if text is None:
            text = self.registry.report_texts(cohort).get(report_id) or ""
        segments = list(self.segments(cohort).get(report_id) or [])

        key = (cohort, report_id, len(text), hash(text), len(segments))
        cached = self._views.get(key)
        if cached is not None:
            self._views.move_to_end(key)
            return cached

        if text and segments:
            from hpo_extraction.curation import spans

            ranges = spans.align_segments(text, segments)
            view = {
                "text": text,
                "segments": segments,
                "display": spans.segment_texts(text, segments, ranges),
                "ranges": ranges,
                "alignment": spans.alignment_stats(ranges),
            }
        else:
            # No alignment is possible, and guessing one is the failure this module exists to
            # avoid. The panel still renders: it draws the tokenized strings, joined by a space.
            ranges = [None] * len(segments)
            view = {
                "text": text,
                "segments": segments,
                "display": list(segments),
                "ranges": ranges,
                "alignment": {"n_segments": len(segments), "n_aligned": 0,
                              "n_unaligned": len(segments)},
            }

        self._views[key] = view
        while len(self._views) > _VIEW_LRU_SIZE:
            self._views.popitem(last=False)
        return view

    # ── prompt ──────────────────────────────────────────────────────────────

    def prompt(self, hpo_id: str, sentence: str) -> str | None:
        """The prompt the SLM was given for one (term, sentence) pair.

        A transcription of ``hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1``, reading the term's
        metadata off the shared tree instead of building a new one per call.
        """
        data = self.registry.view.tree.data
        resolved = self.registry.view.resolve(hpo_id) or hpo_id
        entry = data.get(resolved)
        if entry is None:
            return None
        return build_prompt(entry, sentence)

    # ── pins ────────────────────────────────────────────────────────────────

    def assert_prompt_matches(self, hpo_id: str, sentence: str) -> None:
        """Fix :meth:`prompt` to ``embed_symptom_in_prompt_v1``, character for character."""
        from hpo_extraction.treephenorag.verifier_prompt import embed_symptom_in_prompt_v1

        expected = embed_symptom_in_prompt_v1([sentence], hpo_id)[0]
        actual = self.prompt(hpo_id, sentence)
        assert actual == expected, (
            "evidence.prompt drifted from hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1\n"
            f"  expected: {expected!r}\n  actual:   {actual!r}"
        )

    def assert_segmentation_matches(self, cohort: str) -> None:
        """Fix :meth:`segments` to the exact three calls ``tree_experiment`` makes.

        Cheap insurance against the failure mode that has no visible symptom: a segmentation that
        differs by one sentence still resolves almost every index, and every sentence shown in the
        deep-dive is then the wrong one. This is also what certifies the recovery routes, run it
        once on a machine where Stanza works and the artifact route is no longer a hypothesis.
        """
        from hpo_extraction.data.segmentation import load_stanza, segment_dict, split_sents

        texts = self.registry.report_texts(cohort)
        pipeline = load_stanza(stanza_dir=self.registry.stanza_dir, mode="TOKENIZER")
        expected = split_sents(segment_dict(texts, pipeline))
        actual = self.segments(cohort)
        assert actual == expected, "evidence.segments drifted from hpo_extraction.data.segmentation"


def _read_extraction_sentences(path: str) -> dict[str, dict[int, str]]:
    """``{report_id: {sentence_number: sentence_text}}`` from one ``llm_extractions_*.jsonl``.

    Torn lines are skipped, not raised on, which is the opposite of every other reader in
    this package and is right here: this file is not the measurement, it is a lookup table for
    text a wall-clock kill may have cut in half, and one lost sentence must not cost the cohort
    its segmentation. ``loaders.read_jsonl``'s strictness stays where the numbers are.
    """
    out: dict[str, dict[int, str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                report_id = record.get("patient_id")
                number = record.get("sentence_number")
                text = record.get("sentence_text")
                if report_id is None or number is None or not isinstance(text, str):
                    continue
                out.setdefault(str(report_id), {})[int(number)] = text
    except OSError as exc:
        log.warning("evidence: could not read %s: %s", path, exc)
    return out


def _dense(numbered: dict[int, str]) -> list[str] | None:
    """``{0: a, 1: b}`` → ``[a, b]``; ``None`` when the numbering has a hole.

    A hole means the file is missing a sentence, and a list built by sorting what is there would
    silently shift every later index by one, which is the failure that resolves cleanly
    and shows the wrong sentence.
    """
    if not numbered or set(numbered) != set(range(len(numbered))):
        return None
    return [numbered[i] for i in range(len(numbered))]


def aligns_to_text(sentences: list[str], text: str) -> bool:
    """Whether *sentences* can be a segmentation of *text*, or up to whitespace.

    :func:`verify_segmentation` first, because an exact match is the strongest evidence and costs
    one ``str.find`` per sentence. Where it fails, the whitespace-insensitive alignment
    ``hpo_extraction.curation.spans`` uses is tried instead: Stanza's tokenizer is **not** guaranteed to
    preserve the source bytes, it normalises whitespace and can normalise quotes, so an exact-only
    check rejects perfectly good segmentations of reports that happen to contain a tab or a
    non-breaking space. See that module's docstring. It is the same rule the curation app places
    every annotation by, which is what keeps the two apps' segment 4 the same sentence.

    A segment that cannot be located *at all* still fails. The failure mode this guards against is
    not "slightly different whitespace", it is "these sentences describe a different document".
    """
    if verify_segmentation(sentences, text):
        return True
    try:
        from hpo_extraction.curation.spans import align_segments
    except Exception as exc:  # noqa: BLE001 - without the aligner, exact matching is the answer
        log.warning("evidence: whitespace-insensitive alignment unavailable (%s)", exc)
        return False
    return all(span is not None for span in align_segments(text, sentences))


def verify_segmentation(sentences: list[str], text: str) -> bool:
    """Whether *sentences* can be a segmentation of *text*: each occurs, in order, after the last.

    ``split_sents(segment_dict(...))`` only ever slices its input, Stanza's ``sentence.text`` is
    a span of the document and the newline split cuts those spans further, so every sentence of a
    genuine segmentation is a substring of the report, and they appear in order. A recovered list
    that fails this describes a different document, whatever its indices say.
    """
    cursor = 0
    for sentence in sentences:
        found = text.find(sentence, cursor)
        if found < 0:
            return False
        cursor = found + len(sentence)
    return True


def build_prompt(entry: dict, sentence: str) -> str:
    """One prompt, from a raw ``hpo.json`` entry and a sentence.

    Kept as a module function so the tests can call it on the toy ontology's entries without a
    registry. The string is assembled in the order
    ``hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1`` assembles it, including the two conditional
    clauses and the trailing space.
    """
    label = entry["Name"][0]
    definition = entry["Def"][0] if entry.get("Def") else ""
    synonyms = ", ".join(entry.get("Synonym") or [])

    intro = f"The symptom {label}"
    intro += f" is defined as {definition}." if definition else "."
    if synonyms:
        intro += f" {label} is also referred to as {synonyms}."
    return (
        intro
        + " Does the following text segment explicitly confirm that the patient"
        + f" has this symptom:'{sentence}'. "
    )
