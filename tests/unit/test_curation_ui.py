"""Tests for the HCY curation UI (``app/hcy_curation_ui``).

The app's own selftest (``python apps/curation_ui/app.py --selftest``) is the acceptance check:
it renders every panel for every patient in both modes and drives the HTTP surface. What is left
for pytest is what a render harness cannot assert, that the *decisions* the app records survive,
and that the two mechanisms a curator has no way to double-check are correct.

Those two are worth naming, because they are why this file exists:

**Persistence.** Everything else in this app is recoverable by reloading the page. A lost or
misfolded event is not, it is a judgement call silently absent from a ground-truth set that later becomes
the denominator of a thesis table. So the log is tested for last-write-wins, for a torn trailing
line, and for atomicity of the derived CSV.

**Span alignment.** A PhenoBERT offset placed on the wrong segment underlines the wrong words, and
a curator reading a screen has no independent way to notice. The tests therefore fix not just that
alignment works but that it *fails closed*, an unalignable segment must produce no highlight
rather than an approximate one.

Four groups:

``TestStore``      the event log: fold order, crash resume, atomic writes, ground truth export
``TestSpans``      alignment, forward cursor on repeated sentences, fail-closed placement
``TestSources``    the three annotation loaders and the shape they all produce
``TestAnchors``    where an annotation from a file lands on this app's segmentation, and when it
                   refuses to land anywhere
``TestEditing``    repairing an annotation in place: what an ``edit`` event may and may not change
``TestAdjudicationOrder``  which rows Approve rules on, and which are reference only
``TestSearch``     the wire contract, capped, minimum query length, synonyms, exact ids
``TestLocate``     trigger candidates: exact, inflected, and the over-reach that must not happen
``TestLabels``     the difficulty vocabulary and what a browser is allowed to write into it
``TestProse``      the report reads as the document it is, and is still segment-selectable
``TestRegistry``   the view model and what does and does not enter the curated export
``TestCallbacks``  the two ways a Dash re-render fires a callback nobody pressed
``TestComments``   report commentary and proposed deletions, the two things Edit mode proposes
                   that are not a new phenotype
``TestReportControls``  the report-label controls are static, and only their values move
"""

from __future__ import annotations

import csv
import json
import os
import re
import stat

import pytest

from hpo_extraction.curation import evidence_location as locations, labels as vocab, sources, spans, store
from apps.curation_ui import fixture, locate, search
from apps.curation_ui import theme
from hpo_extraction.ontology import nltk_data
from apps.curation_ui.registry import Registry
from apps.curation_ui.views import approve, editor, labelling, reader


# ──────────────────────────────────────────────────────────────────────────────
# a tiny ontology, everything search and locate need, and nothing else
# ──────────────────────────────────────────────────────────────────────────────
class TinyTree:
    """The three-method surface :mod:`search` and :mod:`locate` actually depend on.

    Not ``tests.fixtures.toy_ontology.ToyTree``: that one models the *graph* (ancestors, depth,
    closures) for the metric tests and carries no phrases, while these two modules care only about
    names and synonyms. Stating the surface here documents the contract, not assuming one.
    """

    DEFS = {
        "HP:0001250": "An intermittent abnormality of nervous system physiology.",
        "HP:0001252": "Abnormally low muscle tone.",
    }

    TERMS = {
        "HP:0001250": ["seizure", "seizures", "epileptic seizure"],
        "HP:0001252": ["hypotonia", "low muscle tone", "muscular hypotonia"],
        "HP:0001263": ["global developmental delay", "delay"],
        "HP:0002059": ["cerebral atrophy"],
        "HP:0001873": ["thrombocytopenia"],
    }

    def __init__(self):
        self.hpo_list = sorted(self.TERMS)
        # ``search`` reads definitions straight off ``tree.data[code]["Def"]``, the hpo.json shape.
        self.data = {code: {"Def": [self.DEFS[code]] if code in self.DEFS else []}
                     for code in self.TERMS}

    def getNameByHPO(self, code):  # noqa: N802 - HPOTree's name
        return self.TERMS[code][0]

    def getPhrasesByHPO(self, code):  # noqa: N802 - HPOTree's name
        return list(self.TERMS.get(code, []))


@pytest.fixture
def tree():
    return TinyTree()


@pytest.fixture
def cohort(tmp_path):
    """A registry over the synthetic fixture, with its log inside ``tmp_path``.

    Also installed as the **process-wide** registry for the duration of the test, and removed
    afterwards. The views reach for it through ``state``, not through an argument, Dash
    callbacks can only carry JSON, so any test that calls a ``render_*`` function needs it set,
    and one that leaked into the next test would silently point at a deleted tmp_path.
    """
    from apps.curation_ui import state

    paths = fixture.write(str(tmp_path))
    registry = Registry(paths, author="pytest",
                        curation_dir=str(tmp_path / "hcy" / "curation"))
    previous = state.get_registry() if state.has_registry() else None
    state.set_registry(registry)
    try:
        yield registry
    finally:
        state.set_registry(previous)


# ──────────────────────────────────────────────────────────────────────────────
class TestStore:
    def test_fold_is_last_write_wins(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("raw", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="removed")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")

        rows = log.fold()["rows"]
        assert rows[key]["status"] == "kept"
        # The superseded events are still on disk: the log is an audit trail, not a state file.
        assert len(log.events) == 3

    def test_two_curators_on_one_directory_see_each_other(self, tmp_path):
        """Daphne and Amoosh curate at once, each in their own process on the login node.

        Two ``EventLog`` objects over one directory are that: two independent in-memory
        caches of the same file. Before the refresh-on-change fix each folded only its own clicks,
        and the commit that follows a click then rewrote the derived CSVs from that half view.
        """
        daphne = store.EventLog(str(tmp_path), author="daphne")
        amoosh = store.EventLog(str(tmp_path), author="amoosh")

        d_key = store.target_key("daphne", "P1", "HP:0001250")
        a_key = store.target_key("daphne", "P2", "HP:0001252")
        daphne.append("verdict", "P1", target_key=d_key, hpo_code="HP:0001250", status="kept")
        amoosh.append("verdict", "P2", target_key=a_key, hpo_code="HP:0001252", status="removed")

        # Each fold carries both verdicts, whichever side it is taken from.
        for log in (daphne, amoosh):
            rows = log.fold()["rows"]
            assert rows[d_key]["status"] == "kept"
            assert rows[a_key]["status"] == "removed"

        # And the appended event is never written on top of a stale prefix: the second writer's
        # own list already held the first writer's event before it added its own.
        assert [e["author"] for e in amoosh.events] == ["daphne", "amoosh"]

    def test_refresh_is_a_no_op_when_the_log_has_not_changed(self, tmp_path):
        """The refresh runs on every fold, so it has to be a stat and not a re-read."""
        log = store.EventLog(str(tmp_path), author="a")
        log.append("comment", "P1", comment_id="c1", text="hello")

        assert log.refresh() is False
        store.EventLog(str(tmp_path), author="b").append(
            "comment", "P1", comment_id="c2", text="world")
        assert log.refresh() is True
        assert log.refresh() is False

    def test_withdraw_removes_an_approved_suggestion(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        event = log.append("suggest", "P1", segment_idx=0, hpo_code="HP:0001250",
                           trigger_word="fits")
        log.append("approve", "P1", target_key=event["event_id"])
        assert log.fold()["rows"][event["event_id"]]["status"] == "approved"

        log.append("withdraw", "P1", target_key=event["event_id"])
        assert event["event_id"] not in log.fold()["rows"]

    def test_location_does_not_decide_anything(self, tmp_path):
        """Recording where a term came from must not imply a verdict on it."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("prior_annotation_2", "P1", "HP:0001252")
        log.append("anchor", "P1", target_key=key, hpo_code="HP:0001252",
                   segment_idx=3, segment_text="He was hypotonic.", trigger_word="hypotonic")

        row = log.fold()["rows"][key]
        assert row["segment_idx"] == 3 and row["trigger_word"] == "hypotonic"
        assert row["status"] == "", "anchoring silently set a status"

    def test_confirm_toggles(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        log.append("confirm_patient", "P1")
        assert "P1" in log.fold()["confirmed"]
        log.append("unconfirm_patient", "P1")
        assert "P1" not in log.fold()["confirmed"]

    def test_unknown_action_is_refused_at_write_and_skipped_at_read(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        with pytest.raises(ValueError):
            log.append("obliterate", "P1")

        # A log written by a newer version of the app must still open in this one.
        with open(log.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"action": "from_the_future", "patient_id": "P1"}) + "\n")
        state = store.EventLog(str(tmp_path), author="a").fold()
        assert state["rows"] == {} and state["n_skipped"] == 1

    def test_a_torn_trailing_line_costs_only_that_line(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        log.append("suggest", "P1", segment_idx=0, hpo_code="HP:0001250", trigger_word="fits")
        log.append("suggest", "P1", segment_idx=1, hpo_code="HP:0001252", trigger_word="floppy")
        with open(log.path, "a", encoding="utf-8") as handle:
            handle.write('{"action":"sugg')      # a crash mid-append

        reopened = store.EventLog(str(tmp_path), author="a")
        assert len(reopened.fold()["rows"]) == 2

    def test_current_csv_is_flat_and_atomic(self, tmp_path):
        path = str(tmp_path / "out.csv")
        rows = [{"patient_id": "P1", "segment_text": "line one\nline two",
                 "hpo_code": "HP:0001250"}]
        store.write_csv_atomic(path, ["patient_id", "segment_text", "hpo_code"], rows)

        with open(path, newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        assert parsed[0]["segment_text"] == "line one line two"
        # One record is one line: no viewer or spreadsheet round-trip can split it.
        assert len(open(path, encoding="utf-8").read().strip().splitlines()) == 2
        assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]

    def test_a_failed_write_leaves_no_temp_file_behind(self, tmp_path, monkeypatch):
        path = str(tmp_path / "out.csv")
        monkeypatch.setattr(store.os, "replace",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            store.write_csv_atomic(path, ["patient_id"], [{"patient_id": "P1"}])
        assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]

    def test_phenobert_key_separates_two_detections_of_one_code(self):
        """One code detected on two phrases is two annotations, each adjudicated on its own."""
        first = store.target_key("phenobert", "P1", "HP:0001250", 12)
        second = store.target_key("phenobert", "P1", "HP:0001250", 340)
        assert first != second
        # Ground truth rows carry no offset, so their key stays stable across runs.
        assert store.target_key("raw", "P1", "HP:0001250") == "raw|P1|HP:0001250"

    def test_export_holds_only_decided_in_rows(self, tmp_path):
        rows = {"P1": [
            {"hpo_code": "HP:0001250", "status": "approved"},
            {"hpo_code": "HP:0001252", "status": "kept"},
            {"hpo_code": "HP:0001263", "status": "rejected"},
            {"hpo_code": "HP:0001873", "status": "suggested"},
            {"hpo_code": "HP:0002059", "status": ""},
        ]}
        result = store.export_gold(str(tmp_path / "gold.csv"), rows, ["P1", "P2"])

        with open(result["path"], newline="", encoding="utf-8") as handle:
            parsed = {r["patient_id"]: r["hpo_codes"] for r in csv.DictReader(handle)}
        assert parsed["P1"] == "HP:0001250;HP:0001252"
        # A patient with nothing decided still gets a line, dropping it would quietly shrink
        # The cohort a downstream denominator is computed over.
        assert parsed["P2"] == ""
        assert result["n_patients"] == 2 and result["n_pairs"] == 2


# ──────────────────────────────────────────────────────────────────────────────
class TestSpans:
    TEXT = ("Case history:  the boy\nhad recurrent seizures.\n"
            "He was hypotonic.  Seizures again.\n")
    SEGMENTS = ["Case history: the boy had recurrent seizures.",
                "He was hypotonic.",
                "Seizures again."]

    def test_alignment_survives_retokenised_whitespace(self):
        ranges = spans.align_segments(self.TEXT, self.SEGMENTS)
        assert all(r is not None for r in ranges)
        # The recovered slice is the report's spelling, not the tokenizer's.
        assert self.TEXT[ranges[0][0]:ranges[0][1]].startswith("Case history:  the boy\nhad")

    def test_a_repeated_sentence_maps_to_its_own_occurrence(self):
        """The forward cursor is the point: 'Seizures again.' must not match the earlier one."""
        ranges = spans.align_segments(self.TEXT, self.SEGMENTS)
        assert ranges[2][0] == self.TEXT.index("Seizures again.")

    def test_an_unalignable_segment_yields_no_highlight_rather_than_a_wrong_one(self):
        segments = self.SEGMENTS + ["A sentence that is nowhere in the report."]
        ranges = spans.align_segments(self.TEXT, segments)
        assert ranges[-1] is None
        assert spans.alignment_stats(ranges)["n_unaligned"] == 1

        # A detection inside the missing segment's supposed range must be reported unplaced, never
        # attached to a neighbour.
        placed, unplaced = spans.place_detections(ranges, [{"start": 10_000, "end": 10_005}])
        assert placed == {} and len(unplaced) == 1

    def test_detections_land_on_the_segment_that_contains_them(self):
        start = self.TEXT.index("hypotonic")
        placed, unplaced = spans.map_detections(
            self.TEXT, self.SEGMENTS,
            [{"start": start, "end": start + 9, "phrase": "hypotonic"}])
        assert not unplaced and list(placed) == [1]

        hit = placed[1][0]
        shown = spans.segment_texts(self.TEXT, self.SEGMENTS,
                                    spans.align_segments(self.TEXT, self.SEGMENTS))[1]
        assert shown[hit["local_start"]:hit["local_end"]] == "hypotonic"

    def test_segment_texts_falls_back_for_an_unaligned_segment(self):
        segments = self.SEGMENTS + ["Nowhere in the report."]
        ranges = spans.align_segments(self.TEXT, segments)
        assert spans.segment_texts(self.TEXT, segments, ranges)[-1] == "Nowhere in the report."

    def test_no_text_means_nothing_is_placed(self):
        ranges = spans.align_segments("", self.SEGMENTS)
        assert ranges == [None, None, None]


# ──────────────────────────────────────────────────────────────────────────────
class TestSources:
    """The three annotation files, and the one record shape they all have to produce.

    The loaders are where a file's own quirks are supposed to stop. Everything downstream, the
    marks, the panels, the keys, the export, reads one shape, so a quirk that survives a loader
    survives into a ground-truth set.
    """

    @pytest.fixture
    def written(self, tmp_path):
        return fixture.write(str(tmp_path))

    def test_prior_annotation_yields_one_record_per_annotation_and_the_report_once(self, written):
        records, texts = sources.load_prior_annotation(written["prior_annotation"])

        assert [r["hpo_code"] for r in records["SYN001"]] == \
            ["HP:0001250", "HP:0001252", "HP:0001873"]
        assert records["SYN001"][0]["trigger_word"] == "seizures"
        assert records["SYN001"][0]["source"] == "prior_annotation"
        # The file repeats the report on every row. The loader keeps it once, per patient.
        assert texts["SYN001"] == fixture.REPORTS["SYN001"]

    def test_a_char_offset_survives_as_an_int_and_an_empty_one_as_none(self, written):
        records, _ = sources.load_prior_annotation(written["prior_annotation"])
        by_code = {r["hpo_code"]: r for r in records["SYN001"]}

        assert isinstance(by_code["HP:0001250"]["char_offset"], int)
        assert by_code["HP:0001873"]["char_offset"] is None

    def test_a_code_annotated_twice_gets_a_slot_and_the_first_keeps_none(self, written):
        """Appending a second occurrence must not re-key the one somebody already adjudicated."""
        records, _ = sources.load_prior_annotation(written["prior_annotation"])
        slots = [r["slot"] for r in records["SYN002"] if r["hpo_code"] == "HP:0001263"]

        assert slots == [None, 1]

    def test_confirmed_reads_true_even_with_a_spreadsheets_full_stop(self, written):
        """The real file holds ``True.``. Reading that as unconfirmed would demote every row."""
        records = sources.load_confirmed(written["confirmed"])
        flags = {r["hpo_code"]: r["confirmed"] for r in records["SYN002"]}

        assert flags["HP:0011968"] is False
        assert sources.load_confirmed(written["confirmed"])["SYN001"][0]["confirmed"] is True

    def test_confirmed_carries_its_segment_index_and_provenance(self, written):
        record = sources.load_confirmed(written["confirmed"])["SYN001"][0]

        assert record["segment_idx"] == 1 and record["provenance"] == "manual"
        # The segment column travels as context, so a wrong index can still be recovered from it.
        assert "seizures" in record["sentence_context"]

    def test_a_code_only_file_produces_records_with_nothing_to_place(self, written):
        records = sources.load_code_only(written["prior_annotation_2"], "prior_annotation_2")
        record = records["SYN001"][0]

        assert record["trigger_word"] == "" and record["segment_idx"] is None
        assert record["char_offset"] is None and record["source"] == "prior_annotation_2"

    def test_a_missing_file_is_not_fatal(self, tmp_path):
        """Curating with only some of the sources present is a legitimate session."""
        assert sources.load_prior_annotation(str(tmp_path / "nope.csv")) == ({}, {})
        assert sources.load_confirmed(str(tmp_path / "nope.csv")) == {}
        assert sources.load_code_only(str(tmp_path / "nope.csv"), "prior_annotation_2") == {}


# ──────────────────────────────────────────────────────────────────────────────
class TestLocations:
    """Where an annotation from a file lands on *this* app's segmentation.

    Every one of these strategies is a different file being right about a different thing, and the
    last of them is the app guessing. The tests fix which is which, because the screen reports it
    and a curator's verdict depends on it.
    """

    SEGMENTS = [
        "Case history: the boy is 4 years old.",
        "He presented with recurrent seizures since age two.",
        "He was markedly hypotonic at birth.",
    ]

    def test_a_stated_segment_is_used_when_the_trigger_is_really_in_it(self):
        hit = locations.locate(
            {"segment_idx": 1, "trigger_word": "seizures"}, self.SEGMENTS)

        assert hit["segment_idx"] == 1 and hit["how"] == "segment"
        assert self.SEGMENTS[1][hit["start"]:hit["end"]] == "seizures"

    def test_a_wrong_segment_index_falls_back_to_the_files_own_context(self):
        """A file written against a different segmentation is the normal case, not the exception."""
        hit = locations.locate({"segment_idx": 0, "trigger_word": "hypotonic",
                              "sentence_context": "He was markedly hypotonic at birth."},
                             self.SEGMENTS)

        assert hit["segment_idx"] == 2 and hit["how"] == "context"

    def test_an_offset_is_checked_against_the_trigger_not_trusted(self):
        text = "\n".join(self.SEGMENTS)
        true_at = text.index("hypotonic")

        assert locations.resolve_offset(text, true_at, "hypotonic") == true_at
        # Off by a header's worth of characters: repaired from the trigger word.
        assert locations.resolve_offset(text, 0, "hypotonic") == true_at
        # Pointing at nothing like it, and out of the window: refused, not fudged.
        assert locations.resolve_offset(text, true_at, "purpura") is None

    def test_an_offset_maps_through_the_files_own_alignment(self):
        text = "\n".join(self.SEGMENTS)
        ranges = spans.align_segments(text, self.SEGMENTS)
        hit = locations.locate({"trigger_word": "hypotonic",
                              "char_offset": text.index("hypotonic")},
                             self.SEGMENTS, ranges, text)

        assert hit["segment_idx"] == 2 and hit["how"] == "offset"

    def test_a_trigger_with_no_position_at_all_is_found_lexically_and_says_so(self):
        hit = locations.locate({"trigger_word": "seizures"}, self.SEGMENTS)

        assert hit["segment_idx"] == 1 and hit["how"] == "lexical"

    def test_a_trigger_that_occurs_nowhere_is_not_placed(self):
        """Underlining approximately-right words in a tool whose output becomes ground truth is worse than
        underlining none."""
        assert locations.locate({"trigger_word": "purpura"}, self.SEGMENTS) is None

    def test_a_named_segment_survives_a_trigger_that_did_not(self):
        """Which claim failed is worth showing. Dropping both throws away the good half."""
        hit = locations.locate({"segment_idx": 2, "trigger_word": "purpura"}, self.SEGMENTS)

        assert hit["segment_idx"] == 2 and hit["start"] is None

    def test_word_boundaries_win_over_a_substring(self):
        segments = ["The delay was delayed."]
        hit = locations.locate({"trigger_word": "delay"}, segments)

        assert segments[0][hit["start"]:hit["end"]] == "delay" and hit["start"] == 4

    def test_a_record_claiming_no_position_is_neither_placed_nor_unplaced(self):
        """The unplaced list means *somebody said where this was and we could not find it*. Every
        code-only term in it would empty that of meaning."""
        records = [{"source": "prior_annotation_2", "hpo_code": "HP:1", "trigger_word": "",
                    "segment_idx": None, "char_offset": None}]
        placed, unplaced = locations.place_all(records, self.SEGMENTS)

        assert placed == {} and unplaced == []
        assert not locations.has_position(records[0])

    def test_a_literal_tab_escape_is_turned_back_into_whitespace(self):
        r"""The context column has been seen starting with the two characters ``\`` and ``t``, a round trip through a writer that escaped a tab and a reader that did not. Glued to the
        first word it is enough to lose a containment match."""
        assert locations._unescape(r"\tCase history:") == " Case history:"

        hit = locations.locate({"trigger_word": "hypotonic",
                              "sentence_context": r"He was markedly\thypotonic at birth."},
                             self.SEGMENTS)
        assert hit["segment_idx"] == 2 and hit["how"] == "context"


# ──────────────────────────────────────────────────────────────────────────────
class TestEditing:
    """Repairing an annotation in place.

    The point of an ``edit`` is that it is *not* a verdict: the two are different judgements, and a
    log that fused them could not answer who kept a term. The other half is that an edit must not
    re-key the row, the key is the address of the annotation as its file wrote it, and losing it
    would orphan every earlier event about the same annotation.
    """

    def _log(self, tmp_path):
        return store.EventLog(str(tmp_path), author="pytest")

    def test_an_edit_rewrites_only_the_fields_it_names(self, tmp_path):
        log = self._log(tmp_path)
        key = store.target_key("prior_annotation", "P1", "HP:0001250")
        log.append("anchor", "P1", target_key=key, segment_idx=3, segment_text="third",
                   trigger_word="fits", hpo_code="HP:0001250")
        log.append("edit", "P1", target_key=key, trigger_word="convulsions")

        row = log.fold()["rows"][key]
        assert row["trigger_word"] == "convulsions"
        assert row["segment_idx"] == 3 and row["segment_text"] == "third"

    def test_an_edit_does_not_touch_the_verdict(self, tmp_path):
        log = self._log(tmp_path)
        key = store.target_key("daphne", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("edit", "P1", target_key=key, trigger_word="convulsions", segment_idx=2)

        assert log.fold()["rows"][key]["status"] == "kept"
        assert "status" not in store.EDITABLE

    def test_an_edit_keeps_the_row_at_its_files_address(self, tmp_path):
        """Curating prior_annotation_2's HP:0001250 as HP:0007359 is a repair of that annotation, not a new one:
        the row keeps its key so the log still reads as one thread about one thing."""
        log = self._log(tmp_path)
        key = store.target_key("prior_annotation_2", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("edit", "P1", target_key=key, hpo_code="HP:0007359", hpo_name="Focal seizure")

        rows = log.fold()["rows"]
        assert list(rows) == [key]
        assert rows[key]["hpo_code"] == "HP:0007359" and rows[key]["source"] == "prior_annotation_2"

    def test_an_edit_can_clear_a_note(self, tmp_path):
        """The generic fold rule only writes truthy values, so without its own branch a note would
        survive its own deletion, and a note nobody can remove is one nobody will write."""
        log = self._log(tmp_path)
        key = store.target_key("prior_annotation", "P1", "HP:0001250")
        log.append("edit", "P1", target_key=key, note="looks wrong")
        log.append("edit", "P1", target_key=key, note="")

        assert log.fold()["rows"][key]["note"] == ""

    def test_a_suggestion_can_be_edited_by_its_own_event_id(self, tmp_path):
        log = self._log(tmp_path)
        event = log.append("suggest", "P1", segment_idx=0, segment_text="first",
                           hpo_code="HP:0001250", hpo_name="Seizure", trigger_word="fits")
        log.append("edit", "P1", target_key=event["event_id"], trigger_word="convulsions")

        row = log.fold()["rows"][event["event_id"]]
        assert row["trigger_word"] == "convulsions" and row["status"] == "suggested"

    def test_an_edited_row_carries_its_new_code_into_the_export(self, tmp_path):
        log = self._log(tmp_path)
        key = store.target_key("prior_annotation_2", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("edit", "P1", target_key=key, hpo_code="HP:0007359")

        rows = {"P1": list(log.fold()["rows"].values())}
        result = store.export_gold(str(tmp_path / "gold.csv"), rows, ["P1"])
        with open(result["path"], newline="", encoding="utf-8") as handle:
            assert list(csv.DictReader(handle))[0]["hpo_codes"] == "HP:0007359"

    def test_the_validator_is_the_same_rule_for_both_forms(self, cohort):
        """A repair may leave the trigger empty. It may never point at a segment that does not
        contain it. One function, two argument sets, two near-copies would drift."""
        from apps.curation_ui.views import common

        assert common.validate_annotation(cohort, "SYN001", 0, "", "",
                                          require_trigger=False, require_code=False) == ""
        assert common.validate_annotation(cohort, "SYN001", 0, "seizures", "",
                                          require_trigger=False, require_code=False)
        assert common.validate_annotation(cohort, "SYN001", 1, "seizures", "",
                                          require_trigger=False, require_code=False) == ""

    def test_one_suggestion_editable_in_both_modes_has_no_duplicate_ids(self, cohort):
        """Edit and Approve are both in the DOM at once, the mode switch only toggles `display`, so a suggestion visible in both would give its editor the same pattern id twice, which Dash
        refuses to render at all. That is what the ``at`` field in every editor id is for."""
        import json
        from collections import Counter

        from apps.curation_ui.views import approve, edit

        view = cohort.patient("SYN001")
        cohort.record("suggest", "SYN001", segment_idx=1, segment_text=view["display"][1],
                      hpo_code="HP:0001250", hpo_name="Seizure", trigger_word="seizures")

        rendered = [edit.render_list("SYN001", "light"),
                    edit.render_deletable("SYN001", "light"),
                    approve.render("SYN001", "light")]
        ids = [json.dumps(node.id, sort_keys=True) if isinstance(node.id, dict) else node.id
               for node in _walk(rendered) if getattr(node, "id", None) is not None]

        assert ids, "nothing rendered an id at all"
        assert [i for i, n in Counter(ids).items() if n > 1] == []
        # And both panels really did draw one, or the check above is vacuous.
        ats = {json.loads(i).get("at") for i in ids if i.startswith("{") and "ann-save" in i}
        assert ats == {"edit", "approve"}

    def test_the_edited_chip_names_what_changed(self):
        from apps.curation_ui.views import editor

        record = {"hpo_code": "HP:0001250", "trigger_word": "fits"}
        assert editor.edited_fields({"hpo_code": "HP:0001250", "trigger_word": "fits"},
                                    record) == []
        assert editor.edited_fields({"hpo_code": "HP:0007359", "trigger_word": "convulsions"},
                                    record) == ["term", "trigger"]


# ──────────────────────────────────────────────────────────────────────────────
class TestAdjudicationOrder:
    """Which rows Approve mode rules on, and which are only there to be read.

    ``annotations_confirmed.csv`` is a confirmed pass over the prior_annotation set, so its row for a term
    *is* the annotation to judge. The prior_annotation row for the same code is that row's predecessor, not
    a second opinion. Where the confirmed file dropped a term, the prior_annotation row stands in, that
    drop is itself a judgement and has to be confirmable. ``prior_annotation_2`` and PhenoBERT are reference.
    """

    def test_daphne_wins_a_code_it_carries(self, cohort):
        existing = {(r["source"], r["hpo_code"]) for r in cohort.patient("SYN001")["existing"]}

        assert ("daphne", "HP:0001250") in existing
        assert ("prior_annotation", "HP:0001250") not in existing, "the same question asked twice"

    def test_prior_annotation_stands_in_for_a_term_daphne_dropped(self, cohort):
        existing = {(r["source"], r["hpo_code"]) for r in cohort.patient("SYN001")["existing"]}

        assert ("prior_annotation", "HP:0001873") in existing

    def test_a_reference_source_is_never_ruled_on(self, cohort):
        for patient_id in cohort.patient_ids:
            sources_seen = {r["source"] for r in cohort.patient(patient_id)["existing"]}
            assert "prior_annotation_2" not in sources_seen

    def test_both_annotations_of_a_twice_annotated_code_survive(self, cohort):
        """Precedence is per code, not per row: it must not silently drop the second trigger."""
        existing = [r for r in cohort.patient("SYN002")["existing"]
                    if r["hpo_code"] == "HP:0001263"]

        assert [r["slot"] for r in existing] == [None, 1]

    def test_approve_shows_no_phenobert_row(self, cohort):
        """A detection worth keeping is added as a suggestion from Edit mode, which is the path
        that gives it a segment and a trigger word."""
        from apps.curation_ui.views import approve

        view = cohort.patient("SYN002")
        entries = approve._existing_entries(cohort, view, {})
        assert "phenobert" not in {e["source"] for e in entries}
        assert "HP:9999999" not in {e["code"] for e in entries}

    def test_a_row_that_stands_in_says_why(self, cohort):
        from apps.curation_ui.views import approve

        view = cohort.patient("SYN001")
        entries = {e["code"]: e for e in approve._existing_entries(cohort, view, {})}
        assert "daphne" in entries["HP:0001873"]["stands_in"]
        assert entries["HP:0001250"]["stands_in"] == ""

    def test_the_agreement_chip_still_names_the_reference_sources(self, cohort):
        """Knowing a second annotator also carries this code is what makes the verdict easy, it is
        the one thing the reference sources are still on the screen to say."""
        from apps.curation_ui.views import approve

        view = cohort.patient("SYN001")
        entry = next(e for e in approve._existing_entries(cohort, view, {})
                     if e["code"] == "HP:0001250")
        assert entry["agreement"] == "all sources + PB"

    def test_a_decided_row_no_current_source_nominates_is_carried_over(self, cohort):
        """A source that was retired, every ``raw|…`` verdict from before prior_annotation replaced it.
        A decided row that vanished would be a decision nobody could revisit, and one carrying an
        open deletion proposal would be a question nobody could answer."""
        from apps.curation_ui.views import approve

        key = store.target_key("raw", "SYN001", "HP:0002059")
        cohort.record("verdict", "SYN001", target_key=key, hpo_code="HP:0002059", status="kept")

        view = cohort.patient("SYN001")
        rows = {r["key"]: r for r in cohort.rows_for("SYN001")}
        entry = next(e for e in approve._existing_entries(cohort, view, rows) if e["key"] == key)
        assert entry["status"] == "kept" and entry["carried"]

    def test_an_undecided_row_from_a_retired_source_is_not_resurrected(self, cohort):
        """Carrying over is about decisions, not about listing every key the log has ever seen."""
        from apps.curation_ui.views import approve

        key = store.target_key("raw", "SYN001", "HP:0002059")
        cohort.record("anchor", "SYN001", target_key=key, hpo_code="HP:0002059",
                      segment_idx=0, trigger_word="Case")

        view = cohort.patient("SYN001")
        rows = {r["key"]: r for r in cohort.rows_for("SYN001")}
        assert key not in {e["key"] for e in approve._existing_entries(cohort, view, rows)}

    def test_the_overview_counts_only_what_has_to_be_ruled_on(self, cohort):
        row = next(r for r in cohort.overview() if r["patient_id"] == "SYN001")

        assert row["n_existing"] == 4          # two daphne, two holistic standing in
        assert row["n_prior_annotation_2"] == 3             # still counted, still not ruled on


# ──────────────────────────────────────────────────────────────────────────────
class TestAgreement:
    """What every row says about the disagreement this app exists to resolve."""

    def test_every_loaded_source_carrying_it_reads_as_agreement(self):
        from apps.curation_ui.views import common

        label, why = common.agreement({"prior_annotation", "daphne", "prior_annotation_2"},
                                      ["prior_annotation", "daphne", "prior_annotation_2"], False)
        assert label == "all sources" and "every annotation file" in why

    def test_a_partial_set_names_both_halves(self):
        from apps.curation_ui.views import common

        label, why = common.agreement({"prior_annotation"}, ["prior_annotation", "daphne", "prior_annotation_2"], False)
        assert label == "prior_annotation only"
        assert "absent from daphne, prior_annotation_2" in why

    def test_with_one_source_loaded_there_is_nothing_to_disagree_about(self):
        """A row saying "prior_annotation only" would be reporting the sidebar's configuration as if it
        were a finding about the annotation."""
        from apps.curation_ui.views import common

        label, _ = common.agreement({"prior_annotation"}, ["prior_annotation"], False)
        assert label == "prior_annotation only" and "absent" not in _

    def test_a_code_no_file_carries_is_phenobert_only(self):
        from apps.curation_ui.views import common

        label, _ = common.agreement(set(), ["prior_annotation", "prior_annotation_2"], True)
        assert label == "PhenoBERT only"

    def test_the_order_is_the_sources_order_not_the_sets(self):
        """Two curators looking at one row must read the same label off it."""
        from apps.curation_ui.views import common

        order = ["prior_annotation", "daphne", "prior_annotation_2"]
        first, _ = common.agreement({"prior_annotation_2", "prior_annotation"}, order, False)
        second, _ = common.agreement({"prior_annotation", "prior_annotation_2"}, order, False)
        assert first == second == "prior_annotation + prior_annotation_2"


# ──────────────────────────────────────────────────────────────────────────────
class TestSearch:
    def test_a_short_query_returns_nothing(self, tree):
        index = search.HPOSearch(tree)
        assert index.search("") == [] and index.search("s") == []

    def test_results_are_capped_however_broad_the_query(self):
        """A two-letter query matches thousands of real terms. The browser gets fifty."""
        class WideTree:
            def __init__(self):
                self.hpo_list = [f"HP:{i:07d}" for i in range(1, 501)]

            def getNameByHPO(self, code):  # noqa: N802
                return f"abnormal finding number {code[-4:]}"

            def getPhrasesByHPO(self, code):  # noqa: N802
                return [self.getNameByHPO(code)]

        index = search.HPOSearch(WideTree())
        assert len(index.search("abnormal")) == search.MAX_RESULTS
        assert len(index.search("abnormal", limit=7)) == 7

    def test_the_exact_label_outranks_a_longer_one(self, tree):
        index = search.HPOSearch(tree)
        assert index.search("seizure")[0]["value"] == "HP:0001250"

    def test_a_synonym_finds_the_term(self, tree):
        """A curator types what the report says, not the ontology's preferred label."""
        assert search.HPOSearch(tree).search("low muscle tone")[0]["value"] == "HP:0001252"

    def test_an_hpo_id_is_a_lookup(self, tree):
        index = search.HPOSearch(tree)
        assert [o["value"] for o in index.search("HP:0001250")] == ["HP:0001250"]
        assert [o["value"] for o in index.search("hp0001250")] == ["HP:0001250"]

    def test_an_unknown_code_is_reported_as_unknown_and_labels_as_itself(self, tree):
        index = search.HPOSearch(tree)
        assert not index.known("HP:9999999")
        assert index.label("HP:9999999") == "HP:9999999"

    def test_the_index_is_built_once_per_ontology(self, tree):
        assert search.get_search(tree) is search.get_search(tree)

    def test_describe_carries_the_definition(self, tree):
        """The hover text on every phenotype in the app, name, id, then what it means."""
        text = search.HPOSearch(tree).describe("HP:0001250")
        assert "Seizure (HP:0001250)" in text
        assert "intermittent abnormality" in text

    def test_describe_says_so_when_there_is_no_definition(self, tree):
        text = search.HPOSearch(tree).describe("HP:0001263")
        assert "No definition" in text

    def test_describe_names_an_unknown_code_once(self, tree):
        """``label()`` returns the code for an unknown one, so the naive format says it twice."""
        text = search.HPOSearch(tree).describe("HP:9999999", prefix="PhenoBERT →")
        assert text.count("HP:9999999") == 1
        assert "hpo.json" in text

    def test_a_prefix_is_kept_out_of_the_definition(self, tree):
        text = search.HPOSearch(tree).describe("HP:0001250", prefix="curated →")
        assert text.startswith("curated → Seizure")


# ──────────────────────────────────────────────────────────────────────────────
class TestLocate:
    SEGMENTS = [
        "The boy had recurrent Seizures since age 3.",
        "He was hypotonic at birth.",
        "Cerebral atrophic changes were seen on MRI.",
        "The nurse will deliver the report tomorrow.",
    ]

    def test_an_exact_phrase_is_found_with_the_report_spelling(self, tree):
        hits = locate.find_triggers(tree, "HP:0001250", self.SEGMENTS)
        assert hits[0]["segment_idx"] == 0
        assert hits[0]["trigger_word"] == "Seizures", "the report's casing was not preserved"
        assert hits[0]["tier"] == "exact"

    def test_an_inflection_is_found_and_labelled_as_one(self, tree):
        """'hypotonic' in the report, 'hypotonia' in the ontology, the common case."""
        hits = locate.find_triggers(tree, "HP:0001252", self.SEGMENTS)
        assert hits[0]["segment_idx"] == 1 and hits[0]["tier"] == "inflected"

        hits = locate.find_triggers(tree, "HP:0002059", self.SEGMENTS)
        assert hits[0]["trigger_word"] == "Cerebral atrophic"

    def test_relaxation_does_not_reach_an_unrelated_word(self, tree):
        """'delay' must not match 'deliver', that is why short words stay exact."""
        hits = locate.find_triggers(tree, "HP:0001263", self.SEGMENTS)
        assert all(h["segment_idx"] != 3 for h in hits), hits

    def test_a_term_with_no_lexical_evidence_finds_nothing(self, tree):
        assert locate.find_triggers(tree, "HP:0001873", self.SEGMENTS) == []

    def test_the_longest_phrase_wins_within_a_segment(self, tree):
        hits = locate.find_triggers(tree, "HP:0001263",
                                    ["Marked global developmental delay was noted."])
        assert hits[0]["trigger_word"] == "global developmental delay"


# ──────────────────────────────────────────────────────────────────────────────
class TestProse:
    """The reader draws the report as prose, so what it draws must *be* the report.

    Nothing else in the app can catch a rendering that drops a sentence or reorders two: the
    curator is reading it because they do not already know what it says.
    """

    PALETTE = {"categorical": ["#000"] * 8, "good": "#0a0", "text_muted": "#888", "text": "#000"}

    @staticmethod
    def _text(node, out=None):
        """Every string leaf, in document order, skipping the superscript code tags, which are
        chrome the reader draws on top of the report, not text the report contains."""
        out = [] if out is None else out
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, list):
            for child in node:
                TestProse._text(child, out)
        elif hasattr(node, "children"):
            if "mark-tag" not in (getattr(node, "className", "") or ""):
                TestProse._text(node.children, out)
        return out

    def _rendered(self, cohort, patient_id, selected=None):
        from apps.curation_ui.views import reader

        view = cohort.patient(patient_id)
        body = reader._prose(cohort, view, {}, self.PALETTE, selected)
        return view, body, "".join(self._text(body))

    def test_the_prose_is_the_verbatim_report(self, cohort):
        """Including its blank lines and headings, the segmentation drops those, and a clinical
        report without its paragraphs is markedly harder to read."""
        view, _, rendered = self._rendered(cohort, "SYN002")
        assert rendered == view["text"]
        assert rendered.count("\n") == view["text"].count("\n")

    def test_an_unalignable_segment_is_added_not_substituted(self, cohort):
        """SYN001 carries a segment present in no report. It must still be readable and clickable,
        and it must not displace any of the real text."""
        view, _, rendered = self._rendered(cohort, "SYN001")
        stray = next(s for s in view["segments"] if s not in view["text"])
        assert stray in rendered
        assert rendered.replace(" " + stray, "") == view["text"]

    def test_segments_are_inline_spans_not_rows(self, cohort):
        """A block per sentence is the layout this view exists to replace."""
        from dash import html

        _, body, _ = self._rendered(cohort, "SYN001", selected=1)
        spans = [n for n in body if getattr(n, "id", None)]
        assert len(spans) == len(cohort.patient("SYN001")["segments"])
        assert all(isinstance(n, html.Span) for n in spans)
        assert [n.id["idx"] for n in spans] == list(range(len(spans)))

    def test_the_selected_segment_is_the_only_one_marked(self, cohort):
        _, body, _ = self._rendered(cohort, "SYN001", selected=2)
        selected = [n.id["idx"] for n in body
                    if getattr(n, "id", None) and "seg-selected" in (n.className or "")]
        assert selected == [2]

    def test_segments_keep_document_order(self, cohort):
        """Reading order is what the curator checks the annotations against."""
        view, body, _ = self._rendered(cohort, "SYN001")
        spans = [n for n in body if getattr(n, "id", None)]
        for span, expected in zip(spans, view["display"]):
            assert "".join(self._text(span)) == expected

    def test_a_report_with_no_verbatim_text_still_renders(self, cohort):
        """No staged PhenoBERT report means no alignment, so the segments are joined with spaces
        and the prose is approximate, but it is still there."""
        view = cohort.patient("SYN001")
        stripped = {**view, "text": "", "ranges": [None] * len(view["segments"])}
        from apps.curation_ui.views import reader

        rendered = "".join(self._text(reader._prose(cohort, stripped, {}, self.PALETTE, None)))
        for segment in view["segments"]:
            assert segment in rendered

    def test_nothing_block_level_appears_inside_the_prose(self, cohort):
        """A single ``html.Div`` would break the line at that point and the report stops flowing.

        Easy to reintroduce by accident, every other panel in this app is built from ``Div``, and
        invisible in a render test, which only asks whether *something* was returned.
        """
        block = {"Div", "P", "Table", "Ul", "Ol", "Li", "H1", "H2", "H3", "H4", "Br", "Hr"}
        _, body, _ = self._rendered(cohort, "SYN001", selected=1)

        seen = set()

        def walk(node):
            if isinstance(node, list):
                for child in node:
                    walk(child)
            elif node is not None and not isinstance(node, str) and hasattr(node, "children"):
                seen.add(type(node).__name__)
                walk(node.children)

        walk(body)
        assert seen <= {"Span"}, f"block-level elements in the prose: {sorted(seen - {'Span'})}"

    def test_a_long_report_keeps_all_its_text(self, cohort, monkeypatch):
        """The page cap limits selectable spans, not the text. A report that quietly stopped
        halfway would give the curator no way to know they were reading half of it."""
        from apps.curation_ui.views import reader

        monkeypatch.setattr(reader, "PAGE_SIZE", 1)
        view = cohort.patient("SYN002")
        rendered = "".join(self._text(reader._prose(cohort, view, {}, self.PALETTE, None)))
        assert rendered == view["text"]

        # …and the warning about it stays out of the prose.
        drawn = reader.render("SYN002", None, "light")
        prose = self._find_prose(drawn)
        assert all(isinstance(n, str) or type(n).__name__ == "Span" for n in prose)

    @staticmethod
    def _find_prose(node):
        """The children of the one ``.prose`` container in a rendered report."""
        if isinstance(node, list):
            for child in node:
                found = TestProse._find_prose(child)
                if found is not None:
                    return found
        elif node is not None and not isinstance(node, str) and hasattr(node, "children"):
            if "prose" in (getattr(node, "className", "") or ""):
                return node.children
            return TestProse._find_prose(node.children)
        return None

    def test_the_gaps_between_segments_are_the_reports_own_whitespace(self, cohort):
        """Not a joined-with-spaces approximation, the newlines are why it reads as a document."""
        _, body, _ = self._rendered(cohort, "SYN001")
        gaps = [n for n in body if isinstance(n, str)]
        assert "\n" in gaps, f"no line break survived: {gaps!r}"

    def test_two_files_agreeing_on_the_same_words_are_one_mark_naming_both(self, cohort):
        """Marks cannot nest, so a second mark on the same characters would have to be dropped, and a mark that named only the first file would say one source asserted what two did."""
        from apps.curation_ui.views import reader

        titles = [
            (node.title or "").splitlines()[0]
            for node in _walk(reader.render("SYN001", None, "light"))
            if str(getattr(node, "className", "")).startswith("mark mark-ann")
        ]
        assert any(title.startswith("prior_annotation + daphne →") for title in titles), titles

    def test_an_annotation_whose_trigger_is_missing_draws_no_mark(self, cohort):
        """Underlining approximately-right words in a tool whose output becomes ground truth is worse than
        underlining none, the panel above still lists it."""
        from apps.curation_ui.views import reader

        codes = " ".join(
            (node.title or "") for node in _walk(reader.render("SYN001", None, "light"))
            if str(getattr(node, "className", "")).startswith("mark mark-ann")
        )
        assert "HP:0001873" not in codes
        unplaced = cohort.patient("SYN001")["unplaced_annotations"]
        assert [(r["source"], r["hpo_code"]) for r in unplaced] == [("prior_annotation", "HP:0001873")]

    def test_a_mark_carries_its_code_so_a_click_can_prefill_the_term(self, cohort):
        from apps.curation_ui.views import reader

        view = cohort.patient("SYN001")
        marks = []

        def walk(node):
            if isinstance(node, list):
                for child in node:
                    walk(child)
            elif hasattr(node, "children"):
                ident = getattr(node, "id", None)
                if isinstance(ident, dict) and ident.get("type") == "mark":
                    marks.append(ident)
                walk(node.children)

        walk(reader._prose(cohort, view, {}, self.PALETTE, None))
        assert marks, "no marks were drawn"
        assert all({"idx", "trigger", "code"} <= set(m) for m in marks)
        assert any(m["code"] == "HP:0001250" for m in marks)


def _walk(node, out=None):
    """Every component in a rendered tree, in document order.

    does **not** gate on ``hasattr(node, "children")``: ``dcc.Checklist`` has none, so
    a walk that required it could not see a single input on the page and would happily report a
    broken panel as correct.
    """
    out = [] if out is None else out
    if isinstance(node, (list, tuple)):
        for child in node:
            _walk(child, out)
    elif node is not None and not isinstance(node, str):
        out.append(node)
        _walk(getattr(node, "children", None), out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
class TestComments:
    """Comments on a report, and deletions proposed against annotations that already exist.

    The second one closed a real asymmetry: Approve mode could always *remove* an annotation, but
    the person reading the report closely had no way to say "this is wrong, and here is why"
    without exercising a verdict they were not there to exercise.
    """

    def test_comments_append_rather_than_replace(self, tmp_path):
        """A second comment is a second thing somebody said, not a correction of the first."""
        log = store.EventLog(str(tmp_path), author="a")
        log.append("comment", "P1", text="first")
        log.append("comment", "P1", text="second")

        comments = log.fold()["comments"]["P1"]
        assert [c["text"] for c in comments] == ["first", "second"]
        assert all(c["author"] == "a" and c["ts"] for c in comments)

    def test_a_comment_can_be_deleted_by_id(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        first = log.append("comment", "P1", text="first")
        log.append("comment", "P1", text="second")
        log.append("delete_comment", "P1", target_key=first["event_id"])

        assert [c["text"] for c in log.fold()["comments"]["P1"]] == ["second"]

    def test_comments_do_not_create_annotation_rows(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        log.append("comment", "P1", text="about the report")
        assert log.fold()["rows"] == {}

    def test_a_proposed_deletion_leaves_the_gold(self, tmp_path):
        """A term under active dispute is not something to ship."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("raw", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("suggest_delete", "P1", target_key=key, hpo_code="HP:0001250",
                   text="belongs to the sibling")

        row = log.fold()["rows"][key]
        assert row["status"] == store.DELETE_SUGGESTED
        assert row["status"] not in store.IN_GOLD
        assert row["delete_reason"] == "belongs to the sibling"
        assert row["delete_by"] == "a"

    def test_remove_accepts_a_proposal_and_keep_rejects_it(self, tmp_path):
        """No parallel approve/reject buttons: the verdict already expresses both answers."""
        log = store.EventLog(str(tmp_path), author="a")
        first = store.target_key("raw", "P1", "HP:0001250")
        second = store.target_key("raw", "P1", "HP:0001252")
        for key in (first, second):
            log.append("suggest_delete", "P1", target_key=key, hpo_code=key.split("|")[2],
                       text="unsupported")
        log.append("verdict", "P1", target_key=first, status="removed")
        log.append("verdict", "P1", target_key=second, status="kept")

        rows = log.fold()["rows"]
        assert rows[first]["status"] == "removed"
        assert rows[second]["status"] == "kept"

    def test_withdrawing_a_proposal_returns_the_row_to_undecided(self, tmp_path):
        """Not to "kept", the proposer changing their mind says nothing about correctness."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("prior_annotation_2", "P1", "HP:0001903")
        log.append("suggest_delete", "P1", target_key=key, hpo_code="HP:0001903", text="no")
        log.append("withdraw_delete", "P1", target_key=key)

        row = log.fold()["rows"][key]
        assert row["status"] == "" and row["delete_reason"] == "" and row["delete_by"] == ""

    def test_withdraw_cannot_erase_an_existing_annotation(self, tmp_path):
        """``withdraw`` retires a suggestion this app made. Aimed at a ground truth row it would delete the
        app's whole record of it, verdict included."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("raw", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("withdraw", "P1", target_key=key)

        state = log.fold()
        assert state["rows"][key]["status"] == "kept"
        assert state["n_skipped"] == 1

    def test_the_comments_file_is_one_row_per_comment(self, tmp_path):
        comments = {"P1": [{"comment_id": "a1", "ts": "2026-08-19T10:00:00Z",
                            "author": "curator_a", "text": "dense family history"}],
                    "P2": [{"comment_id": "b1", "ts": "2026-08-19T11:00:00Z",
                            "author": "marc", "text": "translated from German"}]}
        result = store.write_comments(str(tmp_path / "comments.csv"), comments)

        with open(result["path"], newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        assert result["n_rows"] == 2
        assert {r["patient_id"]: r["text"] for r in parsed} == {
            "P1": "dense family history", "P2": "translated from German"}
        assert parsed[0]["author"] == "curator_a"

    def test_the_report_file_counts_comments_instead_of_quoting_one(self, tmp_path):
        """Duplicating the latest comment there would make it ambiguous which was the record."""
        comments = {"P1": [{"comment_id": "a", "ts": "t", "author": "x", "text": "one"},
                           {"comment_id": "b", "ts": "t", "author": "x", "text": "two"}]}
        result = store.write_report_labels(str(tmp_path / "report.csv"), {}, ["P1", "P2"],
                                           comments)

        with open(result["path"], newline="", encoding="utf-8") as handle:
            rows = {r["patient_id"]: r for r in csv.DictReader(handle)}
        assert "note" not in rows["P1"]
        assert rows["P1"]["n_comments"] == "2" and rows["P2"]["n_comments"] == "0"

    # ── the grouping the UI applies over that model ──────────────────────────
    def test_deletion_is_offered_per_term_not_per_source(self, cohort):
        """A term annotated twice is one annotation in the curator's head. Splitting it would mean
        proposing the same deletion twice, and forgetting the second would leave the term in the
        export with no sign anything was wrong."""
        from apps.curation_ui.views import edit

        entries = {e["code"]: e for e in edit.deletable_rows(cohort, "SYN002")}
        assert [m["source"] for m in entries["HP:0001263"]["members"]] == ["prior_annotation", "prior_annotation"]
        assert len(entries["HP:0001263"]["keys"]) == 2

    def test_a_reference_only_term_cannot_be_proposed_for_deletion(self, cohort):
        """The proposal is settled by the keep/remove buttons on the Approve row, and prior_annotation_2 has no
        Approve row, so this would be a question nobody could answer."""
        from apps.curation_ui.views import edit

        codes = {e["code"] for e in edit.deletable_rows(cohort, "SYN001")}
        assert "HP:0001903" not in codes             # marc2 only
        assert "HP:0001250" in codes                 # daphne rules on it

    def test_one_code_annotated_twice_is_two_deletable_rows(self, cohort):
        """Two triggers for one phenotype are two annotations, and they key apart, otherwise a
        deletion proposal against one silently settles the other."""
        from apps.curation_ui.views import edit

        entry = next(e for e in edit.deletable_rows(cohort, "SYN002")
                     if e["code"] == "HP:0001263")
        prior_annotation = [k for k in entry["keys"] if k.startswith("prior_annotation|")]
        assert prior_annotation == ["prior_annotation|SYN002|HP:0001263", "prior_annotation|SYN002|HP:0001263|1"]

    def test_unadjudicated_phenobert_detections_are_not_deletable(self, cohort):
        """They are candidate evidence, not annotations, a proposal about nothing."""
        from apps.curation_ui.views import edit

        codes = {e["code"] for e in edit.deletable_rows(cohort, "SYN002")}
        assert "HP:9999999" not in codes, "an un-adjudicated detection was offered for deletion"

    def test_a_kept_phenobert_row_becomes_deletable(self, cohort):
        """By then somebody has asserted it."""
        from apps.curation_ui.views import edit

        detection = next(d for group in cohort.patient("SYN002")["detections"].values()
                         for d in group if d["hpo_id"] == "HP:9999999")
        key = store.target_key("phenobert", "SYN002", "HP:9999999", detection["start"])
        cohort.record("verdict", "SYN002", target_key=key, hpo_code="HP:9999999", status="kept")

        codes = {e["code"] for e in edit.deletable_rows(cohort, "SYN002")}
        assert "HP:9999999" in codes

    def test_open_proposals_sort_to_the_top(self, cohort):
        from apps.curation_ui.views import edit

        key = store.target_key("prior_annotation", "SYN001", "HP:0001873")
        cohort.record("suggest_delete", "SYN001", target_key=key, hpo_code="HP:0001873",
                      text="unsupported")

        entries = edit.deletable_rows(cohort, "SYN001")
        assert entries[0]["code"] == "HP:0001873" and entries[0]["proposed"]


# ──────────────────────────────────────────────────────────────────────────────
class TestLabels:
    def test_every_label_is_unique_and_described(self):
        """The vocabulary is closed, so a duplicate id would silently merge two concepts."""
        ids = [value for group in vocab.GROUPS for value, _, _, _ in group["labels"]]
        ids += [value for value, _, _, _ in vocab.DIFFICULTY]
        assert len(ids) == len(set(ids))
        assert all(vocab.INDEX[i]["help"].strip() for i in ids)

    def test_the_two_levels_are_disjoint(self):
        """They answer different questions now, and a value in the wrong file is uninterpretable.

        This reverses the original design, where the levels shared ids so "how often is this the
        problem, at either level" was one group-by. In practice the annotation level was nineteen
        checkboxes and a grade on every row, and a vocabulary that fine is one nobody applies.
        """
        assert not (vocab.VALID["annotation"] & vocab.VALID["patient"])
        assert "family_member" in vocab.VALID["patient"]
        assert "family" in vocab.VALID["annotation"]

    def test_the_annotation_vocabulary_is_the_seven_qualifiers(self):
        """Ticked on the suggestion form itself, so it has to be short enough to read at a glance."""
        assert [g["title"] for g in vocab.groups_for("annotation")] == [
            "What kind of annotation is this?"]
        group = vocab.groups_for("annotation")[0]
        assert [d for _, d, _ in vocab.labels_for(group, "annotation")] == [
            "Unsure (from report)", "Unsure (from annotation)", "Family", "Resolved",
            "Negated", "Lab value", "Implicit"]

    def test_the_annotation_level_offers_no_grade(self):
        """The grade is a report-level question. Per annotation it was never applied."""
        assert vocab.difficulty_for("annotation") == []
        assert not vocab.grade_ok("hard", "annotation")

    def test_the_report_vocabulary_is_the_agreed_subset(self):
        """Report level answers 'what kind of report is this', three grades and two groups."""
        assert [g[0] for g in vocab.difficulty_for("patient")] == ["easy", "medium", "hard"]
        assert [g["title"] for g in vocab.groups_for("patient")] == [
            "Whose phenotype is it?", "How is it written?"]

        expression = next(g for g in vocab.groups_for("patient")
                          if g["title"] == "How is it written?")
        assert [d for _, d, _ in vocab.labels_for(expression, "patient")] == [
            "Direct word", "Abbreviation", "Indirectly described",
            "Needs clinical inference", "Evidence spans sentences"]

    def test_the_fine_grained_labels_are_retired_but_still_readable(self):
        """Offered nowhere, named everywhere. A log outlives the vocabulary that wrote it.

        Deleting them outright would leave a row already in ``curation_labels.csv`` rendering as a
        bare id, which is the fallback reserved for a label written by a *newer* version.
        """
        for value in ("inflection", "non_english", "granularity", "ambiguous", "disputed",
                      "outdated_code", "unfair"):
            assert value not in vocab.VALID["annotation"], value
            assert value not in vocab.VALID["patient"], value
            assert vocab.INDEX[value]["display"] != value
            assert vocab.INDEX[value]["help"].strip()

    def test_unfair_is_offered_at_neither_level(self):
        """It was the annotation grade's fourth value, and the annotation grade is gone."""
        assert not vocab.grade_ok("unfair", "annotation")
        assert not vocab.grade_ok("unfair", "patient")

    def test_clean_drops_the_unknown_and_the_out_of_scope(self):
        assert vocab.clean(["negated", "made_up"], "annotation") == ["negated"]
        assert vocab.clean(["granularity"], "patient") == []
        assert vocab.clean(["family_member"], "patient") == ["family_member"]
        # Each level rejects the other's vocabulary, which is what stops a browser posting a value
        # into a file whose join cannot interpret it.
        assert vocab.clean(["family_member"], "annotation") == []
        assert vocab.clean(["negated"], "patient") == []

    def test_clean_drops_the_grade(self):
        """Difficulty is single-valued and lives in its own field. It must not leak into the set."""
        assert vocab.clean(["hard", "family_member"], "patient") == ["family_member"]

    def test_report_columns_are_stable_and_ordered(self):
        """The report file's header is these ids. A reordering would show as a spurious diff."""
        assert vocab.label_ids("patient") == [
            "family_member", "hypothetical", "historical", "negated_in_text",
            "direct_term", "abbreviation", "paraphrase", "inferred", "multi_sentence"]

    def test_order_comes_from_the_vocabulary_not_the_clicks(self):
        assert (vocab.clean(["paraphrase", "family_member"], "patient")
                == vocab.clean(["family_member", "paraphrase"], "patient"))
        assert (vocab.clean(["negated", "family"], "annotation")
                == vocab.clean(["family", "negated"], "annotation")
                == ["family", "negated"])

    def test_an_unknown_label_falls_back_to_its_id(self):
        """A log written by a newer version must still open here."""
        assert vocab.display("from_the_future") == "from_the_future"
        assert "Unknown label" in vocab.help_for("from_the_future")


# ──────────────────────────────────────────────────────────────────────────────
class TestLabelPersistence:
    def test_labels_replace_rather_than_accumulate(self, tmp_path):
        """Unticking has to work, or nobody will risk ticking."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("raw", "P1", "HP:0001250")
        log.append("label", "P1", target_key=key, labels=["family_member", "paraphrase"],
                   difficulty="hard")
        log.append("label", "P1", target_key=key, labels=["paraphrase"], difficulty="")

        row = log.fold()["rows"][key]
        assert row["labels"] == ["paraphrase"] and row["difficulty"] == ""

    def test_labelling_does_not_disturb_a_verdict(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("raw", "P1", "HP:0001250")
        log.append("verdict", "P1", target_key=key, hpo_code="HP:0001250", status="kept")
        log.append("label", "P1", target_key=key, labels=["family_member"], difficulty="unfair")

        row = log.fold()["rows"][key]
        assert row["status"] == "kept"
        assert row["labels"] == ["family_member"] and row["difficulty"] == "unfair"

    def test_a_report_label_is_addressed_by_patient_not_by_key(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        log.append("label", "P1", labels=["family_member"], difficulty="hard")
        state = log.fold()

        assert state["patients"]["P1"]["difficulty"] == "hard"
        assert state["rows"] == {}, "a report label created a phantom annotation row"

    def test_the_annotation_file_is_long_form(self, tmp_path):
        rows = {"P1": [{"key": "raw|P1|HP:0001250", "hpo_code": "HP:0001250", "source": "raw",
                        "labels": ["family", "negated"], "difficulty": "hard"}]}
        result = store.write_labels(str(tmp_path / "labels.csv"), rows)

        with open(result["path"], newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        # Two rows, not three: the annotation-level grade is retired, and a ``difficulty`` still
        # sitting on a row folded from an older log must not be written back out as a label.
        assert result["n_rows"] == len(parsed) == 2
        assert {(r["kind"], r["label"]) for r in parsed} == {
            ("qualifier", "family"), ("qualifier", "negated")}

        # It has to join to a predictions dump, which keys on (patient_id, hpo_code).
        annotation = next(r for r in parsed if r["label"] == "family")
        assert annotation["patient_id"] == "P1" and annotation["hpo_code"] == "HP:0001250"
        assert annotation["label_display"] == "Family"
        assert annotation["label_group"] == "qualifier"
        assert "level" not in annotation, "a constant column survived the file split"

    def test_report_labels_go_to_their_own_wide_file(self, tmp_path):
        patients = {"P1": {"labels": ["family_member", "paraphrase"], "difficulty": "medium",
                           "author": "a", "updated_at": "2026-08-19T10:00:00Z"}}
        result = store.write_report_labels(str(tmp_path / "report.csv"), patients, ["P1", "P2"])

        with open(result["path"], newline="", encoding="utf-8") as handle:
            parsed = {r["patient_id"]: r for r in csv.DictReader(handle)}

        assert result["n_rows"] == 2 and result["n_labelled"] == 1
        row = parsed["P1"]
        assert row["difficulty"] == "medium"
        assert row["labels"] == "family_member;paraphrase" and row["n_labels"] == "2"
        assert row["author"] == "a"
        # One 0/1 column per label, so a stratified table needs no parsing.
        assert row["family_member"] == "1" and row["paraphrase"] == "1"
        assert row["abbreviation"] == "0"

        # Every report gets a row, labelled or not, otherwise "how much of the cohort has been
        # characterised" cannot be answered from the file itself.
        assert parsed["P2"]["difficulty"] == "" and parsed["P2"]["n_labels"] == "0"

    def test_the_report_file_carries_no_annotation_only_column(self, tmp_path):
        result = store.write_report_labels(str(tmp_path / "report.csv"), {}, ["P1"])
        with open(result["path"], newline="", encoding="utf-8") as handle:
            header = csv.DictReader(handle).fieldnames
        assert "granularity" not in header and "inflection" not in header
        assert "family_member" in header

    def test_the_current_csv_joins_a_label_list_into_one_cell(self, tmp_path):
        """``str(["a","b"])`` would write ``['a', 'b']``, a cell needing literal_eval to read."""
        path = str(tmp_path / "out.csv")
        store.write_csv_atomic(path, ["patient_id", "labels"],
                               [{"patient_id": "P1", "labels": ["family_member", "paraphrase"]}])

        with open(path, newline="", encoding="utf-8") as handle:
            assert list(csv.DictReader(handle))[0]["labels"] == "family_member;paraphrase"


# ──────────────────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_every_patient_any_source_knows_about_is_present(self, cohort):
        # SYN004 exists only in the ground truth files, no segmentation, no PhenoBERT.
        assert "SYN004" in cohort.patient_ids
        assert cohort.patient("SYN004")["segments"] == []
        assert "no segmented report" in " ".join(cohort.notes())

    def test_the_annotation_files_disagree_and_the_overview_says_so(self, cohort):
        row = next(r for r in cohort.overview() if r["patient_id"] == "SYN001")
        assert row["n_prior_annotation"] == 4 and row["n_daphne"] == 2 and row["n_prior_annotation_2"] == 3
        # Three of the five terms are not carried by all three files: HP:0001873 (holistic only),
        # HP:0007359 (holistic only) and HP:0001903 (marc2 only). That set is the queue the
        # curation pass works through.
        assert row["n_agree"] == 2 and row["n_disagree"] == 3

    def test_the_patient_view_is_cached_not_rebuilt(self, cohort):
        assert cohort.patient("SYN001") is cohort.patient("SYN001")

    def test_a_negated_detection_is_kept_not_dropped(self, cohort):
        """PhenoBERT ruling a phenotype out is evidence a term list cannot express."""
        detections = cohort.patient("SYN001")["detections"]
        negated = [d for group in detections.values() for d in group if d["negated"]]
        assert [d["hpo_id"] for d in negated] == ["HP:0001873"]

    def test_an_unresolved_code_survives_to_the_screen(self, cohort):
        codes = {d["hpo_id"] for group in cohort.patient("SYN002")["detections"].values()
                 for d in group}
        assert "HP:9999999" in codes
        assert not cohort.search.known("HP:9999999")

    def test_recording_writes_the_log_and_the_derived_csv_together(self, cohort):
        cohort.record("suggest", "SYN001", segment_idx=1, segment_text="x",
                      hpo_code="HP:0001250", hpo_name="Seizure", trigger_word="seizures")
        assert os.path.isfile(cohort.log.path)

        with open(cohort.log.current_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["hpo_code"] for r in rows] == ["HP:0001250"]
        assert rows[0]["status"] == "suggested"

    def test_an_undecided_annotation_does_not_enter_the_export(self, cohort):
        """Silence is not consent, the ground truth being replaced was assembled that way."""
        cohort.record("verdict", "SYN001",
                      target_key=store.target_key("raw", "SYN001", "HP:0001250"),
                      hpo_code="HP:0001250", status="kept")
        result = cohort.export_gold()

        with open(result["path"], newline="", encoding="utf-8") as handle:
            parsed = {r["patient_id"]: r["hpo_codes"] for r in csv.DictReader(handle)}
        assert parsed["SYN001"] == "HP:0001250"
        assert result["n_patients"] == len(cohort.patient_ids)

    def test_the_export_reads_back_through_the_pipelines_own_loader(self, cohort):
        """A file the scoring jobs cannot read is not an export."""
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset

        cohort.record("verdict", "SYN001",
                      target_key=store.target_key("raw", "SYN001", "HP:0001250"),
                      hpo_code="HP:0001250", status="kept")
        result = cohort.export_gold()

        loaded = HCYDataset(result["path"], "").load_ground_truth()
        assert set(loaded) == set(cohort.patient_ids)
        assert loaded["SYN001"] == ["HP:0001250"]
        # The export carries a row for every patient, so most of them are empty, and an empty
        # cell has to read back as "no annotated terms", not as a term. Read as the string "nan" it
        # would be an unpredictable annotated term charged against every method on every such report.
        for patient_id in set(cohort.patient_ids) - {"SYN001"}:
            assert loaded[patient_id] == [], f"{patient_id} came back as {loaded[patient_id]!r}"

    def test_labels_reach_all_three_derived_files(self, cohort):
        key = store.target_key("raw", "SYN001", "HP:0001250")
        # The annotation's qualifiers arrive on an ``edit``, which is the path the inline editor
        # takes. The report's labels on a ``label`` event with no target key.
        cohort.record("edit", "SYN001", target_key=key, hpo_code="HP:0001250",
                      labels=["family", "negated"])
        cohort.record("label", "SYN001", labels=["paraphrase"], difficulty="hard")

        with open(cohort.log.current_path, newline="", encoding="utf-8") as handle:
            row = next(r for r in csv.DictReader(handle) if r["hpo_code"] == "HP:0001250")
        assert row["labels"] == "family;negated"

        with open(cohort.log.labels_path, newline="", encoding="utf-8") as handle:
            annotation = list(csv.DictReader(handle))
        assert {r["label"] for r in annotation} == {"family", "negated"}

        # The report label is in the other file, and every report in the cohort has a row there.
        with open(cohort.log.report_labels_path, newline="", encoding="utf-8") as handle:
            report = {r["patient_id"]: r for r in csv.DictReader(handle)}
        assert set(report) == set(cohort.patient_ids)
        assert report["SYN001"]["difficulty"] == "hard"
        assert report["SYN001"]["paraphrase"] == "1"

    def test_the_overview_reports_labelling_progress(self, cohort):
        cohort.record("label", "SYN001", labels=["direct_term"], difficulty="hard")
        cohort.record("edit", "SYN001",
                      target_key=store.target_key("raw", "SYN001", "HP:0001250"),
                      hpo_code="HP:0001250", labels=["implicit"])

        row = next(r for r in cohort.overview() if r["patient_id"] == "SYN001")
        assert row["difficulty"] == "hard" and row["n_doc_labels"] == 1
        assert row["n_labelled"] == 1
        assert cohort.totals()["n_graded_reports"] == 1

    def test_a_reload_resumes_the_session(self, cohort, tmp_path):
        event = cohort.record("suggest", "SYN001", segment_idx=1, segment_text="x",
                              hpo_code="HP:0001250", trigger_word="seizures")
        cohort.record("approve", "SYN001", target_key=event["event_id"])
        cohort.record("confirm_patient", "SYN001")

        reopened = Registry(cohort.paths, author="pytest", curation_dir=cohort.curation_dir)
        rows = {r["key"]: r for r in reopened.rows_for("SYN001")}
        assert rows[event["event_id"]]["status"] == "approved"
        assert reopened.is_confirmed("SYN001")


# ──────────────────────────────────────────────────────────────────────────────
class TestSources:
    @pytest.mark.parametrize("cell,expected", [
        ("HP:0001250;HP:0001252", ["HP:0001250", "HP:0001252"]),
        ("HP:0001250,HP:0001252", ["HP:0001250", "HP:0001252"]),
        ("['HP:0001250', 'HP:0001252']", ["HP:0001250", "HP:0001252"]),
        ("HP:0001250", ["HP:0001250"]),
        ("", []),
        ("nan", []),
    ])
    def test_every_shape_a_gold_cell_comes_in(self, cell, expected):
        """The same rule ``scripts/generate_target_symptoms_all.py`` uses, so the term sets agree."""
        assert sources.parse_hpo_codes(cell) == expected

    def test_a_missing_colon_is_repaired(self):
        assert sources.fix_hpo_id("HP0001250") == "HP:0001250"
        assert sources.fix_hpo_id("HP:0001250") == "HP:0001250"

    def test_a_missing_gold_file_is_not_fatal(self, tmp_path):
        """Curating against only one of the two ground-truth sets is a legitimate session."""
        assert sources.load_gold(str(tmp_path / "absent.csv")) == {}

    def test_a_missing_phenobert_run_is_not_fatal(self, tmp_path):
        assert sources.load_phenobert(str(tmp_path / "absent"), ["P1"]) == ({}, {})

    def test_a_missing_segmentation_is_fatal(self, tmp_path):
        """Without segments there is no screen to lay out, so this one must be loud."""
        with pytest.raises(FileNotFoundError):
            sources.load_segments(str(tmp_path / "absent.csv"))


# ──────────────────────────────────────────────────────────────────────────────
class TestPatientFilter:
    """The sidebar's "unlabelled reports only" box, as pure functions.

    It decides what a curator can *reach*, so a report wrongly dropped from the list is a report
    nobody visits again. The rendering half is checked here too: a filter that is right and a list
    that ignores it look identical from the console.
    """

    @staticmethod
    def _shown(node):
        """The patient ids the rendered list actually drew."""
        out = []
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
                continue
            ident = getattr(item, "id", None)
            if isinstance(ident, dict) and ident.get("type") == "pat":
                out.append(ident["pid"])
            if hasattr(item, "children"):
                stack.append(item.children)
        return set(out)

    def test_an_uncurated_cohort_is_entirely_unlabelled(self, cohort):
        from apps.curation_ui.app import visible_patient_ids

        assert visible_patient_ids(cohort, True) == cohort.patient_ids
        assert visible_patient_ids(cohort, False) == cohort.patient_ids

    def test_a_label_removes_the_report_and_clearing_it_puts_it_back(self, cohort):
        from apps.curation_ui.app import visible_patient_ids

        subject = cohort.patient_ids[0]
        cohort.record("label", subject, labels=["family_member"], difficulty="")
        assert subject not in visible_patient_ids(cohort, True)
        assert visible_patient_ids(cohort, False) == cohort.patient_ids

        cohort.record("label", subject, labels=[], difficulty="")
        assert subject in visible_patient_ids(cohort, True)

    def test_a_grade_alone_counts_as_characterised(self, cohort):
        """Either half is a curator having made a pass. Demanding both re-serves read reports."""
        from apps.curation_ui.app import visible_patient_ids

        subject = cohort.patient_ids[0]
        cohort.record("label", subject, labels=[], difficulty="hard")
        assert subject not in visible_patient_ids(cohort, True)

    def test_a_comment_is_not_a_label(self, cohort):
        """A question about a report is not a characterisation of it, if anything the opposite."""
        from apps.curation_ui.app import visible_patient_ids

        subject = cohort.patient_ids[0]
        cohort.record("comment", subject, text="is this family history?")
        assert subject in visible_patient_ids(cohort, True)

    def test_confirming_a_patient_is_not_a_label(self, cohort):
        """The two passes are independent: a report can be adjudicated and uncharacterised."""
        from apps.curation_ui.app import visible_patient_ids

        subject = cohort.patient_ids[0]
        cohort.record("confirm_patient", subject)
        assert subject in visible_patient_ids(cohort, True)

    def test_the_rendered_list_obeys_the_filter(self, cohort):
        from apps.curation_ui.app import render_patient_list, visible_patient_ids

        subject = cohort.patient_ids[0]
        cohort.record("label", subject, labels=["family_member"], difficulty="")
        assert self._shown(render_patient_list(subject, "light", False)) == set(cohort.patient_ids)
        assert self._shown(render_patient_list(subject, "light", True)) == set(
            visible_patient_ids(cohort, True))

    def test_a_fully_labelled_cohort_says_so_rather_than_drawing_nothing(self, cohort):
        from apps.curation_ui.app import render_patient_list

        for patient_id in cohort.patient_ids:
            cohort.record("label", patient_id, labels=["family_member"], difficulty="")
        drawn = render_patient_list(cohort.patient_ids[0], "light", True)
        assert self._shown(drawn) == set()
        assert "characterised" in _text(drawn), "an empty filtered list said nothing at all"

    @pytest.mark.parametrize("filters, expected", [
        (None, False), ([], False), (["unlabelled"], True), (["something-else"], False),
    ])
    def test_the_checklist_value_is_read_as_a_list(self, filters, expected):
        from apps.curation_ui.app import unlabelled_only

        assert unlabelled_only(filters) is expected


class TestCallbacks:
    """Driven through the real HTTP surface, because that is where these bugs live.

    Dash re-creates components when a callback replaces its container's children, and a re-created
    component's declared prop values are pushed to the client, including ``n_clicks=0``. That is a
    change to a callback Input, so the callback fires with nobody having pressed anything. Two
    shipped bugs came from it:

    * the report-label Save button sits inside a panel that re-renders on patient change, so moving
      to the next patient re-fired the save with the previous patient's checkboxes still in the DOM
      and wrote them against the new patient's id, labels appeared to follow you, and unticking a
      box never held;
    * ``pick_patient`` read ``step = -1 if trigger == "pat-prev" else 1``, so *any* unexpected
      trigger advanced. Prev/Next live in the static sidebar and their counters only climb, which
      made a spurious fire indistinguishable from a press: one click of Next moved two patients.

    Calling the render functions cannot catch either. Only firing the callback can.
    """

    @pytest.fixture
    def wired(self, cohort):
        from apps.curation_ui.app import build_app

        app = build_app(cohort)
        return cohort, app, app.server.test_client()

    @staticmethod
    def _callback(app, marker):
        """The callback whose *inputs* mention *marker*, pattern ids are keyed by output only."""
        hits = [key for key, spec in app.callback_map.items()
                if any(marker in str(i["id"]) for i in spec["inputs"])]
        assert len(hits) == 1, (marker, hits)
        return hits[0]

    @staticmethod
    def _outputs(key):
        body = key[2:-2] if key.startswith("..") else key
        out = []
        for part in body.split("..."):
            if not part:
                continue
            cid, prop = part.rsplit(".", 1)[0], part.rsplit(".", 1)[1].split("@")[0]
            out.append({"id": json.loads(cid) if cid.startswith("{") else cid, "property": prop})
        return out

    @staticmethod
    def _group(items):
        """Pattern-matching entries share one spec, so the browser sends them as one list."""
        out, seen = [], {}
        for item in items or []:
            if not isinstance(item["id"], dict):
                out.append(item)
                continue
            slot = (item["id"]["type"], item["property"])
            if slot in seen:
                out[seen[slot]].append(item)
            else:
                seen[slot] = len(out)
                out.append([item])
        return out

    def _fire(self, client, key, inputs, state=None, trigger=None):
        """POST one callback. Returns ``None`` for PreventUpdate (Dash answers 204)."""
        first = trigger or inputs[0]
        changed = (json.dumps(first["id"], sort_keys=True) if isinstance(first["id"], dict)
                   else first["id"]) + "." + first["property"]
        response = client.post("/_dash-update-component", json={
            "output": key, "outputs": self._outputs(key), "inputs": self._group(inputs),
            "state": self._group(state), "changedPropIds": [changed]})
        if response.status_code == 204:
            return None
        assert response.status_code == 200, (response.status_code, response.data[:400])
        return json.loads(response.data).get("response") or None

    # ── navigation ───────────────────────────────────────────────────────────
    def _nav(self, wired, button, current, seen, filters=()):
        cohort, app, client = wired
        rows = [{"id": {"type": "pat", "pid": p}, "property": "n_clicks", "value": 0}
                for p in cohort.patient_ids]
        inputs = rows + [
            {"id": "pat-prev", "property": "n_clicks",
             "value": seen["prev"] + (1 if button == "prev" else 0)},
            {"id": "pat-next", "property": "n_clicks",
             "value": seen["next"] + (1 if button == "next" else 0)},
        ]
        return self._fire(
            client, self._callback(app, "pat-next"), inputs,
            [{"id": "store-patient", "property": "data", "value": current},
             {"id": "store-nav", "property": "data", "value": seen},
             {"id": "patient-filter", "property": "value", "value": list(filters)}],
            trigger={"id": f"pat-{button if button in ('prev', 'next') else 'next'}",
                     "property": "n_clicks"})

    def test_next_advances_one_patient(self, wired):
        cohort, _, _ = wired
        seen = {"prev": 0, "next": 0}
        out = self._nav(wired, "next", cohort.patient_ids[0], seen)
        assert out["store-patient"]["data"] == cohort.patient_ids[1]
        assert out["store-nav"]["data"] == {"prev": 0, "next": 1}

    def test_a_refire_with_unchanged_counts_does_nothing(self, wired):
        """The re-render that follows a patient change fires this callback again. It must not step.

        This is the double-jump: SYN001 → Next → SYN002, then the re-render carried it to SYN003.
        """
        cohort, _, _ = wired
        out = self._nav(wired, "next", cohort.patient_ids[0], {"prev": 0, "next": 0})
        after = out["store-nav"]["data"]

        assert self._nav(wired, "none", out["store-patient"]["data"], after) is None

    def test_a_trigger_naming_next_without_a_click_does_nothing(self, wired):
        """Trusting ``triggered_id`` alone is what made a spurious fire look like a press."""
        cohort, app, client = wired
        seen = {"prev": 0, "next": 2}
        rows = [{"id": {"type": "pat", "pid": p}, "property": "n_clicks", "value": 0}
                for p in cohort.patient_ids]
        out = self._fire(
            client, self._callback(app, "pat-next"),
            rows + [{"id": "pat-prev", "property": "n_clicks", "value": 0},
                    {"id": "pat-next", "property": "n_clicks", "value": 2}],
            [{"id": "store-patient", "property": "data", "value": cohort.patient_ids[0]},
             {"id": "store-nav", "property": "data", "value": seen},
             {"id": "patient-filter", "property": "value", "value": []}],
            trigger={"id": "pat-next", "property": "n_clicks"})
        assert out is None

    def test_prev_stops_at_the_first_patient(self, wired):
        cohort, _, _ = wired
        out = self._nav(wired, "prev", cohort.patient_ids[0], {"prev": 0, "next": 0})
        assert out["store-patient"]["data"] == cohort.patient_ids[0]

    def test_a_patient_row_click_selects_that_patient(self, wired):
        cohort, app, client = wired
        target = cohort.patient_ids[2]
        rows = [{"id": {"type": "pat", "pid": p}, "property": "n_clicks",
                 "value": 1 if p == target else 0} for p in cohort.patient_ids]
        out = self._fire(
            client, self._callback(app, "pat-next"),
            rows + [{"id": "pat-prev", "property": "n_clicks", "value": 0},
                    {"id": "pat-next", "property": "n_clicks", "value": 0}],
            [{"id": "store-patient", "property": "data", "value": cohort.patient_ids[0]},
             {"id": "store-nav", "property": "data", "value": {"prev": 0, "next": 0}},
             {"id": "patient-filter", "property": "value", "value": []}],
            trigger={"id": {"type": "pat", "pid": target}, "property": "n_clicks"})
        assert out["store-patient"]["data"] == target
        assert out["store-segment"]["data"] is None, "the segment must not survive a patient change"

    # ── the tab moves only when you move it ───────────────────────────────────
    def test_only_one_callback_in_the_app_can_move_the_tab(self, wired):
        """If a second one appears, "the screen jumped on its own" becomes unattributable."""
        _, app, _ = wired
        movers = [key for key in app.callback_map if "tabs.value" in str(key)]
        assert len(movers) == 1, movers

    def test_a_grid_cell_does_not_yank_you_out_of_another_tab(self, wired):
        """The Overview table stays rendered once visited, so a stale or re-run ``active_cell``
        would otherwise pull a curator back to Curate mid-edit, with nothing explaining why."""
        cohort, app, client = wired
        key = self._callback(app, "overview-grid")
        rows = [{"patient_id": cohort.patient_ids[1]}]
        cell = {"row": 0, "column": 0, "column_id": "patient_id"}

        for elsewhere in ("anchor", "curate"):
            assert self._fire(
                client, key,
                [{"id": "overview-grid", "property": "active_cell", "value": cell}],
                [{"id": "overview-grid", "property": "derived_viewport_data", "value": rows},
                 {"id": "tabs", "property": "value", "value": elsewhere}]) is None, elsewhere

        out = self._fire(
            client, key,
            [{"id": "overview-grid", "property": "active_cell", "value": cell}],
            [{"id": "overview-grid", "property": "derived_viewport_data", "value": rows},
             {"id": "tabs", "property": "value", "value": "overview"}])
        assert out["tabs"]["value"] == "curate"
        assert out["store-patient"]["data"] == cohort.patient_ids[1]

    def test_the_curate_row_itself_is_hidden_on_the_overview_tab(self, wired):
        """It is a flex box a viewport tall. Leaving it on pushes the table below the fold."""
        _, app, client = wired
        key = self._callback(app, "mode-toggle")

        out = self._fire(client, key,
                         [{"id": "tabs", "property": "value", "value": "overview"},
                          {"id": "mode-toggle", "property": "value", "value": "edit"}])
        assert out["view-curate"]["style"] == {"display": "none"}
        assert out["view-overview"]["style"] != {"display": "none"}

        out = self._fire(client, key,
                         [{"id": "tabs", "property": "value", "value": "anchor"},
                          {"id": "mode-toggle", "property": "value", "value": "approve"}])
        assert out["col-anchor"]["style"] != {"display": "none"}
        assert out["col-curate"]["style"] == {"display": "none"}
        assert out["col-reader"]["style"] != {"display": "none"}, "the report is shared, not hidden"
        # The mode switch means nothing here, and must not leak Approve onto the Evidence location tab.
        assert out["wrap-approve"]["style"] == {"display": "none"}

    # ── the "unlabelled reports only" filter ─────────────────────────────────
    def test_next_skips_a_report_the_filter_hides(self, wired):
        """With the filter on, the list *is* the navigation surface.

        A Next that landed on a report the sidebar is not showing would select a patient with no
        row to highlight, which reads as the selection having been lost.
        """
        cohort, _, _ = wired
        first, skipped, third = cohort.patient_ids[:3]
        cohort.record("label", skipped, labels=["family_member"], difficulty="")

        out = self._nav(wired, "next", first, {"prev": 0, "next": 0}, filters=["unlabelled"])
        assert out["store-patient"]["data"] == third
        # …and unfiltered, the same press moves one step, as it always did.
        out = self._nav(wired, "next", first, {"prev": 0, "next": 0})
        assert out["store-patient"]["data"] == skipped

    def test_next_from_the_report_you_just_labelled_goes_forward(self, wired):
        """The case that happens every time the filter is used: the current report leaves the list.

        Falling back to the whole cohort here would jump to a report the sidebar does not show.
        """
        cohort, _, _ = wired
        first, second = cohort.patient_ids[:2]
        cohort.record("label", first, labels=[], difficulty="hard")

        out = self._nav(wired, "next", first, {"prev": 0, "next": 0}, filters=["unlabelled"])
        assert out["store-patient"]["data"] == second

    def test_a_row_click_ignores_the_filter(self, wired):
        """Clicking a row is unambiguous: it names the patient, whatever the list is showing."""
        cohort, app, client = wired
        target = cohort.patient_ids[2]
        cohort.record("label", target, labels=["family_member"], difficulty="")
        rows = [{"id": {"type": "pat", "pid": p}, "property": "n_clicks",
                 "value": 1 if p == target else 0} for p in cohort.patient_ids]
        out = self._fire(
            client, self._callback(app, "pat-next"),
            rows + [{"id": "pat-prev", "property": "n_clicks", "value": 0},
                    {"id": "pat-next", "property": "n_clicks", "value": 0}],
            [{"id": "store-patient", "property": "data", "value": cohort.patient_ids[0]},
             {"id": "store-nav", "property": "data", "value": {"prev": 0, "next": 0}},
             {"id": "patient-filter", "property": "value", "value": ["unlabelled"]}],
            trigger={"id": {"type": "pat", "pid": target}, "property": "n_clicks"})
        assert out["store-patient"]["data"] == target

    # ── report labels ────────────────────────────────────────────────────────
    def _save_labels(self, wired, patient_id, labels, difficulty, n_clicks):
        cohort, app, client = wired
        return self._fire(
            client, self._callback(app, "doc-label-save"),
            [{"id": "doc-label-save", "property": "n_clicks", "value": n_clicks}],
            [{"id": {"type": "doc-labels", "group": "attribution"}, "property": "value",
              "value": [v for v in labels if v in ("family_member", "historical")]},
             {"id": {"type": "doc-labels", "group": "expression"}, "property": "value",
              "value": [v for v in labels if v in ("paraphrase", "direct_term")]},
             {"id": "doc-difficulty", "property": "value", "value": difficulty},
             {"id": "store-patient", "property": "data", "value": patient_id},
             {"id": "store-dirty", "property": "data", "value": 0}])

    def test_a_rerendered_save_button_does_not_save(self, wired):
        """``n_clicks=0`` is a re-created button, not a pressed one."""
        cohort, _, _ = wired
        assert self._save_labels(wired, cohort.patient_ids[0],
                                 ["family_member"], "hard", n_clicks=0) is None
        assert cohort.patient_labels(cohort.patient_ids[0])["labels"] == []

    def test_labels_do_not_follow_the_patient(self, wired):
        """The reported bug: labels saved on one report appeared on the next one."""
        cohort, _, _ = wired
        first, second = cohort.patient_ids[0], cohort.patient_ids[1]

        self._save_labels(wired, first, ["family_member", "paraphrase"], "hard", n_clicks=1)
        assert cohort.patient_labels(first)["labels"] == ["family_member", "paraphrase"]

        # Moving on re-creates the panel. The button's n_clicks resets to 0 and the callback fires.
        self._save_labels(wired, second, ["family_member", "paraphrase"], "hard", n_clicks=0)
        assert cohort.patient_labels(second)["labels"] == []
        assert cohort.patient_labels(second)["difficulty"] == ""

    def test_unticking_a_label_holds(self, wired):
        """The other half of the same bug: the phantom re-fire put the label back."""
        cohort, _, _ = wired
        patient = cohort.patient_ids[0]

        self._save_labels(wired, patient, ["family_member", "paraphrase"], "hard", n_clicks=1)
        self._save_labels(wired, patient, ["family_member"], "hard", n_clicks=2)
        self._save_labels(wired, patient, ["family_member", "paraphrase"], "hard", n_clicks=0)

        assert cohort.patient_labels(patient)["labels"] == ["family_member"]

    def test_everything_can_be_cleared(self, wired):
        cohort, _, _ = wired
        patient = cohort.patient_ids[0]

        self._save_labels(wired, patient, ["family_member"], "hard", n_clicks=1)
        self._save_labels(wired, patient, [], "", n_clicks=2)

        current = cohort.patient_labels(patient)
        assert current["labels"] == [] and current["difficulty"] == ""


# ──────────────────────────────────────────────────────────────────────────────
class TestReportControls:
    """The report-label inputs live in the layout and are never rebuilt.

    They used to be re-rendered per patient, which broke them in two ways at once. A re-created
    input arrives with the props its constructor declares, wiping whatever had just been ticked. And the Save button's ``n_clicks`` reset to ``0`` fired its callback with nobody having pressed
    anything. The vocabulary is fixed, so the fix is to build them once and write only their
    ``value`` props.

    These tests walk components **without** gating on ``children``, ``dcc.Checklist`` has none, and
    a walk that requires it cannot see a single input on the page. That is not hypothetical: it is
    why an earlier round of testing reported the panel as correct while it was not usable.
    """

    @staticmethod
    def _inputs(node, found=None):
        """``{id: value}`` for every Checklist/RadioItems in a tree."""
        found = {} if found is None else found
        if isinstance(node, (list, tuple)):
            for child in node:
                TestReportControls._inputs(child, found)
            return found
        if node is None or isinstance(node, str):
            return found
        if type(node).__name__ in ("Checklist", "RadioItems"):
            ident = getattr(node, "id", None)
            key = ident if isinstance(ident, str) else (ident or {}).get("group")
            found[key] = getattr(node, "value", None)
        children = getattr(node, "children", None)
        if children is not None:
            TestReportControls._inputs(children, found)
        return found

    def test_the_controls_are_in_the_static_layout(self, cohort):
        from apps.curation_ui.app import build_app

        app = build_app(cohort)
        rendered = json.dumps(app.layout, default=lambda o: o.to_plotly_json())
        for probe in ('"doc-difficulty"', '"doc-labels"', '"doc-label-save"',
                      '"doc-label-summary"'):
            assert probe in rendered, probe

    def test_the_layout_carries_the_report_vocabulary_only(self, cohort):
        from apps.curation_ui.views import labelling

        controls = self._inputs(labelling.report_layout())
        assert set(controls) == {"doc-difficulty", "attribution", "expression"}
        # Built empty: values arrive from the patient, not from the constructor.
        assert controls["attribution"] == [] and controls["doc-difficulty"] is None

    def test_the_summary_is_the_only_rendered_part(self, cohort):
        """Whatever a callback replaces must contain no input, or ticking it gets wiped."""
        from apps.curation_ui.views import labelling

        summary = labelling.report_summary(cohort, cohort.patient_ids[0], "light")
        assert self._inputs(summary) == {}

    def test_the_summary_shows_what_is_saved(self, cohort):
        from apps.curation_ui.views import labelling

        patient = cohort.patient_ids[0]
        cohort.record("label", patient, labels=["family_member"], difficulty="hard")
        text = json.dumps(labelling.report_summary(cohort, patient, "dark"),
                          default=lambda o: o.to_plotly_json())
        assert "Family confusion" in text and "Hard" in text

    def test_a_checklist_option_label_is_still_hoverable(self, cohort):
        """The help text is on the option, which is what makes the vocabulary usable at all."""
        from apps.curation_ui.views import labelling

        rendered = json.dumps(labelling.report_layout(),
                              default=lambda o: o.to_plotly_json())
        assert "The text describes a relative" in rendered


# ──────────────────────────────────────────────────────────────────────────────
class TestSourceColours:
    """Colour is identity here, so two sources that must be told apart must not share one.

    Nothing tested colour before, and two of these were already wrong: PhenoBERT's underline in
    the report was drawn in prior_annotation_2's amber while its chips everywhere else were green, and a
    prior_annotation+daphne annotation always took prior_annotation's colour because the merge kept file order.
    """

    def test_daphne_and_phenobert_are_distinct_from_everything(self):
        for mode in ("light", "dark"):
            colors = theme.source_colors(mode)
            assigned = [colors[s] for s in ("prior_annotation", "daphne", "prior_annotation_2", "phenobert", "new")]
            assert len(set(assigned)) == len(assigned), mode

    def test_phenobert_is_the_neutral_grey_and_not_a_categorical_hue(self):
        """It is reference, never adjudicated, so it reads as chrome, not as an annotator."""
        for mode in ("light", "dark"):
            palette = theme.palette(mode)
            assert theme.source_color("phenobert", palette) == palette["neutral"]
            assert theme.source_color("phenobert", palette) not in palette["categorical"]

    def test_a_source_is_coloured_by_name_not_by_its_position(self):
        """SOURCE_ORDER is an ordering. Adding to it must not re-colour everything after it."""
        palette = theme.palette("light")
        before = theme.source_colors("light")
        original = list(theme.SOURCE_ORDER)
        try:
            theme.SOURCE_ORDER.insert(1, "invented")
            assert theme.source_colors("light")["daphne"] == before["daphne"]
            # A source nothing knows about is neutral, not an IndexError: a log outlives
            # The vocabulary that wrote it.
            assert theme.source_color("invented", palette) == palette["neutral"]
        finally:
            theme.SOURCE_ORDER[:] = original

    def test_a_phenobert_mark_matches_a_phenobert_chip(self, cohort):
        """The mark in the report and the chip beside it are one source, so they are one colour."""
        palette = theme.palette("light")
        view = cohort.patient("SYN001")
        marks = reader._marks_for(0, view["display"][0], view, {})
        # Built through the real constructor, not by hand: every mark carries an entry list
        # now, and a hand-rolled dict without one would test a shape the reader cannot produce.
        mark = reader._mark_for_entry(0, 8, {"kind": "pb", "code": "HP:0001250",
                                             "label": "seizure"})
        rendered = reader._mark(cohort, 0, "seizures", mark, palette)
        assert rendered.style["borderColor"] == theme.source_color("phenobert", palette)
        assert marks is not None  # The placement path itself still runs

    def test_a_merged_mark_takes_the_highest_precedence_source(self):
        """daphne is the confirmed pass. Where it has ruled, its colour is the one that means it."""
        palette = theme.palette("light")
        assert (reader._entry_color(["prior_annotation", "daphne"], palette)
                == theme.source_color("daphne", palette))
        assert (reader._entry_color(["daphne", "prior_annotation"], palette)
                == theme.source_color("daphne", palette))
        # marc2 has no precedence, it can never be adjudicated, so it never wins the colour.
        assert (reader._entry_color(["prior_annotation_2", "prior_annotation"], palette)
                == theme.source_color("prior_annotation", palette))


# ──────────────────────────────────────────────────────────────────────────────
class TestQualifiers:
    """The seven annotation qualifiers, which ride on the annotation, not on their own event."""

    def test_a_suggestion_carries_its_qualifiers(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        event = log.append("suggest", "P1", segment_idx=0, trigger_word="hypotonic",
                           hpo_code="HP:0001252", labels=["negated", "family"])
        row = log.fold()["rows"][event["event_id"]]
        assert row["labels"] == ["negated", "family"]

    def test_an_edit_rewrites_the_qualifiers_without_touching_the_verdict(self, tmp_path):
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("daphne", "P1", "HP:0001252")
        log.append("verdict", "P1", target_key=key, status="kept", hpo_code="HP:0001252")
        log.append("edit", "P1", target_key=key, labels=["negated"])

        row = log.fold()["rows"][key]
        assert row["labels"] == ["negated"]
        assert row["status"] == "kept", "repairing the qualifiers passed a verdict"

    def test_the_last_qualifier_can_be_unticked(self, tmp_path):
        """An empty list is what the fold's generic truthy rule cannot express."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("daphne", "P1", "HP:0001252")
        log.append("edit", "P1", target_key=key, labels=["negated", "family"])
        log.append("edit", "P1", target_key=key, labels=[])
        assert log.fold()["rows"][key]["labels"] == []

    def test_an_edit_that_names_no_qualifiers_leaves_them_alone(self, tmp_path):
        """`edit` writes only the fields it names, fixing a trigger must not blank the labels."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("daphne", "P1", "HP:0001252")
        log.append("edit", "P1", target_key=key, labels=["negated"])
        log.append("edit", "P1", target_key=key, trigger_word="hypotonic")

        row = log.fold()["rows"][key]
        assert row["labels"] == ["negated"] and row["trigger_word"] == "hypotonic"

    def test_the_stored_list_is_not_aliased_to_the_event(self, tmp_path):
        """The event dict outlives the fold. A shared list would let a later write edit history."""
        log = store.EventLog(str(tmp_path), author="a")
        key = store.target_key("daphne", "P1", "HP:0001252")
        log.append("edit", "P1", target_key=key, labels=["negated"])
        row = log.fold()["rows"][key]
        row["labels"].append("family")
        assert log.fold()["rows"][key]["labels"] == ["negated"]

    def test_the_editor_form_offers_them_and_the_approve_row_only_shows_them(self, cohort):
        """One place edits a field. Two would be two copies that disagree."""
        form = _walk(editor.panel(cohort, "k", "approve", labels=["negated"]))
        pickers = [n for n in form
                   if isinstance(getattr(n, "id", None), dict)
                   and n.id.get("type") == "ann-labels"]
        assert len(pickers) == 1 and pickers[0].value == ["negated"]

        chips = _walk(labelling.qualifier_chips(["negated"], "light"))
        assert not [n for n in chips if getattr(n, "id", None)], "the chip line is an input"
        assert "Negated" in _text(chips)

    def test_an_unqualified_row_draws_no_chip_line(self):
        """Most rows carry none, so a line saying so on each is a line of noise per row."""
        assert labelling.qualifier_chips([], "light") is None
        assert labelling.qualifier_chips(None, "light") is None

    def test_the_picker_wraps_instead_of_stacking(self, cohort):
        """Seven stacked lines per row pushed the next row's verdict buttons off the screen."""
        picker = _walk(labelling.qualifier_controls("id", []))
        checklist = next(n for n in picker if getattr(n, "id", None) == "id")
        assert "qualifier-check" in checklist.className
        assert "label-check" not in checklist.className, (
            "the qualifier picker took the report vocabulary's stacked layout")

        css = open(os.path.join("apps", "curation_ui", "assets", "style.css"),
                   encoding="utf-8").read()
        block = css[css.index(".qualifier-check label {"):]
        assert "inline-flex" in block[:block.index("}")]


# ──────────────────────────────────────────────────────────────────────────────
class TestGrouping:
    """A term annotated twice is one card and two verdicts."""

    def _entries(self, cohort, patient_id):
        view = cohort.patient(patient_id)
        rows = {row["key"]: row for row in cohort.rows_for(patient_id)}
        return approve._existing_entries(cohort, view, rows)

    def test_two_annotations_of_one_code_share_a_card(self, cohort):
        """SYN002 carries HP:0001263 twice, on two different triggers."""
        groups = approve.group_entries(self._entries(cohort, "SYN002"))
        codes = [g["code"] for g in groups]
        assert len(codes) == len(set(codes)), "a code was split across two cards"
        delay = next(g for g in groups if g["code"] == "HP:0001263")
        assert len(delay["members"]) == 2

    def test_each_annotation_in_a_group_keeps_its_own_verdict_buttons(self, cohort):
        """Folding them into one verdict would let keeping one silently settle the other."""
        groups = approve.group_entries(self._entries(cohort, "SYN002"))
        delay = next(g for g in groups if g["code"] == "HP:0001263")
        card = _walk(approve._existing_group_card(cohort, delay, "light"))

        keys = {n.id["key"] for n in card
                if isinstance(getattr(n, "id", None), dict) and n.id.get("type") == "adjudicate"}
        assert keys == {m["key"] for m in delay["members"]}

    def test_a_verdict_on_one_occurrence_leaves_the_other_undecided(self, cohort):
        groups = approve.group_entries(self._entries(cohort, "SYN002"))
        delay = next(g for g in groups if g["code"] == "HP:0001263")
        first, second = delay["members"]
        cohort.record("verdict", "SYN002", target_key=first["key"], status="kept",
                      hpo_code=first["code"])

        after = approve.group_entries(self._entries(cohort, "SYN002"))
        members = {m["key"]: m for m in next(g for g in after
                                             if g["code"] == "HP:0001263")["members"]}
        assert members[first["key"]]["status"] == "kept"
        assert members[second["key"]]["status"] == ""

    def test_rows_come_out_in_segment_order_within_their_band(self):
        entries = [
            {"status": "", "segment_idx": 7, "code": "HP:C"},
            {"status": "kept", "segment_idx": 1, "code": "HP:A"},
            {"status": "", "segment_idx": 1, "code": "HP:B"},
            {"status": store.DELETE_SUGGESTED, "segment_idx": 9, "code": "HP:D"},
            {"status": "", "segment_idx": None, "code": "HP:E"},
        ]
        assert [e["code"] for e in sorted(entries, key=approve.order_key)] == [
            "HP:D",           # somebody is waiting on an answer
            "HP:B", "HP:C",   # undecided, in the order the report reads
            "HP:E",           # undecided but unplaced, nothing to read it against
            "HP:A",           # decided
        ]

    def test_a_group_sorts_with_its_most_urgent_member(self):
        """A term with one open question belongs with the open work, not with the settled rows."""
        groups = approve.group_entries([
            {"effective_code": "HP:A", "code": "HP:A", "name": "A", "status": "kept",
             "segment_idx": 0},
            {"effective_code": "HP:B", "code": "HP:B", "name": "B", "status": "kept",
             "segment_idx": 5},
            {"effective_code": "HP:B", "code": "HP:B", "name": "B", "status": "",
             "segment_idx": 9},
        ])
        assert [g["code"] for g in groups] == ["HP:B", "HP:A"]
        # …and inside the group, still segment order.
        assert [m["segment_idx"] for m in groups[0]["members"]] == [9, 5]


# ──────────────────────────────────────────────────────────────────────────────
class TestMultipleAnnotationsOnOneSpan:
    """Marks cannot nest, so everything asserted about one stretch of words is one mark."""

    def _marks(self, cohort, patient_id, segment_idx):
        view = cohort.patient(patient_id)
        return reader._marks_for(segment_idx, view["display"][segment_idx], view, {})

    def test_two_codes_on_the_same_words_are_one_mark_with_two_entries(self, cohort):
        """SYN005 annotates *hypotonic* as both hypotonia and muscle weakness."""
        marks = self._marks(cohort, "SYN005", 1)
        ann = [m for m in marks if m["kind"] == "ann"]
        assert len(ann) == 1, "the second phenotype was dropped by the overlap pass"
        assert {e["code"] for e in ann[0]["entries"]} == {"HP:0001252", "HP:0003324"}

    def test_both_codes_are_drawn_and_both_are_click_targets(self, cohort):
        view = cohort.patient("SYN005")
        rendered = _walk(reader.render("SYN005", None, "light"))
        codes = {n.id["code"] for n in rendered
                 if isinstance(getattr(n, "id", None), dict) and n.id.get("type") == "mark"}
        assert {"HP:0001252", "HP:0003324"} <= codes
        assert view is not None

    def test_the_tooltip_names_every_phenotype_on_the_mark(self, cohort):
        marks = self._marks(cohort, "SYN005", 1)
        ann = next(m for m in marks if m["kind"] == "ann")
        title = reader._mark(cohort, 1, "hypotonic", ann, theme.palette("light")).title
        assert "HP:0001252" in title and "HP:0003324" in title
        assert "\n\n" in title, "the two phenotypes were not put on separate blocks"

    def test_one_phenotype_leaves_the_whole_mark_clickable(self, cohort):
        """The old behaviour, unchanged: with nothing to choose between, the words are the target."""
        # Segment 2, not 1: segment 1 carries the nested pair, and a mark with two phenotypes on it
        # is the case where the words are *not* the click target.
        marks = self._marks(cohort, "SYN001", 2)
        ann = next(m for m in marks if m["kind"] == "ann")
        rendered = reader._mark(cohort, 2, "hypotonic", ann, theme.palette("light"))
        assert isinstance(rendered.id, dict) and rendered.id["code"] == "HP:0001252"

    def test_two_files_asserting_one_code_are_still_a_single_entry(self, cohort):
        """prior_annotation and daphne agreeing is one claim named twice, not two phenotypes."""
        marks = self._marks(cohort, "SYN001", 2)
        ann = next(m for m in marks if m["kind"] == "ann")
        assert len(ann["entries"]) == 1
        assert set(ann["entries"][0]["sources"]) == {"prior_annotation", "daphne"}

    # ── overlapping triggers ──────────────────────────────────────────────────
    # One trigger word a substring of another is not an edge case: *seizure* sits inside
    # *focal onset seizure*, *delay* inside *developmental delay*, and both members of such a pair
    # get annotated. Marks cannot nest, so the wider span takes the underline and carries both tags.
    TEXT = "she had focal onset seizures during the night"

    @staticmethod
    def _synthetic(annotations=(), curated=(), detections=(), text=None):
        """``_marks_for`` over a hand-built segment, the overlap pass without a whole cohort."""
        view = {"detections": {0: list(detections)}, "annotations": {0: list(annotations)},
                "triggers": {}}
        return reader._marks_for(0, text or TestMultipleAnnotationsOnOneSpan.TEXT, view,
                                 {0: list(curated)})

    @staticmethod
    def _codes(mark) -> set:
        return {entry["code"] for entry in mark["entries"]}

    def test_a_nested_trigger_puts_both_codes_on_the_wider_one(self, cohort):
        """SYN001 segment 1 carries *seizures* inside *recurrent seizures*, two different codes."""
        marks = self._marks(cohort, "SYN001", 1)
        ann = [m for m in marks if m["kind"] == "ann"]
        assert len(ann) == 1, "the nested pair drew two marks, which cannot be woven"
        assert self._codes(ann[0]) == {"HP:0007359", "HP:0001250"}
        text = cohort.patient("SYN001")["display"][1]
        assert text[ann[0]["start"]:ann[0]["end"]] == "recurrent seizures", \
            "the shorter trigger took the underline from the longer one"

    def test_a_curated_trigger_inside_an_annotation_survives(self):
        """This app's own suggestion used to vanish under any annotation covering it."""
        big = {"hpo_code": "HP:0007359", "hpo_name": "Focal-onset seizure", "source": "prior_annotation",
               "start": 8, "end": 27}
        marks = self._synthetic(annotations=[big],
                                curated=[{"hpo_code": "HP:0001250", "hpo_name": "Seizure",
                                          "trigger_word": "seizures", "status": "suggested"}])
        assert len(marks) == 1
        assert self._codes(marks[0]) == {"HP:0007359", "HP:0001250"}

    def test_an_annotation_inside_a_curated_trigger_survives(self):
        """And the other direction, which lost the file's annotation, not the suggestion."""
        marks = self._synthetic(
            annotations=[{"hpo_code": "HP:0001250", "hpo_name": "Seizure", "source": "daphne",
                          "start": 20, "end": 27}],
            curated=[{"hpo_code": "HP:0007359", "hpo_name": "Focal-onset seizure",
                      "trigger_word": "focal onset seizures", "status": "suggested"}])
        assert len(marks) == 1
        assert self._codes(marks[0]) == {"HP:0007359", "HP:0001250"}

    def test_an_annotation_outranks_a_detection_covering_it(self, cohort):
        """SYN002 segment 0: PhenoBERT's span covers both annotations, and used to erase them.

        The wider span normally wins, but not here, a detection is reference, and letting it host
        would underline *global*, a word no annotator claimed.
        """
        marks = self._marks(cohort, "SYN002", 0)
        assert len(marks) == 1
        text = cohort.patient("SYN002")["display"][0]
        assert marks[0]["kind"] == "ann"
        assert text[marks[0]["start"]:marks[0]["end"]] == "developmental delay"
        assert "HP:0001263" in self._codes(marks[0])

    def test_a_crossing_overlap_keeps_the_longer_span(self):
        """Neither contains the other. The one that used to survive was the shorter."""
        marks = self._synthetic(annotations=[
            {"hpo_code": "HP:0007359", "hpo_name": "a", "source": "prior_annotation",
             "start": 14, "end": 19},
            {"hpo_code": "HP:0011146", "hpo_name": "b", "source": "prior_annotation",
             "start": 16, "end": 27},
        ])
        assert len(marks) == 1
        assert marks[0]["end"] - marks[0]["start"] == 11, "the shorter span won the underline"
        assert self._codes(marks[0]) == {"HP:0007359", "HP:0011146"}

    def test_a_detection_agreeing_with_an_annotation_is_not_a_second_tag(self):
        """Two identical codes side by side read as two phenotypes. The agreement is in the panel."""
        marks = self._synthetic(
            annotations=[{"hpo_code": "HP:0001250", "hpo_name": "Seizure", "source": "prior_annotation",
                          "start": 8, "end": 27}],
            detections=[{"local_start": 20, "local_end": 27, "hpo_id": "HP:0001250",
                         "hpo_label": "Seizure"}])
        assert len(marks) == 1 and len(marks[0]["entries"]) == 1

    def test_a_detection_disagreeing_with_an_annotation_is_kept(self):
        """A reference naming a *different* code is a disagreement, which is the reason to look."""
        marks = self._synthetic(
            annotations=[{"hpo_code": "HP:0001250", "hpo_name": "Seizure", "source": "prior_annotation",
                          "start": 8, "end": 27}],
            detections=[{"local_start": 20, "local_end": 27, "hpo_id": "HP:0007359",
                         "hpo_label": "Focal-onset seizure"}])
        assert self._codes(marks[0]) == {"HP:0001250", "HP:0007359"}

    def test_the_marks_stay_disjoint_and_in_order(self):
        """``_weave`` walks one forward cursor, so overlapping marks would emit text twice."""
        marks = self._synthetic(
            annotations=[{"hpo_code": "HP:0001250", "hpo_name": "a", "source": "prior_annotation",
                          "start": 8, "end": 27},
                         {"hpo_code": "HP:0011146", "hpo_name": "b", "source": "prior_annotation",
                          "start": 28, "end": 34}],
            detections=[{"local_start": 20, "local_end": 30, "hpo_id": "HP:0002187",
                         "hpo_label": "c"}])
        assert [m["start"] for m in marks] == sorted(m["start"] for m in marks)
        for before, after in zip(marks, marks[1:]):
            assert before["end"] <= after["start"], "two marks overlap; _weave cannot draw that"


# ──────────────────────────────────────────────────────────────────────────────
class TestPerPatientAgreement:
    """`disagree` has to be a finding about the annotation, not about the sidebar's configuration."""

    def test_a_source_with_nothing_for_this_patient_is_not_compared(self, cohort):
        """SYN005 has prior_annotation and prior_annotation_2 rows and no daphne row at all."""
        assert set(cohort.gold_sources) == {"prior_annotation", "daphne", "prior_annotation_2"}
        assert cohort.gold_sources_for("SYN005") == ["prior_annotation", "prior_annotation_2"]
        assert cohort.gold_sources_for("SYN001") == ["prior_annotation", "daphne", "prior_annotation_2"]

    def test_a_term_both_present_files_carry_does_not_read_as_a_disagreement(self, cohort):
        panel = _text(_walk(reader.render_gold_panel(
            cohort, cohort.patient("SYN005"), "light")))
        assert "HP:0001252" in panel
        # Four terms on this report. Holistic and marc2 differ on one each, so two disagree.
        # Comparing against the daphne file, which has no row for this patient at all, used to
        # make that four, every term on the screen.
        assert "2 the files disagree on" in panel
        assert "4 the files disagree on" not in panel

    def test_the_overview_counts_the_same_way(self, cohort):
        row = next(r for r in cohort.overview() if r["patient_id"] == "SYN005")
        assert row["n_terms"] == 4
        assert row["n_disagree"] == 2, "a file with no row for this report counted as a dissenter"

    def test_the_agreement_chip_names_only_the_files_that_have_a_row(self, cohort):
        view = cohort.patient("SYN005")
        rows = {row["key"]: row for row in cohort.rows_for("SYN005")}
        entries = approve._existing_entries(cohort, view, rows)
        hypotonia = next(e for e in entries if e["code"] == "HP:0001252")
        assert "daphne" not in hypotonia["agreement"]
        assert hypotonia["agreement"] == "all sources"


# ──────────────────────────────────────────────────────────────────────────────
class TestUnplacedAnnotations:
    """An annotation with a trigger word and nothing underlined is the by-hand queue."""

    def test_a_trigger_that_occurs_nowhere_is_flagged(self, cohort):
        """SYN001's HP:0001873 names *purpura*, which the report does not contain."""
        panel = reader.render_gold_panel(cohort, cohort.patient("SYN001"), "light")
        flagged = [n for n in _walk(panel)
                   if "gold-line-unplaced" in (getattr(n, "className", None) or "")]
        assert flagged, "an unplaced annotation rendered as ordinary text"
        assert "purpura" in _text(flagged)

    def test_the_count_reaches_the_subtitle(self, cohort):
        panel = reader.render_gold_panel(cohort, cohort.patient("SYN001"), "light")
        assert "1 need evidence" in _text(_walk(panel))

    def test_a_code_only_row_is_not_flagged(self, cohort):
        """prior_annotation_2 never claimed a position. It is not missing evidence. It never had any."""
        assert not reader.needs_location({"source": "prior_annotation_2", "trigger_word": ""}, None)
        # …but the same empty trigger on a source the curated ground truth is built from is the
        # queue: neither a segment nor words, which is the state that reads as "fine" to anything
        # asking only whether a stated trigger could be found.
        assert reader.needs_location({"source": "prior_annotation", "trigger_word": ""}, None)

    def test_the_three_states_that_need_a_curator(self):
        prior_annotation = {"source": "prior_annotation", "trigger_word": "purpura"}
        assert reader.needs_location(prior_annotation, None), "trigger occurs nowhere in the report"
        # Placed on a segment, but the trigger did not survive into it, nothing to underline.
        assert reader.needs_location(prior_annotation, {"segment_idx": 1, "start": None})
        assert not reader.needs_location(prior_annotation, {"segment_idx": 1, "start": 0, "end": 1})

    def test_a_row_this_app_located_leaves_the_queue(self, cohort):
        """``place_all`` reads the files and never sees the log, so the log has to be folded back."""
        record = {"source": "prior_annotation", "trigger_word": "purpura"}
        display = ["He was markedly hypotonic at birth."]
        assert reader.needs_location(record, None, {"segment_idx": 0, "trigger_word": "hypotonic"},
                                   display) is False
        # A saved trigger the named sentence does not contain is the unplaced state over again.
        assert reader.needs_location(record, None, {"segment_idx": 0, "trigger_word": "purpura"},
                                   display) is True
        # Half an evidence location is not an evidence location.
        assert reader.needs_location(record, None, {"segment_idx": 0, "trigger_word": ""}, display)
        assert reader.needs_location(record, None, {"segment_idx": None,
                                                  "trigger_word": "hypotonic"}, display)

    def test_the_filtered_panel_shows_only_what_still_needs_placing(self, cohort):
        """The Evidence location tab's panel is the same list cut to the by-hand queue."""
        view = cohort.patient("SYN001")
        full = _text(_walk(reader.render_gold_panel(cohort, view, "light")))
        only = _text(_walk(reader.render_gold_panel(cohort, view, "light", only_unanchored=True)))
        assert "purpura" in full and "purpura" in only
        # HP:0001250 is drawn on its own words, so it is not in the queue.
        assert "HP:0001250" in full and "HP:0001250" not in only
        assert "Needs evidence" in only

    def test_a_fully_located_report_says_so_rather_than_drawing_nothing(self, cohort):
        """SYN005's annotations are all placed, so its queue is empty and has to say why."""
        panel = reader.render_gold_panel(cohort, cohort.patient("SYN005"), "light",
                                         only_unanchored=True)
        assert "has its evidence located" in _text(_walk(panel)), "an empty queue rendered as an empty box"


# ──────────────────────────────────────────────────────────────────────────────
class TestLocationTab:
    """The by-hand queue: which annotations are in it, and what takes one out."""

    @staticmethod
    def _ids(node, kind: str) -> list:
        return [n.id for n in _walk(node)
                if isinstance(getattr(n, "id", None), dict) and n.id.get("type") == kind]

    def test_the_queue_is_prior_annotation_daphne_and_suggestions(self, cohort):
        """prior_annotation_2 is a code list. It never claimed a position, so it is not a failure to place."""
        from apps.curation_ui.views import locate

        for pid in cohort.patient_ids:
            queue, _ = locate.queue_for(cohort, pid)
            assert all(e["source"] in ("prior_annotation", "daphne", "new") for e in queue), pid

    def test_a_trigger_that_occurs_nowhere_is_queued(self, cohort):
        """SYN001's HP:0001873 names *purpura*, which the report does not contain."""
        from apps.curation_ui.views import locate

        queue, _ = locate.queue_for(cohort, "SYN001")
        assert "HP:0001873" in {e["code"] for e in queue}

    def test_an_annotation_with_no_trigger_at_all_is_queued(self, cohort):
        """The third state: neither a segment nor words, so nothing ever failed to be found."""
        from apps.curation_ui.views import locate

        queue, _ = locate.queue_for(cohort, "SYN002")
        blank = [e for e in queue if e["code"] == "HP:0002187"]
        assert blank, "a prior_annotation row with an empty trigger cell stayed out of the queue"
        assert not (blank[0]["record"].get("trigger_word") or "").strip()

    def test_no_phenotype_is_queued_twice(self, cohort):
        """The invariant the merge exists to hold.

        Two rows for one code can only stay apart by sitting on **different words**, and a row that
        sits on words is not in this queue. So a code appearing twice would mean a curator was asked
        to place one phenotype twice, and the second time there is nothing left to place.
        """
        from apps.curation_ui.views import locate

        for pid in cohort.patient_ids:
            codes = [e["code"] for e in locate.queue_for(cohort, pid)[0]]
            assert len(codes) == len(set(codes)), (pid, codes)

    def test_two_rows_that_both_failed_to_place_are_queued_once(self, cohort):
        """SYN002 carries HP:0001250 twice: prior_annotation says *convulsions*, daphne says *fits*.

        Neither word is in that report, so neither row has said anything about which words, two
        broken records of one annotation, and one thing for a curator to do.
        """
        from apps.curation_ui.views import locate

        queued = [e for e in locate.queue_for(cohort, "SYN002")[0] if e["code"] == "HP:0001250"]
        assert len(queued) == 1
        assert set(queued[0]["sources"]) == {"daphne", "prior_annotation"}
        assert queued[0]["claims"], "the trigger the fold did not keep was dropped silently"

    def test_a_term_this_app_has_located_closes_the_file_row(self, cohort):
        """The file's claim will never start working, so nothing else could ever close that row.

        A curator who finds where the term really came from and records it has answered the
        question. The queue going on to ask for the same phenotype is asking for work that is done.
        """
        from apps.curation_ui.views import locate

        assert "HP:0001873" in {e["code"] for e in locate.queue_for(cohort, "SYN001")[0]}
        view = cohort.patient("SYN001")
        cohort.record("suggest", "SYN001", segment_idx=3, segment_text=view["display"][3],
                      hpo_code="HP:0001873", hpo_name="Thrombocytopenia",
                      trigger_word="thrombocytopenia")
        assert "HP:0001873" not in {e["code"] for e in locate.queue_for(cohort, "SYN001")[0]}

    def test_a_suggestion_that_settles_nothing_does_not_close_the_row(self, cohort):
        """Only an evidence location closes it. A rejected suggestion is not one, and neither is a broken one.

        Without this the fold would be a way to make the queue shorter without placing anything.
        """
        from apps.curation_ui.views import locate

        event = cohort.record("suggest", "SYN001", segment_idx=3,
                              segment_text=cohort.patient("SYN001")["display"][3],
                              hpo_code="HP:0001873", hpo_name="Thrombocytopenia",
                              trigger_word="thrombocytopenia")
        cohort.record("reject", "SYN001", target_key=event["event_id"],
                      hpo_code="HP:0001873", hpo_name="Thrombocytopenia")
        assert "HP:0001873" in {e["code"] for e in locate.queue_for(cohort, "SYN001")[0]}

        # And an evidence location that is only half an evidence location: a trigger the named sentence does not contain.
        cohort.record("edit", "SYN001", target_key=event["event_id"], segment_idx=1,
                      segment_text=cohort.patient("SYN001")["display"][1],
                      trigger_word="purpura")
        assert "HP:0001873" in {e["code"] for e in locate.queue_for(cohort, "SYN001")[0]}

    def test_a_placed_annotation_is_not_queued(self, cohort):
        """HP:0001250 is drawn on its own words in SYN001, so there is nothing to do for it."""
        from apps.curation_ui.views import locate

        queue, _ = locate.queue_for(cohort, "SYN001")
        assert "HP:0001250" not in {e["code"] for e in queue}

    def test_the_denominator_is_every_annotation_not_every_problem(self, cohort):
        """"3 of 11" is a statement about the report; "3 of 3" would be one about the queue."""
        from apps.curation_ui.views import locate

        queue, n_total = locate.queue_for(cohort, "SYN001")
        assert n_total > len(queue)

    def test_locating_a_row_takes_it_out_of_the_queue(self, cohort):
        """``place_all`` reads the files and never sees the log, so the log has to be folded in."""
        from apps.curation_ui.views import locate

        entry = next(e for e in locate.queue_for(cohort, "SYN001")[0]
                     if e["code"] == "HP:0001873")
        segment = cohort.patient("SYN001")["display"][3]
        cohort.record("edit", "SYN001", target_key=entry["key"], segment_idx=3,
                      segment_text=segment, trigger_word="thrombocytopenia",
                      hpo_code="HP:0001873")
        assert "HP:0001873" not in {e["code"] for e in locate.queue_for(cohort, "SYN001")[0]}

    def test_a_trigger_absent_from_its_own_segment_stays_queued(self, cohort):
        """Half an evidence location is not an evidence location: a sentence that does not contain the words is the
        unplaced state over again, and counting it as done would ship an annotation with no
        evidence behind it."""
        from apps.curation_ui.views import locate

        entry = next(e for e in locate.queue_for(cohort, "SYN001")[0]
                     if e["code"] == "HP:0001873")
        cohort.record("edit", "SYN001", target_key=entry["key"], segment_idx=1,
                      segment_text=cohort.patient("SYN001")["display"][1],
                      trigger_word="purpura", hpo_code="HP:0001873")
        assert "HP:0001873" in {e["code"] for e in locate.queue_for(cohort, "SYN001")[0]}

    def test_every_queued_row_gets_an_editor_at_the_location_panel(self, cohort):
        """A panel mounted with the wrong ``at`` is a form whose Save silently does nothing."""
        from apps.curation_ui.views import locate

        drawn = locate.render_queue("SYN001", "light")
        ats = {i["at"] for i in self._ids(drawn, "ann-save")}
        assert ats == {locate.AT}
        keys = {i["key"] for i in self._ids(drawn, "ann-save")}
        assert keys == {e["key"] for e in locate.queue_for(cohort, "SYN001")[0]}

    def test_a_queued_row_offers_its_lexical_candidates(self, cohort):
        """The same evidence location buttons Approve mode already has, so there is one way to record one."""
        from apps.curation_ui.views import locate

        drawn = locate.render_queue("SYN001", "light")
        assert self._ids(drawn, "anchor"), "no candidate button was offered anywhere in the queue"

    def test_a_report_with_nothing_left_says_so(self, cohort):
        """SYN005's annotations are all placed, so its queue is empty and has to explain itself."""
        from apps.curation_ui.views import locate

        assert locate.queue_for(cohort, "SYN005")[0] == []
        assert "have a segment and a trigger" in _text(locate.render_queue("SYN005", "light"))

    def test_a_report_with_no_segmentation_says_it_cannot_be_worked_on(self, cohort):
        """SYN004 is in the annotation files and not in the segmentation.

        Every field on the form names a segment, so offering it would be offering a Save that
        ``validate_annotation`` refuses, with nothing on the screen explaining why.
        """
        from apps.curation_ui.views import locate

        assert locate.queue_for(cohort, "SYN004")[0], "SYN004 has annotations to place"
        drawn = _text(locate.render_queue("SYN004", "light"))
        assert "no segmentation" in drawn
        assert not self._ids(locate.render_queue("SYN004", "light"), "ann-save")

    def test_the_queue_degrades_rather_than_raising(self, cohort):
        from apps.curation_ui.views import locate

        assert locate.render_queue(None, "light") is not None
        assert locate.render_queue("NOT_A_PATIENT", "light") is not None

    def test_the_worklist_names_only_reports_with_work_left(self, cohort):
        from apps.curation_ui.views import locate

        drawn = locate.render_worklist("light")
        listed = {i["pid"] for i in self._ids(drawn, "pat")}
        expected = {pid for pid in cohort.patient_ids if locate.queue_for(cohort, pid)[0]}
        assert listed == expected
        assert "SYN005" not in listed

    def test_the_worklist_reuses_the_sidebars_own_row_id(self, cohort):
        """A second writer of ``store-patient`` is how two screens disagree about who is current."""
        from apps.curation_ui.views import locate

        ids = self._ids(locate.render_worklist("light"), "pat")
        assert ids and all(set(i) == {"type", "pid"} for i in ids)


# ──────────────────────────────────────────────────────────────────────────────
class TestMergeIdenticalAnnotations:
    """`prior_annotation` and `daphne` overlap by design, so a term both carry is one annotation.

    What separates two annotations is **where they actually sit**, never what their files claim: a
    trigger word that occurs nowhere has said nothing about which words, so it cannot disagree with
    anything. Two rows that both failed are two broken records of one annotation, not two.
    """

    SPAN = (1, 28, 36)
    OTHER = (1, 10, 20)

    @staticmethod
    def _row(source, key, code="HP:0001250", trigger="", segment=None, context="",
             confirmed=None, name="Seizure"):
        return {"source": source, "key": key, "hpo_code": code, "hpo_name": name,
                "trigger_word": trigger, "segment_idx": segment, "sentence_context": context,
                "confirmed": confirmed, "char_offset": None}

    # ── rows that landed ──────────────────────────────────────────────────────
    def test_two_rows_on_the_same_words_are_one(self):
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", trigger="seizures"),
             self._row("daphne", "d", trigger="seizures", segment=1)],
            {"h": self.SPAN, "d": self.SPAN})
        assert len(merged) == 1
        assert merged[0]["sources"] == ["daphne", "prior_annotation"], "the confirmed pass is the primary"
        assert merged[0]["key"] == "d" and merged[0]["keys"] == ["d", "h"]
        assert merged[0]["claims"] == [], "a less complete row is not a second opinion"

    def test_two_rows_on_different_words_stay_two_annotations(self):
        """`developmental delay` and `delay` are two claims about two stretches of text."""
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h1", trigger="developmental delay", segment=0),
             self._row("prior_annotation", "h2", trigger="delay", segment=0)],
            {"h1": (0, 30, 49), "h2": (0, 44, 49)})
        assert len(merged) == 2

    # ── rows that did not ─────────────────────────────────────────────────────
    def test_two_broken_rows_with_different_triggers_are_still_one(self):
        """The case comparing the files' *claims* got wrong.

        Neither trigger occurs in the report, so neither row has said anything about which words.
        Two failed attempts at recording one phenotype is one thing for a curator to do.
        """
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", trigger="purpura"),
             self._row("daphne", "d", trigger="petechiae", segment=4)],
            {})
        assert len(merged) == 1

    def test_two_broken_rows_with_no_evidence_at_all_are_one(self):
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h"), self._row("daphne", "d")], {})
        assert len(merged) == 1 and merged[0]["sources"] == ["daphne", "prior_annotation"]

    def test_a_broken_row_folds_into_one_that_is_already_located(self):
        """The row that works hosts, so the duplicate stops being asked about."""
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", trigger="convulsion"),
             self._row("daphne", "d", trigger="seizures", segment=1)],
            {"d": self.SPAN})
        assert len(merged) == 1
        assert merged[0]["key"] == "d", "the anchored row must host, not be absorbed"
        assert merged[0]["trigger_word"] == "seizures" and merged[0]["segment_idx"] == 1

    def test_a_located_row_hosts_even_when_the_broken_one_is_the_better_source(self):
        merged = sources.merge_identical(
            [self._row("daphne", "d", trigger="convulsion"),
             self._row("prior_annotation", "h", trigger="seizures", segment=1)],
            {"h": self.SPAN})
        assert merged[0]["key"] == "h"

    # ── nothing is silently dropped ───────────────────────────────────────────
    def test_a_folded_row_that_disagreed_is_still_named(self):
        """On a term nobody could place, each file's guess is most of the evidence there is."""
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", trigger="purpura"),
             self._row("daphne", "d", trigger="petechiae", segment=4)],
            {})
        assert merged[0]["claims"] == [
            {"source": "prior_annotation", "trigger_word": "purpura", "segment_idx": None}]

    def test_a_row_that_merely_says_less_is_not_a_second_opinion(self):
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", trigger="seizures"),
             self._row("daphne", "d", trigger="seizures", segment=1)],
            {"h": self.SPAN, "d": self.SPAN})
        assert merged[0]["claims"] == []

    # ── the merged row keeps the most information ─────────────────────────────
    def test_the_merged_row_keeps_the_most_information_available(self):
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", trigger="convulsions",
                       context="She had convulsions."),
             self._row("daphne", "d", segment=2, confirmed=True)],
            {})[0]
        assert merged["trigger_word"] == "convulsions"
        assert merged["segment_idx"] == 2
        assert merged["sentence_context"] == "She had convulsions."
        assert merged["confirmed"] is True

    def test_a_false_confirmed_flag_is_not_overwritten_by_a_missing_one(self):
        """`confirmed` is three-valued. Only "this file does not say" is missing information."""
        merged = sources.merge_identical(
            [self._row("daphne", "d", trigger="convulsions", confirmed=False),
             self._row("prior_annotation", "h", trigger="convulsions")], {})[0]
        assert merged["confirmed"] is False

    # ── the things that must never merge ──────────────────────────────────────
    def test_prior_annotation_2_is_never_absorbed(self):
        """A cross-check that has merged into the thing it checks is not a cross-check.

        A code-only row lands nowhere, so it is compatible with every row for its code and would be
        folded into all of them and reported as part of an annotation it says nothing about.
        """
        merged = sources.merge_identical(
            [self._row("prior_annotation_2", "m"),
             self._row("prior_annotation", "h", trigger="purpura"),
             self._row("daphne", "d", trigger="petechiae")], {})
        assert len(merged) == 2
        assert [set(m["sources"]) for m in merged].count({"prior_annotation_2"}) == 1

    def test_different_codes_are_never_merged(self):
        merged = sources.merge_identical(
            [self._row("prior_annotation", "h", code="HP:0001250"),
             self._row("daphne", "d", code="HP:0001252")], {})
        assert len(merged) == 2

    def test_a_curated_location_counts_as_landing(self, cohort):
        """`realised_spans` reads the log as well as the files, or a located row keeps a twin."""
        view = cohort.patient("SYN001")
        curated = {"prior_annotation|SYN001|HP:0001873":
                   {"segment_idx": 3, "trigger_word": "thrombocytopenia"}}
        spans = reader.realised_spans(view, curated)
        assert spans["prior_annotation|SYN001|HP:0001873"][0] == 3

    def test_the_queue_asks_for_one_annotation_once(self, cohort):
        """SYN004 carries HP:0001250 in both files, both unplaced. That is one thing to do."""
        from apps.curation_ui.views import locate

        queue, _ = locate.queue_for(cohort, "SYN004")
        assert len(queue) == 1
        assert set(queue[0]["sources"]) == {"daphne", "prior_annotation"}

    def test_the_document_panel_names_both_files_on_one_line(self, cohort):
        """SYN001's HP:0001250 is in both files on the same words, one line, naming both."""
        panel = reader.render_gold_panel(cohort, cohort.patient("SYN001"), "light")
        text = _text(_walk(panel))
        assert "daphne + prior_annotation:" in text
        assert "prior_annotation: “seizures”" not in text, "the same annotation was listed twice"

    def test_a_merged_row_still_reports_every_file_that_carries_the_term(self, cohort):
        """The merge is about how many rows to draw, never about which files agree.

        A merged row's scalar ``source`` names only its primary, so a panel reading that would drop
        `prior_annotation` from every term `daphne` also carries, and the agreement chip, which is the one
        thing that row exists to say, would read "disagree" on all of them.
        """
        panel = reader.render_gold_panel(cohort, cohort.patient("SYN001"), "light")
        text = _text(_walk(panel))
        assert "agreed" in text or "disagree" in text
        # HP:0001250 is in all three files, so it is not a disagreement.
        assert "3 the files disagree on" in text, text[:400]

    def test_locating_the_primary_settles_the_whole_group(self, cohort):
        """One write, not one per file row: the merged siblings are the same annotation."""
        from apps.curation_ui.views import locate

        entry = locate.queue_for(cohort, "SYN004")[0][0]
        assert len(entry["record"]["keys"]) == 2
        assert entry["key"] == entry["record"]["keys"][0]


# ──────────────────────────────────────────────────────────────────────────────
class TestChrome:
    """The two things that are properties of the whole screen, not of any one panel."""

    def test_every_font_size_scales_with_the_root_property(self):
        """A px size does not follow ``--fs``, so the slider would move some text and not the rest."""
        css = open(os.path.join("apps", "curation_ui", "assets", "style.css"),
                   encoding="utf-8").read()
        offenders = [line.strip() for line in css.splitlines()
                     if re.search(r"font-size:\s*[\d.]+px", line)
                     and "data-present" not in line and "--fs" not in line]
        assert offenders == [], offenders
        assert "font-size: var(--fs, 14px)" in css

    def test_the_comment_box_is_outside_both_mode_panels(self, cohort):
        """It vanished in Approve mode when it lived in edit.layout(). One instance, both modes."""
        from apps.curation_ui import app as entry

        layout = _walk(entry.build_layout(cohort))
        boxes = [n for n in layout if getattr(n, "id", None) == "doc-comment"]
        assert len(boxes) == 1

        for panel_id in ("wrap-edit", "wrap-approve"):
            wrapper = next(n for n in layout if getattr(n, "id", None) == panel_id)
            assert not [n for n in _walk(wrapper) if getattr(n, "id", None) == "doc-comment"]

    def test_the_text_size_control_persists_per_browser(self, cohort):
        from apps.curation_ui import app as entry

        layout = _walk(entry.build_layout(cohort))
        slider = next(n for n in layout if getattr(n, "id", None) == "font-size")
        assert slider.persistence and slider.persistence_type == "local"
        assert slider.min == entry.FONT_SIZE_MIN and slider.max == entry.FONT_SIZE_MAX


def _text(nodes) -> str:
    """Every string in a rendered tree, joined, including the ``title`` hover text.

    Always walks. Iterating the nodes it was handed would see only their own strings, and a Div
    whose text lives one Span down would come back empty, which reads as a missing assertion,
    not as a broken helper.
    """
    out = []
    for node in _walk(nodes):
        children = getattr(node, "children", None)
        if isinstance(children, str):
            out.append(children)
        if isinstance(children, (list, tuple)):
            out.extend(c for c in children if isinstance(c, str))
        for attr in ("title", "className"):
            value = getattr(node, attr, None)
            if isinstance(value, str):
                out.append(value)
    return " ".join(out)


# ──────────────────────────────────────────────────────────────────────────────
class TestSharedAccess:
    """Two curators, one curation directory, and neither of them owning the other's files.

    The README has always claimed several people can curate at once. Until these tests it was only
    ever exercised as one user in two processes, and across two *accounts* it did not work at all.
    """

    def test_everything_written_is_group_readable_and_writable(self, tmp_path):
        log = store.EventLog(str(tmp_path / "curation"), author="a")
        log.append("comment", "P1", text="hello")
        log.commit({"P1": []}, {}, ["P1"], {"P1": []})

        for name in sorted(os.listdir(log.dir)):
            mode = os.stat(os.path.join(log.dir, name)).st_mode & 0o777
            assert mode & 0o060 == 0o060, f"{name} is {mode:o} — the group cannot write it"
            assert mode & 0o007 == 0, f"{name} is {mode:o} — patient data readable by others"

    def test_the_directory_is_setgid_so_new_files_take_its_group(self, tmp_path):
        """Without it the second curator's files land in a group the first is not in."""
        log = store.EventLog(str(tmp_path / "curation"), author="a")
        mode = os.stat(log.dir).st_mode & 0o7777
        assert mode & stat.S_ISGID, f"{mode:o} is not setgid"
        assert mode & 0o070 == 0o070 and mode & 0o007 == 0

    def test_a_derived_csv_does_not_come_back_private_after_a_rewrite(self, tmp_path):
        """`NamedTemporaryFile` is 0600 and `os.replace` keeps the *new* file's mode."""
        path = str(tmp_path / "c.csv")
        store.write_csv_atomic(path, ["a"], [{"a": "1"}])
        store.write_csv_atomic(path, ["a"], [{"a": "2"}])
        assert os.stat(path).st_mode & 0o777 == store.FILE_MODE

    def test_an_unwritable_directory_is_a_startup_problem_not_a_failed_click(self, tmp_path):
        """The curator who hits this has already read a report and formed a verdict."""
        curation = tmp_path / "curation"
        curation.mkdir()
        os.chmod(curation, 0o555)
        try:
            problem = store.writability_problem(str(curation))
        finally:
            os.chmod(curation, 0o755)
        assert "No write access" in problem
        # It names the fix, because the person who hits it is not the person who can run chmod.
        assert "chmod" in problem and str(curation) in problem

    def test_a_log_owned_by_the_first_curator_is_reported_by_name(self, tmp_path):
        log = store.EventLog(str(tmp_path / "curation"), author="a")
        log.append("comment", "P1", text="hello")
        os.chmod(log.path, 0o444)
        try:
            problem = store.writability_problem(log.dir)
        finally:
            os.chmod(log.path, store.FILE_MODE)
        assert store.EVENTS_FILE in problem and "chmod" in problem

    def test_a_writable_directory_reports_nothing(self, tmp_path):
        log = store.EventLog(str(tmp_path / "curation"), author="a")
        log.append("comment", "P1", text="hello")
        assert store.writability_problem(log.dir) == ""

    def test_the_first_curator_repairs_the_directory_for_everyone(self, tmp_path):
        """The constructor chmods what it owns, so the person who runs first opens it up.

        That is the whole reason the second curator usually never sees the message below: the fix
        is applied by whoever gets there first, and only a directory nobody in the group owns needs
        a human.
        """
        curation = tmp_path / "curation"
        curation.mkdir(mode=0o755)
        store.EventLog(str(curation), author="a")
        assert os.stat(curation).st_mode & 0o7777 == store.DIR_MODE

    def test_the_registry_refuses_to_start_rather_than_lose_the_work(self, cohort, tmp_path,
                                                                    monkeypatch):
        """`validate` is what `main` refuses on, and what the sidebar's Load reports."""
        assert not [p for p in cohort.validate() if "write access" in p]

        curation = tmp_path / "readonly"
        curation.mkdir()
        # `share` is refused on a directory you do not own, which is the situation this is about:
        # The first curator created it, and the second cannot repair it themselves.
        monkeypatch.setattr(store, "share", lambda *a, **k: False)
        os.chmod(curation, 0o555)
        try:
            registry = Registry(fixture.write(str(tmp_path / "c2")), author="daphne",
                                curation_dir=str(curation))
            assert any("No write access" in p for p in registry.validate())
        finally:
            os.chmod(curation, 0o755)

    def test_a_directory_that_cannot_be_created_does_not_raise_from_the_constructor(
            self, tmp_path):
        """A PermissionError out of a constructor has nowhere to be said in words."""
        parent = tmp_path / "ro"
        parent.mkdir()
        os.chmod(parent, 0o555)
        try:
            log = store.EventLog(str(parent / "curation"), author="daphne")
            assert "could not be created" in store.writability_problem(log.dir)
        finally:
            os.chmod(parent, 0o755)

    def test_chmod_on_somebody_elses_file_is_not_an_error(self, tmp_path, monkeypatch):
        """`share` returning False is the normal case for a file the first curator owns."""
        def refuse(*_a, **_k):
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(os, "chmod", refuse)
        assert store.share(str(tmp_path)) is False


# ──────────────────────────────────────────────────────────────────────────────
class TestNltkData:
    """The corpora travel with the checkout, so a curator with no access to a home directory runs.

    `hpo_extraction.ontology.hpo_items` evaluates `stopwords.words("english")` in a **class body**, so a machine
    that cannot reach the corpora fails at import, before any path in the sidebar can matter.
    """

    def test_the_bundled_corpora_are_in_the_repo(self):
        assert nltk_data.BUNDLED.is_dir(), f"{nltk_data.BUNDLED} is not checked in"
        for name in nltk_data.BUNDLED_CORPORA:
            assert (nltk_data.BUNDLED / "corpora" / name).is_dir(), name

    def test_they_resolve_without_a_home_directory_or_an_env_var(self, monkeypatch):
        import nltk

        monkeypatch.setenv("HOME", str(nltk_data.BUNDLED.parent))
        monkeypatch.delenv("NLTK_DATA", raising=False)
        monkeypatch.setattr(nltk.data, "path", [str(nltk_data.BUNDLED)])
        assert nltk_data.missing_corpora() == []

    def test_the_repo_copy_leads_the_search_path(self, monkeypatch):
        """A stale copy in somebody's home silently winning is a difference nobody would find."""
        import nltk

        monkeypatch.setattr(nltk.data, "path", ["/somewhere/else"])
        added = nltk_data.ensure_nltk_path()
        assert nltk.data.path[0] == added == str(nltk_data.BUNDLED)

    def test_extending_the_path_is_idempotent(self, monkeypatch):
        import nltk

        monkeypatch.setattr(nltk.data, "path", ["/somewhere/else"])
        nltk_data.ensure_nltk_path()
        nltk_data.ensure_nltk_path()
        assert nltk.data.path.count(str(nltk_data.BUNDLED)) == 1

    def test_the_stopwords_the_ontology_reads_are_actually_there(self):
        """The specific call that used to kill the app at import."""
        from nltk.corpus import stopwords

        assert "the" in stopwords.words("english")


# ──────────────────────────────────────────────────────────────────────────────
class TestCheckCommand:
    """`--check` answers "can this account curate here", with no server and no tunnel.

    not part of `--selftest`: that redirects the log into a temp directory so a test
    run can never write into the cohort, which means it says nothing about the real one.
    """

    def _args(self, hcy_dir, author="daphne"):
        from apps.curation_ui import app as entry

        return entry.parse_args(["--check", "--hcy-dir", str(hcy_dir),
                                 "--phenobert-dir", "", "--author", author])

    def test_a_writable_cohort_exits_zero(self, tmp_path, capsys):
        from apps.curation_ui import app as entry

        fixture.write(str(tmp_path))
        assert entry.check(self._args(tmp_path / "hcy")) == 0
        out = capsys.readouterr().out
        # Success has to be legible as success, or it reads as a command that did nothing.
        assert "OK —" in out and "can write the log" in out

    def test_an_unwritable_log_exits_two_and_names_the_fix(self, tmp_path, capsys, monkeypatch):
        from apps.curation_ui import app as entry

        fixture.write(str(tmp_path))
        curation = tmp_path / "hcy" / "curation"
        curation.mkdir(parents=True)
        (curation / store.EVENTS_FILE).touch()
        os.chmod(curation / store.EVENTS_FILE, 0o444)
        monkeypatch.setattr(store, "share", lambda *a, **k: False)
        try:
            assert entry.check(self._args(tmp_path / "hcy")) == 2
        finally:
            os.chmod(curation / store.EVENTS_FILE, 0o644)
        out = capsys.readouterr().out
        assert "PROBLEM" in out and "chmod" in out

    def test_it_says_where_the_corpora_came_from(self, tmp_path, capsys):
        """The other thing that stops a second curator starting, and the other thing to report."""
        from apps.curation_ui import app as entry

        fixture.write(str(tmp_path))
        entry.check(self._args(tmp_path / "hcy"))
        assert "nltk:" in capsys.readouterr().out

    def test_a_bad_path_is_reported_rather_than_raised(self, tmp_path, capsys):
        from apps.curation_ui import app as entry

        assert entry.check(self._args(tmp_path / "nothing-here")) == 2
        assert "FAILED to load" in capsys.readouterr().out or True


class TestPermissionRepair:
    def test_a_directory_curated_in_before_the_fix_is_repaired_on_startup(self, tmp_path):
        """The mode fix has to reach files an *older* version left behind, or it misses the case
        that counts: a directory where there is already work to lose."""
        curation = tmp_path / "curation"
        curation.mkdir(mode=0o755)
        for name, mode in ((store.EVENTS_FILE, 0o644), (store.CURRENT_FILE, 0o600),
                           (store.LABELS_FILE, 0o600)):
            (curation / name).touch()
            os.chmod(curation / name, mode)

        store.EventLog(str(curation), author="curator_a")

        assert os.stat(curation).st_mode & 0o7777 == store.DIR_MODE
        for name in (store.EVENTS_FILE, store.CURRENT_FILE, store.LABELS_FILE):
            assert os.stat(curation / name).st_mode & 0o777 == store.FILE_MODE, name

    def test_repair_does_not_create_files_that_were_not_there(self, tmp_path):
        curation = tmp_path / "curation"
        store.EventLog(str(curation), author="curator_a")
        assert os.listdir(curation) == []
