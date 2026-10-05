"""The four normalisers behind one interface, and the layout rules that keep the grid honest.

Two of these tests exist because the failure they catch is silent rather than loud:

* ``phenobert_raw`` must reproduce the shipped ``detections_*.jsonl`` byte for byte. It is the
  provenance column, every published ensemble number in the repository was measured under it, so
  a drift there would make the whole grid a comparison against something that was never run.
* ``build_candidate_phenobert_input``'s ``line_spans`` must point at the text they claim. A drift of
  one character attributes every detection to the previous candidate, which produces a plausible
  table and a wrong one.
"""

from __future__ import annotations

import json

import pytest

from hpo_extraction.phenojury.normalisers import (
    GRID_NORMALISERS,
    LINKAGE_FIELDS,
    NORMALISERS,
    build_candidate_phenobert_input,
    candidate_strings,
    cell_extractions,
    cell_models,
    detections_from_linkage,
    detections_name,
    dictionary_linkage,
    linkage_name,
    load_cached_detections,
    normalise_cell,
    phenobert_linkage,
    sapbert_link_records,
    source_cell_dir,
    write_detections,
    write_linkage,
)

pytestmark = pytest.mark.unit


def record(pid="p1", sent=0, output="Microcephaly\nHypotonia", sentence="the patient is small"):
    return {"model": "m", "patient_id": pid, "sentence_number": sent,
            "sentence_text": sentence, "llm_output": output}


# ── Candidate strings, per prompt shape ──────────────────────────────────────

class TestCandidateStrings:
    def test_plain_lines_are_their_own_candidates(self):
        assert candidate_strings(record(), "plain") == [(0, "Microcephaly"), (1, "Hypotonia")]

    def test_list_markers_and_blank_lines_are_stripped(self):
        rec = record(output="1. Microcephaly\n\n- Hypotonia\n")
        assert candidate_strings(rec, "plain") == [(0, "Microcephaly"), (1, "Hypotonia")]

    def test_json_shape_takes_the_term_field_not_the_span(self):
        """q4's whole point: the span is the report's wording, the term is the ontology's.

        The shipped PhenoBERT column never made this split, it was handed the raw JSON, which is
        the confound the ``phenobert_candidates`` reader exists to remove.
        """
        rec = record(output='{"span": "small head", "term": "Microcephaly"}')
        assert candidate_strings(rec, "json") == [(0, "Microcephaly")]

    def test_pipe_shape_takes_the_first_field(self):
        rec = record(output="Microcephaly | Small head | Head below third centile")
        assert candidate_strings(rec, "pipe_first") == [(0, "Microcephaly")]

    def test_line_index_counts_output_lines_not_candidates(self):
        """Two terms on one JSON line share a line index. That is what makes the index useful."""
        rec = record(output='{"term": "Microcephaly"}, {"term": "Hypotonia"}\n{"term": "Ataxia"}')
        indices = [i for i, _ in candidate_strings(rec, "json")]
        assert indices == [0, 0, 1]


# ── The candidate PhenoBERT input layout ─────────────────────────────────────

class TestCandidateInput:
    def test_line_spans_point_at_the_text_they_name(self):
        """The offset arithmetic must mirror the join, or attribution is off by one line."""
        records = [record(sent=0, output="Microcephaly\nHypotonia"),
                   record(sent=1, output="Ataxia")]
        text, spans, line_spans = build_candidate_phenobert_input(records, "p1", "plain")
        assert [text[s:e] for s, e, _, _ in line_spans] == [
            "Microcephaly;", "Hypotonia;", "Ataxia;"]

    def test_spans_carry_the_sentence_and_line_of_each_candidate(self):
        records = [record(sent=0, output="Microcephaly\nHypotonia"),
                   record(sent=1, output="Ataxia")]
        _text, _spans, line_spans = build_candidate_phenobert_input(records, "p1", "plain")
        assert [(sent, line) for _s, _e, sent, line in line_spans] == [(0, 0), (0, 1), (1, 0)]

    def test_block_spans_cover_the_whole_block_including_its_marker(self):
        records = [record(sent=0, output="Microcephaly"), record(sent=1, output="Ataxia")]
        text, spans, _lines = build_candidate_phenobert_input(records, "p1", "plain")
        for start, end, sent in spans:
            assert text[start:end].startswith(f"sentence_number: {sent};")

    def test_every_line_is_terminated_so_phenobert_cannot_fuse_them(self):
        """Unterminated newlines cost the shipped layout all but 3.8 % of the terms handed to it."""
        records = [record(sent=0, output="Microcephaly\nHypotonia")]
        text, _spans, _lines = build_candidate_phenobert_input(records, "p1", "plain")
        assert all(line.endswith((";", ".", ":", ",", "!", "?"))
                   for line in text.split("\n") if line)


# ── Linkage rows ─────────────────────────────────────────────────────────────

TSV = (
    "0\t12\tMicrocephaly\tHP:0000252\t0.99\t0\tlabel\n"
    "13\t22\tHypotonia\tHP:0001252\t0.95\t0\tlabel\tNeg\n"
)


class TestPhenobertLinkage:
    def test_negated_rows_are_kept_as_a_column_not_dropped(self, tmp_path):
        """``detections_*.jsonl`` is the post-Neg view, so this is the only route back.

        An earlier exploratory run records that collapsing "mentioned but negated" with "never mentioned" splits GSC+
        misses 48 against 199, the distinction is worth an entire error class.
        """
        out = tmp_path / "pb"
        out.mkdir()
        (out / "p1.txt").write_text(TSV, encoding="utf-8")
        rows = phenobert_linkage(str(out), {}, normaliser="phenobert_raw")
        assert [r["negated"] for r in rows] == [False, True]
        assert [r["hpo_id"] for r in rows] == ["HP:0000252", "HP:0001252"]

    def test_rows_carry_every_declared_field(self, tmp_path):
        out = tmp_path / "pb"
        out.mkdir()
        (out / "p1.txt").write_text(TSV, encoding="utf-8")
        rows = phenobert_linkage(str(out), {}, normaliser="phenobert_raw")
        assert set(rows[0]) == set(LINKAGE_FIELDS)

    def test_detections_from_linkage_drops_negated_by_default(self, tmp_path):
        out = tmp_path / "pb"
        out.mkdir()
        (out / "p1.txt").write_text(TSV, encoding="utf-8")
        rows = phenobert_linkage(str(out), {}, normaliser="phenobert_raw")
        assert detections_from_linkage(rows) == {"p1": {0: {"HP:0000252": 1}}}
        kept = detections_from_linkage(rows, keep_negated=True)
        assert kept["p1"][0] == {"HP:0000252": 1, "HP:0001252": 1}

    def test_a_missing_output_dir_is_a_warning_and_no_rows(self, tmp_path):
        assert phenobert_linkage(str(tmp_path / "nope"), {}, normaliser="phenobert_raw") == []


def test_dictionary_linkage_keeps_unresolved_lines():
    """An unresolved line is evidence, not absence: it says the reader refused, not that nobody
    wrote anything. Dropping it would make S1's normaliser-union gap unattributable."""
    trace = [{"report_id": "p1", "sentence_number": 0, "line": "Microcephaly",
              "route": "exact", "hpo_ids": ["HP:0000252"]},
             {"report_id": "p1", "sentence_number": 0, "line": "looked unwell",
              "route": "none", "hpo_ids": []}]
    rows = dictionary_linkage(trace)
    assert [r["hpo_id"] for r in rows] == ["HP:0000252", None]
    assert [r["source"] for r in rows] == ["exact", "none"]


def test_dictionary_linkage_splits_a_multi_id_line_into_one_row_each():
    trace = [{"report_id": "p1", "sentence_number": 3, "line": "ASD",
              "route": "exact", "hpo_ids": ["HP:0000729", "HP:0001631"]}]
    rows = dictionary_linkage(trace)
    assert len(rows) == 2
    assert {r["hpo_id"] for r in rows} == {"HP:0000729", "HP:0001631"}


# ── SapBERT ──────────────────────────────────────────────────────────────────

class TestSapbert:
    @staticmethod
    def encoder(vectors):
        """A fake encoder: exact string -> unit vector, anything else orthogonal."""
        import numpy as np

        def encode_batch(texts):
            return np.asarray([vectors.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype="float32")
        return encode_batch

    def test_a_candidate_above_tau_resolves_and_below_tau_does_not(self):
        import numpy as np

        surface_ids = ["HP:0000252", "HP:0001252"]
        surface_vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
        encode = self.encoder({"Microcephaly": [1.0, 0.0, 0.0], "looked unwell": [0.0, 0.0, 1.0]})
        records = [record(output="Microcephaly\nlooked unwell")]
        det, linkage = sapbert_link_records(
            records, ["p1"], surface_ids=surface_ids, surface_vectors=surface_vectors,
            encode_batch=encode, shape="plain", tau=0.8)
        assert det == {"p1": {0: {"HP:0000252": 1}}}
        assert [r["source"] for r in linkage] == ["sapbert", "below_tau"]

    def test_below_tau_rows_keep_their_score(self):
        """The refused rows are the point: they separate "nobody wrote it" from "the reader said no"."""
        import numpy as np

        encode = self.encoder({"Microcephaly": [1.0, 0.0, 0.0]})
        _det, linkage = sapbert_link_records(
            [record(output="looked unwell")], ["p1"], surface_ids=["HP:0000252"],
            surface_vectors=np.asarray([[1.0, 0.0, 0.0]], dtype="float32"),
            encode_batch=encode, shape="plain", tau=0.8)
        assert linkage[0]["hpo_id"] is None
        assert linkage[0]["score"] == pytest.approx(0.0, abs=1e-6)

    def test_reports_outside_the_cohort_are_filtered_not_trusted(self):
        """A cached extractions file may cover a superset of the cohort being scored."""
        import numpy as np

        encode = self.encoder({"Microcephaly": [1.0, 0.0, 0.0]})
        det, _linkage = sapbert_link_records(
            [record(pid="p1"), record(pid="p9")], ["p1"], surface_ids=["HP:0000252"],
            surface_vectors=np.asarray([[1.0, 0.0, 0.0]], dtype="float32"),
            encode_batch=encode, shape="plain", tau=0.8)
        assert set(det) == {"p1"}


# ── Round trips and layout ───────────────────────────────────────────────────

def test_write_detections_round_trips_through_the_shipped_reader(tmp_path):
    detections = {"p1": {0: {"HP:0000252": 1}, 2: {"HP:0001252": 3}}}
    path = write_detections(detections, "llama", tmp_path / "d.jsonl")
    assert load_cached_detections(path) == detections


def test_load_cached_detections_accepts_both_id_spellings(tmp_path):
    """``report_id`` in the detections files, ``patient_id`` in the extractions. Both are real."""
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"report_id": "p1", "sentence_number": 0, "hpo_id": "HP:1", "count": 1}) + "\n"
        + json.dumps({"patient_id": "p2", "sentence_number": 1, "hpo_id": "HP:2", "count": 1})
        + "\n", encoding="utf-8")
    assert load_cached_detections(path) == {"p1": {0: {"HP:1": 1}}, "p2": {1: {"HP:2": 1}}}


def test_write_linkage_emits_declared_fields_in_order(tmp_path):
    path = write_linkage([{"report_id": "p1", "hpo_id": "HP:1"}], tmp_path / "l.jsonl")
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert list(row) == list(LINKAGE_FIELDS)


class TestLayout:
    def test_the_baseline_falls_back_to_exp13_06s_own_directory(self, tmp_path):
        """p0 predates earlier and has no prompt path segment. That is a fact, not a bug to fix."""
        legacy = tmp_path / "phenojury_generation_free_listing" / "hcy"
        legacy.mkdir(parents=True)
        assert source_cell_dir(tmp_path, "hcy", "p0_baseline") == legacy

    def test_a_staged_cell_wins_over_the_legacy_path(self, tmp_path):
        (tmp_path / "phenojury_generation_free_listing" / "hcy").mkdir(parents=True)
        staged = tmp_path / "phenojury_generation_other_prompts" / "hcy" / "p0_baseline"
        staged.mkdir(parents=True)
        assert source_cell_dir(tmp_path, "hcy", "p0_baseline") == staged

    def test_a_missing_cell_raises_rather_than_scoring_zero(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="q2_sentence_last"):
            source_cell_dir(tmp_path, "hcy", "q2_sentence_last")

    def test_cell_models_excludes_an_empty_detections_file(self, tmp_path):
        """The q7 x apertus shape: generation succeeded, grounding did not, the file is 0 bytes.

        ``slm_ensemble_experiment`` excludes such a model from the whole aggregation, so an analysis
        that counted it would report a matched contrast that was not matched.
        """
        (tmp_path / "detections_llama.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "detections_apertus.jsonl").write_text("", encoding="utf-8")
        assert cell_models(tmp_path) == ["llama"]

    def test_cell_extractions_skips_torn_lines(self, tmp_path):
        (tmp_path / "llm_extractions_m.jsonl").write_text(
            json.dumps(record()) + "\n{\"truncated\": \n", encoding="utf-8")
        assert len(cell_extractions(tmp_path, "m")) == 1

    def test_artifact_names_are_stable(self):
        assert detections_name("llama", "dictionary") == "detections_llama__dictionary.jsonl"
        assert linkage_name("llama", "sapbert") == "linkage_llama__sapbert.jsonl"


def test_the_reference_reader_is_not_one_of_the_grid_columns():
    """``phenobert_raw`` reads the whole generation. The grid's three read parsed candidates.

    Putting it inside the grid would compare a reader against a parser.
    """
    assert "phenobert_raw" in NORMALISERS
    assert "phenobert_raw" not in GRID_NORMALISERS
    assert set(GRID_NORMALISERS) < set(NORMALISERS)


def test_an_unknown_normaliser_is_refused():
    with pytest.raises(ValueError, match="unknown normaliser"):
        normalise_cell([], [], "word2vec", "p0_baseline")


def test_dictionary_without_a_surface_index_is_refused():
    with pytest.raises(ValueError, match="surface2hpo"):
        normalise_cell([record()], ["p1"], "dictionary", "p0_baseline")


def test_sapbert_without_an_encoder_is_refused():
    with pytest.raises(ValueError, match="encode_batch"):
        normalise_cell([record()], ["p1"], "sapbert", "p0_baseline")
