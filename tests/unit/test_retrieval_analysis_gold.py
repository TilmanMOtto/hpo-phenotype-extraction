"""``hpo_extraction.evaluation.retrieval_analysis.load_inputs`` against the two HCY annotation formats.

The rank analyses in group 0 (``an earlier exploratory run``, ``an earlier exploratory run``) score retrieval against a segment-level
ground truth, and that ground truth changed when the ``app/hcy_curation_ui`` pass finished. What is tested here is
the seam between the two files, because reading one as the other is silent and wrong in a specific
way: the curated table lists *candidates*, so a loader that ignored ``in_gold`` would score
retrieval against the very annotations the curation policy rejected, and one that ignored
``anchored`` would count an annotation whose trigger word occurs nowhere in the report as a
retrieval miss for a sentence that does not exist.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hpo_extraction.evaluation.retrieval_analysis import load_inputs

SEGMENTS = pd.DataFrame([
    {"patient_id": "SYN001", "sentence_idx": 0, "sentence": "He had recurrent seizures."},
    {"patient_id": "SYN001", "sentence_idx": 1, "sentence": "He was markedly hypotonic."},
    {"patient_id": "SYN002", "sentence_idx": 0, "sentence": "There was no thrombocytopenia."},
])

#: One row per candidate annotation, in the shape ``hcy_ground_truth``'s ``dataset`` stage writes.
CURATED = [
    # in the ground truth, placed by a curator
    {"patient_id": "SYN001", "hpo_code": "HP:0001250", "in_gold": "1", "segment_idx": "0",
     "anchored": "1", "anchor_how": "curated", "trigger_word": "seizures", "source": "daphne"},
    # The same phenotype in the same place, recorded by the other file, one annotation, not two
    {"patient_id": "SYN001", "hpo_code": "HP:0001250", "in_gold": "1", "segment_idx": "0",
     "anchored": "1", "anchor_how": "lexical", "trigger_word": "seizures", "source": "prior_annotation"},
    # in the ground truth, but only a lexical scan found the words
    {"patient_id": "SYN001", "hpo_code": "HP:0001252", "in_gold": "1", "segment_idx": "1",
     "anchored": "1", "anchor_how": "lexical", "trigger_word": "hypotonic", "source": "prior_annotation"},
    # The policy rejected it, a `family` qualifier, a removal verdict, a deletion proposal
    {"patient_id": "SYN001", "hpo_code": "HP:0001903", "in_gold": "0", "segment_idx": "1",
     "anchored": "1", "anchor_how": "curated", "trigger_word": "hypotonic", "source": "daphne"},
    # unanchorable: the file named words the report does not contain, so there is nothing to rank
    {"patient_id": "SYN002", "hpo_code": "HP:0001873", "in_gold": "0", "segment_idx": "",
     "anchored": "0", "anchor_how": "", "trigger_word": "purpura", "source": "prior_annotation"},
]

CONFIRMED = [
    {"patient_id": "SYN001", "segment_idx": "0", "hpo_code": "HP:0001250",
     "trigger_word": "seizures", "confirmed": "1"},
    {"patient_id": "SYN002", "segment_idx": "0", "hpo_code": "HP:0001873",
     "trigger_word": "purpura", "confirmed": "1"},
]


@pytest.fixture
def segments_csv(tmp_path):
    path = tmp_path / "segmented_reports.csv"
    SEGMENTS.to_csv(path, index=False)
    return str(path)


def _write(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def curated_csv(tmp_path):
    return _write(tmp_path, "hcy_curated_annotations.csv", CURATED)


@pytest.fixture
def confirmed_csv(tmp_path):
    return _write(tmp_path, "annotations_confirmed.csv", CONFIRMED)


class TestCuratedGold:
    def test_only_in_gold_located_rows_survive_and_duplicates_fold(self, segments_csv,
                                                                    curated_csv):
        _, ann, patients = load_inputs(segments_csv, curated_csv)

        assert list(ann["hpo_code"]) == ["HP:0001250", "HP:0001252"]
        # SYN002's only annotation was unanchorable, so the report leaves the analysis entirely
        # rather than contributing a phenotype nothing could have retrieved.
        assert patients == ["SYN001"]
        assert ann["segment_idx"].tolist() == [0, 1]
        assert ann["segment_idx"].dtype.kind == "i"

    def test_folded_duplicate_keeps_the_better_placement(self, segments_csv, curated_csv):
        _, ann, _ = load_inputs(segments_csv, curated_csv)
        row = ann[ann["hpo_code"] == "HP:0001250"].iloc[0]
        assert row["anchor_how"] == "curated"

    def test_provenance_counts_every_rule(self, segments_csv, curated_csv):
        _, ann, _ = load_inputs(segments_csv, curated_csv)
        gold = ann.attrs["gold"]
        assert gold["gold_source"] == "curated"
        assert gold["n_candidates"] == 5
        assert gold["n_excluded_by_policy"] == 2
        assert gold["n_unanchored"] == 0      # both unanchored rows were already out of the ground truth
        assert gold["n_folded_duplicates"] == 1
        assert gold["n_annotations"] == 2
        assert gold["n_lexical"] == 1

    def test_exclude_location_how_drops_lexical_placements(self, segments_csv, curated_csv):
        _, ann, _ = load_inputs(segments_csv, curated_csv, exclude_location_how=["lexical"])
        # HP:0001250 survives on its curated row; HP:0001252 had only a lexical one.
        assert list(ann["hpo_code"]) == ["HP:0001250"]
        assert ann.attrs["gold"]["n_excluded_by_anchor_how"] == 1

    def test_exclude_location_how_accepts_a_bare_string(self, segments_csv, curated_csv):
        _, ann, _ = load_inputs(segments_csv, curated_csv, exclude_location_how="lexical")
        assert list(ann["hpo_code"]) == ["HP:0001250"]

    def test_require_location_off_still_needs_a_segment_to_rank(self, segments_csv, tmp_path):
        rows = [dict(r) for r in CURATED]
        rows[4]["in_gold"] = "1"          # a `keep_unanchored` ground truth admits the unplaceable term
        path = _write(tmp_path, "keep_unanchored.csv", rows)

        _, ann, patients = load_inputs(segments_csv, path, require_location=False)
        assert "SYN002" not in patients
        assert ann.attrs["gold"]["n_unanchored"] == 1


class TestConfirmedGold:
    def test_legacy_file_still_loads_unfiltered(self, segments_csv, confirmed_csv):
        _, ann, patients = load_inputs(segments_csv, confirmed_csv)
        assert ann.attrs["gold"]["gold_source"] == "confirmed"
        assert len(ann) == 2
        assert patients == ["SYN001", "SYN002"]


class TestSourceMismatch:
    def test_curated_demanded_but_confirmed_given(self, segments_csv, confirmed_csv):
        with pytest.raises(ValueError, match="in_gold"):
            load_inputs(segments_csv, confirmed_csv, gold_source="curated")

    def test_confirmed_demanded_but_curated_given(self, segments_csv, curated_csv):
        # The dangerous direction: read flat, the curated table's rejected terms become ground truth.
        with pytest.raises(ValueError, match="policy excluded"):
            load_inputs(segments_csv, curated_csv, gold_source="confirmed")

    def test_unknown_source(self, segments_csv, curated_csv):
        with pytest.raises(ValueError, match="auto|curated|confirmed"):
            load_inputs(segments_csv, curated_csv, gold_source="approved")


class TestSegmentationDrift:
    def test_index_beyond_the_report_is_a_hard_error(self, segments_csv, tmp_path):
        rows = [dict(CURATED[0])]
        rows[0]["segment_idx"] = "9"
        path = _write(tmp_path, "drifted.csv", rows)
        with pytest.raises(ValueError, match="out of sync"):
            load_inputs(segments_csv, path)
