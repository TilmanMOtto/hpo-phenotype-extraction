"""The PhenoBERT baseline: PhenoBERT's TSVs turned into the earlier predictions contract.

The driver itself is a subprocess call and a parse. The subprocess is an earlier exploratory run's business, so what
is worth pinning here is everything between the TSV and the predictions file, the places where a
detection can quietly become the wrong prediction, or none at all:

* a report id must survive the round trip through a filename (HCY ids carry a ``:``), because the
  id is what joins a prediction to its ground-truth set;
* PhenoBERT ships an **older HPO release**, so an alt id has to be remapped onto the current
  primary, scored as-is, a correct detection would look like a miss, since no ancestor lookup
  resolves on a merged id;
* a code this release does not have at all must be *counted*, not dropped silently;
* negated mentions must stay in the detections dump (the side-by-side analysis needs "found it and
  ruled it out" separated from "never found it") while staying out of the predicted set;
* a report that predicted nothing must still appear in the predictions file, its summary line is
  the only place it exists, and without it the cohort silently shrinks to the reports with hits.
"""

from __future__ import annotations

import json

import pytest

from hpo_extraction.baselines.phenobert_baseline import (
    _output_is_complete,
    collect_detections,
    predicted_sets,
    safe_name,
    stage_reports,
    write_predictions,
)
from fixtures.toy_ontology import C, D, HALLUCINATION, M, OBSOLETE_C

pytestmark = pytest.mark.unit


REPORTS = {
    "patient:01": "The patient had seizures.\nNo myopia was noted.",
    "patient:02": "Marked anxiety on examination.",
    "1003450": "An unremarkable report.",
}


def _write_tsv(directory, stem: str, rows: list[str]) -> None:
    (directory / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


# ── staging ──────────────────────────────────────────────────────────────────

def test_report_ids_survive_the_round_trip_through_a_filename(tmp_path):
    """``patient:01`` cannot be a filename, but the id map must bring it back unchanged."""
    input_dir = tmp_path / "phenobert_input"
    stem_to_id = stage_reports(REPORTS, sorted(REPORTS), str(input_dir))

    assert stem_to_id == {"patient_01": "patient:01", "patient_02": "patient:02",
                          "1003450": "1003450"}
    assert safe_name("patient:01") == "patient_01"


def test_staged_text_is_byte_identical_to_the_report(tmp_path):
    """The TSV's offsets index this file, normalising the text would invalidate every one."""
    input_dir = tmp_path / "phenobert_input"
    stage_reports(REPORTS, ["patient:01"], str(input_dir))
    written = (input_dir / "patient_01.txt").read_text(encoding="utf-8")
    assert written == REPORTS["patient:01"]


def test_two_ids_that_stage_to_one_filename_are_refused(tmp_path):
    """``a:b`` and ``a_b`` both stage as ``a_b.txt``. One would be annotated and reported as both."""
    reports = {"a:b": "text one", "a_b": "text two"}
    with pytest.raises(ValueError, match="a_b.txt"):
        stage_reports(reports, sorted(reports), str(tmp_path / "in"))


def test_a_partial_output_directory_is_not_complete(tmp_path):
    """A job killed mid-directory leaves detections behind. Reusing them scores a cohort prefix."""
    out = tmp_path / "phenobert_output"
    out.mkdir()
    _write_tsv(out, "patient_01", [f"16\t24\tseizures\t{C}\t0.97"])

    assert not _output_is_complete(str(out), ["patient_01", "patient_02"])
    assert _output_is_complete(str(out), ["patient_01"])


# ── parsing ──────────────────────────────────────────────────────────────────

@pytest.fixture
def detections(tmp_path, toy_tree):
    """Four detections over two reports, plus one output file with no staged report."""
    out = tmp_path / "phenobert_output"
    out.mkdir()
    _write_tsv(out, "patient_01", [
        f"16\t24\tseizures\t{C}\t0.97",
        f"30\t36\tmyopia\t{M}\t0.90\tNeg",          # negated
        f"40\t56\tobsolete finding\t{OBSOLETE_C}\t0.71",   # alt id → C
        f"60\t76\tinvented finding\t{HALLUCINATION}\t0.55",  # not in this release
    ])
    _write_tsv(out, "patient_02", [f"7\t14\tanxiety\t{D}\t0.93"])
    _write_tsv(out, "stray_report", [f"0\t8\tseizures\t{C}\t0.99"])

    stem_to_id = {"patient_01": "patient:01", "patient_02": "patient:02"}
    return collect_detections(str(out), stem_to_id, toy_tree)


def test_an_alt_id_is_remapped_onto_the_current_primary(detections):
    """Scored as the merged id, a correct detection would be a miss, no ancestor resolves on it."""
    records, counts = detections
    alt = next(r for r in records if r["raw_hpo_id"] == OBSOLETE_C)
    assert (alt["hpo_id"], alt["resolved"]) == (C, True)
    assert counts["n_alt_id"] == 1


def test_a_code_absent_from_this_release_is_counted_not_dropped(detections):
    records, counts = detections
    unknown = next(r for r in records if r["raw_hpo_id"] == HALLUCINATION)
    assert unknown["resolved"] is False
    assert counts["n_unmapped"] == 1


def test_negation_score_and_offsets_are_carried_through(detections):
    records, _ = detections
    negated = next(r for r in records if r["phrase"] == "myopia")
    assert negated["negated"] is True
    assert (negated["start"], negated["end"], negated["score"]) == (30, 36, 0.90)
    assert negated["report_id"] == "patient:01"


def test_an_output_file_with_no_staged_report_is_counted_and_skipped(detections):
    """A stale file from an earlier cohort must not smuggle detections into this run."""
    records, counts = detections
    assert counts["n_unknown_file"] == 1
    assert all(r["report_id"] in {"patient:01", "patient:02"} for r in records)


def test_the_predicted_set_drops_negated_and_unresolved_detections(detections):
    records, _ = detections
    predicted = predicted_sets(records, ["patient:01", "patient:02"])
    assert predicted == {"patient:01": [C], "patient:02": [D]}


def test_a_report_with_no_detections_still_gets_an_empty_predicted_set(detections):
    """Reports are keyed from the cohort, not from what PhenoBERT happened to find."""
    records, _ = detections
    predicted = predicted_sets(records, ["patient:01", "patient:02", "patient:03"])
    assert predicted["patient:03"] == []


# ── the predictions contract ─────────────────────────────────────────────────

def test_predictions_file_reads_back_through_the_exp13_07_loader(tmp_path, toy_tree,
                                                                 exp13_modules):
    """The artifact is only useful if ``result_tables`` reads it, so read it with the real loader."""
    path = tmp_path / "phenobert_predictions.jsonl"
    predicted = {"r1": [C, D], "r2": []}
    gold = {"r1": [C, M], "r2": [D]}
    n_pred = write_predictions(str(path), ["r1", "r2"], predicted, gold, toy_tree)

    assert n_pred == 2
    loaded_pred, loaded_gold = exp13_modules.loaders.load_predictions(path)
    assert loaded_pred == {"r1": {C, D}, "r2": set()}
    assert loaded_gold == {"r1": {C, M}, "r2": {D}}


def test_every_report_gets_a_summary_line_including_the_empty_one(tmp_path, toy_tree):
    """The summary line is the only place a report that predicted nothing appears at all."""
    path = tmp_path / "phenobert_predictions.jsonl"
    write_predictions(str(path), ["r1", "r2"], {"r1": [C]}, {"r1": [C], "r2": [D]}, toy_tree)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    summaries = {r["report_id"]: r for r in records if r.get("summary")}
    assert set(summaries) == {"r1", "r2"}
    assert summaries["r2"]["predicted_set"] == []
    assert summaries["r2"]["gold_set"] == [D]


def test_per_term_lines_carry_the_gold_join(tmp_path, toy_tree):
    """``ground_truth`` is the per-term join the error analysis reads before any metric runs."""
    path = tmp_path / "phenobert_predictions.jsonl"
    write_predictions(str(path), ["r1"], {"r1": [C, D]}, {"r1": [C]}, toy_tree)

    terms = [json.loads(line) for line in path.read_text().splitlines()
             if not json.loads(line).get("summary")]
    assert {t["hpo_id"]: t["ground_truth"] for t in terms} == {C: 1, D: 0}
    assert {t["prediction"] for t in terms} == {1}
    assert next(t for t in terms if t["hpo_id"] == C)["hpo_label"] == "Seizure"
