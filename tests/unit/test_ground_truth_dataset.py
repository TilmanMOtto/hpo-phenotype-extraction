"""The ground-truth build: the curated ground truth shipped as a dataset back into the cohort directory.

``curated_gold`` decides what is in the ground truth. This module writes that decision down as something a
second person can pick up a year later, and what is being guarded here is the difference between
those two jobs.

The essential promise is **nothing is deleted, only flagged**. A dataset that has already
applied its own opinions cannot be re-used by anyone who holds a different one, and the ``family``
exclusion in particular is an editorial call about what the task *is* rather than a statement about
evidence. So every excluded annotation has to survive into the annotation table with the reason it
was excluded and a column per qualifier, which means the ground truth is reconstructible from the table,
and so is a different ground truth.

The second promise is that the **README cannot drift from the data**: its counts, its qualifier
vocabulary and its column list all come from the same objects the CSVs are written from, so a
column added without a line of prose is a test failure, not a stale document.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def modules(monkeypatch):
    """``curated_gold`` and ``dataset`` as top-level modules, the way ``run.py`` imports them."""
    exp_dir = _REPO / "experiments" / "03_setup" / "ground_truth"
    monkeypatch.syspath_prepend(str(exp_dir))
    monkeypatch.syspath_prepend(str(_REPO))
    for name in ("curated_gold", "dataset"):
        sys.modules.pop(name, None)
    loaded = (importlib.import_module("curated_ground_truth"), importlib.import_module("dataset"))
    yield loaded
    for name in ("curated_gold", "dataset"):
        sys.modules.pop(name, None)


@pytest.fixture
def hcy_dir(tmp_path):
    from apps.curation_ui import fixture

    paths = fixture.write(str(tmp_path), with_phenobert=False)
    Path(paths["hcy_dir"], "curation").mkdir(parents=True, exist_ok=True)
    return Path(paths["hcy_dir"])


@pytest.fixture
def written(modules, hcy_dir):
    """A dataset on disk, built from the fixture cohort with two qualifier-bearing suggestions."""
    from hpo_extraction.curation import store

    curated_gold, dataset = modules
    log = store.EventLog(str(hcy_dir / "curation"), author="test")
    log.append("suggest", "SYN001", hpo_code="HP:0002014", hpo_name="Diarrhea",
               trigger_word="birth", segment_idx=2, labels=["family"])
    log.append("suggest", "SYN001", hpo_code="HP:0001903", hpo_name="Anemia",
               trigger_word="markedly", segment_idx=2, labels=["unsure_report", "negated"],
               note="the report negates it")
    log.append("label", "SYN002", labels=["family_member", "paraphrase"], difficulty="hard")

    paths = curated_gold.resolve_paths(str(hcy_dir))
    result = curated_gold.build(curated_gold.load_annotations(paths),
                                curated_gold.load_curation(paths["curation_dir"]),
                                corpus=curated_gold.load_corpus(paths))
    payload = dataset.write(str(hcy_dir), result, paths, date="2026-09-08")
    return Path(payload["path"]), payload, result, dataset


def _rows(directory: Path, name: str) -> list:
    with open(directory / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ── what lands, and where ────────────────────────────────────────────────────

def test_the_dataset_is_a_dated_directory_beside_the_cohort(written, hcy_dir):
    """Dated, because the curation log keeps growing and a thesis number has to name its version."""
    out, payload, _, dataset = written
    assert out == hcy_dir / "curated_ground_truth_2026-09-08"
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "hcy_curated_annotations.csv", "hcy_curated_reports.csv",
        "hcy_ground_truth_curated.csv", "manifest.json",
    ]
    assert payload["date"] == "2026-09-08"


def test_the_two_column_gold_is_the_repos_own_shape(written):
    """It has to feed ``HCYDataset`` unchanged, or shipping it buys nothing."""
    from hpo_extraction.evaluation.datasets.hcy import HCYDataset

    out, _, result, _ = written
    loaded = HCYDataset(str(out / "hcy_ground_truth_curated.csv"), "").load_ground_truth()
    assert {str(k): {str(c).strip() for c in v if str(c).strip()}
            for k, v in loaded.items()} == result.gold


# ── nothing is deleted, only flagged ─────────────────────────────────────────

def test_excluded_annotations_survive_with_their_reason(written):
    """The promise the whole file rests on.

    A `family` finding is real and the text supports it. Excluding it is an editorial call about
    what the task is. Shipping only the survivors would make that call on behalf of everyone who
    reads the dataset afterwards, with no way to see it had been made.
    """
    out, _, _, _ = written
    rows = {(r["patient_id"], r["hpo_code"]): r
            for r in _rows(out, "hcy_curated_annotations.csv")}

    family = rows[("SYN001", "HP:0002014")]
    assert family["in_gold"] == "0"
    assert family["exclude_reason"] == "label:family"
    assert family["q_family"] == "1"

    unanchorable = rows[("SYN001", "HP:0001873")]
    assert (unanchorable["in_gold"], unanchorable["exclude_reason"]) == ("0", "no_evidence")


def test_the_shipped_gold_is_rebuildable_from_the_annotation_table(written):
    """`in_gold = 1`, grouped by patient, must reproduce the two-column file.

    If it did not, the two files would be two different claims and the README's worked example
    would be wrong, which is worse than shipping one file, because a reader would trust it.
    """
    out, _, result, _ = written
    rebuilt: dict = {pid: set() for pid in result.gold}
    for row in _rows(out, "hcy_curated_annotations.csv"):
        if row["in_gold"] == "1":
            rebuilt[row["patient_id"]].add(row["hpo_code"])
    assert rebuilt == result.gold


def test_a_reader_can_build_a_different_gold_from_the_columns(written):
    """The point of shipping the qualifiers: `keep_family` is a filter, not a re-run.

    Rebuilding the ``keep_family`` variant out of the table alone is the check that every column a
    policy decision reads is actually present, status, the evidence location flag, and one column per
    qualifier.
    """
    out, _, _, _ = written
    out_statuses = {"removed", "rejected", "needs_work", "delete_suggested"}
    keep_family: dict = {}
    for row in _rows(out, "hcy_curated_annotations.csv"):
        if (row["anchored"] == "1" and row["status"] not in out_statuses
                and row["q_unsure_report"] == "0" and row["q_unsure_annotation"] == "0"):
            keep_family.setdefault(row["patient_id"], set()).add(row["hpo_code"])
    assert "HP:0002014" in keep_family["SYN001"]
    assert "HP:0001903" not in keep_family["SYN001"]   # still unsure, still out


def test_every_qualifier_has_its_own_column(written):
    """One `0`/`1` column per qualifier, named from the app's vocabulary, not listed twice.

    The columns are generated from ``labels.label_ids``, so a qualifier added to the curation app
    cannot silently fail to reach the dataset that is supposed to explain it.
    """
    from hpo_extraction.curation import labels as vocab

    out, _, _, dataset = written
    header = _rows(out, "hcy_curated_annotations.csv")[0].keys()
    assert dataset.QUALIFIERS == tuple(vocab.label_ids("annotation"))
    for name in dataset.QUALIFIERS:
        assert f"q_{name}" in header
    row = next(r for r in _rows(out, "hcy_curated_annotations.csv")
               if r["hpo_code"] == "HP:0001903")
    assert (row["q_unsure_report"], row["q_negated"]) == ("1", "1")
    assert row["qualifiers"] == "unsure_report;negated"
    assert row["note"] == "the report negates it"


def test_the_evidence_travels_with_the_annotation(written):
    """A term names the words it came from, which is the whole difference from the old ground truth."""
    out, _, _, _ = written
    row = next(r for r in _rows(out, "hcy_curated_annotations.csv")
               if r["patient_id"] == "SYN001" and r["hpo_code"] == "HP:0001250")
    assert row["trigger_word"] == "seizures"
    assert row["segment_idx"] == "1"
    assert row["trigger_word"] in row["segment_text"]
    assert row["anchored"] == "1" and row["anchor_how"] in ("segment", "offset", "context",
                                                            "lexical", "curated")


# ── the report table ─────────────────────────────────────────────────────────

def test_the_report_table_lists_the_excluded_reports_too(written):
    """"How much of the cohort is this" has to be answerable from the file itself."""
    out, _, _, _ = written
    rows = {r["patient_id"]: r for r in _rows(out, "hcy_curated_reports.csv")}
    assert rows["SYN003"]["in_cohort"] == "0"
    assert rows["SYN003"]["cohort_reason"] == "no annotation from any source"
    assert rows["SYN001"]["in_cohort"] == "1"


def test_the_report_labels_are_wide_and_joinable(written):
    """One row per report and a column per label, so slicing recall is a `merge` and a `groupby`."""
    out, _, _, dataset = written
    rows = {r["patient_id"]: r for r in _rows(out, "hcy_curated_reports.csv")}
    assert dataset.REPORT_LABELS, "the report vocabulary must not resolve to nothing"
    for name in dataset.REPORT_LABELS:
        assert f"r_{name}" in rows["SYN002"]
    assert rows["SYN002"]["difficulty"] == "hard"
    assert rows["SYN002"]["r_family_member"] == "1"
    assert rows["SYN002"]["r_paraphrase"] == "1"
    assert rows["SYN002"]["r_abbreviation"] == "0"


# ── provenance and prose ─────────────────────────────────────────────────────

def test_the_manifest_names_its_inputs_and_the_log_it_folded(written):
    """A number in a thesis has to be traceable to the version of the log that produced it."""
    out, payload, _, _ = written
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == payload or manifest["dataset"] == payload["dataset"]
    assert manifest["inputs"]["curation_log"].endswith("curation_events.jsonl")
    assert len(manifest["inputs"]["curation_log_sha256_16"]) == 16
    assert manifest["policy"]["require_evidence"] is True
    assert manifest["policy"]["exclude_labels"] == ["unsure_report", "unsure_annotation", "family"]


def test_the_readme_documents_every_column_it_ships(written):
    """The document cannot drift from the data if it is generated from the same objects.

    Checked column by column, not by eyeballing: a column added without a line of prose is
    a dataset whose reader has to guess, which is the failure a README exists to prevent.
    """
    out, _, _, dataset = written
    text = (out / "README.md").read_text(encoding="utf-8")
    for column in dataset.ANNOTATION_FIELDS:
        if column.startswith("q_"):
            continue        # documented as one `q_*` row plus the qualifier table below it
        assert f"`{column}`" in text, f"{column} is shipped but not described"
    for name in dataset.QUALIFIERS:
        assert f"`{name}`" in text
    for value in ("curated", "segment", "offset", "context", "lexical"):
        assert f"`{value}`" in text


def test_the_readme_leads_with_the_patient_data_boundary(written):
    """It carries report sentences. The first screen has to say so, to a person.

    HCY data stays on LeoMed (``docs/cluster.md``). This is the part of that rule that a human
    reads before deciding to copy something.
    """
    out, _, _, _ = written
    head = (out / "README.md").read_text(encoding="utf-8")[:900]
    assert "Patient data" in head
    assert "stays on the cluster" in head


def test_the_readme_counts_agree_with_the_files(written):
    """Prose and table are generated from one ``counts`` call, and this pins that they stay so."""
    out, payload, result, _ = written
    stats = payload["counts"]
    rows = _rows(out, "hcy_curated_annotations.csv")
    assert stats["n_annotations"] == len(rows)
    assert stats["n_terms_unanchored"] == sum(1 for r in rows if r["anchored"] == "0")
    # `n_gold_pairs` counts (report, term). The annotation table counts *annotations*, and one term
    # can have two pieces of evidence. That gap is the README's last note, not a discrepancy,
    # SYN002 carries HP:0001263 on two different triggers.
    assert stats["n_gold_pairs"] == len({(r["patient_id"], r["hpo_code"])
                                         for r in rows if r["in_gold"] == "1"})
    assert stats["n_gold_pairs"] < sum(1 for r in rows if r["in_gold"] == "1")
    text = (out / "README.md").read_text(encoding="utf-8")
    assert f"**{result.n_pairs} (report, term) pairs" in text


def test_writing_twice_on_one_day_replaces_rather_than_appends(written, modules, hcy_dir):
    """Re-running while iterating must not leave two versions of the same day's dataset."""
    curated_gold, dataset = modules
    out, _, result, _ = written
    before = len(_rows(out, "hcy_curated_annotations.csv"))
    paths = curated_gold.resolve_paths(str(hcy_dir))
    dataset.write(str(hcy_dir), result, paths, date="2026-09-08")
    assert len(_rows(out, "hcy_curated_annotations.csv")) == before
