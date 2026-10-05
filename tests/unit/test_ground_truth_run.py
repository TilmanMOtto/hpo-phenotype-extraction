"""End-to-end: ``hcy_ground_truth/run.py`` against a miniature earlier output tree and a hand-built cohort.

``test_exp13_18_curated_gold`` pins the ground truth builder in isolation. This one pins that it composes
with the scorer, that the restriction reaches the predictions, that both ground-truth sets meet the same
report list, and that the delta table is a gold-side difference rather than a change of
denominator dressed up as one.

The scenario is arranged so the answer is checkable by hand:

===========  ===============  ===============  ==============  ==========================
report       predicted        original ground truth    curated ground truth    effect
===========  ===============  ===============  ==============  ==========================
``r1``       ``{C, D}``       ``{C, F}``       ``{C, D}``      one FP and one FN become
                                                               one TP, the curation found
                                                               a real term and dropped a
                                                               wrong one
``r2``       ``{D}``          ``{D}``          ``{D}``         unchanged
``r3``       ``{C}``          ``{C}``, no annotation from any
                                                               source: out of both
===========  ===============  ===============  ==============  ==========================

So over the cohort ``{r1, r2}``: original TP=2 FP=1 FN=1 (micro F1 = 2/3), curated TP=3 FP=0 FN=0
(micro F1 = 1.0). ``r3`` must not appear on either side, or the two columns stop being comparable.

Every trigger word in the fixture really occurs in its segment, so the evidence rule admits every
annotation and the numbers above are about the *policy*. The rule itself is fixed in
``test_exp13_18_curated_gold``, on the curation app's own fixture, which contains the annotations
that cannot be placed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from fixtures.toy_ontology import C, D, F

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_EXP_DIR = _REPO / "experiments" / "03_setup" / "ground_truth"


def _write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


#: One segment per report, and the verbatim report text is that segment. Every trigger word below
#: is a substring of the report it belongs to, the evidence rule is exercised on the curation
#: app's own fixture, not here, and a trigger that failed to evidence location would silently turn a policy
#: test into an locating test.
_REPORTS = {
    "r1": "The boy had recurrent seizures and one odd word.",
    "r2": "Another word appears here.",
    "r3": "Nothing abnormal was noted.",
    "r4": "Nothing abnormal was noted here either.",
}


@pytest.fixture
def hcy_dir(tmp_path):
    """A cohort in the real file shapes: two annotated reports and two nobody annotated.

    ``r3`` and ``r4`` are the two controls for the cohort filter, and the difference between them
    is the whole of it. Neither carries an annotation from any source. ``r3`` is listed in the
    original ground truth **with an empty code list**, that file asserts it has no phenotype, so the
    ``original_gold`` criterion admits it with an empty ground truth and every method takes a false
    positive for anything it predicts there. ``r4`` is in the segmentation and nowhere else:
    nobody has said anything about it, and an empty ground truth would claim it has no phenotypes, not that nobody looked, so it stays out.
    """
    root = tmp_path / "hcy"
    _write_csv(root / "segmented_reports.csv",
               ["patient_id", "sentence_idx", "sentence"],
               [[pid, 0, text] for pid, text in _REPORTS.items()])
    _write_csv(root / "hcy_holistic_ground_truth.csv",
               ["patient_id", "report_text", "hpo_code", "hpo_name", "trigger_word",
                "sentence_context", "char_offset"],
               [["r1", _REPORTS["r1"], C, "Seizure", "seizures", "", ""],
                # The term the confirmed pass dropped, and the curator later removed for good.
                ["r1", _REPORTS["r1"], F, "Wrong term", "word", "", ""],
                ["r2", _REPORTS["r2"], D, "Term D", "word", "", ""]])
    _write_csv(root / "annotations_confirmed.csv",
               ["patient_id", "segment", "segment_idx", "hpo_code", "hpo_name", "trigger_word",
                "provenance", "confirmed"],
               [["r1", _REPORTS["r1"], 0, C, "Seizure", "seizures", "manual", "True."],
                ["r2", _REPORTS["r2"], 0, D, "Term D", "word", "manual", "True."]])
    (root / "curation").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def curated_log(hcy_dir):
    """The curation somebody actually did: one removal, one real suggestion, and two the policy drops."""
    from hpo_extraction.curation import store

    log = store.EventLog(str(hcy_dir / "curation"), author="test")
    # F was in the holistic file and the confirmed pass dropped it. A curator confirmed the drop.
    log.append("verdict", "r1", status="removed", hpo_code=F,
               target_key=store.target_key("prior_annotation", "r1", F))
    # D is real and was never annotated, the recall gap the curation exists to close.
    log.append("suggest", "r1", hpo_code=D, hpo_name="Term D", trigger_word="word", segment_idx=0)
    # And two that must stay out of the denominator: one the curator could not settle, and one
    # that is a relative's finding, not the patient's. Both are properly located, so the
    # reason they are dropped is unambiguously the qualifier.
    log.append("suggest", "r1", hpo_code="HP:0000924", hpo_name="Unsure term",
               trigger_word="boy", segment_idx=0, labels=["unsure_report"])
    log.append("suggest", "r1", hpo_code="HP:0011842", hpo_name="A relative's finding",
               trigger_word="odd", segment_idx=0, labels=["family"])
    return log


@pytest.fixture
def original_gold_csv(tmp_path):
    """The untouched HCY ground truth, in ``HCYDataset``'s two-column shape.

    ``r3`` carries an EMPTY code list, which is the shape that counts: the real file lists 18
    reports that way, and an empty row is that file asserting the report has no phenotype, not omitting it. ``r4`` is omitted entirely, which is the other thing entirely.
    """
    path = tmp_path / "hcy_ground_truth_raw.csv"
    _write_csv(path, ["patient_id", "hpo_codes"],
               [["r1", ";".join([C, F])], ["r2", D], ["r3", ""]])
    return path


@pytest.fixture
def results_dir(tmp_path):
    """The miniature earlier output tree, restricted to the two methods this test reads."""
    from fixtures.exp13_output import build_exp13_tree

    return build_exp13_tree(tmp_path / "output",
                            cohorts=("hcy",),
                            methods=("exp13_00_tree_gate_lr", "baseline_phenobert"))


def _run(overrides: dict, output_dir: Path, monkeypatch) -> Path:
    """Invoke ``run.py``'s Hydra entry point with an in-memory config."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (output_dir / "mlruns").as_uri())
    monkeypatch.syspath_prepend(str(_EXP_DIR))
    monkeypatch.syspath_prepend(str(_REPO))
    for name in ("discovery", "loaders", "segments", "sections", "costs", "report",
                 "curated_gold", "dataset", "run"):
        sys.modules.pop(name, None)

    import run as run_module

    with initialize_config_dir(version_base=None, config_dir=str(_REPO / "configs" / "experiments" / "03_setup")):
        cfg = compose(config_name="ground_truth",
                      overrides=[f"output_dir={output_dir}"]
                      + [f"{k}={v}" for k, v in overrides.items()])
    OmegaConf.set_struct(cfg, False)
    run_module.main.__wrapped__(cfg)
    for name in ("curated_gold", "dataset", "run"):
        sys.modules.pop(name, None)
    return output_dir / "hcy_ground_truth"


@pytest.fixture
def full_run(tmp_path, hcy_dir, curated_log, original_gold_csv, results_dir, monkeypatch):
    return _run(
        {"stages": "[variants,score,report]",
         "results_dir": str(results_dir),
         "hcy_dir": str(hcy_dir),
         "hcy_gt_path": str(original_gold_csv)},
        tmp_path / "out", monkeypatch,
    )


def _rows(out: Path, name: str) -> list:
    with open(out / "tables" / f"{name}.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ── the artifacts ────────────────────────────────────────────────────────────

def test_every_table_lands_on_disk(full_run):
    """One `.csv` and one `.tex` per table, so the thesis can `\\input` any of them."""
    for name in ("cohort_manifest", "gold_terms", "drop_reasons", "anchor_how", "gold_diff",
                 "gold_variants", "core_quality", "core_quality_headline",
                 "curated_vs_original"):
        assert (full_run / "tables" / f"{name}.csv").is_file(), f"missing {name}.csv"
        assert (full_run / "tables" / f"{name}.tex").is_file(), f"missing {name}.tex"
    assert (full_run / "results.md").is_file()
    assert (full_run / "availability.md").is_file()
    assert (full_run / "config_resolved.yaml").is_file()


def test_the_gold_export_is_a_drop_in_for_hcy_gt_path(full_run):
    """The whole point of the two-column export: it feeds any scorer in the repo unchanged."""
    from hpo_extraction.evaluation.datasets.hcy import HCYDataset

    path = full_run / "hcy_ground_truth_curated_eval.csv"
    assert path.is_file()
    gold = HCYDataset(str(path), "").load_ground_truth()
    assert {str(k): {str(c).strip() for c in v if str(c).strip()} for k, v in gold.items()} == \
        {"r1": {C, D}, "r2": {D}, "r3": set()}


# ── the cohort restriction ───────────────────────────────────────────────────

def test_the_report_nobody_asserted_anything_about_is_in_neither_gold(full_run):
    """`r4` carries no annotation from any source and no row in the original ground truth.

    It is still *listed*, the segmentation enumerates the cohort, so a report nobody annotated is
    a visible exclusion, not a row that never existed.
    """
    manifest = {r["patient_id"]: r for r in _rows(full_run, "cohort_manifest")}
    assert manifest["r4"]["in_cohort"] == "False"
    assert manifest["r4"]["reason"] == "no annotation from any source"
    assert "r4" not in (full_run / "hcy_ground_truth_curated_eval.csv").read_text(encoding="utf-8")
    assert not any(r["patient_id"] == "r4" for r in _rows(full_run, "gold_diff"))


def test_a_report_the_original_gold_calls_empty_is_scored_as_empty(full_run):
    """`r3`'s counterpart to the test above, and the reason the two are not the same case.

    Nobody annotated `r3` either, but the original ground truth lists it with an empty code list, that
    file has looked and found nothing. Admitting it costs a false positive for every term any
    method predicts there, which is the measurement, and it keeps the curated cohort a superset of
    the original one so `curated - original` stays a gold-side difference.
    """
    manifest = {r["patient_id"]: r for r in _rows(full_run, "cohort_manifest")}
    assert manifest["r3"]["in_cohort"] == "True"
    assert manifest["r3"]["reason"] == "original_gold"
    assert manifest["r3"]["n_gold_terms"] == "0"
    assert '"r3",""' in (full_run / "hcy_ground_truth_curated_eval.csv").read_text(encoding="utf-8")
    # Empty on both sides, so it contributes no row to the ground truth diff, a report, not a term.
    assert not any(r["patient_id"] == "r3" for r in _rows(full_run, "gold_diff"))


def test_both_ground_truth_sets_are_scored_over_the_same_reports(full_run):
    """The invariant the delta column depends on.

    If `curated` and `original` covered different report lists, the difference between them would
    be part ground truth and part denominator, and no reader could separate the two.
    """
    rows = [r for r in _rows(full_run, "core_quality") if r["status"] == "ok"]
    assert {r["cohort"] for r in rows} == {"curated", "original"}
    per_cohort = {}
    for row in rows:
        per_cohort.setdefault((row["method"], row["operating_point"]), {})[row["cohort"]] = \
            row["n_reports"]
    for key, seen in per_cohort.items():
        # Two, not three: `r3` is in the cohort but the prediction fixture covers only r1 and r2,
        # so it drops out of this method's row. That it drops out of BOTH rows is the invariant,
        # a report missing from one side and present on the other would put a denominator
        # difference inside the delta column.
        assert seen["curated"] == seen["original"] == "2", f"{key}: {seen}"


# ── the numbers ──────────────────────────────────────────────────────────────

def test_the_curated_gold_turns_the_hand_computed_error_into_a_hit(full_run):
    """The tree method at tau_0.1 predicts {C, D} / {D}, which is 2/3 against the original ground truth
    and perfect against the curated one. Both numbers are hand-computed in this module's docstring.
    """
    rows = {(r["cohort"], r["operating_point"]): r
            for r in _rows(full_run, "core_quality")
            if r["method"] == "tree_gate_lr" and r["status"] == "ok"}
    original = rows[("original", "tau_0.1")]
    curated = rows[("curated", "tau_0.1")]

    assert (original["tp"], original["fp"], original["fn"]) == ("2", "1", "1")
    assert float(original["micro_f1"]) == pytest.approx(2 / 3)
    assert (curated["tp"], curated["fp"], curated["fn"]) == ("3", "0", "0")
    assert float(curated["micro_f1"]) == pytest.approx(1.0)


def test_the_delta_table_prices_the_curation(full_run):
    """One row per method, main against main, with the configuration each side chose."""
    delta = {r["method"]: r for r in _rows(full_run, "curated_vs_original")}
    row = delta["tree_gate_lr"]
    assert float(row["micro_f1_original"]) == pytest.approx(2 / 3)
    assert float(row["micro_f1_curated"]) == pytest.approx(1.0)
    assert float(row["delta_micro_f1"]) == pytest.approx(1 / 3)
    assert row["delta_tp"] == "1" and row["delta_fp"] == "-1" and row["delta_fn"] == "-1"


def test_the_drop_reasons_account_for_every_excluded_term(full_run):
    """The three exclusions this scenario exercises, each named, not merely counted."""
    reasons = {r["reason"]: int(r["n_terms"]) for r in _rows(full_run, "drop_reasons")}
    assert reasons["status:removed"] == 1
    assert reasons["label:unsure_report"] == 1
    assert reasons["label:family"] == 1


def test_the_gold_diff_lists_the_change_rather_than_counting_it(full_run):
    """Every gold-side change is one auditable row: which term, which report, which direction."""
    changes = {(r["patient_id"], r["hpo_code"]): r["change"] for r in _rows(full_run, "gold_diff")}
    assert changes[("r1", D)] == "added"
    assert changes[("r1", F)] == "removed"
    assert changes[("r1", C)] == "unchanged"


def test_the_variants_table_prices_each_policy_decision(full_run):
    """Six ground truth definitions over one cohort, each one change from `default`.

    The two label variants are the point: `keep_family` isolates the single editorial call, and
    `keep_all_labels` adds the doubt exclusions on top, so the table says what each is worth
    separately, not as one lump.
    """
    variants = {r["variant"]: r for r in _rows(full_run, "gold_variants")}
    assert set(variants) == {"default", "keep_unanchored", "daphne_first", "keep_family",
                             "keep_all_labels", "approved_only"}
    assert all(int(v["n_reports"]) == 3 for v in variants.values())
    base = int(variants["default"]["n_gold_pairs"])
    assert int(variants["keep_family"]["n_gold_pairs"]) == base + 1
    assert int(variants["keep_all_labels"]["n_gold_pairs"]) == base + 2
    # Every annotation in this fixture evidence locations, so relaxing the evidence rule recovers nothing,
    # which is what makes the other rows attributable to the policy, not to the placement.
    assert int(variants["keep_unanchored"]["n_gold_pairs"]) == base
    # approved_only admits nothing here: the only verdict in the log is a removal.
    assert int(variants["approved_only"]["n_gold_pairs"]) == 0


def test_metrics_json_carries_the_cohort_size_beside_the_numbers(full_run):
    """The provenance every figure from this experiment has to be quoted with.

    The cohort grows as curation continues, so two runs are comparable only when these agree.
    """
    metrics = json.loads((full_run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["gold"]["n_reports_in_cohort"] == 3
    assert metrics["gold"]["n_gold_pairs"] == 3
    assert metrics["gold"]["n_original_pairs_in_cohort"] == 3
    assert metrics["gold"]["n_terms_added"] == 1
    assert metrics["gold"]["n_terms_removed"] == 1
    assert "n_reports_in_cohort" in (full_run / "results.md").read_text(encoding="utf-8") or \
        "reports qualify" in (full_run / "results.md").read_text(encoding="utf-8")


# ── degradation ──────────────────────────────────────────────────────────────

def test_the_hierarchy_stage_runs_and_reports_both_ground_truth_sets(tmp_path, hcy_dir, curated_log,
                                                             original_gold_csv, results_dir,
                                                             monkeypatch):
    """The cluster script's default stage list includes `hierarchy`, so it has to compose.

    Split out from `full_run` because it loads an ontology. The toy graph keeps it fast. hF is
    checked against flat F1, not a literal, the point is that both ground-truth sets reach the
    table with a real number, not what the toy DAG's closure happens to be.
    """
    from fixtures.toy_ontology import build_toy_tree

    hpo_json = tmp_path / "hpo.json"
    hpo_json.write_text(json.dumps(build_toy_tree().data))
    out = _run({"stages": "[score,hierarchy,report]", "results_dir": str(results_dir),
                "hcy_dir": str(hcy_dir), "hcy_gt_path": str(original_gold_csv),
                "hpo_json_path": str(hpo_json)},
               tmp_path / "hier", monkeypatch)

    rows = [r for r in _rows(out, "hierarchy") if r["method"] == "tree_gate_lr"]
    assert {r["cohort"] for r in rows} == {"curated", "original"}
    assert all(float(r["micro_hf"]) >= float(r["micro_f1_flat"]) for r in rows), \
        "closure hF is systematically optimistic; it can never fall below flat F1"
    # `variants` was not requested, so its table must not appear.
    assert not (out / "tables" / "gold_variants.csv").exists()


def test_the_gold_stage_runs_without_results_dir(tmp_path, hcy_dir, curated_log,
                                                 original_gold_csv, monkeypatch):
    """The seconds-long pre-flight: build the ground truth and read the manifest before scoring anything."""
    out = _run({"stages": "[variants]", "hcy_dir": str(hcy_dir),
                "hcy_gt_path": str(original_gold_csv)},
               tmp_path / "preflight", monkeypatch)
    assert (out / "hcy_ground_truth_curated_eval.csv").is_file()
    assert (out / "tables" / "cohort_manifest.csv").is_file()
    assert not (out / "tables" / "core_quality.csv").exists()


def test_the_shipped_criteria_refuse_to_run_without_the_original_gold(tmp_path, hcy_dir,
                                                                     curated_log, monkeypatch):
    """`original_gold` is in the shipped `criteria`, so `hcy_gt_path` is no longer optional.

    Failing open would produce a cohort quietly missing the reports the criterion exists to admit,
    and the only symptom would be a report count nobody has a reason to double-check.
    """
    with pytest.raises(ValueError, match="original_gold"):
        _run({"stages": "[dataset]", "hcy_dir": str(hcy_dir)}, tmp_path / "nogold", monkeypatch)


def test_the_dataset_stage_writes_back_into_the_cohort_directory(tmp_path, hcy_dir, curated_log,
                                                                 original_gold_csv, monkeypatch):
    """The deliverable half of this experiment: a dated dataset beside the files it came from.

    It is written into ``hcy_dir``, not into the run directory on purpose, the run
    directory is one experiment's scratch, and the thing a second person picks up a year later has
    to live with the cohort. The manifest is copied into the run directory so the run is still
    self-describing without duplicating patient data into it.
    """
    out = _run({"stages": "[dataset]", "hcy_dir": str(hcy_dir), "dataset_date": "2026-09-08",
                "hcy_gt_path": str(original_gold_csv)},
               tmp_path / "ship", monkeypatch)

    shipped = hcy_dir / "curated_ground_truth_2026-09-08"
    assert (shipped / "hcy_ground_truth_curated.csv").is_file()
    assert (shipped / "hcy_curated_annotations.csv").is_file()
    assert "Patient data" in (shipped / "README.md").read_text(encoding="utf-8")

    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["path"] == str(shipped)
    assert manifest["counts"]["n_gold_pairs"] == 3
    # No report text leaves the cohort directory: the run directory gets counts and paths only.
    assert "seizures" not in (out / "dataset_manifest.json").read_text(encoding="utf-8")


def test_no_report_sentence_reaches_the_run_directory(full_run):
    """The run directory carries codes, ids and trigger words, never prose.

    ``Tables.add`` writes every field of every row to the CSV whatever the printed column list
    says, so a ``segment_text`` on ``TermRow`` would put patient report sentences into
    ``output/``. The sentence belongs in the dataset written into ``hcy_dir``, under the curation
    log's group-only modes. The boundary is a filter in ``run.py`` and this is what holds it.
    """
    sentence = _REPORTS["r1"]
    for table in ("gold_terms", "cohort_manifest", "drop_reasons", "anchor_how", "gold_diff"):
        text = (full_run / "tables" / f"{table}.csv").read_text(encoding="utf-8")
        assert sentence not in text, f"{table}.csv carries a report sentence"
    assert sentence not in (full_run / "results.md").read_text(encoding="utf-8")


def test_an_empty_cohort_fails_loudly(tmp_path, hcy_dir, original_gold_csv, monkeypatch):
    """No curation log and `criteria=[suggestion]` leaves nothing to score.

    Writing an empty ground truth file and a table of zeros would look like a finished run that found
    nothing, which is the one failure mode this experiment must not have.
    """
    with pytest.raises(RuntimeError, match="no report qualifies"):
        _run({"stages": "[variants]", "hcy_dir": str(hcy_dir),
              "criteria": "[suggestion]", "hcy_gt_path": str(original_gold_csv)},
             tmp_path / "empty", monkeypatch)


def test_a_missing_annotation_directory_names_the_files_it_wanted(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="annotations_confirmed.csv"):
        _run({"stages": "[variants]", "hcy_dir": str(tmp_path / "nowhere")},
             tmp_path / "out", monkeypatch)
