"""Emission and cost extraction.

The property that counts for the thesis is that the CSV, the LaTeX and the markdown are three
renderings of one object, a number cannot be right in one and wrong in another. The other is
that missing and zero stay distinguishable all the way to the page.
"""

from __future__ import annotations

import csv

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def rows():
    return [
        {"method": "tree_gate_lr", "label": "TreePhenoRAG (LR gate)", "cohort": "hcy",
         "operating_point": "tau_0.1", "micro_f1": 0.6666666666, "peak_gpu_gb": None,
         "status": "ok", "is_headline": False},
        {"method": "tree_gate_lr", "label": "TreePhenoRAG (LR gate)", "cohort": "hcy",
         "operating_point": "tau_0.5", "micro_f1": 0.8, "peak_gpu_gb": 14.25,
         "status": "ok", "is_headline": True},
    ]


# ── Missing is not zero ──────────────────────────────────────────────────────

def test_missing_renders_as_na_never_as_zero(exp13_modules):
    """A metric that could not be computed and a metric that measured zero are different facts,
    and a results table that conflates them is worse than one with a gap."""
    fmt = exp13_modules.report.fmt
    assert fmt(None) == "n/a"
    assert fmt(float("nan")) == "n/a"
    assert fmt(0.0) == "0.000"
    assert fmt(0) == "0"


def test_integers_keep_their_exact_value(exp13_modules):
    """Counts must not acquire a decimal point, `tp` of 2 is 2, not 2.000."""
    assert exp13_modules.report.fmt(42) == "42"


def test_very_small_numbers_use_scientific_notation(exp13_modules):
    """An ECE of 1e-5 must not print as 0.000, which reads as perfectly calibrated."""
    assert "e-" in exp13_modules.report.fmt(0.0000123)


def test_booleans_read_as_words(exp13_modules):
    assert exp13_modules.report.fmt(True) == "yes"
    assert exp13_modules.report.fmt(False) == "no"


# ── The three renderings agree ───────────────────────────────────────────────

def test_csv_keeps_full_precision_while_latex_rounds(exp13_modules, rows, tmp_path):
    """The CSV is the auditable artifact. The LaTeX is the readable one."""
    exp13_modules.report.write_csv(rows, tmp_path / "t.csv", ["micro_f1"])
    written = list(csv.DictReader((tmp_path / "t.csv").open()))
    assert written[0]["micro_f1"] == "0.6666666666"

    exp13_modules.report.write_latex(rows, tmp_path / "t.tex", "cap", "tab:x", ["micro_f1"])
    assert "0.667" in (tmp_path / "t.tex").read_text()


def test_the_main_result_star_and_footnote_travel_together(exp13_modules, rows, tmp_path):
    """A `*` without the oracle footnote would present a tuned number as a held-out one."""
    exp13_modules.report.write_latex(rows, tmp_path / "t.tex", "cap", "tab:x",
                                     ["operating_point", "micro_f1"])
    tex = (tmp_path / "t.tex").read_text()
    assert r"tau\_0.5*" in tex
    assert "oracle" in tex

    md = exp13_modules.report.markdown_table(rows, ["operating_point", "micro_f1"])
    assert "tau_0.5*" in md
    assert "oracle" in md


def test_no_footnote_when_no_row_is_a_main_result(exp13_modules, rows, tmp_path):
    plain = [dict(r, is_headline=False) for r in rows]
    exp13_modules.report.write_latex(plain, tmp_path / "t.tex", "cap", "tab:x", ["micro_f1"])
    assert "oracle" not in (tmp_path / "t.tex").read_text()


def test_latex_escapes_the_underscores_in_hpo_and_column_names(exp13_modules, rows, tmp_path):
    """`micro_f1` unescaped is a subscript, and the table silently renders wrong."""
    exp13_modules.report.write_latex(rows, tmp_path / "t.tex", "cap", "tab:x", ["micro_f1"])
    tex = (tmp_path / "t.tex").read_text()
    assert r"micro\ f1" in tex or r"micro f1" in tex
    assert "_" not in tex.split(r"\midrule")[0].split(r"\toprule")[1]


def test_markdown_escapes_pipes_so_a_table_cannot_break(exp13_modules):
    md = exp13_modules.report.markdown_table([{"a": "x|y"}], ["a"])
    assert r"x\|y" in md


# ── Placeholders survive to the page ─────────────────────────────────────────

def test_a_placeholder_row_keeps_its_reason_in_every_format(exp13_modules, tmp_path):
    rows = [{"method": "raghpo_70b", "cohort": "hcy", "status": "placeholder",
             "reason": "no run directory"}]
    tables = exp13_modules.report.Tables(tmp_path)
    tables.add("t", rows, "cap", ["status", "reason"])
    assert "no run directory" in (tmp_path / "tables" / "t.csv").read_text()
    assert "no run directory" in (tmp_path / "tables" / "t.tex").read_text()
    assert "no run directory" in tables.sections[0][1]


def test_an_empty_table_still_emits_with_an_explanation(exp13_modules, tmp_path):
    """A silently absent .tex file would break the thesis build with no clue why."""
    tables = exp13_modules.report.Tables(tmp_path)
    tables.add("empty", [], "cap")
    assert (tmp_path / "tables" / "empty.tex").is_file()
    assert "placeholder" in (tmp_path / "tables" / "empty.csv").read_text()


def test_tables_registers_both_artifacts_per_table(exp13_modules, rows, tmp_path):
    tables = exp13_modules.report.Tables(tmp_path)
    tables.add("t", rows, "cap", ["micro_f1"])
    assert tables.artifacts == ["tables/t.csv", "tables/t.tex"]


def test_the_csv_keeps_every_column_even_when_the_printed_table_is_curated(
        exp13_modules, rows, tmp_path):
    """The printed table shows one metric. The CSV is what a number gets re-checked against, so
    dropping the identifying columns there would leave a quoted value with no auditable record."""
    tables = exp13_modules.report.Tables(tmp_path)
    tables.add("t", rows, "cap", ["micro_f1"])

    written = list(csv.DictReader((tmp_path / "tables" / "t.csv").open()))
    assert {"method", "cohort", "operating_point", "status", "micro_f1"} <= set(written[0])
    # …while the LaTeX shows only what was asked for.
    tex = (tmp_path / "tables" / "t.tex").read_text()
    assert "tree_gate_lr" not in tex


# ── Availability report ──────────────────────────────────────────────────────

def test_availability_md_separates_missing_runs_from_inapplicable_metrics(
        exp13_modules, exp13_results_dir, tmp_path):
    """The two reasons a cell is blank are different facts: one is fixable by running a job,
    the other is a property of the method."""
    avail = exp13_modules.discovery.discover(exp13_results_dir)
    rows = exp13_modules.discovery.availability_rows(avail)
    support = exp13_modules.discovery.metric_support_rows()
    rerun = {(s.key, c): exp13_modules.discovery.rerun_command(s, c)
             for s in exp13_modules.discovery.METHODS for c in ("hcy", "gsc")}

    exp13_modules.report.write_availability_md(
        tmp_path / "availability.md", rows, support, rerun, avail.counts())
    text = (tmp_path / "availability.md").read_text()

    assert "How to fill the gaps" in text
    assert "sbatch cluster/run_hcy_baseline_raghpo_70b.sh" in text
    assert "Which metrics apply to which method" in text
    assert "phrase" in text                 # The RAG-HPO segment-retrieval reason


# ── MLflow flattening ────────────────────────────────────────────────────────

def test_only_main_result_numeric_fields_reach_mlflow(exp13_modules, rows):
    """MLflow should hold a summary, not a copy of every CSV."""
    flat = exp13_modules.report.flatten_metrics({"core": rows})
    assert flat == {"core.tree_gate_lr.hcy.micro_f1": 0.8,
                    "core.tree_gate_lr.hcy.peak_gpu_gb": 14.25}


def test_missing_values_are_not_logged_as_zero(exp13_modules):
    flat = exp13_modules.report.flatten_metrics({"t": [
        {"method": "m", "cohort": "hcy", "is_headline": True, "ece": None,
         "brier": float("nan")}]})
    assert flat == {}


# ── Cost extraction ──────────────────────────────────────────────────────────





















# ── The MLflow source ────────────────────────────────────────────────────────

@pytest.fixture
def mlflow_store(tmp_path):
    """A real local MLflow store holding one run with the metrics the earlier drivers logged.

    Written through mlflow itself rather than by hand: the on-disk layout is an implementation
    detail that has changed across versions, and a hand-built fixture would fix the wrong thing.
    """
    import mlflow

    store = tmp_path / "mlruns"
    mlflow.set_tracking_uri(store.resolve().as_uri())
    mlflow.set_experiment("exp13_treephenorag")
    with mlflow.start_run():
        mlflow.log_metric("peak_gpu_mem_bytes", 15.0 * 1024**3)
        mlflow.log_metric("mean_time_per_report_s", 42.0)
        mlflow.log_param("llama_dir_base", "/models/llama-8b")
    return store




@pytest.fixture
def sharded_mlflow_store(tmp_path):
    """Two cells' worth of runs in one experiment, one of them an 8-way array.

    Mirrors the real store: ``exp13_treephenorag`` holds every tree method and both cohorts, and
    each shard of an array job is its own run.
    """
    import mlflow

    store = tmp_path / "mlruns_sharded"
    mlflow.set_tracking_uri(store.resolve().as_uri())
    mlflow.set_experiment("exp13_treephenorag")

    # The cell we want: 4 shards, uneven report counts, differing peaks.
    for shard, (peak_gb, seconds, reports) in enumerate(
            [(10.0, 100.0, 10), (14.0, 200.0, 30), (12.0, 50.0, 5), (11.0, 50.0, 5)]):
        with mlflow.start_run(run_name=f"exp13_00_tree_gate_lr_hcy_s{shard}of4"):
            mlflow.log_metric("peak_gpu_mem_bytes", peak_gb * 1024**3)
            mlflow.log_metric("duration_seconds", seconds)
            mlflow.log_metric("n_reports", reports)

    # A different cell in the same experiment, with a much larger peak that must not leak.
    with mlflow.start_run(run_name="exp13_01_tree_noisyor_gsc"):
        mlflow.log_metric("peak_gpu_mem_bytes", 79.0 * 1024**3)
        mlflow.log_metric("duration_seconds", 999.0)
        mlflow.log_metric("n_reports", 1)
    return store


















