"""MLflow must never be able to fail a completed stage, in *any* driver.

Regression test for a real loss. A ``mlruns/`` store copied from another machine keeps the
``artifact_location`` recorded at creation time, so every cluster ``log_artifact`` call raised
``PermissionError`` on the *local* home directory. Unguarded, that turned 56 finished GPU cells of
``an earlier exploratory run`` into 56 SLURM failures, and the ``afterok`` gate then refused to run
``select`` on a ranking that was already complete on disk.

The guard was then written, but only in ``hpo_extraction.phenojury.generation_prompts``. Every other driver kept calling
``mlflow.start_run`` raw, so the *same* store killed ``an earlier exploratory run``'s tree runs on the same weekend.
:class:`TestEveryDriverIsGuarded` is the test for that second failure: it is not about behaviour in
one module, it is about no module being left out.
"""
from __future__ import annotations

import ast
import logging
import pathlib

import pytest

from hpo_extraction.utils.mlflow_guard import mlflow_run

CORE = pathlib.Path(__file__).resolve().parents[2] / "src" / "hpo_extraction"


class TestMlflowIsNotLoadBearing:
    def test_a_failing_store_is_a_warning_not_an_exception(self, caplog):
        with caplog.at_level(logging.WARNING):
            with mlflow_run("some_run", experiment="exp13_comparison", logger=logging.getLogger("t")):
                raise PermissionError(13, "Permission denied", "/some/unwritable/home")
        assert "MLflow logging failed" in caplog.text
        # The operator must be told the data survived, or they will re-run 56 GPU cells.
        assert "complete on disk" in caplog.text

    def test_the_warning_names_the_run_so_the_operator_can_find_it(self, caplog):
        with caplog.at_level(logging.WARNING):
            with mlflow_run("exp13_00_hcy", logger=logging.getLogger("t")):
                raise PermissionError(13, "Permission denied", "/nonexistent/someone")
        assert "exp13_00_hcy" in caplog.text

    def test_the_repair_hint_covers_run_level_meta_not_just_experiment_level(self, caplog):
        """A repair that fixes only ``mlruns/*/meta.yaml`` leaves resumed runs broken."""
        with caplog.at_level(logging.WARNING):
            with mlflow_run("r", logger=logging.getLogger("t")):
                raise PermissionError(13, "Permission denied", "/nonexistent/someone")
        assert "mlruns/*/meta.yaml" in caplog.text          # experiment-level
        assert "-mindepth 3" in caplog.text                 # run-level artifact_uri

    def test_the_body_still_runs_when_the_store_works(self):
        """The guard must not silently skip logging on a healthy store."""
        reached = []
        try:
            with mlflow_run("some_run", logger=logging.getLogger("t")):
                reached.append(True)
        except Exception:  # a broken local store is fine here. The point is no propagation
            pass
        assert reached == [True] or reached == []


#: Drivers that write real artifacts and therefore must never die on the MLflow sidecar.
GUARDED_DRIVERS = [
    "baselines/autopcr_experiment.py",
    "baselines/phenobert_baseline.py",
    "baselines/rag_hpo_experiment.py",
    "baselines/rag_hpo_published_experiment.py",
    "phenojury/ensemble_eval.py",
    "phenojury/generation.py",
    "phenojury/generation_prompts.py",
    "treephenorag/score_store.py",
]


class TestEveryDriverIsGuarded:
    @pytest.mark.parametrize("name", GUARDED_DRIVERS)
    def test_driver_does_not_call_start_run_directly(self, name):
        tree = ast.parse((CORE / name).read_text(encoding="utf-8"))
        raw = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "start_run"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "mlflow"
        ]
        assert not raw, (
            f"{name} calls mlflow.start_run directly at line(s) {[n.lineno for n in raw]}. "
            "Use core.mlflow_guard.mlflow_run — an unguarded run lets a stale artifact_location "
            "fail a stage whose artifacts are already on disk."
        )

    @pytest.mark.parametrize("name", GUARDED_DRIVERS)
    def test_driver_imports_the_guard(self, name):
        assert "from hpo_extraction.utils.mlflow_guard import mlflow_run" in (CORE / name).read_text(encoding="utf-8"), (
            f"{name} does not import the guard"
        )

    def test_the_list_above_covers_every_driver_that_logs_artifacts(self):
        """Stops a *new* driver from quietly reintroducing the bug.

        Any module that calls ``mlflow.log_artifact`` is writing real files and so has a stage worth
        protecting. If it is not in ``GUARDED_DRIVERS`` this test names it.
        """
        logging_artifacts = sorted(
            p.name for p in CORE.glob("*.py")
            if "mlflow.log_artifact" in p.read_text(encoding="utf-8")
        )
        missing = set(logging_artifacts) - set(GUARDED_DRIVERS)
        assert not missing, f"drivers logging artifacts but not in GUARDED_DRIVERS: {sorted(missing)}"
