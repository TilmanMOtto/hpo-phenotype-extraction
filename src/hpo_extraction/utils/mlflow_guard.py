"""MLflow as reporting, never as a dependency.

Every experiment driver in ``src/core/`` writes its real artifacts to ``run_output_dir`` *before*
it opens an MLflow run: predictions, retrieved segments, detections, timing, rankings. MLflow is a
convenience index over those files. A store that cannot be written must therefore degrade to a
warning, never to a non-zero exit.

This is not defensive programming for its own sake. It is a bug that has already cost two runs.
An MLflow file store records an absolute ``artifact_location`` at experiment-creation time and an
absolute ``artifact_uri`` per run. Copy the store between machines and those paths still name the
*old* machine, so every ``log_artifact`` call on the new one tries to ``os.makedirs`` a directory it
has no right to and raises ``PermissionError`` on the old home directory.

Unguarded, that turned 56 finished GPU cells of ``an earlier exploratory run`` into 56 SLURM failures, the extractions and the ranking were complete on disk the whole time, and the ``afterok`` gate then
refused to run the ``select`` stage on them. ``experiments/findings/exp13_findings_old.md`` §15
records the same failure an experiment group earlier.

The whole block is guarded rather than each individual call: partial MLflow state is not worth
branching over, and the warning below tells the operator what is missing, what is not, and
how to repair the store.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import mlflow

# Printed verbatim in the warning. The substitution rewrites both the experiment-level
# `artifact_location` and the per-run `artifact_uri`, which is the part a repair usually forgets:
# resuming an existing run reads the run-level value and fails again on an otherwise-fixed store.
REPAIR_HINT = (
    "grep -rl 'file:///home' mlruns/*/meta.yaml | "
    "xargs sed -i \"s|file:///.*/mlruns|file://$PWD/mlruns|\" && "
    "find mlruns -mindepth 3 -maxdepth 3 -name meta.yaml | xargs grep -l 'file:///home' | "
    "xargs sed -i \"s|file:///.*/mlruns|file://$PWD/mlruns|\""
)


@contextmanager
def mlflow_run(run_name: str | None = None, *, experiment: str | None = None, logger=None):
    """An MLflow run whose failure cannot cost a completed stage.

    ``experiment`` is passed to :func:`mlflow.set_experiment` when given. Omit it to log into
    whatever experiment is already active. ``run_name`` is optional for the same reason, two
    callers predate the naming convention and let MLflow generate one.

    Any exception raised inside the block is swallowed with a warning. Callers must therefore put
    only logging inside it, which is the point: if something in the block is essential, it does
    not belong in the block.

    Yields the active run, or ``None`` when MLflow is unavailable, so a caller may either log
    unconditionally inside the block (most drivers) or guard with ``if run is not None`` (the earlier runs).
    Both are correct. The second also skips building the payload.

    **The generator must yield once on every path**, including when setup fails. Opening
    the run inside the ``try`` that also wraps the ``yield`` looks equivalent and is not: a store
    that raises in ``set_experiment``/``start_run`` never reaches the ``yield``, and
    ``contextlib`` then converts the swallowed warning into ``RuntimeError: generator didn't
    yield``, killing the stage this guard exists to protect, in the copied-store case
    it was written for.
    """
    log = logger or logging.getLogger(__name__)

    def _warn(exc: Exception) -> None:
        log.warning(
            "MLflow logging failed for %s (%s: %s). The run's own artifacts are complete on disk; "
            "only the MLflow index is missing. If this is a stale artifact_location from a copied "
            "mlruns/ store, repair it with: %s",
            run_name or "<unnamed run>", type(exc).__name__, exc, REPAIR_HINT,
        )

    active = None
    try:
        if experiment is not None:
            mlflow.set_experiment(experiment)
        active = mlflow.start_run(run_name=run_name)
    except Exception as exc:  # noqa: BLE001, reporting must not propagate
        _warn(exc)

    # One yield site, reached on both paths, so the body's exceptions are swallowed whether or not
    # The store came up. Two separate `yield`s in two branches would leave the failed-setup path
    # re-raising whatever the caller's block raised.
    try:
        if active is None:
            yield None
        else:
            with active:
                yield active
    except Exception as exc:  # noqa: BLE001, reporting must not propagate
        _warn(exc)
