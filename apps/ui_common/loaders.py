"""Raw artifact readers for a tree-pipeline run directory.

Pure IO. Nothing here caches, computes metrics, or knows about Dash, the caching
lives in :mod:`registry` and the metrics in :mod:`report`.

A run directory (``<output_dir>/expGG_SS_name/``) may contain:

    run.log
    {variant}.pkl                       response_dict {patient: {hpo: {i: {...}}}}
    {variant}.csv                       main micro/macro metrics (HCYDataset.evaluate)
    {variant}_predictions.jsonl         one line per (patient, target hpo)
    {variant}_retrieved_segments.jsonl  one line per (patient, hpo, rank)
    timing_records.json                 per-patient latency
    node_conf_by_patient.pkl            {patient: {hpo: confidence}}   (an earlier exploratory run only)
    tau_sweep.csv                       (tau_accept, tau_prune) grid    (an earlier exploratory run only)
    resolved_config.yaml                the run's own config            (an earlier exploratory run only)

``variant`` is ``base_LLM`` or ``fineTuned_LLM``. Not every run has every file. The
loaders return ``None`` rather than raising so capability detection can branch on it.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Any

VARIANTS = ("base_LLM", "fineTuned_LLM")

# Sentinels written into slm_verdict / response for nodes the traversal did not judge.
STATE_SENTINELS = {"PRUNED", "SKIPPED", "ASSUMED_YES"}


# ── discovery ────────────────────────────────────────────────────────────────

def detect_variant(run_dir: str) -> str | None:
    """The file prefix this run used, inferred from whichever artifact exists."""
    for variant in VARIANTS:
        for suffix in ("_predictions.jsonl", "_retrieved_segments.jsonl", ".pkl"):
            if os.path.isfile(os.path.join(run_dir, variant + suffix)):
                return variant
    # Fall back to any *_retrieved_segments.jsonl / *_LLM.pkl with a non-standard prefix.
    for pattern, suffix in (("*_retrieved_segments.jsonl", "_retrieved_segments.jsonl"),
                            ("*_predictions.jsonl", "_predictions.jsonl"),
                            ("*_LLM.pkl", ".pkl")):
        hits = sorted(glob.glob(os.path.join(run_dir, pattern)))
        if hits:
            return os.path.basename(hits[0])[: -len(suffix)]
    return None


def find_runs(output_base: str) -> list[dict]:
    """Every immediate subdirectory of ``output_base`` that looks like a run.

    Returns ``[{"run_id", "run_dir", "variant"}]`` sorted by run_id. A directory
    qualifies if it has a detectable variant, i.e. at least one recognisable artifact.
    """
    if not os.path.isdir(output_base):
        return []
    runs: list[dict] = []
    for name in sorted(os.listdir(output_base)):
        run_dir = os.path.join(output_base, name)
        if not os.path.isdir(run_dir):
            continue
        variant = detect_variant(run_dir)
        if variant is None:
            continue
        runs.append({"run_id": name, "run_dir": run_dir, "variant": variant})
    return runs


def artifacts_mtime(run_dir: str) -> float:
    """Newest mtime among the run's data artifacts, the cache key.

    Logs and models are excluded: ``run*.log`` (one per shard of a SLURM array) keeps
    ticking while a job runs, and ``.checkpoint_*.jsonl`` is progress bookkeeping, not analysed
    data. Either would invalidate the cache on every poll without the data having changed.
    """
    newest = 0.0
    for name in os.listdir(run_dir):
        if (name.startswith("run") and name.endswith(".log")) or name.startswith(".checkpoint_"):
            continue
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            newest = max(newest, os.path.getmtime(path))
    return newest


# ── generic readers ──────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _maybe(run_dir: str, filename: str) -> str | None:
    path = os.path.join(run_dir, filename)
    return path if os.path.isfile(path) else None


# ── per-artifact loaders ─────────────────────────────────────────────────────

def load_predictions(run_dir: str, variant: str) -> list[dict] | None:
    """``{variant}_predictions.jsonl``, (patient, target hpo) with prediction + GT."""
    path = _maybe(run_dir, f"{variant}_predictions.jsonl")
    return _read_jsonl(path) if path else None


def load_segments(run_dir: str, variant: str) -> list[dict] | None:
    """``{variant}_retrieved_segments.jsonl``, (patient, hpo, rank) with text + margin.

    Tree runs also emit rank-0 stub rows whose ``slm_verdict`` is a state sentinel
    (``PRUNED`` / ``SKIPPED`` / ``ASSUMED_YES``) and whose ``text`` is empty.
    """
    path = _maybe(run_dir, f"{variant}_retrieved_segments.jsonl")
    return _read_jsonl(path) if path else None


def load_response_dict(run_dir: str, variant: str) -> dict | None:
    """``{variant}.pkl``, ``{patient: {hpo: {i: {"response", "prompt", ...}}}}``.

    The inner dict's extra keys vary by experiment: ``confidence`` / ``max_margin``
    (an earlier exploratory run, an earlier exploratory run), ``logit_margin`` (an earlier exploratory run, an earlier exploratory run, an earlier exploratory run), or nothing
    beyond prompt+response (the earlier runs, earlier generation runs).
    """
    path = _maybe(run_dir, f"{variant}.pkl")
    if not path:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_main_result_metrics(run_dir: str, variant: str) -> dict | None:
    """``{variant}.csv``, the single row HCYDataset.evaluate wrote for this run."""
    path = _maybe(run_dir, f"{variant}.csv")
    if not path:
        return None
    import pandas as pd

    df = pd.read_csv(path)
    if df.empty:
        return None
    return {k: float(v) for k, v in df.iloc[0].to_dict().items()}


def load_timing(run_dir: str) -> list[dict] | None:
    """``timing_records.json``, per-patient latency and node counts."""
    path = _maybe(run_dir, "timing_records.json")
    if not path:
        return None
    with open(path) as f:
        records = json.load(f)
    return records if isinstance(records, list) else None


def load_node_conf(run_dir: str) -> dict | None:
    """``node_conf_by_patient.pkl``, the unpruned {patient: {hpo: confidence}} table."""
    path = _maybe(run_dir, "node_conf_by_patient.pkl")
    if not path:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tau_sweep(run_dir: str) -> list[dict] | None:
    """``tau_sweep.csv``, one row per (tau_accept, tau_prune) grid point."""
    path = _maybe(run_dir, "tau_sweep.csv")
    if not path:
        return None
    import pandas as pd

    return pd.read_csv(path).to_dict("records")


def load_resolved_config(run_dir: str) -> dict | None:
    """``resolved_config.yaml``, the config the run actually executed with."""
    path = _maybe(run_dir, "resolved_config.yaml")
    if not path:
        return None
    try:
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None


def load_patient_texts(input_dir: str | None) -> dict[str, str]:
    """Raw report text per patient, keyed by filename stem, for the patient view."""
    if not input_dir or not os.path.isdir(input_dir):
        return {}
    texts: dict[str, str] = {}
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, errors="replace") as f:
                texts[os.path.splitext(name)[0]] = f.read()
        except OSError:
            continue
    return texts


# ── bundle ───────────────────────────────────────────────────────────────────

def load_raw(run_dir: str, variant: str) -> dict[str, Any]:
    """Read every artifact a run has. The fat intermediate, never sent to the browser."""
    return {
        "run_dir": run_dir,
        "variant": variant,
        "predictions": load_predictions(run_dir, variant),
        "segments": load_segments(run_dir, variant),
        "response_dict": load_response_dict(run_dir, variant),
        "headline": load_main_result_metrics(run_dir, variant),
        "timing": load_timing(run_dir),
        "node_conf": load_node_conf(run_dir),
        "tau_sweep": load_tau_sweep(run_dir),
        "config": load_resolved_config(run_dir),
    }
