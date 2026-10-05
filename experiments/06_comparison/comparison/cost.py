"""What each system costs to run: seconds per report, model calls, peak GPU, and the setup.

## Why this is not a column in ``t1_overall.csv``

**A method's cost is incurred in a different experiment from the one that scores it.** PhenoJury is
scored from ``phenojury_protocol``'s out-of-fold re-run, which is a CPU job of a few minutes, but the thing
that cost anything is ``phenojury_generation_free_listing``, where eight SLMs generated over every sentence. TreePhenoRAG is
scored from ``treephenorag_protocol``, a pure CPU re-run, while its real cost is ``treephenorag_scores_terminfo`` building the score
cache on a GPU for days. Reading a runtime off the scoring experiment would report the cost of the
analysis rather than the cost of the method, and would make the two most expensive systems in the
roster look like the two cheapest.

So every row here names, in config, **which experiment supplies its cost**, and every number
carries the source it came from.

## What each source can and cannot say

``timing``   ``{exp}/{cohort}/{variant}_timing.jsonl``, written by the driver itself. The most
             direct source, and the only one that records model-load time and LLM call counts.
``runlog``   ``Done | N reports in Xs | … peak GPU Y GB``, the line the tree driver prints. Shards
             are summed for reports and seconds and **maxed** for peak GPU, because the shards ran
             concurrently on separate GPUs: a sum would describe a machine nobody used.
``extract``  the driver's ``Extract done | model=… in Xs | peak GPU Y GB`` lines. The fallback for
             the ensemble, and on this data the route that actually fires, see below. Same
             quantity as ``per-model``, read from the appended log instead of a rewritten file.
``per-model`` ``{exp}/{cohort}/slm_ensemble_timing_{model}.jsonl``, one file per juror, each
             timing that model end to end. **Summed**, not maxed: the jurors ran as an 8-way array
             on 8 devices, so no clock ever measured the ensemble, but the jury is a *set* of
             models and one accelerator running them one after another pays their sum. That is the
             only number a reader deciding whether to deploy it can use, it is additive over
             the members of a condition, and every term in it was measured.
``calls``    ``treephenorag_protocol/tables/core_quality.csv``, the verifier calls per report the protocol
             itself counted, for the one method whose call count is a selected quantity, not a property of the input.
``config``   hardware and model identity. These are facts about the job, not derivable from
             any artifact, so they are declared, not inferred.

Nothing is estimated. A quantity no source has comes back ``None`` and prints as ``n/a``.

## The scope trap, which this module exists to make visible

``baseline_phenobert``'s timing file reports **0.026 s per report** for PhenoBERT. That is not PhenoBERT's
inference time: its ``run.log`` says *"Reusing complete PhenoBERT output"*, so the CNN had already
been run and the file times only the linking and annotation pass over its output. Quoting it would
make PhenoBERT look some six hundred times faster than it is, in the one table whose whole purpose
is a cost comparison.

Hence ``scope``: a required, config-declared string on every row saying what the number covers. A
row whose scope says the expensive stage is excluded is reported with its number **and** that
sentence, never as the method's runtime.

The ensemble has the same trap one level down, and it is the reason the summed juror clocks are
guarded, not simply added. ``slm_ensemble_timing_{model}.jsonl`` is rewritten from scratch
on every submission, and generation is modeled per report, so a juror that resumed after a
12 h wall-clock kill has a file timing only the reports it had left. Summing those would publish a
jury runtime short by hours while looking entirely measured. The driver says which models resumed,
in prose, in its run log; ``resumed_models`` reads it, and a row with any such juror falls through
to the ``extract`` source, not publishing a short sum.

On the shipped the Free Listing generation run artifacts that fallback is not hypothetical: on **both** cohorts the last
submission re-ran all eight jurors with every generation already cached, so every timing file
reports ``extraction_s`` of a few hundredths of a second against 2 713 sentences and times PhenoBERT
alone. The generating passes are still in the run log, which is appended, not rewritten, and
``from_extract_logs`` recovers them, telling generation from re-grounding by the peak GPU, because
a pass with the model resident peaks at 8--21 GB and one that only re-grounds peaks at 0.01 GB.
Two jurors (``deepseek``, ``intelligent_internet`` on HCY; ``intelligent_internet`` on GSC+) hit the
12 h wall clock and were resubmitted, and the incomplete path writes no ``Extract done`` line at
all, so their generation is spread over submissions no single clock covers. Conditions containing them are
marked ``seconds_is_lower_bound`` and printed with a ``≥``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass

from pathlib import Path

logger = logging.getLogger(__name__)

#: ``Done | 3 reports in 90237.2s | 18354 unique nodes | peak GPU 10.36 GB``
_DONE_RE = re.compile(
    r"Done\s*\|\s*(?P<reports>\d+)\s+reports?\s+in\s+(?P<seconds>[\d.]+)\s*s", re.IGNORECASE)
#: ``Done | 116 reports | 1480 predicted terms | 5123.4s (44.17s/report) | peak 71.20 GB``, the
#: RAG-HPO drivers' line. Runs that predate ``rag_hpo_timing.jsonl`` have only this, and
#: ``_DONE_RE`` never matched it, so every RAG-HPO cost cell printed ``n/a``.
_DONE_RAGHPO_RE = re.compile(
    r"Done\s*\|\s*(?P<reports>\d+)\s+reports?\s*\|[^|\n]*\|\s*(?P<seconds>[\d.]+)\s*s\b",
    re.IGNORECASE)
#: ``peak GPU 10.50 GB`` (tree, ensemble) or ``peak 71.20 GB`` (RAG-HPO).
_GPU_RE = re.compile(r"peak(?: GPU)?\s+(?P<gb>[\d.]+)\s*GB", re.IGNORECASE)


@dataclass
class CostRow:
    """One (method, cohort) cost cell. Every numeric field may be ``None``, see the docstring."""

    method: str = ""
    label: str = ""
    cohort: str = ""
    cost_experiment: str = ""
    seconds_per_report: float | None = None
    llm_calls_per_report: float | None = None
    peak_gpu_gb: float | None = None
    model_load_s: float | None = None
    n_reports: int | None = None
    accelerator: str = ""
    checkpoint: str = ""
    scope: str = ""
    #: True only when the scope was narrowed by detection, i.e. The expensive stage did not run.
    #: A boolean, because the table used to infer this by looking for "only" in the scope prose and
    #: flagged every row whose scope happened to say "CPU-only".
    scope_partial: bool = False
    #: True when `seconds_per_report` sums fewer jurors than the condition has, because one has no
    #: complete generating pass on record. The number is then a floor, and the table prints it as
    #: one, a cell that cannot be mistaken for the whole jury is worth more than a blank.
    seconds_is_lower_bound: bool = False
    #: True when part of `seconds_per_report` was apportioned, not timed per document --
    #: RAG-HPO's LLM-mapping stage on a GSC+ subset (see `apply_subset_timing`).
    seconds_is_approximate: bool = False
    #: The tree only: verifier calls per report of the Re-run configuration (the TreePhenoRAG protocol's pooled
    #: out-of-fold traversal). Not the row's cost -- `llm_calls_per_report` is the generating run's,
    #: beside the generating run's seconds -- but the number a deployment at that configuration
    #: would pay, so the table carries it in a footnote.
    operating_point_calls_per_report: float | None = None
    source: str = ""
    note: str = ""


# ── sources ──────────────────────────────────────────────────────────────────

def from_timing(path: Path) -> dict:
    """The driver's own timing record.

    The file is one JSON object per stage. Fields are summed across stages where summing is
    meaningful (durations, call counts) and taken from the last stage where it is not (n_reports),
    because a multi-stage driver writes the same cohort size on every line.
    """
    out: dict = {}
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return out
    total_s = sum(float(r.get("duration_s") or 0.0) for r in rows)
    load_s = sum(float(r.get("model_load_s") or 0.0) for r in rows)
    calls = sum(float(r.get("n_llm_calls") or 0.0) for r in rows)
    n_reports = next((int(r["n_reports"]) for r in reversed(rows) if r.get("n_reports")), None)
    # Maxed, not summed: stages run one after another on the same device.
    peaks = [float(r["peak_gpu_gb"]) for r in rows if r.get("peak_gpu_gb") is not None]
    out["n_reports"] = n_reports
    if peaks:
        out["peak_gpu_gb"] = max(peaks)
    if total_s and n_reports:
        out["seconds_per_report"] = total_s / n_reports
    if load_s:
        out["model_load_s"] = load_s
    if calls and n_reports:
        out["llm_calls_per_report"] = calls / n_reports
    return out


def from_run_logs(paths: list) -> dict:
    """The ``Done | …`` line, summed over shards.

    Reports and seconds add: the shards partition the cohort and each one's seconds are its own
    work. **Peak GPU does not add**, the shards are separate array tasks on separate GPUs, so the
    peak any single device reached is the maximum, and summing would describe hardware that was
    never used.
    """
    reports = seconds = 0
    peak = None
    seen = 0
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        done = None
        for done in _DONE_RE.finditer(text):
            pass                      # The LAST Done line. A resumed shard prints several
        if done is None:
            for done in _DONE_RAGHPO_RE.finditer(text):
                pass
        if done:
            seen += 1
            reports += int(done.group("reports"))
            seconds += float(done.group("seconds"))
        gpus = [float(m.group("gb")) for m in _GPU_RE.finditer(text)]
        if gpus:
            peak = max(gpus) if peak is None else max(peak, max(gpus))
    out: dict = {}
    if seen and reports:
        out["n_reports"] = reports
        out["seconds_per_report"] = seconds / reports
        out["n_shards"] = seen
    if peak is not None:
        out["peak_gpu_gb"] = peak
    return out


def from_protocol_calls(path: Path, config_prefix: str = "V3") -> dict:
    """``calls_per_report`` from the TreePhenoRAG protocol's own quality table.

    The traversal's call count is a *selected* quantity, the protocol chose the threshold that
    produced it, so it is read from the row that reports the selected configuration, not
    counted from an artifact.
    """
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("config", "").startswith(config_prefix) and row.get("calls_per_report"):
                return {"llm_calls_per_report": float(row["calls_per_report"])}
    return {}


def from_cache_calls(path: Path, cohort: str, index: str) -> dict:
    """The generating run's own call count: the TreePhenoRAG protocol's ``cache_calls.csv`` (rows of the cache).

    The tree's seconds come from the cache build's run logs, so its calls must come from the same
    run. The re-run configuration's calls are a different run's number (see CostRow).
    """
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("cohort") == cohort and row.get("retrieval_index") == index:
                return {"llm_calls_per_report": float(row["calls_per_report"])}
    return {}


def count_generation_calls(directory: Path, pattern, n_reports: int | None) -> dict:
    """One generation per line, over every per-model dump matching *pattern*.

    This is the ensemble's call count and there is no other record of it: the driver logs no total.
    Counting lines is a measurement of the artifact, not an estimate, which is the only
    reason it is allowed here.

    *pattern* may be a **list** of patterns, which is how a jury smaller than the whole pool gets a
    real cost, not a fraction of one. `glob` has no alternation, so naming the three members
    of the cost-constrained jury takes three patterns. Multiplying the eight-juror figure by 3/8
    would be an estimate, and the whole claim about that condition is what it costs.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = sorted({p for pat in patterns for p in directory.glob(str(pat))})
    if not files or not n_reports:
        return {}
    total = 0
    for path in files:
        with open(path, "rb") as fh:
            total += sum(1 for line in fh if line.strip())
    # The condition's membership, read off the files, not declared a second time: whichever dumps
    # were counted ARE the jurors this condition's cost must sum over.
    models = sorted(p.stem[len("llm_extractions_"):] for p in files)
    return {"llm_calls_per_report": total / n_reports, "n_generation_files": len(files),
            "models": models}


def resumed_models(paths: list) -> set:
    """Model keys whose generation pass was **continued from cache**, read off the driver's log.

    This counts because ``slm_ensemble_timing_{model}.jsonl`` is rewritten from scratch on every
    submission and its ``duration_s`` starts at that submission's ``t0``. A model that resumed after
    a 12 h wall-clock kill has a timing file covering only the reports left to do, and summing those
    durations would report a jury runtime that is short by hours while looking entirely measured.
    The generation dump has no field that distinguishes the two, but the driver says so in prose, so
    that is where it is read from.
    """
    seen: set = set()
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in _RESUMED_RE:
            seen.update(m.group("model") for m in pattern.finditer(text))
    return seen


def from_per_model_timing(directory: Path, pattern, skip: set | None = None) -> dict:
    """Seconds per report for a jury, **summed** over its members: the sequential runtime.

    Each ``slm_ensemble_timing_{model}.jsonl`` times one juror end to end from its own ``t0`` --
    model load, generation over every sentence of the cohort, and the PhenoBERT grounding of
    what it wrote. The jurors ran as an 8-way array on 8 devices, so no single clock ever measured
    the ensemble. But the jury is a *set* of models, and one A100 running them one after another
    pays their sum. That sum is the number a reader deciding whether to deploy this needs, it is
    additive over the members of the condition (so the core-3 jury gets its own real cost, not three eighths of the pool's), and every term in it was measured.

    Returns nothing when a requested member is missing or in *skip*: a sum over five of eight
    jurors is not a smaller jury's runtime, it is the wrong number for this one.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = sorted({p for pat in patterns for p in directory.glob(str(pat))})
    if not files:
        return {}

    skip = skip or set()
    total_s = 0.0
    n_reports: int | None = None
    models, cached_grounding, excluded = [], [], []
    for path in files:
        stages = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    stages.append(json.loads(line))
        extract = next((s for s in stages if s.get("stage") == "extract"), None)
        if not extract or not extract.get("duration_s"):
            continue
        model = str(extract.get("model") or path.stem)
        if model in skip:
            excluded.append(model)
            continue
        total_s += float(extract["duration_s"])
        models.append(model)
        if extract.get("n_reports"):
            n_reports = max(n_reports or 0, int(extract["n_reports"]))
        # The same trap as the PhenoBERT baseline's, one level down: a juror whose grounding was served from
        # cache never paid for PhenoBERT, so its duration is not what a fresh run would cost.
        if any(s.get("stage", "").startswith("phenobert_") and s.get("status") == "cached"
               for s in stages):
            cached_grounding.append(model)

    if excluded or not models or not n_reports:
        return {"n_timing_files": len(files), "models_excluded": sorted(excluded),
                "n_models_timed": len(models)}
    return {
        "seconds_per_report": total_s / n_reports,
        "n_reports": n_reports,
        "n_timing_files": len(files),
        "n_models_timed": len(models),
        "models_excluded": [],
        "models_cached_grounding": sorted(cached_grounding),
        "sequential_total_s": total_s,
    }


#: The driver's two ways of saying "this model did not generate the whole cohort in this run".
_RESUMED_RE = (
    re.compile(r"Resuming\s+(?P<model>\S+)\s+from\s+\d+\s+cached record"),
    re.compile(r"Extractions for\s+(?P<model>\S+)\s+are complete"),
)

#: ``Extract done | model=phi4 2713 sentences → 1118 detections in 208.8s | peak GPU 20.79 GB``.
#: The detections clause changed format between runs (the non-empty count was added later), so
#: everything between the model and the duration is skipped, not parsed.
_EXTRACT_DONE_RE = re.compile(
    r"Extract done\s*\|\s*model=(?P<model>\S+).*?\sin\s+(?P<seconds>[\d.]+)s\s*\|\s*"
    r"peak GPU\s+(?P<gb>[\d.]+)\s*GB")

#: Peak GPU above which a pass had the model RESIDENT, i.e. actually generated. A pass that
#: only re-grounds cached text reports 0.01 GB. The smallest juror here peaks at 8.1 GB. Anything
#: in between would be a run this rule has not seen, and it is better to drop it than to average
#: a generation and a re-grounding into one number.
_RESIDENT_GPU_GB = 1.0


def from_extract_logs(paths: list) -> dict:
    """Each juror's end-to-end wall clock, recovered from the pass that actually generated.

    The fallback for what ``from_per_model_timing`` cannot use. Those per-model files are rewritten
    on every submission, and on both cohorts the *last* submission re-grounded cached text, ``extraction_s`` is a twentieth of a second against 2 713 sentences, so they time PhenoBERT and
    nothing else. The run log, however, is **appended** across submissions, and the driver's
    ``Extract done`` line from the original generating pass is still in it, carrying that pass's
    duration and its peak GPU.

    Which lines are generating passes is decided by the peak GPU, not by the date: a pass with the
    model resident peaks at 8--21 GB, one that only re-grounds peaks at 0.01 GB. The duration
    on a generating line is the whole juror --- ``t0`` is set before the model loads and the
    duration is taken after grounding has written its detections --- so it is model load, generation
    over every sentence and PhenoBERT, which is the per-juror term the sum needs.

    A juror is returned only when a *complete* generating pass exists. The two reasoning models hit
    the 12 h wall clock and were resubmitted, and the incomplete path never writes this line, so
    their generation is spread over submissions that no single clock covers. They come back absent,
    which is what makes the conditions that contain them a lower bound instead of a wrong number.
    """
    best: dict = {}
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _EXTRACT_DONE_RE.finditer(text):
            if float(match.group("gb")) < _RESIDENT_GPU_GB:
                continue
            # Last generating pass wins: a model regenerated after a config change is costed at
            # what it costs now, not at what an superseded run happened to take.
            best[match.group("model")] = {"seconds": float(match.group("seconds")),
                                          "peak_gpu_gb": float(match.group("gb"))}
    return best


def sequential_from_extract_logs(per_model: dict, models: list, n_reports: int | None) -> dict:
    """Sum *models*' jurors into one sequential runtime, saying which jurors are missing."""
    if not models or not n_reports:
        return {}
    have = [m for m in models if m in per_model]
    missing = [m for m in models if m not in per_model]
    if not have:
        return {"models_missing": missing, "n_models_timed": 0}
    total_s = sum(per_model[m]["seconds"] for m in have)
    return {
        "seconds_per_report": total_s / n_reports,
        "sequential_total_s": total_s,
        "n_models_timed": len(have),
        "models_missing": missing,
        "peak_gpu_gb": max(per_model[m]["peak_gpu_gb"] for m in have),
        "n_reports": n_reports,
    }


# ── assembly ─────────────────────────────────────────────────────────────────

def collect(spec: dict, results_dir: str, cohort: str, artifact_cohort: str) -> CostRow:
    """One cost row, from whichever sources the declared experiment actually has."""
    row = CostRow(cohort=cohort)
    exp_id = str(spec.get("exp_id") or "")
    row.cost_experiment = exp_id
    row.accelerator = str(spec.get("accelerator") or "")
    row.checkpoint = str(spec.get("checkpoint") or "")
    row.scope = str(spec.get("scope") or "")
    row.note = str(spec.get("note") or "")
    if not exp_id:
        row.source = "not declared"
        return row

    base = Path(results_dir) / exp_id
    # `cohort_dir` is a subdirectory of the experiment, or a {cohort: subdirectory} map when one
    # cohort's cost was measured by a separate run (PhenoBERT on HCY: the scored run reused a
    # cached CNN output, so its runtime comes from a dedicated uncached rerun).
    subdir = spec.get("cohort_dir")
    if isinstance(subdir, dict):
        subdir = subdir.get(cohort) or subdir.get(artifact_cohort)
    cohort_dir = base / (str(subdir or artifact_cohort))
    sources: list[str] = []
    found: dict = {}

    variant = str(spec.get("variant") or "")
    dropped_variant_seconds = False
    if variant:
        timing = cohort_dir / f"{variant}_timing.jsonl"
        if timing.exists():
            got = from_timing(timing)
            # `use_timing_seconds: false` says the file's duration covers a stage that is not the
            # method's cost. The ensemble is the case: `slm_ensemble_timing.jsonl` times the
            # one-second vote aggregation, not the eight models generating over every sentence, so
            # taking its duration would report PhenoJury as the cheapest system in the roster by
            # three orders of magnitude. The report count and call count from the same file are
            # still good, which is why this drops one field, not the whole source.
            if got and not bool(spec.get("use_timing_seconds", True)):
                got.pop("seconds_per_report", None)
                dropped_variant_seconds = True
            if got:
                found.update(got)
                sources.append(f"timing:{timing.name}")

    logs = sorted(cohort_dir.glob(str(spec.get("log_glob") or "run*.log")))

    # A scope that depends on what the run actually did, not on what the config assumed.
    # The PhenoBERT baseline reuses a completed PhenoBERT output on one cohort and not on the other, so the same
    # declared scope is true of one row and false of its neighbour. Detecting it from the log keeps
    # The two honest independently.
    detect = spec.get("scope_if_log_contains") or {}
    if detect.get("pattern") and logs:
        marker = str(detect["pattern"])
        if any(marker in Path(p).read_text(encoding="utf-8", errors="replace") for p in logs):
            row.scope = str(detect.get("scope") or row.scope)
            row.scope_partial = True
            sources.append("scope:detected from run.log")

    if logs:
        got = from_run_logs(logs)
        if got:
            # The run log fills only what the timing file lacks. run.log is APPENDED across
            # submissions, so its peaks span every run ever made in the directory -- RAG-HPO 70B's
            # holds the August H200 run beside the 4x RTX 4090 re-run -- and a re-run whose MLflow
            # block failed never writes its own `Done` line, leaving an older run's as the last.
            # The timing file is rewritten per run, so where it has a field, that field is this run's.
            for key, value in got.items():
                found.setdefault(key, value)
            sources.append(f"runlog:{len(logs)} file(s)")

    # `calls_from` reads another experiment's table, which was computed on ONE cohort. Applying it
    # to a second cohort would publish a measurement from HCY under a GSC+ heading: the traversal's
    # call count depends on the documents, so it does not transfer even though the configuration
    # does. Gated on the cohort that produced the table.
    calls_from = spec.get("calls_from")
    calls_cohort = spec.get("calls_from_cohort")
    if calls_from and calls_cohort and str(calls_cohort) != cohort:
        calls_from = None
        row.note = (row.note + " " if row.note else "") + (
            f"calls/report is measured on `{calls_cohort}` only; the traversal's call count depends "
            f"on the documents, so it is not carried across cohorts.")
    if calls_from:
        got = from_protocol_calls(Path(results_dir) / str(calls_from) / "tables"
                                  / "core_quality.csv")
        if got:
            # With a cache source declared, the protocol's count is the configuration's and
            # goes to its own field. The row's calls are the generating run's (below).
            if spec.get("cache_calls_from"):
                row.operating_point_calls_per_report = got["llm_calls_per_report"]
                sources.append(f"operating-point calls:{calls_from}")
            else:
                found.update(got)
                sources.append(f"calls:{calls_from}")

    cache_from = spec.get("cache_calls_from")
    if cache_from:
        got = from_cache_calls(Path(results_dir) / str(cache_from) / "tables" / "cache_calls.csv",
                               cohort, str(spec.get("cache_index") or ""))
        if got:
            found.update(got)
            sources.append(f"calls:{cache_from}/cache_calls.csv")

    generation_glob = spec.get("generation_glob")
    n_generation_files, condition_models = None, []
    if generation_glob and "llm_calls_per_report" not in found:
        got = count_generation_calls(cohort_dir, generation_glob, found.get("n_reports"))
        if got:
            found["llm_calls_per_report"] = got["llm_calls_per_report"]
            n_generation_files = got["n_generation_files"]
            condition_models = got["models"]
            sources.append(f"generations:{n_generation_files} file(s)")

    # The jury's runtime, summed over its members, see `from_per_model_timing`. This is the last
    # source on purpose: it is a direct measurement of every juror, so it overrides the array job's
    # run log (which records one shard's wall clock, not the set's), not deferring to it.
    per_model_glob = spec.get("per_model_timing_glob")
    explained_missing_seconds = False
    if per_model_glob:
        skip = resumed_models(logs)
        got = from_per_model_timing(cohort_dir, per_model_glob, skip=skip)
        n_timed = got.get("n_models_timed", 0)
        # A sum is only this condition's cost if it covers this condition's jurors. Fewer timing files than
        # generation dumps means a member's clock is missing, and a short sum that looks measured
        # is worse than an honest blank.
        complete = bool(got.get("seconds_per_report")) and (
            n_generation_files is None or n_timed == n_generation_files)
        if complete:
            found["seconds_per_report"] = got["seconds_per_report"]
            sources.append(f"per-model timing:{n_timed} file(s), summed")
            row.scope = (row.scope + " " if row.scope else "") + (
                f"The {n_timed} jurors ran as a job array on separate devices; seconds/report is "
                f"their SUM, i.e. what one accelerator pays running them one after another "
                f"({got['sequential_total_s'] / 3600:.1f} GPU-hours over "
                f"{got['n_reports']} reports).")
            if got.get("models_cached_grounding"):
                row.note = (row.note + " " if row.note else "") + (
                    "PhenoBERT grounding was served from cache for "
                    + ", ".join(got["models_cached_grounding"])
                    + ", so those jurors' seconds cover generation only.")
        else:
            # The timing files are unusable, on both cohorts because every juror resumed, so they
            # time the re-grounding and not the generation. Fall back to the generating passes
            # still recorded in the appended run log. Both routes sum the same per-juror quantity;
            # this one just reads it from prose the driver wrote, not from JSON it wrote.
            why = ("every juror resumed from cached generations, so its timing file covers the "
                   "re-grounding rather than the generation"
                   if got.get("models_excluded") else
                   f"only {n_timed} of {n_generation_files} jurors wrote a usable timing file")
            seq = sequential_from_extract_logs(from_extract_logs(logs), condition_models,
                                               found.get("n_reports"))
            if seq.get("seconds_per_report"):
                found["seconds_per_report"] = seq["seconds_per_report"]
                found["peak_gpu_gb"] = seq["peak_gpu_gb"]
                row.seconds_is_lower_bound = bool(seq["models_missing"])
                sources.append(
                    f"extract log:{seq['n_models_timed']} generating pass(es), summed")
                row.scope = (row.scope + " " if row.scope else "") + (
                    f"The jurors ran as a job array on separate devices; seconds/report is their "
                    f"SUM, i.e. what one accelerator pays running them one after another "
                    f"({seq['sequential_total_s'] / 3600:.1f} GPU-hours over "
                    f"{seq['n_reports']} reports). Each term is one juror's complete pass — "
                    f"checkpoint load, generation over every sentence, and PhenoBERT grounding.")
                row.note = (row.note + " " if row.note else "") + (
                    f"Taken from the driver's `Extract done` lines rather than the per-model "
                    f"timing files, because {why}. A generating pass is identified by its peak "
                    f"GPU (8-21 GB with the checkpoint resident, 0.01 GB when only re-grounding).")
                if seq["models_missing"]:
                    row.note += (
                        " LOWER BOUND: " + ", ".join(seq["models_missing"])
                        + " hit the 12 h wall clock and was resubmitted, and the incomplete path "
                          "writes no `Extract done` line, so no single clock covers its "
                          "generation. The figure sums the "
                        + f"{seq['n_models_timed']} jurors that have one.")
            else:
                explained_missing_seconds = True
                row.note = (row.note + " " if row.note else "") + (
                    f"seconds/report is withheld: {why}, and no generating pass survives in the "
                    f"run log either.")

    # The fallback explanation, and only when nothing more specific was said above -- two notes
    # disagreeing about why the same cell is blank is worse than either alone.
    if dropped_variant_seconds and "seconds_per_report" not in found \
            and not explained_missing_seconds:
        row.note = (row.note + " " if row.note else "") + (
            "seconds/report is not recorded for this method: its timing file covers only "
            "the aggregation stage, and the generation dumps carry no wall-clock field.")

    for key in ("seconds_per_report", "llm_calls_per_report", "peak_gpu_gb", "model_load_s",
                "n_reports"):
        if key in found and found[key] is not None:
            setattr(row, key, found[key])
    row.source = "; ".join(sources) if sources else f"no cost artifact under {cohort_dir}"
    return row


def load_subset_timing(path: Path) -> dict:
    """``{(method, cohort): row}`` from ``experiments/06_comparison/recover_gsc_subset_timing.py``; ``{}`` if absent."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {(r["method"], r["cohort"]): r for r in csv.DictReader(fh)}


def apply_subset_timing(row: CostRow, method: str, table: dict, path: Path) -> None:
    """Replace a GSC+ subset's seconds/report with the sum of its own documents' times.

    Every GSC+ system ran once over all 228 abstracts, so the run-level sources above divide one
    total by 228 and give both subsets the same number. The recovery script reads the per-document
    clocks those runs left (tqdm lines, output-file mtimes, the ensemble's per-sentence log lines),
    checks them against the same run totals, and sums them per subset plus the run's one-time
    overhead, charged once. Only seconds change: calls and peak GPU stay the run's.
    """
    hit = table.get((method, row.cohort))
    if hit is None:
        return
    row.seconds_per_report = float(hit["seconds_per_report"])
    # The trace covers every juror, including one whose generation spanned several submissions,
    # so the floor the run-level route had to print no longer applies.
    row.seconds_is_lower_bound = False
    row.seconds_is_approximate = str(hit.get("exact", "True")).lower() != "true"
    row.source = (row.source + "; " if row.source else "") + f"subset timing:{path.name}"
    row.note = (row.note + " " if row.note else "") + (
        f"seconds/report is this subset's own: the sum of its {hit['n_documents']} documents' "
        f"per-document times ({float(hit['document_seconds']):.0f} s) plus the run's one-time "
        f"overhead ({float(hit['overhead_seconds']):.0f} s), recovered by "
        "scripts/recover_gsc_subset_timing.py."
        + (" Part of it (RAG-HPO's LLM-mapping stage) is apportioned by each document's "
           "non-exact findings, not timed per document." if row.seconds_is_approximate else ""))


def rows_to_dicts(rows: list) -> list:
    """The cost rows as plain dicts."""
    return [asdict(r) for r in rows]
