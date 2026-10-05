"""Driver for the Free Listing generation run SLM-ensemble baseline (earlier methodology, earlier harness).

The third method family in the earlier comparison, next to TreePhenoRAG traversal (13_00/01/02),
RAG-HPO (13_03/04) and flat top-M retrieve-then-classify (13_05). No retrieval and no ontology
traversal: each of the eight ensemble SLMs reads every report sentence and freely writes the
signs/symptoms it sees, PhenoBERT grounds that free text to HPO IDs with per-sentence
attribution, and the per-model detections are then aggregated into a report-level term set.

Two stages, selected by ``cfg.stage``:

``extract``
    One model (``cfg.model_key``) on one GPU: generate → PhenoBERT → write this model's
    ``llm_extractions_{model}.jsonl`` and ``detections_{model}.jsonl``. Eight of these run as a
    SLURM job array, so both the generation *and* the PhenoBERT pass parallelise across nodes.

``aggregate``
    CPU-only: every per-model cache hits (no model is loaded), the detection slices are merged
    into one re-run dump, and the aggregation sweep is scored against ground truth.

**Key implementation choice, aggregation is decoupled from inference.** Which models detected
which HPO in which sentence is independent of the voting rule, so it is dumped once
(``slm_ensemble_detections.jsonl``) and every rule is re-run over that dump at zero GPU cost.
This is the same invariant the tree driver uses for its tau sweep: score once, re-run the
decision logic. New aggregation rules therefore never require re-running the models.

Rules swept here: ``vote_k{k}`` for k = 1..n_active (an HPO is predicted iff at least k models
detected it, k=1 is the earlier runs'union, k=n_active is unanimity) and ``agg_plurality``
(the earlier runs'per-sentence plurality). Scoring is the earlier runs'open-set frame: predicted term set vs the
annotated ground-truth set, no ancestor closure, micro/macro P/R/F1.

Deliberate deltas from the earlier runs (documented in the experiment.md):
  * decoding is greedy (``deterministic=True``), where earlier sampled at temperature 0.6;
  * both cohorts (HCY + GSC+), via the tree driver's shared ``_load_dataset``;
  * open-set scoring, not the earlier runs'fixed target-symptom matrix with children-aware matching.

Reuses the earlier building blocks in ``hpo_extraction.phenojury.ensemble_eval`` (PhenoBERT per model, productivity
metrics) but not its ``evaluate_ensemble`` orchestrator, which is bound to HCY and to the
target-symptom label matrix.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict

import mlflow
import pandas as pd
import torch

from hpo_extraction.utils.resume import GracefulStop, open_jsonl
from hpo_extraction.phenojury.ensemble_eval import (
    compute_slm_productivity,
    derive_patient_hpos,
    format_slm_metrics_table,
    merge_slm_metrics,
    run_phenobert_per_model,
    write_slm_metrics,
)
from hpo_extraction.data.segmentation import load_stanza, segment_dict, split_sents
from hpo_extraction.treephenorag.score_store import _load_dataset, _setup_logging
from hpo_extraction.evaluation.set_metrics import calc_metric
from hpo_extraction.models.slm_loaders import load_slm
from hpo_extraction.utils.mlflow_guard import mlflow_run

# Ensemble membership and order, identical to the earlier runs (MODEL_KEYS of their driver). The key is
# essential: it selects the loader family in `load_slm` and names every output file.
MODEL_KEYS = [
    "apertus",
    "deepseek",
    "intelligent_internet",
    "openbiollm",
    "llama",
    "medpsy",
    "medgemma",
    "phi4",
]

# The earlier runs'extraction prompt, verbatim (an earlier exploratory run/run.py). Kept unchanged so
# this is a harness port, not a prompt experiment.
SYSTEM_PROMPT = (
    "You are a medical expert who recognizes signs and symptoms in medical case reports."
)
USER_TEMPLATE = (
    "List any clinical signs or symptoms in this sentence: '{sent}' "
    "Short responses only. If there are no symptoms mentioned, respond 'no phenotype'."
)


# ──────────────────────────────────────────────────────────────────────────────
# Paths / small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extractions_path(run_dir: str, model: str) -> str:
    # The earlier runs'filename, kept so an existing earlier extraction cache can be dropped in.
    return os.path.join(run_dir, f"llm_extractions_{model}.jsonl")


def _detections_path(run_dir: str, model: str) -> str:
    return os.path.join(run_dir, f"detections_{model}.jsonl")


def _configured_models(cfg) -> list:
    """``[(key, path)]`` in MODEL_KEYS order for every key with a non-empty path."""
    slm_dirs = cfg.get("slm_dirs") or {}
    return [(k, slm_dirs.get(k)) for k in MODEL_KEYS if slm_dirs.get(k)]


def _max_new_tokens_for(cfg, model_key: str) -> int:
    """Per-model token budget: reasoning models need headroom or their CoT never terminates
    and ``strip_think`` drops the whole block, silently zeroing their contribution."""
    overrides = cfg.get("max_new_tokens_overrides") or {}
    return int(overrides.get(model_key, cfg.max_new_tokens))


def _report_ids(cfg, reports, logger) -> list:
    """The report cohort, honouring the pilot restriction. Identical in both stages."""
    report_ids = sorted(reports)
    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("PILOT: restricted to %d report(s)", len(report_ids))
    return report_ids


def _segment_reports(cfg, reports, report_ids, logger):
    """``{report_id: [sentence]}`` for the cohort.

    Stanza tokenisation is deterministic, so an extraction job and the later aggregation job
    agree on every ``sentence_number`` without having to persist the segmentation, which is
    why only the extract stage pays for loading Stanza.
    """
    nlp = load_stanza(stanza_dir=cfg.stanza_dir, mode="TOKENIZER")
    sent_dict = split_sents(segment_dict({r: reports[r] for r in report_ids}, nlp))
    logger.info(
        "%d reports | %d sentences", len(report_ids), sum(len(v) for v in sent_dict.values())
    )
    return sent_dict


def _safe_micro_macro(gold_sets, pred_sets) -> dict:
    """Micro/macro P/R/F1 with the degenerate cases guarded.

    Same definition as ``hpo_extraction.evaluation.set_metrics.evaluate_micro_macro`` (and it reuses that module's
    per-report :func:`calc_metric` for the macro half), but every denominator is guarded. The
    shared helper divides by the prediction total, the ground truth total and ``p+r`` unguarded, which is
    fine for the retrieval experiments but not for this sweep: a high vote threshold can
    legitimately predict nothing, or predict terms that never hit ground truth, and both must score 0
    rather than crash the run partway through.
    """
    n = len(gold_sets)
    if n == 0:
        return {k: 0.0 for k in (
            "micro_precision", "micro_recall", "micro_f1",
            "macro_precision", "macro_recall", "macro_f1",
        )}

    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    n_pred = sum(len(p) for p in pred_sets)
    n_gold = sum(len(g) for g in gold_sets)
    micro_p = tp / n_pred if n_pred else 0.0
    micro_r = tp / n_gold if n_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    per_report = [calc_metric(g, p) for g, p in zip(gold_sets, pred_sets)]
    macro_p = sum(x[0] for x in per_report) / n
    macro_r = sum(x[1] for x in per_report) / n
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0

    return {
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1, generative extraction (one model)
# ──────────────────────────────────────────────────────────────────────────────
def _load_cached_records(path: str, logger=None) -> list:
    """Records already in an extractions JSONL, skipping any line killed mid-write.

    A torn line is not necessarily the *last* line: a resubmission appends after whatever the
    SIGKILL left behind, so a run that hit the wall clock twice leaves valid records on both sides
    of the damage (hcy deepseek: line 2551 torn, 170 good records after it). Records are
    independent, keyed by (report_id, sentence_number), so a bad line costs one regenerated
    sentence, whereas stopping at the first one silently discards work that is still on disk.
    """
    records: list = []
    if not os.path.isfile(path):
        return records
    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad and logger is not None:
        logger.warning("%s: skipped %d unparseable line(s) — killed mid-write, "
                       "those sentences will be regenerated", os.path.basename(path), n_bad)
    return records


def _run_extraction(cfg, model_key, model_path, sent_dict, report_ids, run_dir, logger,
                    stop=None, system_prompt=SYSTEM_PROMPT, user_template=USER_TEMPLATE
                    ) -> tuple[list, bool]:
    """Load one SLM, extract over every sentence, unload. Returns ``(records, complete)``.

    ``system_prompt``/``user_template`` default to the earlier runs', so the Free Listing generation run calls this as
    before. They are parameters only so :mod:`hpo_extraction.phenojury.generation_prompts` can sweep the prompt through
    the identical loop, making "earlier is the Free Listing generation run with a different prompt" true of the code,
    not a claim in a docstring.

    Generation is **modeled per report**: records are appended and flushed as each report
    finishes, and a rerun picks up whatever is already on disk. Before, the whole cohort was held
    in memory and written once at the very end (the Free Listing generation run's 12 h array tasks lost hours of
    generation to the wall clock this way).

    ``reuse_extractions`` therefore now means *partial* resume, not all-or-nothing: an
    existing file is continued, not blindly trusted. Set it false to regenerate from scratch.
    ``complete`` is False when a graceful stop cut the pass short, the caller must not treat that
    as a finished model.
    """
    path = _extractions_path(run_dir, model_key)
    resume = bool(cfg.get("reuse_extractions", True))
    cached = _load_cached_records(path) if resume else []
    done_keys = {(str(r.get("patient_id")), int(r.get("sentence_number", -1))) for r in cached}
    todo = [r for r in report_ids
            if any((str(r), i) not in done_keys for i in range(len(sent_dict[r])))]

    if cached:
        logger.info("Resuming %s from %d cached record(s) — %d/%d report(s) still to do",
                    model_key, len(cached), len(todo), len(report_ids))
    if not todo:
        logger.info("Extractions for %s are complete (%d records) — no generation needed",
                    model_key, len(cached))
        return cached, True

    max_new_tokens = _max_new_tokens_for(cfg, model_key)
    batch_size = max(1, int(cfg.get("gen_batch_size") or 1))
    logger.info(
        "Loading %s from %s | greedy | max_new_tokens=%d batch=%d",
        model_key, model_path, max_new_tokens, batch_size,
    )
    slm = load_slm(model_key, model_path, logger, deterministic=True)
    if batch_size > 1 and not slm.supports_batching:
        logger.info("[%s] family has no batched path — falling back to sequential", model_key)

    records: list = list(cached)
    complete = True
    out_f = open_jsonl(path, "a" if cached else "w")
    try:
        for n_done, report_id in enumerate(todo, start=1):
            sents = sent_dict[report_id]
            new_for_report: list = []
            for start in range(0, len(sents), batch_size):
                chunk = sents[start : start + batch_size]
                idx = list(range(start, start + len(chunk)))
                # Skip sentences a previous session already generated.
                keep = [i for i, s in zip(idx, chunk) if (str(report_id), i) not in done_keys]
                if not keep:
                    continue
                keep_sents = [sents[i] for i in keep]
                replies = slm.generate_many(
                    system_prompt,
                    [user_template.format(sent=s) for s in keep_sents],
                    max_new_tokens,
                )
                for sent_num, sent_text, reply in zip(keep, keep_sents, replies):
                    new_for_report.append({
                        "model": model_key,
                        "patient_id": report_id,
                        "sentence_number": sent_num,
                        "sentence_text": sent_text,
                        "llm_output": reply,
                    })
                    logger.debug(
                        "%s | %s | sent %d | %s", model_key, report_id, sent_num, reply[:100]
                    )

            # Durability point: one report's records land together, then hit the disk.
            for rec in new_for_report:
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            records.extend(new_for_report)

            if stop is not None and stop.should_stop():
                complete = False
                logger.warning(
                    "STOPPED_EARLY | %s | %s | %d/%d report(s) generated this session | "
                    "resubmit to resume", model_key, stop.reason, n_done, len(todo),
                )
                break
        logger.info("Saved %d records → %s", len(records), path)
    finally:
        out_f.close()
        slm.unload()
        logger.info("Model %s unloaded.", model_key)

    return records, complete


def _write_detections(model_key, sent_hpos, run_dir) -> int:
    """Write this model's slice of the re-run dump. One line per (report, sentence, HPO)."""
    path = _detections_path(run_dir, model_key)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for report_id in sorted(sent_hpos):
            for sent_num in sorted(sent_hpos[report_id]):
                for hpo_id, count in sorted(sent_hpos[report_id][sent_num].items()):
                    f.write(json.dumps({
                        "report_id": report_id,
                        "model": model_key,
                        "sentence_number": int(sent_num),
                        "hpo_id": hpo_id,
                        "count": int(count),
                    }) + "\n")
                    n += 1
    return n


def _read_detections(model_key, run_dir):
    """Inverse of :func:`_write_detections` → ``{report_id: {sent_num: {hpo_id: count}}}``."""
    path = _detections_path(run_dir, model_key)
    out: dict = defaultdict(lambda: defaultdict(dict))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:  # same mid-write tear as the extractions cache, drop the row, keep the file
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["report_id"]][int(rec["sentence_number"])][rec["hpo_id"]] = int(rec["count"])
    return {r: dict(s) for r, s in out.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2, aggregation rules (pure. Re-run over the detections dump)
# ──────────────────────────────────────────────────────────────────────────────
def vote_k_sets(per_model_report_hpos, active_models, report_ids, k, min_detection_count):
    """``{report_id: set(hpo)}`` where an HPO is kept iff >= k models detected it.

    k=1 reproduces the earlier runs'union. K=len(active_models) is unanimity.
    """
    out: dict = {}
    for rid in report_ids:
        tally: Counter = Counter()
        for model in active_models:
            for hpo_id, count in per_model_report_hpos.get(model, {}).get(rid, {}).items():
                if count >= min_detection_count:
                    tally[hpo_id] += 1
        out[rid] = {h for h, votes in tally.items() if votes >= k}
    return out


def plurality_sets(sent_hpos_by_model, active_models, report_ids, min_detection_count):
    """The earlier runs'per-sentence plurality, in the open-set frame.

    Per (report, sentence) every model votes once for each HPO it detected. Models that detected
    nothing don't vote. The top-voted HPO(s) win (ties keep all winners). A report's set is the
    union of its sentence winners. Returns ``(sets, vote_rows)``.
    """
    selected: dict = {rid: set() for rid in report_ids}
    stats = {m: {"n_votes_cast": 0, "n_votes_won": 0} for m in active_models}

    for rid in report_ids:
        sent_nums: set = set()
        for model in active_models:
            sent_nums |= set(sent_hpos_by_model.get(model, {}).get(rid, {}).keys())
        for sent_num in sent_nums:
            model_votes: dict = {}
            tally: Counter = Counter()
            for model in active_models:
                hpos = {
                    h
                    for h, c in sent_hpos_by_model.get(model, {})
                    .get(rid, {})
                    .get(sent_num, {})
                    .items()
                    if c >= min_detection_count
                }
                if hpos:
                    model_votes[model] = hpos
                    for h in hpos:
                        tally[h] += 1
            if not tally:
                continue
            max_votes = max(tally.values())
            winners = {h for h, c in tally.items() if c == max_votes}
            selected[rid] |= winners
            for model, hpos in model_votes.items():
                for h in hpos:
                    stats[model]["n_votes_cast"] += 1
                    if h in winners:
                        stats[model]["n_votes_won"] += 1

    vote_rows = []
    for model in active_models:
        cast = stats[model]["n_votes_cast"]
        won = stats[model]["n_votes_won"]
        vote_rows.append({
            "model": model,
            "n_votes_cast": cast,
            "n_votes_won": won,
            "pct_votes_won": round(100.0 * won / cast, 2) if cast else 0.0,
        })
    return selected, vote_rows


def _write_rule(run_dir, rule_dir, predicted, gold, report_ids, logger):
    """Write one rule's predictions.jsonl in the earlier contract and return its metrics."""
    out_dir = os.path.join(run_dir, rule_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "slm_ensemble_predictions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rid in report_ids:
            pred = predicted.get(rid, set())
            gold_set = gold.get(rid, set())
            for hpo_id in sorted(pred):
                f.write(json.dumps({
                    "report_id": rid,
                    "hpo_id": hpo_id,
                    "prediction": 1,
                    "ground_truth": int(hpo_id in gold_set),
                }) + "\n")
            f.write(json.dumps({
                "report_id": rid,
                "summary": True,
                "predicted_set": sorted(pred),
                "gold_set": sorted(gold_set),
            }) + "\n")

    metrics = _safe_micro_macro(
        [gold.get(rid, set()) for rid in report_ids],
        [predicted.get(rid, set()) for rid in report_ids],
    )
    logger.info(
        "%-14s | micro_f1=%.4f macro_f1=%.4f | mean |pred|=%.1f",
        rule_dir, metrics["micro_f1"], metrics["macro_f1"],
        sum(len(predicted.get(r, set())) for r in report_ids) / max(1, len(report_ids)),
    )
    return metrics, path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def execute(cfg, exp_id: str) -> None:
    """Run one stage of the juror generation: ``extract`` (one juror) or ``aggregate`` (the vote).

    Args:
        cfg: Hydra config (configs/experiments/05_phenojury/generate_free_listing.yaml).
        exp_id: name of the result folder, created as ``<output_dir>/<exp_id>/<dataset>/``.
    """
    run_dir = os.path.join(cfg.output_dir, exp_id, cfg.dataset)
    os.makedirs(run_dir, exist_ok=True)
    logger = _setup_logging(run_dir)
    stage = str(cfg.get("stage") or "extract")
    if stage not in ("extract", "aggregate"):
        raise ValueError(f"unknown stage {stage!r} (expected 'extract' or 'aggregate')")
    logger.info("Starting %s | dataset=%s stage=%s | out=%s", exp_id, cfg.dataset, stage, run_dir)

    # Validate before the expensive setup: a bad model_key must fail in seconds, not after the
    # dataset load and a Stanza pipeline build.
    model_key = None
    if stage == "extract":
        model_key = _validate_extract_target(cfg)

    reports, gt_dict = _load_dataset(cfg, logger)
    report_ids = _report_ids(cfg, reports, logger)

    if stage == "extract":
        sent_dict = _segment_reports(cfg, reports, report_ids, logger)
        _execute_extract(cfg, exp_id, run_dir, model_key, sent_dict, report_ids, logger)
    else:
        # No Stanza and no model in the aggregate job: report ids come straight from the
        # dataset loader and everything else is re-run from the per-model caches.
        gold = {rid: set(gt_dict.get(rid, [])) for rid in report_ids}
        _execute_aggregate(cfg, exp_id, run_dir, report_ids, gold, logger)


def _validate_extract_target(cfg) -> str:
    """Check ``model_key`` names a real ensemble member with a configured path. Returns the key."""
    model_key = cfg.get("model_key")
    if not model_key:
        raise ValueError("stage=extract requires model_key=<one of %s>" % ",".join(MODEL_KEYS))
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r} (expected one of {MODEL_KEYS})")
    if not (cfg.get("slm_dirs") or {}).get(model_key):
        raise ValueError(f"slm_dirs.{model_key} is not set — nothing to load")
    return str(model_key)


def _execute_extract(cfg, exp_id, run_dir, model_key, sent_dict, report_ids, logger) -> None:
    model_path = cfg.slm_dirs[model_key]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    stop = GracefulStop(max_runtime_s=cfg.get("max_runtime_s"))
    records, complete = _run_extraction(
        cfg, model_key, model_path, sent_dict, report_ids, run_dir, logger, stop=stop
    )
    t_extract = time.time() - t0

    if not complete:
        # Generation was cut short on purpose. Grounding a partial cohort would write a detections
        # slice the aggregation would then treat as this model's complete contribution.
        logger.warning(
            "INCOMPLETE | %s | %d record(s) saved, PhenoBERT skipped | resubmit to continue",
            model_key, len(records),
        )
        return

    timing_records: list = []
    sent_hpos = run_phenobert_per_model(
        model_key, records, report_ids, run_dir, cfg,
        bool(cfg.get("reuse_extractions", True)), timing_records, logger,
    )
    n_det = _write_detections(model_key, sent_hpos, run_dir)
    duration = time.time() - t0
    peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    timing_path = os.path.join(run_dir, f"slm_ensemble_timing_{model_key}.jsonl")
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "stage": "extract", "model": model_key, "n_reports": len(report_ids),
            "n_sentences": len(records), "n_detections": n_det,
            "extraction_s": round(t_extract, 3), "duration_s": round(duration, 3),
        }) + "\n")
        for rec in timing_records:
            f.write(json.dumps(rec) + "\n")

    with mlflow_run(f"{exp_id}_{cfg.dataset}_extract_{model_key}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "dataset": cfg.dataset, "method": "slm_ensemble",
            "stage": "extract", "model_key": model_key, "model_dir": model_path,
            "deterministic": True, "max_new_tokens": _max_new_tokens_for(cfg, model_key),
            "gen_batch_size": cfg.get("gen_batch_size"), "n_reports": len(report_ids),
        })
        mlflow.log_metrics({
            "n_reports": len(report_ids), "n_sentences": len(records), "n_detections": n_det,
            "duration_seconds": duration,
            "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
            "peak_gpu_mem_bytes": peak_mem,
        })
        for path in (_extractions_path(run_dir, model_key), _detections_path(run_dir, model_key),
                     timing_path):
            mlflow.log_artifact(path)

    n_nonempty = sum(1 for r in records if (r.get("llm_output") or "").strip())
    logger.info(
        "Extract done | model=%s %d sentences (%d non-empty) → %d detections in %.1fs | "
        "peak GPU %.2f GB",
        model_key, len(records), n_nonempty, n_det, duration, peak_mem / 1024 ** 3,
    )

    # A model that generated text but grounded to nothing is a pipeline failure, not a result: the
    # aggregation would silently drop it and the array task would still look green. Exit non-zero
    # so it shows up as FAILED (this is what the Free Listing generation run did on both cohorts, 8 models each).
    if records and n_det == 0:
        logger.error(
            "NO DETECTIONS | model=%s | %d record(s), %d non-empty, PhenoBERT grounded 0 HPOs. "
            "Inspect %s and the 'PhenoBERT output' counters above: HP: rows without a sentence "
            "index mean the PhenoBERT marker patch is missing; no rows at all means PhenoBERT saw "
            "empty input.",
            model_key, len(records), n_nonempty,
            os.path.join(run_dir, f"phenobert_output_{model_key}"),
        )
        raise SystemExit(3)


def _execute_aggregate(cfg, exp_id, run_dir, report_ids, gold, logger) -> None:
    configured = _configured_models(cfg)
    logger.info("Configured models (%d): %s", len(configured), [k for k, _ in configured])
    t0 = time.time()

    # Load every model's caches. A model with no detections dump simply didn't finish, the run
    # continues without it (the earlier runs'failure contract), with the survivors recorded in MLflow.
    all_records: dict = {}
    sent_hpos_by_model: dict = {}
    active_models: list = []
    missing_models: list = []
    for model_key, _ in configured:
        det_path = _detections_path(run_dir, model_key)
        ext_path = _extractions_path(run_dir, model_key)
        if not (os.path.isfile(det_path) and os.path.isfile(ext_path)):
            logger.warning("No cache for %s (%s) — excluded from aggregation", model_key, det_path)
            missing_models.append(model_key)
            continue
        if os.path.getsize(det_path) == 0:
            # An empty detections file is a grounding failure, not "this model found nothing":
            # counting it as active would let a broken PhenoBERT pass as a legitimate zero vote.
            logger.warning("Empty detections for %s (%s) — excluded from aggregation",
                           model_key, det_path)
            missing_models.append(model_key)
            continue
        all_records[model_key] = _load_cached_records(ext_path, logger)
        sent_hpos_by_model[model_key] = _read_detections(model_key, run_dir)
        active_models.append(model_key)

    if not active_models:
        raise RuntimeError(
            f"No usable per-model caches in {run_dir} — {len(missing_models)} model(s) missing or "
            f"empty: {missing_models}. Run stage=extract for each model first, and check its log "
            "for a NO DETECTIONS error (grounding produced nothing)."
        )
    logger.info("Active models (%d): %s", len(active_models), active_models)

    # Merge the per-model slices into the single re-run dump.
    merged_path = os.path.join(run_dir, "slm_ensemble_detections.jsonl")
    n_merged = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for model_key in active_models:
            with open(_detections_path(run_dir, model_key), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        n_merged += 1
    logger.info("Replay dump: %d detection rows → %s", n_merged, merged_path)

    per_model_report_hpos = derive_patient_hpos(sent_hpos_by_model)
    min_count = int(cfg.min_detection_count)

    # ── Aggregation sweep ─────────────────────────────────────────────────────
    summary_rows: list = []
    artifact_dirs: list = []
    prev_total = None
    for k in range(1, len(active_models) + 1):
        predicted = vote_k_sets(per_model_report_hpos, active_models, report_ids, k, min_count)
        metrics, _ = _write_rule(run_dir, f"vote_k{k}", predicted, gold, report_ids, logger)
        total = sum(len(v) for v in predicted.values())
        # Self-check: raising the vote threshold can only remove terms.
        if prev_total is not None and total > prev_total:
            raise AssertionError(
                f"vote_k{k} predicted {total} terms > vote_k{k - 1}'s {prev_total} "
                "— the k-of-N sweep must be monotone non-increasing"
            )
        prev_total = total
        summary_rows.append({"rule": f"vote_k{k}", "k": k, "n_predicted": total, **metrics})
        artifact_dirs.append(f"vote_k{k}")

    plurality, vote_rows = plurality_sets(
        sent_hpos_by_model, active_models, report_ids, min_count
    )
    metrics_plur, _ = _write_rule(run_dir, "agg_plurality", plurality, gold, report_ids, logger)
    summary_rows.append({
        "rule": "agg_plurality", "k": "",
        "n_predicted": sum(len(v) for v in plurality.values()), **metrics_plur,
    })
    artifact_dirs.append("agg_plurality")

    summary_path = os.path.join(run_dir, "slm_ensemble_agg_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # ── Per-SLM metrics ───────────────────────────────────────────────────────
    productivity_rows = compute_slm_productivity(all_records, sent_hpos_by_model, active_models)
    slm_rows = merge_slm_metrics(productivity_rows, vote_rows)
    slm_metrics_path = os.path.join(run_dir, "slm_ensemble_slm_metrics.csv")
    write_slm_metrics(slm_rows, slm_metrics_path)
    print(format_slm_metrics_table(slm_rows))

    duration = time.time() - t0
    timing_path = os.path.join(run_dir, "slm_ensemble_timing.jsonl")
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "stage": "aggregate", "n_reports": len(report_ids),
            "n_active_models": len(active_models), "n_detection_rows": n_merged,
            "duration_s": round(duration, 3),
        }) + "\n")

    with mlflow_run(f"{exp_id}_{cfg.dataset}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "dataset": cfg.dataset, "method": "slm_ensemble",
            "stage": "aggregate", "active_models": ",".join(active_models),
            "active_model_count": len(active_models),
            "missing_models": ",".join(missing_models) if missing_models else "none",
            "deterministic": True, "min_detection_count": min_count,
            "max_new_tokens": cfg.max_new_tokens, "phenobert_p1": cfg.phenobert_p1,
            "phenobert_p2": cfg.phenobert_p2, "phenobert_p3": cfg.phenobert_p3,
            "n_reports": len(report_ids),
        })
        metrics_out = {
            "n_reports": len(report_ids), "n_active_models": len(active_models),
            "n_detection_rows": n_merged, "duration_seconds": duration,
            "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
        }
        for row in summary_rows:
            prefix = row["rule"]
            for key in ("micro_f1", "macro_f1", "micro_precision", "micro_recall"):
                metrics_out[f"{prefix}_{key}"] = row[key]
            metrics_out[f"{prefix}_n_predicted"] = row["n_predicted"]
        mlflow.log_metrics(metrics_out)

        for path in (merged_path, summary_path, slm_metrics_path, timing_path):
            mlflow.log_artifact(path)
        for rule_dir in artifact_dirs:
            mlflow.log_artifacts(os.path.join(run_dir, rule_dir), artifact_path=rule_dir)

    best = max(summary_rows, key=lambda r: r["micro_f1"])
    logger.info(
        "Aggregate done | %d reports × %d models in %.1fs | best rule=%s micro_f1=%.4f",
        len(report_ids), len(active_models), duration, best["rule"], best["micro_f1"],
    )
