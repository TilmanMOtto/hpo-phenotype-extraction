"""Driver for the earlier prompt sweep, the same SLM ensemble, one prompt per condition.

The Free Listing generation run runs eight SLMs over every sentence and lets PhenoBERT ground their free text. The
2026-08-11 deep dive found the models mostly echo the sentence, so PhenoBERT sees the string it
would have seen from the raw report and the ensemble cannot structurally beat PhenoBERT standalone.
``experiments/findings/exp13_phenobert_input_format.md`` then measured what restating is worth: a
canonical HPO name grounds at 89.2 % against 58.5 % for the report's own wording.

This driver turns the prompt into a swept axis. It is a **port of**
:mod:`hpo_extraction.phenojury.generation`, not a fork, the extraction loop, the PhenoBERT boundary, the
vote sweep and the scoring frame are imported from it, so an earlier condition and the Free Listing generation run row are the
same pipeline differing in one string. The Free Listing generation run itself is untouched: it is a locked
comparison-table row.

Three stages, selected by ``cfg.stage``:

``extract``
    One (prompt, model) cell on one GPU: generate → PhenoBERT → write this cell's
    ``llm_extractions_{model}.jsonl`` and ``detections_{model}.jsonl`` under
    ``{run_dir}/{prompt_key}/``. Run as a flat SLURM array over prompts × models.

``aggregate``
    CPU-only, one prompt: re-run the vote sweep over that prompt's cached detections, score against
    ground truth, and compute the text-level diagnostics (:mod:`hpo_extraction.phenojury.prompt_diagnostics`).

``select``
    CPU-only, all prompts: rank by best micro-F1 across the k-sweep and decide whether the winner
    beats the baseline prompt by enough to be worth a full-cohort run.

**Why the prompt is a directory level.** Every cell caches its own generation, so the sweep resumes
per (prompt, model) after a wall-clock kill, and the baseline condition can reuse the Free Listing generation run's existing
extractions by dropping them in, no GPU for the control. It also means adding a prompt later never
invalidates the ones already run.

**The promotion gate.** ``select`` writes a verdict, it does not submit anything. The cluster script
reads the verdict and submits. Keeping the decision in Python and the submission in bash means the
threshold is testable without SLURM, and a re-run of ``select`` cannot accidentally launch 32 GPU
tasks twice.
"""

from __future__ import annotations

import json
import os
import time

import mlflow
import pandas as pd
import torch

from hpo_extraction.utils.resume import GracefulStop
from hpo_extraction.phenojury.ensemble_eval import (
    compute_slm_productivity,
    derive_patient_hpos,
    format_slm_metrics_table,
    merge_slm_metrics,
    run_phenobert_per_model,
    write_slm_metrics,
)
from hpo_extraction.phenojury.prompt_diagnostics import diagnose_records, marginal_gold_yield
from hpo_extraction.phenojury.prompts import BASELINE_KEY, PROMPT_KEYS, get_prompt

# The ensemble driver is the source of truth for everything the prompt does not change. Importing
# rather than copying is what makes "earlier is the Free Listing generation run with a different prompt" a fact about the
# code and not a claim in a docstring.
from hpo_extraction.phenojury.generation import (
    MODEL_KEYS,
    _load_cached_records,
    _max_new_tokens_for,
    _run_extraction,
    _segment_reports,
    _write_detections,
    _write_rule,
    plurality_sets,
    vote_k_sets,
)
from hpo_extraction.treephenorag.score_store import _load_dataset, _setup_logging
from hpo_extraction.ontology.hpo_tree import HPOTree
from hpo_extraction.utils.mlflow_guard import mlflow_run

STAGES = ("extract", "aggregate", "select")


# ──────────────────────────────────────────────────────────────────────────────
# Paths, every artifact of a cell lives under {run_dir}/{prompt_key}/
# ──────────────────────────────────────────────────────────────────────────────
def prompt_dir(run_dir: str, prompt_key: str) -> str:
    """Folder of one prompt's cell inside a run folder."""
    return os.path.join(run_dir, prompt_key)


def _extractions_path(run_dir: str, prompt_key: str, model: str) -> str:
    # The Free Listing generation run's filename, unchanged, one directory deeper, so an existing the Free Listing generation run cache can be
    # copied into p0_baseline/ verbatim and picked up as the control's generation.
    return os.path.join(prompt_dir(run_dir, prompt_key), f"llm_extractions_{model}.jsonl")


def _detections_path(run_dir: str, prompt_key: str, model: str) -> str:
    return os.path.join(prompt_dir(run_dir, prompt_key), f"detections_{model}.jsonl")


# ──────────────────────────────────────────────────────────────────────────────
# Cohort selection
# ──────────────────────────────────────────────────────────────────────────────
def select_cohort(cfg, reports, gt_dict, run_dir, logger) -> list:
    """The screening cohort: the first ``max_patients`` reports that actually carry annotations.

    ``slm_ensemble_experiment._report_ids`` truncates the sorted id list directly, which is right
    for a full run but wrong for a 20-report screen: HCY has 118 reports and only 100 annotated
    (``experiments/findings/exp13_findings.md``), so a blind head-20 can include reports whose ground truth
    set is empty. Those score zero precision for every prompt equally and shrink the effective
    sample, the ranking would then rest on fewer reports than it claims.

    Selection is deterministic (sorted ids, no sampling) so a resubmission, the aggregate job and
    phase B all agree without coordination. The chosen ids are written to ``screen_cohort.json``
    anyway, because "which 20 reports" is the first question any later reader of the finding asks.
    """
    report_ids = sorted(reports)
    if cfg.get("require_annotated", True):
        annotated = [r for r in report_ids if gt_dict.get(r)]
        n_dropped = len(report_ids) - len(annotated)
        if n_dropped:
            logger.info("Skipping %d report(s) with no annotation", n_dropped)
        report_ids = annotated

    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("SCREEN: restricted to %d report(s)", len(report_ids))

    if not report_ids:
        raise RuntimeError(
            "Empty cohort. With require_annotated=true this means no report in "
            f"{cfg.dataset} carries a gold annotation — check ground_truth_path."
        )

    cohort_path = os.path.join(run_dir, "screen_cohort.json")
    with open(cohort_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": cfg.dataset,
            "n_reports": len(report_ids),
            "require_annotated": bool(cfg.get("require_annotated", True)),
            "report_ids": list(report_ids),
        }, f, indent=2)
    logger.info("Cohort: %d report(s) → %s", len(report_ids), cohort_path)
    return report_ids


# ──────────────────────────────────────────────────────────────────────────────
# SLURM array decoding
# ──────────────────────────────────────────────────────────────────────────────
def decode_array_index(index: int, prompt_keys, model_keys=MODEL_KEYS) -> tuple[str, str]:
    """``array_task_id`` → ``(prompt_key, model_key)``.

    One flat array over the product, not a nested submission: SLURM has no two-dimensional
    arrays, and a job-per-prompt wrapper would multiply the number of scripts by the number of
    prompts. Prompt-major ordering (``idx // n_models``) keeps one prompt's eight models on
    consecutive indices, so a single prompt can be re-run as one contiguous ``--array=8-15``.
    """
    n_models = len(model_keys)
    total = len(prompt_keys) * n_models
    if not 0 <= index < total:
        raise ValueError(
            f"array index {index} out of range for {len(prompt_keys)} prompt(s) × "
            f"{n_models} model(s) = {total} task(s)"
        )
    return prompt_keys[index // n_models], model_keys[index % n_models]


def _requested_prompts(cfg) -> list:
    """The prompt roster for this run: ``cfg.prompt_keys`` if set, else the whole library.

    Phase B passes the two winners here, which is what makes one driver serve both phases.
    """
    raw = cfg.get("prompt_keys")
    if not raw:
        return list(PROMPT_KEYS)
    keys = [k.strip() for k in str(raw).split(",")] if isinstance(raw, str) else list(raw)
    for k in keys:
        get_prompt(k)  # raises with the valid list on a typo
    return keys


def _configured_models(cfg) -> list:
    """``[(key, path)]`` in MODEL_KEYS order for every key with a non-empty path."""
    slm_dirs = cfg.get("slm_dirs") or {}
    return [(k, slm_dirs.get(k)) for k in MODEL_KEYS if slm_dirs.get(k)]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1, extract one (prompt, model) cell
# ──────────────────────────────────────────────────────────────────────────────
def _execute_extract(cfg, exp_id, run_dir, prompt_key, model_key, sent_dict, report_ids, logger):
    spec = get_prompt(prompt_key)
    model_path = cfg.slm_dirs[model_key]
    cell_dir = prompt_dir(run_dir, prompt_key)
    os.makedirs(cell_dir, exist_ok=True)
    logger.info("Cell %s × %s | %s", prompt_key, model_key, spec.change)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    stop = GracefulStop(max_runtime_s=cfg.get("max_runtime_s"))
    # The extraction loop is the Free Listing generation run's, prompt strings and output directory swapped. It
    # models per report, so a wall-clock kill costs one report, not the cohort.
    records, complete = _run_extraction(
        cfg, model_key, model_path, sent_dict, report_ids, cell_dir, logger, stop=stop,
        system_prompt=spec.system, user_template=spec.user_template,
    )
    t_extract = time.time() - t0

    if not complete:
        # Grounding a partial cohort would write a detections slice that aggregation then treats
        # as this cell's complete contribution.
        logger.warning(
            "INCOMPLETE | %s × %s | %d record(s) saved, PhenoBERT skipped | resubmit to continue",
            prompt_key, model_key, len(records),
        )
        return

    timing_records: list = []
    sent_hpos = run_phenobert_per_model(
        model_key, records, report_ids, cell_dir, cfg,
        bool(cfg.get("reuse_extractions", True)), timing_records, logger,
    )
    n_det = _write_detections(model_key, sent_hpos, cell_dir)
    duration = time.time() - t0
    peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    # Run name is {exp_id}_{prompt_key}_{dataset}_…: the result-table library attributes cost by
    # substring-matching "{exp_id}_{prompt_key}_{cohort}", so the prompt must precede the cohort or
    # both earlier rows would collect the same GPU numbers (discovery.MethodSpec.run_name_prefix).
    with mlflow_run(f"{exp_id}_{prompt_key}_{cfg.dataset}_extract_{model_key}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "dataset": cfg.dataset, "method": "slm_ensemble",
            "stage": "extract", "prompt_key": prompt_key, "prompt_change": spec.change,
            "model_key": model_key, "model_dir": model_path, "deterministic": True,
            "max_new_tokens": _max_new_tokens_for(cfg, model_key),
            "gen_batch_size": cfg.get("gen_batch_size"), "n_reports": len(report_ids),
        })
        mlflow.log_metrics({
            "n_reports": len(report_ids), "n_sentences": len(records), "n_detections": n_det,
            "extraction_seconds": t_extract, "duration_seconds": duration,
            "peak_gpu_mem_bytes": peak_mem,
        })
        for path in (_extractions_path(run_dir, prompt_key, model_key),
                     _detections_path(run_dir, prompt_key, model_key)):
            mlflow.log_artifact(path)

    n_nonempty = sum(1 for r in records if (r.get("llm_output") or "").strip())
    logger.info(
        "Extract done | %s × %s | %d sentences (%d non-empty) → %d detections in %.1fs",
        prompt_key, model_key, len(records), n_nonempty, n_det, duration,
    )

    # A cell that generated text but grounded nothing is a pipeline failure, not a result: the
    # aggregation would drop it silently and the array task would still look green.
    if records and n_det == 0:
        logger.error(
            "NO DETECTIONS | %s × %s | %d record(s), %d non-empty, PhenoBERT grounded 0 HPOs. "
            "Check %s — and note that a prompt whose output is one long line is exactly the "
            "failure mode experiments/findings/exp13_phenobert_input_format.md documents.",
            prompt_key, model_key, len(records), n_nonempty,
            os.path.join(cell_dir, f"phenobert_output_{model_key}"),
        )
        raise SystemExit(3)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2, aggregate one prompt
# ──────────────────────────────────────────────────────────────────────────────
def _load_prompt_caches(cfg, run_dir, prompt_key, report_ids, logger):
    """``(all_records, sent_hpos_by_model, active, missing)`` for one prompt's cells.

    Records are filtered to ``report_ids``, and that filter is essential, not tidy. The
    recommended way to get ``p0_baseline`` for free is to drop the Free Listing generation run's HCY extractions into its
    directory, but those cover all 118 reports, while every other prompt was generated over the
    screen's 20. Without the filter the control's echo rate and label exactness would be measured
    on a different report set from its competitors', and the diagnostics that decide whether the
    ranking is trustworthy would be the least comparable numbers in the table.
    """
    wanted = {str(r) for r in report_ids}
    all_records: dict = {}
    sent_hpos_by_model: dict = {}
    active: list = []
    missing: list = []
    for model_key, _ in _configured_models(cfg):
        det_path = _detections_path(run_dir, prompt_key, model_key)
        ext_path = _extractions_path(run_dir, prompt_key, model_key)
        if not (os.path.isfile(det_path) and os.path.isfile(ext_path)):
            logger.warning("[%s] no cache for %s — excluded", prompt_key, model_key)
            missing.append(model_key)
            continue
        if os.path.getsize(det_path) == 0:
            # An empty detections file is a grounding failure, not "this model found nothing":
            # counting it active would let a broken cell pass as a legitimate zero vote.
            logger.warning("[%s] empty detections for %s — excluded", prompt_key, model_key)
            missing.append(model_key)
            continue
        cached = _load_cached_records(ext_path, logger)
        kept = [r for r in cached if str(r.get("patient_id")) in wanted]
        if len(kept) != len(cached):
            logger.info("[%s/%s] %d of %d cached record(s) are outside the cohort — excluded",
                        prompt_key, model_key, len(cached) - len(kept), len(cached))
        all_records[model_key] = kept
        sent_hpos_by_model[model_key] = _read_detections(run_dir, prompt_key, model_key)
        active.append(model_key)
    return all_records, sent_hpos_by_model, active, missing


def _read_detections(run_dir, prompt_key, model_key):
    """``{report_id: {sent_num: {hpo_id: count}}}`` from one cell's dump."""
    from collections import defaultdict

    out: dict = defaultdict(lambda: defaultdict(dict))
    with open(_detections_path(run_dir, prompt_key, model_key), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:  # a line torn by a mid-write kill costs one row, not the file
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["report_id"]][int(rec["sentence_number"])][rec["hpo_id"]] = int(rec["count"])
    return {r: dict(s) for r, s in out.items()}


def _load_phenobert_baseline(cfg, report_ids, logger) -> dict:
    """The PhenoBERT baseline's predictions as ``{report_id: set(hpo)}``, for the marginal-yield diagnostic.

    Optional by design: the screen must still produce a ranking on a machine where the PhenoBERT baseline has not
    run. A missing file degrades to an empty dict and the diagnostic reports ``pb_available=false``,
    not crediting every ground truth hit as a gain over nothing.
    """
    path = cfg.get("phenobert_predictions_path")
    if not path or not os.path.isfile(str(path)):
        logger.warning(
            "No PhenoBERT baseline at %r — marginal gold yield will be reported as unavailable. "
            "Point phenobert_predictions_path at exp13_08's predictions.jsonl to enable it.", path
        )
        return {}
    wanted = set(report_ids)
    out: dict = {}
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("report_id") or rec.get("patient_id")
            if rid not in wanted or rec.get("summary"):
                continue
            if rec.get("prediction"):
                out.setdefault(rid, set()).add(rec["hpo_id"])
    logger.info("PhenoBERT baseline: %d report(s) with predictions", len(out))
    return out


def _mean_over_cells(diag_rows: list[dict], field: str) -> float | None:
    """Mean of *field* across a prompt's model cells, skipping cells where it does not apply.

    ``diagnose_records`` returns None, not 0.0, for a metric with no meaning under
    the prompt's line shape: ``label_exactness`` on ``q3_rewrite``, whose contract asks for one
    prose sentence and so has no line that could be a term. That distinction has to survive the
    aggregation as well as produce it. Averaging the Nones as zero would seat a fully compliant
    prompt at the bottom of the ranking table for obeying its instructions, indistinguishable from
    one that wrote nonsense, the exact confusion ``line_shape`` was introduced to end.
    """
    vals = [d[field] for d in diag_rows if d[field] is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _fmt_opt(x: float | None) -> str:
    """``%.3f`` for a number, ``n/a`` for a metric this prompt's shape does not define."""
    return "n/a" if x is None else f"{x:.3f}"


def _aggregate_one_prompt(cfg, run_dir, prompt_key, report_ids, gold, phrase_index, pb_pred,
                          logger) -> dict | None:
    """Score one prompt: vote sweep + diagnostics. Returns its summary row, or None if unusable."""
    all_records, sent_hpos_by_model, active, missing = _load_prompt_caches(
        cfg, run_dir, prompt_key, report_ids, logger
    )
    if not active:
        logger.warning("[%s] no usable cells — skipped in the ranking", prompt_key)
        return None
    logger.info("[%s] active models (%d): %s", prompt_key, len(active), active)

    cell_dir = prompt_dir(run_dir, prompt_key)
    per_model_report_hpos = derive_patient_hpos(sent_hpos_by_model)
    min_count = int(cfg.min_detection_count)

    summary_rows: list = []
    best = {"rule": None, "micro_f1": -1.0}
    best_pred: dict = {}
    prev_total = None
    for k in range(1, len(active) + 1):
        predicted = vote_k_sets(per_model_report_hpos, active, report_ids, k, min_count)
        metrics, _ = _write_rule(cell_dir, f"vote_k{k}", predicted, gold, report_ids, logger)
        total = sum(len(v) for v in predicted.values())
        # Raising the vote threshold can only remove terms. A violation means an aggregation bug
        # producing plausible-looking numbers, which is what a screen must not rank on.
        if prev_total is not None and total > prev_total:
            raise AssertionError(
                f"[{prompt_key}] vote_k{k} predicted {total} terms > vote_k{k - 1}'s {prev_total} "
                "— the k-of-N sweep must be monotone non-increasing"
            )
        prev_total = total
        summary_rows.append({"rule": f"vote_k{k}", "k": k, "n_predicted": total, **metrics})
        if metrics["micro_f1"] > best["micro_f1"]:
            best = {"rule": f"vote_k{k}", **metrics}
            best_pred = predicted

    plurality, vote_rows = plurality_sets(sent_hpos_by_model, active, report_ids, min_count)
    metrics_plur, _ = _write_rule(cell_dir, "agg_plurality", plurality, gold, report_ids, logger)
    summary_rows.append({
        "rule": "agg_plurality", "k": "",
        "n_predicted": sum(len(v) for v in plurality.values()), **metrics_plur,
    })
    # Plurality competes for "best rule" too, it is a legitimate configuration, and excluding it
    # would let a prompt lose on a rule the ensemble might actually ship with.
    if metrics_plur["micro_f1"] > best["micro_f1"]:
        best = {"rule": "agg_plurality", **metrics_plur}
        best_pred = plurality

    # The Free Listing generation run's artifact names, not new ones: the result-table library's discovery resolves an ensemble cell's
    # files as {variant}_predictions.jsonl / {variant}_detections.jsonl / {variant}_agg_summary.csv,
    # so writing these under the same names is what lets a promoted prompt appear in the comparison
    # table with no reader changes beyond the extra path segment.
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(cell_dir, "slm_ensemble_agg_summary.csv"), index=False
    )

    # The re-run dump: one merged detections file per prompt, so a new aggregation rule can be
    # tried later without the GPUs, the same "score once, re-run the decision logic" invariant
    # The Free Listing generation run and the tree experiments rely on.
    merged_path = os.path.join(cell_dir, "slm_ensemble_detections.jsonl")
    with open(merged_path, "w", encoding="utf-8") as out:
        for model_key in active:
            with open(_detections_path(run_dir, prompt_key, model_key), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)

    # ── Diagnostics: why this prompt scored what it scored ────────────────────
    diag_rows = []
    for model_key in active:
        diag_rows.append({
            "prompt_key": prompt_key, "model": model_key,
            **diagnose_records(all_records[model_key], prompt_key, phrase_index),
        })
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(cell_dir, "prompt_diagnostics.csv"), index=False
    )

    slm_rows = merge_slm_metrics(
        compute_slm_productivity(all_records, sent_hpos_by_model, active), vote_rows
    )
    write_slm_metrics(slm_rows, os.path.join(cell_dir, "slm_metrics.csv"))
    print(f"\n=== {prompt_key} ===")
    print(format_slm_metrics_table(slm_rows))

    yield_row = marginal_gold_yield(best_pred, gold, pb_pred)
    row = {
        "prompt_key": prompt_key,
        "prompt_change": get_prompt(prompt_key).change,
        "n_active_models": len(active),
        "missing_models": ",".join(missing) if missing else "",
        "best_rule": best["rule"],
        "best_micro_f1": round(best["micro_f1"], 4),
        "best_micro_precision": round(best["micro_precision"], 4),
        "best_micro_recall": round(best["micro_recall"], 4),
        "best_macro_f1": round(best["macro_f1"], 4),
        # Means over the cells, so a prompt's diagnostics are comparable across prompts even when
        # one lost a model to a wall-clock kill.
        "echo_rate": _mean_over_cells(diag_rows, "echo_rate"),
        "label_exactness": _mean_over_cells(diag_rows, "label_exactness"),
        "format_compliance": _mean_over_cells(diag_rows, "format_compliance"),
        "negation_trap_rate": _mean_over_cells(diag_rows, "negation_trap_rate"),
        **yield_row,
    }
    logger.info(
        "[%s] best %s µF1=%.4f | echo=%s exact=%s compliance=%s",
        prompt_key, best["rule"], best["micro_f1"],
        _fmt_opt(row["echo_rate"]), _fmt_opt(row["label_exactness"]),
        _fmt_opt(row["format_compliance"]),
    )
    return row


def _execute_aggregate(cfg, exp_id, run_dir, report_ids, gold, logger) -> None:
    prompts = _requested_prompts(cfg)
    logger.info("Aggregating %d prompt(s): %s", len(prompts), prompts)

    # One HPOTree load for the whole sweep, it parses a 19k-node JSON and every prompt needs the
    # same phrase index.
    phrase_index = HPOTree(cfg.hpo_json_path).p_phrase2HPO if cfg.get("hpo_json_path") else \
        HPOTree().p_phrase2HPO
    pb_pred = _load_phenobert_baseline(cfg, report_ids, logger)

    t0 = time.time()
    rows = []
    for prompt_key in prompts:
        row = _aggregate_one_prompt(
            cfg, run_dir, prompt_key, report_ids, gold, phrase_index, pb_pred, logger
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No prompt produced a usable result in {run_dir}. Run stage=extract for each "
            "(prompt, model) cell first, and check its log for a NO DETECTIONS error."
        )

    df = pd.DataFrame(rows).sort_values("best_micro_f1", ascending=False)
    ranking_path = os.path.join(run_dir, "prompt_screen_ranking.csv")
    df.to_csv(ranking_path, index=False)
    logger.info("Ranking → %s", ranking_path)
    print("\n" + df.to_string(index=False))

    duration = time.time() - t0
    with mlflow_run(f"{exp_id}_{cfg.dataset}_aggregate",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "dataset": cfg.dataset, "method": "slm_ensemble",
            "stage": "aggregate", "prompt_keys": ",".join(prompts),
            "n_reports": len(report_ids), "min_detection_count": int(cfg.min_detection_count),
        })
        mlflow.log_metrics({
            "n_reports": len(report_ids), "n_prompts": len(rows), "duration_seconds": duration,
            **{f"{r['prompt_key']}_micro_f1": r["best_micro_f1"] for r in rows},
            # A metric that does not apply to a prompt's line shape is absent from the run, not
            # logged as zero: mlflow has no null, and a fabricated 0.0 here would read as a real
            # measurement in every chart built off this experiment.
            **{f"{r['prompt_key']}_echo_rate": r["echo_rate"]
               for r in rows if r["echo_rate"] is not None},
            **{f"{r['prompt_key']}_label_exactness": r["label_exactness"]
               for r in rows if r["label_exactness"] is not None},
        })
        mlflow.log_artifact(ranking_path)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3, select
# ──────────────────────────────────────────────────────────────────────────────
def _json_none(x):
    """NaN -> None, so the selection verdict stays parseable JSON.

    ``rank_prompts`` reads its rows back from ``prompt_screen_ranking.csv``, and that round-trip
    turns a not-applicable metric from None into a float NaN. ``json.dump`` writes that as a bare
    ``NaN`` token, which Python re-reads happily and every other JSON parser rejects, so the
    verdict file would be valid only for the tool that wrote it.
    """
    return None if x is None or (isinstance(x, float) and x != x) else x


def rank_prompts(rows: list[dict], baseline_key: str, promote_margin: float, n_promote: int
                 ) -> dict:
    """Rank prompts and decide whether the winner earns a full-cohort run.

    Pure: it takes the ranking rows and returns the verdict, so the threshold is testable without
    a cluster. The gate is a **noise floor, not a preference**. The screen is 20 reports. Phase B is
    32 array tasks at up to 12 h each across two cohorts. A margin that would not survive scaling to
    118 and 228 reports is not worth those hours, so the verdict is False unless the winner clears
    the baseline by ``promote_margin`` micro-F1 points.

    A missing baseline row is not treated as a zero, that would promote on any result at all.
    ``baseline_micro_f1`` comes back None and ``promote`` is False, with the reason recorded.
    """
    ranked = sorted(rows, key=lambda r: r["best_micro_f1"], reverse=True)
    baseline = next((r for r in ranked if r["prompt_key"] == baseline_key), None)

    if baseline is None:
        return {
            "promote": False,
            "reason": (
                f"baseline prompt {baseline_key!r} has no result — nothing to measure a margin "
                "against. Re-run its cells, or set promote_margin=0 to promote unconditionally."
            ),
            "baseline_key": baseline_key,
            "baseline_micro_f1": None,
            "promote_margin": promote_margin,
            "best_margin": None,
            "selected": [],
            "ranking": [
                {"prompt_key": r["prompt_key"], "best_micro_f1": r["best_micro_f1"],
                 "best_rule": r["best_rule"]} for r in ranked
            ],
        }

    base_f1 = baseline["best_micro_f1"]
    contenders = [r for r in ranked if r["prompt_key"] != baseline_key]
    best_margin = round(contenders[0]["best_micro_f1"] - base_f1, 4) if contenders else None
    promote = best_margin is not None and best_margin >= promote_margin
    # Empty unless the gate opened. The cluster script checks `promote` first, but a selection
    # left populated on a closed gate is a loaded gun: any later reader that takes `selected` at
    # face value would submit 32 GPU tasks for prompts the screen explicitly declined.
    selected = contenders[:n_promote] if promote else []

    if promote:
        reason = (
            f"{contenders[0]['prompt_key']} beats {baseline_key} by {best_margin:+.4f} micro-F1 "
            f"(>= {promote_margin:g})"
        )
    elif best_margin is None:
        reason = "no non-baseline prompt produced a result"
    else:
        reason = (
            f"best margin {best_margin:+.4f} over {baseline_key} is below the {promote_margin:g} "
            "threshold — too small to survive the full cohort, not promoting"
        )

    return {
        "promote": promote,
        "reason": reason,
        "baseline_key": baseline_key,
        "baseline_micro_f1": base_f1,
        "promote_margin": promote_margin,
        "best_margin": best_margin,
        # Only meaningful when promote is True. The cluster script gates on `promote` before
        # reading this, so a non-promoting run cannot leak a selection into an sbatch call.
        "selected": [
            {"prompt_key": r["prompt_key"], "best_micro_f1": r["best_micro_f1"],
             "best_rule": r["best_rule"],
             "margin": round(r["best_micro_f1"] - base_f1, 4)}
            for r in selected
        ],
        "ranking": [
            {"prompt_key": r["prompt_key"], "best_micro_f1": r["best_micro_f1"],
             "best_rule": r["best_rule"], "echo_rate": _json_none(r.get("echo_rate")),
             "label_exactness": _json_none(r.get("label_exactness"))} for r in ranked
        ],
    }


def _execute_select(cfg, exp_id, run_dir, logger) -> None:
    ranking_path = os.path.join(run_dir, "prompt_screen_ranking.csv")
    if not os.path.isfile(ranking_path):
        raise FileNotFoundError(
            f"{ranking_path} not found — run stage=aggregate before stage=select."
        )
    rows = pd.read_csv(ranking_path).to_dict("records")
    verdict = rank_prompts(
        rows,
        baseline_key=str(cfg.get("baseline_prompt_key") or BASELINE_KEY),
        promote_margin=float(cfg.get("promote_margin", 0.05)),
        n_promote=int(cfg.get("n_promote", 2)),
    )

    out_path = os.path.join(run_dir, "prompt_screen_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    logger.info("Selection verdict: promote=%s — %s", verdict["promote"], verdict["reason"])
    for entry in verdict["ranking"]:
        logger.info("  %-16s µF1=%.4f (%s)",
                    entry["prompt_key"], entry["best_micro_f1"], entry["best_rule"])
    logger.info("Selection → %s", out_path)

    with mlflow_run(f"{exp_id}_{cfg.dataset}_select",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "dataset": cfg.dataset, "stage": "select",
            "baseline_prompt_key": verdict["baseline_key"],
            "promote_margin": verdict["promote_margin"],
            "promote": verdict["promote"],
            "selected": ",".join(s["prompt_key"] for s in verdict["selected"]) or "none",
        })
        if verdict["best_margin"] is not None:
            mlflow.log_metric("best_margin", verdict["best_margin"])
        mlflow.log_artifact(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def execute(cfg, exp_id: str, gold_source=None) -> None:
    """Run one stage of a prompt screen.

    ``gold_source`` is the extension point an earlier exploratory run uses, and the same shape
    :func:`hpo_extraction.treephenorag.score_store.execute` gives ``build_scores``: an optional
    ``(cfg, reports, gt_dict, run_dir, logger) -> (reports, gt_dict)`` called after the dataset
    load and before cohort selection. It exists so an experiment can swap in a *different ground truth
    set*, an earlier exploratory run's curated HCY ground truth, without this module learning anything about curation,
    and without an earlier exploratory run changing at all: passing nothing keeps the loader's own ground truth.

    A hook that narrows ``reports`` narrows the cohort with it, which is the mechanism that
    restricts an earlier exploratory run to the reports curation has actually reached. It runs for ``extract`` and
    ``aggregate`` alike, both stages must see the same ground truth, or the cells the array generated and
    the ground truth the ranking scores them against would come from different report sets.
    """
    run_dir = os.path.join(cfg.output_dir, exp_id, cfg.dataset)
    os.makedirs(run_dir, exist_ok=True)

    stage = str(cfg.get("stage") or "extract")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} (expected one of {STAGES})")

    # One log file per cell: eight array tasks interleaving into one run.log made it impossible to
    # tell which task a message came from, and which task had stopped.
    suffix = ""
    prompt_key = model_key = None
    if stage == "extract":
        prompt_key, model_key = _resolve_cell(cfg)
        suffix = f"_{prompt_key}_{model_key}"
    logger = _setup_logging(run_dir, suffix)
    logger.info("Starting %s | dataset=%s stage=%s | out=%s", exp_id, cfg.dataset, stage, run_dir)

    if stage == "select":
        # No dataset load: select reads the ranking CSV the aggregate stage already wrote.
        _execute_select(cfg, exp_id, run_dir, logger)
        return

    reports, gt_dict = _load_dataset(cfg, logger)
    if gold_source is not None:
        reports, gt_dict = gold_source(cfg, reports, gt_dict, run_dir, logger)
        logger.info("Gold replaced by %s | %d report(s) | %d with gold terms",
                    getattr(gold_source, "__name__", gold_source),
                    len(reports), sum(1 for v in gt_dict.values() if v))
    report_ids = select_cohort(cfg, reports, gt_dict, run_dir, logger)

    if stage == "extract":
        sent_dict = _segment_reports(cfg, reports, report_ids, logger)
        _execute_extract(
            cfg, exp_id, run_dir, prompt_key, model_key, sent_dict, report_ids, logger
        )
    else:
        gold = {rid: set(gt_dict.get(rid, [])) for rid in report_ids}
        _execute_aggregate(cfg, exp_id, run_dir, report_ids, gold, logger)


def _resolve_cell(cfg) -> tuple[str, str]:
    """The (prompt, model) cell this extract task owns, from the array index or explicit keys.

    Validated before any expensive setup: a bad key must fail in seconds, not after the dataset
    load and a Stanza pipeline build.
    """
    prompts = _requested_prompts(cfg)
    models = [k for k, _ in _configured_models(cfg)]
    if not models:
        raise ValueError("no slm_dirs.* path is set — nothing to load")

    if cfg.get("array_index") is not None:
        # The models list is filtered to what is configured, so a partial roster still decodes to
        # a contiguous index range, not silently skipping cells.
        prompt_key, model_key = decode_array_index(int(cfg.array_index), prompts, models)
    else:
        prompt_key = cfg.get("prompt_key")
        model_key = cfg.get("model_key")
        if not prompt_key or not model_key:
            raise ValueError(
                "stage=extract needs either array_index=<int>, or both prompt_key=<key> and "
                f"model_key=<key>. Prompts: {prompts}. Models: {models}."
            )
    get_prompt(prompt_key)
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r} (expected one of {MODEL_KEYS})")
    if not (cfg.get("slm_dirs") or {}).get(model_key):
        raise ValueError(f"slm_dirs.{model_key} is not set — nothing to load")
    return str(prompt_key), str(model_key)
