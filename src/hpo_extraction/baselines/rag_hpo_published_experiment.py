"""Driver for the faithful RAG-HPO replication (the RAG-HPO reproduction with the published code).

``hpo_extraction.baselines.rag_hpo_experiment`` runs upstream's *current* code (the RAG-HPO 8B baseline / the RAG-HPO 70B baseline). This driver runs
upstream at commit ``25c1ea7`` with the 2024-12-19 prompts and the paper's vector DB recipe, the
configuration behind Garcia et al. 2025, Genome Medicine 17:91, so that the published table has a
like-for-like row in our comparison.

It writes the same two artifacts under the same names as every other earlier method, so the result-table library
scores it with no special case: ``rag_hpo_predictions.jsonl`` (the full predicted set per report,
plus a summary line per report) and ``rag_hpo_retrieved_segments.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch


from hpo_extraction.data.loading import load_cohort
from hpo_extraction.baselines.rag_hpo_published_runner import run_rag_hpo_paper
from hpo_extraction.models.gpu_memory import peak_gpu_bytes_all_devices, reset_peak_all_devices
from hpo_extraction.baselines.rag_hpo_runner import LocalLlamaClient, write_timing
from hpo_extraction.models.llama import LlamaLLM, load_llama
from hpo_extraction.ontology.hpo_tree import HPOTree, hpo_label

import rag_hpo_lib_paper as rag  # importable via rag_hpo_paper_runner (adds src/RAG-HPO to path)
from hpo_extraction.utils.mlflow_guard import mlflow_run


def _setup_logging(output_dir: str) -> logging.Logger:
    log_path = os.path.join(output_dir, "run.log")
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("exp13_raghpo_paper")


def execute(cfg, exp_id: str) -> None:
    """Run RAG-HPO from the code revision behind its published tables, on one cohort.

    Args:
        cfg: Hydra config of one reproduction run (configs/experiments/06_comparison/raghpo_reproduction_*.yaml).
        exp_id: name of the result folder, created as ``<output_dir>/<exp_id>/<dataset>/``.
    """
    run_output_dir = os.path.join(cfg.output_dir, exp_id, cfg.dataset)
    os.makedirs(run_output_dir, exist_ok=True)
    logger = _setup_logging(run_output_dir)
    logger.info("Starting %s | dataset=%s llama_dir=%s 4bit=%s temp=%s | out=%s",
                exp_id, cfg.dataset, cfg.llama_dir, cfg.load_in_4bit, cfg.temperature,
                run_output_dir)

    reports, gt_dict = load_cohort(cfg, logger)
    report_ids = sorted(reports)
    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("PILOT: restricted to %d report(s)", len(report_ids))
    df = pd.DataFrame({
        "patient_id": report_ids,
        "clinical_note": [reports[r] for r in report_ids],
    })

    # The published prompts (2024-12-19), not the ones vendored as system_prompts.json.
    prompts_path = os.path.join(cfg.rag_hpo_dir, cfg.prompts_file)
    with open(prompts_path, encoding="utf-8") as f:
        prompts = json.load(f)
    logger.info("Prompts: %s", prompts_path)

    # Their runs sampled (the API is always called with a temperature). Seed what that makes
    # non-deterministic so a rerun of this experiment reproduces itself.
    torch.manual_seed(int(cfg.seed))
    t_load = time.time()
    tokenizer, model = load_llama(cfg.llama_dir, load_in_4bit=bool(cfg.load_in_4bit))
    rag.llm_client = LocalLlamaClient(
        LlamaLLM(model, tokenizer),
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )

    # Their retrieval stack: BGE-small, L2-normalised, IndexFlatL2, depth 800 → 20 candidates.
    emb_model = rag.initialize_embeddings_model(cfg.embed_model)
    meta_path = os.path.join(cfg.rag_hpo_dir, cfg.meta_file)
    docs, emb_matrix = rag.load_vector_db(
        meta_path=meta_path,
        vec_path=os.path.join(cfg.rag_hpo_dir, cfg.vec_file),
    )
    index = rag.create_faiss_index(emb_matrix)
    with open(meta_path, encoding="utf-8") as f:
        db_source = json.load(f).get("source", {})
    logger.info("Vector DB: %d rows, dim=%d, source=%s", len(docs), emb_matrix.shape[1], db_source)
    load_s = time.time() - t_load

    reset_peak_all_devices()
    t0 = time.time()
    patient_hpos, finding_records = run_rag_hpo_paper(
        df=df, emb_model=emb_model, index=index, docs=docs,
        system_message_I=prompts["system_message_I"],
        system_message_II=prompts["system_message_II"],
        keep_top=cfg.keep_top, faiss_depth=cfg.faiss_depth, logger=logger,
    )
    duration = time.time() - t0
    peak_mem = peak_gpu_bytes_all_devices()
    timing_path = write_timing(run_output_dir, report_ids, patient_hpos, rag.llm_client.n_calls,
                               load_s, duration, peak_mem)

    hpo_tree = HPOTree()
    with mlflow_run(f"{exp_id}_{cfg.dataset}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "method": "rag_hpo_paper", "dataset": cfg.dataset,
            "llama_dir": cfg.llama_dir, "load_in_4bit": bool(cfg.load_in_4bit),
            "max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temperature,
            "seed": cfg.seed, "embed_model": cfg.embed_model, "keep_top": cfg.keep_top,
            "faiss_depth": cfg.faiss_depth, "prompts_file": cfg.prompts_file,
            "hpo_release": db_source.get("hpo_release"), "n_reports": len(report_ids),
            "upstream_revision": "25c1ea7 (2025-04-26) + prompts 6b00779 (2024-12-19)",
        })

        # Full predicted set per report + ground truth, the Step-2 substrate.
        preds_path = os.path.join(run_output_dir, "rag_hpo_predictions.jsonl")
        n_pred = 0
        with open(preds_path, "w", encoding="utf-8") as f:
            for rid in report_ids:
                gold = set(gt_dict.get(rid, []))
                predicted = sorted(patient_hpos.get(rid, set()))
                n_pred += len(predicted)
                for code in predicted:
                    f.write(json.dumps({
                        "report_id": rid, "hpo_id": code, "hpo_label": hpo_label(hpo_tree, code),
                        "prediction": 1, "ground_truth": int(code in gold),
                    }) + "\n")
                f.write(json.dumps({
                    "report_id": rid, "summary": True,
                    "predicted_set": predicted, "gold_set": sorted(gold),
                }) + "\n")
        mlflow.log_artifact(preds_path)

        # Retrieved candidates per extracted phrase (retrieval-quality substrate).
        seg_path = os.path.join(run_output_dir, "rag_hpo_retrieved_segments.jsonl")
        with open(seg_path, "w", encoding="utf-8") as f:
            for rec in finding_records:
                chosen = set(rec["chosen_hpo"])
                for rank, cand in enumerate(rec["candidates"], start=1):
                    hpo_id = cand.get("hp_id")
                    f.write(json.dumps({
                        "report_id": rec["patient_id"], "hpo_id": hpo_id,
                        "hpo_label": cand.get("info"), "rank": rank,
                        "text": rec.get("original_sentence") or rec.get("phrase") or "",
                        "cosine_sim": cand.get("similarity"),
                        "slm_verdict": "Yes" if hpo_id in chosen else "No",
                    }) + "\n")
        mlflow.log_artifact(seg_path)
        mlflow.log_artifact(timing_path)

        mlflow.log_metrics({
            "n_reports": len(report_ids), "n_predictions": n_pred,
            "duration_seconds": duration,
            "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
            "peak_gpu_mem_bytes": peak_mem,
        })
        logger.info("Done | %d reports | %d predicted terms | %.1fs (%.2fs/report) | peak %.2f GB",
                    len(report_ids), n_pred, duration,
                    duration / max(len(report_ids), 1), peak_mem / 1024 ** 3)
