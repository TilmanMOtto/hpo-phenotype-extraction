"""Shared driver for the earlier runs RAG-HPO comparison runs (the RAG-HPO 8B baseline / the RAG-HPO 70B baseline).

RAG-HPO (Garcia et al., Genome Medicine 2025) is the large-LLM retrieval baseline the thesis
positions against. earlier already runs the method with a local LLaMA on HCY, but restricts scoring to
the 9-target projection. For the earlier comparison the method must instead be measured at **ontology
scale on both HCY and GSC+**, emitting the *full* predicted HPO set per report (not a target
projection) plus segments and timing, so the same flat / hierarchy / error-taxonomy metrics apply
to it as to TreePhenoRAG in Step 2.

The RAG-HPO 8B baseline (8B) and the RAG-HPO 70B baseline (70B) differ only by ``llama_dir`` / ``load_in_4bit``. Both call
:func:`execute`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import mlflow
import pandas as pd


from hpo_extraction.data.loading import load_cohort
from hpo_extraction.models.gpu_memory import peak_gpu_bytes_all_devices, reset_peak_all_devices
from hpo_extraction.baselines.rag_hpo_runner import LocalLlamaClient, run_rag_hpo, write_timing
from hpo_extraction.models.llama import LlamaLLM, load_llama
from hpo_extraction.ontology.hpo_tree import HPOTree, hpo_label

import rag_hpo_lib as rag  # importable via rag_hpo_runner (adds src/RAG-HPO to sys.path)
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
    return logging.getLogger("exp13_raghpo")


def execute(cfg, exp_id: str) -> None:
    """Run the RAG-HPO baseline on one cohort and write its predictions.

    Args:
        cfg: Hydra config of a RAG-HPO baseline (configs/experiments/06_comparison/baseline_raghpo_*.yaml).
        exp_id: name of the result folder, created as ``<output_dir>/<exp_id>/<dataset>/``.
    """
    run_output_dir = os.path.join(cfg.output_dir, exp_id, cfg.dataset)
    os.makedirs(run_output_dir, exist_ok=True)
    logger = _setup_logging(run_output_dir)
    logger.info("Starting %s | dataset=%s llama_dir=%s 4bit=%s | out=%s",
                exp_id, cfg.dataset, cfg.llama_dir, cfg.load_in_4bit, run_output_dir)

    reports, gt_dict = load_cohort(cfg, logger)
    report_ids = sorted(reports)
    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("PILOT: restricted to %d report(s)", len(report_ids))
    df = pd.DataFrame({
        "patient_id": report_ids,
        "clinical_note": [reports[r] for r in report_ids],
    })

    # RAG-HPO prompts + local LLaMA client
    with open(os.path.join(cfg.rag_hpo_dir, "system_prompts.json")) as f:
        prompts = json.load(f)
    t_load = time.time()
    tokenizer, model = load_llama(cfg.llama_dir, load_in_4bit=bool(cfg.load_in_4bit))
    rag.llm_client = LocalLlamaClient(LlamaLLM(model, tokenizer), max_new_tokens=cfg.max_new_tokens)

    # RAG-HPO's own SapBERT + FAISS retrieval (unchanged)
    emb_model = rag.initialize_embeddings_model(use_sbert=True, sbert_model=cfg.sbert_model)
    docs, emb_matrix = rag.load_vector_db(
        meta_path=os.path.join(cfg.rag_hpo_dir, "hpo_meta.json"),
        vec_path=os.path.join(cfg.rag_hpo_dir, "hpo_embedded.npz"),
    )
    index = rag.create_faiss_index(emb_matrix, metric="cosine")
    cluster_index = rag.build_cluster_index(docs)
    logger.info("Vector DB: %d docs, dim=%d", len(docs), emb_matrix.shape[1])
    load_s = time.time() - t_load

    reset_peak_all_devices()
    t0 = time.time()
    patient_hpos, finding_records = run_rag_hpo(
        df=df, emb_model=emb_model, index=index, docs=docs, cluster_index=cluster_index,
        system_message_I=prompts["system_message_I"], system_message_II=prompts["system_message_II"],
        keep_top=cfg.keep_top, logger=logger,
    )
    duration = time.time() - t0
    peak_mem = peak_gpu_bytes_all_devices()
    timing_path = write_timing(run_output_dir, report_ids, patient_hpos, rag.llm_client.n_calls,
                               load_s, duration, peak_mem)

    hpo_tree = HPOTree()
    with mlflow_run(f"{exp_id}_{cfg.dataset}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "method": "rag_hpo", "dataset": cfg.dataset,
            "llama_dir": cfg.llama_dir, "load_in_4bit": bool(cfg.load_in_4bit),
            "max_new_tokens": cfg.max_new_tokens, "sbert_model": cfg.sbert_model,
            "keep_top": cfg.keep_top, "n_reports": len(report_ids),
        })

        # Full predicted set per report + ground truth (annotated), the Step-2 substrate.
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
                cands = sorted(rec["candidates"], key=lambda c: (c.get("similarity") or 0.0),
                               reverse=True)
                for rank, cand in enumerate(cands, start=1):
                    hpo_id = cand.get("hp_id")
                    f.write(json.dumps({
                        "report_id": rec["patient_id"], "hpo_id": hpo_id,
                        "hpo_label": cand.get("info"), "rank": rank,
                        "text": rec.get("original_sentence") or rec.get("phrase") or "",
                        "cosine_sim": cand.get("similarity"),
                        "slm_verdict": "Yes" if hpo_id and hpo_id == rec.get("chosen_hpo") else "No",
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
