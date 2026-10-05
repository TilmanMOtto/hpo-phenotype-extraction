"""
PhenoRAG glue for the RAG-HPO benchmark (parallels hpo_extraction.phenojury.phenobert).

RAG-HPO's own logic lives verbatim in ``third_party/RAG-HPO/rag_hpo_lib.py``. Its only external
dependency is a client exposing ``query(user_input, system_message) -> str``. This module
supplies that client backed by a local HuggingFace LLaMA (``LocalLlamaClient``) and a
non-interactive orchestrator (``run_rag_hpo``) that reproduces the notebook's ``main()``
extract → retrieve → map flow without any ``input()`` prompts or pickled models.

``run_rag_hpo`` returns ``{patient_id: set(hpo_id)}`` plus a flat list of per-phrase
"finding records" (chosen HPO + ranked retrieval candidates) used to emit the dashboard's
``*_retrieved_segments.jsonl`` artifact.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# rag_hpo_lib lives in the (hyphenated, non-package) src/RAG-HPO/ directory next to src/core.
from hpo_extraction.paths import RAGHPO_DIR as _RAGHPO_DIR
if str(_RAGHPO_DIR) not in sys.path:
    sys.path.insert(0, str(_RAGHPO_DIR))

import rag_hpo_lib as rag  # noqa: E402

logger = logging.getLogger(__name__)


class LocalLlamaClient:
    """
    Drop-in replacement for RAG-HPO's cloud ``LLMClient``, backed by a local LLaMA.

    Exposes the single method RAG-HPO calls, ``query(user_input, system_message)``, and
    delegates to a ``hpo_extraction.models.llama.LlamaLLM``. Decoding is greedy unless
    ``temperature`` is given: upstream always sends a temperature to the API (class default 0.7,
    interactive default 0.2), so replicating the published run needs sampling, while the RAG-HPO 8B baseline
    / the RAG-HPO 70B baseline runs keep the deterministic default. The API client's rate-limiting sleep and
    tiktoken token accounting are intentionally dropped (irrelevant for local inference).
    """

    def __init__(self, llm, max_new_tokens: int = 1024, temperature: float | None = None):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_calls = 0              # read into rag_hpo_timing.jsonl as the method's LLM cost

    def query(self, user_input: str, system_message: str) -> str:
        """Answer one RAG-HPO request with the local model.

        Args:
            user_input: the user message RAG-HPO built.
            system_message: the system prompt RAG-HPO built.

        Returns:
            The generated reply. Decoding is greedy unless a temperature was configured.
        """
        self.n_calls += 1
        kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature:
            kwargs["temperature"] = float(self.temperature)
        return self.llm.generate(user_input, system_message, **kwargs)


def write_timing(run_output_dir: str, report_ids, patient_hpos: dict, n_llm_calls: int,
                 load_s: float, duration: float, peak_mem_bytes: int) -> str:
    """Write ``rag_hpo_timing.jsonl``, the record the comparison's cost table reads.

    ``duration_s`` is inference only, model and vector-index load is ``model_load_s``, the same
    split AutoPCR's timing file makes, so the two rows compare like for like. Called before the
    MLflow block on purpose: a tracking-store failure must not take the measurement with it.
    """
    path = os.path.join(run_output_dir, "rag_hpo_timing.jsonl")
    n = len(report_ids)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "stage": "annotate", "n_reports": n,
            "n_predictions": sum(len(patient_hpos.get(r, ())) for r in report_ids),
            "n_llm_calls": n_llm_calls,
            "model_load_s": round(load_s, 3), "duration_s": round(duration, 3),
            "mean_time_per_report_s": duration / n if n else 0.0,
            "peak_gpu_gb": round(peak_mem_bytes / 1024 ** 3, 2),
        }) + "\n")
    return path


def _normalize_hpo(hp) -> str | None:
    """Return a canonical ``HP:#######`` id, or None for non-HPO values ('No Candidate Fit', NaN)."""
    if not isinstance(hp, str):
        return None
    hp = hp.replace("HP:HP:", "HP:").strip()
    return hp if hp.startswith("HP:") else None


def run_rag_hpo(
    df: pd.DataFrame,
    emb_model,
    index,
    docs,
    cluster_index,
    system_message_I: str,
    system_message_II: str,
    keep_top: int = 15,
    logger: logging.Logger = logger,
) -> tuple[dict[str, set[str]], list[dict]]:
    """
    Run the full RAG-HPO pipeline over a DataFrame of clinical notes.

    Reproduces notebook ``main()`` steps 5–8 non-interactively:
      5. per patient, LLM phrase extraction (system_message_I) + SapBERT/FAISS retrieval;
      6. local fuzzy HPO mapping (``extract_hpo_term``) → split exact / non-exact;
      7. LLM candidate selection (system_message_II) for the non-exact remainder;
      8. collapse to one HPO set per patient.

    Args:
        df: must have columns ``clinical_note`` and ``patient_id`` (string ids matching the
            evaluation ground-truth keys). ``rag.llm_client`` must be set before calling.
        emb_model / index / docs: SapBERT model, FAISS index, and doc list from
            ``rag.load_vector_db`` / ``rag.create_faiss_index``.
        cluster_index: ``rag.build_cluster_index(docs)`` for the local fuzzy fallback.
        keep_top: number of retrieval candidates per phrase.

    Returns:
        (patient_hpos, finding_records) where
        ``patient_hpos = {patient_id: set(HP:#######)}`` (every input patient present, even
        if empty) and ``finding_records`` is a flat list of
        ``{patient_id, phrase, original_sentence, chosen_hpo, candidates:[{hp_id, info, similarity}]}``.
    """
    if rag.llm_client is None:
        raise RuntimeError(
            "rag_hpo_lib.llm_client is not set — assign a LocalLlamaClient before run_rag_hpo()."
        )

    patient_ids = [str(p) for p in df["patient_id"].tolist()]
    patient_hpos: dict[str, set[str]] = {pid: set() for pid in patient_ids}

    # ── Step 5: extract phenotype phrases + retrieve candidates, per patient ──────
    combined = pd.DataFrame()
    logger.info("RAG-HPO: extracting + retrieving over %d patients …", len(patient_ids))
    for pid in tqdm(patient_ids, desc="Extract+Retrieve", unit="note"):
        note = df.loc[df["patient_id"].astype(str) == pid, "clinical_note"].iloc[0]
        res = rag.process_row(note, system_message_I, emb_model, index, docs)
        if not res.empty:
            res["patient_id"] = pid
            combined = pd.concat([combined, res], ignore_index=True)

    if combined.empty:
        logger.warning("RAG-HPO: no phenotype findings extracted for any patient.")
        return patient_hpos, []

    # ── Step 6: local fuzzy HPO mapping, then split exact vs non-exact ────────────
    if "HPO_Term" not in combined.columns:
        combined["HPO_Term"] = np.nan
    combined["HPO_Term"] = (
        combined.apply(
            lambda r: rag.extract_hpo_term(r["phrase"], r["unique_metadata"], cluster_index)
            if pd.isna(r["HPO_Term"]) else r["HPO_Term"],
            axis=1,
        )
        .astype(object)
        .where(lambda x: pd.notna(x), np.nan)
    )
    exact_df, non_exact_df = rag.split_exact_nonexact(combined, hpo_term_col="HPO_Term")
    logger.info(
        "RAG-HPO: %d findings — %d exact (local match), %d non-exact (LLM mapping)",
        len(combined), len(exact_df), len(non_exact_df),
    )

    # ── Step 7: LLM candidate selection for the non-exact remainder ───────────────
    non_ex = non_exact_df.copy()
    for col in ("llm_parse_reason", "raw_llm_resp"):
        non_ex[col] = non_ex.get(col, pd.Series(dtype="object")).astype("object")
    idxs = non_ex[
        (non_ex["category"] == "Abnormal") & (non_ex["HPO_Term"].isna())
    ].index
    for idx in tqdm(idxs, desc="LLM HPO mapping", unit="phrase"):
        row_df = non_ex.loc[[idx]]
        out_df = rag.generate_hpo_terms(row_df, system_message_II)
        hp = out_df.at[0, "HPO_Terms"][0]["HPO_Term"] if not out_df.empty else None
        if "llm_parse_reason" in out_df.columns:
            non_ex.at[idx, "llm_parse_reason"] = out_df.at[0, "llm_parse_reason"]
        if "raw_llm_resp" in out_df.columns:
            non_ex.at[idx, "raw_llm_resp"] = out_df.at[0, "raw_llm_resp"]
        non_ex.at[idx, "HPO_Term"] = hp or "No Candidate Fit"

    # ── Step 8: collapse to one HPO set per patient + build finding records ───────
    merged = pd.concat([exact_df, non_ex], ignore_index=True)
    finding_records: list[dict] = []
    for _, row in merged.iterrows():
        pid = str(row.get("patient_id"))
        chosen = _normalize_hpo(row.get("HPO_Term"))
        if chosen:
            patient_hpos.setdefault(pid, set()).add(chosen)
        candidates = []
        for m in (row.get("unique_metadata") or []):
            candidates.append({
                "hp_id": m.get("hp_id"),
                "info": m.get("phrase"),
                "similarity": m.get("similarity"),
            })
        finding_records.append({
            "patient_id": pid,
            "phrase": row.get("phrase"),
            "original_sentence": row.get("original_sentence"),
            "chosen_hpo": chosen,
            "candidates": candidates,
        })

    n_pos = sum(len(v) for v in patient_hpos.values())
    logger.info("RAG-HPO: %d predicted HPOs across %d patients", n_pos, len(patient_hpos))
    return patient_hpos, finding_records
