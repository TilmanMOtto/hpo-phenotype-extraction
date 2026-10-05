"""PhenoRAG glue for the **published** RAG-HPO pipeline (parallels ``hpo_extraction.baselines.rag_hpo_runner``).

``rag_hpo_runner`` drives ``rag_hpo_lib``, upstream's current code. This module drives
``rag_hpo_lib_paper``, upstream at commit ``25c1ea7``, the revision behind Garcia et al. 2025's
tables, and reproduces that notebook's ``__main__`` block non-interactively: extract → retrieve →
local shortcut → LLM mapping → one HPO set per patient, minus the pickled models, the
``input()`` prompts and the Groq rate-limit sleeps.

The step order counts and is theirs: ``extract_hpo_term`` runs *before* the mapping call and
mutates each row's candidate list in place, so the prompt sees whatever that left behind.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# rag_hpo_lib_paper lives in the (hyphenated, non-package) src/RAG-HPO/ directory next to src/core.
from hpo_extraction.paths import RAGHPO_DIR as _RAGHPO_DIR
if str(_RAGHPO_DIR) not in sys.path:
    sys.path.insert(0, str(_RAGHPO_DIR))

import rag_hpo_lib_paper as rag  # noqa: E402

logger = logging.getLogger(__name__)

_HPO_RE = re.compile(r"HP:\d+")


def _hpo_ids(value) -> list[str]:
    """Every HPO id in a cell of ``HPO_Term``.

    The mapping stage stores a comma-joined list, so a cell can hold several ids, while the local
    shortcut stores a single bare id. Their ``process_results`` repairs a doubled prefix, which is
    done here too. A cell that holds no id ("No HPO terms found") contributes nothing.
    """
    if not isinstance(value, str):
        return []
    return _HPO_RE.findall(value.replace("HP:HP:", "HP:"))


def run_rag_hpo_paper(
    df: pd.DataFrame,
    emb_model,
    index,
    docs: list[dict[str, str]],
    system_message_I: str,
    system_message_II: str,
    keep_top: int = rag.KEEP_TOP,
    faiss_depth: int = rag.FAISS_DEPTH,
    logger: logging.Logger = logger,
) -> tuple[dict[str, set[str]], list[dict]]:
    """Run the published RAG-HPO pipeline over a DataFrame of clinical notes.

    Args:
        df: columns ``patient_id`` (string ids matching the evaluation ground truth) and
            ``clinical_note``. ``rag_hpo_lib_paper.llm_client`` must be set before calling.
        emb_model / index / docs: BGE-small model, ``IndexFlatL2``, and the row list from
            ``rag.load_vector_db``.
        keep_top / faiss_depth: their 20 candidates out of a search depth of 800.

    Returns:
        ``({patient_id: {HP:#######}}, finding_records)``, every input patient is a key, even
        when it predicted nothing, and ``finding_records`` carries one entry per extracted phrase
        with its ranked candidates, for the ``*_retrieved_segments.jsonl`` artifact.
    """
    if rag.llm_client is None:
        raise RuntimeError(
            "rag_hpo_lib_paper.llm_client is not set — assign a client before run_rag_hpo_paper()."
        )

    patient_ids = [str(p) for p in df["patient_id"].tolist()]
    patient_hpos: dict[str, set[str]] = {pid: set() for pid in patient_ids}

    # ── Extract phrases + retrieve candidates, per patient ────────────────────────
    combined = pd.DataFrame()
    logger.info("RAG-HPO (paper): extracting + retrieving over %d patients …", len(patient_ids))
    for pid in tqdm(patient_ids, desc="Extract+Retrieve", unit="note"):
        note = df.loc[df["patient_id"].astype(str) == pid, "clinical_note"].iloc[0]
        res = rag.process_row(note, system_message_I, emb_model, index, docs,
                              keep_top=keep_top, faiss_depth=faiss_depth)
        if res is not None and not res.empty:
            res["patient_id"] = pid
            combined = pd.concat([combined, res], ignore_index=True)

    if combined.empty:
        logger.warning("RAG-HPO (paper): no phenotype findings extracted for any patient.")
        return patient_hpos, []

    # ── Their normalisation, then the local shortcut ──────────────────────────────
    combined["phrase"] = combined["phrase"].str.lower()
    combined["unique_metadata"] = combined["unique_metadata"].apply(rag.process_unique_metadata)
    combined["HPO_Term"] = combined.apply(
        lambda row: rag.extract_hpo_term(row["phrase"], row["unique_metadata"]), axis=1
    )

    exact_df = combined.dropna(subset=["HPO_Term"])
    non_exact_df = combined[combined["HPO_Term"].isna()].copy()
    logger.info(
        "RAG-HPO (paper): %d findings — %d resolved locally by the exact-match shortcut, "
        "%d sent to the LLM mapper",
        len(combined), len(exact_df), len(non_exact_df),
    )

    # ── LLM mapping for everything the shortcut left ──────────────────────────────
    non_exact_df["raw_llm_resp"] = pd.Series(dtype="object")
    for idx in tqdm(non_exact_df.index, desc="LLM HPO mapping", unit="phrase"):
        out_df = rag.generate_hpo_terms(pd.DataFrame([non_exact_df.loc[idx]]), system_message_II)
        non_exact_df.at[idx, "HPO_Term"] = out_df["response"].iloc[0]
        non_exact_df.at[idx, "raw_llm_resp"] = out_df["raw_llm_resp"].iloc[0]

    # ── Collapse to one set per patient + build the segment records ───────────────
    merged = pd.concat([exact_df, non_exact_df], ignore_index=True)
    finding_records: list[dict] = []
    for _, row in merged.iterrows():
        pid = str(row.get("patient_id"))
        chosen = _hpo_ids(row.get("HPO_Term"))
        patient_hpos.setdefault(pid, set()).update(chosen)

        metadata = row.get("unique_metadata") or []
        similarities = row.get("similarities")
        if not isinstance(similarities, (list, np.ndarray)):
            similarities = []
        # ``extract_hpo_term`` extends ``unique_metadata`` in place with copies of its fuzzy and
        # substring matches, so the list is longer here than what FAISS returned. One similarity
        # was recorded per retrieved row, so that count is the honest cut: without it the copies
        # would enter the retrieval artifact as extra candidates with no score.
        retrieved = metadata[:len(similarities)] if len(similarities) else metadata
        candidates = []
        for rank, item in enumerate(retrieved):
            pair = rag.candidate_pair(item)
            if pair is None:
                continue
            info, hp_id = pair
            candidates.append({
                "hp_id": hp_id,
                "info": info,
                "similarity": float(similarities[rank]) if rank < len(similarities) else None,
            })

        finding_records.append({
            "patient_id": pid,
            "phrase": row.get("phrase"),
            "original_sentence": row.get("original_sentence"),
            "chosen_hpo": chosen,
            "candidates": candidates,
        })

    n_pos = sum(len(v) for v in patient_hpos.values())
    logger.info("RAG-HPO (paper): %d predicted HPOs across %d patients", n_pos, len(patient_hpos))
    return patient_hpos, finding_records
