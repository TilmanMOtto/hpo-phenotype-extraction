"""The candidate chain of the *adapted* RAG-HPO (``rag_hpo_lib``), which an earlier exploratory run/13_03/13_04 run.

The RAG-HPO reproduction with the published code shipped a run in which the mapping prompt listed twenty candidates all named ``info``,
because paper-era code that iterates a candidate dict as ``{<phrase>: 'HP:…'}`` was handed rows
shaped ``{'info': …, 'hp_id': …}``. Nothing failed: the pipeline ran, the artifacts looked
well-formed, and only the F1 was wrong. The retrieval artifact is *not* a witness either, it is
written from the same rows by a reader that parses them correctly, so it looked right throughout.

This module pins the one thing that would have caught it, for the chain that produces the numbers
the paper reproduction is compared against:

    hpo_meta.json ``{hp_id, info}``
      → ``load_vector_db``        ``{hp_id, info, …}``
      → ``_collect_metadata_best``  ``{hp_id, phrase, similarity, …}``   (info → phrase)
      → ``generate_hpo_terms``      ``{term, id}``                        (what the model reads)

Each hop renames the term field, and every hop reads its keys explicitly rather than iterating
``.items()``, which is why this path was never affected. The assertions below are on the *term
text the model is asked to choose between*, so any future rename that silently drops it fails here,
not in a cohort's F1 three days later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_RAGHPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "RAG-HPO"
if str(_RAGHPO_DIR) not in sys.path:
    sys.path.insert(0, str(_RAGHPO_DIR))

import rag_hpo_lib as rag  # noqa: E402

pytestmark = pytest.mark.unit


class RecordingLLM:
    """Answers with a fixed reply and keeps every payload it was handed."""

    def __init__(self, reply: str = '{"hpo_id": "HP:0001250"}'):
        self.reply = reply
        self.payloads: list[str] = []

    def query(self, user_input: str, system_message: str) -> str:
        self.payloads.append(user_input)
        return self.reply


@pytest.fixture(autouse=True)
def _no_leaked_client():
    yield
    rag.llm_client = None


def _row(metadata: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "phrase": "fits",
        "category": "Abnormal",
        "original_sentence": "Fits since birth",
        "unique_metadata": metadata,   # one cell holding the whole candidate list
    }])


def test_the_mapping_payload_names_the_term_not_the_field_it_was_stored_under():
    """The exact defect the RAG-HPO reproduction with the published code shipped, asserted on the payload the model actually receives."""
    rag.llm_client = RecordingLLM()

    rag.generate_hpo_terms(
        _row([{"hp_id": "HP:0001250", "phrase": "seizure", "similarity": 0.9}]), "SYS_II")

    payload = json.loads(rag.llm_client.payloads[-1])
    assert payload["candidates"] == [{"term": "seizure", "id": "HP:0001250"}]
    assert "info" not in {c["term"] for c in payload["candidates"]}


def test_a_candidate_carrying_info_instead_of_phrase_is_still_named_correctly():
    """``m.get('phrase') or m.get('info')``, the retrieval hop renames, so both must work."""
    rag.llm_client = RecordingLLM()

    rag.generate_hpo_terms(_row([{"hp_id": "HP:0001250", "info": "seizure"}]), "SYS_II")

    payload = json.loads(rag.llm_client.payloads[-1])
    assert payload["candidates"] == [{"term": "seizure", "id": "HP:0001250"}]


def test_a_candidate_without_a_term_is_dropped_rather_than_named_none():
    rag.llm_client = RecordingLLM(reply="nothing fits")

    rag.generate_hpo_terms(_row([{"hp_id": "HP:0001250"}]), "SYS_II")

    payload = json.loads(rag.llm_client.payloads[-1])
    assert payload["candidates"] == []


def test_load_vector_db_carries_the_term_text_under_info(tmp_path):
    meta = tmp_path / "hpo_meta.json"
    vec = tmp_path / "hpo_embedded.npz"
    meta.write_text(json.dumps({
        "constants": {"HP:0001250": {"lineage": ["HP:0000001"], "organ_system": "nervous"}},
        "entries": [{"hp_id": "HP:0001250", "info": "seizure", "direction": 0}],
    }), encoding="utf-8")
    np.savez(vec, emb=np.zeros((1, 4), dtype=np.float32))

    docs, _ = rag.load_vector_db(str(meta), str(vec))

    assert docs[0]["hp_id"] == "HP:0001250"
    assert docs[0]["info"] == "seizure"


def test_retrieval_renames_info_to_phrase_which_is_what_the_mapping_hop_reads():
    """The rename is the joint between the two hops. A silent drop here empties every candidate."""
    docs = [
        {"hp_id": "HP:0001250", "info": "seizure"},
        {"hp_id": "HP:0000252", "info": "microcephaly"},
    ]
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = rag.create_faiss_index(emb)
    query = np.array([[1.0, 0.0]], dtype=np.float32)

    results = rag._collect_metadata_best(
        phrase="seizure", query_vec=query, index=index, docs=docs,
        top_k=2, min_unique=2, max_unique=2)

    assert results[0]["phrase"] == "seizure"
    assert results[0]["hp_id"] == "HP:0001250"
    assert all(r.get("phrase") for r in results)
