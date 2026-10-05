"""The RAG-HPO reproduction with the published code: the published RAG-HPO pipeline, fixed where it differs from the current one.

``rag_hpo_lib_paper`` exists because upstream's code moved on after the paper. What these tests
protect is that distance, each case below is a behaviour that the newer
``rag_hpo_lib`` does *differently*, and that a well-meaning "fix" would quietly undo:

* no category filter on extracted phrases, and no null option in the mapping stage, the two
  places where the newer code drops predictions;
* every ``HP:\\d+`` in a mapping reply counts, so one phrase can yield several terms;
* candidates are deduplicated on the ``(phrase, hp_id)`` pair, not on ``hp_id``, so synonyms of one
  term legitimately occupy several of the 20 slots.

The candidate row shape is essential and has its own cases below. Paper-era candidates are
single-key ``{<phrase>: 'HP:…'}`` dicts. The newer ``{'info': …, 'hp_id': …}`` rows are a later
format that this code has no branch for. Feeding it the newer shape type-checks, runs, and
silently destroys the pipeline, the model is asked to choose between candidates all named
``info``, and the local shortcut can never match, so the conversion in :func:`load_vector_db` is
fixed here.
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

import build_vector_db_paper as builder  # noqa: E402
import rag_hpo_lib_paper as rag  # noqa: E402
from hpo_extraction.baselines.rag_hpo_published_runner import _hpo_ids, run_rag_hpo_paper  # noqa: E402

pytestmark = pytest.mark.unit


# ── doubles ──────────────────────────────────────────────────────────────────

class FakeLLM:
    """Returns a canned reply per system message, and records what it was asked.

    An empty note extracts nothing, as a real model would, the runner has to survive a report
    that produces no phrases at all.
    """

    def __init__(self, extraction: str, mapping: str):
        self.extraction = extraction
        self.mapping = mapping
        self.calls: list[tuple[str, str]] = []

    def query(self, user_input: str, system_message: str) -> str:
        self.calls.append((system_message, user_input))
        if system_message == "SYS_I":
            return self.extraction if user_input.strip() else '{"findings": []}'
        return self.mapping


class FakeEmbedder:
    """Maps a phrase to a fixed unit vector, so retrieval order is under the test's control."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True, **_):
        vec = np.asarray(self.vectors[text], dtype=np.float32)
        return vec / np.linalg.norm(vec)


def _tiny_db() -> tuple[list[dict[str, str]], object]:
    """Four rows: two synonyms of one term, plus two other terms.

    In the paper-era shape ``load_vector_db`` hands the pipeline, one key, the phrase itself.
    """
    docs = [
        {"seizure": "HP:0001250"},
        {"epileptic seizure": "HP:0001250"},
        {"microcephaly": "HP:0000252"},
        {"hypotonia": "HP:0001252"},
    ]
    emb = np.array([
        [1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [-1.0, 0.0],
    ], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return docs, rag.create_faiss_index(emb)


@pytest.fixture(autouse=True)
def _no_leaked_client():
    yield
    rag.llm_client = None


# ── extraction ───────────────────────────────────────────────────────────────

def test_findings_are_read_from_the_findings_key_through_code_fences():
    reply = 'Here you go:\n```json\n{"findings": ["seizures", "hypotonia"]}\n```'
    assert rag.extract_findings(reply) == ["seizures", "hypotonia"]


def test_a_reply_without_json_yields_no_findings():
    assert rag.extract_findings("I could not find anything.") == []
    assert rag.extract_findings("") == []


def test_no_category_filter_every_extracted_phrase_survives():
    """The 2025-07-30 prompt would classify these and keep only 'Abnormal'. This one keeps all."""
    docs, index = _tiny_db()
    phrases = ["seizure", "microcephaly", "hypotonia"]
    rag.llm_client = FakeLLM(json.dumps({"findings": phrases}), '{"hpo_id": null}')
    embedder = FakeEmbedder({"seizure": [1.0, 0.0], "microcephaly": [0.0, 1.0],
                             "hypotonia": [-1.0, 0.0]})

    out = rag.process_row("Seizure. Microcephaly. Hypotonia.", "SYS_I", embedder, index, docs)

    assert list(out["phrase"]) == phrases


def test_a_dict_finding_is_unwrapped_and_junk_is_skipped(caplog):
    """Upstream assumed bare strings and would raise. Unattended, that would kill a whole run."""
    docs, index = _tiny_db()
    embedder = FakeEmbedder({"seizure": [1.0, 0.0]})
    findings = [{"phrase": "seizure"}, 42, ""]

    out = rag.process_findings(findings, "Seizure noted.", embedder, index, docs)

    assert list(out["phrase"]) == ["seizure"]


# ── retrieval ────────────────────────────────────────────────────────────────

def test_candidates_dedup_on_the_phrase_hp_id_pair_not_on_hp_id():
    """Both synonyms of HP:0001250 are kept, that is what reaches the prompt."""
    docs, index = _tiny_db()
    embedder = FakeEmbedder({"fits": [1.0, 0.05]})

    out = rag.process_findings(["fits"], "Fits since birth.", embedder, index, docs)

    candidates = [json.loads(m) for m in out["unique_metadata"].iloc[0]]
    assert [rag.candidate_pair(c) for c in candidates[:2]] == [
        ("seizure", "HP:0001250"), ("epileptic seizure", "HP:0001250"),
    ]


def test_keep_top_caps_the_candidate_list():
    docs, index = _tiny_db()
    embedder = FakeEmbedder({"fits": [1.0, 0.05]})

    out = rag.process_findings(["fits"], "Fits.", embedder, index, docs, keep_top=2)

    assert len(out["unique_metadata"].iloc[0]) == 2


def test_the_best_overlapping_sentence_is_attached():
    docs, index = _tiny_db()
    embedder = FakeEmbedder({"microcephaly": [0.0, 1.0]})
    note = "The infant was born at term. Head circumference showed microcephaly at 3 months."

    out = rag.process_findings(["microcephaly"], note, embedder, index, docs)

    assert out["original_sentence"].iloc[0].startswith("Head circumference")


# ── mapping ──────────────────────────────────────────────────────────────────

def _one_row(candidates: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "phrase": "fits",
        "unique_metadata": [json.dumps(c) for c in candidates],
        "original_sentence": "Fits since birth",
    }])


def test_context_renders_one_line_per_candidate_naming_the_phrase():
    """The line the model chooses between must name the term, not the field it was stored under.

    Rendering ``{'info': …, 'hp_id': …}`` here instead produces the pair ``- info (seizure)`` /
    ``- hp_id (HP:0001250)``: twice the lines, and every candidate labelled ``info``.
    """
    rag.llm_client = FakeLLM("", "{}")
    rag.generate_hpo_terms(_one_row([{"seizure": "HP:0001250"}]), "SYS_II")

    _, prompt = rag.llm_client.calls[-1]
    assert "- seizure (HP:0001250)" in prompt
    assert "- info (" not in prompt and "- hp_id (" not in prompt


def test_every_hpo_id_in_the_reply_is_kept():
    """No candidate-membership check and no null option, both arrived after the paper."""
    rag.llm_client = FakeLLM("", "The best matches are HP:0001250 and HP:0001252.")

    out = rag.generate_hpo_terms(_one_row([{"seizure": "HP:0001250"}]), "SYS_II")

    assert out["response"].iloc[0] == "HP:0001250, HP:0001252"


def test_a_reply_without_an_id_is_recorded_as_no_terms():
    rag.llm_client = FakeLLM("", "None of these fit.")

    out = rag.generate_hpo_terms(_one_row([{"seizure": "HP:0001250"}]), "SYS_II")

    assert out["response"].iloc[0] == "No HPO terms found"


# ── the local shortcut ───────────────────────────────────────────────────────

def test_the_shortcut_resolves_a_matching_phrase_without_the_llm():
    metadata = [json.dumps({"seizure": "HP:0001250"})]

    assert rag.extract_hpo_term("seizure", list(metadata)) == "HP:0001250"
    assert rag.extract_hpo_term("Seizure.", list(metadata)) == "HP:0001250"  # their clean_text


def test_the_shortcut_returns_none_on_anything_short_of_exact_equality():
    """Only step 3 returns. Steps 1 and 2 append copies, so a near miss still goes to the LLM."""
    metadata = [json.dumps({"seizure": "HP:0001250"})]

    assert rag.extract_hpo_term("seizures", list(metadata)) is None
    assert rag.extract_hpo_term("microcephaly", list(metadata)) is None


def test_the_shortcut_extends_the_candidate_list_in_place():
    """Callers reading unique_metadata afterwards see the appended copies, the runner cuts them."""
    metadata = [json.dumps({"seizure": "HP:0001250"})]

    rag.extract_hpo_term("seizure", metadata)

    assert len(metadata) > 1


# ── the candidate row shape ──────────────────────────────────────────────────

def test_load_vector_db_hands_the_pipeline_the_paper_era_shape(tmp_path):
    """The on-disk row is two fields. What the pipeline is written for is one key, the phrase.

    Skipping this conversion is not a faithful reproduction but a format mismatch, and it is
    invisible: every stage still runs, on candidates that are all named ``info``.
    """
    meta = tmp_path / "meta.json"
    vec = tmp_path / "emb.npz"
    meta.write_text(json.dumps({"entries": [
        {"hp_id": "HP:0001250", "info": "seizure"},
        {"hp_id": "HP:0000252", "info": "microcephaly"},
    ]}), encoding="utf-8")
    np.savez(vec, emb=np.zeros((2, 4), dtype=np.float32))

    docs, emb = rag.load_vector_db(str(meta), str(vec))

    assert docs == [{"seizure": "HP:0001250"}, {"microcephaly": "HP:0000252"}]
    assert emb.shape == (2, 4)


def test_candidate_pair_reads_both_shapes():
    """Artifact readers get the newer shape too, since that is what the DB file stores."""
    assert rag.candidate_pair({"seizure": "HP:0001250"}) == ("seizure", "HP:0001250")
    assert rag.candidate_pair('{"seizure": "HP:0001250"}') == ("seizure", "HP:0001250")
    assert rag.candidate_pair({"info": "seizure", "hp_id": "HP:0001250"}) == (
        "seizure", "HP:0001250")
    assert rag.candidate_pair({"phrase": "not an id"}) is None
    assert rag.candidate_pair("not json") is None


# ── the vector DB's release guard ────────────────────────────────────────────

def test_the_fixed_release_is_the_one_the_guard_reads_back():
    """The two halves of the guard must speak the same dialect.

    The download URL needs the git tag (``v2024-08-13``). The ontology's own version IRI carries
    the bare date. Pinning the tag made the guard reject the correct file, a build that cannot
    run at all, from the check meant to protect it.
    """
    data = {"graphs": [{"meta": {
        "version": "http://purl.obolibrary.org/obo/hp/releases/2024-08-13/hp.json"}}]}

    assert builder.detect_release(data) == builder.HPO_RELEASE
    assert f"v{builder.HPO_RELEASE}" in builder.HPO_JSON_URL


def test_a_different_release_is_still_detected_as_different():
    data = {"graphs": [{"meta": {
        "version": "http://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.json"}}]}

    assert builder.detect_release(data) == "2026-06-23" != builder.HPO_RELEASE
    assert builder.detect_release({"graphs": [{"meta": {}}]}) is None


# ── the runner ───────────────────────────────────────────────────────────────

def test_ids_are_parsed_per_cell_including_the_doubled_prefix():
    assert _hpo_ids("HP:0001250, HP:0001252") == ["HP:0001250", "HP:0001252"]
    assert _hpo_ids("HP:HP:0001250") == ["HP:0001250"]
    assert _hpo_ids("No HPO terms found") == []
    assert _hpo_ids(None) == []


def test_a_run_collapses_to_one_set_per_patient_and_keeps_empty_reports():
    docs, index = _tiny_db()
    rag.llm_client = FakeLLM(
        json.dumps({"findings": ["fits"]}),
        "HP:0001250 is the best match, HP:0001252 also applies.",
    )
    embedder = FakeEmbedder({"fits": [1.0, 0.05]})
    df = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "clinical_note": ["Fits since birth.", ""],
    })

    patient_hpos, records = run_rag_hpo_paper(
        df=df, emb_model=embedder, index=index, docs=docs,
        system_message_I="SYS_I", system_message_II="SYS_II",
    )

    assert patient_hpos["p1"] == {"HP:0001250", "HP:0001252"}
    assert patient_hpos["p2"] == set()          # a silent report still has to exist
    assert [r["patient_id"] for r in records] == ["p1"]
    assert records[0]["candidates"][0]["hp_id"] == "HP:0001250"
    assert records[0]["candidates"][0]["info"] == "seizure"
    assert records[0]["candidates"][0]["similarity"] == pytest.approx(1.0, abs=0.02)


def test_the_retrieval_artifact_excludes_the_copies_the_shortcut_appended():
    """A locally resolved phrase leaves match copies in unique_metadata. They were never retrieved.

    Writing them out would inflate the retrieval artifact with unscored duplicate candidates, and
    every retrieval-quality number computed from it.
    """
    docs, index = _tiny_db()
    rag.llm_client = FakeLLM(json.dumps({"findings": ["seizure"]}), "HP:0000252")
    embedder = FakeEmbedder({"seizure": [1.0, 0.0]})
    df = pd.DataFrame({"patient_id": ["p1"], "clinical_note": ["Seizure since birth."]})

    patient_hpos, records = run_rag_hpo_paper(
        df=df, emb_model=embedder, index=index, docs=docs,
        system_message_I="SYS_I", system_message_II="SYS_II",
    )

    assert patient_hpos["p1"] == {"HP:0001250"}          # The shortcut answered, not the LLM
    assert len(records[0]["candidates"]) == len(docs)    # not len(docs) + the appended copies
    assert all(c["similarity"] is not None for c in records[0]["candidates"])
