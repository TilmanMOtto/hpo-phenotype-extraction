"""End-to-end tests of the two applications on synthetic sentences.

No language model or sentence encoder is available to the test suite, so both methods run with
stand-ins: a hashed bag-of-words encoder, a verifier that answers *Yes* when a word of the term's
label occurs in the segment, and jurors with scripted replies. Everything between the models
(segmentation, retrieval, prompts, traversal, pooling, parsing, normalisation, voting) is the real
code. The ontology is the release the thesis used, limited to a small subtree.
"""
from __future__ import annotations

import json
import re
import zlib

import numpy as np
import pytest

from hpo_extraction.ontology.hpo_tree import HPOTree
from hpo_extraction.results import ReportResult, read_reports, write_jsonl

SEIZURE = "HP:0001250"
HYPOTONIA = "HP:0001252"

REPORT = ("The boy is four years old. He has had recurrent seizures since the age of two. "
          "His weight and height are within the normal range.")


@pytest.fixture(scope="module")
def tree():
    t = HPOTree()
    t.buildHPOTree()
    return t


class BagOfWords:
    """Stand-in sentence encoder: hashed word counts, one constant dimension against zero rows."""

    def encode(self, texts, **_):
        out = np.zeros((len(texts), 257), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % 256] += 1.0
            out[i, 256] = 0.1
        return out


class WordVerifier:
    """Stand-in verifier: margin +6 when a label word of five or more letters is in the segment."""

    def __init__(self):
        self.n_calls = 0

    def generate_batch(self, prompts, system_prompt):
        out = []
        for prompt in prompts:
            label = re.match(r"The symptom (.+?)(?: is defined as|\. | is also)", prompt).group(1)
            segment = prompt.rsplit(":'", 1)[1].rsplit("'.", 1)[0].lower()
            words = [w for w in re.findall(r"[a-z]+", label.lower()) if len(w) >= 5]
            hit = any(w in segment for w in words)
            out.append({"margin": 6.0 if hit else -6.0})
            self.n_calls += 1
        return out


# ── TreePhenoRAG ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def treephenorag(tree):
    from hpo_extraction.treephenorag import pipeline as tp
    from hpo_extraction.treephenorag import verifier_prompt

    # The prompt builder reads the ontology file on every call. Hand it the loaded tree instead.
    original = verifier_prompt.HPOTree
    verifier_prompt.HPOTree = lambda: tree
    try:
        settings = tp.TreePhenoRAGSettings(subtree_root=SEIZURE, segments_per_term=3)
        method = tp.TreePhenoRAG(tree, BagOfWords(), WordVerifier(), settings)
        yield method
    finally:
        verifier_prompt.HPOTree = original


def test_treephenorag_finds_the_named_term_with_its_evidence(treephenorag):
    from hpo_extraction.treephenorag import verifier_prompt

    original = verifier_prompt.HPOTree
    verifier_prompt.HPOTree = lambda: treephenorag.tree
    try:
        result = treephenorag.extract(REPORT, "syn_1")
    finally:
        verifier_prompt.HPOTree = original
    assert isinstance(result, ReportResult) and result.method == "treephenorag"
    ids = [t.hpo_id for t in result.terms]
    assert SEIZURE in ids
    seizure = result.terms[ids.index(SEIZURE)]
    assert 0.99 <= seizure.score <= 1.0
    assert seizure.evidence[0].text.startswith("He has had recurrent seizures")
    # Every accepted term names a seizure. Nothing else in the report matches a label word.
    assert all("seizure" in t.label.lower() for t in result.terms)
    assert result.n_model_calls > 0
    assert all(0.0 <= t.score <= 1.0 for t in result.terms)


def test_treephenorag_report_without_findings_returns_no_terms(treephenorag):
    from hpo_extraction.treephenorag import verifier_prompt

    original = verifier_prompt.HPOTree
    verifier_prompt.HPOTree = lambda: treephenorag.tree
    try:
        result = treephenorag.extract("Routine visit. Nothing abnormal was noted.", "syn_2")
    finally:
        verifier_prompt.HPOTree = original
    assert result.terms == []


def test_online_pooling_equals_the_stored_score_pooling():
    """The application pools with pooling.py. The thesis numbers came from stored_scores.py."""
    from hpo_extraction.treephenorag import stored_scores
    from hpo_extraction.treephenorag.pipeline import POOLINGS

    rng = np.random.default_rng(0)
    n_nodes, s_max = 40, 10
    margins = rng.normal(0.0, 4.0, size=(n_nodes, s_max))
    lengths = rng.integers(1, s_max + 1, size=n_nodes)
    mask = np.arange(s_max)[None, :] < lengths[:, None]
    cache = stored_scores.ReportCache("r", [f"n{i}" for i in range(n_nodes)],
                                      np.where(mask, margins, 0.0), mask)
    offline = {
        "P0": stored_scores.pool_p0(cache), "P1": stored_scores.pool_p1(cache),
        "P2": stored_scores.pool_p2(cache), "P3_1": stored_scores.pool_p3(cache, kappa=1.0),
        "P3_2": stored_scores.pool_p3(cache, kappa=2.0),
        "P3_S": stored_scores.pool_p3(cache, kappa=mask.sum(axis=1)),
        "P4": stored_scores.pool_p4(cache), "lse_beta1": stored_scores.pool_lse_beta1(cache),
    }
    for name, values in offline.items():
        online = [POOLINGS[name](margins[i, :lengths[i]]) for i in range(n_nodes)]
        np.testing.assert_allclose(online, values, rtol=1e-12, atol=1e-12, err_msg=name)


# ── PhenoJury ────────────────────────────────────────────────────────────────

class ScriptedJuror:
    """Stand-in juror: names Seizure and Hypotonia when the sentence mentions them."""

    supports_batching = True

    def __init__(self, names_hypotonia: bool):
        self.names_hypotonia = names_hypotonia

    def generate_many(self, system_prompt, user_msgs, max_new_tokens):
        replies = []
        for msg in user_msgs:
            sentence = msg.lower()
            lines = []
            if "seizure" in sentence:
                lines.append("Seizure")
            if self.names_hypotonia and "hypotonic" in sentence:
                lines.append("Hypotonia")
            replies.append("\n".join(lines) or "NONE")
        return replies

    def unload(self):
        pass


def test_phenojury_keeps_the_terms_enough_jurors_name(tree, monkeypatch):
    from hpo_extraction.phenojury import generation
    from hpo_extraction.phenojury import pipeline as pj

    # Juror a names both findings. B and c name only the seizure. With k = 2 the vote keeps
    # Seizure (three jurors) and drops Hypotonia (one juror).
    jurors = {"a": ScriptedJuror(True), "b": ScriptedJuror(False), "c": ScriptedJuror(False)}
    monkeypatch.setattr(generation, "load_slm",
                        lambda key, path, logger=None, deterministic=False: jurors[key])
    settings = pj.PhenoJurySettings(jurors=list(jurors), k=2, unit="report")
    method = pj.PhenoJury(tree, {k: f"/models/{k}" for k in jurors},
                          pj.dictionary_normaliser(tree), settings)
    text = ("He has had recurrent seizures since the age of two. "
            "He was markedly hypotonic at birth.")
    result = method.extract(text, "syn_3")
    ids = {t.hpo_id: t for t in result.terms}
    assert SEIZURE in ids and HYPOTONIA not in ids
    assert ids[SEIZURE].score == pytest.approx(1.0)
    assert ids[SEIZURE].evidence[0].segment_index == 0
    assert result.n_model_calls == 3 * result.n_segments


def test_results_round_trip_through_json_lines(tmp_path):
    from hpo_extraction.results import Evidence, TermScore

    result = ReportResult("syn_4", [TermScore(SEIZURE, "Seizure", 0.75,
                                              [Evidence(0, "He has seizures.", 3.0)])],
                          n_segments=1, n_model_calls=8, method="phenojury")
    path = write_jsonl([result], tmp_path / "out.jsonl")
    line = json.loads(path.read_text(encoding="utf-8"))
    assert line["terms"][0]["hpo_id"] == SEIZURE
    assert line["terms"][0]["evidence"][0]["segment_index"] == 0


def test_reports_are_read_from_a_folder(tmp_path):
    (tmp_path / "b.txt").write_text("Second.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("First.", encoding="utf-8")
    assert list(read_reports(tmp_path)) == ["a", "b"]
