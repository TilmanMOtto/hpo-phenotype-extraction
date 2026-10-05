"""Unit tests for hpo_extraction.retrieval.similarity.SymptomScoreCalculator."""

import numpy as np
import pytest

from hpo_extraction.retrieval.similarity import SymptomScoreCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calc_max():
    """Calculator with two 2-D context vectors for HP:0001234."""
    context_dict = {
        "HP:0001234": np.array([[1.0, 0.0], [0.0, 1.0]])
    }
    return SymptomScoreCalculator(context_dict, sentScoring="max")


@pytest.fixture
def calc_mean():
    context_dict = {
        "HP:0001234": np.array([[1.0, 0.0], [0.0, 1.0]])
    }
    return SymptomScoreCalculator(context_dict, sentScoring="mean")


# ---------------------------------------------------------------------------
# cos_sim
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cos_sim_parallel_unit_vectors_returns_one(calc_max):
    A = np.array([1.0, 0.0])
    B = np.array([1.0, 0.0])
    assert calc_max.cos_sim(A, B) == pytest.approx(1.0)


@pytest.mark.unit
def test_cos_sim_orthogonal_vectors_returns_zero(calc_max):
    A = np.array([1.0, 0.0])
    B = np.array([0.0, 1.0])
    assert calc_max.cos_sim(A, B) == pytest.approx(0.0)


@pytest.mark.unit
def test_cos_sim_antiparallel_returns_negative_one(calc_max):
    A = np.array([1.0, 0.0])
    B = np.array([-1.0, 0.0])
    assert calc_max.cos_sim(A, B) == pytest.approx(-1.0)


@pytest.mark.unit
def test_cos_sim_arbitrary_vectors_within_range(calc_max):
    A = np.array([3.0, 4.0])  # norm = 5
    B = np.array([1.0, 0.0])
    result = calc_max.cos_sim(A, B)
    # cos(angle) = (3*1 + 4*0) / (5*1) = 3/5 = 0.6
    assert result == pytest.approx(0.6, abs=1e-6)


@pytest.mark.unit
def test_cos_sim_zero_vector_returns_nan(calc_max):
    """Bug: zero-vector input → nan (silent). Documented as known issue."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    with pytest.warns(RuntimeWarning):
        result = calc_max.cos_sim(A, B)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# sent_2_context, mode="max"
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_sent_2_context_max_returns_highest_similarity(calc_max):
    # context: [[1,0],[0,1]], sent=[1,0]
    # sim([1,0],[1,0])=1.0, sim([0,1],[1,0])=0.0 → max=1.0
    sent = np.array([1.0, 0.0])
    result = calc_max.sent_2_context("HP:0001234", sent)
    assert result == pytest.approx(1.0)


@pytest.mark.unit
def test_sent_2_context_max_diagonal_sentence(calc_max):
    # context: [[1,0],[0,1]], sent=[1/√2, 1/√2]
    # sim([1,0],sent) = 1/√2, sim([0,1],sent) = 1/√2 → max = 1/√2
    sent = np.array([1.0, 1.0]) / np.sqrt(2)
    result = calc_max.sent_2_context("HP:0001234", sent)
    assert result == pytest.approx(1 / np.sqrt(2), abs=1e-6)


# ---------------------------------------------------------------------------
# sent_2_context, mode="mean"
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_sent_2_context_mean_averages_scores(calc_mean):
    # context: [[1,0],[0,1]], sent=[1,0]
    # scores = [1.0, 0.0] → mean = 0.5
    sent = np.array([1.0, 0.0])
    result = calc_mean.sent_2_context("HP:0001234", sent)
    assert result == pytest.approx(0.5)


@pytest.mark.unit
def test_sent_2_context_mean_top_n_truncates(calc_mean):
    calc_mean.top_n = 1
    # context: [[1,0],[0,1]], sent=[1,0]
    # scores = [1.0, 0.0], sorted = [0.0, 1.0], top-1 = [1.0] → mean = 1.0
    sent = np.array([1.0, 0.0])
    result = calc_mean.sent_2_context("HP:0001234", sent)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# text_2_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_text_2_context_returns_list_of_same_length(calc_max):
    sents = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    result = calc_max.text_2_context(sents, "HP:0001234")
    assert len(result) == 3


@pytest.mark.unit
def test_text_2_context_all_values_are_floats(calc_max):
    sents = [np.array([1.0, 0.0])]
    result = calc_max.text_2_context(sents, "HP:0001234")
    assert isinstance(result[0], float)


@pytest.mark.unit
def test_text_2_context_empty_list_returns_empty(calc_max):
    assert calc_max.text_2_context([], "HP:0001234") == []


# ---------------------------------------------------------------------------
# symptom_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_symptom_score_returns_index_of_highest_similarity(calc_max):
    enc_dict = {"p1": [np.array([0.0, 1.0]), np.array([1.0, 0.0])]}
    # sent[0] vs HP:0001234 context [[1,0],[0,1]]: max(0.0, 1.0) = 1.0
    # sent[1] vs HP:0001234 context [[1,0],[0,1]]: max(1.0, 0.0) = 1.0
    # Both equal. Index_of_max = 0 (first encountered max)
    idx, val = calc_max.symptom_score("HP:0001234", enc_dict, "p1")
    assert isinstance(idx, int)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


@pytest.mark.unit
def test_symptom_score_correctly_identifies_best_sentence(calc_max):
    # sent[0] = [0,1]: sim w/ [1,0]=0, w/ [0,1]=1 → max=1.0
    # sent[1] = [-1,0]: sim w/ [1,0]=-1, w/ [0,1]=0 → max=0.0
    # sent[0] wins
    enc_dict = {"p1": [np.array([0.0, 1.0]), np.array([-1.0, 0.0])]}
    idx, val = calc_max.symptom_score("HP:0001234", enc_dict, "p1")
    assert idx == 0
    assert val == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# resetSentScoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reset_sent_scoring_updates_mode():
    calc = SymptomScoreCalculator({}, sentScoring="max")
    calc.resetSentScoring("mean", 3)  # positional: set_mode, set_top_n
    assert calc.mode == "mean"
    assert calc.top_n == 3


# ---------------------------------------------------------------------------
# symptom_sents / retrieve, with monkeypatched HPOTree
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_symptom_sents_returns_correct_top_n(patch_retrieval_hpo_tree):
    calc = SymptomScoreCalculator(
        {"HP:0001234": np.array([[1.0, 0.0]])}, sentScoring="max"
    )
    enc = {"p1": [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]}
    sents = {"p1": ["sent_a", "sent_b", "sent_c"]}
    result = calc.symptom_sents(enc, sents, ["HP:0001234"], top_n=2)
    assert len(result["p1"]["HP:0001234"]["top_sents"]) == 2


@pytest.mark.unit
def test_symptom_sents_deduplicates_identical_sentences(patch_retrieval_hpo_tree):
    # All 3 sentences identical → only 1 unique returned even if top_n=3
    calc = SymptomScoreCalculator(
        {"HP:0001234": np.array([[1.0, 0.0]])}, sentScoring="max"
    )
    enc = {"p1": [np.array([1.0, 0.0])] * 3}
    sents = {"p1": ["same_text", "same_text", "same_text"]}
    result = calc.symptom_sents(enc, sents, ["HP:0001234"], top_n=3)
    assert len(result["p1"]["HP:0001234"]["top_sents"]) == 1


@pytest.mark.unit
def test_retrieve_is_alias_for_symptom_sents(patch_retrieval_hpo_tree):
    calc = SymptomScoreCalculator(
        {"HP:0001234": np.array([[1.0, 0.0]])}, sentScoring="max"
    )
    enc = {"p1": [np.array([1.0, 0.0])]}
    sents = {"p1": ["fever sentence"]}
    r1 = calc.symptom_sents(enc, sents, ["HP:0001234"], top_n=1)
    r2 = calc.retrieve(enc, sents, ["HP:0001234"], top_n=1)
    assert r1["p1"]["HP:0001234"]["top_sents"] == r2["p1"]["HP:0001234"]["top_sents"]


@pytest.mark.unit
def test_symptom_sents_output_schema(patch_retrieval_hpo_tree):
    calc = SymptomScoreCalculator(
        {"HP:0001234": np.array([[1.0, 0.0]])}, sentScoring="max"
    )
    enc = {"p1": [np.array([1.0, 0.0])]}
    sents = {"p1": ["fever sentence"]}
    result = calc.symptom_sents(enc, sents, ["HP:0001234"], top_n=1)
    entry = result["p1"]["HP:0001234"]
    assert "top_sents" in entry
    assert "top_scores" in entry
    assert "top_indices" in entry
    assert "symptom name" in entry


@pytest.mark.unit
def test_symptom_sents_missing_hpo_key_raises_key_error():
    calc = SymptomScoreCalculator({}, sentScoring="max")
    with pytest.raises(KeyError):
        calc.text_2_context([np.array([1.0])], "HP:NONEXISTENT")
