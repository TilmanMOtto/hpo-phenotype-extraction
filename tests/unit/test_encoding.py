"""Unit tests for hpo_extraction.retrieval.encoding.encode_dict.

SentenceTransformer is mocked, no model weights needed.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from hpo_extraction.retrieval.encoding import encode_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    model = MagicMock()
    # encode([str1, str2]) → ndarray with shape (2, 4)
    model.encode.side_effect = lambda sents: np.zeros((len(sents), 4))
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_encode_dict_returns_dict(mock_model):
    result = encode_dict({"p1": ["hello", "world"]}, mock_model)
    assert isinstance(result, dict)


@pytest.mark.unit
def test_encode_dict_preserves_keys(mock_model):
    result = encode_dict({"p1": ["a"], "p2": ["b"]}, mock_model)
    assert "p1" in result
    assert "p2" in result


@pytest.mark.unit
def test_encode_dict_shape_matches_sentence_count(mock_model):
    sents = ["sent one", "sent two", "sent three"]
    result = encode_dict({"p1": sents}, mock_model)
    # shape should be (3, embedding_dim)
    assert result["p1"].shape[0] == 3


@pytest.mark.unit
def test_encode_dict_calls_model_encode_per_patient(mock_model):
    encode_dict({"p1": ["a"], "p2": ["b", "c"]}, mock_model)
    assert mock_model.encode.call_count == 2


@pytest.mark.unit
def test_encode_dict_empty_dict_returns_empty(mock_model):
    result = encode_dict({}, mock_model)
    assert result == {}


@pytest.mark.unit
def test_encode_dict_empty_sentence_list(mock_model):
    mock_model.encode.side_effect = lambda sents: np.zeros((len(sents), 4))
    result = encode_dict({"p1": []}, mock_model)
    assert result["p1"].shape == (0, 4)


@pytest.mark.unit
def test_encode_dict_result_values_are_ndarrays(mock_model):
    result = encode_dict({"p1": ["a", "b"]}, mock_model)
    assert isinstance(result["p1"], np.ndarray)
