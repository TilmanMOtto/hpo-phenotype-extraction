"""Sentence encoding with SentenceTransformer models."""

import numpy as np
from sentence_transformers import SentenceTransformer


def encode_dict(sent_dict: dict[str, list[str]], model: SentenceTransformer) -> dict[str, np.ndarray]:
    """
    Encode each sentence list in sent_dict into a 2-D embedding array.

    Args:
        sent_dict: {id → [sentence_strings]}
        model: Loaded SentenceTransformer model.

    Returns:
        {id → ndarray[N_sents × embedding_dim]}
    """
    return {key: model.encode(sents) for key, sents in sent_dict.items()}
