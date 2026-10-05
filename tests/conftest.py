"""Shared fixtures for the PhenoRAG test suite.

Rules:
- No real network calls.
- No real file I/O outside tmp_path.
- All random seeds fixed per-test, not globally.
- Heavy resources (GPU, model weights) live in @slow tests only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# HPO mock data, used wherever HPOTree is instantiated inside a function
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hpo_data() -> dict:
    """Minimal HPO data dict with two entries for unit tests."""
    return {
        "HP:0001234": {
            "Id": "HP:0001234",
            "Name": ["Fever"],
            "Alt_id": [],
            "Def": ["Elevated body temperature above the normal range."],
            "Comment": [],
            "Synonym": ["High temperature", "Pyrexia"],
            "Xref": [],
            "Is_a": ["HP:0000118"],
            "Father": {"HP:0000118": True},
            "Child": {},
            "Son": {},
        },
        "HP:0005678": {
            "Id": "HP:0005678",
            "Name": ["Headache"],
            "Alt_id": [],
            "Def": [],          # intentionally empty, tests the no-definition branch
            "Comment": [],
            "Synonym": [],      # intentionally empty, tests the no-synonym branch
            "Xref": [],
            "Is_a": ["HP:0000118"],
            "Father": {"HP:0000118": True},
            "Child": {},
            "Son": {},
        },
    }


@pytest.fixture
def mock_hpo_tree(mock_hpo_data) -> MagicMock:
    """A MagicMock HPOTree whose .data attribute is mock_hpo_data."""
    mock = MagicMock()
    mock.data = mock_hpo_data
    return mock


@pytest.fixture
def patch_retrieval_hpo_tree(monkeypatch, mock_hpo_tree):
    """Patch HPOTree inside hpo_extraction.retrieval.similarity."""
    monkeypatch.setattr("hpo_extraction.retrieval.similarity.HPOTree", lambda: mock_hpo_tree)
    return mock_hpo_tree


@pytest.fixture
def patch_prompting_hpo_tree(monkeypatch, mock_hpo_tree):
    """Patch HPOTree inside hpo_extraction.treephenorag.verifier_prompt."""
    monkeypatch.setattr("hpo_extraction.treephenorag.verifier_prompt.HPOTree", lambda: mock_hpo_tree)
    return mock_hpo_tree


@pytest.fixture
def patch_augmentation_hpo_tree(monkeypatch, mock_hpo_tree):
    """Patch HPOTree inside training.augmentation."""
    monkeypatch.setattr("training.augmentation.HPOTree", lambda: mock_hpo_tree)
    return mock_hpo_tree


# ---------------------------------------------------------------------------
# Real HPOTree, loaded once per session from the repo's own hpo.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_hpo_tree():
    """Session-scoped HPOTree using the real hpo.json (no network. Local file)."""
    from hpo_extraction.ontology.hpo_tree import HPOTree
    return HPOTree()


# ---------------------------------------------------------------------------
# Toy ontology, the hand-workable graph the thesis-metric tests run against
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_tree():
    """A ten-node DAG with a multi-parent node, an alt id and an out-of-subtree term.

    See ``tests/fixtures/toy_ontology`` for the diagram and the hand-computed reference values.
    """
    from fixtures.toy_ontology import build_toy_tree
    return build_toy_tree()


@pytest.fixture
def toy_view(toy_tree):
    """``OntologyView`` over :func:`toy_tree`, the argument every thesis-metric function takes."""
    from hpo_extraction.evaluation.metrics import OntologyView
    return OntologyView(toy_tree)


# ---------------------------------------------------------------------------
# earlier artifacts, a miniature output tree in the drivers' exact schemas
# ---------------------------------------------------------------------------

@pytest.fixture
def exp13_modules(monkeypatch):
    """Import ``experiments/result_tables`` as top-level modules.

    The experiment's own modules import each other by bare name (``import loaders``), the way
    ``run.py`` sets them up, so the package directory has to be on ``sys.path``.
    """
    import importlib
    from pathlib import Path

    names = ("discovery", "loaders", "sections", "report")
    modules = {name: importlib.import_module(f"hpo_extraction.evaluation.result_tables.{name}")
               for name in names}
    yield type("Exp13Modules", (), modules)


@pytest.fixture
def exp13_results_dir(tmp_path):
    """A complete miniature earlier output tree, see ``tests/fixtures/exp13_output``."""
    from fixtures.exp13_output import build_exp13_tree
    return build_exp13_tree(tmp_path / "output")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_vec_x() -> np.ndarray:
    return np.array([1.0, 0.0])


@pytest.fixture
def unit_vec_y() -> np.ndarray:
    return np.array([0.0, 1.0])


# ---------------------------------------------------------------------------
# Segment records factory
# ---------------------------------------------------------------------------

def make_segment_record(
    patient_id: str,
    hpo_id: str,
    rank: int,
    is_gt_relevant=None,
    slm_verdict: str = "No",
    cosine_sim: float = 0.5,
) -> dict:
    """Build a segment record dict matching the dashboard schema."""
    return {
        "patient_id": patient_id,
        "hpo_id": hpo_id,
        "rank": rank,
        "cosine_sim": cosine_sim,
        "slm_verdict": slm_verdict,
        "is_gt_relevant": is_gt_relevant,
    }


@pytest.fixture
def make_seg():
    """Return the make_segment_record factory so tests can call it."""
    return make_segment_record
