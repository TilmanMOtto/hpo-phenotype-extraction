"""Unit tests for ``hpo_extraction.retrieval.term_information_index``, the R3 query representation the term-information score store retrieves with.

R3 is "one string, one vector": ``label + " " + definition + " " + synonyms`` embedded once. The
failure mode these tests exist for is drift, either from the retrieval-curve analysis's condition (which would make the
retrieval experiment and the traversal incomparable) or into the *other* ontology-derived
representation in the codebase, ``tree_experiment._ontology_context_sentences``, which reads the
same fields but emits a LIST of strings and therefore several vectors per term.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpo_extraction.retrieval.term_information_index import R3_TEMPLATE, build_r3_index, r3_text

pytestmark = pytest.mark.unit


class _FakeTree:
    """Just enough of ``HPOTree`` for ``HPO_class`` to read name/def/synonym off ``data``."""

    def __init__(self, data):
        self.data = data


def _node(hpo_id, name=None, definition=None, synonym=None):
    """One ``tree.data`` entry, in ``hpo.json``'s own capitalised key spelling."""
    return {
        "Id": hpo_id,
        "Name": list(name or []),
        "Alt_id": [],
        "Def": list(definition or []),
        "Comment": [],
        "Synonym": list(synonym or []),
        "Xref": [],
        "Is_a": [],
        "Father": {},
        "Child": {},
        "Son": {},
    }


@pytest.fixture()
def tree():
    return _FakeTree({
        "HP:0000001": _node("HP:0000001", name=["Seizure"],
                            definition=["A sudden discharge."],
                            synonym=["Seizures", "Epileptic fit"]),
        "HP:0000002": _node("HP:0000002", name=["Microcytosis"]),          # label only
        "HP:0000003": _node("HP:0000003", name=["Hypotonia"],
                            synonym=["Hypotonia", "Low tone"]),
        "HP:0000004": _node("HP:0000004", name=["Ataxia"],
                            definition=["Impaired coordination."]),
    })


class TestR3Text:
    def test_all_three_parts_joined_in_template_order(self, tree):
        assert r3_text(tree, "HP:0000001") == (
            "Seizure A sudden discharge. Seizures Epileptic fit"
        )

    def test_missing_parts_are_dropped_not_padded(self, tree):
        # No double spaces, no empty leading/trailing fragment.
        assert r3_text(tree, "HP:0000004") == "Ataxia Impaired coordination."
        assert "  " not in r3_text(tree, "HP:0000004")

    def test_label_only_term_falls_back_to_the_label(self, tree):
        # A term skipped here would score as a guaranteed miss and the condition would be measuring
        # ontology *coverage* rather than ontology *wording*.
        assert r3_text(tree, "HP:0000002") == "Microcytosis"

    def test_a_synonym_equal_to_the_label_is_not_repeated(self, tree):
        assert r3_text(tree, "HP:0000003") == "Hypotonia Low tone"

    def test_every_term_gets_a_non_empty_string(self, tree):
        for hpo in tree.data:
            assert r3_text(tree, hpo).strip()


class TestBuildR3Index:
    def test_one_row_per_term_aligned_to_the_input_order(self, tree):
        hpos = ["HP:0000003", "HP:0000001", "HP:0000002"]
        seen = []

        def encode(texts):
            seen.extend(texts)
            return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1.0

        vectors, prov = build_r3_index(tree, hpos, encode)
        assert vectors.shape == (3, 4)
        assert seen == [r3_text(tree, h) for h in hpos]
        assert prov["n_terms"] == 3

    def test_rows_are_l2_normalised(self, tree):
        hpos = sorted(tree.data)

        def encode(texts):
            return np.asarray([[float(i + 1), 2.0, -3.0, 0.5] for i in range(len(texts))])

        vectors, _ = build_r3_index(tree, hpos, encode)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    def test_an_all_zero_embedding_stays_zero_rather_than_dividing_by_zero(self, tree):
        hpos = ["HP:0000001", "HP:0000002"]

        def encode(texts):
            return np.zeros((len(texts), 3), dtype=np.float32)

        vectors, _ = build_r3_index(tree, hpos, encode)
        assert np.isfinite(vectors).all()
        assert not vectors.any()

    def test_provenance_counts_the_conditions_ceiling(self, tree):
        hpos = sorted(tree.data)

        def encode(texts):
            return np.ones((len(texts), 2), dtype=np.float32)

        _, prov = build_r3_index(tree, hpos, encode)
        assert prov["template"] == R3_TEMPLATE
        assert prov["n_terms"] == 4
        assert prov["n_without_definition"] == 2        # HP:...02 and HP:...03
        assert prov["n_without_synonym"] == 2           # HP:...02 and HP:...04
        assert prov["n_label_only"] == 1                # HP:...02 alone
        assert prov["median_chars"] > 0


class TestNotTheOtherOntologyRepresentation:
    """R3 must stay one vector per term, the distinction the term-information score store's atomic change rests on."""

    def test_r3_is_a_single_string_where_the_fallback_is_a_list(self, tree):
        from hpo_extraction.treephenorag.score_store import _ontology_context_sentences

        fallback = _ontology_context_sentences("HP:0000001", tree)
        assert isinstance(fallback, list) and len(fallback) > 1
        assert isinstance(r3_text(tree, "HP:0000001"), str)
        # Same material, different shape: the fallback's parts all occur in R3's one string.
        for part in fallback:
            assert part in r3_text(tree, "HP:0000001")


class TestBuildUniverseHonoursRetrievalIndex:
    """``retrieval_index`` is the term-information score store's whole atomic change, one key, wired in one place."""

    @staticmethod
    def _tree():
        tree = _FakeTree({
            "HP:0000118": _node("HP:0000118", name=["Phenotypic abnormality"]),
            "HP:0000707": _node("HP:0000707", name=["Nervous system"],
                                definition=["Of the nervous system."]),
            "HP:0001250": _node("HP:0001250", name=["Seizure"], synonym=["Seizures"]),
        })
        tree.data["HP:0000707"]["Son"] = {"HP:0001250": 1}
        tree.root = "HP:0000118"
        tree.layer1 = ["HP:0000707"]
        tree.phenotypic_abnormality = {"HP:0000118", "HP:0000707", "HP:0001250"}
        return tree

    @staticmethod
    def _cfg(**over):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"context_dir": "/does/not/exist", "restrict_to_subtree": None,
                                **over})
        OmegaConf.set_struct(cfg, True)
        return cfg

    class _Model:
        """A stand-in encoder: 4-dim, deterministic, and it records what it was asked to embed."""

        def __init__(self):
            self.seen = []

        def encode(self, texts, **kwargs):
            self.seen.append(list(texts))
            return np.asarray([[float(len(t)), 1.0, 0.0, -1.0] for t in texts],
                              dtype=np.float32)

    def test_ontology_r3_gives_one_context_vector_per_node(self, caplog):
        import logging

        from hpo_extraction.treephenorag.score_store import _build_universe

        model = self._Model()
        scorer, ctx, _children, _roots, subtree = _build_universe(
            self._cfg(retrieval_index="ontology_r3"), self._tree(), model,
            logging.getLogger("t"),
        )
        universe = sorted(subtree)
        assert sorted(ctx) == universe
        scorer.prepare()
        assert set(scorer.n_context_sentences.values()) == {1}
        # It never touched context_dir, which does not exist, that is the condition's claim.
        assert model.seen == [[r3_text(self._tree(), h) for h in universe]]

    def test_default_is_the_synthetic_index(self):
        # The shipped runs must be unaffected: with no key set, the missing context_dir is reached
        # and raises, i.e. The synthetic-sentence path is the one taken.
        import logging

        from hpo_extraction.treephenorag.score_store import _build_universe

        with pytest.raises((FileNotFoundError, OSError)):
            _build_universe(self._cfg(), self._tree(), self._Model(), logging.getLogger("t"))

    def test_an_unknown_index_name_is_rejected_before_any_work(self):
        import logging

        from hpo_extraction.treephenorag.score_store import _build_universe

        with pytest.raises(ValueError, match="retrieval_index"):
            _build_universe(self._cfg(retrieval_index="r3"), self._tree(), self._Model(),
                            logging.getLogger("t"))
