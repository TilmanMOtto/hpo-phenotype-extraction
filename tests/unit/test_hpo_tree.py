"""Unit tests for hpo_extraction.ontology.hpo_tree.HPOTree and helpers.

Uses the session-scoped real_hpo_tree fixture (loaded from resources/util/hpo.json).
No network access. File is part of the repository.
"""

import pytest

from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree, getNames


# ---------------------------------------------------------------------------
# HPOTree initialisation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hpo_tree_loads_successfully(real_hpo_tree):
    assert real_hpo_tree is not None


@pytest.mark.unit
def test_hpo_tree_root_is_phenotypic_abnormality(real_hpo_tree):
    assert real_hpo_tree.root == "HP:0000118"


@pytest.mark.unit
def test_hpo_tree_hpo_list_is_non_empty(real_hpo_tree):
    assert len(real_hpo_tree.hpo_list) > 0


@pytest.mark.unit
def test_hpo_tree_hpo_list_sorted(real_hpo_tree):
    assert real_hpo_tree.hpo_list == sorted(real_hpo_tree.hpo_list)


@pytest.mark.unit
def test_hpo_tree_n_concept_matches_hpo_list_length(real_hpo_tree):
    assert real_hpo_tree.n_concept == len(real_hpo_tree.hpo_list)


@pytest.mark.unit
def test_hpo_tree_root_in_hpo_list(real_hpo_tree):
    assert real_hpo_tree.root in real_hpo_tree.hpo_list


# ---------------------------------------------------------------------------
# Index mapping round-trips
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hpo2idx_and_idx2hpo_are_inverse(real_hpo_tree):
    hpo = real_hpo_tree.hpo_list[0]
    idx = real_hpo_tree.getHPO2idx(hpo)
    assert real_hpo_tree.getIdx2HPO(idx) == hpo


@pytest.mark.unit
def test_all_hpo_list_entries_round_trip_through_index(real_hpo_tree):
    for hpo in real_hpo_tree.hpo_list[:20]:  # spot-check first 20
        idx = real_hpo_tree.getHPO2idx(hpo)
        assert real_hpo_tree.getIdx2HPO(idx) == hpo


@pytest.mark.unit
def test_hpo2idx_none_sentinel(real_hpo_tree):
    # "None" key is always added as the last index
    assert "None" in real_hpo_tree.hpo2idx


# ---------------------------------------------------------------------------
# getNameByHPO
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_getNameByHPO_returns_string(real_hpo_tree):
    name = real_hpo_tree.getNameByHPO("HP:0000118")
    assert isinstance(name, str)


@pytest.mark.unit
def test_getNameByHPO_is_lowercase(real_hpo_tree):
    name = real_hpo_tree.getNameByHPO("HP:0000118")
    assert name == name.lower()


# ---------------------------------------------------------------------------
# getAllFatherHPOByHPO
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_all_father_hpo_returns_set(real_hpo_tree):
    result = real_hpo_tree.getAllFatherHPOByHPO("HP:0000118")
    assert isinstance(result, set)


@pytest.mark.unit
def test_get_all_father_hpo_of_root_returns_empty_set(real_hpo_tree):
    # HP:0000118 is the root of phenotypic_abnormality. It's not in phenotypic_abnormalityNT
    result = real_hpo_tree.getAllFatherHPOByHPO("HP:0000118")
    assert result == set()


# ---------------------------------------------------------------------------
# getPhrasesByHPO
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_phrases_by_hpo_returns_list_of_strings(real_hpo_tree):
    phrases = real_hpo_tree.getPhrasesByHPO("HP:0000118")
    assert isinstance(phrases, list)
    assert all(isinstance(p, str) for p in phrases)


@pytest.mark.unit
def test_get_phrases_by_hpo_all_lowercase(real_hpo_tree):
    phrases = real_hpo_tree.getPhrasesByHPO("HP:0000118")
    for p in phrases:
        assert p == p.lower()


# ---------------------------------------------------------------------------
# getLayer1HPOByHPO
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_layer1_hpo_returns_list(real_hpo_tree):
    result = real_hpo_tree.getLayer1HPOByHPO("HP:0000118")
    assert isinstance(result, list)


@pytest.mark.unit
def test_get_layer1_hpo_unknown_node_returns_none_list(real_hpo_tree):
    result = real_hpo_tree.getLayer1HPOByHPO("HP:0000001")
    assert result == ["None"]


# ---------------------------------------------------------------------------
# HPO_class
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hpo_class_stores_id(real_hpo_tree):
    struct = HPO_class(real_hpo_tree.data["HP:0000118"])
    assert struct.id == "HP:0000118"


@pytest.mark.unit
def test_hpo_class_child_is_set(real_hpo_tree):
    struct = HPO_class(real_hpo_tree.data["HP:0000118"])
    assert isinstance(struct.child, set)


@pytest.mark.unit
def test_hpo_class_father_is_set(real_hpo_tree):
    struct = HPO_class(real_hpo_tree.data["HP:0000118"])
    assert isinstance(struct.father, set)


# ---------------------------------------------------------------------------
# getNames helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_names_returns_list(real_hpo_tree):
    struct = HPO_class(real_hpo_tree.data["HP:0000118"])
    names = getNames(struct)
    assert isinstance(names, list)


@pytest.mark.unit
def test_get_names_no_duplicates(real_hpo_tree):
    struct = HPO_class(real_hpo_tree.data["HP:0000118"])
    names = getNames(struct)
    assert len(names) == len(set(names))


@pytest.mark.unit
def test_get_names_includes_primary_name(real_hpo_tree):
    struct = HPO_class(real_hpo_tree.data["HP:0000118"])
    names = getNames(struct)
    assert struct.name[0] in names


# ---------------------------------------------------------------------------
# buildHPOTree (BFS depth computation)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_hpo_tree_populates_depth_dict(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    assert isinstance(real_hpo_tree.depth_dict, dict)
    assert len(real_hpo_tree.depth_dict) > 0


@pytest.mark.unit
def test_build_hpo_tree_root_at_depth_zero(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    assert real_hpo_tree.depth_dict[real_hpo_tree.root] == 0


# ---------------------------------------------------------------------------
# getFatherHPOByHPO
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_father_hpo_returns_none_for_root(real_hpo_tree):
    # HP:0000118 is root of phenotypic abnormality, not in phenotypic_abnormalityNT
    result = real_hpo_tree.getFatherHPOByHPO("HP:0000118")
    assert result is None


@pytest.mark.unit
def test_get_father_hpo_returns_list_for_child_node(real_hpo_tree):
    # A child node should return its Is_a list
    # Use first node in hpo_list that is not the root
    for hpo in real_hpo_tree.hpo_list:
        if hpo != real_hpo_tree.root:
            result = real_hpo_tree.getFatherHPOByHPO(hpo)
            assert result is not None
            assert isinstance(result, list)
            break


# ---------------------------------------------------------------------------
# matchPhrase2HPO
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_match_phrase_to_hpo_returns_string(real_hpo_tree):
    result = real_hpo_tree.matchPhrase2HPO("phenotypic abnormality")
    assert isinstance(result, str)


@pytest.mark.unit
def test_match_phrase_unknown_returns_empty_string(real_hpo_tree):
    result = real_hpo_tree.matchPhrase2HPO("xxxxxx_nonexistent_yyyyyyy")
    assert result == ""


# ---------------------------------------------------------------------------
# getAllPhrasesAbnorm
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_all_phrases_abnorm_returns_list(real_hpo_tree):
    phrases = real_hpo_tree.getAllPhrasesAbnorm()
    assert isinstance(phrases, list)
    assert len(phrases) > 0


# ---------------------------------------------------------------------------
# Layer-1 index
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hpo2idx_l1_and_idx2hpo_l1_are_inverse(real_hpo_tree):
    hpo = real_hpo_tree.layer1[0]
    idx = real_hpo_tree.getHPO2idx_l1(hpo)
    assert real_hpo_tree.getIdx2HPO_l1(idx) == hpo


# ---------------------------------------------------------------------------
# getNodeSimilarityByID, requires buildHPOTree
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_node_similarity_same_node_returns_one_or_one(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    # Root compared with itself should return 1.0 (documented special case)
    sim = real_hpo_tree.getNodeSimilarityByID(real_hpo_tree.root, real_hpo_tree.root)
    assert sim == pytest.approx(1.0)


@pytest.mark.unit
def test_get_node_similarity_out_of_range_returns_zero(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    # HP:0000001 is "All", not in phenotypic_abnormality
    sim = real_hpo_tree.getNodeSimilarityByID("HP:0000001", real_hpo_tree.root)
    assert sim == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# getHPO_set_similarity_max
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hpo_set_similarity_both_empty_returns_one(real_hpo_tree):
    assert real_hpo_tree.getHPO_set_similarity_max(set(), set()) == pytest.approx(1.0)


@pytest.mark.unit
def test_hpo_set_similarity_one_empty_returns_zero(real_hpo_tree):
    assert real_hpo_tree.getHPO_set_similarity_max({"HP:0000118"}, set()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Wu-Palmer boundedness on the DAG (regression)
#
# HPO is a DAG, and depth_dict is the *shortest* path from the root, so a node's
# lowest common subsumer can sit at a greater depth_dict value than the nodes it
# subsumes (true for ~1.4% of ancestor/descendant pairs). Wu-Palmer divides by
# those depths, which drove getNodeSimilarityByID up to 1.6, well outside the
# [0, 1] its callers in hpo_extraction.evaluation.tree_metrics document and rely on.
#
# The fix is depth_long (longest path from the root), which is strictly monotone
# along ancestry and therefore keeps the ratio bounded.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_depth_long_is_strictly_monotone_along_ancestry(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    for hpo in real_hpo_tree.hpo_list[::250]:  # spot-check across the ontology
        for ancestor in real_hpo_tree.getAllFatherHPOByHPO(hpo):
            if ancestor in real_hpo_tree.depth_long:
                assert real_hpo_tree.depth_long[ancestor] < real_hpo_tree.depth_long[hpo], (
                    f"{ancestor} is an ancestor of {hpo} but is not strictly shallower"
                )


@pytest.mark.unit
def test_node_similarity_never_exceeds_one(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    codes = real_hpo_tree.hpo_list[::120]
    for i, a in enumerate(codes):
        for b in codes[i:]:
            sim = real_hpo_tree.getNodeSimilarityByID(a, b)
            assert 0.0 <= sim <= 1.0, f"similarity({a}, {b}) = {sim} is outside [0, 1]"


@pytest.mark.unit
def test_node_similarity_self_is_one(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    assert real_hpo_tree.getNodeSimilarityByID("HP:0001250", "HP:0001250") == pytest.approx(1.0)


@pytest.mark.unit
def test_node_similarity_parent_beats_unrelated(real_hpo_tree):
    real_hpo_tree.buildHPOTree()
    seizure = "HP:0001250"
    parent = sorted(real_hpo_tree.data[seizure]["Is_a"])[0]
    unrelated = "HP:0001873"  # thrombocytopenia, a different organ system
    assert (real_hpo_tree.getNodeSimilarityByID(seizure, parent)
            > real_hpo_tree.getNodeSimilarityByID(seizure, unrelated))
