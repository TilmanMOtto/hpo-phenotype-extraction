"""Unit tests for hpo_extraction.treephenorag.verifier_prompt, embed_symptom_in_prompt_v1 and BaselinePrompter."""

import pytest

from hpo_extraction.treephenorag.verifier_prompt import BaselinePrompter, embed_symptom_in_prompt_v1


# ---------------------------------------------------------------------------
# embed_symptom_in_prompt_v1, HPOTree patched via conftest fixture
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prompt_list_same_length_as_sents(patch_prompting_hpo_tree):
    sents = ["Patient has fever.", "No symptoms found.", "Third sentence."]
    result = embed_symptom_in_prompt_v1(sents, "HP:0001234")
    assert len(result) == 3


@pytest.mark.unit
def test_prompt_empty_sents_returns_empty_list(patch_prompting_hpo_tree):
    result = embed_symptom_in_prompt_v1([], "HP:0001234")
    assert result == []


@pytest.mark.unit
def test_prompt_contains_hpo_label(patch_prompting_hpo_tree):
    result = embed_symptom_in_prompt_v1(["test"], "HP:0001234")
    assert "Fever" in result[0]


@pytest.mark.unit
def test_prompt_contains_sentence_text(patch_prompting_hpo_tree):
    result = embed_symptom_in_prompt_v1(["unique_marker_xyz_42"], "HP:0001234")
    assert "unique_marker_xyz_42" in result[0]


@pytest.mark.unit
def test_prompt_with_definition_includes_defined_as(patch_prompting_hpo_tree):
    # HP:0001234 has Def = ["Elevated body temperature..."]
    result = embed_symptom_in_prompt_v1(["test sentence"], "HP:0001234")
    assert "is defined as" in result[0]


@pytest.mark.unit
def test_prompt_with_definition_includes_definition_text(patch_prompting_hpo_tree):
    result = embed_symptom_in_prompt_v1(["test sentence"], "HP:0001234")
    assert "Elevated body temperature" in result[0]


@pytest.mark.unit
def test_prompt_without_definition_excludes_defined_as(patch_prompting_hpo_tree):
    # HP:0005678 has Def = []
    result = embed_symptom_in_prompt_v1(["test sentence"], "HP:0005678")
    assert "is defined as" not in result[0]


@pytest.mark.unit
def test_prompt_with_synonyms_includes_referred_to_as(patch_prompting_hpo_tree):
    # HP:0001234 has Synonym = ["High temperature", "Pyrexia"]
    result = embed_symptom_in_prompt_v1(["test"], "HP:0001234")
    assert "referred to as" in result[0]


@pytest.mark.unit
def test_prompt_with_synonyms_includes_synonym_text(patch_prompting_hpo_tree):
    result = embed_symptom_in_prompt_v1(["test"], "HP:0001234")
    assert "Pyrexia" in result[0] or "High temperature" in result[0]


@pytest.mark.unit
def test_prompt_without_synonyms_excludes_referred_to_as(patch_prompting_hpo_tree):
    # HP:0005678 has Synonym = []
    result = embed_symptom_in_prompt_v1(["test"], "HP:0005678")
    assert "referred to as" not in result[0]


@pytest.mark.unit
def test_each_prompt_contains_its_own_sentence(patch_prompting_hpo_tree):
    sents = ["sentence_alpha", "sentence_beta"]
    result = embed_symptom_in_prompt_v1(sents, "HP:0001234")
    assert "sentence_alpha" in result[0]
    assert "sentence_beta" in result[1]
    # Sentences should not bleed across prompts
    assert "sentence_beta" not in result[0]
    assert "sentence_alpha" not in result[1]


# ---------------------------------------------------------------------------
# BaselinePrompter
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_baseline_prompter_returns_list(patch_prompting_hpo_tree):
    prompter = BaselinePrompter()
    result = prompter.build_prompts(["test"], "HP:0001234")
    assert isinstance(result, list)


@pytest.mark.unit
def test_baseline_prompter_same_length_as_sents(patch_prompting_hpo_tree):
    prompter = BaselinePrompter()
    result = prompter.build_prompts(["a", "b", "c"], "HP:0001234")
    assert len(result) == 3


@pytest.mark.unit
def test_baseline_prompter_contains_hpo_label(patch_prompting_hpo_tree):
    prompter = BaselinePrompter()
    result = prompter.build_prompts(["test"], "HP:0001234")
    assert "Fever" in result[0]


@pytest.mark.unit
def test_baseline_prompter_delegates_to_embed_function(patch_prompting_hpo_tree):
    """BaselinePrompter.build_prompts and embed_symptom_in_prompt_v1 must agree."""
    sents = ["some text"]
    prompter = BaselinePrompter()
    direct = embed_symptom_in_prompt_v1(sents, "HP:0001234")
    via_class = prompter.build_prompts(sents, "HP:0001234")
    assert direct == via_class
