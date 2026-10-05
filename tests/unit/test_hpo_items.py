"""Unit tests for hpo_extraction.ontology.hpo_items, string utilities, WordItem, PhraseItem.

NLTK resources are used from the local environment (no network. Already installed).
"""

import pytest

from hpo_extraction.ontology.hpo_items import (
    PhraseItem,
    WordItem,
    containNum,
    getNegativeWords,
    isNum,
    processStr,
    strip_accents,
)


# ---------------------------------------------------------------------------
# strip_accents
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_strip_accents_removes_accent_from_e():
    assert strip_accents("café") == "cafe"


@pytest.mark.unit
def test_strip_accents_removes_accent_from_n():
    assert strip_accents("señor") == "senor"


@pytest.mark.unit
def test_strip_accents_ascii_string_unchanged():
    assert strip_accents("hello world") == "hello world"


@pytest.mark.unit
def test_strip_accents_empty_string_returns_empty():
    assert strip_accents("") == ""


# ---------------------------------------------------------------------------
# processStr
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_processStr_lowercases_input():
    tokens = processStr("FEVER")
    assert all(t == t.lower() for t in tokens)


@pytest.mark.unit
def test_processStr_strips_accents():
    tokens = processStr("Café")
    assert "cafe" in tokens


@pytest.mark.unit
def test_processStr_removes_hyphen_between_uppercase_and_digit():
    # "HP-123" → re removes hyphen between uppercase letter and digit → "HP123" → "hp123"
    tokens = processStr("HP-123")
    assert "hp123" in tokens


@pytest.mark.unit
def test_processStr_replaces_dash_with_space_in_lowercase_word():
    # "hypo-tension", no uppercase before dash, so hyphen → space → two tokens
    tokens = processStr("hypo-tension")
    assert "hypo" in tokens
    assert "tension" in tokens


@pytest.mark.unit
def test_processStr_splits_on_whitespace():
    tokens = processStr("fever headache nausea")
    assert tokens == ["fever", "headache", "nausea"]


@pytest.mark.unit
def test_processStr_returns_list():
    assert isinstance(processStr("test"), list)


@pytest.mark.unit
def test_processStr_strips_leading_trailing_whitespace():
    tokens = processStr("  fever  ")
    assert tokens == ["fever"]


# ---------------------------------------------------------------------------
# isNum
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [
    ("3.14", True),
    ("42", True),
    ("-1", True),
    ("0", True),
    ("1e3", True),
    ("abc", False),
    ("3abc", False),
    ("", False),
])
def test_isNum_parametrized(value, expected):
    assert isNum(value) is expected


# ---------------------------------------------------------------------------
# containNum
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [
    ("abc3def", True),
    ("123", True),
    ("abc", False),
    ("", False),
    ("!", False),
])
def test_containNum_parametrized(value, expected):
    assert containNum(value) is expected


# ---------------------------------------------------------------------------
# getNegativeWords
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_getNegativeWords_returns_set():
    result = getNegativeWords()
    assert isinstance(result, set)


@pytest.mark.unit
@pytest.mark.parametrize("word", ["no", "not", "none", "negative", "never"])
def test_getNegativeWords_contains_core_negatives(word):
    assert word in getNegativeWords()


# ---------------------------------------------------------------------------
# WordItem
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_word_item_stores_text_as_lowercase():
    wi = WordItem("Hello", 0, 5)
    assert wi.text == "hello"


@pytest.mark.unit
def test_word_item_already_lowercase_unchanged():
    wi = WordItem("fever", 0, 5)
    assert wi.text == "fever"


@pytest.mark.unit
def test_word_item_stores_start_position():
    wi = WordItem("word", 10, 14)
    assert wi.start == 10


@pytest.mark.unit
def test_word_item_stores_end_position():
    wi = WordItem("word", 10, 14)
    assert wi.end == 14


# ---------------------------------------------------------------------------
# PhraseItem
# ---------------------------------------------------------------------------

@pytest.fixture
def two_word_phrase():
    """PhraseItem with two words at positions [0,5) and [6,11)."""
    words = [WordItem("hello", 0, 5), WordItem("world", 6, 11)]
    return PhraseItem(words)


@pytest.fixture
def single_word_phrase():
    return PhraseItem([WordItem("pain", 0, 4)])


@pytest.mark.unit
def test_phrase_item_to_string(two_word_phrase):
    assert two_word_phrase.toString() == "hello world"


@pytest.mark.unit
def test_phrase_item_len_returns_word_count(two_word_phrase):
    assert len(two_word_phrase) == 2


@pytest.mark.unit
def test_phrase_item_single_word_len(single_word_phrase):
    assert len(single_word_phrase) == 1


@pytest.mark.unit
def test_phrase_item_no_flag_initially_false(single_word_phrase):
    assert single_word_phrase.no_flag is False


@pytest.mark.unit
def test_phrase_item_set_no_flag(single_word_phrase):
    single_word_phrase.set_no_flag()
    assert single_word_phrase.no_flag is True


@pytest.mark.unit
def test_phrase_item_locs_set_contains_word_starts(two_word_phrase):
    assert 0 in two_word_phrase.locs_set
    assert 6 in two_word_phrase.locs_set


@pytest.mark.unit
def test_phrase_item_start_loc_is_first_word_start(two_word_phrase):
    assert two_word_phrase.start_loc == 0


@pytest.mark.unit
def test_phrase_item_end_loc_is_last_word_end(two_word_phrase):
    assert two_word_phrase.end_loc == 11


@pytest.mark.unit
def test_phrase_item_issubset_true_for_proper_subset():
    # p1 = {0}, p2 = {0, 6} → p1 ⊆ p2
    p1 = PhraseItem([WordItem("a", 0, 1)])
    p2 = PhraseItem([WordItem("a", 0, 1), WordItem("b", 6, 7)])
    assert p1.issubset(p2) is True


@pytest.mark.unit
def test_phrase_item_issubset_false_for_superset():
    p1 = PhraseItem([WordItem("a", 0, 1)])
    p2 = PhraseItem([WordItem("a", 0, 1), WordItem("b", 6, 7)])
    assert p2.issubset(p1) is False


@pytest.mark.unit
def test_phrase_item_issubset_true_for_equal_sets():
    p1 = PhraseItem([WordItem("a", 0, 1)])
    p2 = PhraseItem([WordItem("a", 0, 1)])
    assert p1.issubset(p2) is True


@pytest.mark.unit
def test_phrase_item_include_returns_true_for_subset():
    p1 = PhraseItem([WordItem("a", 0, 1)])
    p2 = PhraseItem([WordItem("a", 0, 1), WordItem("b", 6, 7)])
    assert p1.include(p2) is True


@pytest.mark.unit
def test_phrase_item_include_returns_true_for_superset():
    p1 = PhraseItem([WordItem("a", 0, 1)])
    p2 = PhraseItem([WordItem("a", 0, 1), WordItem("b", 6, 7)])
    assert p2.include(p1) is True


@pytest.mark.unit
def test_phrase_item_include_returns_false_for_disjoint():
    p1 = PhraseItem([WordItem("a", 0, 1)])
    p2 = PhraseItem([WordItem("b", 5, 6)])
    assert p1.include(p2) is False


@pytest.mark.unit
def test_phrase_item_to_simple_string_excludes_stopwords():
    # "the" is a stopword; "fever" is not
    words = [WordItem("the", 0, 3), WordItem("fever", 4, 9)]
    phrase = PhraseItem(words)
    simple = phrase.toSimpleString()
    assert "fever" in simple
    assert "the" not in simple


@pytest.mark.unit
def test_phrase_item_to_simple_string_excludes_numeric_tokens():
    words = [WordItem("42", 0, 2), WordItem("degrees", 3, 10)]
    phrase = PhraseItem(words)
    simple = phrase.toSimpleString()
    assert "42" not in simple
    assert "degrees" in simple
