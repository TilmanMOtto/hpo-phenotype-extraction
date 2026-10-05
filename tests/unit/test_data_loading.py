"""Unit tests for hpo_extraction.data.loading, load_txt, prepare_context, LazyContextDict."""

import os
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from hpo_extraction.data.loading import LazyContextDict, load_txt, prepare_context


# ---------------------------------------------------------------------------
# load_txt
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_txt_key_converts_underscore_to_colon(tmp_path):
    (tmp_path / "HP_0001234.txt").write_text("content", encoding="latin1")
    result = load_txt(str(tmp_path))
    assert "HP:0001234" in result


@pytest.mark.unit
def test_load_txt_value_is_file_content(tmp_path):
    (tmp_path / "HP_0001234.txt").write_text("hello world", encoding="latin1")
    result = load_txt(str(tmp_path))
    assert result["HP:0001234"] == "hello world"


@pytest.mark.unit
def test_load_txt_multiple_files(tmp_path):
    (tmp_path / "HP_0001.txt").write_text("a", encoding="latin1")
    (tmp_path / "HP_0002.txt").write_text("b", encoding="latin1")
    result = load_txt(str(tmp_path))
    assert len(result) == 2
    assert "HP:0001" in result
    assert "HP:0002" in result


@pytest.mark.unit
def test_load_txt_ignores_non_txt_files(tmp_path):
    (tmp_path / "HP_0001.txt").write_text("ok", encoding="latin1")
    (tmp_path / "notes.csv").write_text("skip", encoding="latin1")
    (tmp_path / "README.md").write_text("skip", encoding="latin1")
    result = load_txt(str(tmp_path))
    assert len(result) == 1


@pytest.mark.unit
def test_load_txt_empty_directory_returns_empty_dict(tmp_path):
    assert load_txt(str(tmp_path)) == {}


@pytest.mark.unit
def test_load_txt_multiple_underscores_in_filename(tmp_path):
    # Only the first dot separates name from extension
    (tmp_path / "HP_000_1234.txt").write_text("x", encoding="latin1")
    result = load_txt(str(tmp_path))
    # Key: "HP_000_1234" → "HP:000:1234"
    assert "HP:000:1234" in result


@pytest.mark.unit
def test_load_txt_reads_latin1_encoding(tmp_path):
    # latin1 byte 0xe9 = é
    content = "caf\xe9"
    (tmp_path / "test.txt").write_bytes(content.encode("latin1"))
    result = load_txt(str(tmp_path))
    assert result["test"] == content


# ---------------------------------------------------------------------------
# prepare_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prepare_context_strips_asterisk_bullet():
    d = {"HP:1": "* item one\n* item two"}
    result = prepare_context(d)
    assert result["HP:1"] == ["item one", "item two"]


@pytest.mark.unit
def test_prepare_context_strips_dash_bullet():
    d = {"HP:1": "- item one"}
    result = prepare_context(d)
    assert result["HP:1"] == ["item one"]


@pytest.mark.unit
def test_prepare_context_strips_bullet_char():
    d = {"HP:1": "• item a"}
    result = prepare_context(d)
    assert result["HP:1"] == ["item a"]


@pytest.mark.unit
def test_prepare_context_strips_middle_dot_bullet():
    d = {"HP:1": "· item b"}
    result = prepare_context(d)
    assert result["HP:1"] == ["item b"]


@pytest.mark.unit
def test_prepare_context_discards_empty_lines():
    d = {"HP:1": "item one\n\n\nitem two"}
    result = prepare_context(d)
    assert result["HP:1"] == ["item one", "item two"]


@pytest.mark.unit
def test_prepare_context_plain_line_preserved_as_is():
    d = {"HP:1": "no bullet here"}
    result = prepare_context(d)
    assert result["HP:1"] == ["no bullet here"]


@pytest.mark.unit
def test_prepare_context_modifies_dict_in_place():
    d = {"HP:1": "line"}
    returned = prepare_context(d)
    assert returned is d


@pytest.mark.unit
def test_prepare_context_empty_string_yields_empty_list():
    d = {"HP:1": ""}
    result = prepare_context(d)
    assert result["HP:1"] == []


@pytest.mark.unit
def test_prepare_context_multiple_hpo_keys():
    d = {
        "HP:1": "* fever",
        "HP:2": "- headache",
    }
    result = prepare_context(d)
    assert result["HP:1"] == ["fever"]
    assert result["HP:2"] == ["headache"]


@pytest.mark.unit
def test_prepare_context_strips_leading_whitespace_after_bullet():
    d = {"HP:1": "*    padded item"}
    result = prepare_context(d)
    assert result["HP:1"] == ["padded item"]


@pytest.mark.unit
def test_prepare_context_mixed_bullets_in_same_entry():
    d = {"HP:1": "* one\n- two\n• three\n· four"}
    result = prepare_context(d)
    assert result["HP:1"] == ["one", "two", "three", "four"]


# ---------------------------------------------------------------------------
# LazyContextDict
# ---------------------------------------------------------------------------

def _make_model(embedding=None):
    model = MagicMock()
    model.encode.return_value = np.zeros((3, 4)) if embedding is None else embedding
    return model


def _write_hpo(tmp_path, stem="HP_0001234", content="* sentence one\n* sentence two\n* sentence three"):
    (tmp_path / f"{stem}.txt").write_bytes(content.encode("latin1"))
    return stem.replace("_", ":")  # return HPO key


@pytest.mark.unit
def test_lazy_first_access_calls_encode_and_writes_npy(tmp_path):
    key = _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    result = lcd[key]

    model.encode.assert_called_once()
    assert (tmp_path / ".cache" / "HP_0001234.npy").exists()
    assert isinstance(result, np.ndarray)


@pytest.mark.unit
def test_lazy_second_access_uses_memory_cache(tmp_path):
    key = _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    lcd[key]
    lcd[key]

    assert model.encode.call_count == 1


@pytest.mark.unit
def test_lazy_disk_cache_hit_skips_encode(tmp_path):
    key = _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)
    lcd[key]  # populate disk cache

    # New instance: no in-memory cache, but .npy exists and is newer
    model2 = _make_model()
    lcd2 = LazyContextDict(str(tmp_path), model2)
    lcd2[key]

    model2.encode.assert_not_called()


@pytest.mark.unit
def test_lazy_stale_npy_triggers_reencode(tmp_path):
    key = _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)
    lcd[key]  # writes .npy

    # Touch source .txt to make it newer than the .npy
    time.sleep(0.01)
    txt_path = tmp_path / "HP_0001234.txt"
    txt_path.touch()

    model2 = _make_model()
    lcd2 = LazyContextDict(str(tmp_path), model2)
    lcd2[key]

    model2.encode.assert_called_once()


@pytest.mark.unit
def test_lazy_missing_key_raises_key_error(tmp_path):
    _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    with pytest.raises(KeyError):
        lcd["HP:9999999"]


@pytest.mark.unit
def test_lazy_contains_txt_backed_key(tmp_path):
    key = _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    assert key in lcd
    assert "HP:9999999" not in lcd


@pytest.mark.unit
def test_lazy_set_encoded_accessible(tmp_path):
    _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    injected = np.ones((2, 4))
    lcd.set_encoded("HP:custom", injected)

    assert "HP:custom" in lcd
    np.testing.assert_array_equal(lcd["HP:custom"], injected)
    model.encode.assert_not_called()


@pytest.mark.unit
def test_lazy_keys_includes_txt_files(tmp_path):
    _write_hpo(tmp_path, "HP_0001234")
    _write_hpo(tmp_path, "HP_0005678")
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    assert {"HP:0001234", "HP:0005678"} == set(lcd.keys())


@pytest.mark.unit
def test_lazy_cache_dir_created_automatically(tmp_path):
    key = _write_hpo(tmp_path)
    model = _make_model()
    lcd = LazyContextDict(str(tmp_path), model)

    assert not (tmp_path / ".cache").exists()
    lcd[key]
    assert (tmp_path / ".cache").exists()


# ---------------------------------------------------------------------------
# getReports
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_reports_returns_two_dicts(tmp_path):
    from hpo_extraction.data.loading import getReports

    # Create patient subfolder with translation file
    patient_dir = tmp_path / "patient_001"
    patient_dir.mkdir()
    (patient_dir / "report_translation.txt").write_text("Patient has fever.")

    report_dict, summary_dict = getReports(str(tmp_path))
    assert "patient_001" in report_dict
    assert "patient_001" in summary_dict


@pytest.mark.unit
def test_get_reports_ignores_non_translation_files(tmp_path):
    from hpo_extraction.data.loading import getReports

    patient_dir = tmp_path / "p1"
    patient_dir.mkdir()
    (patient_dir / "notes.txt").write_text("ignored")
    (patient_dir / "report_translation.txt").write_text("included")

    report_dict, _ = getReports(str(tmp_path))
    assert len(report_dict["p1"]) == 1


@pytest.mark.unit
def test_get_reports_concatenates_multiple_files(tmp_path):
    from hpo_extraction.data.loading import getReports

    patient_dir = tmp_path / "p1"
    patient_dir.mkdir()
    (patient_dir / "a_translation.txt").write_text("Part A.")
    (patient_dir / "b_translation.txt").write_text("Part B.")

    _, summary_dict = getReports(str(tmp_path))
    assert "Part A." in summary_dict["p1"]
    assert "Part B." in summary_dict["p1"]


@pytest.mark.unit
def test_get_reports_empty_dir_returns_empty_dicts(tmp_path):
    from hpo_extraction.data.loading import getReports

    report_dict, summary_dict = getReports(str(tmp_path))
    assert report_dict == {}
    assert summary_dict == {}
