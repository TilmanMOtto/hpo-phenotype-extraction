"""The stored curation log uses two source labels that were renamed in this repository.

``holistic`` is now ``prior_annotation`` and ``marc2`` is ``prior_annotation_2``. The log on the
cluster is append-only and is not rewritten, so these tests check that reading an old log gives
the same state as reading the same log written with the new labels.
"""
import json

from hpo_extraction.curation import store
from hpo_extraction.curation.labels import translate_stored_labels

OLD_EVENTS = [
    {"event_id": "e1", "action": "reject", "patient_id": "SYN004", "ts": "t1",
     "target_key": "holistic|SYN004|HP:0001250", "reason": "not in the report"},
    {"event_id": "e2", "action": "approve", "patient_id": "SYN004", "ts": "t2",
     "target_key": "marc2|SYN004|HP:0001252"},
    {"event_id": "e3", "action": "approve", "patient_id": "SYN005", "ts": "t3",
     "target_key": "daphne|SYN005|HP:0004322"},
]


def _new(event):
    text = json.dumps(event).replace('"holistic|', '"prior_annotation|').replace(
        '"marc2|', '"prior_annotation_2|')
    return json.loads(text)


def _write(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return str(path)


def test_old_and_new_labels_fold_to_the_same_state(tmp_path):
    old = store.fold_events(store.read_events(_write(tmp_path / "old.jsonl", OLD_EVENTS)))
    new = store.fold_events(store.read_events(
        _write(tmp_path / "new.jsonl", [_new(e) for e in OLD_EVENTS])))
    assert old == new


def test_only_the_two_renamed_labels_change():
    assert translate_stored_labels("holistic") == "prior_annotation"
    assert translate_stored_labels("marc2|SYN001|HP:0000001") == "prior_annotation_2|SYN001|HP:0000001"
    for unchanged in ("daphne", "daphne|SYN001|HP:0000001", "new", "phenobert|SYN1|HP:1|3",
                      "holistically", "a holistic note"):
        assert translate_stored_labels(unchanged) == unchanged


def test_nested_values_are_translated():
    event = {"target_key": "holistic|SYN001|HP:0000001", "claims": [{"source": "marc2"}]}
    assert translate_stored_labels(event) == {
        "target_key": "prior_annotation|SYN001|HP:0000001",
        "claims": [{"source": "prior_annotation_2"}]}
