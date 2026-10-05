"""A synthetic HCY-shaped cohort, written to a temp directory.

The real data lives on the cluster and is patient data, so neither CI nor a laptop can open it. The
fixture exists so the acceptance check has something to run against everywhere, and it is built to
contain the awkward cases rather than a happy path, a screen that only ever renders well-formed
input is a screen whose failure modes are discovered by a curator.

What it contains:

* annotation files that **disagree** (the case the whole app is about), a term only ``prior_annotation``
  carries, one only ``daphne`` carries, one only ``prior_annotation_2`` carries;
* a prior_annotation row whose ``char_offset`` is **wrong**, so it has to be checked against the trigger
  word and repaired, not trusted;
* a prior_annotation row whose trigger word occurs **nowhere in the report**, so the unplaced path is
  exercised and a lexical candidate is still offered for it;
* a prior_annotation code annotated **twice** in one report, so the ``slot`` in the key is exercised and
  two marks compete for overlapping characters;
* a daphne row whose ``segment_idx`` points at the **wrong segment**, so the fallback to its own
  ``segment`` column as context is exercised;
* a daphne row that is **not confirmed**;
* a patient with **no annotation at all** and one with **no PhenoBERT detection**;
* a **negated** detection and one whose code the ontology does not resolve;
* a segment that **cannot be aligned** to the verbatim report, so the no-highlight path is
  exercised, not assumed;
* a patient present in the annotation files but **absent from the segmentation**, whose report text
  therefore comes from the prior_annotation file, not from a staged PhenoBERT copy;
* a patient the **confirmed pass never reached**, prior_annotation and prior_annotation_2 rows, no daphne row at all, so "disagree" has to be computed against the sources that carry something for *this* report,
  not against every file that loaded;
* **two different phenotypes annotated on the same words**, which marks cannot nest to express and
  which used to be resolved by drawing one of them and dropping the other;
* **two different phenotypes on *nested* words**, one trigger a substring of the other, which is
  the same problem one step harder: the wider span has to take the underline and carry both tags;
* a prior_annotation row with **no trigger word at all**, which has neither a segment nor words and so
  belongs in the locating queue even though nothing about it ever failed to be found;
* **one code carried by both files with two different trigger words, neither of which occurs in the
  report**, two broken records of one annotation, which comparing the files' claims turns into two
  things to do.
"""

from __future__ import annotations

import csv
import json
import os

#: Report text, verbatim, this is what a PhenoBERT ``phenobert_input/<stem>.txt`` holds, and what
#: The holistic file repeats in its ``report_text`` column.
REPORTS = {
    "SYN001": (
        "Case history: the boy is 4 years old.\n"
        "He presented with recurrent seizures since age two.\n"
        "He was markedly hypotonic at birth.\n"
        "There was no evidence of thrombocytopenia.\n"
    ),
    "SYN002": (
        "The girl showed marked global developmental delay.\n"
        "Feeding was difficult throughout infancy.\n"
    ),
    "SYN003": (
        "Routine follow-up. Nothing abnormal was noted on examination.\n"
    ),
    # No segmentation and no staged PhenoBERT report: everything this patient has comes from the
    # holistic file, which is the case that used to render as a bare list of nothing.
    "SYN004": (
        "First visit at eleven weeks. Recurrent convulsions occurred during sleep.\n"
    ),
    # The confirmed pass never reached this report. Everything it has comes from holistic and
    # marc2, and those two do have something to compare, which is the case that used to put a
    # "disagree" chip on every term of it, reporting the sidebar's configuration as a finding.
    "SYN005": (
        "Referred at nine months.\n"
        "She was persistently hypotonic and could not hold her head up.\n"
        "Growth has been along the third centile.\n"
    ),
}

#: Segments as ``experiments/03_setup/segment_reports.py`` would write them, tokenized, so whitespace differs
#: from the report. SYN001 gets one segment that is *not* in its report at all, and SYN004 gets no
#: segmentation whatsoever.
SEGMENTS = {
    "SYN001": [
        "Case history: the boy is 4 years old.",
        "He presented with recurrent seizures since age two.",
        "He was markedly hypotonic at birth.",
        "There was no evidence of thrombocytopenia.",
        "A segment that never appears in the report.",
    ],
    "SYN002": [
        "The girl showed marked global developmental delay.",
        "Feeding was difficult throughout infancy.",
    ],
    "SYN003": ["Routine follow-up.", "Nothing abnormal was noted on examination."],
    "SYN005": [
        "Referred at nine months.",
        "She was persistently hypotonic and could not hold her head up.",
        "Growth has been along the third centile.",
    ],
}

#: ``(patient, code, name, trigger, context, offset)``. ``offset`` may be ``None`` (the file said
#: nothing) or a **wrong** number, and both must be survivable, see ``anchors.resolve_offset``.
#:
#: ``HP:0001873`` on SYN001 is annotated on a word the report does not contain, which is what a
#: report re-exported after annotation looks like. It must come back unplaced, not land
#: somewhere plausible.
HOLISTIC: list[tuple] = [
    ("SYN001", "HP:0001250", "Seizure", "seizures", "He presented with recurrent seizures", "auto"),
    # A **nested** trigger with a *different* code: "seizures" sits inside "recurrent seizures", so
    # The two spans overlap without being equal. Marks cannot nest, so the wider one has to take the
    # underline and carry both tags, which is what the whole overlap pass in ``reader._marks_for``
    # exists to get right, and what it used to get wrong by dropping one of the two outright.
    ("SYN001", "HP:0007359", "Focal-onset seizure", "recurrent seizures",
     "He presented with recurrent seizures since age two.", "auto"),
    ("SYN001", "HP:0001252", "Hypotonia", "hypotonic", "He was markedly hypotonic at birth.", 0),
    ("SYN001", "HP:0001873", "Thrombocytopenia", "purpura", "", None),
    ("SYN002", "HP:0001263", "Global developmental delay", "developmental delay",
     "The girl showed marked global developmental delay.", "auto"),
    # The same code again, on overlapping words: two annotations, two keys, one mark surviving.
    ("SYN002", "HP:0001263", "Global developmental delay", "delay", "", "auto"),
    # A row with **no trigger word at all**. Its file named a code and nothing else, so it has
    # neither a segment nor words, the third state the locating queue has to catch, and the one
    # that reads as "already fine" to anything testing only whether a stated trigger could be found.
    ("SYN002", "HP:0002187", "Profound global developmental delay", "", "", None),
    # Two files, one code, **two different trigger words, neither of which occurs in the report**.
    # Comparing what the files claim makes this two annotations, because neither claim
    # works, and a curator is then asked to place one phenotype twice. See ``DAPHNE`` below for
    # The other half of the pair, and ``sources.same_annotation`` for why it is one.
    ("SYN002", "HP:0001250", "Seizure", "convulsions", "", None),
    ("SYN004", "HP:0001250", "Seizure", "convulsions", "Recurrent convulsions occurred during sleep.",
     "auto"),
    # Two *different* phenotypes on the same words. Marks cannot nest, so this has to become one
    # mark carrying two tags, it used to be a collision the overlap pass resolved by drawing one
    # of them and silently losing the other.
    ("SYN005", "HP:0001252", "Hypotonia", "hypotonic",
     "She was persistently hypotonic and could not hold her head up.", "auto"),
    ("SYN005", "HP:0003324", "Generalized muscle weakness", "hypotonic",
     "She was persistently hypotonic and could not hold her head up.", "auto"),
    ("SYN005", "HP:0004322", "Short stature", "third centile",
     "Growth has been along the third centile.", "auto"),
]

#: ``(patient, segment_idx, code, name, trigger, provenance, confirmed)``. The SYN001 hypotonia row
#: names the wrong segment on purpose: its own ``segment`` column has to rescue it, which is the
#: only route by which the ``context`` strategy is reached.
DAPHNE: list[tuple] = [
    ("SYN001", 1, "HP:0001250", "Seizure", "seizures", "manual", True),
    ("SYN001", 0, "HP:0001252", "Hypotonia", "hypotonic", "manual", True),
    ("SYN002", 1, "HP:0011968", "Feeding difficulties", "Feeding", "model", False),
    # The other half of the broken pair: same code as the holistic row above, different words,
    # equally absent from the report.
    ("SYN002", 0, "HP:0001250", "Seizure", "fits", "model", True),
    ("SYN004", 2, "HP:0001250", "Seizure", "convulsions", "manual", True),
]

#: The one code-only source left. It carries a term nothing in SYN001's text supports
#: (``HP:0001903``, anemia), which is what keeps the lexical-candidate and deletion paths live.
GOLD_MARC2 = {
    "SYN001": ["HP:0001250", "HP:0001252", "HP:0001903"],
    "SYN002": ["HP:0001263", "HP:0011968"],
    "SYN003": [],
    "SYN004": ["HP:0001250"],
    # Agrees with holistic on two of three. With no daphne row for this patient, that is the whole
    # comparison, and the one term they differ on is the only one that should read as a
    # disagreement.
    "SYN005": ["HP:0001252", "HP:0003324", "HP:0001903"],
}

#: The segment a daphne row claims to come from, for its ``segment`` column. Taken from SEGMENTS
#: where the index is valid, so the context is a real sentence even when the index is wrong.
def _segment_text(patient_id: str, segment_idx: int, trigger: str) -> str:
    segments = SEGMENTS.get(patient_id, [])
    for segment in segments:
        if trigger.lower() in segment.lower():
            return segment
    return segments[segment_idx] if 0 <= segment_idx < len(segments) else ""


def _offset(patient_id: str, trigger: str, stated) -> str:
    """The ``char_offset`` cell: ``"auto"`` means the true one, a number means take it literally."""
    if stated is None:
        return ""
    if stated != "auto":
        return str(stated)
    text = REPORTS.get(patient_id, "")
    at = text.lower().find(trigger.lower())
    return str(at) if at >= 0 else ""


def _detections() -> list[dict]:
    rows = []

    def add(report_id, phrase, hpo_id, label, negated=False, resolved=True):
        text = REPORTS[report_id]
        start = text.index(phrase)
        rows.append({
            "report_id": report_id, "hpo_id": hpo_id, "hpo_label": label,
            "raw_hpo_id": hpo_id, "resolved": resolved, "phrase": phrase,
            "start": start, "end": start + len(phrase), "score": None, "negated": negated,
        })

    add("SYN001", "seizures", "HP:0001250", "seizure")
    add("SYN001", "hypotonic", "HP:0001252", "hypotonia")
    add("SYN001", "thrombocytopenia", "HP:0001873", "thrombocytopenia", negated=True)
    add("SYN002", "global developmental delay", "HP:0001263", "global developmental delay")
    # A code this hpo.json does not resolve, PhenoBERT ships an older HPO release.
    add("SYN002", "Feeding", "HP:9999999", "feeding difficulties", resolved=False)
    return rows


def write(directory: str, with_phenobert: bool = True) -> dict:
    """Write the fixture under *directory* and return the path set the app takes."""
    hcy_dir = os.path.join(directory, "hcy")
    pb_dir = os.path.join(directory, "output", "baseline_phenobert", "hcy")
    os.makedirs(hcy_dir, exist_ok=True)

    segments_path = os.path.join(hcy_dir, "segmented_reports.csv")
    with open(segments_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "sentence_idx", "sentence"])
        for patient_id in sorted(SEGMENTS):
            for idx, sentence in enumerate(SEGMENTS[patient_id]):
                writer.writerow([patient_id, idx, sentence])

    prior_annotation_path = os.path.join(hcy_dir, "hcy_holistic_ground_truth.csv")
    with open(prior_annotation_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "report_text", "hpo_code", "hpo_name", "trigger_word",
                         "sentence_context", "char_offset"])
        for patient_id, code, name, trigger, context, offset in HOLISTIC:
            writer.writerow([patient_id, REPORTS.get(patient_id, ""), code, name, trigger,
                             context, _offset(patient_id, trigger, offset)])

    confirmed_path = os.path.join(hcy_dir, "annotations_confirmed.csv")
    with open(confirmed_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "segment", "segment_idx", "hpo_code", "hpo_name",
                         "trigger_word", "provenance", "confirmed"])
        for patient_id, segment_idx, code, name, trigger, provenance, confirmed in DAPHNE:
            # "True." with the trailing full stop, because that is what the real file holds and a
            # loader that read it as unconfirmed would silently demote every row.
            writer.writerow([patient_id, _segment_text(patient_id, segment_idx, trigger),
                             segment_idx, code, name, trigger, provenance,
                             "True." if confirmed else "False"])

    prior_annotation_2_path = os.path.join(hcy_dir, "hcy_ground_truth_marc2.csv")
    with open(prior_annotation_2_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "hpo_codes"])
        for patient_id in sorted(GOLD_MARC2):
            writer.writerow([patient_id, ";".join(GOLD_MARC2[patient_id])])

    if with_phenobert:
        os.makedirs(os.path.join(pb_dir, "phenobert_input"), exist_ok=True)
        # SYN004 is not staged: its verbatim text has to come from the holistic file.
        for report_id, text in REPORTS.items():
            if report_id == "SYN004":
                continue
            stem = report_id.replace(":", "_")
            with open(os.path.join(pb_dir, "phenobert_input", f"{stem}.txt"),
                      "w", encoding="utf-8") as handle:
                handle.write(text)

        detections = _detections()
        with open(os.path.join(pb_dir, "phenobert_detections.jsonl"), "w",
                  encoding="utf-8") as handle:
            for row in detections:
                handle.write(json.dumps(row) + "\n")

        with open(os.path.join(pb_dir, "phenobert_predictions.jsonl"), "w",
                  encoding="utf-8") as handle:
            for report_id in sorted(REPORTS):
                predicted = sorted({d["hpo_id"] for d in detections
                                    if d["report_id"] == report_id and not d["negated"]})
                handle.write(json.dumps({
                    "report_id": report_id, "summary": True,
                    "predicted_set": predicted,
                    "gold_set": sorted(GOLD_MARC2.get(report_id, [])),
                }) + "\n")

    return {
        "hcy_dir": hcy_dir,
        "segments": segments_path,
        "prior_annotation": prior_annotation_path,
        "confirmed": confirmed_path,
        "prior_annotation_2": prior_annotation_2_path,
        "phenobert": pb_dir if with_phenobert else "",
    }
