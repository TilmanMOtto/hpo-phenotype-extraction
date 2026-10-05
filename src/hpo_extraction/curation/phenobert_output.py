"""The PhenoBERT baseline PhenoBERT-standalone baseline, loaded beside the Free Listing generation run ensemble.

The PhenoBERT baseline runs PhenoBERT over the **raw report**. The Free Listing generation run runs it over what eight SLMs *wrote*
about each sentence. Same grounder, opposite inputs, which makes the PhenoBERT baseline the one baseline the
ensemble has to beat, and the reason this app needs to read a second experiment's output.

Three things it gives that nothing in the Free Listing generation run directory can:

``phenobert_input/<stem>.txt``
    the report text **verbatim**, ``phenobert_experiment.stage_reports`` writes the report
    unchanged, and a driver test pins the bytes. The Free Listing generation run only holds sentences, so this is
    the app's only source for "the report as a document".
``phenobert_detections.jsonl``
    one row per detection with ``start``/``end`` indexing that file, so ``text[start:end] ==
    phrase`` holds and a highlight needs no string search. It also keeps the two kinds
    of evidence the predicted set throws away: ``negated`` mentions (PhenoBERT found the phenotype
    and ruled it out) and ``resolved: false`` codes (PhenoBERT ships an older HPO release).
``phenobert_predictions.jsonl``
    a ``summary`` line per report carrying ``predicted_set`` and ``gold_set``, written for every
    report the driver processed *including* the ones that predicted nothing. Those lines are
    authoritative. The per-term rows are a projection of them.

Stdlib only, like :mod:`loaders`, this app must stay loadable without torch, mlflow or stanza.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

#: The experiment directory this module reads.
EXP_ID = "baseline_phenobert"

_PREDICTIONS = "phenobert_predictions.jsonl"
_DETECTIONS = "phenobert_detections.jsonl"
_TIMING = "phenobert_timing.jsonl"
_INPUT_DIR = "phenobert_input"

#: How far up from a Free Listing generation run directory to look for the PhenoBERT baseline sibling. Three levels covers
#: every layout ``loaders.find_runs`` accepts: ``<base>/phenojury_generation_free_listing/<cohort>`` needs two,
#: ``<base>/<cohort>`` one, a bare cohort directory zero, plus one of slack.
_MAX_WALK_UP = 4

_WS = re.compile(r"\s+")


def _normalise(s: str) -> str:
    """The whitespace-insensitive comparison ``phenobert_runner._sentence_from_offset`` uses.

    Kept identical on purpose: the driver decides an offset is trustworthy by this rule, and a UI
    that used a stricter one would call the driver's own accepted attributions broken.
    """
    return _WS.sub(" ", s).strip().lower()


def safe_name(report_id: str) -> str:
    """``phenobert_experiment.safe_name``, a report id as a filename stem."""
    return report_id.replace(":", "_")


# ──────────────────────────────────────────────────────────────────────────────
# discovery
# ──────────────────────────────────────────────────────────────────────────────
def is_run_dir(path: str) -> bool:
    """True if *path* holds the one artifact this module cannot work without."""
    return bool(path) and os.path.isfile(os.path.join(path, _PREDICTIONS))


def find_run(run_dir: str, cohort: str | None = None, override: str | None = None) -> str | None:
    """The PhenoBERT baseline run directory matching a Free Listing generation run one, or ``None``.

    With *override* set, only the override is considered, a reader who typed a path wants that
    path, and silently falling back to a derived one would show them another run's numbers under
    the label they chose. Both ``<override>`` and ``<override>/<cohort>`` are accepted, so pointing
    at the experiment directory and at the cohort directory both work.

    Without one, the sibling is derived: walk up from *run_dir* looking for an ``baseline_phenobert``
    directory, then prefer its ``<cohort>`` subdirectory. Deriving rather than requiring a second
    path is what makes the comparison appear by itself on the cluster layout the runs actually use.
    """
    if override:
        for candidate in _override_candidates(override, cohort):
            if is_run_dir(candidate):
                return candidate
        return None

    if not run_dir:
        return None
    current = os.path.abspath(run_dir)
    for _ in range(_MAX_WALK_UP):
        current = os.path.dirname(current)
        if not current or current == os.path.dirname(current):
            break
        exp_dir = os.path.join(current, EXP_ID)
        if not os.path.isdir(exp_dir):
            continue
        for candidate in _sibling_candidates(exp_dir, cohort):
            if is_run_dir(candidate):
                return candidate
    return None


def _override_candidates(override: str, cohort: str | None) -> list[str]:
    out = [override]
    if cohort:
        out.insert(0, os.path.join(override, cohort))
        out.append(os.path.join(override, EXP_ID, cohort))
    out.append(os.path.join(override, EXP_ID))
    return out


def _sibling_candidates(exp_dir: str, cohort: str | None) -> list[str]:
    """``<the PhenoBERT baseline>/<cohort>`` first, then the experiment directory itself.

    The bare experiment directory is a real layout: a run launched with ``output_dir`` already
    pointing at one cohort writes its artifacts there directly.
    """
    out = []
    if cohort:
        out.append(os.path.join(exp_dir, cohort))
    out.append(exp_dir)
    return out


def searched_paths(run_dir: str, cohort: str | None = None,
                   override: str | None = None) -> list[str]:
    """Where :func:`find_run` would have looked, shown on screen when it found nothing.

    Naming the paths is the difference between "no baseline" and "no baseline *here*", and the
    second is the only one a reader can act on.
    """
    if override:
        return _override_candidates(override, cohort)

    paths = []
    current = os.path.abspath(run_dir) if run_dir else ""
    for _ in range(_MAX_WALK_UP):
        if not current or current == os.path.dirname(current):
            break
        current = os.path.dirname(current)
        paths.extend(_sibling_candidates(os.path.join(current, EXP_ID), cohort))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# reading
# ──────────────────────────────────────────────────────────────────────────────
def _read_jsonl(path: str, what: str) -> list[dict]:
    """Every parseable line. Torn lines are counted and skipped, never fatal."""
    records: list[dict] = []
    if not os.path.isfile(path):
        return records
    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        logger.warning("%s: skipped %d unparseable %s line(s)", os.path.basename(path), n_bad, what)
    return records


def load_predictions(run_dir: str) -> dict:
    """``{predicted, gold, labels, report_ids}`` from the summary lines.

    The summary lines are the authority: they list every report the driver processed, including the
    ones that predicted nothing, which the per-term rows cannot express. Term rows are read only for
    their ``hpo_label``, so a cohort loaded without the ontology still shows names.
    """
    predicted: dict[str, set[str]] = {}
    gold: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    order: list[str] = []

    for rec in _read_jsonl(os.path.join(run_dir, _PREDICTIONS), "prediction"):
        report_id = str(rec.get("report_id", ""))
        if not report_id:
            continue
        if rec.get("summary"):
            if report_id in predicted:
                continue            # first summary wins, as elsewhere in the earlier runs
            predicted[report_id] = set(rec.get("predicted_set") or ())
            gold[report_id] = set(rec.get("gold_set") or ())
            order.append(report_id)
        elif rec.get("hpo_id") and rec.get("hpo_label"):
            labels.setdefault(rec["hpo_id"], rec["hpo_label"])

    return {"predicted": predicted, "gold": gold, "labels": labels, "report_ids": order}


def load_detections(run_dir: str) -> list[dict]:
    """Every detection, negated and unresolved included, ``phenobert_experiment.collect_detections``.

    Rows: ``report_id, hpo_id, hpo_label, raw_hpo_id, resolved, phrase, start, end, score,
    negated``. ``score`` may be ``None``: it is PhenoBERT's fixed filter threshold, not a
    per-term confidence, and an install that writes no score column leaves it absent.
    """
    rows = []
    for rec in _read_jsonl(os.path.join(run_dir, _DETECTIONS), "detection"):
        rows.append({
            "report_id": str(rec.get("report_id", "")),
            "hpo_id": rec.get("hpo_id", ""),
            "hpo_label": rec.get("hpo_label", ""),
            "raw_hpo_id": rec.get("raw_hpo_id", rec.get("hpo_id", "")),
            "resolved": bool(rec.get("resolved", True)),
            "phrase": rec.get("phrase", ""),
            "start": int(rec.get("start", 0)),
            "end": int(rec.get("end", 0)),
            "score": rec.get("score"),
            "negated": bool(rec.get("negated", False)),
        })
    return rows


def load_texts(run_dir: str, report_ids) -> dict[str, str]:
    """``{report_id: report text}`` from ``phenobert_input/``.

    Looked up **per known report id** via :func:`safe_name`, never by inverting a filename: an HCY
    id can contain a colon that the stem replaced with an _ character, and two ids could stage to one
    stem (the driver refuses that case, but this module must not assume it was checked).
    """
    text_dir = os.path.join(run_dir, _INPUT_DIR)
    texts: dict[str, str] = {}
    if not os.path.isdir(text_dir):
        return texts
    for report_id in report_ids:
        path = os.path.join(text_dir, f"{safe_name(report_id)}.txt")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                texts[report_id] = f.read()
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
    return texts


def load_timing(run_dir: str) -> dict | None:
    """The single ``annotate`` row. The PhenoBERT baseline times one subprocess over the whole cohort."""
    rows = _read_jsonl(os.path.join(run_dir, _TIMING), "timing")
    return rows[0] if rows else None


def load(run_dir: str) -> dict:
    """Everything the views need, plus the span check that says whether it can be trusted."""
    preds = load_predictions(run_dir)
    detections = load_detections(run_dir)
    texts = load_texts(run_dir, preds["report_ids"])

    by_report: dict[str, list[dict]] = {}
    for row in detections:
        by_report.setdefault(row["report_id"], []).append(row)
    for rows in by_report.values():
        rows.sort(key=lambda r: (r["start"], r["end"]))

    span_checked, span_bad = _check_spans(detections, texts)

    labels = dict(preds["labels"])
    for row in detections:
        if row["hpo_label"]:
            labels.setdefault(row["hpo_id"], row["hpo_label"])

    return {
        "run_dir": run_dir,
        "predicted": preds["predicted"],
        "gold": preds["gold"],
        "report_ids": preds["report_ids"],
        "labels": labels,
        "detections": detections,
        "by_report": by_report,
        "texts": texts,
        "timing": load_timing(run_dir),
        "span_checked": span_checked,
        "span_bad": span_bad,
    }


def _check_spans(detections: list[dict], texts: dict[str, str]) -> tuple[int, list[dict]]:
    """Verify ``text[start:end] == phrase`` for every detection whose report text is on disk.

    This is the whole basis of the annotated-report panel. If it does not hold, the offsets belong
    to some other text, a stale ``phenobert_input/`` beside a rerun ``phenobert_output/``, say, and highlighting with them would underline the wrong words with complete confidence. Checked
    once at load, reported by gate G6, and the panel refuses to draw when it fails.
    """
    checked = 0
    bad = []
    for row in detections:
        text = texts.get(row["report_id"])
        if text is None:
            continue
        checked += 1
        got = text[row["start"]:row["end"]]
        if got != row["phrase"] and _normalise(got) != _normalise(row["phrase"]):
            bad.append({"report_id": row["report_id"], "hpo_id": row["hpo_id"],
                        "start": row["start"], "end": row["end"],
                        "expected": row["phrase"], "found": got})
    return checked, bad


def report_ok(pb: dict | None, report_id: str) -> bool:
    """True when *report_id*'s text and offsets are both present and consistent."""
    if not pb or report_id not in pb.get("texts", {}):
        return False
    return not any(b["report_id"] == report_id for b in pb["span_bad"])


# ──────────────────────────────────────────────────────────────────────────────
# tying the PhenoBERT baseline's report offsets to the Free Listing generation run's sentences
# ──────────────────────────────────────────────────────────────────────────────
def align_sentences(report_text: str, sentences: dict[int, str]) -> dict[int, tuple[int, int]] | None:
    """``{sentence_number: (start, end)}`` in *report_text*, or ``None`` if any sentence is unplaceable.

    The two experiments never exchanged offsets: the Free Listing generation run stores sentences, the PhenoBERT baseline stores the
    document they came from. Placing one in the other is what lets a report-level detection be
    attributed to a sentence, which is what the sentence picker and the term cross-filter both need.

    Matching is a forward scan with a moving cursor, so a sentence occurring twice is placed at its
    own occurrence, not at the first one. The fallback retries on whitespace-collapsed text,
    because the pipeline's splitter normalises runs of whitespace that the raw report still has.

    All-or-nothing on purpose: a partial map would silently drop evidence for the sentences it could
    not place, and a reader has no way to see the difference between "nothing here" and "not
    aligned". The caller degrades to a plain sentence list instead, and says so.
    """
    if not report_text or not sentences:
        return None

    spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    flat, raw_at = _collapse(report_text)
    flat_cursor = 0

    for sent_num in sorted(sentences):
        text = sentences[sent_num]
        if not text.strip():
            continue

        found = report_text.find(text, cursor)
        if found >= 0:
            spans[sent_num] = (found, found + len(text))
            cursor = found + len(text)
            flat_cursor = _flat_position(raw_at, cursor)
            continue

        needle, _ = _collapse(text)
        if not needle:
            continue
        hit = flat.find(needle, flat_cursor)
        if hit < 0:
            return None
        start = raw_at[hit]
        # The last matched character is never the synthetic space (`needle` is stripped), so its
        # raw offset + 1 is the exclusive end. Using the *next* index instead would swallow the
        # whitespace that follows.
        end = raw_at[hit + len(needle) - 1] + 1
        spans[sent_num] = (start, end)
        cursor = end
        flat_cursor = hit + len(needle)

    return spans or None


def _collapse(text: str) -> tuple[str, list[int]]:
    """``_normalise(text)`` together with, per output character, its offset in *text*.

    Built character by character, not by calling :func:`_normalise` and indexing afterwards,
    because ``str.lower`` is not length-preserving for every codepoint. One character that grew
    would shift every subsequent offset, and a span computed from it would underline the wrong
    words with no sign that anything went wrong.
    """
    out: list[str] = []
    positions: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            positions.append(i)     # The single space this whitespace run collapsed to
            pending_space = False
        lowered = ch.lower()
        out.append(lowered)
        positions.extend([i] * len(lowered))
    return "".join(out), positions


def _flat_position(raw_at: list[int], raw_offset: int) -> int:
    """The collapsed-string index at or after *raw_offset*. Linear, called once per sentence."""
    for i, pos in enumerate(raw_at):
        if pos >= raw_offset:
            return i
    return len(raw_at)


def sentence_of(spans: dict[int, tuple[int, int]] | None, offset: int) -> int | None:
    """The sentence a report-level character offset falls in, or ``None``."""
    if not spans:
        return None
    for sent_num, (start, end) in spans.items():
        if start <= offset < end:
            return sent_num
    return None
