"""PhenoBERT: a tagger, so its reasoning is a span and a score.

There is no natural-language rationale here and none is claimed. What ``phenobert_detections.jsonl``
records is what a CNN+BERT tagger can tell you: which characters it matched, which HPO code
it mapped them to, how confident it was, and whether it read the span as negated. That turns out to
be enough to answer most of the interesting questions about it, because PhenoBERT's characteristic
failures are visible in the span alone -- ``relative microcephaly`` mapped to *Relative
macrocephaly*, a lab value matched on the analyte and not the direction.

Two fields carry decisions that never reach the prediction set, and both are worth surfacing:
``negated`` detections and ``resolved: false`` ones (codes absent from the fixed ``hpo.json``) are
dropped by the driver. An annotated term PhenoBERT *detected and then discarded* is a completely
different miss from one it never saw, and only this file can tell them apart.

Offsets index the staged report, which ``phenobert_experiment.stage_reports`` writes **verbatim**
-- so they index the report text this app reads, and no second alignment is needed. That is
asserted per detection rather than assumed: ``base.span_spells`` checks the characters against the
recorded phrase, and a disagreement falls back to locating the phrase instead of underlining the
wrong three words.
"""

from __future__ import annotations

from apps.compare_ui.methods import base


class PhenoBertAdapter(base.Adapter):
    """Reads PhenoBERT predictions and places each term on its character offset."""
    key = "phenobert"

    def load(self, report_ids) -> None:
        """Read the PhenoBERT predictions of *report_ids*. Sets ``status`` to ``missing`` when absent."""
        paths = self.ctx.paths
        variant = self.method.variant
        predictions = paths.cohort_exp(self.method.exp_id, variant + "_predictions.jsonl")
        detections = paths.cohort_exp(self.method.exp_id, variant + "_detections.jsonl")

        self._predictions = base.load_prediction_sets(predictions)
        if not self._predictions:
            self.missing(predictions)
            return
        self.ctx.note_input("phenobert_predictions", predictions)

        self._detections = base.group_jsonl(detections, report_ids)
        if not self._detections:
            self.note = ("no detections beside the predictions, so only the verdict is shown and "
                         "not the span behind it")
        else:
            self.ctx.note_input("phenobert_detections", detections)

    def evidence(self, rv, outcomes, gold_rows):
        """``(marks, reasons)`` of one report."""
        rows = getattr(self, "_detections", {}).get(rv.report_id, [])
        by_code: dict = {}
        for row in rows:
            by_code.setdefault(row.get("hpo_id"), []).append(row)

        placed = _place(rv, rows)
        reasons = {}
        for code, outcome in outcomes.items():
            reasons[code] = _reason(rv, code, outcome, by_code.get(code) or [],
                                    placed.get(code))
        return self.marks_from(outcomes, placed, gold_rows), reasons


def _place(rv, rows) -> dict:
    """First detection per code, placed on the report -- trusting offsets only where they check."""
    out: dict = {}
    for row in rows:
        code = row.get("hpo_id")
        if not code or code in out:
            continue
        where = _locate(rv, row)
        if where is not None:
            out[code] = where
    return out


def _locate(rv, row):
    phrase = str(row.get("phrase") or "")
    try:
        start, end = int(row["start"]), int(row["end"])
    except (KeyError, TypeError, ValueError):
        start = end = None

    if start is not None and base.span_spells(rv.text, start, end, phrase):
        local = rv.local(start, end)
        if local is not None:
            idx, s, e = local
            return {"segment_idx": idx, "span": [s, e], "phrase": rv.display[idx][s:e]}

    # Either no offsets, or offsets that do not spell the phrase they claim -- which means they
    # were measured against a different staging of this report. Fall back to the phrase itself and
    # say so, not underlining characters nobody detected.
    idx, span = base.locate_phrase(rv, phrase)
    if idx is None:
        return None
    return {"segment_idx": idx, "span": span, "phrase": phrase, "located_by": "phrase"}


def _reason(rv, code, outcome, rows, where) -> dict:
    """What PhenoBERT can say about one term on one report."""
    detections = []
    for row in rows:
        detections.append({
            "phrase": str(row.get("phrase") or ""),
            "score": _number(row.get("score")),
            "negated": bool(row.get("negated")),
            "resolved": bool(row.get("resolved", True)),
            "raw_hpo_id": str(row.get("raw_hpo_id") or ""),
            "start": _int(row.get("start")),
            "end": _int(row.get("end")),
        })

    reason = {
        "kind": "tagger",
        "source": "phenobert_detections.jsonl",
        "detections": detections,
        "segment_idx": (where or {}).get("segment_idx"),
    }

    if outcome == "fn":
        reason["why"] = _why_missed(detections)
    elif detections:
        reason["why"] = ("matched {!r}".format(detections[0]["phrase"])
                         + (" (score {:.2f})".format(detections[0]["score"])
                            if detections[0]["score"] is not None else ""))
    else:
        reason["why"] = ("predicted with no detection recorded -- the term reached the prediction "
                         "set through the driver, not through a span")
    return reason


def _why_missed(detections) -> str:
    """The three ways PhenoBERT loses an annotated term, distinguished, not merged.

    Detected-and-dropped is a tuning question. Never-detected is a vocabulary or a wording
    question. Reporting both as "missed" is what makes a recall table unactionable.
    """
    if not detections:
        return "never detected -- no span in this report was mapped to this term"
    if all(d["negated"] for d in detections):
        return ("detected on {!r} and discarded as negated".format(detections[0]["phrase"]))
    if all(not d["resolved"] for d in detections):
        return ("detected as {}, which the fixed HPO release does not carry, so the driver "
                "dropped it".format(detections[0]["raw_hpo_id"] or "an unresolvable id"))
    return "detected on {!r} but not emitted".format(detections[0]["phrase"])


def _number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None       # NaN is not JSON, and is not a score


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
