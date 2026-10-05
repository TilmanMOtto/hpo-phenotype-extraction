"""AutoPCR: the interesting field is ``linked_by``, and the missing one is the answer.

AutoPCR's contribution is where it puts the language model -- not at extraction, but at *linking*:
a phrase whose top retrieval candidate scores in the band ``[tau_2, tau_1)`` is handed to the LLM
with a menu of candidates, and the model picks one. Everything above ``tau_1`` is taken on the
retriever's word and anything the dictionary matches never reaches either. So the single
most informative thing about an AutoPCR decision is which of those three routes fired, and
``autopcr_detections.jsonl`` records it: ``linked_by`` is ``dictionary``, ``retrieval`` or ``llm``,
with ``score`` carrying the cosine for the middle one and the sentinel ``-1.0`` for the last.

**The answer is not on disk, and this module does not pretend otherwise.**
``hpo_extraction.baselines.autopcr_runner.LocalPrompter.__call__`` increments a counter and discards both the prompt it
was given and the text the model returned. The prompt is reconstructible -- upstream's
``gen_grounding_prompt2`` builds it deterministically from the candidate menu, and notably does
*not* include the report text, which is a real property of the method worth showing -- so it is
rebuilt here and badged. The answer is gone; ``answer_recorded: false`` travels with every panel so
a reader cannot mistake our reconstruction for the model's reasoning.

``autopcr_retrieved_segments.jsonl`` holds the menu, but only for phrases whose top candidate
cleared ``tau_1`` -- the full n-gram enumeration is millions of rows. So a term with no menu here is
not evidence of nothing being retrieved.

Offsets index the corpus AutoPCR was staged into, and ``autopcr_runner.flatten_for_corpus`` is
length-preserving by design, so they index the report text this app reads.
"""

from __future__ import annotations

from apps.compare_ui.methods import base

#: ``score`` carries this when the LLM chose, rather than a cosine. Upstream's sentinel, not ours.
LLM_SENTINEL = -1.0

ROUTE_HELP = {
    "dictionary": "an exact dictionary hit -- no retrieval and no model call",
    "retrieval": "the retriever's top candidate cleared tau_1, so it was taken without a model",
    "llm": "the top candidate fell in the [tau_2, tau_1) band, so the linker chose from a menu",
    "unknown": "the route was not recorded",
}


class AutoPcrAdapter(base.Adapter):
    """Reads AutoPCR predictions and places each term on its linked phrase."""
    key = "autopcr"

    def load(self, report_ids) -> None:
        """Read the AutoPCR predictions of *report_ids*. Sets ``status`` to ``missing`` when absent."""
        paths = self.ctx.paths
        variant = self.method.variant
        predictions = paths.cohort_exp(self.method.exp_id, variant + "_predictions.jsonl")
        detections = paths.cohort_exp(self.method.exp_id, variant + "_detections.jsonl")
        segments = paths.cohort_exp(self.method.exp_id, variant + "_retrieved_segments.jsonl")

        self._predictions = base.load_prediction_sets(predictions)
        if not self._predictions:
            self.missing(predictions)
            return
        self.ctx.note_input("autopcr_predictions", predictions)

        self._detections = base.group_jsonl(detections, report_ids)
        if self._detections:
            self.ctx.note_input("autopcr_detections", detections)
        self._menus = _menus_by_report(segments, report_ids)
        if self._menus:
            self.ctx.note_input("autopcr_retrieved_segments", segments)

    def evidence(self, rv, outcomes, gold_rows):
        """``(marks, reasons)`` of one report."""
        rows = getattr(self, "_detections", {}).get(rv.report_id, [])
        menus = getattr(self, "_menus", {}).get(rv.report_id, {})

        by_code: dict = {}
        for row in rows:
            by_code.setdefault(row.get("hpo_id"), []).append(row)
        placed = base.place_by_offsets(rv, rows)

        reasons = {}
        for code, outcome in outcomes.items():
            reasons[code] = _reason(code, outcome, by_code.get(code) or [], menus,
                                    placed.get(code))
        return self.marks_from(outcomes, placed, gold_rows), reasons


def _menus_by_report(path: str, report_ids) -> dict:
    """``{report: {phrase: [candidate, ...]}}`` -- the ranked menu, keyed by the phrase it was for.

    Keyed by phrase because that is the unit AutoPCR retrieves on: one phrase, one menu, one
    choice. Keying by HPO id instead would split one decision across its own candidates.
    """
    out: dict = {}
    for rid, row in base.read_jsonl(path, report_ids):
        phrase = str(row.get("text") or "")
        if not phrase:
            continue
        menu = out.setdefault(rid, {}).setdefault(phrase, [])
        if len(menu) >= base.MAX_CANDIDATES:
            continue
        menu.append({
            "rank": _int(row.get("rank")),
            "hpo_id": str(row.get("hpo_id") or ""),
            "hpo_label": str(row.get("hpo_label") or ""),
            "cosine_sim": _number(row.get("cosine_sim")),
            "chosen": str(row.get("slm_verdict") or "").strip().lower() == "yes",
        })
    for menus in out.values():
        for menu in menus.values():
            menu.sort(key=lambda c: (c["rank"] is None, c["rank"]))
    return out


def _reason(code, outcome, rows, menus, where) -> dict:
    mentions = []
    for row in rows:
        phrase = str(row.get("phrase") or "")
        route = str(row.get("linked_by") or "unknown")
        mentions.append({
            "phrase": phrase,
            "route": route,
            "route_help": ROUTE_HELP.get(route, ROUTE_HELP["unknown"]),
            "score": _number(row.get("score")),
            "resolved": bool(row.get("resolved", True)),
            "raw_hpo_id": str(row.get("raw_hpo_id") or ""),
            "menu": menus.get(phrase) or [],
        })

    reason = {
        "kind": "linker",
        "source": "autopcr_detections.jsonl + autopcr_retrieved_segments.jsonl",
        "mentions": mentions,
        "segment_idx": (where or {}).get("segment_idx"),
        "answer_recorded": False,
        "answer_note": ("AutoPCR's linker was asked and answered, but hpo_extraction.baselines.autopcr_runner keeps "
                        "only a call count -- neither the prompt nor the answer was written. The "
                        "menu below is what it was choosing from; the choice is the marked row."),
    }

    if outcome == "fn":
        reason["why"] = _why_missed(code, menus)
    elif mentions:
        first = mentions[0]
        reason["why"] = "linked {!r} by {} ({})".format(
            first["phrase"], first["route"], first["route_help"])
    else:
        reason["why"] = "predicted with no mention recorded"
    return reason


def _why_missed(code: str, menus: dict) -> str:
    """Whether the term was ever on a menu, which is the one thing this file can settle.

    An annotated term that appeared as a candidate and lost is a linking failure. One that never appeared
    is a retrieval or extraction failure. They want opposite fixes, and a bare miss says neither.
    """
    near = []
    for phrase, menu in menus.items():
        for candidate in menu:
            if candidate["hpo_id"] == code:
                near.append((phrase, candidate))
    if not near:
        return ("never a candidate -- no phrase in this report retrieved this term into its menu "
                "(note: menus are only written for phrases that cleared tau_1)")
    phrase, candidate = near[0]
    rank = candidate.get("rank")
    return "was candidate{} for {!r} and not chosen".format(
        "" if rank is None else " #{}".format(rank), phrase)


def _number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
