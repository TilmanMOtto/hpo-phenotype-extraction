"""RAG-HPO: a phrase, a menu, and a Yes -- with the rationale computed and then dropped.

RAG-HPO extracts findings from the report, retrieves HPO candidates for each by embedding
similarity, and asks the model to pick. ``rag_hpo_retrieved_segments.jsonl`` records the middle two
steps faithfully: one row per (finding, candidate) with the source sentence, the rank, the cosine
and ``slm_verdict``, which is ``Yes`` on the candidate the pipeline took.

**The rationale exists and is not written.** ``hpo_extraction.baselines.rag_hpo_runner.run_rag_hpo`` stores
``raw_llm_resp`` and ``llm_parse_reason`` per non-exact phrase and then builds ``finding_records``
from a narrower set of columns, so neither reaches disk. Persisting them is a few lines. Paying for
them is a 70B re-run, which is why this panel shows the menu and the verdict and says plainly that
the reasoning behind the verdict was not kept.

There is a second gap worth naming on screen rather than in a commit message: nothing records
whether a finding was resolved by the local fuzzy matcher or by the model. AutoPCR has ``linked_by``
and RAG-HPO has no equivalent, so "the model chose this" is not a claim this column can make.

Placement is by phrase, not by offset -- RAG-HPO records the sentence a finding came from and no
character range. :func:`base.locate_phrase` searches the sentence first and the report second, and
what it returns is *a* location, not *the* location. That is the honest reading: with no
offset recorded, a term whose phrase occurs three times has three equally good homes.
"""

from __future__ import annotations

from apps.compare_ui.methods import base


class RagHpoAdapter(base.Adapter):
    """Reads RAG-HPO predictions and places each term on its extracted phrase."""
    key = "raghpo"

    def load(self, report_ids) -> None:
        """Read the RAG-HPO predictions of *report_ids*. Sets ``status`` to ``missing`` when absent."""
        paths = self.ctx.paths
        variant = self.method.variant
        predictions = paths.cohort_exp(self.method.exp_id, variant + "_predictions.jsonl")
        segments = paths.cohort_exp(self.method.exp_id, variant + "_retrieved_segments.jsonl")

        self._predictions = base.load_prediction_sets(predictions)
        if not self._predictions:
            self.missing(predictions)
            return
        self.ctx.note_input("raghpo_predictions", predictions)

        self._findings = _findings_by_report(segments, report_ids)
        if self._findings:
            self.ctx.note_input("raghpo_retrieved_segments", segments)
        else:
            self.note = ("no retrieved-segments artifact, so this column can show the verdict but "
                         "not the candidates behind it")

    def evidence(self, rv, outcomes, gold_rows):
        """``(marks, reasons)`` of one report."""
        findings = getattr(self, "_findings", {}).get(rv.report_id, {})
        chosen = _chosen_by_code(findings)
        placed = _place(rv, chosen)

        reasons = {}
        for code, outcome in outcomes.items():
            reasons[code] = _reason(code, outcome, chosen.get(code) or [], findings,
                                    placed.get(code))
        return self.marks_from(outcomes, placed, gold_rows), reasons


def _findings_by_report(path: str, report_ids) -> dict:
    """``{report: {source_text: [candidate, ...]}}`` -- one entry per finding, ranked.

    Grouped on the source text because that is the finding: RAG-HPO retrieves for a phrase and
    chooses once. The rows arrive in rank order and are kept in it.
    """
    out: dict = {}
    for rid, row in base.read_jsonl(path, report_ids):
        source = str(row.get("text") or "")
        if not source:
            continue
        menu = out.setdefault(rid, {}).setdefault(source, [])
        candidate = {
            "rank": _int(row.get("rank")),
            "hpo_id": str(row.get("hpo_id") or ""),
            "hpo_label": str(row.get("hpo_label") or ""),
            "cosine_sim": _number(row.get("cosine_sim")),
            "chosen": str(row.get("slm_verdict") or "").strip().lower() == "yes",
        }
        # The chosen row is kept whatever its rank. The rest are capped. A menu truncated before
        # its winner would show a decision with the decision missing.
        if len(menu) < base.MAX_CANDIDATES or candidate["chosen"]:
            menu.append(candidate)
    for menus in out.values():
        for menu in menus.values():
            menu.sort(key=lambda c: (c["rank"] is None, c["rank"]))
    return out


def _chosen_by_code(findings: dict) -> dict:
    """``{hpo_id: [(source_text, candidate), ...]}`` for the candidates the pipeline took."""
    out: dict = {}
    for source, menu in findings.items():
        for candidate in menu:
            if candidate["chosen"] and candidate["hpo_id"]:
                out.setdefault(candidate["hpo_id"], []).append((source, candidate))
    return out


def _place(rv, chosen: dict) -> dict:
    out: dict = {}
    for code, hits in chosen.items():
        source = hits[0][0]
        idx, span = base.locate_phrase(rv, source)
        if idx is None:
            # The finding's own text is a chunk RAG-HPO built, not necessarily a sentence of the
            # report -- so when it cannot be found verbatim, fall back to the sentence that
            # contains the most of it, not giving up on placing the term at all.
            idx = _best_segment(rv, source)
            span = None
        if idx is not None:
            out[code] = {"segment_idx": idx, "span": span, "phrase": source if span else ""}
    return out


def _best_segment(rv, source: str):
    """The segment sharing the most words with *source*, or ``None`` when none shares any."""
    words = {w for w in _words(source) if len(w) > 3}
    if not words:
        return None
    best, best_score = None, 0
    for idx, text in enumerate(rv.display):
        score = len(words & set(_words(text)))
        if score > best_score:
            best, best_score = idx, score
    return best


def _words(text: str):
    return [w.strip(".,;:()[]%").lower() for w in str(text).split()]


def _reason(code, outcome, hits, findings: dict, where) -> dict:
    selections = []
    for source, candidate in hits:
        selections.append({
            "source_text": base.clip(source, 600)[0],
            "rank": candidate.get("rank"),
            "cosine_sim": candidate.get("cosine_sim"),
            "menu": findings.get(source) or [],
        })

    reason = {
        "kind": "rag",
        "source": "rag_hpo_retrieved_segments.jsonl",
        "selections": selections,
        "segment_idx": (where or {}).get("segment_idx"),
        "answer_recorded": False,
        "answer_note": ("RAG-HPO records a rationale per selection (raw_llm_resp, "
                        "llm_parse_reason) and hpo_extraction.baselines.rag_hpo_runner drops both before writing. "
                        "There is also no record of whether the fuzzy matcher or the model made "
                        "this call, so the route is unknown -- unlike AutoPCR's linked_by."),
    }

    if outcome == "fn":
        reason["why"] = _why_missed(code, findings)
    elif selections:
        first = selections[0]
        reason["why"] = "chose it for {!r}{}".format(
            _short(first["source_text"]),
            "" if first["rank"] is None else " from rank #{}".format(first["rank"]))
    else:
        reason["why"] = "predicted with no finding recorded"
    return reason


def _why_missed(code: str, findings: dict) -> str:
    near = []
    for source, menu in findings.items():
        for candidate in menu:
            if candidate["hpo_id"] == code:
                near.append((source, candidate))
    if not near:
        return "never retrieved -- no finding in this report put this term on a candidate list"
    source, candidate = near[0]
    rank = candidate.get("rank")
    return "was candidate{} for {!r} and lost".format(
        "" if rank is None else " #{}".format(rank), _short(source))


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
