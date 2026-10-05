"""The adapter lifecycle, and the placement rules all five columns must agree about.

Placement is where a comparison UI quietly goes wrong. Each method recorded its evidence in its own
frame, and a gutter that puts AutoPCR's term one segment away from PhenoBERT's identical term is
not a small error -- it is the app inventing a disagreement the methods did not have. So the two
rules below are here once, not five times:

**A term is placed where its own evidence says, and where that is silent, where the ground truth says.**
A true positive and a false negative both concern an annotated term, and the ground truth already carries a
curated segment and trigger word. Drawing the method's row on that same segment is what puts the
five gutters on one line with the ground truth column, which is the whole layout. A false positive has no
ground truth row by definition, so it falls back to the method's own span -- and when the method kept none
(PhenoJury before its jurors are joined, TreePhenoRAG before its calls are read) it is placed
nowhere and listed in the column's footer instead of being drawn somewhere plausible.

**Membership is ontological, never a string compare.** A ground truth file predates the release it is
scored against, so a ground truth code and a predicted code can be the same phenotype spelled two ways.
``terms.same_term`` resolves alt ids. Comparing the raw strings would paint a term the method
actually found in red.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

#: Cap on a free-text field copied into a bundle -- a model generation, a reconstructed prompt.
#: Generous enough that no real HCY sentence generation is touched, small enough that one
#: pathological repetition loop cannot blow the bundle cap on its own. Truncation is visible: the
#: bundle carries ``truncated: true`` and the panel says so.
MAX_TEXT = 4000

#: Cap on how many ranked candidates of a menu are kept. AutoPCR shows k=5; RAG-HPO writes far
#: more, and past the first handful the reader is looking at cosine noise.
MAX_CANDIDATES = 12


def clip(text, limit: int = MAX_TEXT):
    """``(text, truncated)`` -- shorten a free-text field and say whether it was shortened."""
    if text is None:
        return "", False
    text = str(text)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def read_jsonl(path: str, report_ids=None, id_fields=("report_id", "patient_id")):
    """Stream one JSONL, yielding ``(report_id, row)`` for the reports asked for.

    Streaming rather than loading: these files run to several megabytes and the builder holds five
    of them plus an ontology. Filtering here, not after is what keeps the whole build inside
    a login node's memory.

    Both id spellings are accepted because the artifacts disagree -- detections files say
    ``report_id`` and extraction files say ``patient_id`` -- and a reader that handles one of them
    fails on half the inputs, silently.
    """
    if not path or not os.path.isfile(path):
        return
    wanted = set(report_ids) if report_ids is not None else None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rid = ""
            for field in id_fields:
                if row.get(field):
                    rid = str(row[field])
                    break
            if not rid or (wanted is not None and rid not in wanted):
                continue
            yield rid, row


def group_jsonl(path: str, report_ids=None, id_fields=("report_id", "patient_id")) -> dict:
    """:func:`read_jsonl` collected into ``{report_id: [row, ...]}``, order preserved."""
    out: dict = {}
    for rid, row in read_jsonl(path, report_ids, id_fields):
        out.setdefault(rid, []).append(row)
    return out


def load_prediction_sets(path: str) -> dict:
    """``{report_id: set}`` from either shape a method's predictions come in.

    Two shapes exist and both are authoritative for their own methods: the fixed drivers write
    ``*_predictions.jsonl`` whose ``summary: true`` line is the only record of a report that
    predicted **nothing**, and the protocol drivers write the two-column CSV because their
    prediction depends on the fold a report fell in and there is no per-term row to attach it to.
    Reading the per-term lines of the first would hand a method free precision on every report it
    said nothing about, which is the trap ``result_tables.loaders.load_predictions`` avoids by trusting
    the summary line over the rows.
    """
    if not path or not os.path.isfile(path):
        return {}
    if path.endswith(".csv"):
        from hpo_extraction.evaluation.prediction_sets import read_prediction_sets

        return read_prediction_sets(path)

    predicted: dict = {}
    seen_summary: set = set()
    for rid, row in read_jsonl(path):
        if row.get("summary"):
            predicted[rid] = set(row.get("predicted_set") or ())
            seen_summary.add(rid)
        elif rid not in seen_summary and row.get("prediction"):
            predicted.setdefault(rid, set()).add(row["hpo_id"])
    return predicted


# -- placement ---------------------------------------------------------------

def place_by_offsets(rv, rows, start_key="start", end_key="end") -> dict:
    """``{hpo_id: {segment_idx, span, phrase}}`` for rows carrying offsets into a staged text.

    *rv* must be the :class:`~apps.compare_ui.sources.ReportView` for the same text the offsets were
    recorded against. PhenoBERT and AutoPCR both stage the report verbatim, so that is the report
    view -- but the caller checks, because an offset measured against a differently staged file
    lands a few characters off and looks like a tokenizer quirk, not a bug.

    Where one term was detected several times the **first** occurrence wins, matching the order the
    detections file was written in: it is the one the reader will find by scanning down.
    """
    out: dict = {}
    for row in rows:
        code = row.get("hpo_id")
        if not code or code in out:
            continue
        try:
            start, end = int(row[start_key]), int(row[end_key])
        except (KeyError, TypeError, ValueError):
            continue
        local = rv.local(start, end)
        if local is None:
            continue
        idx, s, e = local
        out[code] = {"segment_idx": idx, "span": [s, e],
                     "phrase": rv.display[idx][s:e]}
    return out


def place_by_sentence(rv, code_to_sentence: dict) -> dict:
    """``{hpo_id: {segment_idx, span: None}}`` for methods that name a sentence and no offset.

    PhenoJury and TreePhenoRAG both work per sentence, so a sentence index is all they have and all
    they need: the gutter cell sits on that row, and the reasoning panel shows the sentence. No
    span is invented -- underlining a guessed phrase would be the app claiming evidence nobody
    recorded.
    """
    out: dict = {}
    for code, sent in code_to_sentence.items():
        if sent is None:
            continue
        idx = int(sent)
        if 0 <= idx < rv.n_segments:
            out[code] = {"segment_idx": idx, "span": None}
    return out


def span_spells(text: str, start, end, phrase: str) -> bool:
    """Do recorded offsets actually spell the phrase they claim, in *text*?

    Every set of offsets this builder is handed was measured against *some* staging of the
    document, and the whole placement story rests on that staging being the one the reader sees.
    Where it is not, the offsets land a few characters off and the result reads as a tokenizer
    quirk, not as a bug -- so the caller checks first and falls back to locating the phrase.

    Compared with line terminators collapsed, because a mention may straddle a line break that the
    document spells ``
`` and the phrase spells as a space.
    """
    if not phrase or start is None or end is None:
        return False
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return False
    if not 0 <= start < end <= len(text):
        return False
    return " ".join(text[start:end].split()) == " ".join(str(phrase).split())


def locate_phrase(rv, phrase: str, prefer_segment=None):
    """``(segment_idx, span)`` for a literal phrase, or ``(None, None)``.

    For RAG-HPO, whose records carry the sentence a finding came from but no offset into the
    report. Searched in *prefer_segment* first when the caller has narrowed it down, then across
    the segments in order -- and the first hit wins, which is honest about being a location, not the location: RAG-HPO did not record which occurrence it meant.
    """
    if not phrase:
        return None, None
    needle = phrase.strip()
    if not needle:
        return None, None
    order = range(rv.n_segments)
    if prefer_segment is not None and 0 <= prefer_segment < rv.n_segments:
        order = [prefer_segment] + [i for i in range(rv.n_segments) if i != prefer_segment]
    lowered = needle.lower()
    for idx in order:
        at = rv.display[idx].lower().find(lowered)
        if at >= 0:
            return idx, [at, at + len(needle)]
    return None, None


# -- the lifecycle -----------------------------------------------------------

class Adapter:
    """One column's reader.

    Lifecycle is two phases on purpose. ``load`` runs once and may open cohort-wide artifacts;
    ``evidence`` runs per report and must not. The split is what keeps the build linear in the
    number of reports, not quadratic, and it is also what lets the expensive adapter
    (:mod:`apps.compare_ui.methods.tree`, which re-runs a 585 MB cache) pay for its setup once.
    """

    #: Overridden by each adapter. Used in log lines and in the bundle's ``status`` note.
    key = ""

    def __init__(self, ctx, method):
        self.ctx = ctx
        self.method = method
        self.status = "ok"
        self.note = ""
        self._predictions: dict = {}

    # -- phase 1 -------------------------------------------------------------

    def load(self, report_ids) -> None:
        """Open whatever this method wrote, restricted to *report_ids*. Must not raise.

        A method whose artifacts are absent sets ``status`` to ``missing`` and a ``note`` saying
        which file it wanted. The column is then greyed out and the other four still work, which
        is the behaviour that counts when one experiment is still queued on the cluster.
        """
        raise NotImplementedError

    def predictions(self) -> dict:
        """``{report_id: set}`` -- populated by :meth:`load`."""
        return self._predictions

    # -- phase 2 -------------------------------------------------------------

    def evidence(self, rv, outcomes: dict, gold_rows: dict):
        """``(marks, reasons)`` for one report.

        *outcomes* is ``{hpo_id: "tp"|"fp"|"fn"}``, computed once in :mod:`apps.compare_ui.build`
        so that five adapters cannot disagree about what a true positive is. *gold_rows* is
        ``{hpo_id: {segment_idx, span}}`` from the curated ground truth, which is where a ``tp`` or an
        ``fn`` is drawn unless this method has better evidence of its own.
        """
        raise NotImplementedError

    # -- shared -------------------------------------------------------------

    def missing(self, path: str) -> None:
        """Record that this column cannot be drawn, and why, without failing the build."""
        self.status = "missing"
        self.note = "no artifact at " + str(path)
        logger.warning("%s: %s", self.key or self.method.key, self.note)

    def marks_from(self, outcomes: dict, placed: dict, gold_rows: dict) -> list:
        """Assemble the drawn chips from this method's placements and the ground truth's.

        The precedence is the rule stated in the module docstring: a ``fp`` uses the method's own
        evidence. A ``tp`` or ``fn`` uses the ground truth's position, because that is what lines the
        gutter up with the ground truth column. A ``tp`` whose method evidence sits somewhere *else* is
        not lost -- the reasoning panel shows it, and that disagreement is itself interesting.
        """
        marks = []
        for code, outcome in sorted(outcomes.items()):
            if outcome == "fp":
                where = placed.get(code) or {}
            else:
                where = gold_rows.get(code) or placed.get(code) or {}
            marks.append({
                "hpo_id": code,
                "outcome": outcome,
                "segment_idx": where.get("segment_idx"),
                "span": where.get("span") if where.get("segment_idx") is not None else None,
                "phrase": where.get("phrase") or "",
            })
        return marks
