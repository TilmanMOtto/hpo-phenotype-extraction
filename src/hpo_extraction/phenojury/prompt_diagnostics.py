"""Per-(prompt, model) diagnostics for the earlier prompt screen.

The screen ranks prompts by micro-F1, but that number rests on 20 reports and cannot say *why* a
prompt won or lost. These metrics can, and every one of them is computed from the cached
extractions with no GPU and no second PhenoBERT pass:

``echo_rate``
    How much of the output is lifted verbatim from the source sentence. This is the quantity the
    2026-08-11 meeting could only describe ("they mostly regurgitate the sentence wording") and the
    thing every restating prompt is trying to move. It also settles the per-model behaviour
    question in the same pass, MedGemma listing cleanly vs OpenBioLLM handing the sentence back
    becomes a column rather than an impression.

``label_exactness``
    Fraction of output lines that *are* an HPO term name or synonym. The most direct possible
    measure of restatement, and the one closest to what the format finding measured: PhenoBERT
    recovers 89.2 % of ids from a canonical name against 58.5 % from the report's own wording.

``format_compliance``
    Fraction of outputs shaped the way the prompt asked. Without it, "this prompt is worse" and
    "this model ignores instructions" are the same observation, and they call for opposite
    responses.

``negation_trap_rate``
    Fraction of lines carrying a token from PhenoBERT's ``getNegativeWords()``, each of which is
    discarded as ``Neg`` however well it reads. A prompt can score well on every metric above and
    still ground to nothing through this door.

``marginal_gold_yield``
    Annotated terms this cell finds that PhenoBERT standalone misses on the same reports. The only
    metric here that measures what the ensemble uniquely contributes. If it is near zero, no prompt
    can make the ensemble worth its GPU hours.

All rates are over records where the model actually wrote something (``wrote_something``), so a
model that mostly answers "no phenotype" is not credited with a perfect echo rate for staying
silent. ``n_wrote`` is reported alongside, so a rate computed over three lines is visible as such.
"""
from __future__ import annotations

import re

from hpo_extraction.phenojury.ensemble_eval import wrote_something
from hpo_extraction.phenojury.prompts import (
    NEGATION_TRIGGER_WORDS,
    SHAPE_FREE,
    SHAPE_JSON,
    SHAPE_PIPE_FIRST,
    SHAPE_PLAIN,
    SHAPE_SENTENCE,
    SHAPE_TWO_COLUMN,
    get_prompt,
)
from hpo_extraction.models.verdict import strip_think
from hpo_extraction.ontology.hpo_items import processStr

#: A leading list marker the templates ask models *not* to produce, plus the ones they produce
#: anyway. Stripped before a line is judged, since "- Microcephaly" and "Microcephaly" are the same
#: answer as far as restatement goes, the layout is `terminate_lines`' problem, not the prompt's.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

#: A numbered line, which the format finding measures at 21.9 % recovery, the templates forbid it
#: explicitly, so producing one is a compliance failure, not a style choice.
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")

#: p3's two-column separator.
_TWO_COLUMN = "=>"

#: q5/q6's field separator. Both put the ontology term in the FIRST field, q5's alternatives are
#: synonyms of it, q6's second field is its ancestor, so both are read the same way.
_PIPE = "|"


def _shape_of(prompt_key: str) -> str:
    """The declared ``line_shape`` for *prompt_key*, defaulting to plain for an unknown key.

    Unknown keys reach here from tests and from a cached cell whose prompt was later renamed. The
    default is the conservative one: judge the whole line, which is what every metric here did
    before shapes existed.
    """
    try:
        return get_prompt(prompt_key).line_shape
    except ValueError:
        return SHAPE_PLAIN


def term_candidates(line: str, shape: str) -> list[str]:
    """The substrings of *line* that are supposed to BE an ontology term, under *shape*.

    This is the whole reason ``line_shape`` exists. ``Microcephaly | Small head circumference |
    Head smaller than normal`` is a correct answer from q5 and scores zero label exactness if the
    metric reads the line whole, indistinguishable, in a ranking table, from a model that wrote
    nonsense. Splitting first makes a low score mean what it says.

    Returns a list because a JSON reply carries several terms in one *line*. Every other shape
    yields one candidate.
    """
    if shape == SHAPE_TWO_COLUMN:
        # The left side is *supposed* to be the sentence's wording, so judging it would penalise
        # p3 for doing its job.
        return [line.split(_TWO_COLUMN, 1)[1]] if _TWO_COLUMN in line else [line]
    if shape == SHAPE_PIPE_FIRST:
        return [line.split(_PIPE, 1)[0]] if _PIPE in line else [line]
    if shape == SHAPE_JSON:
        return _json_terms(line)
    # SHAPE_PLAIN, SHAPE_FREE, SHAPE_SENTENCE, the line is the candidate.
    return [line]


def _json_terms(line: str) -> list[str]:
    """The ``term`` values on one line of a q4 reply, without requiring the reply to parse.

    A regex, not ``json.loads`` on purpose: the whole point of measuring this condition is that
    an 8B model's JSON is often *nearly* valid, a trailing comma, a missing brace, prose before
    the object, and a strict parse would report every such reply as zero terms, which is a
    property of the parser and not of the prompt. The grounding path downstream is PhenoBERT,
    which never parses the JSON either.
    """
    return _JSON_TERM.findall(line)

#: A ``"term": "…"`` field of a q4 reply. Tolerant of whitespace and of single quotes, which small
#: models substitute for double ones often enough to matter.
_JSON_TERM = re.compile(r"""["']term["']\s*:\s*["']([^"']+)["']""")

#: Sentence-ending punctuation followed by more text, a line holding prose, not one
#: finding. A trailing period is fine (``terminate_lines`` adds one anyway). It is the *internal*
#: boundary that says the model wrote a sentence and ignored the one-per-line contract.
_PROSE = re.compile(r"[.!?;]\s+\S")


def _output_lines(text: str | None) -> list[str]:
    """The non-empty lines of one reply, reasoning stripped and list markers removed.

    ``strip_think`` first, for the same reason ``run_phenobert_per_model`` applies it at the
    grounding boundary: a reasoning model's CoT is not its answer, and counting the prose it
    thinks in would make the three reasoning models look like the worst echoers in the
    ensemble regardless of what they finally wrote.
    """
    if not text:
        return []
    out = []
    for raw in strip_think(text).split("\n"):
        line = _LIST_MARKER.sub("", raw).strip()
        if line:
            out.append(line)
    return out


def _tokens(text: str) -> list[str]:
    """Normalised tokens, using the ontology's own normalisation.

    ``processStr`` is what ``HPOTree`` runs over every term name when it builds ``p_phrase2HPO``,
    so using it here means a line and an ontology label are compared under one definition of
    "same string", not two that drift.
    """
    return processStr(text)


def echo_rate(lines: list[str], sentence: str) -> float:
    """Share of the output's tokens that already appear in the source sentence.

    Token containment, not substring matching: a model that reorders or re-inflects while
    still copying is echoing, and a literal-substring measure would score it as a clean restatement.
    Returns 0.0 for an empty output, nothing was echoed because nothing was written.
    """
    src = set(_tokens(sentence))
    toks = [t for line in lines for t in _tokens(line)]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in src) / len(toks)


def label_exactness(lines: list[str], phrase_index: dict[str, str],
                    shape: str = SHAPE_PLAIN) -> float:
    """Share of a reply's term candidates that match an HPO term name or synonym.

    ``phrase_index`` is ``HPOTree.p_phrase2HPO``, sorted normalised tokens → HPO id, built from
    every node's name and synonyms. Reusing it (rather than re-reading ``hpo.json``) is what keeps
    "is this a term" answered the same way here and in the ontology code.

    *shape* selects which part of each line is judged (:func:`term_candidates`): the right half of
    a ``span => Term`` line, the first field of a ``Term | synonym | gloss`` line, the ``term``
    values of a JSON reply, the whole line otherwise. The denominator is the number of candidates,
    not of lines, so a JSON line carrying three terms of which two are canonical scores 2/3, not 0 or 1.

    Returns ``None`` for :data:`SHAPE_SENTENCE`, which asks for prose and therefore has no line
    that could be a term. ``None`` and ``0.0`` are different claims, "not applicable" against
    "wrote nothing an ontology recognises", and collapsing them would put a rewriting prompt at
    the bottom of a column it never entered.
    """
    if shape == SHAPE_SENTENCE:
        return None
    if not lines:
        return 0.0
    candidates = [c for line in lines for c in term_candidates(line, shape)]
    if not candidates:
        return 0.0
    n_hit = sum(1 for c in candidates if " ".join(sorted(_tokens(c))) in phrase_index)
    return n_hit / len(candidates)


def negation_trap_rate(lines: list[str]) -> float:
    """Share of output lines carrying a token PhenoBERT treats as negation.

    Matched as bare tokens, as ``getNegativeWords()`` is applied, which is why
    ``non-verbal`` counts: the hyphen becomes a space upstream and leaves the bare token ``non``.
    """
    if not lines:
        return 0.0
    n_trap = sum(
        1 for line in lines if NEGATION_TRIGGER_WORDS & set(_tokens(line))
    )
    return n_trap / len(lines)


def format_compliance(lines: list[str], prompt_key: str) -> float:
    """Share of output lines shaped the way ``prompt_key``'s template asked for.

    shallow, it checks the constraints the format finding shows actually cost
    recovery, not prose quality:

    * every variant except the baseline forbids numbering (``Term\\n2.`` corrupts the last word of
      every item) and asks for short lines, so a line over 12 tokens is a sentence, not a finding;
    * a line carrying **internal sentence punctuation** is prose the model failed to split. Length
      alone does not catch this, "The boy had a small head." is only six tokens, and it is the
      single most common way a model ignores the one-per-line contract while looking compliant;
    * a shape with a separator (``p3_two_column``'s ``=>``, q5/q6's ``|``) additionally has to
      contain it.

    A prompt that requested no shape at all (:data:`SHAPE_FREE`, i.e. ``p0_baseline``) scores 1.0
    by definition. That is not a free pass, it means compliance is uninformative for the control,
    and its echo rate carries the comparison instead.

    Prompts asking for something other than a list of terms are judged on **their own** contract,
    via ``line_shape``: a rewriting prompt is compliant when it produced one sentence, a
    JSON prompt when a ``term`` field came out of it. Holding every condition to the list contract would
    score the two that obeyed their instructions at 0.0, which is the ranking table's worst
    available failure, a wrong number that looks like a finding.
    """
    shape = _shape_of(prompt_key)
    if shape == SHAPE_FREE:
        return 1.0
    if shape == SHAPE_SENTENCE:
        # A rewriting prompt asked for one prose sentence, so every constraint below is
        # inverted: internal punctuation is correct, a 20-token line is correct, and more than one
        # line is the failure. Judging it on the list contract would report the condition as 0.0
        # compliant for obeying its instructions.
        return 1.0 if len(lines) == 1 else 0.0
    if shape == SHAPE_JSON:
        # Compliance for JSON is "did a term field come out of it", not line length: the whole
        # reply is legitimately one long line, and _PROSE fires on `", "` inside every object.
        return 1.0 if any(_json_terms(line) for line in lines) else 0.0
    if not lines:
        return 0.0
    n_ok = 0
    for line in lines:
        if _NUMBERED.match(line):
            continue
        if shape == SHAPE_PIPE_FIRST and _PIPE not in line:
            continue
        # Judged on the term-bearing field, not the whole line: q5 asks for three names, whose
        # combined length says nothing about whether the model obeyed the contract.
        parts = term_candidates(line, shape)
        if any(len(_tokens(p)) > 12 for p in parts):
            continue
        if any(_PROSE.search(p) for p in parts):
            continue
        if shape == SHAPE_TWO_COLUMN and _TWO_COLUMN not in line:
            continue
        n_ok += 1
    return n_ok / len(lines)


def diagnose_records(records: list[dict], prompt_key: str, phrase_index: dict[str, str]) -> dict:
    """Aggregate every text-level diagnostic over one (prompt, model) cell's extractions.

    Rates are means over the records where the model wrote something real, so silence neither
    helps nor hurts a prompt's echo rate. ``n_records`` and ``n_wrote`` are carried alongside for
    the reason the screen needs them: a 1.00 label exactness over four lines is not a
    result, and the ranking table has to show that.
    """
    n_records = len(records)
    echoes: list[float] = []
    exactness: list[float] = []
    traps: list[float] = []
    compliance: list[float] = []
    n_lines = 0

    shape = _shape_of(prompt_key)

    for rec in records:
        if not wrote_something(rec.get("llm_output")):
            continue
        lines = _output_lines(rec.get("llm_output"))
        if not lines:
            continue
        n_lines += len(lines)
        echoes.append(echo_rate(lines, rec.get("sentence_text") or ""))
        # None for a shape the metric does not apply to. Dropped, not averaged as zero.
        hit = label_exactness(lines, phrase_index, shape)
        if hit is not None:
            exactness.append(hit)
        traps.append(negation_trap_rate(lines))
        compliance.append(format_compliance(lines, prompt_key))

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_records": n_records,
        "n_wrote": len(echoes),
        "n_lines": n_lines,
        "mean_lines_per_wrote": round(n_lines / len(echoes), 2) if echoes else 0.0,
        "echo_rate": _mean(echoes),
        # None, not 0.0, when the prompt's shape has no line that could be a term. A ranking table
        # cannot tell those apart once they are both zero, and one of them is a real failure.
        "label_exactness": _mean(exactness) if shape != SHAPE_SENTENCE else None,
        "negation_trap_rate": _mean(traps),
        "format_compliance": _mean(compliance),
    }


def marginal_gold_yield(predicted: dict, gold: dict, phenobert_predicted: dict) -> dict:
    """What this prompt's predictions add over PhenoBERT standalone, on the same reports.

    ``predicted``/``gold``/``phenobert_predicted`` are all ``{report_id: set(hpo_id)}``.

    Three counts, because "adds terms" and "adds *correct* terms" are different claims:

    ``n_gold_gained``
        annotated terms found here that PhenoBERT missed, the ensemble's reason to exist.
    ``n_gold_lost``
        annotated terms PhenoBERT found that this prompt did not. A prompt can gain implicit phenotypes
        while losing explicit ones, and a net figure would hide that trade entirely.
    ``n_extra_wrong``
        non-annotated terms added over PhenoBERT, what the gain costs in precision.

    When ``phenobert_predicted`` is empty (the PhenoBERT baseline run for this cohort is missing) the
    subtrahend is an empty set, which would silently report every ground truth hit as "gained". Callers get
    ``pb_available: False`` to key on, not a plausible-looking number.
    """
    pb_available = bool(phenobert_predicted)
    gained = lost = extra = 0
    for rid, gold_set in gold.items():
        pred = predicted.get(rid, set())
        pb = phenobert_predicted.get(rid, set())
        gained += len((pred & gold_set) - pb)
        lost += len((pb & gold_set) - pred)
        extra += len((pred - gold_set) - pb)
    return {
        "pb_available": pb_available,
        "n_gold_gained": gained,
        "n_gold_lost": lost,
        "n_extra_wrong": extra,
    }
