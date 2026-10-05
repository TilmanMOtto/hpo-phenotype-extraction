"""Stage 0 (preprocessing) and Stage A (span detection) of the earlier typed-span pipeline.

Stage 0 is deterministic and normalises the input *surface* without destroying evidence. Stage A is
the first model call: it over-generates candidate spans, because a span that is marked
and later discarded costs nothing while a span never marked can never be recovered.

Two decisions in here are worth reading before using the module.

**The source text is the only coordinate system, and it is never rewritten.** The specification's
Stage 0 says to apply a substitution table for optical-recognition damage (``rn``->``m``,
``um01``->``µmol``, ...). Applied literally that is incompatible with invariant I3, every emitted
record must carry a character offset into the source text, and ``rn``->``m`` changes the length of
the string, so every offset after the first repair would point at the wrong place. Worse, the drift
is silent: the span still looks like a span.

So Stage 0 here **detects and flags** corruption and never edits. The repair travels as an
annotation with its own offsets, and the actual repair happens at *query* time in Stage C's fuzzy
pass, which is what the specification's own §4.3 asks for anyway ("handled at the retrieval layer
rather than repaired in the text") for the neighbouring problem of spelling and register variants.
``homocvsteinaemia`` therefore stays ``homocvsteinaemia`` in the record, and still retrieves
*homocysteinemia*.

**An ambiguous unit is never resolved.** ``mol/mol`` could be ``mmol/mol`` or ``µmol/mmol``. The two
differ by a factor of a thousand. It is left intact, flagged, and escalated (§3.7). A unit that is
corrupt but *recoverable* is flagged too, and the flag is what tells a reader which of the two they
are looking at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sentence:
    """One segmented sentence, with its offsets into the *source* report text.

    ``text`` is a verbatim slice: ``report_text[start:end] == text`` holds by design, and
    :func:`anchor_sentences` raises, not return a Sentence for which it does not.
    """

    idx: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Span:
    """A candidate clinically-meaningful span, in source-text coordinates."""

    report_id: str
    sent_idx: int
    start: int
    end: int
    text: str
    source: str = "slm"  # slm | ner | lexical, which detector proposed it


@dataclass(frozen=True)
class Repair:
    """A detected character corruption. Recorded, never applied (see module docstring)."""

    start: int
    end: int
    observed: str
    suggested: str | None  # None when the reading is ambiguous
    kind: str              # ocr | unit
    ambiguous: bool = False


@dataclass
class Document:
    """Stage 0's output contract for one report."""

    report_id: str
    text: str
    sentences: list[Sentence]
    repairs: list[Repair] = field(default_factory=list)
    unit_flags: list[Repair] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    patient_meta: dict = field(default_factory=dict)
    escalations: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Offset locating, the I3 gate
# ---------------------------------------------------------------------------------------------


class OffsetError(ValueError):
    """A span or sentence whose offsets do not slice back to its own text."""


def location_sentences(text: str, sentence_texts: list[str]) -> list[Sentence]:
    """Locate each sentence in ``text`` by exact sequential forward search.

    Exactness is available to us because every step of the legacy segmentation chain
    (``segment_dict`` -> ``split_sents``) preserves verbatim substrings in document order: Stanza
    returns slices, ``str.split("\\n")`` returns slices, and the length filter only *drops*. The
    same argument is what lets ``result_tables/segments.py:segment_spans`` recover GSC+ spans, and this
    function is the same algorithm so the two agree.

    Searching forward from the previous match (rather than ``text.find`` from zero) is what keeps a
    repeated sentence, "No further metabolic crises." twice in one letter, located to its own
    occurrence instead of collapsing both onto the first.

    Raises:
        OffsetError: if a sentence cannot be found at or after the previous match. That means the
            segmentation is no longer a slice of the text, which invalidates every offset
            downstream, so it aborts, not guessing.
    """
    sentences: list[Sentence] = []
    cursor = 0
    for idx, sent_text in enumerate(sentence_texts):
        start = text.find(sent_text, cursor)
        if start < 0:
            raise OffsetError(
                f"sentence {idx} is not a verbatim slice of the report at or after offset "
                f"{cursor}: {sent_text[:60]!r}"
            )
        end = start + len(sent_text)
        sentences.append(Sentence(idx=idx, start=start, end=end, text=sent_text))
        cursor = end
    return sentences


def assert_offsets(text: str, spans: list[Span]) -> None:
    """Invariant I3, as a hard gate: ``text[start:end]`` must be the span's own text.

    Called on every report before anything is written. This is the analogue of ``an earlier exploratory run``'s
    verbatim-quote check, and it is not ceremonial: ``hpo_items.strip_accents`` (NFD, then drop
    combining marks) is only *usually* length-preserving, so any component that normalises before
    locating will drift by a character or two on the accented, non-native-speaker text this
    corpus is full of. A drifted span still looks like a span.

    Raises:
        OffsetError: naming the report and the first offending span.
    """
    for span in spans:
        if span.start < 0 or span.end > len(text) or span.start >= span.end:
            raise OffsetError(
                f"{span.report_id}: span offsets [{span.start}, {span.end}) are outside "
                f"[0, {len(text)}) or empty — {span.text!r}"
            )
        actual = text[span.start : span.end]
        if actual != span.text:
            raise OffsetError(
                f"{span.report_id}: span at [{span.start}, {span.end}) slices to {actual!r} "
                f"but carries text {span.text!r}"
            )


# ---------------------------------------------------------------------------------------------
# Stage 0, character and unit damage (detected, never applied)
# ---------------------------------------------------------------------------------------------

# Optical-recognition damage from the specification's §3.0. Each entry is (pattern, suggestion).
# `l`<->`1`<->`I` and `0`<->`O` are NOT blanket rules: applied to running text they fire
# on every capital I and every zero in the letter. They are scoped to the one place the confusion is
# both frequent and decidable, inside a token that is otherwise a unit or a number.
_OCR_RULES: tuple[tuple[re.Pattern, str], ...] = (
    # "rn" read for "m" inside a word: "homocysteinernia" -> "homocysteinemia"
    (re.compile(r"(?<=[a-z])rn(?=ia\b|ic\b|al\b)"), "m"),
    (re.compile(r"\bgmol\b"), "µmol"),
    (re.compile(r"\bum0?1\b", re.IGNORECASE), "µmol"),
    (re.compile(r"\bumol\b"), "µmol"),
    (re.compile(r"\bpmo1\b"), "pmol"),
)

# A unit that is ambiguous between two valid readings is flagged and left intact, never repaired.
# `mol/mol creatinine` could be `mmol/mol` or `µmol/mmol`, which differ by 1000x.
_AMBIGUOUS_UNIT = re.compile(r"\bmol\s*/\s*mol(?:\s+creat\w*)?", re.IGNORECASE)


def detect_repairs(text: str) -> tuple[list[Repair], list[Repair]]:
    """Find optical-recognition damage and ambiguous units. Returns ``(repairs, unit_flags)``.

    Nothing is substituted. ``repairs`` carry a ``suggested`` reading that Stage C may use to widen
    a query; ``unit_flags`` carry ``suggested=None`` and ``ambiguous=True``, and every one of them
    is an escalation (§3.7, "Which unit was intended?").
    """
    repairs: list[Repair] = []
    for pattern, suggestion in _OCR_RULES:
        for match in pattern.finditer(text):
            kind = "unit" if suggestion.endswith("mol") else "ocr"
            repairs.append(
                Repair(
                    start=match.start(),
                    end=match.end(),
                    observed=match.group(0),
                    suggested=suggestion,
                    kind=kind,
                )
            )

    unit_flags = [
        Repair(
            start=match.start(),
            end=match.end(),
            observed=match.group(0),
            suggested=None,
            kind="unit",
            ambiguous=True,
        )
        for match in _AMBIGUOUS_UNIT.finditer(text)
    ]
    return sorted(repairs, key=lambda r: r.start), unit_flags


def repair_density(repairs: list[Repair], sentences: list[Sentence]) -> dict[int, int]:
    """Repairs per sentence index. Feeds the §3.0 escalation trigger on repair density."""
    density: dict[int, int] = {}
    for repair in repairs:
        for sent in sentences:
            if sent.start <= repair.start < sent.end:
                density[sent.idx] = density.get(sent.idx, 0) + 1
                break
    return density


# ---------------------------------------------------------------------------------------------
# Stage 0, whitespace laboratory tables
# ---------------------------------------------------------------------------------------------

_VALUE_RE = re.compile(r"[<>]?\s*\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?)?")
_TABLE_SPLIT = re.compile(r"\s{2,}|\t")


def _cells(line: str) -> list[str]:
    """Split a whitespace-aligned row into cells. Two spaces or a tab is a column boundary."""
    return [c.strip() for c in _TABLE_SPLIT.split(line) if c.strip()]


def _is_value(cell: str) -> bool:
    return bool(_VALUE_RE.fullmatch(cell.replace(" ", "")))


def detect_tables(text: str) -> list[dict]:
    """Whitespace-aligned laboratory tables, extracted *before* linearisation.

    Two shapes occur in this correspondence and both have to be handled, because they fail
    differently:

    * **Inline**, ``Methionine    20    umol/L``, labels and values on one row. Alignment is
      local and a row that carries both is self-describing.
    * **Block**, a header row of labels followed by one or more rows of values. This is the shape
      the specification's escalation trigger is about ("four labels, three values"), and it is the
      dangerous one: linearising it into a sentence pairs the analytes with the wrong numbers, and
      the result reads perfectly.

    A count mismatch is **flagged, never zipped**. Pairing is the operation that would be
    wrong, so a misaligned block is an escalation, not a guess.
    """
    tables: list[dict] = []
    offsets: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        offsets.append((offset, offset + len(stripped), stripped))
        offset += len(line)

    pending: tuple[int, int, list[str]] | None = None  # An unconsumed label-only header row
    for start, end, line in offsets:
        cells = _cells(line)
        if len(cells) < 2:
            pending = None
            continue
        values = [c for c in cells if _is_value(c)]
        labels = [c for c in cells if not _is_value(c)]

        if values and labels:  # inline row
            tables.append(
                {
                    "kind": "inline",
                    "start": start,
                    "end": end,
                    "line": line,
                    "labels": labels,
                    "values": values,
                    "aligned": len(labels) == len(values) or len(values) == 1,
                }
            )
            pending = None
        elif labels and not values:  # candidate header
            pending = (start, end, labels)
        elif values and not labels and pending is not None:  # value row under a header
            h_start, _, h_labels = pending
            tables.append(
                {
                    "kind": "block",
                    "start": h_start,
                    "end": end,
                    "line": line,
                    "labels": h_labels,
                    "values": values,
                    "aligned": len(h_labels) == len(values),
                }
            )
        else:
            pending = None

    return tables
