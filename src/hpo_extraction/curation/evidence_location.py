"""Putting a rich annotation back on the sentence it came from.

``hcy_holistic_ground_truth.csv`` and ``annotations_confirmed.csv`` each say where their annotation
lives, but in a coordinate system that is not the one this app draws. The prior_annotation file gives a
character offset into **its own** ``report_text`` column. The confirmed file gives a segment index
from **its own** segmentation run. Neither is guaranteed to agree with ``segmented_reports.csv``,
which is what every screen here is laid out on.

So each record is located against four falling-back strategies, and the one that worked travels
with the hit as ``how``, because "this mark is where the file said" and "this mark is where the
trigger happens to occur" are different claims, and a curator adjudicating an annotation is
entitled to know which they are looking at:

``segment``   the record's own ``segment_idx``, and the trigger is really in that segment
``offset``    the record's ``char_offset``, mapped through an alignment of the segments against the
              record's own report text
``context``   the record's sentence context matched to a segment, then the trigger found inside it
``lexical``   nothing but the trigger word, scanned across the report

An annotation whose trigger word occurs nowhere is **not** placed. It is returned unplaced and the
screen says so, for the same reason :mod:`spans` refuses to place a PhenoBERT offset it cannot
trust: a mark under approximately-the-right words, in a tool whose output becomes ground truth, is
worse than no mark. The curator can then locate it by hand, which is the honest outcome.

A record that names a segment but whose trigger cannot be found in it is still placed **on that
segment**, without character offsets. The file's claim about which sentence is worth keeping even
when its claim about which words did not survive re-tokenization. What is dropped is only the
underline, never the annotation.
"""

from __future__ import annotations

import re

from .spans import normalise

#: How far either side of a stated ``char_offset`` to look for the trigger word before giving up on
#: The offset. Generous enough to survive a report that gained or lost a header, tight enough that
#: it cannot silently reach a different occurrence of a common word in a long report.
OFFSET_WINDOW = 400


def find_trigger(text: str, trigger: str) -> tuple[int, int] | None:
    """``(start, end)`` of *trigger* in *text*, word-boundary first, or ``None``.

    Two passes rather than one: a word-boundary match is what the annotation meant, but a trigger
    that is a fragment ("hypotoni") or that carries punctuation still has to be findable, so a
    case-insensitive match without boundaries is the fallback. Boundaries first means *delay* never
    matches inside *delayed* while an exact-boundary hit exists elsewhere in the sentence.

    **Whitespace inside the trigger matches a run, not itself.** Reports are pasted out of lab
    systems and align their columns with tabs and runs of spaces, ``Serum Ammonia<TAB>87``, but
    every route a trigger arrives by has already flattened them: HTML collapses runs when it
    renders, so the browser's text selection, a click on a mark and anything a curator retypes all
    carry single spaces. Matching literally would refuse the trigger and unplace the mark over a
    difference that is invisible on screen and means nothing.

    The range indexes *text*, so the caller can always recover the report's own spelling from it.
    That is what :func:`views.common.snap_trigger` does, and why nothing is written from the typed
    string directly.
    """
    trigger = (trigger or "").strip()
    if not trigger or not text:
        return None
    pattern = r"\s+".join(re.escape(part) for part in trigger.split())
    match = re.search(r"\b" + pattern + r"\b", text, re.IGNORECASE)
    if match is None:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            return None
    return match.start(), match.end()


def resolve_offset(report_text: str, offset, trigger: str):
    """The offset at which *trigger* actually sits, or ``None`` if it does not sit near *offset*.

    The stated offset is checked, not trusted. A file whose offsets are right returns them
    unchanged. One whose report text has drifted by a header's worth of characters is repaired from
    the trigger word. One whose offset points at unrelated text is refused, and the record falls
    through to the context and lexical strategies. Refusing is the point, an unchecked offset is
    how a mark lands under the wrong words.
    """
    if offset is None or not report_text or not trigger:
        return None
    offset = int(offset)
    if report_text[offset:offset + len(trigger)].lower() == trigger.lower():
        return offset

    low = max(0, offset - OFFSET_WINDOW)
    window = report_text[low:offset + OFFSET_WINDOW + len(trigger)]
    hit = find_trigger(window, trigger)
    return None if hit is None else low + hit[0]


def segment_for_offset(ranges, offset) -> int | None:
    """Which segment contains *offset*, given an alignment in the same coordinate system."""
    if offset is None or not ranges:
        return None
    for idx, span in enumerate(ranges):
        if span is not None and span[0] <= offset < span[1]:
            return idx
    return None


def segment_for_context(display: list[str], context: str) -> int | None:
    """Which segment the record's sentence context refers to.

    Containment either way first, a context window is often a *slice* of a sentence, and a segment
    is often a slice of a context window, then the best token overlap, which is what survives a
    tokenizer that split one sentence into two. Whitespace and case are collapsed on both sides
    because the two files were written by different pipelines and neither preserved the other's.
    """
    context = normalise(_unescape(context))
    if not context:
        return None

    normalised = [normalise(segment) for segment in display]
    for idx, segment in enumerate(normalised):
        if segment and (segment in context or context in segment):
            return idx

    wanted = set(context.split())
    if not wanted:
        return None
    best, best_score = None, 0.0
    for idx, segment in enumerate(normalised):
        tokens = set(segment.split())
        if not tokens:
            continue
        score = len(tokens & wanted) / len(tokens | wanted)
        if score > best_score:
            best, best_score = idx, score
    # Half the vocabulary in common is a sentence, not a coincidence. Below that the "context" is
    # matching on stopwords and a guess would be indistinguishable from a finding.
    return best if best_score >= 0.5 else None


def locate(record: dict, display: list[str], ranges=None, report_text: str = "") -> dict | None:
    """Where *record* belongs: ``{"segment_idx", "start", "end", "how"}``, or ``None``.

    *ranges* is an alignment of *display* against the record's **own** report text, passed
    alongside as *report_text*, so a ``char_offset`` can be mapped. Pass neither when the source
    carries no text of its own, and the offset strategy is skipped, not guessed at.

    ``start``/``end`` index the segment string in *display* and are ``None`` when the segment is
    known but the trigger could not be found inside it.
    """
    if not display:
        return None
    trigger = (record.get("trigger_word") or "").strip()

    stated = record.get("segment_idx")
    guesses: list[tuple[str, int]] = []
    if stated is not None and 0 <= int(stated) < len(display):
        guesses.append(("segment", int(stated)))

    offset = resolve_offset(report_text or record.get("report_text", ""),
                            record.get("char_offset"), trigger)
    from_offset = segment_for_offset(ranges, offset)
    if from_offset is not None:
        guesses.append(("offset", from_offset))

    from_context = segment_for_context(display, record.get("sentence_context") or "")
    if from_context is not None:
        guesses.append(("context", from_context))

    for how, idx in guesses:
        span = find_trigger(display[idx], trigger)
        if span is not None:
            return {"segment_idx": idx, "start": span[0], "end": span[1], "how": how}

    if trigger:
        for idx, segment in enumerate(display):
            span = find_trigger(segment, trigger)
            if span is not None:
                return {"segment_idx": idx, "start": span[0], "end": span[1], "how": "lexical"}

    # The file named a sentence and the trigger did not survive into it. Keep the sentence: which
    # claim failed is worth showing, and dropping both would throw away the good half.
    if guesses:
        how, idx = guesses[0]
        return {"segment_idx": idx, "start": None, "end": None, "how": how}
    return None


def has_position(record: dict) -> bool:
    """Whether *record* claims a position at all, a trigger word, a segment, or an offset.

    A code-only source claims none, and that is not a failure to place: there was nothing to place.
    Counting those as "unplaced" would put every ``prior_annotation_2`` term in a list whose whole meaning is
    *somebody said where this was and we could not find it*, which is the one thing a curator has to
    go and look at by hand.
    """
    return bool((record.get("trigger_word") or "").strip()
                or record.get("segment_idx") is not None
                or record.get("char_offset") is not None)


def place_all(records, display: list[str], ranges_by_source=None, texts_by_source=None):
    """``({segment_idx: [placed, …]}, unplaced)`` for one patient's annotation records.

    A *placed* record is the original dict plus ``segment_idx``, ``start``, ``end`` and ``how``.
    Unplaced ones come back whole, so the screen can still list an annotation it could not draw,
    not quietly losing it, an annotation absent from both the report and the panel is one
    nobody will ever adjudicate.

    Records that never claimed a position (:func:`has_position`) appear in neither list. They are
    not lost, every panel lists them from the records themselves, they are simply not part of the
    question this function answers.
    """
    ranges_by_source = ranges_by_source or {}
    texts_by_source = texts_by_source or {}
    placed: dict[int, list[dict]] = {}
    unplaced: list[dict] = []

    for record in records:
        if not has_position(record):
            continue
        source = record.get("source")
        hit = locate(record, display, ranges_by_source.get(source),
                     texts_by_source.get(source, ""))
        if hit is None:
            unplaced.append(dict(record))
            continue
        placed.setdefault(hit["segment_idx"], []).append({**record, **hit})

    for rows in placed.values():
        rows.sort(key=lambda r: (r["start"] if r["start"] is not None else -1,
                                 r["end"] if r["end"] is not None else -1))
    return placed, unplaced


_ESCAPES = (("\\t", " "), ("\\n", " "), ("\\r", " "))


def _unescape(text: str) -> str:
    r"""Turn a literal ``\t`` written into a CSV cell back into whitespace.

    The context column has been seen starting with the two characters ``\`` and ``t``, not a
    tab, a round trip through a writer that escaped it and a reader that did not. Left alone it is
    two junk characters glued to the first word, which is enough to lose a containment match.
    """
    for needle, replacement in _ESCAPES:
        text = text.replace(needle, replacement)
    return text
