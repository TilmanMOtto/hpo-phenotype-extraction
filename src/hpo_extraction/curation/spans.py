"""Aligning the Stanza segments to the verbatim report, so an offset lands on the right words.

PhenoBERT's ``start``/``end`` index the report file ``baseline_phenobert`` staged, the raw text, unchanged.
The segments this app lays a screen out on come from ``experiments/03_setup/segment_reports.py``: Stanza
tokenization followed by a newline split. Tokenization is *not* guaranteed to preserve the source
bytes, it normalises whitespace, and can normalise quotes, so a segment is not reliably a
substring of the report and ``text.find(segment)`` is not a safe way to place it.

The alignment here is therefore whitespace-insensitive, using the same collapse rule
``pbstandalone._normalise`` uses (and, through it, the rule ``phenobert_runner`` itself trusts an
offset by). Each segment is located in the normalised report with a **forward cursor**, so a
sentence repeated twice in a report maps to its own occurrence rather than to the first one.

What counts more than the hit rate is the failure mode. A segment that cannot be located gets no
PhenoBERT highlight at all, and the count of such segments is reported to the caller so the screen
can say so. Underlining approximately-right words in a curation tool is worse than underlining
none: the whole point of the screen is that what it shows can be trusted without re-deriving it.
This mirrors ``phenojury_generation_free_listing.verify.gate_pb_standalone``, which refuses to draw, not draw
wrong.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """The whitespace-insensitive form. Identical to ``pbstandalone._normalise`` by design."""
    return _WS.sub(" ", text).strip().lower()


def _normalised_index(text: str) -> tuple[str, list[int]]:
    """``(normalised text, raw index of each normalised character)``.

    The map is what makes the alignment invertible: a position in the normalised string can be
    turned back into a position in the report, which is what the caller actually needs.
    """
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for pos, char in enumerate(text):
        if char.isspace():
            pending_space = bool(out)     # never a leading space
            continue
        if pending_space:
            out.append(" ")
            index.append(pos)
            pending_space = False
        out.append(char.lower())
        index.append(pos)
    return "".join(out), index


def align_segments(text: str, segments: list[str]) -> list[tuple[int, int] | None]:
    """Per segment, its ``(start, end)`` char range in *text*, or ``None`` if unlocatable.

    Ranges are half-open and index *text* directly, so ``text[start:end]`` is the segment as the
    report spells it (which may differ from the segment string in whitespace).
    """
    if not text or not segments:
        return [None] * len(segments)

    norm_text, index = _normalised_index(text)
    ranges: list[tuple[int, int] | None] = []
    cursor = 0
    for segment in segments:
        needle = normalise(segment)
        if not needle:
            ranges.append(None)
            continue
        at = norm_text.find(needle, cursor)
        if at < 0:
            # A sentence out of document order (or reworded by tokenization) still gets one chance
            # from the top, but never rewinds the cursor, so later segments keep moving forward.
            at = norm_text.find(needle)
            if at < 0:
                ranges.append(None)
                continue
        start = index[at]
        end = index[at + len(needle) - 1] + 1
        ranges.append((start, end))
        cursor = max(cursor, at + len(needle))
    return ranges


def map_detections(
    text: str,
    segments: list[str],
    detections: list[dict],
) -> tuple[dict[int, list[dict]], list[dict]]:
    """:func:`align_segments` then :func:`place_detections`, the whole job in one call."""
    return place_detections(align_segments(text, segments), detections)


def place_detections(
    ranges: list[tuple[int, int] | None],
    detections: list[dict],
) -> tuple[dict[int, list[dict]], list[dict]]:
    """Place each detection on a segment, given the alignment.

    Returns ``({segment_idx: [placed, …]}, unplaced)``. A *placed* detection is the original dict
    plus ``segment_idx``, ``local_start`` and ``local_end``, offsets into the segment **as the
    report spells it** (:func:`segment_texts`), which is the string the reader will see. Detections
    whose start falls in no aligned segment come back in *unplaced*, carrying the phrase so the
    screen can still list them, not dropping evidence it could not place.
    """
    placed: dict[int, list[dict]] = {}
    unplaced: list[dict] = []

    for det in detections:
        start, end = int(det.get("start", 0)), int(det.get("end", 0))
        target = None
        for idx, span in enumerate(ranges):
            if span is not None and span[0] <= start < span[1]:
                target = (idx, span)
                break
        if target is None:
            unplaced.append(dict(det))
            continue
        idx, (seg_start, seg_end) = target
        # A detection may run past its segment's end (an abbreviation Stanza split on). Clamp
        #, not drop: the phrase's head is in this segment and that is where it belongs.
        placed.setdefault(idx, []).append({
            **det,
            "segment_idx": idx,
            "local_start": start - seg_start,
            "local_end": min(end, seg_end) - seg_start,
        })

    for rows in placed.values():
        rows.sort(key=lambda r: (r["local_start"], r["local_end"]))
    return placed, unplaced


def alignment_stats(ranges: list[tuple[int, int] | None]) -> dict:
    """``{n_segments, n_aligned, n_unaligned}``, what the sidebar reports about this patient."""
    aligned = sum(1 for r in ranges if r is not None)
    return {
        "n_segments": len(ranges),
        "n_aligned": aligned,
        "n_unaligned": len(ranges) - aligned,
    }


def segment_texts(
    text: str,
    segments: list[str],
    ranges: list[tuple[int, int] | None],
) -> list[str]:
    """Each segment as the *report* spells it, falling back to the tokenized string when unaligned.

    Highlight offsets index these strings, never the tokenized ones, mixing the two is the
    off-by-a-few-characters bug this module exists to prevent. The fallback is safe because an
    unaligned segment carries no offsets to be wrong about: :func:`place_detections` gives it none.
    """
    out = []
    for idx, span in enumerate(ranges):
        out.append(text[span[0]:span[1]] if span is not None else segments[idx])
    return out
