"""Helpers every panel shares. Not a panel, it has no ``layout`` and registers no callbacks.

The important one is :func:`registry_or_none`. A Dash callback that propagates an exception returns
a 500 for one component and the page renders around the hole, so every render path here degrades to
a visible message instead: a curator who cannot tell "nothing to show" from "this broke" will
eventually record a decision against a screen that was lying to them.
"""

from __future__ import annotations

import logging
import traceback

from dash import html

from .. import state, theme
from hpo_extraction.curation import evidence_location as locations

logger = logging.getLogger(__name__)

NO_PATIENT = "Pick a patient in the sidebar."

#: The last traceback per render path, so a broken panel reports its own reason on screen.
_FAILED: dict[str, str] = {}


def registry_or_none():
    """The installed registry, or None before one is installed."""
    return state.get_registry() if state.has_registry() else None


def guard(name: str, fn, *args, **kwargs):
    """Run *fn*, or return an error panel naming what broke. Never raises."""
    try:
        result = fn(*args, **kwargs)
        _FAILED.pop(name, None)
        return result
    except Exception as exc:  # noqa: BLE001 - surfaced on screen, not hidden
        _FAILED[name] = traceback.format_exc()
        logger.error("%s failed: %s\n%s", name, exc, _FAILED[name])
        return theme.note(f"{name} failed to render, {exc}", "danger")


def clear_failures() -> None:
    """Forget the recorded render failures."""
    _FAILED.clear()


def failures() -> dict:
    """``{view: error message}`` of the views that failed to render since the last clear."""
    return dict(_FAILED)


# ── small chrome ──────────────────────────────────────────────────────────────
def chip(text: str, color: str, title: str = "") -> html.Span:
    """A small coloured label. The app's one badge. Every status and source uses it.

    Everything but the colour lives in the ``.chip`` class. The size in particular *has* to: an
    inline ``fontSize`` in px does not follow ``--fs``, so the Display slider would scale the
    report and leave every chip on it the same size.
    """
    return html.Span(text, title=title, className="chip", style={"backgroundColor": color})


def code_chip(code: str, label: str, color: str, title: str = "") -> html.Span:
    """``Seizure (HP:0001250)``, the term as it is written everywhere in this app.

    *title* is the hover text, and callers should pass ``registry.search.describe(code)`` so that
    every mention of a phenotype in this app, a mark in the report, a chip in the ground truth panel, a
    row in Approve mode, hovers to the same name and the same definition.
    """
    return html.Span(
        [html.Span(label, style={"fontWeight": "600"}),
         html.Span(f" {code}", style={"opacity": ".7", "fontVariantNumeric": "tabular-nums"})],
        title=title,
        className="code-chip code-chip-text" if title else "code-chip-text",
        style={"color": color},
    )


def agreement(present, gold_sources, in_pb: bool) -> tuple[str, str]:
    """``(short label, explanation)`` for which annotation sources carry a code.

    The disagreement between the annotation files is the reason this app exists, so it is named on
    every row rather than left to be inferred from three separate columns.

    *gold_sources* is the set actually loaded, not the set this version knows about: with one file
    on disk there is nothing to disagree about, and a row that said "prior_annotation only" would be
    reporting the sidebar's configuration as if it were a finding about the annotation.
    """
    gold_sources = list(gold_sources)
    present = [source for source in gold_sources if source in set(present)]

    if not present:
        return "PhenoBERT only", "in no annotation file"
    if len(gold_sources) > 1 and len(present) == len(gold_sources):
        base, why = "all sources", "in every annotation file loaded"
    elif len(present) == 1:
        absent = [s for s in gold_sources if s != present[0]]
        base = f"{present[0]} only"
        why = f"in {present[0]}" + (f", absent from {', '.join(absent)}" if absent else "")
    else:
        base = " + ".join(present)
        absent = [s for s in gold_sources if s not in present]
        why = f"in {', '.join(present)}" + (f", absent from {', '.join(absent)}" if absent else "")
    if in_pb:
        base += " + PB"
        why += "; PhenoBERT also detected it"
    return base, why


def validate_annotation(registry, patient_id, segment_idx, trigger, code,
                        require_trigger: bool = True, require_code: bool = True) -> str:
    """The message to show, or ``""`` when the annotation is complete and consistent.

    One rule for both forms. Edit mode's *new phenotype* needs all three parts, the files this app
    repairs are code-only and a suggestion without evidence would reproduce that, while
    repairing an existing annotation may legitimately touch only some of them, so the two
    requirements are arguments, not two near-copies of this function that can drift.

    The trigger must occur in the segment it is attached to. That is the one bad record this app
    can produce: a trigger word that is not in its sentence is evidence pointing nowhere, and it
    would be discovered by whoever later tries to score against it.

    "Occurs" is :func:`anchors.find_trigger`, not ``in``, so a run of whitespace in the segment is
    satisfied by a single space in the trigger. A plain substring test refused
    ``Serum Ammonia 87`` against a report that separates the columns with a tab, a difference
    the screen cannot show, on the field the app most wants people to fill in.
    """
    if segment_idx is None:
        if require_trigger or (trigger or "").strip():
            return "Pick a segment in the report first, click the sentence this comes from."
        return ""
    view = registry.patient(patient_id)
    if view is None or not (0 <= int(segment_idx) < len(view["display"])):
        return "That segment is not in this report any more. Pick one again."
    trigger = (trigger or "").strip()
    if not trigger:
        if require_trigger:
            return "A trigger word is required, the words in the sentence that say this."
    elif locations.find_trigger(view["display"][int(segment_idx)], trigger) is None:
        return (f"“{trigger}” does not occur in segment #{segment_idx}. "
                "The trigger must be text from the segment it is attached to.")
    if require_code and not code:
        return "Pick an HPO term."
    return ""


def snap_trigger(registry, patient_id, segment_idx, trigger: str) -> str:
    """*trigger* respelled as the report spells it, or the typed string when it is not found.

    The last step before any trigger is written, and the reason validation can afford to be
    whitespace-insensitive: what gets recorded is a **verbatim slice of the report**, tabs and
    column padding included, not the flattened thing that came back from the browser. A trigger is
    a quote, and the file this app produces is scored by matching it against the report, so the
    one place a difference in whitespace counts is the place it is repaired.

    Falling back to the typed string, not refusing: this runs after
    :func:`validate_annotation` on every path that requires a trigger, so a miss here means the
    caller did not require one (an edit that only touches the note), and blanking their field would
    be a worse answer than leaving it alone.
    """
    trigger = (trigger or "").strip()
    view = registry.patient(patient_id) if registry is not None and patient_id else None
    if not trigger or view is None or segment_idx is None:
        return trigger
    if not (0 <= int(segment_idx) < len(view["display"])):
        return trigger
    segment = view["display"][int(segment_idx)]
    span = locations.find_trigger(segment, trigger)
    return segment[span[0]:span[1]] if span else trigger


def segment_line(idx: int, text: str, max_chars: int = 160) -> str:
    """``#7 · He presented with recurrent seizures.``, how a segment is named in a list."""
    shown = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    return f"#{idx} · {shown}"
