"""Chrome, re-exported from ``app/tree_ui`` plus the two colour maps this app owns.

The palette, the panel/stat components and the note box are shared with the three analysis apps
on purpose: four UIs that look and read the same are one UI a reader has to learn once, and the
palette is a validated CVD-separated set that must not be extended by hand.

What this module adds is the vocabulary specific to curation:

``status_colors``  the lifecycle of a row, ``suggested`` → ``approved``/``rejected``, and the
                   verdicts ``kept``/``removed``/``needs_work`` on annotations that already exist
``source_colors``  where a row came from, one of the annotation files, PhenoBERT, or this app
"""

from __future__ import annotations

from dash import html

from apps.ui_common.theme import (  # noqa: F401 - re-exported as this app's chrome API
    DARK,
    LIGHT,
    empty,
    note,
    palette,
    panel,
    stat,
    stat_row,
)

#: Row lifecycle, worst → best is not the ordering here. This is a *status* map, so the reading is
#: "what has been decided", not "how bad is it".
STATUS_ORDER = ["suggested", "delete_suggested", "approved", "kept", "needs_work",
                "rejected", "removed"]

STATUS_HELP = {
    "suggested": "proposed in Edit mode, not yet adjudicated",
    "delete_suggested": "somebody has proposed removing this annotation, Remove accepts the "
                        "proposal, Keep rejects it. Out of the ground truth while it is open",
    "approved": "a suggestion accepted in Approve mode, enters the curated ground truth",
    "rejected": "a suggestion turned down, kept in the log, out of the ground truth",
    "kept": "an existing annotation confirmed correct, enters the curated ground truth",
    "removed": "an existing annotation judged wrong, out of the ground truth",
    "needs_work": "flagged for a second look; neither in nor out",
}

#: The provenances a row can carry. ``new`` is this app's own. The rest are read-only inputs.
#: ``raw`` is retired, ``holistic`` replaced it, but stays listed so a verdict recorded against it
#: before the swap still renders with a name and a colour instead of falling over on a lookup.
SOURCE_ORDER = ["prior_annotation", "daphne", "prior_annotation_2", "phenobert", "new", "raw"]

SOURCE_HELP = {
    "prior_annotation": "hcy_holistic_ground_truth.csv, one row per annotation, with the trigger word, "
                "the sentence it sits in and a character offset into the report",
    "daphne": "annotations_confirmed.csv, one row per annotation, already located to a segment, "
              "carrying a provenance and a confirmed flag",
    "prior_annotation_2": "hcy_ground_truth_marc2.csv, the second annotator's set, code-only: no trigger word "
             "and no sentence, so its terms get a lexical candidate origin instead of a mark",
    "phenobert": "PhenoBERT baseline detections, with character offsets into the report",
    "new": "proposed in this app",
    "raw": "hcy_ground_truth_raw.csv, the retired code-only ground truth, replaced by "
           "hcy_holistic_ground_truth.csv. Only rows adjudicated before the swap carry it",
}


def status_colors(mode: str = "light") -> dict:
    """``{status: colour}``. Decided-in green, decided-out muted, undecided amber."""
    p = palette(mode)
    return {
        "suggested": p["warning"],
        "delete_suggested": p["critical"],
        "approved": p["good"],
        "kept": p["good"],
        "needs_work": p["serious"],
        "rejected": p["text_muted"],
        "removed": p["text_muted"],
    }


#: Which validated hue each source owns, as an index into ``palette()["categorical"]``.
#:
#: **Named rather than positional.** This used to be ``SOURCE_ORDER.index(source)``, which meant a
#: source inserted into that list silently re-coloured every source after it, and colour here is
#: identity, so a curator who has learnt "violet is the confirmed set" would have been quietly
#: taught something false by an unrelated edit.
#:
#: ``daphne`` takes the violet slot and ``new`` the green one it vacated: the confirmed set is the
#: thing being adjudicated on most screens, and it has to be the one that separates.
SOURCE_SLOT = {"prior_annotation": 0, "new": 1, "prior_annotation_2": 2, "daphne": 4, "raw": 5}

#: Sources that take a *named* colour instead of a categorical one. PhenoBERT is reference, it
#: never takes a verdict and nothing about it can enter the curated ground truth, so it reads as chrome
#:, not as one more annotator competing for attention with the files that can.
SOURCE_NAMED = {"phenobert": "neutral"}


def source_color(source: str, colors: dict) -> str:
    """One source's colour, given a palette *dict*, not a mode.

    The reader draws marks from a palette it was handed, so it cannot re-derive the mode. Taking
    the dict keeps the colour of a source identical between a mark in the report and a chip beside
    it, the whole point of a source having a colour at all. A source this version does not know is
    neutral, not an ``IndexError``: a log outlives the vocabulary that wrote it.
    """
    cats = colors["categorical"]
    named = SOURCE_NAMED.get(source)
    if named:
        return colors.get(named) or cats[0]
    slot = SOURCE_SLOT.get(source)
    if slot is None:
        return colors.get("neutral") or cats[0]
    return cats[slot % len(cats)]


def source_colors(mode: str = "light") -> dict:
    """``{source: colour}``. Identity, not severity, no source is better than another.

    Built from ``SOURCE_ORDER``, not written out, so adding a source is one entry in one
    list. Callers still read it with ``.get`` and a fallback: a log written against a source this
    version does not know about must render, not raise.
    """
    colors = palette(mode)
    return {source: source_color(source, colors) for source in SOURCE_ORDER}


def card(title: str, body, subtitle: str = "") -> html.Div:
    """A titled block, without the copy toolbar ``panel`` carries.

    The analysis apps' ``panel`` exists so a figure and its numbers can be lifted into a slide or a
    findings file. Nothing on a curation screen is going into a slide, and a Copy-Markdown button
    on every card would be four buttons of noise around the one button that counts. ``panel``
    stays re-exported for the Overview tab, which does export.
    """
    head = [html.Div(title, className="card-title")]
    if subtitle:
        head.append(html.Div(subtitle, className="card-sub"))
    return html.Div([html.Div(head, className="card-head"), body], className="card")
