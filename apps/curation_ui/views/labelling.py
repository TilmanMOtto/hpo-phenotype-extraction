"""The report-label controls, and the read-only chips an annotation's qualifiers render as.

This module used to own both *levels*, the same widgets, the same event, differing only in scope.
It no longer does, and the asymmetry is deliberate. A report is characterised once,
from a fixed vocabulary, so it earns a panel with a Save button. An annotation's qualifiers are
ticked while the annotation is being written and repaired in the same form as its trigger word, so
what is left here for them is :func:`qualifier_controls` (the picker both those forms mount) and
:func:`qualifier_chips` (what a row shows).

That replaced a collapsed block on every Approve row carrying nineteen checkboxes, a grade and its
own Save button. A vocabulary that fine is one nobody applies, and an unapplied label aggregates to
nothing. Worse, it was a second job done *after* the judgement rather than part of it.

The report panel sits at the **top of the right column, outside the mode switch**, so it is there in
Edit and in Approve alike. One instance, not one per mode: two would be two sets of component
ids for one piece of state, and the copy that was not on screen when Save was pressed would quietly
win. Deciding a report is *heavy on family history* happens while reading it, which is a thing you
do in both modes.

The comment thread sits directly under it, in :mod:`apps.curation_ui.views.comments`, for the
same reason and by the same mechanism.

**Its controls are static, and only their ``value`` props are driven by callbacks.** The report
vocabulary is fixed, so the checklists and the grade radio can live in the layout for the life of
the page. Switching patient writes their values, not rebuilding them. That is not a
micro-optimisation, it is the fix for a whole class of bug. Re-rendering a container re-creates the
inputs inside it, and a re-created input arrives with the props its constructor declares, wiping
whatever the curator had just ticked, and resetting the Save button's ``n_clicks`` to ``0``, which
fires its callback with nobody having pressed anything. Both of those shipped. The per-row
qualifier pickers cannot use this trick, there is one per annotation and the set changes with the
patient, so the forms that mount them keep the ``n_clicks`` guards instead.

The controls **replace**, not accumulate: what the checkboxes say is what gets recorded.
Merging would make unticking impossible, and a label that cannot be removed is a label nobody will
risk applying.

Saving is explicit, a Save button, not an on-change write. Ticking four boxes would otherwise be
four events and four CSV rewrites, and the intermediate three describe a judgement nobody made.

**The Save callback checks that ``n_clicks`` is truthy.** This button lives inside a block this
app re-renders, and Dash resets a recreated button's ``n_clicks`` to the ``0`` its constructor
declares, which is a change to a callback Input, so the callback fires. Without the guard, moving
to the next patient re-fires the save with the *previous* patient's checkboxes still in the DOM and
writes them against the new patient's id. That is not a hypothetical: it is what made report labels
appear to follow you from one report to the next, and what made unticking a box impossible.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from hpo_extraction.curation import labels as vocab
from .. import theme
from . import common

VIEW_ID = "labelling"


def controls(scope: str, ids, labels_value=None, difficulty_value: str = "",
             compact: bool = False) -> html.Div:
    """The grade radio and the label checklists for *scope*.

    *ids* is a callable ``(kind, group_id) -> component id``, which is the only thing that differs
    between the two levels: the sidebar's controls are addressed by a fixed id, an Approve row's by
    a pattern that carries the row's key. One implementation, not two near-copies, because
    two would drift and the drift would show up as a level whose labels quietly stop saving.

    Grouped by the question each group answers, with the group's help line above it: the difference
    between *Family confusion* and *Suspected* has to be readable at the moment of ticking, not
    looked up in a README afterwards.
    """
    blocks = [html.Div([
        html.Div("Difficulty", className="label-group-title"),
        dcc.RadioItems(
            id=ids("difficulty", None),
            options=[{"label": html.Span(f" {display}", title=help_text), "value": value}
                     for value, display, help_text in vocab.difficulty_for(scope)],
            value=difficulty_value or None,
            className="check-list",
        ),
    ], className="label-group")]

    for group in vocab.groups_for(scope):
        offered = vocab.labels_for(group, scope)
        in_group = {value for value, _, _ in offered}
        blocks.append(html.Div([
            html.Div(group["title"], className="label-group-title", title=group["help"]),
            html.Div(group["help"], className="stat-sub") if not compact else None,
            dcc.Checklist(
                id=ids("labels", group["id"]),
                options=[{"label": html.Span(display, title=help_text), "value": value}
                         for value, display, help_text in offered],
                value=[v for v in (labels_value or []) if v in in_group],
                className="check-list label-check",
            ),
        ], className="label-group"))

    return html.Div(blocks, className="label-controls")


def chips(values, difficulty: str, mode: str = "light"):
    """The current labels as hoverable chips, what a collapsed row shows."""
    palette = theme.palette(mode)
    out = []
    if difficulty:
        out.append(common.chip(vocab.display(difficulty), difficulty_color(difficulty, palette),
                               title=vocab.help_for(difficulty)))
    for value in values or ():
        out.append(common.chip(vocab.display(value), palette["neutral"],
                               title=vocab.help_for(value)))
    return out


def difficulty_color(value: str, palette: dict) -> str:
    """Easy → good, hard → serious, unfair → critical.

    A grade is a severity, so it reads as one, including in the sidebar's patient list, which is
    the other caller and the reason this is not private.
    """
    return {
        "easy": palette["good"],
        "medium": palette["warning"],
        "hard": palette["serious"],
        "unfair": palette["critical"],
    }.get(value, palette["neutral"])


def collect(values_by_group, scope: str) -> list[str]:
    """Flatten the per-group checklists into one validated, vocabulary-ordered list.

    A group that arrives as a bare string, not a list is taken as one value. Without that,
    ``extend`` would iterate it character by character and :func:`vocab.clean` would drop every
    character as unknown, the labels would vanish with no error anywhere, which is the one failure
    mode this feature cannot afford.
    """
    flat: list[str] = []
    for values in values_by_group or []:
        if isinstance(values, str):
            flat.append(values)
        else:
            flat.extend(values or [])
    return vocab.clean(flat, scope)


# ──────────────────────────────────────────────────────────────────────────────
def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("doc-difficulty", "value"),
        Output({"type": "doc-labels", "group": ALL}, "value"),
        Input("store-patient", "data"),
        State({"type": "doc-labels", "group": ALL}, "id"),
    )
    def load_report_labels(patient_id, ids):
        """Write the current patient's labels into the static controls.

        Driven by the patient alone. Not by ``store-dirty``: a save is the curator's own ticks
        being written down, so echoing them back would at best be a no-op and at worst would fight
        whatever they have typed since. And any *other* action bumping ``store-dirty``, adding a
        suggestion, passing a verdict, has nothing to say about this report's labels.

        The values are ordered by the matched ids, not by the vocabulary, because Dash pairs
        an ``ALL`` output with its components positionally and the two orders are not the same.
        """
        registry = common.registry_or_none()
        current = registry.patient_labels(patient_id) if (registry and patient_id) else {}
        chosen = set(current.get("labels") or ())

        by_group = {}
        for group in vocab.groups_for("patient"):
            by_group[group["id"]] = [value for value, _, _ in vocab.labels_for(group, "patient")
                                     if value in chosen]
        return current.get("difficulty") or None, [by_group.get((i or {}).get("group"), [])
                                                   for i in (ids or [])]

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Output("doc-label-status", "children", allow_duplicate=True),
        Input("doc-label-save", "n_clicks"),
        State({"type": "doc-labels", "group": ALL}, "value"),
        State("doc-difficulty", "value"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def save_document_labels(n_clicks, values, difficulty, patient_id, dirty):
        # ``n_clicks`` is 0 when this button was re-created, not pressed, see the module
        # docstring. Acting on it writes one patient's labels onto another.
        registry = common.registry_or_none()
        if not n_clicks or registry is None or not patient_id:
            raise PreventUpdate
        chosen = collect(values, "patient")
        # ``unfair`` is not offered at report level, and a payload that carries it anyway is not
        # something to store: the grade is validated against its own scope, not the global list.
        grade = difficulty if vocab.grade_ok(difficulty, "patient") else ""
        # Free text about a report is a *comment*, kept in its own file with an author and a
        # timestamp. Duplicating the latest one onto the label row would make it ambiguous which
        # was the record.
        registry.record("label", patient_id, labels=chosen, difficulty=grade)
        summary = ", ".join(vocab.display(v) for v in ([grade] if grade else []) + chosen)
        return (dirty or 0) + 1, theme.note(
            f"Saved for {patient_id}, {summary or 'nothing selected'}.", "info")


def report_layout() -> html.Div:
    """The whole report-label block, static.

    Nothing here is rebuilt while the app runs: the vocabulary is fixed, so the controls are built
    once and their values are written by :func:`register`'s ``load_report_labels``. The summary
    chips and the status line are the only children a callback replaces, and neither contains an
    input.
    """
    return html.Div([
        html.Details([
            html.Summary(html.Span(id="doc-label-summary"),
                         className="details-summary row-label-summary"),
            html.Div([
                html.Div("What kind of report is this? Applied once, it characterises every "
                         "annotation in it at once.", className="stat-sub"),
                controls("patient", _document_ids, compact=True),
                html.Button("Save report labels", id="doc-label-save", className="primary-btn",
                            n_clicks=0, style={"marginTop": "6px", "width": "100%"}),
            ], style={"paddingTop": "6px"}),
        ], open=False),
        html.Div(id="doc-label-status"),
    ], className="report-labels")


def report_summary(registry, patient_id, mode: str = "light"):
    """The chips in the panel's own ``<summary>``, the state, readable without opening it."""
    if registry is None or not patient_id:
        return html.Span("no patient", className="stat-sub")
    current = registry.patient_labels(patient_id)
    marks = chips(current["labels"], current["difficulty"], mode) or [
        html.Span("not characterised yet", className="stat-sub")]
    return html.Span([html.Span(f"{patient_id}, report labels ", style={"fontWeight": "600"}),
                      *marks])


def qualifier_chips(values, mode: str = "light"):
    """One annotation's qualifiers, as a read-only chip line, or nothing, when there are none.

    Read-only because there is one place to change them, the *Edit evidence* form on the
    same row, and a field editable from two places is a field whose two copies disagree. This used
    to be a collapsed ``Details`` carrying its own nineteen checkboxes, a grade and a Save button. The qualifiers now ride on the annotation itself, so what is left to do here is show them.

    **Nothing at all when the set is empty.** Most rows carry no qualifier, so a line saying so on
    each of them is a line of noise per row on the screen this app exists to make scannable, and
    it is not the only place the state is legible: the *Edit evidence* summary on the same row
    names them when set, and opening it shows the empty picker when not.
    """
    marks = chips(values, "", mode)
    return html.Div(marks, className="row-qualifiers") if marks else None


def qualifier_controls(component_id, values=None) -> html.Div:
    """The seven-checkbox qualifier picker, for the suggestion form and the inline editor.

    One implementation for both, and the vocabulary comes from :mod:`labels`, not being
    written out here, so the two forms cannot drift into offering different sets.

    Laid out as **inline pills that wrap**, not as a stack. There is one of these on the suggestion
    form and one inside every row's editor, in a 400px column. Seven stacked lines per row pushed
    the verdict buttons of the row below off the screen, and a control you have to scroll past on
    every annotation is one that stops being read. The report-level checklists keep the stacked
    layout, their labels are sentences ("Needs clinical inference") and there are nineteen of them,
    so wrapping would produce a ragged block that is harder to scan, not easier.
    """
    offered = vocab.labels_for(vocab.GROUPS[0], "annotation")
    chosen = set(values or ())
    return html.Div([
        dcc.Checklist(
            id=component_id,
            options=[{"label": html.Span(display, title=help_text), "value": value}
                     for value, display, help_text in offered],
            value=[value for value, _, _ in offered if value in chosen],
            className="check-list qualifier-check",
        ),
    ], className="label-group qualifier-controls")


def _document_ids(kind: str, group_id):
    """Fixed ids, there is one report panel on screen, so it needs no key."""
    return "doc-difficulty" if kind == "difficulty" else {"type": "doc-labels", "group": group_id}
