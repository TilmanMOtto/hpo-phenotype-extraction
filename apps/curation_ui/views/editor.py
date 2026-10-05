"""Repairing an annotation in place, its segment, its trigger word, its term, its note.

Every annotation on this screen arrived from somewhere: two of the files carry a trigger word and a
position, one carries only a code, PhenoBERT carries an offset, and this app's own suggestions carry
whatever was typed. All of them can be **wrong in the same four ways**, the trigger points at the
wrong words, the sentence index belongs to a different segmentation, the code is a level too coarse,
the note says something that is no longer true, and until this module existed the only repair
available was a candidate button that offered a lexical guess, or nothing at all.

So the editor is one component, used at both levels: on a pending suggestion in Edit mode, and on
anything at all in Approve mode. One form rather than two means the rule about what a valid
annotation is (:func:`common.validate_annotation`) is enforced in one place, which is the only way
it stays true of both.

It also owns the row's **qualifiers**, negated, family, lab value, implicit, and the two kinds of
unsure. They are a property of the annotation, ticked on the form that creates it, so the form that
repairs it is the one place they change. The Approve row shows them as chips and offers no second
way to edit them, because a field editable from two places is a field whose two copies disagree.

**Jump to** is the missing half of *Use selected*. That button brings the report's selection into
the form. This one takes the form's segment out to the report, selecting it and scrolling to it.
Deciding whether a code fits means reading the sentence it came from, and on a long report that
sentence is off-screen, a verdict passed without going to look at it is a verdict guessed.

**An edit is not a verdict.** It writes an ``edit`` event, which rewrites only the fields it names
and never touches ``status``, repairing the evidence for a term and deciding whether that term
belongs are two judgements, and a log that fused them could not answer who kept it. Fixing a trigger
word on an annotation nobody has ruled on leaves it as undecided as it was.

**Editing is not deleting, either.** Changing the code on a row does not remove the original: the
row keeps its key, the address of the annotation *as its file wrote it*, and the screen shows both,
so "prior_annotation_2 said HP:0001250, we curated it as HP:0007359" stays legible in the export and in the log.
An annotation that should not exist at all is a deletion proposal, which is a different button with
a reason attached.

Two Dash mechanics this module has to respect, both of which have already cost this app a bug:

**Duplicate ids.** Edit mode and Approve mode are both in the layout at once, the mode switch only
toggles ``display``, so a suggestion visible in both would give its editor's components the same
pattern id twice, which Dash refuses to render at all. Every id therefore carries ``at``, the panel
it was drawn in, and every callback matches on the whole id, not on the key.

**Re-created inputs.** These panels live inside blocks that re-render, so a recreated button arrives
with ``n_clicks=0``, a change to a callback Input, which fires the callback with nobody having
pressed anything. Every callback here checks ``any(clicks)`` first. Without that, paging to the next
patient would re-fire Save with the previous patient's fields still in the DOM.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from hpo_extraction.curation import labels as vocab
from .. import search as search_mod
from .. import theme
from hpo_extraction.curation import store
from . import common, labelling

VIEW_ID = "editor"

#: Hover text for the trigger-word label, shared with :mod:`apps.curation_ui.views.edit`.
#: The behaviour itself lives in ``assets/trigger_complete.js``.
TAB_HELP = ("Tab completes the word from this segment, from the first letter, or from an empty "
            "field; Tab again appends the next word, so a whole phrase costs one key, spelled "
            "as the report spells it. ↑/↓ cycle the alternatives, Enter jumps to the "
            "HPO term, Esc dismisses the suggestion.")


def ident(kind: str, key: str, at: str) -> dict:
    """The pattern id of one field of one row's editor. ``at`` is what keeps the two panels apart."""
    return {"type": f"ann-{kind}", "key": key, "at": at}


def panel(registry, key: str, at: str, segment_idx=None, trigger_word: str = "",
          hpo_code: str = "", note: str = "", labels=None, mode: str = "light") -> html.Details:
    """The collapsed editor for one annotation.

    Collapsed because adjudicating is the job and repairing is the exception: an open form on every
    row would bury the verdict buttons that most rows only need. The summary line says what the row
    currently claims, so the common case, reading it and moving on, costs no clicks.
    """
    options = []
    if hpo_code and registry is not None and registry.search.known(hpo_code):
        options = [registry.search.option(hpo_code)]

    shown_segment = "" if segment_idx is None else str(segment_idx)
    summary = "Edit evidence"
    if shown_segment or trigger_word:
        summary += f", #{shown_segment or '?'} “{trigger_word or '—'}”"
    if labels:
        # Named in the collapsed summary as well as shown as chips on the row, because this is the
        # form that changes them and a curator looking for where to do that should not have to open
        # every panel to find out which one owns the field.
        summary += " · " + ", ".join(vocab.display(value) for value in labels)

    return html.Details([
        html.Summary(summary, className="details-summary"),
        html.Div([
            html.Div([
                html.Label("Segment", className="field-label"),
                dcc.Input(id=ident("seg", key, at), type="number", min=0, step=1,
                          value=segment_idx, debounce=True, className="path-input seg-input"),
                html.Button("Use selected", id=ident("useseg", key, at), className="copy-btn",
                            n_clicks=0,
                            title="take the segment currently selected in the report"),
                # The other direction, and the one that was missing. Checking whether a code fits
                # means reading the sentence it came from, and on a long report that sentence is
                # somewhere off-screen, so the only way to check was to scroll hunting for it,
                # which is the point at which a verdict starts being guessed instead.
                html.Button("Jump to", id=ident("jumpseg", key, at), className="copy-btn",
                            n_clicks=0,
                            title="select this row's segment in the report and scroll to it"),
            ], className="row-actions"),

            html.Label("Trigger word", className="field-label", title=TAB_HELP),
            html.Div([
                dcc.Input(id=ident("trigger", key, at), type="text", value=trigger_word,
                          debounce=True, className="path-input", autoComplete="off",
                          placeholder="start typing, then Tab"),
                html.Button("Use selection", id=ident("usesel", key, at), className="copy-btn",
                            n_clicks=0,
                            title="take whatever you have highlighted in the report"),
            ], className="row-actions"),

            html.Label("HPO term", className="field-label"),
            dcc.Dropdown(id=ident("hpo", key, at), options=options, value=hpo_code or None,
                         className="hpo-pick",
                         placeholder="type ≥2 characters, label, synonym, or an id like 0003645"),

            html.Label("Note", className="field-label"),
            dcc.Input(id=ident("note", key, at), type="text", value=note, debounce=True,
                      className="path-input", placeholder="what you changed, or why"),

            html.Label("Qualifiers", className="field-label"),
            labelling.qualifier_controls(ident("labels", key, at), labels),

            html.Div([
                html.Button("Save edit", id=ident("save", key, at), className="primary-btn",
                            n_clicks=0),
            ], className="row-actions"),
            html.Div(id=ident("feedback", key, at)),
        ], className="ann-editor"),
    ], className="ann-editor-wrap")


# ──────────────────────────────────────────────────────────────────────────────
def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output({"type": "ann-hpo", "key": ALL, "at": ALL}, "options"),
        Input({"type": "ann-hpo", "key": ALL, "at": ALL}, "search_value"),
        State({"type": "ann-hpo", "key": ALL, "at": ALL}, "value"),
    )
    def search_hpo(queries, chosen):
        """Server-side search, per row. Only the dropdown being typed into is answered.

        Every other row's options come back ``no_update``, filling them all would ship one page of
        the ontology per annotation on every keystroke, which is the cost this app's whole search
        design exists to avoid.
        """
        registry = common.registry_or_none()
        fired = _fired_index()
        if registry is None or fired is None:
            raise PreventUpdate
        at, blank = fired
        options = registry.search.search(queries[at] or "")
        current = chosen[at] if at < len(chosen) else None
        # Keep the chosen value in the option list, or Dash renders the selection as blank the
        # moment the query that found it stops matching.
        if current and all(option["value"] != current for option in options):
            options = [registry.search.option(current)] + options
        blank[at] = options[: search_mod.MAX_RESULTS]
        return blank

    @app.callback(
        Output({"type": "ann-seg", "key": ALL, "at": ALL}, "value"),
        Input({"type": "ann-useseg", "key": ALL, "at": ALL}, "n_clicks"),
        State("store-segment", "data"),
        prevent_initial_call=True,
    )
    def use_selected_segment(clicks, segment_idx):
        fired = _fired_index()
        if fired is None or not any(clicks or []) or segment_idx is None:
            raise PreventUpdate
        at, blank = fired
        blank[at] = int(segment_idx)
        return blank

    @app.callback(
        Output("store-segment", "data", allow_duplicate=True),
        Input({"type": "ann-jumpseg", "key": ALL, "at": ALL}, "n_clicks"),
        State({"type": "ann-seg", "key": ALL, "at": ALL}, "value"),
        prevent_initial_call=True,
    )
    def jump_to_segment(clicks, segments):
        """Make this row's segment the selected one. The clientside half then scrolls to it.

        Selecting, not only scrolling is deliberate: the segment lights up, so the sentence
        the row is claiming is unambiguous once you arrive. It reads the row's *input*, not
        the row's stored value, so jumping to a segment you have just typed in, to check it before
        saving, works, which is the case this is most useful in.
        """
        fired = _fired_index()
        if fired is None or not any(clicks or []):
            raise PreventUpdate
        at, _ = fired
        segment_idx = _int_or_none(segments[at])
        if segment_idx is None:
            raise PreventUpdate
        return segment_idx

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Output({"type": "ann-feedback", "key": ALL, "at": ALL}, "children"),
        Input({"type": "ann-save", "key": ALL, "at": ALL}, "n_clicks"),
        State({"type": "ann-seg", "key": ALL, "at": ALL}, "value"),
        State({"type": "ann-trigger", "key": ALL, "at": ALL}, "value"),
        State({"type": "ann-hpo", "key": ALL, "at": ALL}, "value"),
        State({"type": "ann-note", "key": ALL, "at": ALL}, "value"),
        State({"type": "ann-labels", "key": ALL, "at": ALL}, "value"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def save(clicks, segments, triggers, codes, notes, qualifiers, patient_id, dirty):
        """Write one ``edit`` event for the row whose Save was pressed.

        The whole form is sent, not a diff: the event names every field the curator can see, so the
        log line is a complete statement of what the annotation says after the edit, not
        something only reconstructable by re-running what it said before.
        """
        registry = common.registry_or_none()
        fired = _fired_index()
        triggered = dash.ctx.triggered_id
        # A falsy click count means these buttons were re-created, not pressed.
        if registry is None or fired is None or not any(clicks or []) or not patient_id:
            raise PreventUpdate
        at, blank = fired

        segment_idx = _int_or_none(segments[at])
        trigger = (triggers[at] or "").strip()
        code = codes[at] or ""
        note = (notes[at] or "").strip()

        problem = common.validate_annotation(registry, patient_id, segment_idx, trigger, code,
                                             require_trigger=False, require_code=False)
        if problem:
            blank[at] = theme.note(problem, "warn")
            return dash.no_update, blank

        view = registry.patient(patient_id)
        # ``labels`` is always named, even when empty, that is what makes unticking the last
        # qualifier possible. The generic "write any truthy field" rule the fold applies to other
        # actions cannot express clearing anything, which is why ``edit`` does not use it.
        # Same rule as a new suggestion: what is written is the report's own spelling of the
        # trigger, whitespace and all, never the flattened string the browser posted back.
        fields = {"note": note,
                  "trigger_word": common.snap_trigger(registry, patient_id, segment_idx, trigger),
                  "labels": vocab.clean(qualifiers[at], "annotation")}
        if segment_idx is not None and view is not None and 0 <= segment_idx < len(view["display"]):
            fields["segment_idx"] = segment_idx
            fields["segment_text"] = view["display"][segment_idx]
        if code:
            fields["hpo_code"] = code
            fields["hpo_name"] = registry.search.label(code)

        registry.record("edit", patient_id, target_key=triggered["key"], **fields)
        # No feedback line: the panel re-renders on the dirty bump and would wipe it anyway, and a
        # message that survives its own row being rebuilt is a message about the wrong row.
        return (dirty or 0) + 1, blank

    # Copying the browser's text selection into one row's trigger field. Clientside because the
    # selection exists only in the browser. Pattern-matching because there is one field per row, and
    # ``outputs_list`` is what tells the browser which of them the pressed button belongs to.
    app.clientside_callback(
        """
        function(clicks) {
            const ctx = window.dash_clientside.callback_context;
            const outputs = ctx.outputs_list || [];
            const blank = outputs.map(() => window.dash_clientside.no_update);
            if (!clicks || !clicks.some(function (c) { return c; })) { return blank; }
            const fired = ctx.triggered && ctx.triggered.length ? ctx.triggered[0] : null;
            if (!fired || !fired.value) { return blank; }
            const text = (window.getSelection ? String(window.getSelection()) : '').trim();
            if (!text) { return blank; }
            const id = JSON.parse(fired.prop_id.slice(0, fired.prop_id.lastIndexOf('.')));
            for (let i = 0; i < outputs.length; i += 1) {
                if (outputs[i].id.key === id.key && outputs[i].id.at === id.at) {
                    blank[i] = text;
                }
            }
            return blank;
        }
        """,
        Output({"type": "ann-trigger", "key": ALL, "at": ALL}, "value"),
        Input({"type": "ann-usesel", "key": ALL, "at": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )


def edited_fields(row: dict, record: dict | None) -> list[str]:
    """Which of an annotation's fields this app has changed away from what its file said.

    Shown as a chip on the row. A curated ground-truth set whose provenance is "somebody edited it at some
    point" is not much better than one with no provenance at all, and the log is the wrong place to
    look while deciding whether to keep the term.
    """
    if record is None or not row:
        return []
    changed = []
    if row.get("hpo_code") and row["hpo_code"] != record.get("hpo_code"):
        changed.append("term")
    if (row.get("trigger_word") or "") != (record.get("trigger_word") or ""):
        changed.append("trigger")
    return changed


#: Re-exported so a caller can name the action without importing ``store`` for one constant.
EDIT_ACTION = "edit"
EDITABLE = store.EDITABLE


def _fired_index():
    """``(index, [no_update, …])`` for the row whose control fired, or ``None``.

    The index comes from the **first Input's** wildcard list, never from ``outputs_list``. Dash
    orders every wildcard list in a callback the same way, so one index addresses the States and the
    pattern Output alike, but ``outputs_list`` is shaped like the *outputs*, so a callback with a
    plain output beside a pattern one gets a nested list there and indexing it would silently
    address the wrong row.

    Matched on ``key`` **and** ``at``: one annotation can be on screen in both panels at once, and
    answering the copy that was not clicked is as wrong as answering a different annotation.
    """
    triggered = dash.ctx.triggered_id
    inputs = (dash.ctx.inputs_list or [None])[0]
    if not isinstance(triggered, dict) or not isinstance(inputs, list):
        return None
    for index, entry in enumerate(inputs):
        ident = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(ident, dict) and ident.get("key") == triggered.get("key") \
                and ident.get("at") == triggered.get("at"):
            return index, [dash.no_update] * len(inputs)
    return None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
