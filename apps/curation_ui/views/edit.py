"""Edit mode, proposing changes to the ground-truth set.

Two proposals live here, and neither decides anything: each waits for Approve mode.

**Add a phenotype.** A segment, a trigger word, an HPO term, and any qualifiers that apply.
**Remove an annotation.** Any term Approve mode would rule on, with a reason.

The third thing that used to live here, commenting on the report, has moved to
:mod:`apps.curation_ui.views.comments`, mounted *outside* the mode switch. It was disappearing
the moment anybody switched to Approve, which is backwards: the reader who notices something about
a whole report is usually the one adjudicating it.

The asymmetry that used to exist is what the second one fixes: Approve mode could always *remove*
an annotation outright, but the person reading the report closely had no way to say "this one is
wrong, and here is why", they had to either hold it in their head until they switched modes, or
exercise a verdict they were not there to exercise. A proposal with a reason attached is the thing
that actually travels between the reader and the adjudicator.


A suggestion is three things and the panel refuses to record it without all three: the **segment**
it comes from, the **trigger word** in that segment, and the **HPO term**. That is not paperwork.
One of the files this app repairs is still code-only, and the two that are not were built by
annotators whose evidence is the whole reason they are worth curating. Every question anyone has
asked of a bare code, is this term right, did the model have a chance to find it, is this a
labelling error or a retrieval failure, needs the sentence and the words. A suggestion recorded
without them would reproduce the file being fixed.

The **qualifiers** are optional and ride along with it, ticked on the same form and saved by the
same click. Whether a finding belongs to a relative, or is negated, or rests only on a lab value,
is decided in the same breath as the term, asking for it afterwards, on another screen, is how it
goes unrecorded. They clear on save while the trigger word does not: two phenotypes from one phrase
is a normal pass, but carrying "negated" onto the next suggestion would record a claim nobody made.

The HPO field is a ``dcc.Dropdown`` whose options start empty and are filled by
:mod:`apps.curation_ui.search` from the typed query, capped at fifty. Nothing about the ontology
is sent to the browser. What crosses the tunnel is a query and a short list. This is the panel that
would otherwise ship 19 000 options on every page load, which is what makes ``app/annotation_ui``
unusable over SSH.

The form's controls live in the static layout rather than inside a rendered block, so their
callbacks always have a target and typing is never interrupted by a redraw of the panel around
them. Buttons that *are* inside rendered blocks all check ``n_clicks``: Dash resets a re-created
button's count to zero, which fires its callback with nobody having pressed anything.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from hpo_extraction.curation import labels as vocab
from .. import search as search_mod
from .. import theme
from hpo_extraction.curation import store
from . import approve, common, editor, labelling

VIEW_ID = "edit"

#: Hover text for the trigger-word label, one string, so both trigger fields describe the same
#: key. The behaviour itself lives in ``assets/trigger_complete.js``.
TAB_HELP = editor.TAB_HELP

#: Where in this file's own layout the editor panels live. Edit and Approve are both in the DOM at
#: once, so a suggestion visible in both would give its editor the same pattern id twice, which
#: Dash refuses to render. See ``editor.ident``.
AT = "edit"


def layout() -> html.Div:
    """The Dash layout of this view."""
    return html.Div(
        [
            theme.card(
                "New phenotype",
                html.Div([
                    html.Div(id="edit-seg-text", className="edit-seg"),
                    html.Label("Trigger word", className="field-label", title=TAB_HELP),
                    # ``autoComplete="off"``: the browser's own history dropdown opens over the
                    # completion hint and eats the same Tab, so the two cannot both be on.
                    dcc.Input(id="edit-trigger", type="text", debounce=False, autoComplete="off",
                              placeholder="start typing, then Tab, or select the words in the report",
                              className="path-input"),
                    html.Button("Use selection", id="edit-use-selection", className="copy-btn",
                                title="Copy whatever you have highlighted in the report into the "
                                      "trigger field"),
                    html.Label("HPO term", className="field-label", style={"marginTop": "10px"}),
                    dcc.Dropdown(id="edit-hpo", options=[], value=None, className="hpo-pick",
                                 placeholder="type ≥2 characters, label, synonym, "
                                             "or an id like 0003645"),
                    html.Label("Note (optional)", className="field-label",
                               style={"marginTop": "10px"}),
                    dcc.Input(id="edit-note", type="text", debounce=True, className="path-input",
                              placeholder="why this is right, or what you are unsure about"),
                    # The qualifiers ride on the suggestion, not being a second job done
                    # after it. Whether a finding belongs to a relative, or is negated, or is only
                    # a lab value, is decided in the same breath as the term itself, asking for it
                    # later, on another screen, is how it goes unrecorded.
                    html.Label("Qualifiers (optional)", className="field-label",
                               style={"marginTop": "10px"}),
                    labelling.qualifier_controls("edit-labels"),
                    html.Button("Add suggestion", id="edit-add", className="primary-btn",
                                style={"marginTop": "10px"}),
                    html.Div(id="edit-feedback"),
                ]),
                subtitle="Segment, trigger word and term are all required, a code without "
                         "evidence is the problem this app was built to fix.",
            ),
            html.Div(id="edit-list", style={"marginTop": "12px"}),

            theme.card(
                "Propose a deletion",
                html.Div(id="edit-delete-list"),
                subtitle="The terms Approve mode rules on, the confirmed set, plus the prior_annotation "
                         "annotation for anything it no longer carries, and anything already "
                         "accepted in. Give a reason: it is what the adjudicator reads.",
            ),
        ],
        id="edit-panel",
    )


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("edit-hpo", "options"),
        Input("edit-hpo", "search_value"),
        State("edit-hpo", "value"),
    )
    def search_hpo(query, chosen):
        """Server-side search. The dropdown never holds more than one page of the ontology."""
        registry = common.registry_or_none()
        if registry is None:
            return []
        options = registry.search.search(query or "")
        # Keep the chosen value in the option list, or Dash renders the selection as blank the
        # moment the query that found it stops matching.
        if chosen and all(o["value"] != chosen for o in options):
            options = [registry.search.option(chosen)] + options
        return options[: search_mod.MAX_RESULTS]

    @app.callback(
        Output("edit-seg-text", "children"),
        Input("store-patient", "data"),
        Input("store-segment", "data"),
    )
    def show_segment(patient_id, segment_idx):
        return common.guard("Selected segment", render_selected, patient_id, segment_idx)

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Output("edit-feedback", "children"),
        Output("edit-hpo", "value"),
        Output("edit-trigger", "value", allow_duplicate=True),
        Output("edit-note", "value"),
        Output("edit-labels", "value"),
        Input("edit-add", "n_clicks"),
        State("store-patient", "data"),
        State("store-segment", "data"),
        State("edit-trigger", "value"),
        State("edit-hpo", "value"),
        State("edit-note", "value"),
        State("edit-labels", "value"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def add(n_clicks, patient_id, segment_idx, trigger, code, note, qualifiers, dirty):
        # A falsy ``n_clicks`` means this button was re-created, not pressed. It is static today,
        # but a callback that records a judgement must not depend on that staying true.
        registry = common.registry_or_none()
        if not n_clicks or registry is None or not patient_id:
            raise PreventUpdate
        problem = _validate(registry, patient_id, segment_idx, trigger, code)
        if problem:
            return dash.no_update, theme.note(problem, "warn"), dash.no_update, \
                dash.no_update, dash.no_update, dash.no_update

        view = registry.patient(patient_id)
        registry.record(
            "suggest", patient_id,
            segment_idx=int(segment_idx),
            segment_text=view["display"][int(segment_idx)],
            hpo_code=code,
            hpo_name=registry.search.label(code),
            # Recorded as the *report* spells it, not as it was typed: the browser collapses the
            # tabs a lab table is padded with, and a quote that does not match the source is the
            # one defect this file cannot survive. See ``common.snap_trigger``.
            trigger_word=common.snap_trigger(registry, patient_id, segment_idx, trigger),
            note=(note or "").strip(),
            # Validated here, not trusted: what arrives is whatever the browser posted, and
            # a label from another scope, or from a newer version of this app, is dropped rather
            # than written into a file something will later group by.
            labels=vocab.clean(qualifiers, "annotation"),
        )
        saved = theme.note(f"Saved, {registry.search.label(code)} ({code}) on segment "
                           f"#{segment_idx}.", "info")
        # The term, the note and the qualifiers clear. The trigger does not. Two phenotypes from one
        # phrase is a normal pass ("developmental delay" → delay and its severity), and retyping it
        # is friction. The qualifiers clear because they are a claim about *this* term: carrying
        # "negated" onto the next suggestion is how a wrong one gets recorded silently.
        return (dirty or 0) + 1, saved, None, dash.no_update, "", []

    @app.callback(
        Output("edit-list", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
    )
    def draw_list(patient_id, _dirty, mode):
        return common.guard("Suggestions", render_list, patient_id, mode or "light")

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Input({"type": "withdraw", "key": ALL}, "n_clicks"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def withdraw(_clicks, patient_id, dirty):
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate
        registry.record("withdraw", patient_id, target_key=triggered["key"])
        return (dirty or 0) + 1

    # ── deletion proposals ───────────────────────────────────────────────────
    @app.callback(
        Output("edit-delete-list", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
    )
    def draw_deletable(patient_id, _dirty, mode):
        return common.guard("Deletable", render_deletable, patient_id, mode or "light")

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Input({"type": "del-propose", "code": ALL}, "n_clicks"),
        State({"type": "del-reason", "code": ALL}, "value"),
        State({"type": "del-reason", "code": ALL}, "id"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def propose_delete(_clicks, reasons, reason_ids, patient_id, dirty):
        """One callback for every row's button. The clicked *code* selects its own reason box.

        Every row's reason arrives in the state list, Dash builds pattern callbacks once, and the
        rows do not exist at that point, so the clicked id is what picks the right one out.
        """
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate

        code = triggered["code"]
        reason = (next((r for r, i in zip(reasons, reason_ids) if i.get("code") == code), "")
                  or "").strip()
        entry = next((e for e in deletable_rows(registry, patient_id) if e["code"] == code), None)
        if entry is None:
            raise PreventUpdate

        # One event per underlying row. The grouping is presentation. The log stays faithful to the
        # data model, so a later reader can still see which source's annotation was disputed.
        for key in entry["keys"]:
            registry.record("suggest_delete", patient_id, target_key=key, text=reason,
                            hpo_code=code, hpo_name=entry["name"])
        return (dirty or 0) + 1

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Input({"type": "del-withdraw", "code": ALL}, "n_clicks"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def withdraw_delete(_clicks, patient_id, dirty):
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate
        entry = next((e for e in deletable_rows(registry, patient_id)
                      if e["code"] == triggered["code"]), None)
        if entry is None:
            raise PreventUpdate
        for key in entry["keys"]:
            registry.record("withdraw_delete", patient_id, target_key=key)
        return (dirty or 0) + 1

    # Copying the browser's text selection into the trigger field. Clientside because the selection
    # only exists in the browser, there is no server-side way to know what the curator highlighted.
    app.clientside_callback(
        """
        function(n) {
            if (!n) { return window.dash_clientside.no_update; }
            const text = (window.getSelection ? String(window.getSelection()) : '').trim();
            return text ? text : window.dash_clientside.no_update;
        }
        """,
        Output("edit-trigger", "value", allow_duplicate=True),
        Input("edit-use-selection", "n_clicks"),
        prevent_initial_call=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# rendering, pure
# ──────────────────────────────────────────────────────────────────────────────
def _validate(registry, patient_id, segment_idx, trigger, code) -> str:
    """The message to show, or ``""`` when the suggestion is complete and consistent.

    A *new* phenotype needs all three parts, so this is :func:`common.validate_annotation` with both
    requirements on. Repairing an existing annotation turns them off, see
    :mod:`apps.curation_ui.views.editor`, and the rule about a trigger belonging to its segment
    is the same one in both cases because it is written once.
    """
    return common.validate_annotation(registry, patient_id, segment_idx, trigger, code,
                                      require_trigger=True, require_code=True)


def render_selected(patient_id, segment_idx):
    """The selected segment of *patient_id*, ready for a new suggestion."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return theme.note(common.NO_PATIENT, "info")
    if segment_idx is None:
        return theme.note("No segment selected, click a sentence in the report.", "warn")
    view = registry.patient(patient_id)
    if view is None or not (0 <= int(segment_idx) < len(view["display"])):
        return theme.note("Selected segment is out of range for this report.", "warn")
    return html.Div([
        html.Div(f"segment #{segment_idx}", className="stat-sub"),
        html.Div(view["display"][int(segment_idx)], className="edit-seg-body"),
    ])


def render_list(patient_id, mode: str = "light"):
    """The pending suggestions of *patient_id*."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return None
    rows = sorted((r for r in registry.rows_for(patient_id)
                   if r.get("source") == "new" and r.get("status") == "suggested"),
                  key=lambda r: (_int(r.get("segment_idx")), r.get("hpo_code", "")))
    if not rows:
        return theme.card("Open suggestions", theme.empty(
            "Nothing proposed for this patient yet."))

    colors = theme.status_colors(mode)
    items = [
        html.Div([
            html.Div([
                common.chip("suggested", colors["suggested"],
                            title=theme.STATUS_HELP["suggested"]),
                common.code_chip(row["hpo_code"], row.get("hpo_name") or row["hpo_code"],
                                 theme.palette(mode)["text"],
                                 title=registry.search.describe(row["hpo_code"])),
            ], className="row-head"),
            html.Div(common.segment_line(_int(row.get("segment_idx")),
                                         str(row.get("segment_text", ""))),
                     className="stat-sub"),
            html.Div(f"trigger: “{row.get('trigger_word', '')}”", className="stat-sub"),
            html.Div(row["note"], className="stat-sub") if row.get("note") else None,
            html.Button("Withdraw", id={"type": "withdraw", "key": row["key"]},
                        className="copy-btn", n_clicks=0),
            # Repairable here as well as in Approve mode: noticing a typo in your own suggestion
            # happens while you are still reading the report, and making that a mode switch is how
            # it stops being fixed at all.
            editor.panel(registry, row["key"], AT,
                         segment_idx=_none_int(row.get("segment_idx")),
                         trigger_word=row.get("trigger_word", ""),
                         hpo_code=row.get("hpo_code", ""), note=row.get("note", ""),
                         labels=row.get("labels"), mode=mode),
            labelling.qualifier_chips(row.get("labels"), mode),
        ], className="row-card")
        for row in rows
    ]
    return theme.card(f"Open suggestions ({len(rows)})", html.Div(items),
                      subtitle="In segment order, like the report. Open until adjudicated in "
                               "Approve mode; withdrawing removes the row and the log keeps the "
                               "history.")


def deletable_rows(registry, patient_id: str) -> list[dict]:
    """The annotations a deletion can be proposed against, **one entry per HPO term**.

    Grouped by code, not by (source, code). A term two annotation files carry is two rows in the
    data model and one annotation in the curator's head. Leaving it split would mean proposing the
    same deletion twice, and forgetting the second one would leave the term in the curated export with
    no sign anything was wrong. The proposal is recorded against every underlying row, so the
    grouping is presentation over a faithful log, not a shortcut through it.

    "Existing" is the same set Approve mode rules on (``view["existing"]``) plus anything this app
    has already accepted into the curated set, not every record the inputs carry. A
    proposal is settled by the keep/remove buttons on the Approve row, so proposing a deletion
    against something Approve does not show would be a question nobody could answer.

    That rules out ``prior_annotation_2`` and un-adjudicated PhenoBERT detections. Both are reference: candidate
    *evidence*, not annotations that can enter the ground truth, and proposing to delete something nobody
    ever asserted would be a proposal about nothing. A PhenoBERT row that has been kept does appear, by then somebody has asserted it, and Approve carries it too.
    """
    view = registry.patient(patient_id)
    if view is None:
        return []
    rows = {row["key"]: row for row in registry.rows_for(patient_id)}

    #: ``{code: [(source, key)]}`` in a stable order, the rows Approve rules on first, then
    #: anything this app has curated in.
    by_code: dict[str, list[tuple[str, str]]] = {}
    for record in view["existing"]:
        by_code.setdefault(record["hpo_code"], []).append((record["source"], record["key"]))

    claimed = {key for pairs in by_code.values() for _, key in pairs}
    for key, row in rows.items():
        if key in claimed or row.get("status") not in store.IN_GOLD:
            continue
        by_code.setdefault(row.get("hpo_code", ""), []).append((row.get("source", ""), key))

    entries: list[dict] = []
    for code, pairs in by_code.items():
        if not code:
            continue
        members = [{"source": source, "key": key, "status": rows.get(key, {}).get("status", "")}
                   for source, key in pairs]
        proposed = [rows[key] for _, key in pairs
                    if rows.get(key, {}).get("status") == store.DELETE_SUGGESTED]
        name = next((rows[key].get("hpo_name") for _, key in pairs
                     if rows.get(key, {}).get("hpo_name")), "") or registry.search.label(code)
        entries.append({
            "code": code,
            "name": name,
            "keys": [key for _, key in pairs],
            "members": members,
            "proposed": bool(proposed),
            "delete_reason": proposed[0].get("delete_reason", "") if proposed else "",
            "delete_by": proposed[0].get("delete_by", "") if proposed else "",
        })

    # Open proposals first, then terms nobody has ruled on, then the settled ones, and inside
    # each band, the order the report reads, which is the order every other list in this column
    # is now in.
    def rank(entry):
        if entry["proposed"]:
            return 0
        return 1 if any(not m["status"] for m in entry["members"]) else 2

    def segment(entry):
        found = [rows[key].get("segment_idx") for key in entry["keys"] if key in rows]
        placed = [_int(value) for value in found if _int(value) >= 0]
        return min(placed) if placed else approve.NO_SEGMENT

    entries.sort(key=lambda e: (rank(e), segment(e), e["code"]))
    return entries


def render_deletable(patient_id, mode: str = "light"):
    """The suggestions of *patient_id* that the curator may still delete."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return theme.empty(common.NO_PATIENT)
    entries = deletable_rows(registry, patient_id)
    if not entries:
        return theme.empty("No existing annotation to remove for this patient.")

    palette = theme.palette(mode)
    colors = theme.status_colors(mode)

    cards = []
    for entry in entries:
        head = [common.code_chip(entry["code"], entry["name"], palette["text"],
                                 title=registry.search.describe(entry["code"]))]
        for member in entry["members"]:
            label = member["source"]
            if member["status"] and member["status"] != store.DELETE_SUGGESTED:
                label += f" · {member['status']}"
            head.append(common.chip(
                label,
                colors.get(member["status"], theme.source_color(member["source"], palette)),
                title=theme.SOURCE_HELP.get(member["source"], "")))
        body = [html.Div(head, className="row-head")]

        if entry["proposed"]:
            body.append(html.Div([
                html.Span("Deletion proposed", className="banner-title"),
                html.Div(f"“{entry['delete_reason'] or 'no reason given'}”",
                         className="banner-text"),
                html.Div(f", {entry['delete_by']}", className="stat-sub"),
            ], className="delete-banner"))
            body.append(html.Div([
                html.Button("Withdraw proposal",
                            id={"type": "del-withdraw", "code": entry["code"]},
                            className="copy-btn", n_clicks=0),
            ], className="row-actions"))
        else:
            body.append(html.Div([
                dcc.Input(id={"type": "del-reason", "code": entry["code"]}, type="text",
                          value="", debounce=True, className="path-input",
                          placeholder="why should this be removed?"),
                html.Button("Propose deletion",
                            id={"type": "del-propose", "code": entry["code"]},
                            className="copy-btn", n_clicks=0),
            ], className="row-actions"))
        cards.append(html.Div(body, className="row-card"))
    return html.Div(cards)


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _none_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
