"""HCY gold-set curation UI, Dash entry point.

    python apps/curation_ui/app.py --port 8055        # on the cluster login node
    ssh -L 8055:localhost:8055 <cluster>                 # then open http://localhost:8055

CPU-only: no GPU, no model, no PhenoBERT rerun. Every source is read from disk, and the heaviest
thing in the process is one ``HPOTree``.

    python apps/curation_ui/app.py --selftest         # every panel, headless

runs the app against a synthetic fixture (or ``--hcy-dir``) and exits non-zero on the first
problem. That is the acceptance check. See ``README.md``.

**One patient, two modes.** The sidebar picks the patient and the mode. The report is the same
either way, and only the right column changes. Edit proposes, Approve adjudicates. Both write to
the same append-only log, so switching modes mid-patient costs nothing and loses nothing.

**Nothing is batched.** Every click is one appended event and one atomic CSV rewrite, both done
before the screen says "saved". A curation session that lies about what it has persisted is worse
than one that is slow, and at this cohort's size it is not slow.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import socket
import sys

# Importable without `pip install -e .`, which is how the other analysis apps are run.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import dash  # noqa: E402
from dash import ALL, Input, Output, State, dcc, html  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402

from hpo_extraction.curation import labels as vocab, sources  # noqa: E402
from apps.curation_ui import state, theme  # noqa: E402
from hpo_extraction.ontology import nltk_data  # noqa: E402
from apps.curation_ui.registry import Registry  # noqa: E402
from apps.curation_ui.views import (  # noqa: E402
    approve, comments, common, edit, editor, labelling, locate, overview, reader,
)  # noqa: E402

logger = logging.getLogger(__name__)

MODES = [("edit", "Edit, propose"), ("approve", "Approve, adjudicate")]

#: The sidebar's editable paths, in load order. Derived from ``sources.GOLD_SOURCES`` so adding an
#: annotation file is one entry in that tuple rather than one here, one in the loader, one in the
#: registry and one in this callback's signature, which is how the fifth one ends up half-wired.
SOURCE_FIELDS = [
    ("segments", sources.SEGMENTS_FILE),
    *[(spec["path_key"], spec["label"]) for spec in sources.GOLD_SOURCES],
    ("phenobert", "PhenoBERT baseline run directory"),
    ("hcy_dir", "hcy directory (the log is written to <hcy>/curation/)"),
]
SOURCE_KEYS = [key for key, _ in SOURCE_FIELDS]

#: The patient-list filter. One value, so the checklist is either empty or carries this.
FILTER_UNLABELLED = "unlabelled"
FILTER_HELP = ("show only reports with no difficulty grade and no report label yet, the queue "
               "for a characterisation pass")

#: The root font size, in px, that every ``rem`` in ``assets/style.css`` is a fraction of.
#: 14 is the stylesheet's own baseline, so the default is a no-op. The range stops where the
#: 400px right column would start wrapping its own buttons.
FONT_SIZE_MIN, FONT_SIZE_MAX, FONT_SIZE_DEFAULT = 11, 20, 14


def build_layout(registry: Registry) -> html.Div:
    """The page layout: report list, report reader, and the Edit and Approve panels."""
    paths = registry.paths
    first = registry.patient_ids[0] if registry.patient_ids else None

    sidebar = html.Div([
        html.H1("HCY curation", className="brand"),
        html.Div("HCY ground truth, report by report", className="brand-sub"),

        html.Div([
            html.Label("Mode", className="field-label"),
            dcc.RadioItems(id="mode-toggle", value="edit",
                           options=[{"label": f" {label}", "value": value}
                                    for value, label in MODES]),
        ], className="sidebar-section"),

        html.Div([
            html.Div([
                html.Button("◀ Prev", id="pat-prev", className="copy-btn", n_clicks=0),
                html.Button("Next ▶", id="pat-next", className="copy-btn", n_clicks=0),
            ], className="row-actions"),
            html.Button("Mark patient confirmed", id="pat-confirm", className="primary-btn",
                        n_clicks=0, style={"marginTop": "6px", "width": "100%"}),
            html.Div(id="pat-status"),
            html.Label("Patients", className="field-label", style={"marginTop": "10px"}),
            # One box, unticked by default: the cohort is the thing being worked through, so the
            # unfiltered list stays what the app opens on. Ticked, it becomes the queue of reports
            # nobody has characterised yet, which is the pass this vocabulary exists for, and the
            # one question the flags in the list itself cannot answer at a glance.
            #
            # *not* persisted across reloads, unlike the text size. The page opens on
            # ``patient_ids[0]``, which a restored filter could well hide, so the app would come
            # back up showing a list with nothing selected in it, for a reason last week's session
            # knows and this one does not.
            dcc.Checklist(id="patient-filter", value=[], className="check-list pat-filter",
                          options=[{"label": html.Span(" Unlabelled reports only",
                                                       title=FILTER_HELP),
                                    "value": FILTER_UNLABELLED}]),
            html.Div(id="patient-list", className="patient-list"),
        ], className="sidebar-section"),

        html.Details([
            html.Summary("Data source", className="field-label details-summary"),
            html.Div([
                *[html.Div([
                    html.Label(label, className="field-label"),
                    dcc.Input(id=f"src-{key}", value=paths.get(key, ""), type="text",
                              debounce=True, className="path-input"),
                ]) for key, label in SOURCE_FIELDS],
                html.Button("Load", id="src-load", className="copy-btn",
                            style={"marginTop": "8px"}),
                html.Div(f"Curation log: {registry.log.path}", className="stat-sub",
                         style={"marginTop": "8px"}),
            ], style={"paddingTop": "10px"}),
        ], open=not registry.patient_ids),

        html.Div(id="src-status"),
        html.Div(id="save-status", className="sidebar-section"),

        html.Div([
            html.Label("Display", className="field-label"),
            dcc.RadioItems(id="theme-toggle", inline=True, value="light",
                           options=[{"label": " Light", "value": "light"},
                                    {"label": " Dark", "value": "dark"}]),
            # Curation happens over an SSH tunnel on whatever monitor the curator has, and the
            # report is the thing being read closely for an hour at a time. The slider drives one
            # custom property. The stylesheet is entirely `rem`, so the report, the chips and the
            # Overview grid move together, not the report alone.
            html.Label("Text size", className="field-label", style={"marginTop": "12px"}),
            dcc.Slider(id="font-size", min=FONT_SIZE_MIN, max=FONT_SIZE_MAX, step=1,
                       value=FONT_SIZE_DEFAULT, included=False,
                       marks={size: {"label": str(size)}
                              for size in (FONT_SIZE_MIN, FONT_SIZE_DEFAULT, FONT_SIZE_MAX)},
                       tooltip={"placement": "bottom"},
                       # Dash's own persistence, not a second store fed back into the slider:
                       # a store that both writes and is written by one component is a circular
                       # dependency and Dash refuses to start at all. The restored value fires the
                       # callback below on load, which is what puts it on the page.
                       persistence=True, persistence_type="local"),
        ], className="sidebar-section"),
    ], className="sidebar")

    # The report column is a **sibling** of the right-hand panels, not living inside the
    # Curate tab, because Curate and Evidence location both read the same report and ``reader.layout()`` cannot
    # be mounted twice: it owns ``reader-body`` and its segments and marks are pattern ids, so a
    # second copy is a duplicate-id layout, which Dash refuses to render at all. One reader, two
    # right-hand columns, and the tab decides which column is on screen.
    curate = html.Div([
        html.Div(reader.layout(), className="curate-reader", id="col-reader"),
        html.Div([
            # Above the mode switch on purpose: characterising the report, and saying out loud
            # what confused you about it, are things you do while reading it, which happens in
            # Edit and in Approve alike. One instance each, not one per mode: two would be two sets
            # of component ids for one piece of state, and the copy not on screen when Save was
            # pressed would quietly win.
            labelling.report_layout(),
            comments.layout(),
            html.Div(edit.layout(), id="wrap-edit"),
            html.Div(approve.layout(), id="wrap-approve"),
        ], className="curate-panel", id="col-curate"),
        html.Div(locate.layout(), className="curate-panel", id="col-anchor"),
    ], className="curate", id="view-curate")

    main = html.Div([
        dcc.Tabs(id="tabs", value="curate", className="tabs-bar", children=[
            dcc.Tab(label="Curate", value="curate"),
            dcc.Tab(label="Locate evidence", value="anchor"),
            dcc.Tab(label="Overview", value="overview"),
        ]),
        html.Div([
            curate,
            html.Div(overview.layout(), id="view-overview"),
        ]),
    ], className="main")

    return html.Div([
        dcc.Store(id="store-theme", data="light"),
        # The current text size, written by the slider's clientside callback. The *persistence*
        # across reloads is the slider's own (``persistence_type="local"``), not this store's.
        dcc.Store(id="store-fontsize", data=FONT_SIZE_DEFAULT),
        # The segment the report was last scrolled to. Written by the scroll callback purely so it
        # has somewhere to put its Output, the effect is the scroll itself.
        dcc.Store(id="store-scrolled", data=None),
        # Browser state, and small: a patient id, a segment index, a mode and a
        # counter. The reports, the ontology and the folded log stay in the process.
        dcc.Store(id="store-patient", data=first),
        dcc.Store(id="store-segment", data=None),
        dcc.Store(id="store-dirty", data=0),
        # The Prev/Next click counts this app has already acted on. See ``pick_patient``.
        dcc.Store(id="store-nav", data={"prev": 0, "next": 0}),
        html.Div([sidebar, main], className="shell"),
    ])


def register_callbacks(app: dash.Dash) -> None:
    """Register the page-level callbacks (report selection, mode switch, confirmation, theme)."""
    for module in (reader, edit, approve, locate, editor, labelling, comments, overview):
        module.register(app)

    @app.callback(
        Output("view-curate", "style"),
        Output("col-reader", "style"),
        Output("col-curate", "style"),
        Output("col-anchor", "style"),
        Output("view-overview", "style"),
        Output("wrap-edit", "style"),
        Output("wrap-approve", "style"),
        Input("tabs", "value"),
        Input("mode-toggle", "value"),
    )
    def switch_view(tab, mode):
        """Which columns are on screen. Every panel stays in the layout. Only ``display`` moves.

        A panel removed from the layout takes its callbacks' targets with it, and Dash then errors
        on every callback bound to it, which on this screen would mean the HPO search silently
        dying the first time anyone visited Approve mode.

        Tab and mode are answered by **one** callback, not two. They are not independent: the
        Edit/Approve switch means nothing on the Evidence location tab, and two callbacks each writing part of
        the visibility is how a panel ends up on screen in the wrong tab, whichever fired last
        wins, and neither knows what the other decided.
        """
        show, hide = {"display": "block"}, {"display": "none"}
        curating = tab not in ("overview", "anchor")
        # The row itself, not only its columns. It is a ``display: flex`` box a viewport tall, so
        # leaving it on with every column hidden reserves the whole screen above the Overview
        # table, the table then opens below the fold, looking like a panel that failed to render.
        return (
            hide if tab == "overview" else {"display": "flex"},
            hide if tab == "overview" else show,       # The report, shared by Curate and Evidence location
            show if curating else hide,                # labels, comments, Edit/Approve
            show if tab == "anchor" else hide,
            show if tab == "overview" else hide,
            show if curating and mode != "approve" else hide,
            show if curating and mode == "approve" else hide,
        )

    @app.callback(
        Output("doc-label-summary", "children"),
        Output("doc-label-status", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
    )
    def doc_labels(patient_id, _dirty, mode):
        """The summary chips, and the confirmation line cleared on a *patient* change only.

        Only these two, the controls themselves are static (see ``labelling.report_layout``).
        Clearing the status on every trigger would wipe the "Saved" line the save itself just
        wrote, since saving bumps ``store-dirty`` and so re-enters here. Never clearing it would
        show "Saved for SYN001" while SYN002 is on screen.
        """
        summary = common.guard("Report labels", render_doc_labels, patient_id, mode or "light")
        clear = dash.ctx.triggered_id == "store-patient"
        return summary, (None if clear else dash.no_update)

    @app.callback(
        Output("patient-list", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
        Input("patient-filter", "value"),
    )
    def patient_list(current, _dirty, mode, filters):
        return common.guard("Patient list", render_patient_list, current, mode or "light",
                            unlabelled_only(filters))

    @app.callback(
        Output("store-patient", "data", allow_duplicate=True),
        Output("store-segment", "data", allow_duplicate=True),
        Output("store-nav", "data"),
        Input({"type": "pat", "pid": ALL}, "n_clicks"),
        Input("pat-prev", "n_clicks"),
        Input("pat-next", "n_clicks"),
        State("store-patient", "data"),
        State("store-nav", "data"),
        State("patient-filter", "value"),
        prevent_initial_call=True,
    )
    def pick_patient(row_clicks, prev_clicks, next_clicks, current, seen, filters):
        """The one writer of the current patient. Changing patient always clears the segment.

        Carrying a segment index across patients would point at a different sentence in a different
        report, and the Edit form would happily attach a trigger to it.

        **Prev/Next step on the click count having gone up, not on Dash naming the trigger.** Those
        two buttons live in the static sidebar, so their ``n_clicks`` only ever climbs, which makes
        a spurious re-fire indistinguishable from a real press if you trust ``triggered_id`` alone.
        It used to be worse than indistinguishable: the old code read ``-1 if trigger == "pat-prev"
        else 1``, so *anything* unexpected advanced, and one press of Next moved two patients.
        Comparing against the counts we last acted on makes a re-fire a no-op by design.

        **Prev/Next step through the list as filtered, not through the cohort.** The list is the
        navigation surface. With the filter on, a Next that landed on a report not in it would be a
        patient the sidebar cannot show you have selected. See :func:`step_patient` for the one case
        that needs care, the report you just labelled, which leaves the list under you.
        """
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or triggered is None:
            raise PreventUpdate
        ids = registry.patient_ids
        if not ids:
            raise PreventUpdate

        seen = seen or {}
        prev_clicks, next_clicks = int(prev_clicks or 0), int(next_clicks or 0)
        counts = {"prev": prev_clicks, "next": next_clicks}

        if prev_clicks > int(seen.get("prev", 0)):
            step = -1
        elif next_clicks > int(seen.get("next", 0)):
            step = 1
        elif isinstance(triggered, dict) and any(row_clicks or []):
            # A row in the patient list. Its ``n_clicks`` is reset to 0 by every re-render, so a
            # non-zero one is a press that has not been redrawn away yet.
            return triggered["pid"], None, counts
        else:
            raise PreventUpdate

        visible = visible_patient_ids(registry, unlabelled_only(filters))
        return step_patient(current, visible, ids, step), None, counts

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Input("pat-confirm", "n_clicks"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def toggle_confirm(n_clicks, patient_id, dirty):
        registry = common.registry_or_none()
        if not n_clicks or registry is None or not patient_id:
            raise PreventUpdate
        action = "unconfirm_patient" if registry.is_confirmed(patient_id) else "confirm_patient"
        registry.record(action, patient_id)
        return (dirty or 0) + 1

    @app.callback(
        Output("pat-status", "children"),
        Output("pat-confirm", "children"),
        Output("save-status", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
    )
    def status(patient_id, _dirty):
        return common.guard("Status", render_status, patient_id)

    @app.callback(
        Output("src-status", "children"),
        Output("store-patient", "data", allow_duplicate=True),
        Output("store-dirty", "data", allow_duplicate=True),
        Input("src-load", "n_clicks"),
        *[State(f"src-{key}", "value") for key in SOURCE_KEYS],
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def load_source(n_clicks, *values):
        """Repoint the app without a restart.

        A bad path keeps the previous registry: losing a working session to a typo in a text box is
        a worse failure than the typo, and this app's session holds unfinished judgement calls.

        The paths arrive positionally in ``SOURCE_KEYS`` order, not as named arguments, so a
        new source cannot be added to the sidebar and silently left out of the dict built here.
        """
        if not n_clicks:
            raise PreventUpdate
        *path_values, dirty = values
        paths = dict(zip(SOURCE_KEYS, path_values))
        current = state.get_registry() if state.has_registry() else None
        try:
            candidate = Registry(paths, author=current.author if current else "unknown")
        except Exception as exc:  # noqa: BLE001
            return theme.note(f"Could not load: {exc}", "danger"), dash.no_update, dash.no_update

        problems = candidate.validate()
        if problems:
            return theme.note(" ".join(problems), "danger"), dash.no_update, dash.no_update

        state.set_registry(candidate)
        common.clear_failures()
        notes = candidate.notes()
        message = f"Loaded {len(candidate.patient_ids)} patients. " + " ".join(notes)
        first = candidate.patient_ids[0] if candidate.patient_ids else None
        return theme.note(message, "warn" if notes else "info"), first, (dirty or 0) + 1

    app.clientside_callback(
        """
        function(mode) {
            document.body.setAttribute('data-theme', mode || 'light');
            return mode;
        }
        """,
        Output("store-theme", "data"),
        Input("theme-toggle", "value"),
    )

    # Scrolling the report to the selected segment. Every path that selects one arrives here,
    # a Jump-to button on an Approve row, an evidence location candidate, a click on a mark, so the report
    # follows the work without each of those needing to say so.
    #
    # Dash renders a pattern id as ``JSON.stringify`` of the dict with its keys **sorted**, which
    # is what makes ``{"idx": n, "type": "seg"}`` the element's real DOM id. A miss is a no-op
    #, not an error: the segment may be past ``reader.PAGE_SIZE``, or the report may not
    # have been redrawn yet.
    app.clientside_callback(
        """
        function(idx) {
            if (idx === null || idx === undefined) {
                return window.dash_clientside.no_update;
            }
            const el = document.getElementById(JSON.stringify({idx: idx, type: 'seg'}));
            if (!el) { return window.dash_clientside.no_update; }
            el.scrollIntoView({behavior: 'smooth', block: 'center'});
            return idx;
        }
        """,
        Output("store-scrolled", "data"),
        Input("store-segment", "data"),
    )

    # Every size in ``assets/style.css`` is a ``rem``, so setting one custom property on the root
    # element scales the report, the row cards, the chips and the Overview grid together. The store
    # is written, not read: it exists so this callback has an Output, and so the current
    # size is visible to anything that later wants it.
    app.clientside_callback(
        """
        function(size) {
            const px = parseInt(size, 10);
            if (!px) { return window.dash_clientside.no_update; }
            document.documentElement.style.setProperty('--fs', px + 'px');
            return px;
        }
        """,
        Output("store-fontsize", "data"),
        Input("font-size", "value"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# sidebar rendering, pure
# ──────────────────────────────────────────────────────────────────────────────
def render_doc_labels(patient_id, mode: str = "light"):
    """The report-label panel of *patient_id*."""
    return labelling.report_summary(common.registry_or_none(), patient_id, mode)


def unlabelled_only(filters) -> bool:
    """Is the sidebar's filter ticked? A checklist arrives as a list, or as ``None`` before it has
    been rendered, both of which mean *unfiltered*."""
    return FILTER_UNLABELLED in (filters or [])


def is_labelled(row: dict) -> bool:
    """Has this report been characterised at all?

    Either half counts. A grade with no labels and labels with no grade are both a curator having
    made a pass over the report, and a filter that demanded both would keep handing back reports
    somebody had already read.

    Comments do not count. A note about a report is not a characterisation of it, and a report
    somebody asked a question about is the kind that still needs the vocabulary applied.
    """
    return bool(row.get("difficulty")) or bool(row.get("n_doc_labels"))


def visible_patient_ids(registry, only_unlabelled: bool) -> list[str]:
    """The patient ids the sidebar is showing, in cohort order.

    Off the ``overview`` rows, not a per-patient build: the filter has to be re-evaluated on
    every render and on every Prev/Next, and warming sixteen patients into the LRU to answer "which
    of these has a label" would evict the one being worked on.
    """
    if not only_unlabelled:
        return list(registry.patient_ids)
    return [row["patient_id"] for row in registry.overview() if not is_labelled(row)]


def step_patient(current, visible: list[str], order: list[str], step: int) -> str:
    """One Prev/Next press: the neighbouring *visible* patient, clamped at both ends.

    The case that needs care is ``current`` not being visible, which is not an edge case at all, it is what happens the moment you save labels for the report you are on with the filter ticked.
    Falling back to the whole cohort there would jump to a report the list does not show. Instead we
    walk the cohort from where ``current`` sits and take the first patient still in the list, so
    Next means "the next one still to do". With nothing left in that direction, we stay put, which
    is the same thing the clamp does at the end of the list.
    """
    if not visible:
        return current
    if current in visible:
        at = visible.index(current)
        return visible[max(0, min(len(visible) - 1, at + step))]
    try:
        at = order.index(current)
    except ValueError:
        return visible[0] if step > 0 else visible[-1]
    remaining = order[at + 1:] if step > 0 else order[:at][::-1]
    seen = set(visible)
    return next((pid for pid in remaining if pid in seen), current)


def render_patient_list(current, mode: str = "light", only_unlabelled: bool = False):
    """The report list, marking *current*. With *only_unlabelled* only reports without labels."""
    registry = common.registry_or_none()
    if registry is None:
        return theme.empty("No data loaded.")
    palette = theme.palette(mode)
    colors = theme.status_colors(mode)

    rows = registry.overview()
    shown = [row for row in rows if not (only_unlabelled and is_labelled(row))]
    if only_unlabelled and not shown:
        return theme.empty(f"Every one of the {len(rows)} reports has been characterised.")

    items = []
    # Said out loud, because a short list and a broken filter look identical, and because the
    # report you just labelled leaves this list, which is easier to read as a bug than as progress.
    if only_unlabelled:
        items.append(html.Div(f"{len(shown)} of {len(rows)} reports still unlabelled",
                              className="stat-sub pat-filter-count"))
    for row in shown:
        pid = row["patient_id"]
        flags = []
        if row["n_suggested"]:
            flags.append(common.chip(str(row["n_suggested"]), colors["suggested"],
                                     title="open suggestions"))
        if row["n_disagree"]:
            flags.append(common.chip(f"≠{row['n_disagree']}", palette["serious"],
                                     title="terms not every loaded annotation file carries"))
        if row["difficulty"]:
            flags.append(common.chip(row["difficulty"][:1].upper(),
                                     labelling.difficulty_color(row["difficulty"], palette),
                                     title=f"report graded {vocab.display(row['difficulty'])}"))
        if row["confirmed"]:
            flags.append(common.chip("✓", colors["approved"], title="marked confirmed"))
        items.append(html.Div(
            [html.Span(pid, className="pat-id"), html.Span(flags, className="pat-flags")],
            id={"type": "pat", "pid": pid}, n_clicks=0,
            className="pat" + (" pat-current" if pid == current else ""),
        ))
    return items


def render_status(patient_id):
    """``(status, confirm button label, autosave note)`` for *patient_id*."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return theme.note(common.NO_PATIENT, "info"), "Mark patient confirmed", None

    confirmed = registry.is_confirmed(patient_id)
    ids = registry.patient_ids
    at = ids.index(patient_id) + 1 if patient_id in ids else 0
    rows = registry.rows_for(patient_id)
    open_count = sum(1 for r in rows if r.get("status") == "suggested")

    status = html.Div([
        html.Div(f"{patient_id}, {at} of {len(ids)}", className="stat-sub"),
        html.Div(f"{open_count} open · {len(rows) - open_count} decided", className="stat-sub"),
    ])
    totals = registry.totals()
    saved = html.Div([
        html.Div(f"autosaved · {totals['n_events']} events", className="stat-sub"),
        html.Div(os.path.basename(registry.log.current_path), className="stat-sub"),
    ])
    return status, ("Unmark confirmed" if confirmed else "Mark patient confirmed"), saved


# ──────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────
def build_app(registry: Registry) -> dash.Dash:
    """Create the Dash app over *registry* and register every callback."""
    state.set_registry(registry)
    app = dash.Dash(
        __name__,
        title="HCY curation",
        assets_folder=os.path.join(_HERE, "assets"),
        # The panels are rendered into divs, so a callback's target may not exist at import time.
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(registry)
    register_callbacks(app)
    return app


def _port_is_free(host: str, port: int) -> bool:
    """Can we bind ``port``? Several curators share one login node, so this is a real question.

    Dash's own failure here is a bare ``OSError`` traceback out of werkzeug, which reads as "the
    tool is broken", not "pick another port", and the curators this is written for are not
    the people who should have to tell those apart.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _first_free_port(host: str, start: int, span: int = 50) -> int:
    """The first bindable port at or above ``start``, a suggestion, not a binding."""
    for port in range(start + 1, start + span):
        if _port_is_free(host, port):
            return port
    return start + span


def parse_args(argv=None):
    """Parse the command-line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hcy-dir", default=sources.DEFAULT_HCY_DIR,
                        help="directory holding the segments, the annotation files, and curation/")
    parser.add_argument("--phenobert-dir", default=sources.DEFAULT_PB_DIR,
                        help="PhenoBERT baseline run directory (baseline_phenobert); blank to run without the PhenoBERT column")
    parser.add_argument("--curation-dir", default=None,
                        help="override where the log is written (default <hcy-dir>/curation)")
    parser.add_argument("--author", default=None,
                        help="recorded on every event (default: the OS user)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8055)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="render every panel headlessly and exit non-zero on the first problem")
    parser.add_argument("--check", action="store_true",
                        help="check the paths and that this account can write the curation log, "
                             "then exit, no server, no tunnel")
    return parser.parse_args(argv)


def check(args) -> int:
    """Report whether this account could curate here, and exit. No server.

    Prints what it found, not only what is wrong: "everything is fine" has to be
    distinguishable from "the check did not run", or a silent success reads as a broken command.
    """
    paths = sources.default_paths(args.hcy_dir, args.phenobert_dir)
    try:
        author = args.author or getpass.getuser()
    except Exception:  # noqa: BLE001
        author = "unknown"

    print(f"Curating as   {author}")
    print(f"hcy directory {args.hcy_dir}")
    print(nltk_data.describe())

    try:
        registry = Registry(paths, author=author, curation_dir=args.curation_dir)
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it, not to raise it
        print(f"\nFAILED to load: {exc}")
        return 2

    print(f"log directory {registry.log.dir}")
    problems, notes = registry.validate(), registry.notes()
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print()
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        print("\nNot ready. Send the lines above to whoever owns the directory.")
        return 2

    print(f"\nOK — {len(registry.patient_ids)} patients, and this account can write the log.")
    return 0


def main(argv=None) -> int:
    """Load the curation sources, then serve the app. Returns the exit status."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.selftest:
        from apps.curation_ui import selftest

        return selftest.run(hcy_dir=args.hcy_dir, phenobert_dir=args.phenobert_dir)

    if args.check:
        # *not* part of --selftest. That one redirects the log into a temp directory
        # so a test run can never write into the cohort, which is right, and which means it says
        # nothing at all about whether this account can write the real one. That is the question a
        # new curator actually has, it is the one failure that costs work, not time, and it
        # needs no browser and no tunnel to answer.
        return check(args)

    try:
        author = args.author or getpass.getuser()
    except Exception:  # noqa: BLE001 - getuser raises on a container with no passwd entry
        author = "unknown"

    paths = sources.default_paths(args.hcy_dir, args.phenobert_dir)
    registry = Registry(paths, author=author, curation_dir=args.curation_dir)
    problems = registry.validate()
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        logger.error("Refusing to start. Fix the paths, or pass --hcy-dir.")
        return 2
    for note in registry.notes():
        logger.warning("%s", note)
    # Where the ontology's corpora came from, the repo's own copy, or this machine's. Worth one
    # line in the log: it is the difference between a checkout that runs anywhere and one that runs
    # only for the person whose home directory the corpora were downloaded into.
    logger.info("%s", nltk_data.describe())

    if not _port_is_free(args.host, args.port):
        free = _first_free_port(args.host, args.port)
        logger.error(
            "Port %d on %s is already taken, most likely another curator's session on this "
            "login node. Re-run with --port %d, and tunnel that port instead: "
            "ssh -L %d:localhost:%d <cluster>",
            args.port, args.host, free, free, free,
        )
        return 3

    app = build_app(registry)
    logger.info("Curating as %r, log at %s", author, registry.log.path)
    logger.info("http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
