"""The report comment thread, free text about the document as a whole.

It lived in Edit mode until now, as the last card of :mod:`apps.curation_ui.views.edit`, which
meant it vanished the moment anybody switched to Approve. That is backwards: the reader who
notices "this whole report is family history, and I am not sure any of these belong to the patient"
is usually the one *adjudicating*, and making them switch modes to write it down is how it stops
being written down.

So it moves here, and is mounted **outside the mode switch**, directly under the report-label panel, one instance for the life of the page, in Edit and Approve alike, for the same reason and by the
same mechanism as those labels. Two copies would be two sets of component ids for one piece of
state, and the copy not on screen when Add was pressed would quietly win.

A comment has an author and a time and there can be several, so they form a **thread** rather than
one overwritten field: a second comment is a second thing somebody said, not a correction of the
first. They live in ``curation_comments.csv``, not on the report-label row, which is why that row
records ``n_comments``, not quoting the latest one.

The Add button lives inside a static card, but its callback still checks ``n_clicks``, this app has
shipped that bug twice, and a callback that writes to the log is not the place to assume a component
will stay static.
"""

from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import theme
from . import common

VIEW_ID = "comments"


def layout() -> html.Div:
    """The Dash layout of this view."""
    return theme.card(
        "Report comment",
        html.Div([
            dcc.Textarea(
                id="doc-comment", value="", className="comment-box",
                placeholder="anything about this report as a whole, what confused you, "
                            "what a later reader should know",
            ),
            html.Button("Add comment", id="doc-comment-add", className="primary-btn",
                        n_clicks=0, style={"marginTop": "6px"}),
            html.Div(id="doc-comment-status"),
            html.Div(id="doc-comment-list", style={"marginTop": "8px"}),
        ]),
        subtitle="About the document, not about one term. Appended with your name and the time, "
                 "so a later reader knows who said it and when. Available in both modes.",
    )


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Output("doc-comment", "value"),
        Output("doc-comment-status", "children"),
        Input("doc-comment-add", "n_clicks"),
        State("doc-comment", "value"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def add_comment(n_clicks, text, patient_id, dirty):
        registry = common.registry_or_none()
        if not n_clicks or registry is None or not patient_id:
            raise PreventUpdate
        text = (text or "").strip()
        if not text:
            return dash.no_update, dash.no_update, theme.note(
                "Nothing to save, write something first.", "warn")
        registry.record("comment", patient_id, text=text)
        return (dirty or 0) + 1, "", theme.note(f"Comment added to {patient_id}.", "info")

    @app.callback(
        Output("doc-comment-list", "children"),
        Input("store-patient", "data"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
    )
    def draw_comments(patient_id, _dirty, mode):
        return common.guard("Comments", render_comments, patient_id, mode or "light")

    @app.callback(
        Output("store-dirty", "data", allow_duplicate=True),
        Input({"type": "comment-del", "cid": ALL}, "n_clicks"),
        State("store-patient", "data"),
        State("store-dirty", "data"),
        prevent_initial_call=True,
    )
    def delete_comment(_clicks, patient_id, dirty):
        registry = common.registry_or_none()
        triggered = dash.ctx.triggered_id
        if registry is None or not triggered or not any(_clicks or []):
            raise PreventUpdate
        registry.record("delete_comment", patient_id, target_key=triggered["cid"])
        return (dirty or 0) + 1


# ──────────────────────────────────────────────────────────────────────────────
# rendering, pure
# ──────────────────────────────────────────────────────────────────────────────
def render_comments(patient_id, mode: str = "light"):
    """The report's comment thread, oldest first, each with who said it and when."""
    registry = common.registry_or_none()
    if registry is None or not patient_id:
        return None
    comments = registry.comments_for(patient_id)
    if not comments:
        return theme.empty("No comments on this report yet.")

    palette = theme.palette(mode)
    return html.Div([
        html.Div([
            html.Div(comment["text"], className="comment-text"),
            html.Div([
                html.Span(f"{comment.get('author', '')} · {comment.get('ts', '')}",
                          className="stat-sub"),
                html.Button("Delete", id={"type": "comment-del", "cid": comment["comment_id"]},
                            className="copy-btn", n_clicks=0),
            ], className="row-actions"),
        ], className="row-card", style={"borderLeft": f"3px solid {palette['neutral']}"})
        for comment in comments
    ])
