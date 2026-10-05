"""What was built, from what, and what each column can and cannot show.

A third tab rather than a footnote, for a reason that came out of the 2026-09-16 advisor meeting:
*"someone who takes this table and wants to reproduce it needs to know what the parameters were."*
The same applies with more force to a screen that shows five methods' reasoning side by side, where
two of those columns are thinner than the other three and the difference is a property of the
artifacts, not of the methods.

So this tab prints three things and hides none of them:

- the **frame**: which twenty documents, drawn when, from which ground truth, under which parameters;
- the **inputs**: every artifact the bundle was built from, with a digest, and which of them have
  moved since;
- the **columns**: for each method, what its panel shows and the note that travels with it --
  AutoPCR's and RAG-HPO's model answers were never written, and TreePhenoRAG's pooled prediction
  set has no writer in tracked code.
"""

from __future__ import annotations

from dash import dcc, html

from apps.compare_ui import theme
from apps.compare_ui.views import common

VIEW_ID = "provenance"


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="provenance-body"),
        dcc.Store(id="sig-provenance", data=None),
    ])


def render(registry, report_id):
    """The inputs each column of *report_id* was built from."""
    return html.Div([
        common.guard(VIEW_ID, "frame", _frame, registry),
        common.guard(VIEW_ID, "columns", _columns, registry),
        common.guard(VIEW_ID, "inputs", _inputs, registry, report_id),
    ])


def _frame(registry):
    index = registry.index
    frame = index.get("frame") or {}
    rows = [
        ["dataset", "{} ({})".format(registry.label(), registry.cohort.key)],
        ["scored as", registry.scored_cohort()],
        ["bundles built", index.get("built_at") or "—"],
        ["ground truth", index.get("gold_dataset") or "—"],
        ["frame directory", frame.get("dir") or ",  none; documents were passed explicitly"],
        ["frame drawn", frame.get("generated") or "—"],
    ]
    for key, value in sorted((frame.get("params") or {}).items()):
        rows.append(["frame parameter · " + key, str(value)])
    for key, value in sorted((frame.get("inputs") or {}).items()):
        rows.append(["frame input digest · " + key, value])

    body = [common.table(["", ""], [[html.Span(k, className="cmp-key"), v] for k, v in rows])]
    if registry.gold_note():
        body.append(theme.note(registry.gold_note(), "info"))
    if frame:
        body.append(theme.note(
            "The draw is purposive and fixed in advance: cells, seed and input digests were written "
            "before anything was read. If the frame's ground truth digest differs from the ground truth these "
            "bundles were scored against, the draw was made on an earlier version of the ground truth, "
            "which is legitimate, and worth saying out loud in any write-up.", "info"))
    return theme.panel("The sample", html.Div(body), panel_id="prov-frame",
                       subtitle="which reports are on screen, and why",
                       markdown=common.md_table(["field", "value"], rows))


def _columns(registry):
    rows = []
    for method in registry.methods():
        rows.append([
            html.Span(method.get("label") or method["key"]),
            html.Span(method.get("family") or "", className="cmp-flag",
                      **{"data-tone": theme.FAMILY_TONE.get(method.get("family"), "neutral")},
                      title=theme.FAMILY_HELP.get(method.get("family"), "")),
            method.get("evidence") or "—",
            method.get("evidence_note") or "—",
        ])
    return theme.panel(
        "What each column can show", html.Div([
            common.table(["method", "", "shows", "caveat"], rows),
            theme.note("Two of these columns are thinner than the others and it is not because "
                       "those methods reason less. AutoPCR's linker and RAG-HPO's verifier were "
                       "both asked and both answered; neither driver wrote the answer down. What "
                       "is shown for them is the menu and the decision, never a reconstructed "
                       "rationale presented as the model's own.", "warn"),
        ]),
        panel_id="prov-columns", subtitle="and the note that travels with each",
        markdown=common.md_table(["method", "family", "shows", "caveat"],
                                 [[r[0], "", r[2], r[3]] for r in rows]))


def _inputs(registry, report_id):
    bundle = registry.bundle(report_id) if report_id else None
    inputs = (bundle or {}).get("inputs") or registry.index.get("inputs") or {}
    if not inputs:
        return theme.panel("Inputs", theme.empty("This bundle records no inputs."),
                           panel_id="prov-inputs")

    drift = set(registry.drift(report_id) if report_id and bundle else ())
    rows = []
    for name, record in sorted(inputs.items()):
        rows.append([
            html.Span(name, className="cmp-key"),
            html.Code((record or {}).get("path") or "—"),
            (record or {}).get("sha256") or "—",
            "{:,}".format((record or {}).get("size") or 0),
            html.Span("moved", className="cmp-flag", **{"data-tone": "critical"})
            if name in drift else "",
        ])
    body = [common.table(["input", "path", "sha256", "bytes", ""], rows)]
    if drift:
        body.append(theme.note(
            "The marked artifacts differ from what this bundle was built from. Rebuild before "
            "quoting anything on the reader tab.", "danger"))
    else:
        body.append(theme.note(
            "Paths are relative to the output base. An input that is not reachable from this "
            "machine is not checked and not reported as moved, absence of evidence is not "
            "evidence of change.", "info"))

    return theme.panel(
        "Inputs for {}".format(report_id or "—"), html.Div(body), panel_id="prov-inputs",
        subtitle="every artifact this bundle was built from",
        markdown=common.md_table(
            ["input", "path", "sha256"],
            [[name, (rec or {}).get("path", ""), (rec or {}).get("sha256", "")]
             for name, rec in sorted(inputs.items())]))


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    from dash import Input, Output, State
    from dash.exceptions import PreventUpdate

    from apps.compare_ui import state

    @app.callback(
        Output("provenance-body", "children"),
        Output("sig-provenance", "data"),
        Input("tabs", "value"),
        Input("cohort-picker", "value"),
        Input("report-picker", "value"),
        Input("store-theme", "data"),
        State("sig-provenance", "data"),
    )
    def render_view(tab, cohort, report_id, mode, previous):
        key = common.render_key(VIEW_ID, cohort, report_id, mode)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        return render(state.registry(cohort), report_id), key
