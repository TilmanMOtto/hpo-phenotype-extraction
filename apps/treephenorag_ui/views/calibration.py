"""Do the two scores mean anything? One diagram per score, each against its own label.

The earlier runs'tree runs threshold two scores, and the whole design rests on them being separately
meaningful. ``accept_score`` predicts node presence; ``prune_score`` predicts subtree presence.
Judging either against the other's label produces a confident, wrong answer, see
:mod:`apps.treephenorag_ui.calib`, so this view always names which label it used, on the panel, next to
the number.

The accept-threshold curve is exact and the prune one is absent. τ_accept never
influenced which nodes were scored, so sweeping it over the scored nodes reproduces what a rerun
would report. τ_prune decided which nodes exist at all, and the run holds no score for the ones it
never visited. That frontier is read from the configurations on disk in the Pruning autopsy tab
instead of being invented here.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .. import active, copyexport, state, theme


def layout():
    """The Dash layout of this view."""
    # The reliability/AURC block is the most expensive thing this app computes and is deferred
    # until this tab is opened, so the first render of a cell takes seconds. Without a spinner
    # that reads as a hang.
    return dcc.Loading(html.Div(id="cal-content"), type="dot", delay_show=200)


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _reliability_figure(summary: dict, mode: str) -> go.Figure:
    palette = theme.palette(mode)
    # ``bin_summary`` names them confidence / accuracy / count.
    bins = summary.get("bins") or []
    x = [b.get("confidence") for b in bins]
    y = [b.get("accuracy") for b in bins]
    n = [b.get("count", 0) for b in bins]

    figure = go.Figure()
    figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                       line={"color": palette["text_muted"], "width": 1, "dash": "dash"},
                       hoverinfo="skip")
    figure.add_scatter(
        x=x, y=y, mode="lines+markers", name="observed",
        line={"color": palette["categorical"][0], "width": 2},
        marker={"size": [max(6, min(26, (c or 0) ** 0.5)) for c in n]},
        customdata=n,
        hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<br>%{customdata} nodes<extra></extra>",
    )
    figure.update_layout(**theme.plotly_layout(
        mode, height=330,
        xaxis={"title": "predicted probability", "range": [0, 1]},
        yaxis={"title": "observed frequency", "range": [0, 1]},
    ))
    return figure


def _separation_figure(separation: dict, mode: str, label_column: str) -> go.Figure:
    palette = theme.palette(mode)
    edges = separation.get("edges") or []
    if len(edges) < 2:
        return go.Figure(layout=theme.plotly_layout(mode, height=300))
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    figure = go.Figure()
    figure.add_bar(x=centres, y=separation.get("negative") or [], name=f"{label_column} = false",
                   marker_color=palette["neutral"], opacity=0.85,
                   hovertemplate="score≈%{x:.2f}<br>%{y} nodes<extra></extra>")
    figure.add_bar(x=centres, y=separation.get("positive") or [], name=f"{label_column} = true",
                   marker_color=palette["good"], opacity=0.85,
                   hovertemplate="score≈%{x:.2f}<br>%{y} nodes<extra></extra>")
    figure.update_layout(**theme.plotly_layout(
        mode, height=300, barmode="overlay",
        xaxis={"title": "score", "range": [0, 1]}, yaxis={"title": "nodes", "type": "log"}))
    return figure


def _accept_curve_figure(rows: list[dict], mode: str) -> go.Figure:
    palette = theme.palette(mode)
    tau = [r["tau"] for r in rows]
    figure = go.Figure()
    figure.add_scatter(x=tau, y=[r.get("precision") for r in rows], name="precision",
                       mode="lines", line={"color": palette["categorical"][0], "width": 2},
                       hovertemplate="τ_accept %{x:.2f}<br>precision %{y:.3f}<extra></extra>")
    figure.add_scatter(x=tau, y=[r.get("recall") for r in rows], name="recall",
                       mode="lines", line={"color": palette["good"], "width": 2},
                       hovertemplate="τ_accept %{x:.2f}<br>recall %{y:.3f}<extra></extra>")
    figure.update_layout(**theme.plotly_layout(
        mode, height=320, xaxis={"title": "τ_accept", "range": [0, 1]},
        yaxis={"title": "share of scored nodes", "range": [0, 1]}))
    return figure


def _score_panels(report: dict, mode: str) -> list:
    panels = []
    calibration = report.get("calibration") or {}
    separation = report.get("separation") or {}
    for name in ("accept", "prune"):
        summary = calibration.get(name)
        if not summary:
            panels.append(theme.panel(
                f"{name}_score", theme.empty(
                    "Not measurable on this cell, the scored nodes are all one class, so "
                    "calibration, Brier and AUROC are all undefined."),
                panel_id=f"cal-{name}"))
            continue
        panels.append(theme.panel(
            f"{name}_score, predicting {summary.get('meaning')}",
            html.Div([
                theme.stat_row([
                    theme.stat("nodes", f"{summary.get('n'):,}",
                               f"prevalence {summary.get('prevalence', 0) * 100:.2f}%"),
                    theme.stat("ECE", _fmt(summary.get("ece_equal_mass")), "equal-mass, 15 bins",
                               tone="critical" if (summary.get("ece_equal_mass") or 0) > 0.15
                               else None),
                    theme.stat("Brier", _fmt(summary.get("brier"))),
                    theme.stat("AUROC", _fmt(summary.get("auroc") or summary.get("auc")),
                               "discrimination",
                               tone="critical" if (summary.get("auroc")
                                                   or summary.get("auc") or 1) < 0.6 else None),
                    theme.stat("AURC", _fmt(summary.get("aurc")), "risk--coverage"),
                    theme.stat("mean score", f"{summary.get('mean_confidence_positive', 0):.3f} "
                                             f"/ {summary.get('mean_confidence_negative', 0):.3f}",
                               "positive / negative",
                               help_text="If these two are close, the score cannot separate the "
                                         "classes no matter where the threshold goes."),
                ]),
                html.Div([
                    html.Div(dcc.Graph(id=f"cal-rel-{name}",
                                       figure=_reliability_figure(summary, mode),
                                       config=theme.GRAPH_CONFIG)),
                    html.Div(dcc.Graph(
                        id=f"cal-sep-{name}",
                        figure=_separation_figure(separation.get(name) or {}, mode,
                                                  summary.get("label_column", "label")),
                        config=theme.GRAPH_CONFIG)),
                ], className="split"),
                theme.note([
                    "Scored against ", html.Code(summary.get("label_column")),
                    f", {summary.get('meaning')}. ",
                    "Nodes the traversal never reached are excluded: they carry no score, and "
                    "imputing zero for them would enter predictions the model never made.",
                ], tone="info"),
            ]),
            panel_id=f"cal-{name}", graph_id=f"cal-rel-{name}",
            markdown=copyexport.calibration_md(report),
        ))
    return panels


def _build(report: dict, mode: str, restriction=None):
    if not report:
        return theme.empty("Select a run.")
    if not report.get("is_tree"):
        return theme.note(
            f"{report.get('method_label')} emits no per-node score, so there is nothing to "
            "calibrate. The calibration view applies to an earlier exploratory run, an earlier exploratory run, an earlier exploratory run and "
            "exp13_09.",
            tone="info")

    children = _score_panels(report, mode)
    if restriction is not None:
        children.insert(0, restriction)
    curve = report.get("accept_curve") or []
    if curve:
        children.append(theme.panel(
            "τ_accept sweep",
            html.Div([
                dcc.Graph(id="cal-accept-curve", figure=_accept_curve_figure(curve, mode),
                          config=theme.GRAPH_CONFIG),
                theme.note(
                    "Exact. τ_accept never influenced which nodes were scored, so sweeping it "
                    "over the scored nodes reproduces what a rerun at that threshold "
                    "would report. There is no τ_prune slider here, that threshold "
                    "decided which nodes exist, and the run holds no score for the ones it never "
                    "visited. Its frontier is in the Pruning autopsy tab, read from the "
                    "directories the sweep actually wrote.", tone="info"),
            ]),
            panel_id="cal-accept", graph_id="cal-accept-curve",
            subtitle="Precision and recall over the scored nodes, against the accept threshold.",
        ))
    return html.Div(children)


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("cal-content", "children"),
        Input("run-a", "value"), Input("op-a", "value"), Input("store-theme", "data"),
        Input("store-subset", "data"), active.TAB_INPUT,
    )
    def _render(cell_id, operating_point, mode, subset, tab):
        active.guard(tab, "calibration")
        if not cell_id:
            return theme.empty("Select a run in the sidebar.")
        registry = state.get_registry()
        keep = registry.aggregate_filter(cell_id, subset)
        return _build(registry.get_report(cell_id, operating_point, heavy=True,
                                          report_filter=keep),
                      mode or "light", registry.subset_note(cell_id, subset))
