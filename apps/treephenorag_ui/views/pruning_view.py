"""The pruning autopsy, which decision severed which annotated term, and at what score.

This is the view the whole app exists for. Four questions, in the order a diagnosis asks them:

1. **How deep was the damage?** A blocking-depth histogram. Concentration at depth 1 means the
   gate is failing at the organ-system level, where a broad term's surface expression in a
   narrative is weakest. A flat distribution means diffuse error no single threshold will fix.
2. **Which nodes did it?** The culprit leaderboard, with each node's own ``prune_score`` and the
   highest swept τ_prune that would have opened it. That last column is the difference between
   "lower the threshold" and "the score is wrong".
3. **What would a different τ have cost?** The frontier, read from the configurations the sweep
   materialised on disk. No re-run, no imputation, every point is a run that happened.
4. **Where does the cost go?** The per-depth traversal table: frontier size, survival rate,
   cumulative SLM calls, cumulative reachability recall.

Panels 1, 2 and 4 describe the selected configuration. Panel 3 describes the whole sweep and is
loaded separately, because it means reading every ``tau_*`` directory of the cell.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .. import active, copyexport, state, theme


def layout():
    """The Dash layout of this view."""
    return html.Div([
        dcc.Loading(html.Div(id="pr-content"), type="dot", delay_show=200),
        dcc.Loading(html.Div(id="pr-frontier"), type="dot"),
    ])


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def _blocking_figure(report: dict, mode: str) -> go.Figure:
    blocking = report.get("blocking") or {}
    histogram = blocking.get("histogram") or {}
    palette = theme.palette(mode)
    depths = sorted(int(k) for k in histogram)
    values = [histogram[str(d)] if str(d) in histogram else histogram.get(d, 0) for d in depths]

    figure = go.Figure()
    figure.add_bar(x=depths, y=values, marker_color=palette["critical"],
                   text=values, textposition="outside",
                   hovertemplate="depth %{x}<br>%{y} annotated terms blocked<extra></extra>")
    figure.update_layout(**theme.plotly_layout(
        mode, height=320,
        xaxis={"title": "ontology depth of the severing node", "dtick": 1},
        yaxis={"title": "annotated terms blocked"},
    ))
    return figure


def _frontier_figure(rows: list[dict], mode: str) -> go.Figure:
    """Recall, the pruning ceiling and cost against τ, the trade the sweep actually made."""
    palette = theme.palette(mode)
    labels = [r["operating_point"] for r in rows]
    x = [r["op_value"] if r["op_value"] is not None else i for i, r in enumerate(rows)]
    # A 2-D sweep's directory names are ``tau_0.00015_acc_0.9``. Five of those as tick labels
    # collide into a smear. The slice is one τ_accept, so the tick is the τ_prune it varies,
    # The full name stays in the hover, where there is room for it.
    ticks = [f"{r['op_value']:g}" if r.get("op_accept") is not None else label
             for r, label in zip(rows, labels)]

    figure = go.Figure()
    figure.add_scatter(x=x, y=[r.get("recall_ceiling") for r in rows], name="recall ceiling",
                       mode="lines+markers", line={"color": palette["critical"], "width": 2},
                       customdata=labels,
                       hovertemplate="%{customdata}<br>ceiling %{y:.3f}<extra></extra>")
    figure.add_scatter(x=x, y=[r.get("micro_recall") for r in rows], name="micro recall",
                       mode="lines+markers", line={"color": palette["good"], "width": 2},
                       customdata=labels,
                       hovertemplate="%{customdata}<br>recall %{y:.3f}<extra></extra>")
    figure.add_scatter(x=x, y=[r.get("micro_precision") for r in rows], name="micro precision",
                       mode="lines+markers",
                       line={"color": palette["categorical"][0], "width": 2, "dash": "dot"},
                       customdata=labels,
                       hovertemplate="%{customdata}<br>precision %{y:.3f}<extra></extra>")
    calls = [r.get("mean_slm_calls") for r in rows]
    if any(c is not None for c in calls):
        figure.add_scatter(x=x, y=calls, name="SLM calls / report", yaxis="y2",
                           mode="lines+markers",
                           line={"color": palette["text_muted"], "width": 1, "dash": "dash"},
                           customdata=labels,
                           hovertemplate="%{customdata}<br>%{y:.0f} calls<extra></extra>")
    layout_kwargs = theme.plotly_layout(
        mode, height=380,
        xaxis={"title": "τ_prune", "tickmode": "array", "tickvals": x, "ticktext": ticks},
        yaxis={"title": "share", "range": [0, 1]},
    )
    layout_kwargs["yaxis2"] = {"title": "SLM calls / report", "overlaying": "y", "side": "right",
                              "showgrid": False,
                              "tickfont": {"color": theme.palette(mode)["text_muted"]}}
    figure.update_layout(**layout_kwargs)
    return figure


def _depth_cost_figure(report: dict, mode: str) -> go.Figure:
    rows = (report.get("traversal") or {}).get("depth_cost") or []
    if not rows:
        return go.Figure(layout=theme.plotly_layout(mode, height=300))
    palette = theme.palette(mode)
    depths = [r.get("depth") for r in rows]

    figure = go.Figure()
    figure.add_bar(x=depths, y=[r.get("n_visited") or r.get("frontier_size") for r in rows],
                   name="nodes scored", marker_color=palette["sequential"][2],
                   hovertemplate="depth %{x}<br>%{y} nodes scored<extra></extra>")
    figure.add_bar(x=depths, y=[r.get("n_expanded") or r.get("n_survivors") for r in rows],
                   name="expanded", marker_color=palette["sequential"][5],
                   hovertemplate="depth %{x}<br>%{y} expanded<extra></extra>")
    recall_key = next((k for k in ("cumulative_reachability_recall", "reachability_recall")
                       if rows and k in rows[0]), None)
    if recall_key:
        figure.add_scatter(x=depths, y=[r.get(recall_key) for r in rows],
                           name="cumulative reachability recall", yaxis="y2", mode="lines+markers",
                           line={"color": palette["good"], "width": 2},
                           hovertemplate="depth %{x}<br>recall %{y:.3f}<extra></extra>")
    layout_kwargs = theme.plotly_layout(
        mode, height=340, barmode="group",
        xaxis={"title": "ontology depth", "dtick": 1}, yaxis={"title": "nodes"})
    layout_kwargs["yaxis2"] = {"title": "reachability recall", "overlaying": "y", "side": "right",
                               "range": [0, 1], "showgrid": False}
    figure.update_layout(**layout_kwargs)
    return figure


def _culprit_table(report: dict):
    culprits = report.get("culprits") or []
    if not culprits:
        return theme.empty(
            "No annotated term was blocked at this configuration, nothing pruning did cost recall "
            "here. Every miss is an identification failure.")
    rows = []
    for c in culprits:
        tau = c.get("tau_would_expand_all")
        rows.append([
            html.Div([html.Code(c["hpo_id"]), html.Br(),
                      html.Span(c.get("hpo_label") or "", className="mono")]),
            c.get("depth"),
            c["n_gold_lost"],
            c["n_reports"],
            _fmt(c.get("prune_score_min")),
            _fmt(c.get("prune_score_max")),
            html.Span(f"τ ≤ {tau:g}" if tau is not None else "no swept τ opens it",
                      className="pill" if tau is not None else None),
            html.Span(", ".join(g["hpo_label"] for g in c["gold_lost"][:4])
                      + ("…" if len(c["gold_lost"]) > 4 else ""), className="mono"),
        ])
    return theme.table(
        ["node that refused to expand", "depth", "annotated terms lost", "reports",
         "prune score (min)", "(max)", "would open at", "what it cost"],
        rows, align={1: "num", 2: "num", 3: "num", 4: "num", 5: "num"})


def _fp_factory_table(report: dict):
    factory = report.get("fp_factory") or []
    if not factory:
        return theme.empty("No false positives to attribute at this configuration.")
    rows = [[
        html.Div([html.Code(f["hpo_id"]), html.Br(),
                  html.Span(f.get("hpo_label") or "", className="mono")]),
        f["n_fp"], f["n_reports"], _fmt(f.get("prune_score_mean")),
        html.Span(", ".join(e["hpo_label"] for e in f["examples"][:4])
                  + ("…" if len(f["examples"]) > 4 else ""), className="mono"),
    ] for f in factory]
    return theme.table(
        ["expanded node", "FPs directly below it", "reports", "mean prune score", "examples"],
        rows, align={1: "num", 2: "num", 3: "num"})


def _build(report: dict, mode: str, restriction=None):
    if not report:
        return theme.empty("Select a run.")
    if not report.get("is_tree"):
        return theme.note(
            f"{report.get('method_label')} performs no ontology traversal, so it makes no pruning "
            "decisions and none can be audited. Use it as a reference column in Compare A/B "
            "instead. The pruning views apply to an earlier exploratory run, an earlier exploratory run, an earlier exploratory run and an earlier exploratory run.",
            tone="info")

    blocking = report.get("blocking") or {}
    ledger = report.get("ledger") or {}
    n_blocked = blocking.get("n_blocked", 0)

    return html.Div([
        *([restriction] if restriction is not None else []),
        theme.stat_row([
            theme.stat("annotated terms blocked", str(n_blocked),
                       _pct(ledger.get("recall_lost_to_pruning")) + " of all ground truth",
                       tone="critical" if n_blocked else "good",
                       help_text="Never reached: every root-to-term path was severed."),
            theme.stat("median blocking depth", _fmt(blocking.get("median"), 1),
                       f"mean {_fmt(blocking.get('mean'), 2)}",
                       help_text="Depth of the severing node on the path that got furthest."),
            theme.stat("cut at depth 1", _pct(blocking.get("fraction_at_depth_1")),
                       "organ-system level",
                       tone="critical" if (blocking.get("fraction_at_depth_1") or 0) > 0.5
                       else None,
                       help_text="A high share here means the gate is failing on the broadest "
                                 "terms, where narrative evidence is weakest."),
            theme.stat("distinct culprits", str(len(report.get("culprits") or [])),
                       "nodes that refused to expand",
                       help_text="How concentrated the damage is. A handful of nodes causing "
                                 "most of it is a fixable problem."),
        ]),
        theme.panel(
            "Blocking depth", dcc.Graph(id="pr-depth",
                                        figure=_blocking_figure(report, mode),
                                        config=theme.GRAPH_CONFIG),
            panel_id="pr-depth", graph_id="pr-depth",
            subtitle="At what ontology depth each missed annotated term's last surviving path was cut.",
            markdown=copyexport.pruning_md(report),
        ),
        theme.panel(
            "Culprits, the decisions that cost recall", _culprit_table(report),
            panel_id="pr-culprits",
            subtitle="Ranked by annotated terms lost. `would open at` is the highest τ_prune in this "
                     "run's sweep that sits at or below the node's own prune score in every "
                     "report where it blocked something.",
            markdown=copyexport.pruning_md(report),
        ),
        theme.panel(
            "The mirror image, expansions that cost precision", _fp_factory_table(report),
            panel_id="pr-factory",
            subtitle="Every false positive is directly below a node the gate chose to expand. "
                     "Read this beside the culprit table before concluding that τ_prune should "
                     "simply be lower.",
            markdown=copyexport.fp_md(report),
        ),
        theme.panel(
            "Traversal cost by depth",
            dcc.Graph(id="pr-cost", figure=_depth_cost_figure(report, mode),
                      config=theme.GRAPH_CONFIG),
            panel_id="pr-cost", graph_id="pr-cost",
            subtitle="Where the frontier, and therefore the inference bill, actually is.",
        ),
    ])


def _accept_note(report: dict, rows: list[dict]):
    """Name the accept slice these rows are, when the cell swept τ_accept as well.

    Without it the panel would claim to show "all configurations" while showing a fifth of them,
    and the reader has no way to tell a 1-D sweep from one slice of a grid by looking.
    """
    accepts = {r.get("op_accept") for r in rows}
    if accepts == {None}:
        return None
    n_points = len(report.get("operating_points") or rows)
    shown = next(iter(accepts)) if len(accepts) == 1 else None
    return theme.note(
        f"This cell sweeps τ_accept as well: {n_points} configurations in total, of which the "
        f"{len(rows)} below are the ones at "
        + (f"τ_accept = {shown:g}" if shown is not None else "the selected τ_accept")
        + ". τ_accept changes which nodes are kept, never which are visited, so the cost and "
          "recall-ceiling columns are the same in every slice, precision and recall are not. "
          "Switch slices with the sidebar's operating-point dropdown.",
        tone="info")


def _build_frontier(cell_id: str, report: dict, rows: list[dict], mode: str):
    if not rows:
        return html.Div()
    accept_note = _accept_note(report, rows)
    return theme.panel(
        "τ_prune frontier",
        html.Div([
            dcc.Graph(id="pr-frontier-graph", figure=_frontier_figure(rows, mode),
                      config=theme.GRAPH_CONFIG),
            accept_note or html.Div(),
            theme.table(
                ["τ", "P", "R", "F1", "recall ceiling", "annotated terms blocked", "SLM calls / report"],
                [[html.B(r["operating_point"]) if r["operating_point"] ==
                  report.get("operating_point") else r["operating_point"],
                  _fmt(r.get("micro_precision")), _fmt(r.get("micro_recall")),
                  _fmt(r.get("micro_f1")), _pct(r.get("recall_ceiling")),
                  r.get("n_blocked"), _fmt(r.get("mean_slm_calls"), 0)] for r in rows],
                align={1: "num", 2: "num", 3: "num", 4: "num", 5: "num", 6: "num"}),
            theme.note(
                "Every point here is a run that happened. The run wrote the whole sweep to disk, "
                "one directory per τ_prune, so this frontier is read rather than re-run, "
                "which is why τ_prune is a free variable here and fixed in other views. A re-run "
                "would have to invent scores for the nodes each τ never visited, and those nodes "
                "were never asked.", tone="info"),
        ]),
        panel_id="pr-frontier-panel", graph_id="pr-frontier-graph",
        subtitle=(f"{'All ' if accept_note is None else ''}{len(rows)} configuration"
                  f"{'' if len(rows) == 1 else 's'} of {report.get('method_label')} "
                  f"on {report.get('cohort', '').upper()}"
                  + ("." if accept_note is None else ", at one τ_accept.")),
        markdown=copyexport.frontier_md(report, rows),
    )


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("pr-content", "children"),
        Input("run-a", "value"),
        Input("op-a", "value"),
        Input("store-theme", "data"),
        Input("store-subset", "data"),
        active.TAB_INPUT,
    )
    def _render(cell_id, operating_point, mode, subset, tab):
        active.guard(tab, "pruning")
        if not cell_id:
            return theme.empty("Select a run in the sidebar.")
        registry = state.get_registry()
        keep = registry.aggregate_filter(cell_id, subset)
        return _build(registry.get_report(cell_id, operating_point, report_filter=keep),
                      mode or "light", registry.subset_note(cell_id, subset))

    @app.callback(
        Output("pr-frontier", "children"),
        Input("run-a", "value"),
        Input("op-a", "value"),
        Input("store-theme", "data"),
        active.TAB_INPUT,
    )
    def _render_frontier(cell_id, operating_point, mode, tab):
        active.guard(tab, "pruning")
        if not cell_id:
            return html.Div()
        registry = state.get_registry()
        report = registry.get_report(cell_id, operating_point)
        if not report or not report.get("is_tree"):
            return html.Div()
        return _build_frontier(cell_id, report,
                               registry.tau_frontier(cell_id, operating_point), mode or "light")
