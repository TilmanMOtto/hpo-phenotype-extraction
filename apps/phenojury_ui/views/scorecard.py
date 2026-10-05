"""Scorecard, what this cohort scores, and whether the app can be believed about it.

The verification panel comes first on purpose. Every number in this app is recomputed from the raw
artifacts rather than read out of ``slm_ensemble_agg_summary.csv``, because the CSV cannot answer
any of the questions the app exists for. That makes a bug here indistinguishable from a finding, so
the first thing the screen says is whether the recomputation reproduces what the driver shipped, on this cohort, this load, these files. See :mod:`apps.phenojury_ui.verify`.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import theme, votes
from . import common

VIEW_ID = "scorecard"


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="sc-gates"),
        html.Div(id="sc-stats"),
        html.Div(id="sc-body"),
    ], className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("sc-gates", "children"),
        Output("sc-stats", "children"),
        Output("sc-body", "children"),
        Output(f"sig-{VIEW_ID}", "data"),
        Input("tabs", "value"),
        Input("run-a", "value"),
        Input("store-config", "data"),
        Input("store-theme", "data"),
        State(f"sig-{VIEW_ID}", "data"),
    )
    def render(tab, run_a, config, mode, previous):
        key = common.render_key(VIEW_ID, run_a, config, mode)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return common.no_data(run_a, "Scorecard"), None, None, key
        cfg = common.resolve(config, bundle)
        mode = mode or "light"
        return (render_gates(bundle, mode),
                render_stats(bundle, cfg),
                html.Div([render_sweep(bundle, cfg, mode),
                          render_rule_table(bundle, cfg),
                          render_report_spread(bundle, cfg, mode)]),
                key)


def render_gates(bundle: dict, mode: str):
    """The verification verdict, gate by gate."""
    summary = bundle["gate_summary"]
    tone = {"pass": "info", "fail": "danger", "skip": "warn"}[summary["status"]]
    rows = [[g["title"], g["status"].upper(), g["detail"]] for g in bundle["gates"]]
    body = html.Div([
        theme.note(summary["headline"], tone),
        common.table(["Gate", "Result", "Detail"], rows, numeric_from=99)
        if rows else theme.empty("No gates ran, this cohort has no aggregate-stage artifacts."),
    ])
    return theme.panel(
        "Verification", body, panel_id="sc-gates",
        subtitle="Recomputed from the raw artifacts, compared against what the driver shipped",
        markdown=common.md_table(["Gate", "Result", "Detail"], rows),
    )


def render_stats(bundle: dict, cfg: dict):
    """Micro precision, recall and F1 of the cohort at the selected voting configuration."""
    predicted = votes.predicted_sets(bundle, cfg)
    ids = common.report_ids(bundle, cfg)
    metrics = votes.score(bundle["gold"], predicted, ids)
    counts = votes._counts(bundle["gold"], predicted, ids)
    shipped = common.is_shipped(cfg, bundle)
    return html.Div([
        theme.stat_row([
            theme.stat("micro F1", common.num(metrics["micro_f1"]),
                       common.describe(cfg),
                       tone="good" if shipped else None,
                       help_text="Pooled over reports: TP / (TP+FP), TP / (TP+FN)."),
            theme.stat("micro precision", common.num(metrics["micro_precision"]),
                       f"{counts['n_tp']} of {counts['n_predicted']} predicted terms"),
            theme.stat("micro recall", common.num(metrics["micro_recall"]),
                       f"{counts['n_tp']} of {counts['n_gold']} annotated terms"),
            theme.stat("macro F1", common.num(metrics["macro_f1"]),
                       "mean over reports, not over terms"),
            theme.stat("missed", str(counts["n_fn"]),
                       "annotated terms not predicted", tone="warning"),
        ]),
        theme.note(
            "This is a configuration the driver actually wrote a rule directory for, so these "
            "numbers are the shipped ones." if shipped else
            "Exploratory configuration, the driver never ran this one, so these numbers exist "
            "only here. Reset in the sidebar to return to the shipped configuration.",
            "info" if shipped else "warn"),
    ])


def render_sweep(bundle: dict, cfg: dict, mode: str):
    """Precision, recall and F1 against k, the whole trade in one figure."""
    rows = votes.sweep_k(bundle, bundle["gold"], cfg["models"], cfg["min_count"],
                         common.report_ids(bundle, cfg))
    k_rows = [r for r in rows if r["k"] is not None]
    if not k_rows:
        return common.empty_panel("No voting models in this cohort.", "sc-sweep", "The k trade")

    p = theme.palette(mode)
    ks = [r["k"] for r in k_rows]
    fig = go.Figure()
    for name, key, colour in (("precision", "micro_precision", p["categorical"][0]),
                              ("recall", "micro_recall", p["categorical"][1]),
                              ("F1", "micro_f1", p["categorical"][5])):
        fig.add_trace(go.Scatter(x=ks, y=[r[key] for r in k_rows], name=name, mode="lines+markers",
                                 line={"color": colour, "width": 2}))
    fig.add_trace(go.Bar(x=ks, y=[r["n_predicted"] for r in k_rows], name="terms predicted",
                         marker_color=p["neutral"], opacity=0.30, yaxis="y2"))
    if cfg["rule"] != "plurality":
        fig.add_vline(x=cfg["k"], line_dash="dot", line_color=p["text_muted"])
    fig.update_layout(**theme.plotly_layout(
        mode, xaxis={"title": "k, models that must agree", "dtick": 1},
        yaxis={"title": "micro metric", "range": [0, 1]},
        yaxis2={"title": "terms predicted", "overlaying": "y", "side": "right",
                "showgrid": False, "tickfont": {"color": p["text_muted"]}},
        height=380))

    plurality = next((r for r in rows if r["k"] is None), None)
    subtitle = ("Raising k trades recall for precision; the dotted line is the current operating "
                "point.")
    if plurality:
        subtitle += (f" Per-sentence plurality scores F1 {plurality['micro_f1']:.3f} at "
                     f"{plurality['n_predicted']} terms.")
    return theme.panel(
        "The k trade", dcc.Graph(id="sc-sweep-fig", figure=fig, config=theme.GRAPH_CONFIG),
        panel_id="sc-sweep", subtitle=subtitle, graph_id="sc-sweep-fig",
        markdown=common.md_table(
            ["rule", "predicted", "TP", "FP", "FN", "microP", "microR", "microF1"],
            [[r["rule"], r["n_predicted"], r["n_tp"], r["n_fp"], r["n_fn"],
              common.num(r["micro_precision"]), common.num(r["micro_recall"]),
              common.num(r["micro_f1"])] for r in rows]),
    )


def render_rule_table(bundle: dict, cfg: dict):
    """Every rule in counts as well as rates, at high k the rates barely move and the counts fall
    off a cliff, and only the counts show it."""
    rows = votes.sweep_k(bundle, bundle["gold"], cfg["models"], cfg["min_count"],
                         common.report_ids(bundle, cfg))
    headers = ["rule", "predicted", "TP", "FP", "FN", "micro P", "micro R", "micro F1", "macro F1"]
    table_rows = [[r["rule"], r["n_predicted"], r["n_tp"], r["n_fp"], r["n_fn"],
                   common.num(r["micro_precision"]), common.num(r["micro_recall"]),
                   common.num(r["micro_f1"]), common.num(r["macro_f1"])] for r in rows]
    return theme.panel("Every rule", common.table(headers, table_rows), panel_id="sc-rules",
                       subtitle=common.describe(cfg),
                       markdown=common.md_table(headers, table_rows))


def render_report_spread(bundle: dict, cfg: dict, mode: str):
    """Per-report F1, a mean of 0.53 over a bimodal cohort is not a typical report."""
    from hpo_extraction.evaluation.set_metrics import calc_metric

    predicted = votes.predicted_sets(bundle, cfg)
    scored = []
    ids = common.report_ids(bundle, cfg)
    for report_id in ids:
        gold = set(bundle["gold"].get(report_id, ()))
        if not gold:
            continue   # F1 is undefined against an empty annotation. Counted separately below
        scored.append((report_id, calc_metric(gold, predicted.get(report_id, set()))[2]))
    n_no_gold = len(ids) - len(scored)
    if not scored:
        return common.empty_panel("No report in this cohort has annotations.", "sc-spread",
                                  "Per-report F1")

    p = theme.palette(mode)
    fig = go.Figure(go.Histogram(x=[f1 for _, f1 in scored], nbinsx=20,
                                 marker_color=p["categorical"][0]))
    fig.update_layout(**theme.plotly_layout(
        mode, xaxis={"title": "report F1", "range": [0, 1]}, yaxis={"title": "reports"},
        height=280))

    worst = sorted(scored, key=lambda x: x[1])[:5]
    return theme.panel(
        "Per-report F1", html.Div([
            dcc.Graph(id="sc-spread-fig", figure=fig, config=theme.GRAPH_CONFIG),
            common.table(["worst reports", "F1"], [[r, common.num(f1)] for r, f1 in worst]),
        ]),
        panel_id="sc-spread", graph_id="sc-spread-fig",
        subtitle=(f"{len(scored)} annotated reports"
                  + (f"; {n_no_gold} with no annotations are excluded (F1 undefined)"
                     if n_no_gold else "")),
        markdown=common.md_table(["report", "F1"], [[r, common.num(f1)] for r, f1 in worst]),
    )
