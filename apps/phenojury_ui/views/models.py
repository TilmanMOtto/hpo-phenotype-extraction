"""Models, the character of each of the eight ensemble members.

Four questions, in the order they matter when deciding whether an ensemble member earns its GPU:

1. **Does it produce anything?** The funnel sentences → wrote something → grounded to an HPO. A
   reasoning model that never closes its ``<think>`` block scores zero here while looking busy in
   the logs, which is the Free Listing generation run failure the driver now raises ``SystemExit(3)`` on.
2. **What kind of detector is it?** Precision against recall over the terms it alone proposes, sprayers up and left, conservatives down and right.
3. **What does it contribute to the ensemble?** Exact Shapley, leave-one-out, and unique recall.
   These disagree by design: a model can have negative leave-one-out (removing it *helps*) and
   still hold the only route to a handful of annotated terms.
4. **How does it behave?** Reply length, ``<think>`` rate, empty-after-strip rate.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import theme, votes
from . import common

VIEW_ID = "models"


def layout():
    """The Dash layout of this view."""
    return html.Div(id="mo-body", className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("mo-body", "children"),
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
            return common.no_data(run_a), key
        cfg = common.resolve(config, bundle)
        mode = mode or "light"
        return html.Div([
            render_funnel(bundle, cfg, mode),
            render_archetypes(bundle, cfg, mode),
            render_contribution(bundle, cfg, mode),
            render_behaviour(bundle, cfg),
        ]), key


def _colors(bundle: dict, mode: str) -> dict:
    return theme.model_colors(bundle["all_models"], mode)


def render_funnel(bundle: dict, cfg: dict, mode: str):
    """sentences → wrote something → grounded, per model."""
    from ..detections import productivity

    rows = productivity(bundle["built"], bundle["sent_hpos"], cfg["models"])
    if not rows:
        return common.empty_panel("No extractions on disk for the selected models.",
                                  "mo-funnel", "Productivity funnel")

    colors = _colors(bundle, mode)
    p = theme.palette(mode)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["model"] for r in rows], y=[r["n_sentences"] for r in rows],
                         name="sentences seen", marker_color=p["neutral"], opacity=0.35))
    fig.add_trace(go.Bar(x=[r["model"] for r in rows], y=[r["n_wrote_something"] for r in rows],
                         name="wrote something", marker_color=p["categorical"][2], opacity=0.7))
    fig.add_trace(go.Bar(x=[r["model"] for r in rows], y=[r["n_wrote_and_hpo"] for r in rows],
                         name="grounded to an HPO",
                         marker_color=[colors.get(r["model"], p["categorical"][0]) for r in rows]))
    fig.update_layout(**theme.plotly_layout(mode, barmode="overlay", yaxis={"title": "sentences"},
                                            height=340))

    headers = ["model", "sentences", "wrote", "→ HPO", "% of wrote", "% of all"]
    table_rows = [[r["model"], r["n_sentences"], r["n_wrote_something"], r["n_wrote_and_hpo"],
                   f"{r['pct_of_wrote']:.1f}", f"{r['pct_of_all']:.1f}"] for r in rows]
    return theme.panel(
        "Productivity funnel",
        html.Div([dcc.Graph(id="mo-funnel-fig", figure=fig, config=theme.GRAPH_CONFIG),
                  common.table(headers, table_rows)]),
        panel_id="mo-funnel", graph_id="mo-funnel-fig",
        subtitle="A wide gap between the middle and inner bar is a grounding problem, not a "
                 "generation one, the model wrote prose PhenoBERT could not link.",
        markdown=common.md_table(headers, table_rows))


def render_archetypes(bundle: dict, cfg: dict, mode: str):
    """Each model alone: precision against recall over the terms it would predict by itself."""
    masks = bundle["masks"](cfg["min_count"])
    gold = bundle["gold"]
    colors = _colors(bundle, mode)
    points = []
    for model in cfg["models"]:
        bit = 1 << bundle["models"].index(model)
        tp = n_pred = n_gold = 0
        for report_id in common.report_ids(bundle, cfg):
            gold_set = set(gold.get(report_id, ()))
            solo = {h for h, mask in masks.get(report_id, {}).items() if mask & bit}
            tp += len(solo & gold_set)
            n_pred += len(solo)
            n_gold += len(gold_set)
        points.append({
            "model": model,
            "precision": tp / n_pred if n_pred else 0.0,
            "recall": tp / n_gold if n_gold else 0.0,
            "n_predicted": n_pred,
        })

    if not points:
        return common.empty_panel("No voting models selected.", "mo-arch", "Archetypes")

    p = theme.palette(mode)
    fig = go.Figure()
    for point in points:
        fig.add_trace(go.Scatter(
            x=[point["recall"]], y=[point["precision"]], mode="markers+text",
            text=[point["model"]], textposition="top center", name=point["model"],
            marker={"size": 14, "color": colors.get(point["model"], p["categorical"][0])},
            hovertemplate=(f"{point['model']}<br>precision %{{y:.3f}}<br>recall %{{x:.3f}}"
                           f"<br>{point['n_predicted']} terms<extra></extra>"),
            showlegend=False))
    fig.update_layout(**theme.plotly_layout(
        mode, xaxis={"title": "recall alone", "range": [0, 1]},
        yaxis={"title": "precision alone", "range": [0, 1]}, height=380))

    headers = ["model", "precision alone", "recall alone", "terms proposed"]
    table_rows = [[q["model"], common.num(q["precision"]), common.num(q["recall"]),
                   q["n_predicted"]] for q in sorted(points, key=lambda q: -q["recall"])]
    return theme.panel(
        "Archetypes", html.Div([dcc.Graph(id="mo-arch-fig", figure=fig, config=theme.GRAPH_CONFIG),
                                common.table(headers, table_rows)]),
        panel_id="mo-arch", graph_id="mo-arch-fig",
        subtitle="Each model scored as if it were the whole ensemble (k=1, itself only). Upper "
                 "left is a sprayer, lower right a conservative, the vote is what reconciles them.",
        markdown=common.md_table(headers, table_rows))


def render_contribution(bundle: dict, cfg: dict, mode: str):
    """Shapley, leave-one-out and unique recall, side by side because they disagree."""
    k = cfg["k"] if cfg["rule"] != "plurality" else 1
    shap = {r["model"]: r["shapley"] for r in
            votes.shapley(bundle, bundle["gold"], cfg["models"], k, cfg["min_count"],
                          common.report_ids(bundle, cfg))}
    loo = {r["model"]: r for r in
           votes.leave_one_out(bundle, bundle["gold"], cfg["models"], k, cfg["min_count"],
                               common.report_ids(bundle, cfg))}
    uniq = {r["model"]: r for r in
            votes.unique_recall(bundle, bundle["gold"], cfg["models"], cfg["min_count"],
                                common.report_ids(bundle, cfg))}
    if not shap:
        return common.empty_panel("No voting models selected.", "mo-contrib", "Contribution")

    colors = _colors(bundle, mode)
    p = theme.palette(mode)
    ordered = sorted(cfg["models"], key=lambda m: -shap.get(m, 0.0))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ordered, y=[shap.get(m, 0.0) for m in ordered], name="Shapley",
                         marker_color=[colors.get(m, p["categorical"][0]) for m in ordered]))
    fig.add_trace(go.Bar(x=ordered, y=[loo[m]["delta"] for m in ordered],
                         name="leave-one-out Δ F1", marker_color=p["neutral"], opacity=0.55))
    fig.update_layout(**theme.plotly_layout(mode, barmode="group", yaxis={"title": "Δ micro F1"},
                                            height=340))

    headers = ["model", "Shapley", "F1 without it", "leave-one-out Δ", "ground truth found",
               "ground truth only it found"]
    table_rows = [[m, common.num(shap.get(m)), common.num(loo[m]["f1_without"]),
                   common.signed(loo[m]["delta"]), uniq[m]["n_gold_found"], uniq[m]["n_unique"]]
                  for m in ordered]
    return theme.panel(
        "Contribution to the ensemble",
        html.Div([dcc.Graph(id="mo-contrib-fig", figure=fig, config=theme.GRAPH_CONFIG),
                  common.table(headers, table_rows),
                  theme.note(
                      "Shapley averages a model's marginal F1 over all 2^n coalitions with k "
                      "clamped to the coalition size, without the clamp every proper subset "
                      "scores zero at high k and the values collapse to F1/n. A negative "
                      "leave-one-out Δ means the ensemble is better without that model at this k; "
                      "check its unique-recall column before dropping it.", "info")]),
        panel_id="mo-contrib", graph_id="mo-contrib-fig",
        subtitle=common.describe(cfg), markdown=common.md_table(headers, table_rows))


def render_behaviour(bundle: dict, cfg: dict):
    """Reply length, reasoning-block rate, and how often stripping empties a reply outright."""
    rows = []
    for model in cfg["models"]:
        replies = [v for (m, _r, _s), v in bundle["built"]["replies"].items() if m == model]
        if not replies:
            continue
        n = len(replies)
        n_think = sum(1 for v in replies if len(v["raw"]) != len(v["stripped"]))
        n_emptied = sum(1 for v in replies if v["raw"].strip() and not v["stripped"].strip())
        rows.append([
            model, n,
            f"{sum(len(v['stripped']) for v in replies) / n:.0f}",
            f"{100.0 * n_think / n:.1f}",
            f"{100.0 * n_emptied / n:.1f}",
            f"{100.0 * sum(1 for v in replies if v['wrote']) / n:.1f}",
        ])
    if not rows:
        return common.empty_panel("No extractions on disk.", "mo-behav", "Generation behaviour")

    headers = ["model", "replies", "mean chars (stripped)", "% with a reasoning block",
               "% emptied by stripping", "% counted as writing something"]
    return theme.panel(
        "Generation behaviour", html.Div([
            common.table(headers, rows),
            theme.note(
                "'Emptied by stripping' is the silent killer: strip_think drops an unterminated "
                "reasoning block whole, so a model that ran out of tokens mid-thought contributes "
                "nothing while its logs look healthy. Anything above a few percent means "
                "max_new_tokens is too low for that model.", "warn"),
        ]),
        panel_id="mo-behav", markdown=common.md_table(headers, rows))
