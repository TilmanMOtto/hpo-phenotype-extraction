"""Missed (FN), the recall autopsy.

Recall is where this method loses, and "recall is 0.52" is not actionable. Every annotated term the
ensemble failed to predict died at one place in the generate → ground → vote chain, and
which one decides what would fix it: more tokens, a better linker, a negation fix, a lower
threshold, or nothing at all. See :mod:`apps.phenojury_ui.autopsy` for the ladder and its two stated
limitations.

The panel that closes the view is the honest form of "just lower k": every candidate k is priced in
the false positives it buys as well as the annotated terms it recovers.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import autopsy, theme, votes
from . import common

VIEW_ID = "missed"


def layout():
    """The Dash layout of this view."""
    return html.Div(id="mi-body", className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("mi-body", "children"),
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
        predicted = votes.predicted_sets(bundle, cfg)
        result = autopsy.build(bundle, cfg, bundle["gold"], predicted,
                              common.report_ids(bundle, cfg))
        return html.Div([
            render_fates(bundle, cfg, result, mode),
            render_blame(bundle, cfg, result, mode),
            render_recoverable(bundle, cfg, mode),
            render_examples(bundle, cfg, result),
        ]), key


def render_fates(bundle: dict, cfg: dict, result: dict, mode: str):
    """Where each missed annotated term was lost (generation, normalisation or vote)."""
    counts = result["counts"]
    total = sum(counts.values())
    n_gold = sum(len(v) for v in bundle["gold"].values())
    if not total:
        return theme.panel("Cause of death",
                           theme.empty("Nothing was missed at this configuration."),
                           panel_id="mi-fates")

    colors = theme.fate_colors(mode)
    order = [f for f in autopsy.FATE_ORDER if counts.get(f)]
    fig = go.Figure(go.Bar(
        x=[counts[f] for f in order][::-1], y=[f for f in order][::-1], orientation="h",
        marker_color=[colors[f] for f in order][::-1],
        hovertemplate="%{y}: %{x} terms<extra></extra>"))
    fig.update_layout(**theme.plotly_layout(mode, xaxis={"title": "annotated terms lost"},
                                            height=90 + 40 * len(order)))

    headers = ["fate", "terms", "% of misses", "% of all annotated", "what it means"]
    table_rows = [[f, counts[f], f"{100.0 * counts[f] / total:.1f}",
                   f"{100.0 * counts[f] / n_gold:.1f}" if n_gold else "—", autopsy.FATE_HELP[f]]
                  for f in autopsy.FATE_ORDER if counts.get(f)]

    body = [dcc.Graph(id="mi-fates-fig", figure=fig, config=theme.GRAPH_CONFIG),
            common.table(headers, table_rows, numeric_from=1)]
    if result["degraded"]:
        body.append(theme.note(
            "This run kept no phenobert_output_* directories, so 'not linked', 'negated' and "
            "'below min count' cannot be separated, all three are reported as 'not linked'.",
            "warn"))
    body.append(theme.note(
        "'not written' is a per-report verdict: the generation run has no segment-level annotation, so "
        "nothing on disk says which sentence a given annotated term should have come from. The "
        "strongest defensible statement is that no model wrote anything anywhere in that report.",
        "info"))

    return theme.panel(
        "Cause of death", html.Div(body), panel_id="mi-fates", graph_id="mi-fates-fig",
        subtitle=f"{total} of {n_gold} annotated terms missed at {common.describe(cfg)}",
        markdown=common.md_table(headers, table_rows))


def render_blame(bundle: dict, cfg: dict, result: dict, mode: str):
    """For the recoverable misses, which models already found them and were outvoted."""
    rows = autopsy.blame(result["rows"], cfg["models"])
    n_recoverable = rows[0]["n_recoverable_total"] if rows else 0
    if not n_recoverable:
        return theme.panel(
            "Outvoted evidence",
            theme.empty("No missed term was found by any model at this configuration, nothing "
                        "here is recoverable by lowering k."),
            panel_id="mi-blame")

    colors = theme.model_colors(bundle["all_models"], mode)
    ordered = sorted(rows, key=lambda r: -r["n_found_but_outvoted"])
    fig = go.Figure(go.Bar(
        x=[r["model"] for r in ordered], y=[r["n_found_but_outvoted"] for r in ordered],
        marker_color=[colors.get(r["model"], "#888") for r in ordered]))
    fig.update_layout(**theme.plotly_layout(mode, yaxis={"title": "annotated terms it found "
                                                                  "but that lost the vote"},
                                            height=320))

    headers = ["model", "found but outvoted", "% of the recoverable misses"]
    table_rows = [[r["model"], r["n_found_but_outvoted"],
                   f"{100.0 * r['n_found_but_outvoted'] / n_recoverable:.1f}"] for r in ordered]
    return theme.panel(
        "Outvoted evidence", html.Div([
            dcc.Graph(id="mi-blame-fig", figure=fig, config=theme.GRAPH_CONFIG),
            common.table(headers, table_rows)]),
        panel_id="mi-blame", graph_id="mi-blame-fig",
        subtitle=f"{n_recoverable} missed term(s) were found by 1..{cfg['k'] - 1} model(s). A "
                 "model high here is finding annotated terms the rest of the ensemble is not, and "
                 "the threshold is throwing them away.",
        markdown=common.md_table(headers, table_rows))


def render_recoverable(bundle: dict, cfg: dict, mode: str):
    """What each lower k would recover, priced in the false positives it buys."""
    if cfg["rule"] == "plurality":
        return theme.panel(
            "What a lower k would buy",
            theme.note("Plurality has no k. Switch to the k-of-N rule to price the trade.", "info"),
            panel_id="mi-recover")
    rows = autopsy.recoverable_by_k(bundle, cfg, bundle["gold"],
                                    common.report_ids(bundle, cfg))
    p = theme.palette(mode)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["k"] for r in rows], y=[r["gained_tp"] for r in rows],
                         name="annotated terms recovered", marker_color=p["good"]))
    fig.add_trace(go.Bar(x=[r["k"] for r in rows], y=[-r["gained_fp"] for r in rows],
                         name="false positives bought", marker_color=p["critical"]))
    fig.update_layout(**theme.plotly_layout(
        mode, barmode="relative", xaxis={"title": "k", "dtick": 1},
        yaxis={"title": "terms gained (+) / bought (−)"}, height=320))

    headers = ["k", "annotated recovered", "false positives bought", "TP per FP"]
    table_rows = [[r["k"], r["gained_tp"], r["gained_fp"],
                   common.num(r["ratio"]) if r["ratio"] is not None else "—"] for r in rows]
    return theme.panel(
        "What a lower k would buy",
        html.Div([dcc.Graph(id="mi-recover-fig", figure=fig, config=theme.GRAPH_CONFIG),
                  common.table(headers, table_rows)]),
        panel_id="mi-recover", graph_id="mi-recover-fig",
        subtitle=f"Relative to the current k={cfg['k']}. Each row is what moving the threshold "
                 "there would change, recall gained against the precision paid for it.",
        markdown=common.md_table(headers, table_rows))


def render_examples(bundle: dict, cfg: dict, result: dict, limit: int = 40):
    """The individual misses, worst fate first, with the models that did find them."""
    rows = result["rows"]
    if not rows:
        return common.empty_panel("Nothing was missed at this configuration.", "mi-examples",
                                  "The misses")
    colors = theme.model_colors(bundle["all_models"], "light")
    order = {fate: i for i, fate in enumerate(autopsy.FATE_ORDER)}
    ordered = sorted(rows, key=lambda r: (order.get(r["fate"], 99), r["report_id"]))[:limit]

    table_rows = [[
        r["report_id"], common.term(bundle, r["hpo_id"]), r["fate"], r["n_votes"],
        common.model_chips([m for i, m in enumerate(cfg["models"]) if r["voters"] >> i & 1],
                           colors) if r["voters"] else "—",
        r["n_negated_models"] or "",
    ] for r in ordered]
    headers = ["report", "annotated term", "fate", "votes", "found by", "models that negated it"]
    return theme.panel(
        "The misses", common.table(headers, table_rows, numeric_from=3),
        panel_id="mi-examples",
        subtitle=f"{min(limit, len(rows))} of {len(rows)}, grouped by fate. Open a report in the "
                 "deep dive to see what each model actually wrote on its sentences.",
        markdown=common.md_table(
            ["report", "term", "fate", "votes"],
            [[r["report_id"], r["hpo_id"], r["fate"], r["n_votes"]] for r in ordered]))
