"""Voters, which models actually made each decision.

The k-of-N rule hides its own mechanics: two configurations with the same F1 can be built from
completely different coalitions, and "the ensemble predicted this term" says nothing about whether
seven models agreed or two scraped over a threshold of two. This view opens that up.

The central idea is **decisiveness**. A model is decisive for a term when removing it would drop
that term below k, which, for a threshold on a count, means the term sits at k votes and
the model is one of its voters. Terms above the threshold have no decisive voters at all: they
survive any single model's removal. So "decisive" measures essential-ness at the current
configuration, and a model whose decisive votes are mostly false positives is doing active
damage that no aggregate precision number attributes to it.
"""

from __future__ import annotations

import itertools

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import theme, votes
from . import common

VIEW_ID = "voters"


def layout():
    """The Dash layout of this view."""
    return html.Div(id="vo-body", className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("vo-body", "children"),
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
        rows = votes.decisive_rows(bundle, cfg, bundle["gold"], common.report_ids(bundle, cfg))
        return html.Div([
            render_decisive(bundle, cfg, rows, mode),
            render_vote_histogram(bundle, cfg, rows, mode),
            render_coalitions(bundle, cfg, rows, mode),
            render_agreement(bundle, cfg, mode),
        ]), key


def render_decisive(bundle: dict, cfg: dict, rows: list[dict], mode: str):
    """Per model: votes cast, votes that were essential, and what they bought."""
    if cfg["rule"] == "plurality":
        return theme.panel(
            "Decisive voters",
            theme.note("Plurality picks a per-sentence argmax rather than crossing a threshold, "
                       "so no single model's removal has a defined effect on a term. Switch to "
                       "the k-of-N rule in the sidebar to see decisiveness.", "info"),
            panel_id="vo-decisive")
    if not rows:
        return common.empty_panel("Nothing is predicted at this configuration.", "vo-decisive",
                                  "Decisive voters")

    profile = votes.voter_profile(rows, cfg["models"])
    # Coloured by outcome, not by model: the question here is what a decisive vote bought, and
    # status colour answers it directly. Model identity is carried by the x axis.
    p = theme.palette(mode)
    ordered = sorted(profile, key=lambda r: -r["n_decisive"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["model"] for r in ordered], y=[r["n_decisive_tp"] for r in ordered],
                         name="decisive → true positive", marker_color=p["good"]))
    fig.add_trace(go.Bar(x=[r["model"] for r in ordered], y=[r["n_decisive_fp"] for r in ordered],
                         name="decisive → false positive", marker_color=p["critical"]))
    fig.add_trace(go.Bar(x=[r["model"] for r in ordered],
                         y=[r["n_voted"] - r["n_decisive"] for r in ordered],
                         name="voted, not decisive", marker_color=p["neutral"], opacity=0.4))
    fig.update_layout(**theme.plotly_layout(mode, barmode="stack", yaxis={"title": "terms"},
                                            height=360))

    headers = ["model", "voted", "of which TP", "decisive", "decisive → TP", "decisive → FP",
               "% of its votes decisive"]
    table_rows = [[r["model"], r["n_voted"], r["n_voted_tp"], r["n_decisive"],
                   r["n_decisive_tp"], r["n_decisive_fp"], f"{r['pct_decisive']:.1f}"]
                  for r in ordered]
    n_decisive_terms = sum(1 for r in rows if r["n_decisive"])
    return theme.panel(
        "Decisive voters",
        html.Div([dcc.Graph(id="vo-dec-fig", figure=fig, config=theme.GRAPH_CONFIG),
                  common.table(headers, table_rows)]),
        panel_id="vo-decisive", graph_id="vo-dec-fig",
        subtitle=(f"{n_decisive_terms} of {len(rows)} predicted terms sit on k={cfg['k']} "
                  f"and would be lost if any one of their voters were removed. "
                  + common.describe(cfg)),
        markdown=common.md_table(headers, table_rows))


def render_vote_histogram(bundle: dict, cfg: dict, rows: list[dict], mode: str):
    """How many models backed each predicted term, split by whether it was right."""
    if not rows:
        return common.empty_panel("Nothing is predicted at this configuration.", "vo-hist",
                                  "Vote counts")
    n = len(cfg["models"])
    tp = [0] * (n + 1)
    fp = [0] * (n + 1)
    for r in rows:
        (tp if r["outcome"] == "TP" else fp)[r["n_votes"]] += 1

    p = theme.palette(mode)
    xs = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=tp[1:], name="true positive", marker_color=p["good"]))
    fig.add_trace(go.Bar(x=xs, y=fp[1:], name="false positive", marker_color=p["critical"]))
    fig.update_layout(**theme.plotly_layout(
        mode, barmode="stack", xaxis={"title": "models that detected the term", "dtick": 1},
        yaxis={"title": "terms"}, height=320))

    headers = ["votes", "TP", "FP", "precision at this vote count"]
    table_rows = [[v, tp[v], fp[v],
                   common.num(tp[v] / (tp[v] + fp[v])) if tp[v] + fp[v] else "—"]
                  for v in xs]
    return theme.panel(
        "Vote count vs correctness",
        html.Div([dcc.Graph(id="vo-hist-fig", figure=fig, config=theme.GRAPH_CONFIG),
                  common.table(headers, table_rows)]),
        panel_id="vo-hist", graph_id="vo-hist-fig",
        subtitle="Agreement is the ensemble's only confidence signal, and it is discrete, at most "
                 "N+1 distinct values. This is the curve the k slider is sliding along.",
        markdown=common.md_table(headers, table_rows))


def render_coalitions(bundle: dict, cfg: dict, rows: list[dict], mode: str):
    """Which *specific* model sets keep recurring, and how right each one is."""
    if not rows:
        return common.empty_panel("Nothing is predicted at this configuration.", "vo-coal",
                                  "Voter coalitions")
    coalitions = votes.coalitions(rows, cfg["models"])
    colors = theme.model_colors(bundle["all_models"], mode)
    p = theme.palette(mode)

    labels = ["+".join(m[:4] for m in c["models"]) or "∅" for c in coalitions]
    fig = go.Figure(go.Bar(
        x=[c["n"] for c in coalitions][::-1], y=labels[::-1], orientation="h",
        marker_color=[p["good"] if c["precision"] >= 0.5 else p["critical"]
                      for c in coalitions][::-1],
        hovertemplate="%{y}<br>%{x} terms<extra></extra>"))
    fig.update_layout(**theme.plotly_layout(mode, xaxis={"title": "terms"},
                                            height=max(280, 26 * len(coalitions))))

    headers = ["coalition", "size", "terms", "TP", "precision"]
    table_rows = [[", ".join(c["models"]), c["size"], c["n"], c["n_tp"], common.num(c["precision"])]
                  for c in coalitions]
    return theme.panel(
        "Voter coalitions",
        html.Div([
            dcc.Graph(id="vo-coal-fig", figure=fig, config=theme.GRAPH_CONFIG),
            common.model_chips(cfg["models"], colors),
            common.table(headers, table_rows),
        ]),
        panel_id="vo-coal", graph_id="vo-coal-fig",
        subtitle="The commonest exact voter sets among predicted terms. A large low-precision "
                 "coalition is a correlated failure mode, the same models being wrong together.",
        markdown=common.md_table(headers, table_rows))


def render_agreement(bundle: dict, cfg: dict, mode: str):
    """Pairwise agreement on terms, and pairwise agreement on *errors*.

    The second counts more: an ensemble is only worth eight GPUs if its members fail
    independently, and two models with high error correlation are one model priced as two.
    """
    models = cfg["models"]
    if len(models) < 2:
        return common.empty_panel("Select at least two models.", "vo-agree", "Agreement")

    masks = bundle["masks"](cfg["min_count"])
    index = {m: i for i, m in enumerate(bundle["models"])}
    detected: dict[str, set] = {m: set() for m in models}
    wrong: dict[str, set] = {m: set() for m in models}
    for report_id in common.report_ids(bundle, cfg):
        gold = set(bundle["gold"].get(report_id, ()))
        for hpo_id, mask in masks.get(report_id, {}).items():
            for model in models:
                if mask >> index[model] & 1:
                    detected[model].add((report_id, hpo_id))
                    if hpo_id not in gold:
                        wrong[model].add((report_id, hpo_id))

    def jaccard(sets: dict) -> list[list[float]]:
        grid = []
        for a in models:
            row = []
            for b in models:
                union = sets[a] | sets[b]
                row.append(len(sets[a] & sets[b]) / len(union) if union else 0.0)
            grid.append(row)
        return grid

    p = theme.palette(mode)
    figs = []
    for name, sets, gid in (("terms detected", detected, "vo-agree-fig"),
                            ("false positives", wrong, "vo-err-fig")):
        fig = go.Figure(go.Heatmap(z=jaccard(sets), x=models, y=models, zmin=0, zmax=1,
                                   colorscale=[[i / 6, c] for i, c in enumerate(p["sequential"])],
                                   hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
        fig.update_layout(**theme.plotly_layout(mode, height=110 + 42 * len(models)))
        figs.append((name, dcc.Graph(id=gid, figure=fig, config=theme.GRAPH_CONFIG), gid))

    worst = max(
        (((a, b), len(wrong[a] & wrong[b]) / len(wrong[a] | wrong[b]))
         for a, b in itertools.combinations(models, 2) if wrong[a] | wrong[b]),
        key=lambda x: x[1], default=(None, 0.0))
    subtitle = "Jaccard overlap. "
    if worst[0]:
        subtitle += (f"Most correlated failures: {worst[0][0]} and {worst[0][1]} "
                     f"({worst[1]:.2f}), an ensemble pays for eight models and gets the "
                     "independence it actually has.")
    return theme.panel(
        "Agreement",
        html.Div([html.Div([html.Div(name, className="stat-label"), graph])
                  for name, graph, _ in figs]),
        panel_id="vo-agree", graph_id="vo-agree-fig", subtitle=subtitle)
