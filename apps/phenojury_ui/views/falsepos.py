"""False positives, how wrong each one is.

Open-set scoring compares term identity, so predicting "Seizure" where the annotation says "Focal
seizure" costs as much as predicting a term from an unrelated organ system. That is the
right scoring rule and a misleading summary, because the two errors say opposite things about the
model: one is a granularity mismatch, the other is a hallucination. This view separates them using
the ontology, and then asks the question the k slider actually turns on, *which kind* of error
does raising k remove?

If raising k mostly removes near-misses, the threshold is discarding good evidence and the fix is
elsewhere. If it mostly removes unrelated terms, the vote is doing its job.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .. import memo, relations, theme, votes
from ..detections import popcount
from . import common

VIEW_ID = "falsepos"


def layout():
    """The Dash layout of this view."""
    return html.Div(id="fp-body", className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("fp-body", "children"),
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
        if bundle.get("tree") is None:
            return theme.panel(
                "False positives",
                theme.note("The ontology could not be loaded (resources/util/hpo.json), so false "
                           "positives cannot be classified. Every other view still works.",
                           "danger"),
                panel_id="fp-none"), key
        cfg = common.resolve(config, bundle)
        mode = mode or "light"
        rows = classify_all(bundle, cfg)
        return html.Div([
            render_mix(bundle, cfg, rows, mode),
            render_distance(bundle, cfg, rows, mode),
            render_by_model(bundle, cfg, rows, mode),
            render_across_k(bundle, cfg, mode),
            render_examples(bundle, cfg, rows),
        ]), key


def classify_all(bundle: dict, cfg: dict) -> list[dict]:
    """One row per false positive: relation class, near-miss distance and its voters.

    A bounded BFS through the HPO DAG per false positive, the most expensive thing any view does,
    so it is computed once per cohort and every k reads from that.

    Under ``vote_k``, the classification at k is the k=1 rows that got at least k votes: the
    predicted sets shrink monotonically in k, and a term's relation to the annotations does not
    depend on the threshold that let it through. So the whole k slider costs one ontology walk.
    Plurality is not a threshold on a count and gets its own walk.
    """
    if cfg["rule"] == "plurality":
        return memo.cached(bundle, ("fp.classify", memo.config_key(cfg)),
                           lambda: _classify_all(bundle, cfg))

    base = memo.cached(bundle, ("fp.classify.k1", memo.config_key({**cfg, "k": 1})),
                       lambda: _classify_all(bundle, {**cfg, "k": 1}))
    k = max(1, min(int(cfg["k"]), len(cfg["models"]) or 1))
    if k == 1:
        return base
    return [row for row in base if popcount(row["voters"]) >= k]


def _classify_all(bundle: dict, cfg: dict) -> list[dict]:
    tree = bundle["tree"]
    predicted = votes.predicted_sets(bundle, cfg)
    masks = bundle["masks"](cfg["min_count"])
    subset_mask = bundle["mask_of"](cfg["models"])
    out: list[dict] = []
    for report_id in common.report_ids(bundle, cfg):
        gold = set(bundle["gold"].get(report_id, ()))
        for row in relations.classify_report(tree, predicted.get(report_id, set()), gold):
            out.append({
                **row, "report_id": report_id,
                "voters": masks.get(report_id, {}).get(row["hpo_id"], 0) & subset_mask,
            })
    return out


def counts_across_k(bundle: dict, cfg: dict) -> tuple[list[int], dict[str, list[int]]]:
    """``(ks, {relation: [count at each k]})`` from a *single* classification.

    Classifying once per k is the obvious implementation and the wrong one. Two facts make it
    unnecessary: ``vote_k`` predictions shrink monotonically in k, so every false positive at any k
    is already a false positive at k=1. And a term's relation class is a function of the term, the
    report's annotations and the ontology, never of k. So the k=1 rows carry their own vote count,
    and each one contributes to the prefix of k values it survives.

    On a full cohort that is one ontology walk instead of eight.
    """
    n = max(1, len(cfg["models"]))
    ks = list(range(1, n + 1))
    rows = classify_all(bundle, {**cfg, "rule": "vote_k", "k": 1})
    series = {relation: [0] * n for relation in relations.RELATION_ORDER}
    for row in rows:
        votes_for = min(popcount(row["voters"]), n)
        row_series = series[row["relation"]]
        for i in range(votes_for):
            row_series[i] += 1
    return ks, series


def render_mix(bundle: dict, cfg: dict, rows: list[dict], mode: str):
    """Share of false positives per category."""
    if not rows:
        return theme.panel("Error mix", theme.empty("No false positives at this configuration."),
                           panel_id="fp-mix")
    counts = {r: 0 for r in relations.RELATION_ORDER}
    for row in rows:
        counts[row["relation"]] += 1
    total = len(rows)
    colors = theme.relation_colors(mode)
    order = [r for r in relations.RELATION_ORDER if counts[r]]

    fig = go.Figure(go.Bar(
        x=[counts[r] for r in order][::-1], y=order[::-1], orientation="h",
        marker_color=[colors[r] for r in order][::-1],
        hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(**theme.plotly_layout(mode, xaxis={"title": "false positives"},
                                            height=90 + 40 * len(order)))

    n_near = sum(counts[r] for r in relations.NEAR_MISS)
    headers = ["class", "count", "% of FPs", "meaning"]
    table_rows = [[r, counts[r], f"{100.0 * counts[r] / total:.1f}", relations.RELATION_HELP[r]]
                  for r in relations.RELATION_ORDER if counts[r]]
    return theme.panel(
        "Error mix", html.Div([dcc.Graph(id="fp-mix-fig", figure=fig, config=theme.GRAPH_CONFIG),
                               common.table(headers, table_rows)]),
        panel_id="fp-mix", graph_id="fp-mix-fig",
        subtitle=(f"{total} false positives at {common.describe(cfg)}. "
                  f"{n_near} ({100.0 * n_near / total:.0f}%) are near-misses, an ancestor, a "
                  "descendant or a sibling of something the report really is annotated with."),
        markdown=common.md_table(headers, table_rows))


def render_distance(bundle: dict, cfg: dict, rows: list[dict], mode: str):
    """How far each false positive sits from the nearest annotated term, in ontology hops."""
    measured = [r["distance"] for r in rows if r["distance"] is not None]
    n_far = sum(1 for r in rows if r["distance"] is None and r["relation"] != "no_gt")
    n_no_gt = sum(1 for r in rows if r["relation"] == "no_gt")
    if not measured:
        return theme.panel(
            "Near-miss distance",
            theme.empty("No false positive is within "
                        f"{relations.MAX_HOPS} hops of an annotated term."),
            panel_id="fp-dist")

    p = theme.palette(mode)
    hist = {d: measured.count(d) for d in sorted(set(measured))}
    fig = go.Figure(go.Bar(x=list(hist), y=list(hist.values()), marker_color=p["categorical"][0]))
    fig.update_layout(**theme.plotly_layout(
        mode, xaxis={"title": "ontology hops to the nearest annotated term", "dtick": 1},
        yaxis={"title": "false positives"}, height=300))

    headers = ["hops", "false positives"]
    table_rows = [[d, n] for d, n in hist.items()]
    table_rows.append([f"> {relations.MAX_HOPS}", n_far])
    table_rows.append(["no annotations in the report", n_no_gt])
    return theme.panel(
        "Near-miss distance",
        html.Div([dcc.Graph(id="fp-dist-fig", figure=fig, config=theme.GRAPH_CONFIG),
                  common.table(headers, table_rows)]),
        panel_id="fp-dist", graph_id="fp-dist-fig",
        subtitle=f"Undirected distance in the HPO DAG, measured up to {relations.MAX_HOPS} hops. "
                 f"Mean over the measured ones: {sum(measured) / len(measured):.2f}.",
        markdown=common.md_table(headers, table_rows))


def render_by_model(bundle: dict, cfg: dict, rows: list[dict], mode: str):
    """Which models drive which class of error."""
    if not rows:
        return common.empty_panel("No false positives at this configuration.", "fp-model",
                                  "Who proposes what")
    colors = theme.relation_colors(mode)
    per_model = {m: {r: 0 for r in relations.RELATION_ORDER} for m in cfg["models"]}
    for row in rows:
        for i, model in enumerate(cfg["models"]):
            if row["voters"] >> i & 1:
                per_model[model][row["relation"]] += 1

    fig = go.Figure()
    for relation in relations.RELATION_ORDER:
        values = [per_model[m][relation] for m in cfg["models"]]
        if any(values):
            fig.add_trace(go.Bar(x=cfg["models"], y=values, name=relation,
                                 marker_color=colors[relation]))
    fig.update_layout(**theme.plotly_layout(mode, barmode="stack",
                                            yaxis={"title": "false positives it voted for"},
                                            height=340))

    headers = ["model"] + [r for r in relations.RELATION_ORDER
                           if any(per_model[m][r] for m in cfg["models"])]
    table_rows = [[m] + [per_model[m][r] for r in headers[1:]] for m in cfg["models"]]
    return theme.panel(
        "Who proposes what", html.Div([
            dcc.Graph(id="fp-model-fig", figure=fig, config=theme.GRAPH_CONFIG),
            common.table(headers, table_rows)]),
        panel_id="fp-model", graph_id="fp-model-fig",
        subtitle="A model whose errors are mostly ancestors and descendants is reading the report "
                 "correctly and naming it at the wrong granularity, a different problem from a "
                 "model whose errors are unrelated terms.",
        markdown=common.md_table(headers, table_rows))


def render_across_k(bundle: dict, cfg: dict, mode: str):
    """The error mix as k rises, does the vote remove hallucinations or near-misses first?"""
    if cfg["rule"] == "plurality":
        return theme.panel(
            "What raising k removes",
            theme.note("Plurality has no k, it picks a per-sentence argmax. Switch to the k-of-N "
                       "rule in the sidebar to see how the error mix moves with the threshold.",
                       "info"),
            panel_id="fp-acrossk")
    if not cfg["models"]:
        return common.empty_panel("No voting models in this cohort.", "fp-acrossk",
                                  "What raising k removes")
    colors = theme.relation_colors(mode)
    ks, series = counts_across_k(bundle, cfg)

    fig = go.Figure()
    for relation in relations.RELATION_ORDER:
        if any(series[relation]):
            fig.add_trace(go.Scatter(x=ks, y=series[relation], name=relation, mode="lines+markers",
                                     line={"color": colors[relation], "width": 2}))
    fig.update_layout(**theme.plotly_layout(mode, xaxis={"title": "k", "dtick": 1},
                                            yaxis={"title": "false positives"}, height=340))

    first, last = ks[0], ks[-1]
    near_drop = sum(series[r][0] - series[r][-1] for r in relations.NEAR_MISS)
    far_drop = (series["unrelated"][0] - series["unrelated"][-1]
                + series["no_gt"][0] - series["no_gt"][-1])
    verdict = ("mostly near-misses, the threshold is discarding evidence the model got roughly "
               "right" if near_drop > far_drop else
               "mostly unrelated terms, the vote is filtering what it should")
    return theme.panel(
        "What raising k removes",
        dcc.Graph(id="fp-acrossk-fig", figure=fig, config=theme.GRAPH_CONFIG),
        panel_id="fp-acrossk", graph_id="fp-acrossk-fig",
        subtitle=f"Going from k={first} to k={last} removes {near_drop} near-miss and {far_drop} "
                 f"unrelated/no-annotation false positives: {verdict}.")


def render_examples(bundle: dict, cfg: dict, rows: list[dict], limit: int = 40):
    """Up to *limit* example false positives with their evidence."""
    if not rows:
        return common.empty_panel("No false positives at this configuration.", "fp-examples",
                                  "The false positives")
    colors = theme.model_colors(bundle["all_models"], "light")
    order = {r: i for i, r in enumerate(relations.RELATION_ORDER)}
    ordered = sorted(rows, key=lambda r: (order[r["relation"]], r["report_id"]))[:limit]
    headers = ["report", "predicted term", "class", "hops", "voted for by"]
    table_rows = [[
        r["report_id"], common.term(bundle, r["hpo_id"]), r["relation"],
        r["distance"] if r["distance"] is not None else "—",
        common.model_chips([m for i, m in enumerate(cfg["models"]) if r["voters"] >> i & 1],
                           colors),
    ] for r in ordered]
    return theme.panel(
        "The false positives", common.table(headers, table_rows, numeric_from=3),
        panel_id="fp-examples",
        subtitle=f"{min(limit, len(rows))} of {len(rows)}, near-miss first.",
        markdown=common.md_table(
            ["report", "term", "class", "hops"],
            [[r["report_id"], r["hpo_id"], r["relation"], r["distance"]] for r in ordered]))
