"""A against B, over the reports they share.

Two uses, and the second is the one that motivated including the non-tree methods at all:

**Ablation.** Two tree cells, different gate, or the same gate at two τ, scored over their
shared reports, with a delta scorecard and the list of annotated terms whose fate changed. The fate
delta is the useful part: a term moving from ``blocked`` to ``rejected`` means the traversal fix
worked and the identification is still wrong, which no F1 delta can tell you.

**Reference.** A tree cell against RAG-HPO, PhenoBERT, flat top-M or the ensemble. The question is
not which
F1 is higher, the thesis tables already answer that, and unkindly, but *which annotated terms the
comparator found that the tree missed, and how the tree lost them*. A term RAG-HPO found and the
tree blocked is evidence about pruning. A term RAG-HPO found and the tree rejected is evidence
about the SLM. Those are different projects.

Everything is computed over the intersection of the two runs' report sets. Comparing a finished
run against a half-finished shard array on their union would attribute the missing shards to the
method.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .. import active, copyexport, state, theme


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="cmp-precondition"),
        dcc.Loading(html.Div(id="cmp-content"), type="dot", delay_show=200),
    ])


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def _delta(a, b, digits: int = 3):
    if a is None or b is None:
        return html.Span("—")
    diff = float(b) - float(a)
    if abs(diff) < 10 ** -(digits + 1):
        return html.Span("=", className="mono")
    return html.Span(f"{diff:+.{digits}f}",
                     className="delta-pos" if diff > 0 else "delta-neg")


def _fate_delta_figure(a: dict, b: dict, mode: str) -> go.Figure:
    """The recall decomposition of both runs, side by side."""
    colors = theme.fate_colors(mode)
    figure = go.Figure()
    for index, (report, name) in enumerate(((a, "A"), (b, "B"))):
        counts = (report.get("ledger") or {}).get("fate_counts") or {}
        for fate in theme.FATE_ORDER:
            value = counts.get(fate, 0)
            if not value:
                continue
            figure.add_bar(
                x=[value], y=[name], orientation="h", name=fate,
                marker_color=colors[fate], showlegend=index == 0,
                legendgroup=fate,
                hovertemplate=f"<b>{fate}</b> · {name}<br>%{{x}} annotated terms<extra></extra>",
            )
    figure.update_layout(**theme.plotly_layout(
        mode, barmode="stack", height=210,
        margin={"l": 40, "r": 20, "t": 46, "b": 34},
        xaxis={"title": "annotated terms"}, yaxis={"showgrid": False, "autorange": "reversed"}))
    return figure


def _scorecard(a: dict, b: dict, shared: int):
    def row(name, key, group="flat", fmt=_fmt):
        va = (a.get(group) or {}).get(key)
        vb = (b.get(group) or {}).get(key)
        return [name, fmt(va), fmt(vb), _delta(va, vb)]

    return theme.table(
        ["metric", "A", "B", "Δ (B − A)"],
        [
            row("micro precision", "micro_precision"),
            row("micro recall", "micro_recall"),
            row("micro F1", "micro_f1"),
            row("macro F1", "macro_f1"),
            row("hF (closure)", "micro_hf", "hierarchy"),
            row("recall achieved", "recall_actual", "ledger", _pct),
            row("recall ceiling", "recall_ceiling", "ledger", _pct),
            row("lost to pruning", "recall_lost_to_pruning", "ledger", _pct),
            ["TP", (a.get("flat") or {}).get("tp"), (b.get("flat") or {}).get("tp"),
             _delta((a.get("flat") or {}).get("tp"), (b.get("flat") or {}).get("tp"), 0)],
            ["FP", (a.get("flat") or {}).get("fp"), (b.get("flat") or {}).get("fp"),
             _delta((a.get("flat") or {}).get("fp"), (b.get("flat") or {}).get("fp"), 0)],
            ["FN", (a.get("flat") or {}).get("fn"), (b.get("flat") or {}).get("fn"),
             _delta((a.get("flat") or {}).get("fn"), (b.get("flat") or {}).get("fn"), 0)],
        ],
        align={1: "num", 2: "num", 3: "num"})


def _fate_changes(a_bundle, b_bundle, shared: set[str], limit: int = 200) -> list[list]:
    """Annotated terms whose fate differs between the two runs, worst change first.

    Sorted so the movements that matter lead: a term B found and A did not, then the reverse,
    then everything else. The sort key is a rank of the (fate_a, fate_b) pair rather than a
    string sort, because alphabetical order would bury ``blocked → found`` under ``found →
    blocked``.
    """
    rank = {"found": 0, "rejected": 1, "unreached": 2, "blocked": 3, "outside_graph": 4}
    a_rows = {(r["report_id"], r["hpo_id"]): r for r in a_bundle["ledger"].get("rows", ())}
    b_rows = {(r["report_id"], r["hpo_id"]): r for r in b_bundle["ledger"].get("rows", ())}

    changes = []
    for key in sorted(set(a_rows) & set(b_rows)):
        if key[0] not in shared:
            continue
        fate_a, fate_b = a_rows[key]["fate"], b_rows[key]["fate"]
        if fate_a == fate_b:
            continue
        improvement = rank.get(fate_a, 9) - rank.get(fate_b, 9)
        changes.append((-improvement, key, fate_a, fate_b,
                        b_rows[key].get("culprit") or a_rows[key].get("culprit")))

    changes.sort()
    rows = []
    for _, (report_id, hpo_id), fate_a, fate_b, culprit in changes[:limit]:
        rows.append([
            report_id, html.Code(hpo_id),
            html.Span(fate_a, className="pill"),
            html.Span("→"),
            html.Span(fate_b, className="pill"),
            html.Code(culprit) if culprit else "—",
        ])
    return rows, len(changes)


def _reference_split(a_bundle, b_bundle, shared: set[str], view):
    """What B found that A missed, bucketed by *how* A missed it.

    The whole reason the non-tree methods are loadable. An annotated term the comparator retrieved and
    the tree blocked indicts the gate. One the comparator retrieved and the tree rejected indicts
    the SLM. The two counts point at different work.
    """
    a_rows = {(r["report_id"], r["hpo_id"]): r["fate"]
              for r in a_bundle["ledger"].get("rows", ())}
    b_rows = {(r["report_id"], r["hpo_id"]): r["fate"]
              for r in b_bundle["ledger"].get("rows", ())}

    buckets: dict[str, int] = {}
    for key, fate_b in b_rows.items():
        if key[0] not in shared or fate_b != "found":
            continue
        fate_a = a_rows.get(key)
        if fate_a is None or fate_a == "found":
            continue
        buckets[fate_a] = buckets.get(fate_a, 0) + 1

    reverse = sum(1 for key, fate_a in a_rows.items()
                  if key[0] in shared and fate_a == "found" and b_rows.get(key) != "found")
    return buckets, reverse


def _build(cell_a, op_a, cell_b, op_b, mode: str):
    registry = state.get_registry()
    a = registry.get_report(cell_a, op_a)
    b = registry.get_report(cell_b, op_b)
    if not a or not b:
        return theme.empty("Select two runs.")

    a_bundle = registry.bundle(cell_a, op_a)
    b_bundle = registry.bundle(cell_b, op_b)
    shared = set(a_bundle["gold"]) & set(b_bundle["gold"])
    if not shared:
        return theme.note(
            "These two runs share no reports, so nothing can be compared. That normally means "
            "one of them is a shard array that has not been merged, run "
            "`python src/hpo_extraction/evaluation/merge_shards.py <run_dir>`.", tone="danger")

    changes, n_changes = _fate_changes(a_bundle, b_bundle, shared)
    found_by_b, found_by_a = _reference_split(a_bundle, b_bundle, shared, registry.view)

    children = [
        theme.note([
            html.B(f"{len(shared)} shared reports. "),
            f"A = {a['label']} @ {a.get('operating_point') or '—'} "
            f"({len(a_bundle['gold'])} reports); ",
            f"B = {b['label']} @ {b.get('operating_point') or '—'} "
            f"({len(b_bundle['gold'])} reports). ",
            "Every number below is computed on the intersection.",
        ], tone="info" if len(shared) == len(a_bundle["gold"]) == len(b_bundle["gold"])
            else "warn"),
        theme.panel("Delta scorecard", _scorecard(a, b, len(shared)), panel_id="cmp-score",
                    markdown=copyexport.compare_md(a, b)),
        theme.panel(
            "Where the annotated terms went, in both runs",
            dcc.Graph(id="cmp-fate", figure=_fate_delta_figure(a, b, mode),
                      config=theme.GRAPH_CONFIG),
            panel_id="cmp-fate", graph_id="cmp-fate",
            subtitle="A recall decomposition each. The comparator methods perform no traversal, "
                     "so their `blocked` band is structurally zero, not missing.",
            markdown=copyexport.compare_md(a, b)),
        theme.panel(
            "What B found that A did not",
            html.Div([
                theme.stat_row([
                    theme.stat("B found, A blocked", str(found_by_b.get("blocked", 0)),
                               "pruning cost these", tone="critical",
                               help_text="A never reached these terms. Evidence about the gate."),
                    theme.stat("B found, A rejected", str(found_by_b.get("rejected", 0)),
                               "identification cost these", tone="warning",
                               help_text="A reached and scored these, and said no. Evidence "
                                         "about the SLM, not the traversal."),
                    theme.stat("B found, A never reached",
                               str(found_by_b.get("unreached", 0) +
                                   found_by_b.get("outside_graph", 0)),
                               "coverage gap"),
                    theme.stat("A found, B did not", str(found_by_a), "the reverse direction"),
                ]),
                theme.note(
                    "Read the first two tiles against each other. If the terms a comparator "
                    "finds are mostly ones the tree *blocked*, the traversal is the bottleneck. "
                    "If they are mostly ones it *rejected*, the threshold and the gate are not "
                    "the problem, the classifier is.", tone="info"),
            ]),
            panel_id="cmp-split"),
        theme.panel(
            f"Annotated terms whose fate changed ({n_changes})",
            theme.table(["report", "HPO", "in A", "", "in B", "culprit"], changes)
            if changes else theme.empty("No annotated term changed fate between these two runs."),
            panel_id="cmp-flips",
            subtitle="Improvements first. A move from `blocked` to `rejected` means the "
                     "traversal fix landed and the identification is still wrong."),
    ]
    return html.Div(children)


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("cmp-precondition", "children"),
        Output("cmp-content", "children"),
        Input("run-a", "value"), Input("op-a", "value"),
        Input("run-b", "value"), Input("op-b", "value"),
        Input("store-theme", "data"), active.TAB_INPUT,
    )
    def _render(cell_a, op_a, cell_b, op_b, mode, tab):
        active.guard(tab, "compare")
        if not cell_a or not cell_b:
            return theme.empty("Pick a Run A and a Run B in the sidebar."), html.Div()
        if cell_a == cell_b and op_a == op_b:
            return theme.note("Run A and Run B are the same cell at the same configuration, "
                              "pick a different method or a different τ.", tone="warn"), html.Div()
        return html.Div(), _build(cell_a, op_a, cell_b, op_b, mode or "light")
