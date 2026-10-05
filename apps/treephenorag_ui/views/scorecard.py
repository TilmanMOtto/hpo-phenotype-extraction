"""What happened, and where recall went.

The main of an earlier tree run is not its F1, it is the decomposition behind its recall. A
micro recall of 0.10 can mean the SLM rejected nine annotated terms in ten, or that it never got to see
nine in ten. Those call for opposite fixes, and no aggregate table distinguishes them. The fate
panel does: every annotated term in the cohort lands in one of ``found`` / ``rejected`` /
``blocked`` / ``unreached`` / ``outside_graph``, and the gap between the actual recall and the
pruning ceiling is the part of the miss rate that τ_accept could still recover.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, html

from .. import active, copyexport, state, theme
from ..curated import EXCLUDE_HELP as _EXCLUDE_HELP


def layout():
    """The Dash layout of this view."""
    from dash import dcc

    return dcc.Loading(html.Div(id="scorecard-content"), type="dot", delay_show=200)


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def _fate_figure(report: dict, mode: str) -> go.Figure:
    """A single stacked bar: every annotated term of the cohort, by cause of death."""
    counts = (report.get("ledger") or {}).get("fate_counts") or {}
    colors = theme.fate_colors(mode)
    total = sum(counts.values()) or 1

    figure = go.Figure()
    for fate in theme.FATE_ORDER:
        value = counts.get(fate, 0)
        if not value:
            continue
        figure.add_bar(
            x=[value], y=["annotated terms"], orientation="h", name=fate,
            marker_color=colors[fate],
            text=[f"{fate}: {value} ({value / total * 100:.0f}%)"],
            textposition="inside", insidetextanchor="middle",
            hovertemplate=f"<b>{fate}</b><br>{theme.FATE_HELP.get(fate, '')}"
                          f"<br>%{{x}} terms<extra></extra>",
        )
    figure.update_layout(**theme.plotly_layout(
        mode, barmode="stack", height=150, showlegend=False,
        margin={"l": 90, "r": 20, "t": 10, "b": 30},
        xaxis={"title": "annotated terms"}, yaxis={"showgrid": False},
    ))
    return figure


def _recall_gap_figure(report: dict, mode: str) -> go.Figure:
    """Actual recall, the ceiling pruning imposes, and 1.0, the two gaps, side by side."""
    ledger = report.get("ledger") or {}
    palette = theme.palette(mode)
    actual = ledger.get("recall_actual") or 0.0
    ceiling = ledger.get("recall_ceiling") or 0.0

    figure = go.Figure()
    figure.add_bar(x=[actual], y=[""], orientation="h", marker_color=palette["good"],
                   name="achieved", hovertemplate="achieved recall %{x:.3f}<extra></extra>")
    figure.add_bar(x=[max(ceiling - actual, 0.0)], y=[""], orientation="h",
                   marker_color=palette["warning"], name="recoverable by τ_accept",
                   hovertemplate="identification gap %{x:.3f}<extra></extra>")
    figure.add_bar(x=[max(1.0 - ceiling, 0.0)], y=[""], orientation="h",
                   marker_color=palette["critical"], name="unreachable, pruning",
                   hovertemplate="traversal gap %{x:.3f}<extra></extra>")
    figure.update_layout(**theme.plotly_layout(
        mode, barmode="stack", height=170,
        margin={"l": 20, "r": 20, "t": 40, "b": 36},
        xaxis={"title": "share of annotated terms", "range": [0, 1]},
        yaxis={"showgrid": False, "showticklabels": False},
    ))
    return figure


def _main_result_stats(report: dict) -> html.Div:
    flat = report.get("flat") or {}
    hier = report.get("hierarchy") or {}
    ledger = report.get("ledger") or {}
    counts = report.get("counts") or {}
    f1 = flat.get("micro_f1")

    return theme.stat_row([
        theme.stat("micro F1", _fmt(f1), f"P {_fmt(flat.get('micro_precision'))} · "
                                        f"R {_fmt(flat.get('micro_recall'))}",
                   tone="critical" if (f1 or 0) < 0.1 else None,
                   help_text="Pooled over every report, no ancestor closure."),
        theme.stat("hF", _fmt(hier.get("micro_hf")),
                   f"hP {_fmt(hier.get('micro_hp'))} · hR {_fmt(hier.get('micro_hr'))}",
                   help_text="Ancestor-closure F, credits a predicted parent of the right term."),
        theme.stat("TP / FP / FN",
                   f"{flat.get('tp')} / {flat.get('fp')} / {flat.get('fn')}",
                   f"{counts.get('n_gold_terms')} annotated terms",
                   help_text="Pooled counts over the scored reports."),
        theme.stat("recall ceiling", _pct(ledger.get("recall_ceiling")),
                   f"achieved {_pct(ledger.get('recall_actual'))}",
                   tone="warning",
                   help_text="The highest recall this run could reach at any τ_accept, given "
                             "the pruning it performed."),
        theme.stat("lost to pruning", _pct(ledger.get("recall_lost_to_pruning")),
                   f"{(ledger.get('fate_counts') or {}).get('blocked', 0)} terms never reached",
                   tone="critical" if (ledger.get("recall_lost_to_pruning") or 0) > 0.1 else None,
                   help_text="Annotated terms whose every root-to-term path was severed by a "
                             "pruning decision."),
        theme.stat("reports", str(counts.get("n_reports_scored")),
                   f"{counts.get('n_visited_nodes')} nodes scored",
                   help_text="Reports with both a ground truth entry and a prediction summary line."),
    ])


def _cost_stats(report: dict) -> html.Div:
    traversal = report.get("traversal") or {}
    if not traversal.get("mean_slm_calls"):
        return theme.empty("No traversal artifact, inference cost is not recorded for this cell.")
    return theme.stat_row([
        theme.stat("SLM calls / report", _fmt(traversal.get("mean_slm_calls"), 0),
                   f"{traversal.get('total_slm_calls')} total",
                   help_text="From *_traversal.jsonl, which is per-τ. The calls artifact is a "
                             "union over the whole sweep and would overcount."),
        theme.stat("nodes visited / report", _fmt(traversal.get("mean_nodes_visited"), 0),
                   help_text="Unique nodes scored, DAG-deduplicated."),
        theme.stat("frontier insertions / report", _fmt(traversal.get("mean_frontier_insertions"), 0),
                   help_text="Enqueues including duplicates; the excess over nodes visited is "
                             "the DAG's multi-parent structure."),
    ])


def _banner(report: dict):
    """The notes that change how every number below should be read."""
    notes = []
    status = report.get("status")
    if status and status != "ok":
        notes.append(theme.note(
            f"Cell status: {status}. " + " ".join(report.get("problems") or []),
            tone="danger" if status in ("partial", "missing") else "warn"))

    disagreements = (report.get("counts") or {}).get("gold_disagreements") or 0
    if disagreements:
        notes.append(theme.note([
            html.B(f"{disagreements} nodes "),
            "carry an ", html.Code("is_gold"), " flag that disagrees with the ground truth file loaded "
            "here. The artifact flag records whichever ground-truth set was configured at inference time; "
            "every number on this page uses ",
            html.Code(report.get("gold_source") or "?"),
            ". If that is not the file you mean to score against, change it in Data sources.",
        ], tone="warn"))
    return notes


#: Policy keys worth showing, and what each one decided. The manifest carries them under
#: ``policy``. Spelling out what they *mean* is the difference between a provenance line a reader
#: can check and one they have to take on faith.
_POLICY_HELP: dict[str, str] = {
    "criteria": "Which annotation sources make a report a cohort member at all.",
    "prior_annotation_fallback": "Fall back to the prior_annotation annotation for a term the confirmed set no "
                         "longer carries.",
    "require_evidence": "A term needs a segment and a trigger word to enter the ground truth.",
    "include_undecided": "Keep annotations nobody passed a keep/remove verdict on.",
    "include_suggested": "Keep terms a curator proposed themselves.",
    "exclude_labels": "Qualifiers that disqualify an annotation outright.",
}


def _curated_panel(report: dict) -> list:
    """What the curated ground truth covers, and what it charged this cell for, or nothing at all.

    Two numbers here are not visible anywhere else in the app and are the reason this panel
    exists rather than a line in the banner.

    **Reports outside the cohort.** The curated ground truth holds only the reports curation reached, and
    ``scoring.align`` drops the rest before any metric sees them. That is the right behaviour, a report nobody annotated is not a report with no phenotypes, but it is silent, and a
    scorecard reading "48 reports" over a 60-report run otherwise looks like a broken run.

    **False positives a curator had already seen.** A predicted term the policy dropped for a
    ``family`` qualifier or a missing evidence location is charged as a false positive here as an
    invented term is. That is a defensible scoring choice and a misleading error count, so the
    split is shown: it is usually the whole of the precision difference between this ground truth and the
    raw one.
    """
    block = report.get("curated") or {}
    if not block:
        return []

    outside = block.get("n_reports_outside") or 0
    named = block.get("reports_outside") or []
    excluded_fp = block.get("n_excluded_fp") or 0
    fp_total = (report.get("flat") or {}).get("fp")

    stats = theme.stat_row([
        theme.stat("cohort", str(block.get("n_reports_in_cohort")),
                   f"{block.get('n_gold_pairs')} annotated terms",
                   help_text="Reports the curation pass reached. A report outside it is absent "
                             "from the ground truth file rather than present with an empty set."),
        theme.stat("outside the cohort", str(outside),
                   "not scored here" if outside else "every report is curated",
                   tone="critical" if outside else None,
                   help_text="Reports this run predicted for that the curated ground truth does not "
                             "cover. They are excluded from every metric on this page."),
        theme.stat("excluded annotations", str(block.get("n_annotations_excluded")),
                   "seen by a curator, dropped by the policy",
                   help_text="Candidates in the annotation table that did not enter the ground truth: "
                             "family findings, unsure rows, and terms with no located evidence."),
        theme.stat("of which predicted here", str(excluded_fp),
                   (f"{excluded_fp / fp_total * 100:.0f}% of this cell's {fp_total} FP"
                    if excluded_fp and fp_total else "—"),
                   tone="critical" if excluded_fp else None,
                   help_text="False positives that are annotations a curator saw and the policy "
                             "dropped, rather than terms the method invented."),
    ])

    rows = [[reason, count, _EXCLUDE_HELP.get(reason, reason)]
            for reason, count in (block.get("excluded_fp_by_reason") or {}).items()]

    body = [stats]
    if rows:
        body.append(theme.table(["dropped for", "FP", "what that means"], rows, align={1: "num"}))
    if outside:
        listed = ", ".join(named)
        if outside > len(named):
            listed += f", … ({outside - len(named)} more)"
        body.append(theme.note(
            [html.B(f"{outside} report(s) "), "in this run are outside the curated cohort and are "
             "not scored: ", html.Code(listed), ". Every rate on this page is over the ",
             html.B(f"{block.get('n_reports_in_cohort')} "), "curated reports only."],
            tone="warn"))
    body.append(theme.table(
        ["policy key", "value", "what it decided"],
        [[key, _policy_value(value), _POLICY_HELP.get(key, "")]
         for key, value in (block.get("policy") or {}).items()]))

    return [theme.panel(
        f"The curated ground truth, {block.get('name')}",
        html.Div(body),
        panel_id="sc-curated",
        subtitle=f"{block.get('describe')}. Built by "
                 f"`experiments/03_setup/ground_truth` from the completed "
                 f"`apps/curation_ui` pass; the ground truth below is the applied policy, and the "
                 f"annotation table beside it carries every candidate including the dropped ones.",
        markdown=copyexport.curated_md(report),
    )]


def _policy_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "—"
    return str(value)


def _build(report: dict, mode: str, restriction=None):
    if not report:
        return theme.empty("Select a run.")

    body = [
        *([restriction] if restriction is not None else []),
        *_banner(report),
        _main_result_stats(report),
        theme.panel(
            "Where every annotated term went",
            html.Div([
                _graph("sc-fate", _fate_figure(report, mode)),
                theme.table(
                    ["fate", "n", "meaning"],
                    [[fate, (report.get("ledger") or {}).get("fate_counts", {}).get(fate, 0),
                      theme.FATE_HELP[fate]] for fate in theme.FATE_ORDER],
                    align={1: "num"}),
            ]),
            panel_id="sc-fate", graph_id="sc-fate",
            subtitle="Every annotated term of the cohort lands in one bucket. "
                     "`rejected` is an identification failure; `blocked` is a pruning failure.",
            markdown=copyexport.scorecard_md(report),
        ),
        theme.panel(
            "The two recall gaps",
            html.Div([
                _graph("sc-gap", _recall_gap_figure(report, mode)),
                theme.note(
                    "The amber band is what a different τ_accept could still recover, those "
                    "terms were reached and scored. The red band is what no τ_accept can reach: "
                    "the traversal never got there, so the run holds no score for them at all.",
                    tone="info"),
            ]),
            panel_id="sc-gap", graph_id="sc-gap",
            subtitle="Identification error and traversal error, separated.",
            markdown=copyexport.scorecard_md(report),
        ),
        *_curated_panel(report),
        theme.panel("Inference cost", _cost_stats(report), panel_id="sc-cost",
                    subtitle="Per report, at this configuration."),
        theme.panel(
            "Quality metrics",
            theme.table(
                ["metric", "micro", "macro (by report)", "macro (by term)"],
                _metric_rows(report), align={1: "num", 2: "num", 3: "num"}),
            panel_id="sc-metrics",
            subtitle="Flat P/R/F1 and the ancestor-closure and CoPHE variants, the same "
                     "functions that produce the thesis tables.",
            markdown=copyexport.scorecard_md(report),
        ),
    ]
    return html.Div(body)


def _metric_rows(report: dict) -> list[list[str]]:
    flat = report.get("flat") or {}
    hier = report.get("hierarchy") or {}
    return [
        ["precision", _fmt(flat.get("micro_precision")), _fmt(flat.get("macro_precision")),
         _fmt(flat.get("macro_term_precision"))],
        ["recall", _fmt(flat.get("micro_recall")), _fmt(flat.get("macro_recall")),
         _fmt(flat.get("macro_term_recall"))],
        ["F1", _fmt(flat.get("micro_f1")), _fmt(flat.get("macro_f1")),
         _fmt(flat.get("macro_term_f1"))],
        ["hP (closure)", _fmt(hier.get("micro_hp")), _fmt(hier.get("macro_hp")), "—"],
        ["hR (closure)", _fmt(hier.get("micro_hr")), _fmt(hier.get("macro_hr")), "—"],
        ["hF (closure)", _fmt(hier.get("micro_hf")), _fmt(hier.get("macro_hf")), "—"],
        ["CoPHE F1", _fmt(hier.get("micro_cophe_f1")), _fmt(hier.get("macro_cophe_f1")), "—"],
    ]


def _graph(graph_id: str, figure: go.Figure):
    from dash import dcc

    return dcc.Graph(id=graph_id, figure=figure, config=theme.GRAPH_CONFIG)


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("scorecard-content", "children"),
        Input("run-a", "value"),
        Input("op-a", "value"),
        Input("store-theme", "data"),
        Input("store-subset", "data"),
        active.TAB_INPUT,
    )
    def _render(cell_id, operating_point, mode, subset, tab):
        active.guard(tab, "scorecard")
        if not cell_id:
            return theme.empty("Select a run in the sidebar.")
        registry = state.get_registry()
        keep = registry.aggregate_filter(cell_id, subset)
        return _build(registry.get_report(cell_id, operating_point, report_filter=keep),
                      mode or "light", registry.subset_note(cell_id, subset))
