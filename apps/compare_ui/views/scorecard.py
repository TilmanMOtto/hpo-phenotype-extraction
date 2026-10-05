"""The numbers, kept thin -- and the line between the two kinds of them.

This app is for reading reports, not for measuring methods, and the scorecard is built so that it
cannot be mistaken for the second. There are two tables:

**Per report.** TP / FP / FN and the rates for the report currently open. These are a property of
one document and are meant to be read as one: "PhenoBERT got two of five here" is a fact about this
report, and it is the number a reader wants while looking at it.

**Cohort-wide.** Micro precision, recall and F1 over the whole cohort -- 118 HCY reports, or the
114 GSC+ documents of RAG-HPO's subset -- read **straight out of**
``comparison/tables/t1_overall.csv``, matched on the bundle's own
``scored_cohort``. Not recomputed: printed from the file the results chapter prints from, which
is what guarantees the app and the thesis cannot disagree.

And one table that does not exist: a mean over the twenty documents on screen. Both
sampling frames draw into cells defined on the very outcome such a mean would report -- two of
HCY's cells *are* PhenoBERT's own best and worst scores, and every GSC+ cell is defined on which
of three methods got a term right -- so any rate over them is a rate over the selection rule.
Twenty documents is ~200 annotated pairs, which cannot separate methods whose micro-F1 differs by
0.009. The totals row here therefore prints counts and says so. It prints no F1.
"""

from __future__ import annotations

from dash import dcc, html

from apps.compare_ui import theme
from apps.compare_ui.views import common

VIEW_ID = "scorecard"


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="scorecard-body"),
        dcc.Store(id="sig-scorecard", data=None),
    ])


def render(registry, report_id):
    """Precision, recall and F1 of every method on *report_id*, beside the cohort values."""
    return html.Div([
        common.guard(VIEW_ID, "per-report", _per_report, registry, report_id),
        common.guard(VIEW_ID, "cohort", _cohort, registry),
        common.guard(VIEW_ID, "coverage", _coverage, registry),
    ])


# -- one report --------------------------------------------------------------

def _per_report(registry, report_id):
    bundle = registry.bundle(report_id) if report_id else None
    if bundle is None:
        return theme.empty("Pick a report.")

    headers = ["method", "TP", "FP", "FN", "precision", "recall", "F1"]
    rows = []
    for method in registry.methods():
        key = method["key"]
        score = (bundle.get("metrics") or {}).get(key) or {}
        rows.append([
            html.Span(method.get("label") or key, title=method.get("evidence") or ""),
            score.get("tp", 0), score.get("fp", 0), score.get("fn", 0),
            common.fmt(score.get("precision")), common.fmt(score.get("recall")),
            common.fmt(score.get("f1")),
        ])
    return theme.panel(
        "This {}: {}".format(registry.unit(1), bundle["report_id"]),
        html.Div([
            common.table(headers, rows),
            theme.note("One document. A rate over a single document is not an estimate of "
                       "anything, it is a description of this page.", "info"),
        ]),
        panel_id="scorecard-report",
        subtitle="{} annotated term(s), cell “{}”".format(
            len(bundle.get("gold") or ()), bundle.get("cell") or "—"),
        markdown=common.md_table(headers, rows))


# -- the cohort --------------------------------------------------------------

def _cohort(registry):
    rows_in = registry.comparison_rows()
    heading = "The whole {} cohort".format(registry.label())
    if not rows_in:
        return theme.panel(
            heading, theme.note(
                "No row for cohort {!r} in {}. The quotable numbers live in the comparison's own "
                "output; this app does not recompute them.".format(
                    registry.scored_cohort(), registry.comparison_path()), "warn"),
            panel_id="scorecard-cohort")

    by_key = {str(r.get("method")): r for r in rows_in}
    headers = ["method", "n reports", "TP", "FP", "FN", "micro P", "micro R", "micro F1",
               "95% CI on F1"]
    rows = []
    for method in registry.methods():
        row = by_key.get(method["key"])
        if row is None:
            rows.append([method.get("label") or method["key"], "—", "—", "—", "—", "—", "—", "—",
                         "not in the comparison table"])
            continue
        rows.append([
            html.Span(row.get("label") or method.get("label"),
                      title=row.get("operating_point") or ""),
            row.get("n_reports"), row.get("tp"), row.get("fp"), row.get("fn"),
            common.fmt(row.get("micro_precision")), common.fmt(row.get("micro_recall")),
            common.fmt(row.get("micro_f1")),
            "[{}, {}]".format(common.fmt(row.get("micro_f1_lo")),
                              common.fmt(row.get("micro_f1_hi"))),
        ])

    notes = [row.get("note") for row in rows_in if row.get("note")]
    body = [common.table(headers, rows),
            theme.note("Read verbatim from the comparison's t1_overall.csv. These are the numbers the "
                       "results chapter prints; nothing here recomputes them.", "info")]
    for note in sorted(set(notes)):
        body.append(theme.note(note, "warn"))

    n = _n_reports(rows_in)
    return theme.panel(
        heading, html.Div(body), panel_id="scorecard-cohort",
        subtitle="micro-averaged over {} · scored as “{}” · the only quotable rates".format(
            "{} {}".format(n, registry.unit(n)) if n else "the full cohort",
            registry.scored_cohort()),
                       markdown=common.md_table(headers, rows))


# -- what is on screen -------------------------------------------------------

def _coverage(registry):
    """Counts over the drawn reports, and no rates. See the module docstring."""
    rows_by_id = registry.rows()
    headers = ["method", "TP", "FP", "FN"]
    totals = {}
    for row in rows_by_id.values():
        for key, score in (row.get("metrics") or {}).items():
            bucket = totals.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
            for field in bucket:
                bucket[field] += int(score.get(field) or 0)

    rows = [[method.get("label") or method["key"],
             totals.get(method["key"], {}).get("tp", 0),
             totals.get(method["key"], {}).get("fp", 0),
             totals.get(method["key"], {}).get("fn", 0)]
            for method in registry.methods()]

    skipped = registry.skipped()
    body = [
        common.table(headers, rows),
        theme.note("Counts only, over the {} {} in this sample. No rate is shown because the "
                   "sample is purposive: its cells are defined on the outcome, so a rate here "
                   "measures the selection rule. Use the cohort table above.".format(
                       len(rows_by_id), registry.unit(len(rows_by_id))), "warn"),
    ]
    if skipped:
        body.append(html.Div("Documents the builder did not draw", className="cmp-subhead"))
        body.append(common.table(["document", "why"],
                                 [[rid, why] for rid, why in sorted(skipped.items())]))

    return theme.panel("The {} {} on screen".format(len(rows_by_id),
                                                    registry.unit(len(rows_by_id))),
                       html.Div(body), panel_id="scorecard-coverage",
                       subtitle="the deep-dive sample, mechanisms, not measurements",
                       markdown=common.md_table(headers, rows))


def _n_reports(rows) -> int:
    """The cohort size ``comparison`` reports, when its rows agree about it. ``0`` when not.

    They can legitimately disagree: a method with no prediction file for this cohort
    contributes a row with an empty count. Printing the maximum would be a guess dressed as a
    fact, so a disagreement prints nothing instead.
    """
    seen = {int(row["n_reports"]) for row in rows
            if str(row.get("n_reports") or "").strip().isdigit()}
    return seen.pop() if len(seen) == 1 else 0


# -- callbacks ---------------------------------------------------------------

def register(app) -> None:
    """Register this view's callbacks on *app*."""
    from dash import Input, Output, State
    from dash.exceptions import PreventUpdate

    from apps.compare_ui import state

    @app.callback(
        Output("scorecard-body", "children"),
        Output("sig-scorecard", "data"),
        Input("tabs", "value"),
        Input("cohort-picker", "value"),
        Input("report-picker", "value"),
        Input("store-theme", "data"),
        State("sig-scorecard", "data"),
    )
    def render_view(tab, cohort, report_id, mode, previous):
        key = common.render_key(VIEW_ID, cohort, report_id, mode)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        return render(state.registry(cohort), report_id), key
