"""FP and FN anatomy, what kind of wrong, and how close.

Two tabs' worth of question sharing one module because they are the same measurement read from
opposite ends: the Garcia et al. taxonomy asks what each false positive was *near*, and the
false-negative taxonomy asks what was predicted near each missed annotated term. A method whose FPs are
mostly ancestors and whose FNs mostly have an ancestor predicted does not have a detection
problem, it has a granularity problem, and no threshold fixes that.

The panels that separate this from the thesis's aggregate tables:

**FP by accept score.** If the false positives sit just above τ_accept, raising it is a fix. If
they are confident, it is a trade. The histogram answers that in one look.

**FN by how close it came.** For every missed annotated term the traversal *did* reach, the actual
``accept_score`` against τ_accept. "Missed by 0.02" and "scored 0.01" call for different work.

Both tables page server-side. A tree cell can carry millions of rows and none of them may cross a
callback boundary.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html

from .. import active, copyexport, state, terms, theme

PAGE_SIZE = 25

FP_COLUMNS = [
    {"name": "report", "id": "report_id"},
    {"name": "HPO", "id": "hpo_id"},
    {"name": "label", "id": "hpo_label"},
    {"name": "bucket", "id": "fp_class"},
    {"name": "nearest ground truth", "id": "fp_reference"},
    {"name": "hops", "id": "fp_distance"},
    {"name": "depth", "id": "depth"},
    {"name": "accept", "id": "accept_score"},
    {"name": "prune", "id": "prune_score"},
    {"name": "organ", "id": "layer1_organ"},
]

FN_COLUMNS = [
    {"name": "report", "id": "report_id"},
    {"name": "HPO", "id": "hpo_id"},
    {"name": "label", "id": "hpo_label"},
    {"name": "fate", "id": "fate"},
    {"name": "predicted nearby", "id": "fn_class"},
    {"name": "depth", "id": "depth"},
    {"name": "accept", "id": "accept_score"},
    {"name": "blocked by", "id": "culprit"},
    {"name": "its prune score", "id": "culprit_prune_score"},
]

#: Added after ``label`` when a curated ground truth is loaded. One column, two different claims depending
#: on the side, which is why it is not one shared constant: on the FP side it says whether a
#: curator had *already seen and dropped* this term, the difference between a method inventing a
#: phenotype and a method finding a family-history finding the policy excludes. On the FN side it
#: says which words the missed term was annotated on, which is what turns a page of missed ids
#: into something a reader can go and look at in the report.
CURATED_COLUMN = {"FP": {"name": "curator dropped it as", "id": "curated_note"},
                  "FN": {"name": "annotated on", "id": "curated_note"}}


def _columns(side: str, curated=None) -> list[dict]:
    base = list(FP_COLUMNS if side == "FP" else FN_COLUMNS)
    if curated is None:
        return base
    return base[:3] + [CURATED_COLUMN[side]] + base[3:]


def _curated_note(curated, side: str, report_id: str, hpo_id: str) -> str:
    """The one cell, or ``""``, an empty cell means the dataset has no record, which is a claim."""
    if side == "FN":
        row = curated.find(report_id, hpo_id)
        return row.describe() if row is not None and row.in_gold else ""
    row = curated.excluded(report_id, hpo_id)
    if row is None:
        return ""
    return f"{row.exclude_reason or 'excluded'} · {row.describe()}"


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Show", className="field-label"),
                dcc.RadioItems(
                    id="err-side",
                    options=[{"label": " False positives", "value": "FP"},
                             {"label": " False negatives", "value": "FN"}],
                    value="FP", inline=True,
                ),
            ], className="field"),
            html.Div([
                html.Label("Bucket", className="field-label"),
                dcc.Dropdown(id="err-bucket", options=[], value=None, placeholder="all",
                             clearable=True),
            ], className="field"),
            html.Div([
                html.Label("Report", className="field-label"),
                dcc.Dropdown(id="err-report", options=[], value=None, placeholder="all",
                             clearable=True),
            ], className="field"),
        ], className="sidebar-section"),
        # Classification walks the ontology once per false positive and is deferred to this tab,
        # so the first open of a cell is seconds of work.
        dcc.Loading(html.Div(id="err-summary"), type="dot", delay_show=200),
        dcc.Loading(html.Div(id="err-charts"), type="dot", delay_show=200),
        theme.panel("Every error, one row each",
                    html.Div([
                        dash_table.DataTable(
                            id="err-table", columns=FP_COLUMNS, data=[],
                            page_action="custom", page_current=0, page_size=PAGE_SIZE,
                            page_count=1, sort_action="custom", sort_mode="single",
                            row_selectable="single", selected_rows=[],
                            style_as_list_view=True,
                            tooltip_data=[], tooltip_delay=300, tooltip_duration=None,
                            style_cell={"fontFamily": "ui-monospace, SFMono-Regular, monospace",
                                        "fontSize": "12px", "padding": "6px 10px",
                                        "textAlign": "left", "maxWidth": "260px",
                                        "overflow": "hidden", "textOverflow": "ellipsis"},
                        ),
                        html.Div(id="err-hint", className="stat-sub"),
                    ]),
                    panel_id="err-table-panel",
                    subtitle="Paged server-side. Select a row to open that report in the "
                             "deep-dive. With a curated ground truth loaded, each row also carries what "
                             "the curation pass recorded: the words a missed term was annotated "
                             "on, or the reason a curator had already dropped a term this method "
                             "predicted."),
    ])


def _frame(cell_id, operating_point, report_filter=None):
    """The node table *with* FP/FN classification, this is the only view that needs it."""
    registry = state.get_registry()
    bundle = registry.bundle(cell_id, operating_point, heavy=True, report_filter=report_filter)
    return (bundle or {}).get("node_table")


def _subset_rows(registry, cell_id, subset, frame):
    """Restrict the *table* to the chosen subset, whether or not the metrics are restricted.

    The row list is a browsing aid, so narrowing it changes no number and needs no opt-in, the
    same asymmetry as the deep-dive's report dropdown. When the metrics *are* restricted the frame
    is already narrowed upstream and this is a no-op.
    """
    subset = subset or {}
    chosen = registry.subset_ids(cell_id, subset.get("mode"), subset.get("cell"))
    if chosen is None or frame is None or frame.empty:
        return frame
    keep = frame.loc[frame["report_id"].astype(str).isin(set(chosen))]
    return keep if not keep.empty else frame


def _taxonomy_figure(report: dict, side: str, mode: str) -> go.Figure:
    errors = report.get("errors") or {}
    if side == "FP":
        counts = (errors.get("fp_taxonomy") or {}).get("counts") or {}
        order, colors, helps = theme.FP_ORDER, theme.fp_colors(mode), theme.FP_HELP
    else:
        counts = (errors.get("fn_taxonomy") or {}).get("counts") or {}
        order, colors, helps = theme.FN_ORDER, theme.fn_colors(mode), theme.FN_HELP

    labels = [b for b in order if counts.get(b)]
    values = [counts[b] for b in labels]
    figure = go.Figure()
    figure.add_bar(
        x=values, y=labels, orientation="h",
        marker_color=[colors[b] for b in labels],
        text=values, textposition="outside",
        customdata=[helps.get(b, "") for b in labels],
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x}<extra></extra>",
    )
    figure.update_layout(**theme.plotly_layout(
        mode, height=max(220, 44 * len(labels) + 60),
        margin={"l": 160, "r": 40, "t": 16, "b": 40},
        xaxis={"title": "count"}, yaxis={"autorange": "reversed"},
    ))
    return figure


def _score_figure(report: dict, side: str, frame, mode: str) -> go.Figure:
    """FP: are they confident? FN: how far below τ_accept did the reached ones land?"""
    palette = theme.palette(mode)
    figure = go.Figure()
    if side == "FP":
        histogram = (report.get("geometry") or {}).get("fp_by_accept_score") or {}
        edges = histogram.get("edges") or []
        if len(edges) < 2:
            return go.Figure(layout=theme.plotly_layout(mode, height=260))
        centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        figure.add_bar(x=centres, y=histogram.get("FP") or [], name="false positives",
                       marker_color=palette["critical"], opacity=0.85,
                       hovertemplate="accept≈%{x:.2f}<br>%{y} FP<extra></extra>")
        figure.add_bar(x=centres, y=histogram.get("TP") or [], name="true positives",
                       marker_color=palette["good"], opacity=0.85,
                       hovertemplate="accept≈%{x:.2f}<br>%{y} TP<extra></extra>")
        title = "accept_score"
    else:
        if frame is None or frame.empty:
            return go.Figure(layout=theme.plotly_layout(mode, height=260))
        reached = frame.loc[(frame["outcome"] == "FN") & frame["accept_score"].notna(),
                            "accept_score"].to_numpy(dtype=float)
        if reached.size == 0:
            return go.Figure(layout=theme.plotly_layout(mode, height=260))
        edges = np.linspace(0.0, 1.0, 21)
        counts, _ = np.histogram(reached, bins=edges)
        centres = (edges[:-1] + edges[1:]) / 2
        figure.add_bar(x=centres, y=counts, marker_color=palette["warning"],
                       name="missed, but reached and scored",
                       hovertemplate="accept≈%{x:.2f}<br>%{y} missed<extra></extra>")
        title = "accept_score of annotated terms that were reached but not accepted"

    figure.update_layout(**theme.plotly_layout(
        mode, height=300, barmode="overlay",
        xaxis={"title": title, "range": [0, 1]}, yaxis={"title": "nodes"}))
    return figure


def _summary(report: dict, side: str):
    errors = report.get("errors") or {}
    if side == "FP":
        taxonomy = errors.get("fp_taxonomy") or {}
        near = errors.get("near_miss") or {}
        return theme.stat_row([
            theme.stat("false positives", str(taxonomy.get("n_fp", 0)),
                       f"precision {report.get('flat', {}).get('micro_precision', 0):.3f}",
                       tone="critical"),
            theme.stat("median distance to ground truth", _fmt(near.get("median"), 1), "hops",
                       help_text="Undirected shortest path to the nearest annotated term of the "
                                 "same report."),
            theme.stat("within 2 hops", _pct(near.get("fraction_within_2")), "near misses",
                       help_text="A high share means a granularity problem, not a detection one."),
            theme.stat("hallucinated codes",
                       str((taxonomy.get("counts") or {}).get("hallucination", 0)),
                       "not in the ontology",
                       help_text="Structurally impossible for a traversal method, a non-zero "
                                 "value here means a code-handling bug."),
        ])
    taxonomy = errors.get("fn_taxonomy") or {}
    ledger = report.get("ledger") or {}
    fates = ledger.get("fate_counts") or {}
    return theme.stat_row([
        theme.stat("missed annotated terms", str(taxonomy.get("n_fn", 0)),
                   f"recall {report.get('flat', {}).get('micro_recall', 0):.3f}", tone="warning"),
        theme.stat("rejected", str(fates.get("rejected", 0)), "reached, asked, said no",
                   help_text="Identification failures, recoverable by τ_accept."),
        theme.stat("blocked", str(fates.get("blocked", 0)), "never reached",
                   tone="critical" if fates.get("blocked") else None,
                   help_text="Traversal failures, not recoverable by τ_accept at all."),
        theme.stat("nothing predicted nearby",
                   str((taxonomy.get("counts") or {}).get("nothing_near", 0)),
                   "no ancestor, descendant or sibling",
                   help_text="The system did not see this region of the ontology."),
    ])


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _report_label(sample_frame, report_id: str) -> str:
    """``1003450 · pb_worst`` when the report is in the deep-dive sample, otherwise the bare id.

    The value stays the bare id, ``_rows`` matches on it and ``_drill`` hands it to the deep-dive.
    """
    cell = sample_frame.label(report_id) if sample_frame is not None else ""
    return f"{report_id} · {cell}" if cell else str(report_id)


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def _rows(frame, side: str, bucket, report_id, sort_by, page, curated=None, view=None):
    """One page of rows, filtered and sorted server-side, plus the tooltips for that page.

    Returns ``(records, n_pages, total, tooltips)``.

    The curated note and the term tooltip are both built **after** paging, for the twenty-five rows
    on screen and no others. A tree cell's node table runs to millions of rows, and an ontology
    lookup per row over all of them to populate something nobody has scrolled to would undo the
    reason this table pages server-side in the first place. The cost is that the note cannot be
    sorted on, which is correct anyway: it is evidence, not a measurement.
    """
    if frame is None or frame.empty:
        return [], 1, 0, []
    subset = frame.loc[frame["outcome"] == side]
    bucket_column = "fp_class" if side == "FP" else "fn_class"
    if bucket:
        subset = subset.loc[subset[bucket_column] == bucket]
    if report_id:
        subset = subset.loc[subset["report_id"] == report_id]

    total = len(subset)
    if sort_by:
        column, direction = sort_by[0]["column_id"], sort_by[0]["direction"]
        if column in subset.columns:
            subset = subset.sort_values(column, ascending=direction == "asc",
                                        na_position="last", kind="stable")
    else:
        default = "fp_distance" if side == "FP" else "accept_score"
        if default in subset.columns:
            subset = subset.sort_values(default, ascending=side == "FP",
                                        na_position="last", kind="stable")

    columns = [c["id"] for c in (FP_COLUMNS if side == "FP" else FN_COLUMNS)]
    page_rows = subset.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    records = []
    for record in page_rows.to_dict("records"):
        row = {}
        for key in columns:
            value = record.get(key)
            if isinstance(value, float) and value == value:
                row[key] = round(value, 4)
            elif value != value or value is None:  # NaN / None
                row[key] = ""
            else:
                row[key] = value
        if curated is not None:
            row["curated_note"] = _curated_note(curated, side, str(record.get("report_id")),
                                                str(record.get("hpo_id")))
        records.append(row)

    # A row here names a term and says it was wrong, without ever saying what the term *is*,
    # which is the one thing needed to judge whether it was. `dash_table` has no `title`, so the
    # definition goes in its own tooltip layer, keyed by row index and column.
    tooltips = []
    if view is not None:
        for record in page_rows.to_dict("records"):
            text = terms.tooltip(view, str(record.get("hpo_id")))
            tooltips.append({"hpo_id": {"value": text, "type": "markdown"},
                             "hpo_label": {"value": text, "type": "markdown"}})
    return records, max(1, -(-total // PAGE_SIZE)), total, tooltips


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("err-bucket", "options"),
        Output("err-report", "options"),
        Input("run-a", "value"), Input("op-a", "value"), Input("err-side", "value"),
        Input("store-subset", "data"), active.TAB_INPUT,
    )
    def _filters(cell_id, operating_point, side, chosen_subset, tab):
        active.guard(tab, "errors")
        if not cell_id:
            return [], []
        registry = state.get_registry()
        frame = _frame(cell_id, operating_point,
                       registry.aggregate_filter(cell_id, chosen_subset))
        frame = _subset_rows(registry, cell_id, chosen_subset, frame)
        if frame is None or frame.empty:
            return [], []
        column = "fp_class" if side == "FP" else "fn_class"
        subset = frame.loc[frame["outcome"] == side]
        buckets = sorted({b for b in subset[column].dropna().unique()})
        sample_frame = registry.sample
        reports = sorted({str(r) for r in subset["report_id"].dropna().unique()})
        return ([{"label": b, "value": b} for b in buckets],
                [{"label": _report_label(sample_frame, r), "value": r} for r in reports])

    @app.callback(
        Output("err-summary", "children"),
        Output("err-charts", "children"),
        Input("run-a", "value"), Input("op-a", "value"),
        Input("err-side", "value"), Input("store-theme", "data"),
        Input("store-subset", "data"), active.TAB_INPUT,
    )
    def _charts(cell_id, operating_point, side, mode, subset, tab):
        active.guard(tab, "errors")
        if not cell_id:
            return theme.empty("Select a run in the sidebar."), html.Div()
        registry = state.get_registry()
        keep = registry.aggregate_filter(cell_id, subset)
        report = registry.get_report(cell_id, operating_point, heavy=True, report_filter=keep)
        if not report:
            return theme.empty("Nothing loaded."), html.Div()
        mode = mode or "light"
        frame = _frame(cell_id, operating_point, keep)
        markdown = copyexport.fp_md(report) if side == "FP" else copyexport.fn_md(report)
        charts = html.Div([
            theme.panel(
                "What kind of wrong",
                dcc.Graph(id="err-taxonomy", figure=_taxonomy_figure(report, side, mode),
                          config=theme.GRAPH_CONFIG),
                panel_id="err-taxonomy", graph_id="err-taxonomy",
                subtitle=("Each false positive against the nearest annotated term of its report."
                          if side == "FP" else
                          "Each missed annotated term against what was predicted around it."),
                markdown=markdown,
            ),
            theme.panel(
                "Confident, or borderline?",
                dcc.Graph(id="err-score", figure=_score_figure(report, side, frame, mode),
                          config=theme.GRAPH_CONFIG),
                panel_id="err-score", graph_id="err-score",
                subtitle=("If the false positives sit just above τ_accept, raising it is a fix; "
                          "if they are confident, it is a trade." if side == "FP" else
                          "How close the reached-but-rejected annotated terms came. Terms that were "
                          "never reached have no score and are not in this histogram."),
                markdown=markdown,
            ),
        ])
        restriction = registry.subset_note(cell_id, subset)
        summary = _summary(report, side)
        return (html.Div([restriction, summary]) if restriction is not None else summary), charts

    @app.callback(
        Output("err-table", "columns"),
        Output("err-table", "data"),
        Output("err-table", "page_count"),
        Output("err-hint", "children"),
        Output("err-table", "tooltip_data"),
        Input("run-a", "value"), Input("op-a", "value"), Input("err-side", "value"),
        Input("err-bucket", "value"), Input("err-report", "value"),
        Input("err-table", "page_current"), Input("err-table", "sort_by"),
        Input("store-subset", "data"), active.TAB_INPUT,
    )
    def _table(cell_id, operating_point, side, bucket, report_id, page, sort_by, subset, tab):
        active.guard(tab, "errors")
        if not cell_id:
            return _columns(side), [], 1, "", []
        registry = state.get_registry()
        cell = registry.cell(cell_id)
        curated = registry.curated_for(cell.cohort) if cell else None
        frame = _frame(cell_id, operating_point, registry.aggregate_filter(cell_id, subset))
        frame = _subset_rows(registry, cell_id, subset, frame)
        rows, pages, total, tooltips = _rows(frame, side, bucket, report_id, sort_by, page or 0,
                                             curated, registry.view)
        return _columns(side, curated), rows, pages, f"{total} rows", tooltips

    @app.callback(
        Output("store-focus", "data", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input("err-table", "selected_rows"),
        State("err-table", "data"),
        prevent_initial_call=True,
    )
    def _drill(selected, data):
        from dash.exceptions import PreventUpdate

        if not selected or not data:
            raise PreventUpdate
        row = data[selected[0]]
        return {"report_id": row.get("report_id"), "hpo_id": row.get("hpo_id")}, "deepdive"
