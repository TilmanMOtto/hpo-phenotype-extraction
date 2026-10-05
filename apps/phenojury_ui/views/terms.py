"""Term explorer, every (report, term) decision, filterable and paged.

The other views answer questions. This one answers *"show me the ones like that"*: filter to
false positives that only phi4 voted for, or to annotated terms lost with three votes out of four,
and page through them. Clicking a row sets ``store-focus``, which the deep dive picks up, so
"which report is this" is one click rather than a search.

Paging is server-side and deliberate. A GSC+ cohort at k=1 has thousands of rows. Shipping them all
to a browser over an SSH tunnel is the failure mode this app's architecture exists to avoid, so
what crosses the wire is one page.
"""

from __future__ import annotations

from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from .. import autopsy, memo, relations, theme, votes
from ..detections import popcount
from . import common

VIEW_ID = "terms"

PAGE_SIZE = 25

COLUMNS = [
    {"name": "report", "id": "report_id"},
    {"name": "term", "id": "term"},
    {"name": "outcome", "id": "outcome"},
    {"name": "votes", "id": "n_votes", "type": "numeric"},
    {"name": "decisive", "id": "n_decisive", "type": "numeric"},
    {"name": "voters", "id": "voters_text"},
    {"name": "class / fate", "id": "note"},
]


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Outcome", className="field-label"),
                dcc.Checklist(id="te-outcome", inline=True,
                              options=[{"label": " TP", "value": "TP"},
                                       {"label": " FP", "value": "FP"},
                                       {"label": " FN", "value": "FN"}],
                              value=["FP", "FN"]),
            ], className="field"),
            html.Div([
                html.Label("Voted for by (any of)", className="field-label"),
                dcc.Dropdown(id="te-models", options=[], value=[], multi=True,
                             placeholder="any model"),
            ], className="field"),
            html.Div([
                html.Label("Class / fate", className="field-label"),
                dcc.Dropdown(id="te-note", options=[], value=[], multi=True,
                             placeholder="any"),
            ], className="field"),
        ], className="filter-row"),
        html.Div(id="te-body"),
    ], className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("te-models", "options"),
        Output("te-note", "options"),
        Input("run-a", "value"),
        Input("store-config", "data"),
    )
    def filters(run_a, config):
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return [], []
        cfg = common.resolve(config, bundle)
        notes = list(relations.RELATION_ORDER) + list(autopsy.FATE_ORDER)
        return ([{"label": m, "value": m} for m in cfg["models"]],
                [{"label": n, "value": n} for n in notes])

    @app.callback(
        Output("te-body", "children"),
        Output(f"sig-{VIEW_ID}", "data"),
        Input("tabs", "value"),
        Input("run-a", "value"),
        Input("store-config", "data"),
        Input("te-outcome", "value"),
        Input("te-models", "value"),
        Input("te-note", "value"),
        State(f"sig-{VIEW_ID}", "data"),
    )
    def table(tab, run_a, config, outcomes, models, notes, previous):
        key = common.render_key(VIEW_ID, run_a, config, outcomes, models, notes)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return common.no_data(run_a), key
        cfg = common.resolve(config, bundle)
        rows = rows_for(bundle, cfg, outcomes, models, notes)
        return render_table(rows, cfg), key

    @app.callback(
        Output("te-table", "data"),
        Input("te-table", "page_current"),
        Input("te-table", "sort_by"),
        State("run-a", "value"),
        State("store-config", "data"),
        State("te-outcome", "value"),
        State("te-models", "value"),
        State("te-note", "value"),
        prevent_initial_call=True,
    )
    def page(page_current, sort_by, run_a, config, outcomes, models, notes):
        """Serve one page. The rows never leave the process. The browser gets 25 at a time."""
        bundle = common.bundle_for(run_a)
        if bundle is None:
            raise PreventUpdate
        cfg = common.resolve(config, bundle)
        rows = sort_rows(rows_for(bundle, cfg, outcomes, models, notes), sort_by)
        return page_of(rows, page_current or 0)

    @app.callback(
        Output("store-focus", "data"),
        Input("te-table", "active_cell"),
        State("te-table", "data"),
        prevent_initial_call=True,
    )
    def focus(active_cell, data):
        """Clicking any cell sends its report to the deep dive."""
        if not active_cell or not data:
            return None
        index = active_cell["row"]
        if index >= len(data):
            return None    # The page shrank under the click (a filter landed first)
        return data[index].get("report_id")


def rows_for(bundle: dict, cfg: dict, outcomes, models, notes) -> list[dict]:
    """The filtered rows for a configuration, with the unfiltered build memoised on the bundle.

    Two callbacks want them, the one that draws the panel and the one that serves a page, and
    the filters are a list scan, so only the whole-cohort build is worth remembering.
    """
    rows = memo.cached(bundle, ("terms.rows", memo.config_key(cfg)),
                       lambda: build_rows(bundle, cfg))
    return filter_rows(rows, outcomes, models, notes)


def build_rows(bundle: dict, cfg: dict) -> list[dict]:
    """Every decision in the cohort: the predicted terms and the missed annotated ones."""
    predicted = votes.predicted_sets(bundle, cfg)
    masks = bundle["masks"](cfg["min_count"])
    subset_mask = bundle["mask_of"](cfg["models"])
    k = cfg["k"] if cfg["rule"] != "plurality" else 1
    tree = bundle.get("tree")
    fates = {(r["report_id"], r["hpo_id"]): r["fate"]
             for r in autopsy.build(bundle, cfg, bundle["gold"], predicted,
                                   common.report_ids(bundle, cfg))["rows"]}

    rows: list[dict] = []
    for report_id in common.report_ids(bundle, cfg):
        gold = set(bundle["gold"].get(report_id, ()))
        pred = predicted.get(report_id, set())
        for hpo_id in sorted(pred | gold):
            mask = masks.get(report_id, {}).get(hpo_id, 0) & subset_mask
            in_pred = hpo_id in pred
            outcome = ("TP" if in_pred and hpo_id in gold else "FP" if in_pred else "FN")
            if outcome == "FP":
                note = relations.classify(tree, hpo_id, gold) if tree is not None else ""
            elif outcome == "FN":
                note = fates.get((report_id, hpo_id), "")
            else:
                note = ""
            decisive = votes.decisive_mask(mask, subset_mask, k) if in_pred else 0
            voters = [m for i, m in enumerate(cfg["models"]) if mask >> i & 1]
            rows.append({
                "report_id": report_id,
                "term": common.term(bundle, hpo_id),
                "hpo_id": hpo_id,
                "outcome": outcome,
                "n_votes": popcount(mask),
                "n_decisive": popcount(decisive),
                "voters_text": ", ".join(voters) or "—",
                "voters": voters,
                "note": note,
            })
    return rows


def filter_rows(rows: list[dict], outcomes, models, notes) -> list[dict]:
    """*rows* restricted to the chosen outcomes, jurors and notes (an empty choice keeps all)."""
    wanted_outcomes = set(outcomes or ())
    wanted_models = set(models or ())
    wanted_notes = set(notes or ())
    out = []
    for row in rows:
        if wanted_outcomes and row["outcome"] not in wanted_outcomes:
            continue
        if wanted_models and not wanted_models & set(row["voters"]):
            continue
        if wanted_notes and row["note"] not in wanted_notes:
            continue
        out.append(row)
    return out


def sort_rows(rows: list[dict], sort_by) -> list[dict]:
    """Apply the table's sort server-side. Empty ``sort_by`` keeps the natural report order."""
    if not sort_by:
        return rows
    ordered = list(rows)
    for rule in reversed(sort_by):        # last column is the weakest key
        column = rule.get("column_id")
        if column not in _SORTABLE:
            continue
        ordered.sort(key=lambda r: _sort_value(r, column),
                     reverse=rule.get("direction") == "desc")
    return ordered


def _sort_value(row: dict, column: str):
    """A total order within a column, numbers as numbers, everything else as text."""
    value = row.get(column)
    return value if isinstance(value, (int, float)) else str(value or "")


def page_of(rows: list[dict], page_current: int) -> list[dict]:
    """The ``page_current``-th slice, stripped of the fields the browser has no use for."""
    start = max(0, page_current) * PAGE_SIZE
    return [{k: v for k, v in row.items() if k not in _SERVER_ONLY}
            for row in rows[start:start + PAGE_SIZE]]


#: Kept server-side: `voters` is the filter's input and `hpo_id` is already inside `term`.
_SERVER_ONLY = ("voters", "hpo_id")

_SORTABLE = {c["id"] for c in COLUMNS}


def render_table(rows: list[dict], cfg: dict):
    """The term table of *rows*."""
    # The DataTable is rendered even when empty: the focus callback is bound to its `active_cell`,
    # and a filter that matches nothing must not remove the component that callback targets.
    if not rows:
        return theme.panel(
            "Decisions",
            html.Div([theme.empty("Nothing matches these filters."),
                      dash_table.DataTable(id="te-table", data=[], columns=COLUMNS,
                                           page_action="custom", sort_action="custom",
                                           page_size=PAGE_SIZE, page_count=1)]),
            panel_id="te-table-panel")

    # Paging and sorting are `custom`, i.e. server-side. A GSC+ cohort at k=1 has thousands of
    # decisions; `native` would serialise every one of them into the page and page a copy the
    # browser already holds. What crosses the tunnel is PAGE_SIZE rows, and the rest stays in the
    # bundle's memo where the next page comes from.
    n_pages = max(1, -(-len(rows) // PAGE_SIZE))
    return theme.panel(
        "Decisions",
        dash_table.DataTable(
            id="te-table",
            data=page_of(rows, 0),
            columns=COLUMNS,
            page_size=PAGE_SIZE,
            page_current=0,
            page_count=n_pages,
            page_action="custom",
            sort_action="custom",
            sort_mode="single",
            style_as_list_view=True,
            style_cell={"fontFamily": "inherit", "fontSize": "13px", "padding": "6px",
                        "backgroundColor": "transparent", "border": "none",
                        "textAlign": "left", "maxWidth": "320px", "whiteSpace": "normal"},
            style_header={"fontWeight": "600", "backgroundColor": "transparent",
                          "borderBottom": "1px solid #dededa"},
            style_data_conditional=[
                {"if": {"filter_query": '{outcome} = "FP"'}, "color": "#d03b3b"},
                {"if": {"filter_query": '{outcome} = "FN"'}, "color": "#c98500"},
            ],
        ),
        panel_id="te-table-panel",
        subtitle=f"{len(rows)} decision(s) at {common.describe(cfg)}, {PAGE_SIZE} per page. Click "
                 "any row to open that report in the deep dive.")
