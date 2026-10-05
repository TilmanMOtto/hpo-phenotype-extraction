"""vs PhenoBERT, the ensemble against the baseline it has to beat.

The PhenoBERT baseline runs PhenoBERT over the raw report. The Free Listing generation run runs the same grounder over what eight SLMs
*wrote* about each sentence. Same ontology linker, opposite inputs, so the difference between them
is attributable to the generation step and nothing else, which is the only reason the comparison
is worth making at this level of detail.

Four questions, in the order they get asked:

**Is it better?**       micro and macro P/R/F1 side by side, at the current configuration.
**Where does it win?**  the agreement split, every annotated pair is found by both, by one, or by
                        neither, and the same for the false positives. Two systems with the same F1
                        and disjoint hits are a very different finding from two that agree.
**Which reports?**      one row per report, sorted by ΔF1, clickable through to the deep dive.
**Which terms?**        one row per HPO term across the cohort, so a systematic blind spot on either
                        side shows up as a term, not as a scatter of individual misses.

Scoring is over the **intersection** of the two report-id sets. A rerun of one cohort at a different
size is a real situation, and scoring one method on reports the other never saw would show a
difference that is entirely bookkeeping. The count is on screen either way.

The whole page degrades to a single note when no PhenoBERT baseline run was found: it names the paths that
were searched, because "not found" a reader can act on and "unavailable" they cannot.
"""

from __future__ import annotations

from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from .. import memo, theme, votes
from .. import frame as frame_mod
from . import common

VIEW_ID = "pbcompare"

PAGE_SIZE = 25

#: Which side a term went to. Fixed order: it is the severity axis these tables are read along.
SIDE_ORDER = ("both", "ensemble_only", "phenobert_only", "neither")

SIDE_LABEL = {
    "both": "both found it",
    "ensemble_only": "ensemble only",
    "phenobert_only": "PhenoBERT only",
    "neither": "neither found it",
}

REPORT_COLUMNS = [
    {"name": "report", "id": "report_id"},
    {"name": "annotated", "id": "n_gold", "type": "numeric"},
    {"name": "ens TP", "id": "ens_tp", "type": "numeric"},
    {"name": "ens FP", "id": "ens_fp", "type": "numeric"},
    {"name": "ens FN", "id": "ens_fn", "type": "numeric"},
    {"name": "ens F1", "id": "ens_f1", "type": "numeric"},
    {"name": "pb TP", "id": "pb_tp", "type": "numeric"},
    {"name": "pb FP", "id": "pb_fp", "type": "numeric"},
    {"name": "pb FN", "id": "pb_fn", "type": "numeric"},
    {"name": "pb F1", "id": "pb_f1", "type": "numeric"},
    {"name": "ΔF1", "id": "delta_f1", "type": "numeric"},
    {"name": "only the ensemble got", "id": "ens_only_text"},
    {"name": "only PhenoBERT got", "id": "pb_only_text"},
]

TERM_COLUMNS = [
    {"name": "term", "id": "term"},
    {"name": "annotated in", "id": "n_gold", "type": "numeric"},
    {"name": "ens hits", "id": "ens_hits", "type": "numeric"},
    {"name": "pb hits", "id": "pb_hits", "type": "numeric"},
    {"name": "ens only", "id": "ens_only", "type": "numeric"},
    {"name": "pb only", "id": "pb_only", "type": "numeric"},
    {"name": "both missed", "id": "both_missed", "type": "numeric"},
    {"name": "ens FP", "id": "ens_fp", "type": "numeric"},
    {"name": "pb FP", "id": "pb_fp", "type": "numeric"},
]

#: Kept in the process: the term lists are already rendered into the ``*_text`` columns, and
#: ``both``/``neither`` are the raw counters the displayed columns are derived from.
_REPORT_SERVER_ONLY = ("ens_only", "pb_only")
_TERM_SERVER_ONLY = ("hpo_id", "both", "neither")


def layout():
    """The Dash layout of this view."""
    return html.Div([
        html.Div([
            html.Label("Sort reports by", className="field-label"),
            dcc.RadioItems(
                id="pb-sort", inline=True, value="delta",
                options=[{"label": " biggest PhenoBERT win", "value": "delta"},
                         {"label": " biggest ensemble win", "value": "-delta"},
                         {"label": " report number", "value": "report"}],
            ),
        ], className="field"),
        html.Div(id="pb-body"),
    ], className="view")


def register(app) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("pb-body", "children"),
        Output(f"sig-{VIEW_ID}", "data"),
        Input("tabs", "value"),
        Input("run-a", "value"),
        Input("store-config", "data"),
        Input("store-theme", "data"),
        Input("pb-sort", "value"),
        State(f"sig-{VIEW_ID}", "data"),
    )
    def body(tab, run_a, config, mode, sort_key, previous):
        key = common.render_key(VIEW_ID, run_a, config, mode, sort_key)
        if not common.should_render(VIEW_ID, tab, key, previous):
            raise PreventUpdate
        bundle = common.bundle_for(run_a)
        if bundle is None:
            return common.no_data(run_a), key
        cfg = common.resolve(config, bundle)
        return render_all(bundle, cfg, mode or "light", sort_key or "delta"), key

    @app.callback(
        Output("pb-report-table", "data"),
        Input("pb-report-table", "page_current"),
        Input("pb-report-table", "sort_by"),
        State("run-a", "value"),
        State("store-config", "data"),
        State("pb-sort", "value"),
        prevent_initial_call=True,
    )
    def page_reports(page_current, sort_by, run_a, config, sort_key):
        bundle = common.bundle_for(run_a)
        if bundle is None or not bundle.get("pb_standalone"):
            # No rows rather than PreventUpdate: without a baseline the table is not on the page at
            # all, so there is nothing to protect, and "this callback never runs" is a report the
            # selftest is right to make about a callback that is simply unreachable.
            return []
        cfg = common.resolve(config, bundle)
        rows = order_reports(report_rows(bundle, cfg), sort_key or "delta")
        return _page(_sorted(rows, sort_by, REPORT_COLUMNS), page_current or 0,
                     _REPORT_SERVER_ONLY)

    @app.callback(
        Output("pb-term-table", "data"),
        Input("pb-term-table", "page_current"),
        Input("pb-term-table", "sort_by"),
        State("run-a", "value"),
        State("store-config", "data"),
        prevent_initial_call=True,
    )
    def page_terms(page_current, sort_by, run_a, config):
        bundle = common.bundle_for(run_a)
        if bundle is None or not bundle.get("pb_standalone"):
            # No rows, not PreventUpdate: without a baseline the table is not on the page at
            # all, so there is nothing to protect, and "this callback never runs" is a report the
            # selftest is right to make about a callback that is simply unreachable.
            return []
        cfg = common.resolve(config, bundle)
        return _page(_sorted(term_rows(bundle, cfg), sort_by, TERM_COLUMNS), page_current or 0,
                     _TERM_SERVER_ONLY)

    @app.callback(
        Output("store-focus", "data", allow_duplicate=True),
        Input("pb-report-table", "active_cell"),
        State("pb-report-table", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def focus(active_cell, data):
        """Clicking a row opens that report in the deep dive, the same contract as the term explorer."""
        if not active_cell or not data:
            return None
        index = active_cell["row"]
        if index >= len(data):
            return None       # The page moved under the click
        return data[index].get("report_id")


# ──────────────────────────────────────────────────────────────────────────────
# The comparison itself
# ──────────────────────────────────────────────────────────────────────────────
def shared_reports(bundle: dict, cfg: dict | None = None) -> list[str]:
    """Reports both experiments processed, in the ensemble's own order.

    Intersected with the deep-dive subset when one is active, so this page scores the same reports
    as every other. *cfg* is optional because the intersection is the interesting part and a caller
    that has no configuration (a test asking which reports the two runs share) should not have to
    invent one.
    """
    pb = bundle.get("pb_standalone")
    if not pb:
        return []
    theirs = set(pb["report_ids"])
    return [r for r in common.report_ids(bundle, cfg) if r in theirs]


def compare(bundle: dict, cfg: dict) -> dict:
    """The whole comparison, memoised: pairs, per-report rows, per-term rows and the totals.

    One walk over the shared reports produces all four, and every panel on the page reads from it, computing them separately would be four walks answering the same question.
    """
    return memo.cached(bundle, ("pbcompare", memo.config_key(cfg)),
                       lambda: _compare(bundle, cfg))


def _compare(bundle: dict, cfg: dict) -> dict:
    from hpo_extraction.evaluation.set_metrics import calc_metric

    pb = bundle["pb_standalone"]
    ens_sets = votes.predicted_sets(bundle, cfg)
    reports = shared_reports(bundle, cfg)

    gold_split = {side: 0 for side in SIDE_ORDER}
    fp_split = {"both": 0, "ensemble_only": 0, "phenobert_only": 0}
    per_report: list[dict] = []
    per_term: dict[str, dict] = {}

    for report_id in reports:
        gold = set(bundle["gold"].get(report_id, ()))
        ens = set(ens_sets.get(report_id, ()))
        pbp = set(pb["predicted"].get(report_id, ()))

        for hpo_id in gold:
            side = ("both" if hpo_id in ens and hpo_id in pbp else
                    "ensemble_only" if hpo_id in ens else
                    "phenobert_only" if hpo_id in pbp else "neither")
            gold_split[side] += 1
            _term(per_term, hpo_id)["n_gold"] += 1
            _term(per_term, hpo_id)[_TERM_FIELD[side]] += 1

        for hpo_id in (ens | pbp) - gold:
            side = ("both" if hpo_id in ens and hpo_id in pbp else
                    "ensemble_only" if hpo_id in ens else "phenobert_only")
            fp_split[side] += 1
            row = _term(per_term, hpo_id)
            row["ens_fp"] += int(hpo_id in ens)
            row["pb_fp"] += int(hpo_id in pbp)

        ens_p, ens_r, ens_f1 = calc_metric(gold, ens) if gold else (None, None, None)
        pb_p, pb_r, pb_f1 = calc_metric(gold, pbp) if gold else (None, None, None)
        ens_only = sorted((gold & ens) - pbp)
        pb_only = sorted((gold & pbp) - ens)
        per_report.append({
            "report_id": report_id,
            "n_gold": len(gold),
            "ens_tp": len(gold & ens), "ens_fp": len(ens - gold), "ens_fn": len(gold - ens),
            "pb_tp": len(gold & pbp), "pb_fp": len(pbp - gold), "pb_fn": len(gold - pbp),
            "ens_f1": _round(ens_f1), "pb_f1": _round(pb_f1),
            "delta_f1": _round(pb_f1 - ens_f1) if gold else None,
            "ens_only": ens_only, "pb_only": pb_only,
            "ens_only_text": ", ".join(_short(bundle, h) for h in ens_only) or "—",
            "pb_only_text": ", ".join(_short(bundle, h) for h in pb_only) or "—",
        })

    terms = []
    for hpo_id, row in per_term.items():
        row = dict(row, hpo_id=hpo_id, term=common.term(bundle, hpo_id))
        row["ens_hits"] = row["both"] + row["ens_only"]
        row["pb_hits"] = row["both"] + row["pb_only"]
        row["both_missed"] = row["neither"]
        terms.append(row)
    terms.sort(key=lambda r: (-abs(r["ens_hits"] - r["pb_hits"]), -r["n_gold"], r["hpo_id"]))

    return {
        "reports": reports,
        "gold_split": gold_split,
        "fp_split": fp_split,
        "per_report": per_report,
        "per_term": terms,
        "totals": _totals(bundle, cfg, reports, ens_sets, pb),
    }


#: Which per-term counter each agreement side increments.
_TERM_FIELD = {"both": "both", "ensemble_only": "ens_only",
               "phenobert_only": "pb_only", "neither": "neither"}


def _term(per_term: dict, hpo_id: str) -> dict:
    return per_term.setdefault(hpo_id, {
        "n_gold": 0, "both": 0, "ens_only": 0, "pb_only": 0, "neither": 0,
        "ens_fp": 0, "pb_fp": 0,
    })


def _totals(bundle: dict, cfg: dict, reports, ens_sets, pb) -> dict:
    """Micro and macro P/R/F1 for both methods over the shared reports.

    Through ``votes.score``, which is the driver's own ``_safe_micro_macro``: every denominator is
    guarded, and a configuration that predicts nothing at all, a high k on a small ensemble, is a
    real configuration, not a crash. Using the same scorer as every other panel also means
    the ensemble column here cannot disagree with the scorecard.
    """
    if not reports:
        return {"ensemble": {}, "phenobert": {}}
    return {
        "ensemble": votes.score(bundle["gold"], ens_sets, reports),
        "phenobert": votes.score(bundle["gold"], pb["predicted"], reports),
    }


def _round(x):
    return None if x is None else round(float(x), 4)


def _short(bundle: dict, hpo_id: str) -> str:
    """A term short enough for a table cell: the label where there is one, else the id."""
    tree = bundle.get("tree")
    if tree is None:
        return hpo_id
    from .. import relations

    name = relations.label(tree, hpo_id)
    return hpo_id if name == hpo_id else name


def report_rows(bundle: dict, cfg: dict) -> list[dict]:
    """Per-report comparison rows of the jury against PhenoBERT alone."""
    return compare(bundle, cfg)["per_report"]


def term_rows(bundle: dict, cfg: dict) -> list[dict]:
    """Per-term comparison rows of the jury against PhenoBERT alone."""
    return compare(bundle, cfg)["per_term"]


def order_reports(rows: list[dict], sort_key: str) -> list[dict]:
    """The reader's chosen default order, before any column sort they apply on top."""
    if sort_key == "report":
        return sorted(rows, key=lambda r: _report_key(r["report_id"]))
    reverse = sort_key == "delta"
    return sorted(rows, key=lambda r: (r["delta_f1"] is None,
                                       -(r["delta_f1"] or 0) if reverse else (r["delta_f1"] or 0)))


def _report_key(report_id: str):
    """Numeric where the id is a number. GSC ids are variable-length digit strings, so a plain
    lexicographic sort puts ``10051003`` before ``1003450``."""
    return (0, int(report_id), "") if report_id.isdigit() else (1, 0, report_id)


def _sorted(rows: list[dict], sort_by, columns) -> list[dict]:
    """Apply the DataTable's own sort server-side."""
    if not sort_by:
        return rows
    sortable = {c["id"] for c in columns}
    ordered = list(rows)
    for rule in reversed(sort_by):
        column = rule.get("column_id")
        if column not in sortable:
            continue
        ordered.sort(key=lambda r: _sort_value(r, column),
                     reverse=rule.get("direction") == "desc")
    return ordered


def _sort_value(row: dict, column: str):
    value = row.get(column)
    if value is None:
        return float("-inf")
    return value if isinstance(value, (int, float)) else str(value)


def _page(rows: list[dict], page_current: int, server_only) -> list[dict]:
    start = max(0, page_current) * PAGE_SIZE
    return [{k: v for k, v in row.items() if k not in server_only}
            for row in rows[start:start + PAGE_SIZE]]


# ──────────────────────────────────────────────────────────────────────────────
# rendering
# ──────────────────────────────────────────────────────────────────────────────
def render_all(bundle: dict, cfg: dict, mode: str, sort_key: str = "delta"):
    """Every panel of the PhenoBERT comparison."""
    if not bundle.get("pb_standalone"):
        return render_missing(bundle)
    return html.Div([
        render_totals(bundle, cfg),
        render_agreement(bundle, cfg, mode),
        render_reports(bundle, cfg, sort_key),
        render_terms(bundle, cfg),
    ])


def render_missing(bundle: dict):
    """No baseline. Say where it was looked for, that is the only actionable part."""
    searched = bundle.get("pb_searched") or []
    return theme.panel(
        "No PhenoBERT baseline for this cohort",
        html.Div([
            theme.note("This page compares the ensemble against baseline_phenobert, which runs "
                       "PhenoBERT over the raw report. No such run was found for this cohort, so "
                       "there is nothing to compare against. Every other tab is unaffected.",
                       "warn"),
            html.Div("Looked in:", className="stat-label", style={"marginTop": "10px"}),
            html.Ul([html.Li(html.Code(p)) for p in searched]
                    or [html.Li("nowhere, this cohort has no run directory")]),
            html.Div("Set the baseline_phenobert folder under Data source in the sidebar and press Load.",
                     className="stat-sub"),
        ]),
        panel_id="pb-missing")


def render_totals(bundle: dict, cfg: dict):
    """Micro and macro, both methods, one row of tiles each."""
    result = compare(bundle, cfg)
    totals = result["totals"]
    ens = totals.get("ensemble") or {}
    pb = totals.get("phenobert") or {}
    if not result["reports"]:
        return common.empty_panel(
            "The two runs share no report, this baseline is a different cohort.",
            "pb-totals", "Head to head")

    tiles = []
    for name, key in (("micro F1", "micro_f1"), ("micro precision", "micro_precision"),
                      ("micro recall", "micro_recall"), ("macro F1", "macro_f1")):
        a, b = ens.get(key), pb.get(key)
        delta = None if a is None or b is None else b - a
        tiles.append(theme.stat(
            name, f"{common.num(a)} → {common.num(b)}",
            f"PhenoBERT {common.signed(delta)}",
            tone=_tone(delta),
            help_text=f"ensemble {common.num(a)} at {common.describe(cfg)}; "
                      f"PhenoBERT standalone {common.num(b)}. The arrow reads "
                      "ensemble → PhenoBERT."))

    n_reports = len(result["reports"])
    # over the *whole* cohort, not the subset: this counts what each run processed,
    # which is a fact about the two runs and does not change because a filter is on.
    only_ens = len(set(bundle["report_ids"]) - set(bundle["pb_standalone"]["report_ids"]))
    only_pb = len(set(bundle["pb_standalone"]["report_ids"]) - set(bundle["report_ids"]))
    scoped = ("" if (cfg.get("subset") or frame_mod.ALL) == frame_mod.ALL
              else f", within the {common.subset_label(cfg.get('subset'))}")
    subtitle = (f"{n_reports} report(s) scored, the ones both runs processed{scoped}. Ensemble at "
                f"{common.describe(cfg)}.")
    if only_ens or only_pb:
        subtitle += (f" {only_ens} report(s) are only in the ensemble run and {only_pb} only in "
                     "the baseline; neither is scored here.")

    return theme.panel(
        "Head to head", theme.stat_row(tiles), panel_id="pb-totals", subtitle=subtitle,
        markdown=common.md_table(
            ["metric", "ensemble", "PhenoBERT", "Δ"],
            [[k, common.num(ens.get(k)), common.num(pb.get(k)),
              common.signed(None if ens.get(k) is None or pb.get(k) is None
                            else pb[k] - ens[k])]
             for k in ("micro_precision", "micro_recall", "micro_f1",
                       "macro_precision", "macro_recall", "macro_f1")]))


def _tone(delta):
    """Red when the baseline is ahead, green when the ensemble is, this app's subject is the
    ensemble, so 'PhenoBERT wins' is the result that needs explaining."""
    if delta is None or abs(delta) < 5e-4:
        return None
    return "critical" if delta > 0 else "good"


def render_agreement(bundle: dict, cfg: dict, mode: str):
    """Who found what, the split two equal F1 scores can hide."""
    result = compare(bundle, cfg)
    gold_split = result["gold_split"]
    fp_split = result["fp_split"]
    n_gold = sum(gold_split.values()) or 1

    gold_rows = [[SIDE_LABEL[side], gold_split[side], common.pct(gold_split[side] / n_gold)]
                 for side in SIDE_ORDER]
    n_fp = sum(fp_split.values()) or 1
    fp_rows = [[SIDE_LABEL[side], fp_split[side], common.pct(fp_split[side] / n_fp)]
               for side in ("both", "ensemble_only", "phenobert_only")]

    return theme.panel(
        "Who found what",
        html.Div([
            html.Div("annotated (report, term) pairs", className="stat-label"),
            common.table(["found by", "pairs", "share"], gold_rows),
            html.Div("false positives", className="stat-label", style={"marginTop": "12px"}),
            common.table(["predicted by", "pairs", "share"], fp_rows),
            theme.note("The two 'only' rows are the whole argument for an ensemble of the two: "
                       "terms one method finds and the other never does are recall an "
                       "either-way union would gain, and precision it would lose. 'Neither found "
                       "it' is the joint ceiling, no combination of these two reaches those.",
                       "info"),
        ]),
        panel_id="pb-agreement",
        subtitle=f"{sum(gold_split.values())} annotated pair(s) and {sum(fp_split.values())} "
                 f"false positive(s) over {len(result['reports'])} shared report(s)",
        markdown=common.md_table(["found by", "pairs"],
                                 [[SIDE_LABEL[s], gold_split[s]] for s in SIDE_ORDER]))


def render_reports(bundle: dict, cfg: dict, sort_key: str = "delta"):
    """The per-report table, sorted by *sort_key*."""
    rows = order_reports(report_rows(bundle, cfg), sort_key)
    return theme.panel(
        "Report by report",
        dash_table.DataTable(
            id="pb-report-table",
            data=_page(rows, 0, _REPORT_SERVER_ONLY),
            columns=REPORT_COLUMNS,
            page_size=PAGE_SIZE,
            page_current=0,
            page_count=max(1, -(-len(rows) // PAGE_SIZE)),
            page_action="custom",
            sort_action="custom",
            sort_mode="single",
            style_as_list_view=True,
            style_cell={"fontFamily": "inherit", "fontSize": "13px", "padding": "6px",
                        "backgroundColor": "transparent", "border": "none",
                        "textAlign": "left", "maxWidth": "260px", "whiteSpace": "normal"},
            style_header={"fontWeight": "600", "backgroundColor": "transparent",
                          "borderBottom": "1px solid #dededa"},
            style_data_conditional=[
                {"if": {"filter_query": "{delta_f1} > 0", "column_id": "delta_f1"},
                 "color": "#d03b3b"},
                {"if": {"filter_query": "{delta_f1} < 0", "column_id": "delta_f1"},
                 "color": "#0ca30c"},
            ],
        ),
        panel_id="pb-report-table-panel",
        subtitle=f"{len(rows)} report(s), {PAGE_SIZE} per page. ΔF1 is PhenoBERT minus the "
                 "ensemble, so a positive number is a report the baseline wins. Click any row to "
                 "open it in the deep dive.")


def render_terms(bundle: dict, cfg: dict):
    """Per term across the cohort, where a blind spot is systematic, not incidental."""
    rows = term_rows(bundle, cfg)
    tree = bundle.get("tree")
    tooltips = None
    if tree is not None:
        page = rows[:PAGE_SIZE]
        tooltips = [{"term": {"value": common.hpo_tooltip(bundle, r["hpo_id"]),
                              "type": "text"}} for r in page]
    return theme.panel(
        "Term by term",
        dash_table.DataTable(
            id="pb-term-table",
            data=_page(rows, 0, _TERM_SERVER_ONLY),
            columns=TERM_COLUMNS,
            tooltip_data=tooltips,
            tooltip_duration=None,
            page_size=PAGE_SIZE,
            page_current=0,
            page_count=max(1, -(-len(rows) // PAGE_SIZE)),
            page_action="custom",
            sort_action="custom",
            sort_mode="single",
            style_as_list_view=True,
            style_cell={"fontFamily": "inherit", "fontSize": "13px", "padding": "6px",
                        "backgroundColor": "transparent", "border": "none",
                        "textAlign": "left", "maxWidth": "300px", "whiteSpace": "normal"},
            style_header={"fontWeight": "600", "backgroundColor": "transparent",
                          "borderBottom": "1px solid #dededa"},
        ),
        panel_id="pb-term-table-panel",
        subtitle=f"{len(rows)} term(s) in play, ordered by how far the two methods disagree on "
                 "them. 'annotated in' counts reports where the term is ground truth; 'ens only' and "
                 "'pb only' are the annotated occurrences one method recovered.",
        markdown=common.md_table(
            ["term", "annotated in", "ens hits", "pb hits", "ens only", "pb only"],
            [[r["term"], r["n_gold"], r["ens_hits"], r["pb_hits"], r["ens_only"], r["pb_only"]]
             for r in rows[:60]]))
