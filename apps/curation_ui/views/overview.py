"""The cohort at a glance, progress, disagreement, and the two exports.

Three questions this answers that the per-patient screen cannot:

* **how far along is this?**, confirmed patients, undecided terms, open suggestions;
* **where do the annotation files actually disagree?**, per patient, the terms not every loaded
  source carries, sorted worst first, which is the queue this curation pass exists to work through;
* **what comes out of it?**, the curated ground truth, written in the two-column shape every cluster
  script's ``ground_truth_path=`` already reads, so the result of the pass is a drop-in file rather
  than something needing an adapter, plus the two difficulty-label files, which make a stratified
  recall table a ``merge`` and a ``groupby``, not a re-read of every report.

The table is built from counts, never from per-patient view models: opening this tab must not warm
sixteen patients into the LRU and evict the one being worked on.
"""

from __future__ import annotations

import os

import dash
from dash import Input, Output, State, dash_table, html
from dash.exceptions import PreventUpdate

from hpo_extraction.curation import labels as vocab
from .. import theme
from hpo_extraction.curation import store
from . import common

VIEW_ID = "overview"

#: The columns either side of the per-source counts. The source columns themselves depend on which
#: files were actually loaded, so the header is built per registry, not written out here,
#: a fixed ``raw``/``marc2`` header outlived the files it named once already.
LEAD_COLUMNS = [("patient_id", "Patient"), ("n_segments", "Segs")]
TAIL_COLUMNS = [
    ("n_existing", "Rule on"),
    ("n_disagree", "≠"),
    ("n_phenobert", "PB"),
    ("n_suggested", "Open"),
    ("n_decided", "Decided"),
    ("n_in_gold", "In ground truth"),
    ("n_delete_suggested", "Del?"),
    ("n_comments", "Notes"),
    ("difficulty", "Grade"),
    ("n_labelled", "Labelled"),
    ("confirmed", "✓"),
]


def columns(registry) -> list[tuple[str, str]]:
    """``[(key, header), …]``, one count column per annotation source that loaded."""
    sources = registry.gold_sources if registry is not None else []
    return LEAD_COLUMNS + [(f"n_{source}", source) for source in sources] + TAIL_COLUMNS


def layout() -> html.Div:
    """The Dash layout of this view."""
    return html.Div([
        html.Div(id="overview-stats"),
        html.Div(id="overview-difficulty"),
        theme.card(
            "Export",
            html.Div([
                html.Div([
                    html.Button("Write curated ground truth", id="ov-export", className="primary-btn",
                                n_clicks=0),
                    html.Button("Write difficulty labels", id="ov-export-labels",
                                className="copy-btn", n_clicks=0),
                ], className="row-actions"),
                html.Div(id="ov-export-status"),
            ]),
            subtitle="hcy_ground_truth_curated.csv, patient_id, hpo_codes, one line per patient "
                     "including the empty ones. curation_labels.csv, annotation labels, long "
                     "form, joined on (patient_id, hpo_code). curation_report_labels.csv, one "
                     "wide row per report, joined on patient_id. curation_comments.csv, one row "
                     "per comment. All are rewritten on every action anyway; these buttons are "
                     "for writing them now.",
        ),
        html.Div(id="overview-table"),
    ], id="overview-panel")


def register(app: dash.Dash) -> None:
    """Register this view's callbacks on *app*."""
    @app.callback(
        Output("overview-stats", "children"),
        Output("overview-table", "children"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
        Input("tabs", "value"),
    )
    def draw(_dirty, mode, tab):
        if tab != "overview":
            raise PreventUpdate
        return (common.guard("Cohort stats", render_stats, mode or "light"),
                common.guard("Cohort table", render_table))

    @app.callback(
        Output("overview-difficulty", "children"),
        Input("store-dirty", "data"),
        Input("store-theme", "data"),
        Input("tabs", "value"),
    )
    def draw_difficulty(_dirty, mode, tab):
        if tab != "overview":
            raise PreventUpdate
        return common.guard("Difficulty", render_difficulty, mode or "light")

    @app.callback(
        Output("ov-export-status", "children"),
        Input("ov-export", "n_clicks"),
        Input("ov-export-labels", "n_clicks"),
        prevent_initial_call=True,
    )
    def export(gold_clicks, label_clicks):
        registry = common.registry_or_none()
        which = dash.ctx.triggered_id
        pressed = gold_clicks if which == "ov-export" else label_clicks
        if not pressed or registry is None:
            raise PreventUpdate
        try:
            if which == "ov-export-labels":
                result = registry.write_labels()
                return theme.note(
                    f"Wrote {result['annotation']['n_rows']} annotation label row(s), "
                    f"{result['report']['n_rows']} report row(s) "
                    f"({result['report']['n_labelled']} characterised) and "
                    f"{result['comments']['n_rows']} comment(s) to "
                    f"{os.path.dirname(result['report']['path'])}", "info")
            result = registry.export_gold()
        except OSError as exc:
            return theme.note(f"Could not write the export: {exc}", "danger")
        return theme.note(
            f"Wrote {result['n_patients']} patients / {result['n_pairs']} terms to "
            f"{result['path']}", "info")

    @app.callback(
        Output("store-patient", "data", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input("overview-grid", "active_cell"),
        State("overview-grid", "derived_viewport_data"),
        State("tabs", "value"),
        prevent_initial_call=True,
    )
    def jump(active_cell, rows, tab):
        """Clicking a row opens that patient, the table is a queue, not a readout.

        Gated on actually being on this tab, and that guard is the whole reason the state is read.
        This is the **only** callback in the app that writes ``tabs.value``, so anything that made
        ``active_cell`` change while the curator was somewhere else, the grid being rebuilt, a
        stale value re-run, a cell still selected from a previous visit, would yank them back to
        the Curate tab mid-edit, with nothing on screen to explain it. A jump is a thing you asked
        for by clicking a row you can see.
        """
        if not active_cell or not rows or tab != "overview":
            raise PreventUpdate
        row = rows[active_cell["row"]]
        return row.get("patient_id"), "curate"


# ──────────────────────────────────────────────────────────────────────────────
# rendering, pure
# ──────────────────────────────────────────────────────────────────────────────
def render_stats(mode: str = "light"):
    """Progress counts over the whole cohort."""
    registry = common.registry_or_none()
    if registry is None:
        return theme.empty("No data loaded.")
    totals = registry.totals()
    rows = registry.overview()
    # Over the rows Approve actually rules on, not every record the inputs carry: counting the
    # reference sources here would report a backlog nobody is ever expected to work through.
    n_undecided = sum(r["n_existing"] - r["n_decided"] for r in rows)
    return theme.stat_row([
        theme.stat("Patients", str(totals["n_patients"]),
                   sub=f"{totals['n_confirmed']} confirmed"),
        theme.stat("Open suggestions", str(totals["n_open"])),
        theme.stat("Terms in curated ground truth", str(totals["n_in_gold"])),
        theme.stat("Annotations undecided", str(max(n_undecided, 0)),
                   help_text="rows Approve mode rules on with no verdict yet, the confirmed set "
                             "plus the prior_annotation annotation for anything it no longer carries. "
                             "These do not enter the export; prior_annotation_2 and PhenoBERT are reference and "
                             "are not counted"),
        theme.stat("Deletions proposed", str(totals["n_delete_suggested"]),
                   sub=f"{totals['n_comments']} comment(s)",
                   help_text="annotations somebody has proposed removing, out of the ground truth until "
                             "an approver settles it with Remove or Keep"),
        theme.stat("Difficulty-labelled", str(totals["n_labelled"]),
                   sub=f"{totals['n_graded_reports']} reports graded",
                   help_text="annotations carrying a label or a grade, the rows a stratified "
                             "recall table can split on"),
        theme.stat("Log", str(totals["n_events"]), sub="events"),
    ])


def render_difficulty(mode: str = "light"):
    """How the labels are distributed, so the stratification can be sanity-checked before it is used.

    Both levels: the annotation **qualifiers**, negated, family, lab value, implicit, and the
    report-level difficulty grade and labels. They are separate files and separate joins, so they
    are separate blocks here.

    A recall slice over a bucket of three annotations is noise, and the only honest place to
    notice that is before the slice is computed.
    """
    registry = common.registry_or_none()
    if registry is None:
        return theme.empty("No data loaded.")

    counts: dict[str, dict[str, int]] = {}
    n_annotations = 0
    for rows in registry.rows_by_patient().values():
        for row in rows:
            values = row.get("labels") or ()
            n_annotations += 1 if values else 0
            for value in values:
                group = vocab.INDEX.get(value, {}).get("group", "unknown")
                counts.setdefault(group, {})
                counts[group][value] = counts[group].get(value, 0) + 1

    doc_counts: dict[str, int] = {}
    for entry in registry.state()["patients"].values():
        for value in entry.get("labels") or ():
            doc_counts[value] = doc_counts.get(value, 0) + 1
        if entry.get("difficulty"):
            doc_counts[entry["difficulty"]] = doc_counts.get(entry["difficulty"], 0) + 1

    if not counts and not doc_counts:
        return theme.card(
            "Labels", theme.empty("Nothing labelled yet."),
            subtitle="Tick a qualifier on a suggestion or in a row's Edit-evidence form, or "
                     "characterise a report in the right column, and the distribution appears "
                     "here.")

    palette = theme.palette(mode)
    blocks = []
    for group_id, group_counts in counts.items():
        # ``difficulty`` only appears for a row labelled before the annotation-level grade was
        # retired. It is still counted and still named, because a log outlives its vocabulary.
        title = ("Difficulty (retired)" if group_id == "difficulty"
                 else next((g["title"] for g in vocab.GROUPS if g["id"] == group_id), group_id))
        blocks.append(html.Div([
            html.Div(title, className="label-group-title"),
            html.Div([
                html.Div([
                    common.chip(vocab.display(value), palette["neutral"],
                                title=vocab.help_for(value)),
                    html.Span(f" {count}", className="stat-sub"),
                ], className="dist-row")
                for value, count in sorted(group_counts.items(), key=lambda kv: -kv[1])
            ]),
        ], className="label-group"))
    if doc_counts:
        blocks.append(html.Div([
            html.Div("Report level", className="label-group-title",
                     title="from curation_report_labels.csv, one row per report"),
            html.Div([
                html.Div([
                    common.chip(vocab.display(value), palette["neutral"],
                                title=vocab.help_for(value)),
                    html.Span(f" {count}", className="stat-sub"),
                ], className="dist-row")
                for value, count in sorted(doc_counts.items(), key=lambda kv: -kv[1])
            ]),
        ], className="label-group"))

    return theme.card(
        f"Labels ({n_annotations} annotations qualified)",
        html.Div(blocks, className="dist-grid"),
        subtitle="Annotation qualifiers from curation_labels.csv, report labels from "
                 "curation_report_labels.csv. A slice over a bucket of three is noise, check the "
                 "sizes here before stratifying anything on them.",
    )


def render_table():
    """One row per report with its curation progress."""
    registry = common.registry_or_none()
    if registry is None:
        return theme.empty("No data loaded.")
    rows = registry.overview()
    if not rows:
        return theme.empty("No patients.")

    data = [
        {**row, "confirmed": "✓" if row["confirmed"] else ""}
        for row in sorted(rows, key=lambda r: (r["confirmed"], -r["n_disagree"], r["patient_id"]))
    ]
    return theme.card(
        f"Patients ({len(data)})",
        dash_table.DataTable(
            id="overview-grid",
            data=data,
            columns=[{"name": title, "id": key} for key, title in columns(registry)],
            page_size=25,
            sort_action="native",
            filter_action="native",
            style_as_list_view=True,
            style_cell={"fontSize": "0.857rem", "padding": "4px 8px", "fontFamily": "inherit"},
            style_header={"fontWeight": "600"},
            style_data_conditional=[
                {"if": {"filter_query": "{n_suggested} > 0", "column_id": "n_suggested"},
                 "fontWeight": "700"},
                {"if": {"filter_query": "{n_delete_suggested} > 0",
                        "column_id": "n_delete_suggested"}, "fontWeight": "700"},
                {"if": {"filter_query": "{confirmed} = '✓'"}, "opacity": "0.55"},
            ],
        ),
        subtitle="Unconfirmed first, then by how much the annotation files disagree. "
                 "Click a row to open that patient.",
    )


def markdown_summary() -> str:
    """The cohort table as Markdown, what a findings file wants pasted into it."""
    registry = common.registry_or_none()
    if registry is None:
        return ""
    totals = registry.totals()
    cols = columns(registry)
    lines = [
        f"| {' | '.join(title for _, title in cols)} |",
        f"|{'---|' * len(cols)}",
    ]
    for row in registry.overview():
        lines.append("| " + " | ".join(
            "✓" if key == "confirmed" and row.get(key)
            else ("" if key == "confirmed" else str(row.get(key, "")))
            for key, _ in cols
        ) + " |")
    header = (f"{totals['n_patients']} patients, {totals['n_confirmed']} confirmed, "
              f"{totals['n_in_gold']} terms in the curated ground truth, "
              f"{totals['n_events']} log events.")
    return header + "\n\n" + "\n".join(lines)


#: Re-exported so ``selftest`` can assert the export path without importing ``store`` itself.
EXPORT_FILE = store.EXPORT_FILE
