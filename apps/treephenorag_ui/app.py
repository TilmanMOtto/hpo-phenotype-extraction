#!/usr/bin/env python
"""The earlier tree deep-dive UI.

    python apps/treephenorag_ui/app.py --port 8053
    python apps/treephenorag_ui/app.py --selfcheck        # validate the run directories and exit

Run it on the cluster login node, where the output directory and the reports are readable, and
reach it from a laptop with ``ssh -L 8053:localhost:8053 <cluster>``. Everything is CPU-only: no
GPU, no model loading, no re-inference.

**No paths are needed on the command line.** The defaults below are lifted from
``cluster/run_{hcy,gsc}_exp13_*.sh``, the scripts that produced the artifacts, so the app finds
the runs, the ground truth and the reports without configuration. All of them are editable in the
sidebar under Data sources. Change one, press Load, and everything is rescanned and rescored
without a restart.

The one choice that changes every number is the HCY ground-truth file, and every candidate is
offered explicitly:

``curated_ground_truth_<date>/hcy_ground_truth_curated.csv``
    The ground truth ``experiments/03_setup/ground_truth`` built from the completed curation pass.
    **This is the default wherever one is on disk**, newest first, because the original HCY ground truth
    is known to be incomplete and the curated one is the set the project now scores against.
    Choosing it also brings its sidecars, see :mod:`apps.treephenorag_ui.curated`: the cohort panel, the
    trigger word behind every annotated term, and the reason behind every false positive a curator had
    already seen and dropped.
``hcy_ground_truth_raw.csv``
    What ``result_tables`` scores the thesis tables against, so a number read here and a number in the
    thesis are the same measurement. The default before a curated dataset exists.
``hcy_ground_truth.csv``
    What the inference jobs ran against, and what the artifacts' baked-in ``is_gold`` flag
    records. The count of nodes where it and the loaded ground truth disagree is on the scorecard.

Whichever file is loaded is named on every panel's copy-Markdown provenance line, and the
selfcheck's cross-check against the thesis tables reads ``result_tables``'s own config rather than
assuming the app and the thesis chose the same file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for path in (os.path.join(_ROOT, "src"), _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import dash                                                        # noqa: E402
from dash import Input, Output, State, dcc, html                   # noqa: E402

from apps.treephenorag_ui import curated, discovery, sample, state, theme  # noqa: E402
from apps.treephenorag_ui.registry import Registry                         # noqa: E402
from apps.treephenorag_ui.views import (                                   # noqa: E402
    calibration, compare, deepdive, errors_view, pruning_view, scorecard,
)

log = logging.getLogger("exp13_ui")

from hpo_extraction.paths import lookup as _lookup  # noqa: E402

_RESULTS = _lookup("results_dir")

#: Straight out of the run scripts. ``output_dir=$REPO/output`` (every earlier script),
#: ``input_dir=$BASE/hcy/input`` (run_hcy_exp13_00_tree_gate_lr.sh:49-60),
#: ``gsc_dir=$REPO/resources/data/GSC_2024`` (run_gsc_*), ``stanza_dir=$BASE/stanza_resources``.
#: The ground truth file is an exception: it is ``run_exp13_07_thesis_metrics.sh:56``'s
#: ``hcy_ground_truth_raw.csv``, not the inference jobs' ``hcy_ground_truth.csv``. It is
#: also the *fallback*, not the default, :func:`resolve_hcy_gt` prefers a curated dataset
#: where one exists, and this is what it falls back to where one does not.
DEFAULTS = {
    "output_base": _RESULTS,
    "hcy_dir": _lookup("hcy.dir"),
    "hcy_gt": _lookup("hcy.raw_ground_truth"),
    "hcy_input": _lookup("hcy.input_dir"),
    "gsc_dir": "resources/data/GSC_2024",
    "stanza_dir": _lookup("stanza_dir"),
    "thesis_tables": f"{_RESULTS}/exp13_07_thesis_metrics/tables",
    # apps/compare_ui/select_hcy_documents.py's own --out default, under the same output folder.
    "deepdive_frame": f"{_RESULTS}/hcy_deepdive_frame",
}

#: The two legacy HCY ground truth files, in the order the sidebar offers them under the curated datasets.
#: Offered, not typed because switching ground truth is the single most consequential thing a user
#: of this app can do, and typing a path by hand invites getting it subtly wrong.
HCY_GT_LEGACY = [
    (_lookup("hcy.raw_ground_truth"),
     "hcy_ground_truth_raw.csv, what the result tables score"),
    (f"{_lookup('hcy.dir')}/hcy_ground_truth.csv", "hcy_ground_truth.csv, what inference used"),
]


def hcy_gt_alternatives(hcy_gt_path: str) -> list[tuple[str, str]]:
    """``[(path, label), …]`` for the ground truth dropdown, curated datasets first, newest first.

    Built at runtime from whichever ground truth is loaded, not listed as a constant, because the curated
    datasets are **dated directories**: a constant would name one date, go stale the next time
    the ground-truth build runs, and quietly stop offering the newest ground truth. The legacy files are appended
    whatever happens, and the currently loaded path is appended when it is neither, so the control
    never shows a value it does not have an option for.
    """
    hcy_dir = curated.hcy_dir_of(hcy_gt_path) or DEFAULTS["hcy_dir"]
    options = [(os.path.join(d, curated.GOLD_FILE), curated.label_for(d))
               for d in curated.discover(hcy_dir)]
    options += HCY_GT_LEGACY
    if hcy_gt_path and not any(hcy_gt_path == p for p, _ in options):
        options.append((hcy_gt_path, f"{os.path.basename(hcy_gt_path)}, configured on the "
                                     f"command line"))
    return options


def resolve_hcy_gt(hcy_gt_path: str) -> str:
    """The ground truth file to open on, preferring the newest curated dataset beside it.

    Applied **only to the untouched default**, an explicitly passed ``--hcy-gt`` is never
    substituted, as an explicitly passed ``--gsc-dir`` is not. Silently scoring against a
    different ground truth than the one named on the command line is the kind of help nobody wants, and
    the ground truth is the one path here where that help would change every number on screen.
    """
    if hcy_gt_path != DEFAULTS["hcy_gt"]:
        return hcy_gt_path
    return curated.gold_path_for(curated.hcy_dir_of(hcy_gt_path)) or hcy_gt_path


#: Local fallbacks, so the app is runnable from a workstation with an rsynced output folder.
_LOCAL_GSC = os.path.join(_ROOT, "resources", "data", "GSC_2024")

VIEWS = [
    ("scorecard", "Scorecard", scorecard),
    ("pruning", "Pruning autopsy", pruning_view),
    ("errors", "Error anatomy", errors_view),
    ("calibration", "Calibration", calibration),
    ("deepdive", "Report deep-dive", deepdive),
    ("compare", "Compare A/B", compare),
]


def build_layout(registry: Registry) -> html.Div:
    """The page layout: run and report selectors and one tab per view."""
    options = registry.cell_options()
    default_a = _default_cell(registry, options)
    default_b = _default_cell(registry, options, avoid=default_a)

    sidebar = html.Div([
        html.Div("TreePhenoRAG deep dive", className="brand"),
        html.Div("TreePhenoRAG, where pruning went wrong", className="brand-sub"),

        html.Div([
            html.Label("Run A", className="field-label"),
            dcc.Dropdown(id="run-a", options=options, value=default_a, clearable=False),
            html.Label("τ / configuration", className="field-label"),
            dcc.Dropdown(id="op-a", options=[], value=None, clearable=False),
        ], className="field"),

        html.Div([
            html.Label("Run B (compare)", className="field-label"),
            dcc.Dropdown(id="run-b", options=options, value=default_b, clearable=True),
            html.Label("τ / configuration", className="field-label"),
            dcc.Dropdown(id="op-b", options=[], value=None, clearable=True),
        ], className="field"),

        html.Details([
            html.Summary("Data sources", className="details-summary"),
            _path_field("Output folder", "src-output", registry.output_base),
            html.Label("HCY ground truth", className="field-label"),
            dcc.Dropdown(
                id="src-gt-preset",
                options=_gt_options(registry.hcy_gt_path),
                value=registry.hcy_gt_path,
                placeholder="or type a path below", clearable=True),
            dcc.Input(id="src-gt", type="text", value=registry.hcy_gt_path,
                      className="path-input", debounce=True),
            _path_field("HCY reports", "src-input", registry.hcy_input_dir),
            _path_field("GSC+ folder", "src-gsc", registry.gsc_dir),
            _path_field("Stanza resources", "src-stanza", registry.stanza_dir),
            _path_field("HCY deep-dive sample", "src-frame", registry.hcy_frame_dir),
            html.Button("Load", id="src-load", n_clicks=0, className="copy-btn"),
        ], open=not options, className="sidebar-section"),

        html.Div(id="src-status"),
        _subset_section(registry),
        html.Div(id="run-summary"),

        html.Div([
            # Built on demand, not on every run change: a findings block includes the error
            # taxonomy and the calibration block, which are the two most expensive things this
            # app computes. Filling a passive clipboard would have made every dropdown change
            # pay for them whether or not anyone ever pressed copy.
            html.Button("Build findings block", id="findings-build", n_clicks=0,
                        className="copy-btn"),
            dcc.Clipboard(id="copy-findings", title="Copy the findings block for Run A",
                          className="copy-btn copy-clip"),
            html.Div(id="findings-status", className="stat-sub"),
        ], className="sidebar-section"),

        html.Div([
            dcc.RadioItems(id="theme-toggle",
                           options=[{"label": " Light", "value": "light"},
                                    {"label": " Dark", "value": "dark"}],
                           value="light", inline=True),
            html.Button("Presentation mode", id="present-btn", n_clicks=0, className="copy-btn"),
        ], className="sidebar-section"),
    ], className="sidebar")

    main = html.Div([
        dcc.Tabs(id="tabs", value="scorecard", className="tabs-bar",
                 children=[dcc.Tab(label=label, value=vid, className="tab")
                           for vid, label, _ in VIEWS]),
        html.Div([
            html.Div(module.layout(), id=f"view-{vid}")
            for vid, _, module in VIEWS
        ]),
    ], className="main")

    return html.Div([
        sidebar, main,
        dcc.Store(id="store-theme", data="light"),
        dcc.Store(id="store-present", data=0),
        dcc.Store(id="store-focus", data=None),
        dcc.Store(id="store-subset", data=DEFAULT_SUBSET),
        # Which sentence of the report the deep-dive last sent the reader to. Written by the
        # calls panel's `sent #N` badges, read by the clientside scroll below.
        dcc.Store(id="store-sentence", data=None),
        dcc.Store(id="store-scrolled", data=None),
    ], className="shell")


#: The unrestricted state, and what every view sees when no frame is configured.
DEFAULT_SUBSET = {"mode": "all", "cell": None, "aggregates": False}


def _subset_options(registry: Registry) -> list[dict]:
    """``All reports`` / the whole sample / one option per drawn cell."""
    frame = registry.sample
    options = [{"label": "All reports", "value": "all"}]
    if frame is None:
        return options
    options.append({"label": f"Deep-dive sample ({len(frame.selected)})", "value": "sample"})
    options += [{"label": f"  · {cell} ({len(frame.picks[cell])})", "value": f"cell:{cell}"}
                for cell in frame.cells]
    return options


def _subset_section(registry: Registry) -> html.Div:
    """The subset picker, and the separate opt-in that lets it reach the metrics.

    Two controls, not one because they are different claims. Restricting which reports you
    can page through changes nothing about any number on screen. Restricting the *metrics* makes
    every rate a rate over a purposive selection rule, which is why it is a second, explicit click
    and why turning it on paints :data:`sample.WARNING` across the aggregate views.

    Both are always rendered, disabled when no frame is configured, not absent. A control
    that vanishes reads as a feature that does not exist. A disabled one with the reason under it
    reads as a path that has not been set, which is what it is, and it is fixable in the field
    directly above.
    """
    frame = registry.sample
    missing = frame is None
    return html.Div([
        html.Label("Report subset", className="field-label"),
        dcc.Dropdown(id="subset-mode", options=_subset_options(registry), value="all",
                     clearable=False, disabled=missing),
        dcc.Checklist(
            id="subset-aggregates",
            options=[{"label": " Also restrict the metrics", "value": "on"}],
            value=[], className="field-check",
        ),
        html.Div(
            theme.note("No deep-dive sampling frame found. Point 'HCY deep-dive sample' above at "
                       "the folder apps/compare_ui/select_hcy_documents.py wrote, then press Load.",
                       tone="info") if missing else "",
            id="subset-status", className="stat-sub"),
    ], className="sidebar-section", id="subset-section")


def _gold_name(report: dict) -> str:
    """The ground truth as a name a reader can act on.

    A curated ground truth's file is called ``hcy_ground_truth_curated.csv`` in every dataset ever built,
    so the basename alone says nothing about *which* one is loaded, the date is in the directory.
    The dataset name is shown instead wherever there is one.
    """
    block = report.get("curated") or {}
    return block.get("name") or os.path.basename(report.get("gold_source") or "?")


def _reports_kv(report: dict, counts: dict):
    """``48``, or ``48 of 60`` when the curated cohort is what left the other twelve out."""
    scored = counts.get("n_reports_scored")
    block = report.get("curated") or {}
    outside = block.get("n_reports_outside") or 0
    if not outside:
        return str(scored)
    return html.Span([str(scored), html.Span(f" of {int(scored or 0) + int(outside)}",
                                             className="stat-sub")],
                     title=f"{outside} report(s) in this run are outside the curated cohort and "
                           f"are not scored, see the scorecard's curated-ground truth panel.")


def _gt_options(hcy_gt_path: str) -> list[dict]:
    return [{"label": label, "value": path}
            for path, label in hcy_gt_alternatives(hcy_gt_path)]


def _path_field(label: str, field_id: str, value: str) -> html.Div:
    return html.Div([
        html.Label(label, className="field-label"),
        dcc.Input(id=field_id, type="text", value=value, className="path-input", debounce=True),
    ], className="field")


def _parse_subset(mode):
    """``"cell:pb_worst"`` -> ``("cell", "pb_worst")``. Anything else -> ``(mode, None)``."""
    if isinstance(mode, str) and mode.startswith("cell:"):
        return "cell", mode.split(":", 1)[1]
    return (mode or "all"), None


def _default_cell(registry: Registry, options, avoid=None):
    """Prefer a tree cell, those are the ones this app is about."""
    values = [o["value"] for o in options]
    for value in values:
        cell = registry.cell(value)
        if cell and cell.spec.is_tree and value != avoid:
            return value
    for value in values:
        if value != avoid:
            return value
    return None


def register_callbacks(app: dash.Dash) -> None:
    """Register the page-level callbacks (run, configuration and report selection, theme)."""
    for _, _, module in VIEWS:
        module.register(app)

    @app.callback(
        [Output(f"view-{vid}", "style") for vid, _, _ in VIEWS],
        Input("tabs", "value"),
    )
    def _switch(active):
        return [{"display": "block"} if vid == active else {"display": "none"}
                for vid, _, _ in VIEWS]

    @app.callback(
        Output("op-a", "options"), Output("op-a", "value"),
        Input("run-a", "value"), State("op-a", "value"),
    )
    def _ops_a(cell_id, current):
        return _op_options(cell_id, current)

    @app.callback(
        Output("op-b", "options"), Output("op-b", "value"),
        Input("run-b", "value"), State("op-b", "value"),
    )
    def _ops_b(cell_id, current):
        return _op_options(cell_id, current)

    @app.callback(
        Output("store-subset", "data"),
        Output("subset-status", "children"),
        Input("subset-mode", "value"), Input("subset-aggregates", "value"),
        Input("run-a", "value"),
    )
    def _subset(mode, aggregates, cell_id):
        """Translate the two controls into the state every view reads.

        The run id is an input because the frame is an HCY draw: on a GSC+ cell the choice is
        reported as not applying, not quietly filtering nothing, which would otherwise show
        as an empty report list with no visible cause.
        """
        chosen, cell = _parse_subset(mode)
        restrict = bool(aggregates) and chosen != "all"
        data = {"mode": chosen, "cell": cell, "aggregates": restrict}

        registry = state.get_registry()
        frame = registry.sample
        run = registry.cell(cell_id) if cell_id else None
        if frame is None:
            # no_update, not "": the layout already carries the "point this at a folder" note, and
            # blanking it on first paint would leave a disabled control with no explanation.
            return DEFAULT_SUBSET, dash.no_update
        if chosen == "all":
            return data, ""
        if run is not None and not frame.applies_to(run.cohort):
            return data, theme.note(
                f"The deep-dive sample is an {sample.COHORT.upper()} draw; this run is "
                f"{run.cohort.upper()}, so no filter is applied.", tone="warn")

        ids = frame.ids(chosen, cell)
        note = f"{len(ids or ())} report(s) · {frame.describe(chosen, cell)}"
        if restrict:
            return data, theme.note(f"Metrics restricted to {note}. {sample.WARNING}", tone="warn")
        return data, html.Div(f"Report lists limited to {note}. Metrics stay on the full cohort.",
                              className="stat-sub")

    @app.callback(
        Output("src-gt", "value"),
        Input("src-gt-preset", "value"),
        prevent_initial_call=True,
    )
    def _preset(path):
        from dash.exceptions import PreventUpdate

        if not path:
            raise PreventUpdate
        return path

    @app.callback(
        Output("src-status", "children"),
        Output("run-a", "options"), Output("run-b", "options"),
        Output("subset-mode", "options"), Output("src-gt-preset", "options"),
        Input("src-load", "n_clicks"),
        State("src-output", "value"), State("src-gt", "value"), State("src-input", "value"),
        State("src-gsc", "value"), State("src-stanza", "value"), State("src-frame", "value"),
        prevent_initial_call=True,
    )
    def _load(_clicks, output_base, hcy_gt, hcy_input, gsc_dir, stanza_dir, frame_dir):
        current = state.get_registry()
        registry = Registry(
            output_base or "", hcy_gt_path=hcy_gt or "", hcy_input_dir=hcy_input or "",
            gsc_dir=gsc_dir or "", stanza_dir=stanza_dir or "", cache_dir=current.cache_dir,
            thesis_tables_dir=current.thesis_tables_dir, hcy_frame_dir=frame_dir or "",
        )
        problems = registry.validate()
        blocking = [p for p in problems if p.startswith(("Output folder", "No TreePhenoRAG"))]
        if blocking:
            return (theme.note(" ".join(problems), tone="danger"),
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update)

        state.set_registry(registry)
        options = registry.cell_options()
        # Rescanned, not kept: pointing the ground truth field at a different cohort directory,
        # or running the ground-truth build again while the app is open, changes which curated datasets exist,
        # and a dropdown still listing the old ones is a control that lies about the filesystem.
        message = (theme.note(" ".join(problems), tone="warn") if problems
                   else theme.note(f"Loaded {len(options)} run(s) from {output_base}"
                                   + (f" · ground truth: {registry.curated.describe()}"
                                      if registry.curated else ""), tone="info"))
        return (message, options, options, _subset_options(registry),
                _gt_options(registry.hcy_gt_path))

    @app.callback(
        Output("run-summary", "children"),
        Input("run-a", "value"), Input("op-a", "value"),
    )
    def _summary(cell_id, operating_point):
        if not cell_id:
            return theme.empty("No run selected.")
        registry = state.get_registry()
        cell = registry.cell(cell_id)
        report = registry.get_report(cell_id, operating_point)
        if not cell or not report:
            return theme.empty("Nothing loaded for this run.")
        counts = report.get("counts") or {}
        rows = [
            html.Div([html.B("experiment "), html.Code(cell.spec.exp_id)], className="kv"),
            html.Div([html.B("variant "), html.Code(cell.spec.variant)], className="kv"),
            html.Div([html.B("gold "), html.Code(_gold_name(report))], className="kv"),
            html.Div([html.B("reports "), _reports_kv(report, counts)], className="kv"),
            html.Div([html.B("status "), html.Span(cell.status, className="pill")],
                     className="kv"),
        ]
        if cell.problems:
            rows.append(theme.note(" · ".join(cell.problems), tone="warn"))
        return html.Div(rows, className="sidebar-section")

    @app.callback(
        Output("copy-findings", "content"),
        Output("findings-status", "children"),
        Input("findings-build", "n_clicks"),
        State("run-a", "value"), State("op-a", "value"),
        prevent_initial_call=True,
    )
    def _findings(_clicks, cell_id, operating_point):
        from dash.exceptions import PreventUpdate

        from apps.treephenorag_ui import copyexport

        if not cell_id:
            raise PreventUpdate
        registry = state.get_registry()
        report = registry.get_report(cell_id, operating_point, heavy=True)
        if not report:
            raise PreventUpdate
        frontier = (registry.tau_frontier(cell_id, operating_point)
                    if report.get("is_tree") else None)
        return (copyexport.findings_block(report, frontier),
                f"ready for {report.get('operating_point') or '—'}, press the copy icon")

    app.clientside_callback(
        "function(mode) { document.body.setAttribute('data-theme', mode || 'light');"
        " return mode || 'light'; }",
        Output("store-theme", "data"), Input("theme-toggle", "value"),
    )

    app.clientside_callback(
        "function(n) { const on = (n || 0) % 2 === 1;"
        " document.body.setAttribute('data-present', on ? '1' : '0'); return n || 0; }",
        Output("store-present", "data"), Input("present-btn", "n_clicks"),
    )

    # Scroll the report to a sentence, and light it. Dash renders a pattern id as
    # ``JSON.stringify`` of the dict with its keys **sorted**, which is what makes
    # ``{"idx": n, "type": "dd-seg"}`` the element's real DOM id. A miss is a no-op, not an
    # error: the sentence may be past ``deepdive.MAX_SENTENCES_SHOWN``, the report may not have been
    # redrawn yet, or the reader may have switched tabs. Ported from
    # ``apps/curation_ui/app.py``'s scroll-to-segment, including this note.
    app.clientside_callback(
        """
        function(focus) {
            if (!focus || focus.idx === null || focus.idx === undefined) {
                return window.dash_clientside.no_update;
            }
            document.querySelectorAll('.seg-selected').forEach(function (node) {
                node.classList.remove('seg-selected');
            });
            const el = document.getElementById(
                JSON.stringify({idx: focus.idx, type: 'dd-seg'}));
            if (!el) { return window.dash_clientside.no_update; }
            el.classList.add('seg-selected');
            el.scrollIntoView({behavior: 'smooth', block: 'center'});
            return focus.idx;
        }
        """,
        Output("store-scrolled", "data"),
        Input("store-sentence", "data"),
    )

    app.clientside_callback(
        """
        function(n, id) {
            if (!n) { return window.dash_clientside.no_update; }
            const el = document.getElementById(
                JSON.stringify(id, Object.keys(id).sort()));
            if (window.phenorag) { window.phenorag.copyPng(id.graph, el); }
            return '';
        }
        """,
        Output({"type": "copy-sink", "graph": dash.MATCH}, "children"),
        Input({"type": "copy-png", "graph": dash.MATCH}, "n_clicks"),
        State({"type": "copy-png", "graph": dash.MATCH}, "id"),
    )


def _op_options(cell_id, current):
    """Sweep dropdown options, with never-written points labelled, not hidden.

    Hiding them would be a lie about what is on disk. Leaving them unlabelled invites picking one
    and concluding the run is broken. They are shown, marked, and never chosen as the default.
    """
    if not cell_id:
        return [], None
    registry = state.get_registry()
    points = registry.operating_points(cell_id)
    if not points:
        return [{"label": "—", "value": "-"}], "-"
    written = set(registry.written_operating_points(cell_id))
    options = [{"label": discovery.op_label(p) if p in written
                else f"{discovery.op_label(p)}  (empty)", "value": p} for p in points]
    return options, registry.resolve_op(cell_id, current)


def build_registry(args) -> Registry:
    """The registry over the run folders and ground truths named in *args*."""
    gsc_dir = args.gsc_dir
    # Fall back to the repo's own copy only when the cluster default was left untouched. An
    # explicitly passed path is never substituted: silently scoring against a different GSC+ than
    # The one named on the command line is the kind of help nobody wants.
    if gsc_dir == DEFAULTS["gsc_dir"] and not os.path.isdir(gsc_dir) and os.path.isdir(_LOCAL_GSC):
        gsc_dir = _LOCAL_GSC
    return Registry(
        args.output_base, hcy_gt_path=resolve_hcy_gt(args.hcy_gt),
        hcy_input_dir=args.hcy_input,
        hcy_segments_path=args.hcy_segments,
        gsc_dir=gsc_dir, stanza_dir=args.stanza_dir, cache_dir=args.cache_dir,
        thesis_tables_dir=args.thesis_tables, hcy_frame_dir=args.deepdive_frame,
    )


def parse_args(argv=None):
    """Parse the command-line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-base", default=DEFAULTS["output_base"],
                        help="folder holding <exp_id>/<cohort>/ run directories")
    parser.add_argument("--hcy-gt", default=DEFAULTS["hcy_gt"],
                        help="HCY ground truth. Left at its default, the newest curated_ground_truth_<date>/ "
                             "beside it wins; passed explicitly, it is used as given")
    parser.add_argument("--hcy-input", default=DEFAULTS["hcy_input"])
    parser.add_argument("--hcy-segments", default="",
                        help="segmented_reports.csv. Empty (the default) derives it from the HCY "
                             "ground truth's cohort directory, which is the one the curated ground truth's "
                             "segment_idx indexes")
    parser.add_argument("--gsc-dir", default=DEFAULTS["gsc_dir"])
    parser.add_argument("--stanza-dir", default=DEFAULTS["stanza_dir"])
    parser.add_argument("--thesis-tables", default=DEFAULTS["thesis_tables"],
                        help="tables/ folder of the result tables, for the selfcheck's cross-check")
    parser.add_argument("--deepdive-frame", default=DEFAULTS["deepdive_frame"],
                        help="folder apps/compare_ui/select_hcy_documents.py wrote (frame.json, pools.csv)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--port", type=int, default=8053)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate every discovered run and exit without serving")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Discover the runs, then serve the app. Returns the exit status."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    registry = build_registry(args)
    state.set_registry(registry)

    if args.selfcheck:
        from apps.treephenorag_ui import selfcheck

        return selfcheck.run(registry)

    for problem in registry.validate():
        log.warning("%s", problem)

    log.info("building the HPO tree and the traversal graph (once) …")
    _ = registry.graph
    theme.assert_vocabularies_match()
    log.info("ready, %d run(s) discovered", len(registry.cells))

    app = dash.Dash(__name__, suppress_callback_exceptions=True,
                    assets_folder=os.path.join(_HERE, "assets"),
                    title="TreePhenoRAG deep dive")
    app.layout = build_layout(registry)
    register_callbacks(app)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
