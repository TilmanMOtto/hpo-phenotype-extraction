"""The Free Listing generation run SLM-ensemble deep-dive UI, Dash entry point.

    python apps/phenojury_ui/app.py --port 8054      # on the cluster login node
    ssh -L 8054:localhost:8054 <cluster>           # then open http://localhost:8054

CPU-only: no GPU, no model reload, no PhenoBERT rerun. Everything is re-run from the artifacts
``slm_ensemble_experiment`` already wrote, which is the invariant the driver was built
around, score once, re-run the decision logic.

    python apps/phenojury_ui/app.py --selftest       # verification gates + every view, headless

runs the whole app against whatever data ``--output-base`` points at and exits non-zero on the
first problem. That is the acceptance check. See ``README.md``.

**One control set, every view.** The sidebar's cohort, rule, k, model subset and
``min_detection_count`` live in ``store-config``, and every view renders at that configuration. So
a subset chosen once is the subset the scorecard, the voters, the autopsy and the term explorer all
speak about, and the provenance line under each Copy-Markdown block names it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Importable without `pip install -e .`, which is how the other two analysis apps are run.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import dash  # noqa: E402
from dash import MATCH, Input, Output, State, dcc, html  # noqa: E402

from apps.phenojury_ui import frame as frame_mod  # noqa: E402
from apps.phenojury_ui import state, theme, votes  # noqa: E402
from apps.phenojury_ui.registry import Registry  # noqa: E402
from apps.phenojury_ui.views import (  # noqa: E402
    annotate,
    compare_pb,
    falsepos,
    missed,
    models as models_view,
    patient,
    scorecard,
    terms,
    voters,
)

logger = logging.getLogger(__name__)

#: The cluster layout the experiments actually ran with (cluster/run_*_exp13_06_*.sh).
from hpo_extraction.paths import lookup as _lookup  # noqa: E402

DEFAULT_OUTPUT_BASE = _lookup("results_dir")

VIEWS = [
    ("scorecard", "Scorecard", scorecard),
    ("models", "Models", models_view),
    ("voters", "Voters", voters),
    ("missed", "Missed (FN)", missed),
    ("falsepos", "False positives", falsepos),
    ("patient", "Report deep dive", patient),
    ("terms", "Term explorer", terms),
    ("pbcompare", "vs PhenoBERT", compare_pb),
    # Last, and the only tab that writes: everything before it re-runs what the driver shipped,
    # this one records what a person read in it.
    ("annotate", "Annotate", annotate),
]


def run_label(run: dict) -> str:
    """How a run reads in the cohort dropdown.

    The id has to be unique across experiments, so it carries the full path
    (``phenojury_generation_other_prompts/hcy/q4_span_json``). The label does not, and reading it as
    ``phenojury_generation_other_prompts - hcy - q4_span_json`` is the difference between a dropdown you can scan and one you
    have to parse.
    """
    exp_id = run.get("exp_id", "")
    parts = [exp_id.split("_prompt")[0].split("_slm")[0] if exp_id else "", run.get("cohort", "")]
    if run.get("prompt_key"):
        parts.append(run["prompt_key"])
    label = " · ".join(part for part in parts if part)
    return label or run["run_id"]


def run_options(registry: Registry) -> list[dict]:
    """Dropdown options for the runs of *registry*."""
    return [{"label": run_label(r), "value": r["run_id"]} for r in registry.list_runs()]


def build_layout(registry: Registry) -> html.Div:
    """The page layout: run and voting-configuration controls and one tab per view."""
    runs = run_options(registry)
    default_run = runs[0]["value"] if runs else None

    sidebar = html.Div(
        [
            html.H1("SLM ensemble", className="brand"),
            html.Div("PhenoJury · Free Listing · 8 jurors, k-of-N vote", className="brand-sub"),

            html.Div([
                html.Label("Cohort", className="field-label"),
                dcc.Dropdown(id="run-a", options=runs, value=default_run, clearable=False,
                             placeholder="no Free Listing generation runs found"),
            ], className="field"),

            # The configuration. Populated per cohort, because the k range is the ensemble size
            # and the model list is whichever models actually grounded.
            html.Div([
                html.Label("Aggregation rule", className="field-label"),
                dcc.RadioItems(
                    id="cfg-rule", inline=True, value="vote_k",
                    options=[{"label": " k of N", "value": "vote_k"},
                             {"label": " plurality", "value": "plurality"}],
                ),
                html.Div(id="cfg-k-wrap", children=[
                    html.Label("k, models that must agree", className="field-label"),
                    dcc.Slider(id="cfg-k", min=1, max=8, step=1, value=1,
                               marks={i: str(i) for i in range(1, 9)}),
                ], style={"marginTop": "10px"}),
                html.Label("Voting models", className="field-label",
                           style={"marginTop": "10px"}),
                dcc.Checklist(id="cfg-models", options=[], value=[], className="check-list"),
                html.Label("min_detection_count", className="field-label",
                           style={"marginTop": "10px"}),
                dcc.Slider(id="cfg-mincount", min=1, max=3, step=1, value=1,
                           marks={i: str(i) for i in (1, 2, 3)}),
                html.Button("Reset to the shipped configuration", id="cfg-reset",
                            className="copy-btn", style={"marginTop": "10px"}),
            ], className="sidebar-section"),

            # The deep-dive subset. Separate from the configuration above because it is not one:
            # The rule, k and the model set choose *how* the ensemble decides, this chooses *which
            # reports are on screen*. Reset lives with the configuration and does not
            # touch this, a reader working the sample does not want it cleared by a rule change.
            html.Div([
                html.Label("Report subset", className="field-label"),
                dcc.Dropdown(id="cfg-subset", options=[], value=frame_mod.ALL, clearable=False),
                html.Div(id="cfg-subset-note", className="stat-sub"),
            ], className="sidebar-section", id="cfg-subset-wrap"),

            html.Details([
                html.Summary("Data source", className="field-label details-summary"),
                html.Div([
                    html.Label("Experiment output folder", className="field-label"),
                    dcc.Input(id="src-output", value=registry.output_base, type="text",
                              debounce=True, className="path-input"),
                    html.Div("Cohorts are the folders under phenojury_generation_free_listing/ that hold "
                             "detections_*.jsonl. No ground-truth file is needed, the generation run writes "
                             "the ground-truth set into every predictions file.",
                             className="stat-sub"),
                    html.Label("PhenoBERT baseline (baseline_phenobert)", className="field-label",
                               style={"marginTop": "10px"}),
                    dcc.Input(id="src-pb", value=registry.pb_base, type="text", debounce=True,
                              className="path-input",
                              placeholder="blank = derive from the cohort"),
                    html.Div("Left blank, baseline_phenobert/<cohort>/ is found beside the "
                             "ensemble run, which is the cluster layout, so it usually just "
                             "appears. Fill it in to compare against a baseline living elsewhere.",
                             className="stat-sub"),
                    html.Label("Deep-dive frame", className="field-label",
                               style={"marginTop": "10px"}),
                    dcc.Input(id="src-frame", value=registry.frame_base, type="text",
                              debounce=True, className="path-input",
                              placeholder="blank = find hcy_deepdive_frame/ beside the output"),
                    html.Div("apps/compare_ui/select_hcy_documents.py writes the 20-report HCY draw. Left "
                             "blank, hcy_deepdive_frame/ is found beside the experiment folders. "
                             "Without one the subset control simply does not appear.",
                             className="stat-sub"),
                    html.Button("Load", id="src-load", className="copy-btn",
                                style={"marginTop": "8px"}),
                ], style={"paddingTop": "10px"}),
            ], open=not runs),

            html.Div(id="src-status"),
            html.Div(id="run-summary", className="sidebar-section"),

            html.Div([
                html.Label("Display", className="field-label"),
                dcc.RadioItems(
                    id="theme-toggle", inline=True, value="light",
                    options=[{"label": " Light", "value": "light"},
                             {"label": " Dark", "value": "dark"}],
                ),
                html.Button("Presentation mode", id="present-btn", className="copy-btn",
                            style={"marginTop": "8px"}),
            ], className="sidebar-section"),
        ],
        className="sidebar",
    )

    main = html.Div(
        [
            # One banner for eight views, rather than eight copies of one sentence. It is a claim
            # about the data on every screen at once, so it lives above the tab bar where it cannot
            # be scrolled past or forgotten by whichever view was written last.
            html.Div(id="subset-banner"),
            dcc.Tabs(id="tabs", value="scorecard", className="tabs-bar",
                     children=[dcc.Tab(label=label, value=vid) for vid, label, _ in VIEWS]),
            # Every view is in the layout at once and toggled by display, so a callback bound to
            # one of its components always has a target. Only the *open* one computes and draws:
            # each render callback takes `tabs` as an input and gates on it, see
            # views/common.should_render.
            html.Div([html.Div(module.layout(), id=f"view-{vid}") for vid, _, module in VIEWS]),
        ],
        className="main",
    )

    return html.Div([
        dcc.Store(id="store-theme", data="light"),
        dcc.Store(id="store-present", data=0),
        dcc.Store(id="store-config", data=None),
        # Cross-view drill-down: the term explorer and the autopsy write a report id here and the
        # deep-dive picks it up, so "which report is this row" is one click, not a search.
        dcc.Store(id="store-focus", data=None),
        # What each view last drew, so re-opening an unchanged tab redraws nothing. Browser state
        #, not process state: two open readers must not suppress each other's renders.
        *[dcc.Store(id=f"sig-{vid}", data=None) for vid, _, _ in VIEWS],
        html.Div([sidebar, main], className="shell"),
    ])


def register_callbacks(app: dash.Dash) -> None:
    """Register the page-level callbacks (run, voting configuration, theme)."""
    for _, _, module in VIEWS:
        module.register(app)

    @app.callback(
        [Output(f"view-{vid}", "style") for vid, _, _ in VIEWS],
        Input("tabs", "value"),
    )
    def switch_view(active):
        return [{"display": "block"} if vid == active else {"display": "none"}
                for vid, _, _ in VIEWS]

    @app.callback(
        Output("run-a", "options"),
        Output("run-a", "value"),
        Output("src-status", "children"),
        Input("src-load", "n_clicks"),
        State("src-output", "value"),
        State("src-pb", "value"),
        State("src-frame", "value"),
        prevent_initial_call=True,
    )
    def load_source(_n, output_base, pb_base, frame_base):
        """Repoint the app at another output folder without a restart.

        A bad path keeps the previous registry: losing the working view because of a typo in a
        text box is a worse failure than the typo.
        """
        from apps.phenojury_ui.views.common import clear_failures

        try:
            previous = state.get_registry()
            candidate = Registry(output_base=output_base,
                                 cache_dir=previous.cache_dir,
                                 pb_base=pb_base,
                                 frame_base=frame_base,
                                 annotations_dir=previous.annotations_dir)
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, dash.no_update, theme.note(f"Could not load: {exc}", "danger")

        problems = candidate.validate()
        if problems:
            return dash.no_update, dash.no_update, theme.note(" ".join(problems), "danger")

        state.set_registry(candidate)
        clear_failures()   # a new source means the old cohorts' failures no longer apply
        runs = run_options(candidate)
        note = f"Loaded {len(runs)} cohort(s)."
        if candidate.frame:
            note += f" Deep-dive frame: {candidate.frame_path}."
        elif frame_base:
            note += " No frame at the path given, so the subset control stays hidden."
        return runs, runs[0]["value"], theme.note(note, "info")

    @app.callback(
        Output("cfg-models", "options"),
        Output("cfg-models", "value"),
        Output("cfg-k", "max"),
        Output("cfg-k", "marks"),
        Output("cfg-k", "value"),
        Output("cfg-rule", "value"),
        Output("cfg-mincount", "value"),
        Input("run-a", "value"),
        Input("cfg-reset", "n_clicks"),
    )
    def reset_controls(run_a, _n):
        """Point the controls at this cohort's shipped configuration.

        Fires on cohort change *and* on Reset, because both mean the same thing: forget whatever
        was being explored and go back to the configuration the experiment reported.
        """
        # Absolute, not relative: this module is __main__ when the app is launched as a script,
        # so it has no parent package and `from .views...` raises ImportError inside the callback
        #, a 500 on one panel, not a startup failure, which is why it needs a test.
        from apps.phenojury_ui.views.common import bundle_for

        bundle = bundle_for(run_a)
        if bundle is None or not bundle["models"]:
            return [], [], 8, {i: str(i) for i in range(1, 9)}, 1, "vote_k", 1
        default = bundle["default_config"]
        n = len(bundle["models"])
        options = [{"label": f" {m}", "value": m} for m in bundle["models"]]
        return (options, list(default["models"]), n, {i: str(i) for i in range(1, n + 1)},
                default["k"], default["rule"], default["min_count"])

    @app.callback(
        Output("cfg-subset", "options"),
        Output("cfg-subset", "value"),
        Output("cfg-subset", "disabled"),
        Output("cfg-subset-note", "children"),
        Input("run-a", "value"),
    )
    def subset_options(run_a):
        """Rebuild the subset choices for this cohort, and reset the selection.

        Resetting on every cohort change is the point: the frame is drawn on HCY, so a cell carried
        over to GSC+ would name no report here and empty every screen. Instead of let that happen
        and explain it afterwards, the control goes back to "all reports" and says why it is
        disabled.
        """
        from apps.phenojury_ui.views.common import bundle_for

        registry = state.get_registry() if state.has_registry() else None
        frame = getattr(registry, "frame", None) if registry else None
        all_option = [{"label": "All reports", "value": frame_mod.ALL}]
        if not frame:
            searched = registry.frame_searched() if registry else []
            return (all_option, frame_mod.ALL, True,
                    "No deep-dive frame loaded. Looked in: "
                    + ("; ".join(searched[:3]) if searched else "nowhere"))

        bundle = bundle_for(run_a)
        ids = bundle["report_ids"] if bundle else []
        cells = frame_mod.cells_present(frame, ids)
        if not cells:
            return (all_option, frame_mod.ALL, True,
                    f"The frame names none of this cohort's reports, it was drawn on "
                    f"{len(frame['selected'])} HCY reports. {frame_mod.summary(frame, ids)}")

        n_here = len(frame_mod.select(frame, frame_mod.SAMPLE, ids))
        options = all_option + [
            {"label": f"Deep-dive sample ({n_here})", "value": frame_mod.SAMPLE},
        ] + [{"label": f"  cell: {cell} ({len(frame_mod.select(frame, cell, ids))})",
              "value": cell} for cell in cells]
        return options, frame_mod.ALL, False, frame_mod.summary(frame, ids)

    @app.callback(
        Output("subset-banner", "children"),
        Input("store-config", "data"),
        Input("run-a", "value"),
    )
    def subset_banner(config, run_a):
        """The one place the purposive-sample warning is written."""
        from apps.phenojury_ui.views.common import bundle_for, subset_label

        bundle = bundle_for(run_a)
        if bundle is None:
            return None
        subset = (config or {}).get("subset") or frame_mod.ALL
        if subset == frame_mod.ALL:
            return None
        n = len(frame_mod.select(state.get_registry().frame, subset, bundle["report_ids"]))
        return theme.note(
            f"Showing the {subset_label(subset)}, {n} report(s) of "
            f"{len(bundle['report_ids'])}. {frame_mod.WARNING} The verification gates below are "
            "unaffected: they compare against the shipped artifacts and always run on the full "
            "cohort.", "warn")

    @app.callback(
        Output("store-config", "data"),
        Output("cfg-k-wrap", "style"),
        Input("run-a", "value"),
        Input("cfg-rule", "value"),
        Input("cfg-k", "value"),
        Input("cfg-models", "value"),
        Input("cfg-mincount", "value"),
        Input("cfg-subset", "value"),
    )
    def update_config(run_a, rule, k, models, min_count, subset):
        """The only writer of ``store-config``. k is hidden under plurality, which has no k."""
        config = {"run_id": run_a, "rule": rule or "vote_k", "k": k or 1,
                  "models": models or [], "min_count": min_count or 1,
                  "subset": subset or frame_mod.ALL}
        hide = {"display": "none"} if rule == "plurality" else {"marginTop": "10px"}
        return config, hide

    @app.callback(
        Output("run-summary", "children"),
        Input("run-a", "value"),
        Input("store-theme", "data"),
    )
    def run_summary(run_a, mode):
        # Absolute, not relative: this module is __main__ when the app is launched as a script,
        # so it has no parent package and `from .views...` raises ImportError inside the callback
        #, a 500 on one panel, not a startup failure, which is why it needs a test.
        from apps.phenojury_ui.views.common import bundle_for, load_failure

        bundle = bundle_for(run_a)
        if bundle is None:
            failure = load_failure(run_a)
            if failure:
                # Name the exception here, not in the tab body: the sidebar is the one part
                # of the page visible no matter which view is open.
                return theme.note(f"{run_a}: failed to load, "
                                  f"{failure.strip().splitlines()[-1]}", "danger")
            return theme.note("No cohort loaded.", "warn")
        return _summary_block(bundle, mode or "light")

    # ── clientside: theme, presentation mode, copy-PNG ───────────────────────
    app.clientside_callback(
        """
        function(mode) {
            document.body.setAttribute('data-theme', mode || 'light');
            return mode;
        }
        """,
        Output("store-theme", "data"),
        Input("theme-toggle", "value"),
    )

    app.clientside_callback(
        """
        function(n) {
            const on = (n || 0) % 2;
            document.body.setAttribute('data-present', on ? '1' : '0');
            return on;
        }
        """,
        Output("store-present", "data"),
        Input("present-btn", "n_clicks"),
    )

    # One registration covers every Copy-PNG button on every panel: the button's own id carries
    # The graph it belongs to.
    app.clientside_callback(
        """
        function(n_clicks, id) {
            if (!n_clicks) { return window.dash_clientside.no_update; }
            const domId = JSON.stringify(id, Object.keys(id).sort());
            window.phenorag.copyPng(id.graph, document.getElementById(domId));
            return '';
        }
        """,
        Output({"type": "copy-sink", "graph": MATCH}, "children"),
        Input({"type": "copy-png", "graph": MATCH}, "n_clicks"),
        State({"type": "copy-png", "graph": MATCH}, "id"),
        prevent_initial_call=True,
    )


def _summary_block(bundle: dict, mode: str) -> html.Div:
    """The sidebar's "what is this cohort" block, including the verification verdict."""
    info = bundle["models_info"]
    silent = [m for m in bundle["all_models"] if m not in bundle["models"]]
    summary = bundle["gate_summary"]
    tone = {"pass": "info", "fail": "danger", "skip": "warn"}[summary["status"]]
    return html.Div([
        html.Label("Cohort", className="field-label"),
        html.Div(f"{len(bundle['report_ids'])} reports · "
                 f"{sum(len(v) for v in bundle['gold'].values())} annotated terms · "
                 f"{sum(len(v) for v in bundle['sentences'].values())} sentences",
                 className="stat-sub"),
        html.Div(f"{len(bundle['models'])} voting model(s)"
                 + (f", {len(silent)} extracted but not grounded: {', '.join(silent)}"
                    if silent else ""),
                 className="stat-sub"),
        html.Div(f"min_detection_count = {bundle['min_count']}"
                 + ("" if bundle["min_count_confirmed"] else " (assumed, could not be confirmed "
                    "against the shipped sets)"),
                 className="stat-sub"),
        theme.note(summary["headline"], tone),
        html.Div("PhenoBERT evidence: " + (
            "available" if bundle["has_phenobert"]
            else "not kept, the negated / not-linked split is unavailable"),
            className="stat-sub"),
        html.Div(_pb_baseline_line(bundle), className="stat-sub"),
    ])


def _pb_baseline_line(bundle: dict) -> str:
    """One line on whether the PhenoBERT baseline comparison is live for this cohort."""
    pb = bundle.get("pb_standalone")
    if not pb:
        return ("PhenoBERT baseline: not found, so the comparison tab and the annotated "
                "report are unavailable. Set the path under Data source.")
    return (f"PhenoBERT baseline: {len(pb['report_ids'])} reports, "
            f"{len(pb['detections'])} detections, {len(pb['texts'])} report texts")


def build_app(registry: Registry) -> dash.Dash:
    """Construct the Dash app. Separated from :func:`main` so the selftest can build it too."""
    app = dash.Dash(
        __name__,
        title="PhenoJury · Free Listing generation run",
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(_HERE, "assets"),
    )
    app.layout = build_layout(registry)
    register_callbacks(app)
    return app


def main() -> None:
    """Discover the runs, then serve the app."""
    ap = argparse.ArgumentParser(description="PhenoJury deep-dive UI for the Free Listing generation run")
    ap.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE,
                    help="Folder holding phenojury_generation_free_listing/<cohort>/ (also accepts the "
                         "experiment folder or a single cohort folder)")
    ap.add_argument("--pb-base", default=None,
                    help="Folder holding the PhenoBERT baseline (baseline_phenobert). Omit to derive it from "
                         "each cohort's run directory, which is what the cluster layout gives.")
    ap.add_argument("--frame", default=None,
                    help="Folder holding the deep-dive sampling frame (frame.json). Omit to find "
                         "hcy_deepdive_frame/ beside the output base, which is where "
                         "apps/compare_ui/select_hcy_documents.py puts it by default.")
    ap.add_argument("--annotations-dir", default=None,
                    help="Where manual SLM annotations are recorded. Omit to keep them beside "
                         "each run, in <run_dir>/slm_annotations/.")
    ap.add_argument("--cache-dir", default=None,
                    help="Where run reports are cached (default ~/.cache/phenorag_exp13_06_ui)")
    ap.add_argument("--port", type=int, default=8054)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the verification gates and render every view headlessly, then exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    registry = Registry(output_base=args.output_base, cache_dir=args.cache_dir,
                        pb_base=args.pb_base, frame_base=args.frame,
                        annotations_dir=args.annotations_dir)
    state.set_registry(registry)

    for problem in registry.validate():
        logger.warning("%s", problem)
    logger.info("Found %d cohort(s) under %s",
                len(registry.list_runs()), args.output_base)
    if registry.frame:
        logger.info("Deep-dive frame: %s (%d reports, %d cells)", registry.frame_path,
                    len(registry.frame["selected"]), len(registry.frame["cells"]))
    else:
        logger.info("No deep-dive frame found, the subset control will be hidden.")

    if args.selftest:
        from apps.phenojury_ui.selftest import run_selftest

        sys.exit(0 if run_selftest(registry, build_app) else 1)

    # Both slow imports are independent of the run, so pay them here, not inside the first
    # callback, where the cost lands on a click that looks like it should be instant, and where a
    # failure looks like a broken cohort.
    registry.get_tree_safe()
    # Both the ontology and the search index over it are independent of the run, so pay them here
    #, not inside the first callback, where an 18 000-term scan lands on a keystroke.
    registry.search
    pinned, why_not = votes.driver_status()
    if not pinned:
        logger.warning("Running unpinned: %s", why_not)

    app = build_app(registry)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
