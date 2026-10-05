"""Five main methods against one document, side by side. Login node + ``ssh -L``.

    python apps/compare_ui/app.py --port 8057
    # locally:  ssh -L 8057:localhost:8057 <cluster>  ->  http://localhost:8057

CPU-only, and lighter than every other app here: it reads the bundles
``apps/compare_ui/build.py`` wrote and opens no experiment directory, no ontology and no model. That
is the whole point of the split -- the expensive half ran once on the cluster, and what is left is
a few hundred kilobytes per document.

``--bundles`` points at the *parent* of the per-cohort bundle directories, and every build found
under it becomes an entry in the **Dataset** picker. Today that is two: the twenty HCY reports of
the deep-dive frame, and the twenty GSC+ abstracts drawn on where PhenoJury, RAG-HPO and AutoPCR
agree and disagree. They are separate registries rather than one merged list, because they have
different ground truths, different document sets and different rows in ``comparison``'s table -- merging them
would produce a screen on which no number means one thing. Pointing ``--bundles`` at a single
cohort's directory still works and gives a picker with one entry.

Run the builder first::

    sbatch slurm/compare_ui_bundles.sbatch            # both cohorts, on LeoMed

Self-verifying, with no data at all::

    python apps/compare_ui/app.py --selftest

Exit codes: 0 ok, 1 selftest failed, 2 nothing to serve (bad paths -- it refuses to start, not serving an empty shell), 3 the port is taken.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import dash  # noqa: E402
from dash import Input, Output, dcc, html  # noqa: E402

# Absolute imports on purpose: a relative one breaks when this file is ``__main__``, which is
# how it is run.
from apps.compare_ui import bundles, registry as registry_mod, sources, state, theme  # noqa: E402
from apps.compare_ui.registry import Registry  # noqa: E402
from apps.compare_ui.views import provenance, reader, scorecard  # noqa: E402

logger = logging.getLogger(__name__)

#: The next free port. 8050-8056 are taken by the other apps. See their READMEs.
DEFAULT_PORT = 8057

#: ``(id, label, module)``. Every view is mounted at once and toggled by ``display`` -- a panel
#: removed from the layout takes its callbacks' targets with it, and Dash then errors on every
#: callback bound to it.
VIEWS = (
    (reader.VIEW_ID, "Reader", reader),
    (scorecard.VIEW_ID, "Scorecard", scorecard),
    (provenance.VIEW_ID, "Provenance", provenance),
)


def build_layout(registries: dict, default_key: str = "") -> html.Div:
    """The shell: a sidebar of controls and a main column of tabs.

    The stores sit **outside** ``.shell``, matching ``app/phenojury_generation_free_listing``. ``.shell`` is a flex row
    whose two children are the sidebar and the main column (``app/tree_ui/assets/style.css``), so
    anything else placed in it becomes a third flex item.

    The dataset picker is always in the layout, even with one dataset. Rendering a control only
    when it has a choice to offer means every callback that reads it has to defend against its
    absence, and ``suppress_callback_exceptions`` turns that defence into silence, not an
    error. With one dataset it is disabled and still says which one is open.
    """
    keys = registry_mod.order(registries)
    default_key = default_key or (keys[0] if keys else "")
    current = registries.get(default_key)
    dataset_options = [{"label": registries[k].label(), "value": k} for k in keys]

    sidebar = html.Div(
        [
            html.Div([
                html.Div("Method comparison", className="brand"),
                html.Div(current.blurb() if current else "no bundles", className="brand-sub",
                         id="sidebar-blurb"),
            ]),

            html.Div([
                html.Div("Dataset", className="field-label"),
                dcc.Dropdown(id="cohort-picker", options=dataset_options, value=default_key,
                             clearable=False, disabled=len(dataset_options) < 2),
            ], className="field"),

            html.Div([
                html.Div("Document", className="field-label"),
                dcc.Dropdown(id="report-picker", options=[], value=None, clearable=False),
            ], className="field"),

            html.Div(id="sidebar-summary", className="sidebar-section"),

            html.Details([
                html.Summary("Data source", className="details-summary"),
                html.Div(id="sidebar-source"),
            ], className="sidebar-section"),

            html.Div([
                dcc.RadioItems(id="theme-toggle", value="light", inline=True,
                               options=[{"label": "light", "value": "light"},
                                        {"label": "dark", "value": "dark"}]),
                dcc.Checklist(id="present-toggle", value=[], inline=True,
                              options=[{"label": "presentation", "value": "1"}]),
            ], className="sidebar-section"),
        ],
        className="sidebar",
    )

    main = html.Div(
        [
            dcc.Tabs(id="tabs", value=VIEWS[0][0], className="tabs-bar",
                     children=[dcc.Tab(label=label, value=vid) for vid, label, _ in VIEWS]),
            # Every view is in the layout at once and toggled by display, so a callback bound to
            # one of its components always has a target. Only the open one draws -- each render
            # callback gates on `tabs` through views.common.should_render.
            html.Div([html.Div(module.layout(), id="view-" + vid) for vid, _, module in VIEWS]),
        ],
        className="main",
    )

    return html.Div([
        dcc.Store(id="store-theme", data="light"),
        # Browser state, and small: a cohort key, a document id and a clicked term.
        # The bundles hold patient report text and stay in the process. What crosses the wire is
        # one rendered page.
        dcc.Store(id="store-selection", data=None),
        html.Div(id="sink-clientside", style={"display": "none"}),
        html.Div([sidebar, main], className="shell"),
    ])


def report_options(registry) -> list:
    """Dropdown options for the reports of *registry*."""
    if registry is None:
        return []
    return [{"label": _report_label(registry, rid), "value": rid}
            for rid in registry.report_ids()]


def _report_label(registry, report_id: str) -> str:
    row = registry.rows().get(report_id) or {}
    cell = row.get("cell")
    return "{}  ·  {} annotated{}".format(report_id, row.get("n_gold", 0),
                                     "  ·  " + cell if cell else "")


def register_callbacks(app) -> None:
    """Register the page-level callbacks (dataset, report and tab selection)."""
    for _vid, _label, module in VIEWS:
        module.register(app)

    @app.callback([Output("view-" + vid, "style") for vid, _, _ in VIEWS],
                  Input("tabs", "value"))
    def switch_view(active):
        return [{"display": "block"} if vid == active else {"display": "none"}
                for vid, _, _ in VIEWS]

    # Switching dataset replaces the document list, not filtering it. The two cohorts share
    # no document ids today, but relying on that would be relying on an accident: a stale id left
    # in the picker resolves to no bundle and the page renders empty with nothing to explain it.
    @app.callback(Output("report-picker", "options"),
                  Output("report-picker", "value"),
                  Output("sidebar-blurb", "children"),
                  Output("sidebar-source", "children"),
                  Input("cohort-picker", "value"))
    def switch_cohort(cohort):
        registry = state.registry(cohort)
        options = report_options(registry)
        return (options,
                options[0]["value"] if options else None,
                registry.blurb() if registry else "no bundles",
                _source_panel(registry))

    @app.callback(Output("sidebar-summary", "children"),
                  Input("cohort-picker", "value"),
                  Input("report-picker", "value"))
    def summary(cohort, report_id):
        registry = state.registry(cohort)
        if registry is None:
            return theme.empty("No bundles.")
        rows = registry.rows()
        bits = [theme.stat(registry.unit(len(rows)), len(rows), "in this sample")]
        skipped = registry.skipped()
        if skipped:
            bits.append(theme.stat("skipped", len(skipped), "not drawable", tone="warning"))
        row = rows.get(report_id) or {}
        if row:
            bits.append(theme.stat("annotated terms", row.get("n_gold", 0), report_id or ""))
        return theme.stat_row(bits)

    # Clearing the selection when the document changes is a correctness fix, not a nicety: a term
    # id is not unique to a document, so a stale selection would open a reasoning panel for a term
    # The new document happens to share, under the previous one's evidence. The same argument
    # applies with more force across datasets.
    @app.callback(Output("store-selection", "data", allow_duplicate=True),
                  Input("report-picker", "value"),
                  Input("cohort-picker", "value"), prevent_initial_call=True)
    def clear_selection(_report_id, _cohort):
        return None

    app.clientside_callback(
        "function(mode) { document.body.setAttribute('data-theme', mode || 'light'); return mode; }",
        Output("store-theme", "data"), Input("theme-toggle", "value"))

    app.clientside_callback(
        "function(v) { document.body.setAttribute('data-present', (v && v.length) ? '1' : '0');"
        " return ''; }",
        Output("sink-clientside", "children"), Input("present-toggle", "value"))


def _source_panel(registry):
    if registry is None:
        return html.Div("no bundles", className="mono cmp-path")
    return html.Div([
        html.Div("bundles", className="field-label"),
        html.Div(registry.bundles_dir, className="mono cmp-path"),
        html.Div("comparison tables", className="field-label"),
        html.Div(registry.comparison_path(), className="mono cmp-path"),
        html.Div("scored as", className="field-label"),
        html.Div(registry.scored_cohort(), className="mono cmp-path"),
    ])


def build_app(registries, default_key: str = "") -> dash.Dash:
    """Construct the Dash app. Separated from :func:`main` so the selftest can build it too.

    Accepts a single ``Registry`` as well as a map, so a caller with one cohort in hand does not
    have to know about the map at all.
    """
    if isinstance(registries, Registry):
        registries = {registries.cohort.key: registries}
    app = dash.Dash(
        __name__,
        title="Method comparison",
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(_HERE, "assets"),
    )
    app.layout = build_layout(registries, default_key or state.default_key())
    register_callbacks(app)
    return app


# -- CLI ---------------------------------------------------------------------

def parse_args(argv=None):
    """Parse the command-line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundles", default=sources.DEFAULT_BUNDLE_ROOT,
                        help="the directory apps/compare_ui/build.py wrote -- either the parent of "
                             "the per-cohort directories, or one of them")
    parser.add_argument("--output-base", default=sources.DEFAULT_OUTPUT_BASE,
                        help="only used to resolve a bundle's recorded input paths and to read "
                             "the comparison tables")
    parser.add_argument("--cohort", default="",
                        help="which dataset to open on; default is the first one found")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 on a login node, where the tunnel terminates locally")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="render every panel against a synthetic cohort and exit")
    parser.add_argument("--check", action="store_true",
                        help="report on the configured bundles and exit")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def check(args) -> int:
    """Print what each discovered bundle set contains, without serving. Returns the exit status."""
    registries = registry_mod.discover(args.bundles, args.output_base)
    if not registries:
        print("ERROR: " + _nothing_found(args), file=sys.stderr)
        return 2
    for key in registry_mod.order(registries):
        registry = registries[key]
        print("")
        print("dataset       {}  ({})".format(key, registry.label()))
        print("bundles       {}".format(registry.bundles_dir))
        print("built         {}".format(registry.index.get("built_at")))
        print("ground truth          {}".format(registry.index.get("gold_dataset")))
        print("scored as     {}".format(registry.scored_cohort()))
        print("documents     {}".format(len(registry.report_ids())))
        for report_id in registry.report_ids():
            drift = registry.drift(report_id)
            print("  {}  {}  {}".format(
                report_id, registry.cell_of(report_id),
                "DRIFTED: " + ", ".join(drift) if drift else "ok"))
        for report_id, why in sorted(registry.skipped().items()):
            print("  {}  SKIPPED: {}".format(report_id, why))
        rows = registry.comparison_rows()
        print("comparison    {} row(s) from {}".format(len(rows), registry.comparison_path()))
        if not rows:
            # Not an error: the cohort table is a nice-to-have on a machine that has the bundles
            # but not the experiment tree. The reader tab is unaffected. Only the Scorecard loses
            # a panel.
            print("              (the Scorecard's cohort table will be empty)")
    return 0


def _nothing_found(args) -> str:
    """Why there is nothing to serve, in the words of the fix."""
    if not os.path.isdir(args.bundles):
        return ("no bundle directory at {} -- run slurm/compare_ui_bundles.sbatch on LeoMed "
                "first".format(args.bundles))
    return ("{} holds no readable build -- a build is a directory with an index.json of schema "
            "{}. Looked at it and at its immediate subdirectories, and no deeper.".format(
                args.bundles, bundles.SCHEMA_VERSION))


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def main(argv=None) -> int:
    """Discover the bundles, then serve the app (or run ``--check``). Returns the exit status."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    if args.selftest:
        from apps.compare_ui import selftest

        return selftest.run_app()
    if args.check:
        return check(args)

    registries = registry_mod.discover(args.bundles, args.output_base)
    if not registries:
        # Refuse, not serving an empty shell: a reader who reaches a page with an empty
        # dropdown assumes the build produced nothing, which is a different problem.
        print("ERROR: " + _nothing_found(args), file=sys.stderr)
        return 2
    order = registry_mod.order(registries)
    default_key = args.cohort if args.cohort in registries else order[0]
    if args.cohort and args.cohort not in registries:
        print("ERROR: no build for cohort {!r} under {}; found {}".format(
            args.cohort, args.bundles, ", ".join(order)), file=sys.stderr)
        return 2
    state.set_registries(registries, default_key)

    if not _port_free(args.host, args.port):
        print("ERROR: port {} is already in use. Try --port {}, and tunnel that instead:\n"
              "  ssh -L {p}:localhost:{p} <cluster>".format(
                  args.port, args.port + 1, p=args.port + 1), file=sys.stderr)
        return 3

    app = build_app(registries, default_key)
    for key in order:
        logger.info("dataset %-4s %d %s from %s", key, len(registries[key].report_ids()),
                    registries[key].unit(), registries[key].bundles_dir)
    logger.info("serving on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
