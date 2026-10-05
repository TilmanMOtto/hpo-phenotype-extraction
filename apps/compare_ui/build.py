"""Build the per-report bundles. Runs where the artifacts are. The app never runs this.

    python apps/compare_ui/build.py \\
        --output-base $REPO/output \\
        --hcy-dir     $BASE/hcy \\
        --frame       $REPO/output/hcy_deepdive_frame_2026-09-12 \\
        --out         $REPO/output/compare_ui_bundles/hcy

One pass, and the order of the two loops is the whole design. Adapters ``load`` once over the
cohort -- that is where a 585 MB score cache is re-run and eight juror files are read -- and then
``evidence`` per report, which only slices what is already in memory. Inverting that would reopen
every artifact twenty times.

Two decisions are made here rather than in any adapter, because five adapters must not be able to
disagree about them:

**What counts as a true positive.** Membership is tested on *resolved* codes
(``terms.resolve_all``), so a ground truth file written against an older HPO release and a method emitting
the current spelling of the same phenotype agree. Comparing raw strings would paint terms the
method actually found in red, and it would do so differently for different methods depending on
which release their vocabulary came from.

**Where the reader's coordinate system comes from.** ``sources.Context.report_view`` recovers it
and verifies it, and a document whose segmentation does not align to its own text is **skipped and
named** in the index, not drawn against a segmentation that is not the run's.

What the ground truth is, and therefore what a drawn underline means, is the one thing that genuinely
differs between the cohorts -- see :mod:`apps.compare_ui.gold`. Everything else a cohort changes is
a path, and those are named in :mod:`apps.compare_ui.cohorts`.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from apps.compare_ui import bundles, cohorts, roster, sources  # noqa: E402
from apps.compare_ui.methods import adapter_for  # noqa: E402

logger = logging.getLogger(__name__)


def build_all(ctx, report_ids, cells=None) -> dict:
    """Build every bundle in memory. Returns ``{report_id: bundle}`` plus a ``_skipped`` key.

    In memory, not streamed to disk because the caller decides where -- and because the
    selftest builds bundles it never writes. Twenty bundles of a few hundred kilobytes is not a
    memory problem. The adapters' cohort-wide state, which is already loaded, dwarfs it.
    """
    report_ids = [str(r) for r in report_ids]
    cells = cells or {}

    adapters = {}
    for method in roster.ROSTER:
        adapter = adapter_for(method.key)(ctx, method)
        try:
            adapter.load(report_ids)
        except Exception as exc:                          # pragma: no cover - defensive
            logger.exception("%s failed to load: %s", method.key, exc)
            adapter.status, adapter.note = "missing", "failed to load: {}".format(exc)
        adapters[method.key] = adapter

    out, skipped = {}, {}
    for report_id in report_ids:
        rv = ctx.report_view(report_id)
        if rv is None:
            skipped[report_id] = _skip_reason(ctx, report_id)
            continue
        out[report_id] = build_one(ctx, adapters, rv, cells.get(report_id, ""))
    out["_skipped"] = skipped
    return out


def build_one(ctx, adapters, rv, cell: str = "") -> dict:
    """One report's bundle."""
    from apps.treephenorag_ui import terms as terms_mod

    view = ctx.view
    gold_rows, gold_unplaced = _gold(ctx, rv)
    gold_codes = set(gold_rows) | set(gold_unplaced)

    methods, metrics = {}, {}
    mentioned = set(gold_codes)
    for key in roster.ORDER:
        adapter = adapters[key]
        method = roster.get(key)
        predicted = _resolve_map(view, adapter.predictions().get(rv.report_id) or set())
        outcomes = _outcomes(gold_codes, set(predicted))

        if adapter.status == "ok":
            marks, reasons = adapter.evidence(rv, outcomes, gold_rows)
            extras = adapter.block_extras() if hasattr(adapter, "block_extras") else {}
        else:
            marks, reasons, extras = [], {}, {}

        methods[key] = dict({
            "label": method.label,
            "short": method.short,
            "family": method.family,
            "status": adapter.status,
            "note": adapter.note,
            "evidence": method.evidence,
            "evidence_note": method.note_for(ctx.cohort),
            "predicted": sorted(predicted),
            "marks": marks,
            "reasons": reasons,
        }, **extras)
        metrics[key] = _score(outcomes)
        mentioned.update(outcomes)

    return {
        "schema": bundles.SCHEMA_VERSION,
        "report_id": rv.report_id,
        "cohort": ctx.cohort.key,
        "cell": cell,
        "built_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "inputs": dict(ctx.inputs),
        "gold_dataset": getattr(ctx.gold, "name", ""),
        "segments": _segments(rv),
        "tail": _tail(rv),
        "terms": {code: _term(terms_mod, view, code) for code in sorted(mentioned)},
        "gold": [gold_rows[c] for c in sorted(gold_rows)]
                + [gold_unplaced[c] for c in sorted(gold_unplaced)],
        "methods": methods,
        "metrics": metrics,
    }


# -- segments ----------------------------------------------------------------

def _segments(rv) -> list:
    """The segments, each carrying whatever text precedes it and belongs to no segment.

    Sentence tokenization does not partition the report: headers, list bullets, blank lines and
    the newline between paragraphs fall between segments. The reader is reading a clinical
    document, so dropping those would quietly rewrite it -- a heading that said "Labor:" would
    vanish and the values under it would read as prose. ``gap_before`` carries them, and the view
    renders whitespace as a break and anything else as dimmed text that is visibly not a segment.
    """
    out = []
    cursor = 0
    for idx, text in enumerate(rv.display):
        span = rv.ranges[idx]
        gap = ""
        if span is not None:
            gap = rv.text[cursor:span[0]] if span[0] > cursor else ""
            cursor = max(cursor, span[1])
        out.append({"idx": idx, "text": text, "gap_before": gap})
    return out


def _tail(rv) -> str:
    """Whatever follows the last aligned segment -- usually a trailing newline, sometimes not."""
    last = next((span for span in reversed(rv.ranges) if span is not None), None)
    return rv.text[last[1]:] if last is not None else ""


# -- ground truth --------------------------------------------------------------------

def _gold(ctx, rv):
    """``({resolved: row}, {resolved: row})`` -- this document's annotated terms, placed or not.

    One line, because the two cohorts' answers to "where was this phenotype read" are genuinely
    different procedures and each lives with its own ground truth. ``gold.CuratedGold`` runs the curated
    dataset's own placement ladder and evidence locations a trigger word inside the segment the curator named;
    ``gold.RagHpoGold`` falls back through the GSC+ corpus's mention offsets and then RAG-HPO's own
    description. Neither invents a position, and both report what they could not place.
    """
    if ctx.gold is None:
        return {}, {}
    return ctx.gold.rows(rv, lambda code: _resolve(ctx.view, code))


# -- outcomes and scoring ----------------------------------------------------

def _outcomes(gold: set, predicted: set) -> dict:
    """``{hpo_id: "tp"|"fp"|"fn"}`` over the union, in resolved ids."""
    out = {}
    for code in gold & predicted:
        out[code] = "tp"
    for code in predicted - gold:
        out[code] = "fp"
    for code in gold - predicted:
        out[code] = "fn"
    return out


def _score(outcomes: dict) -> dict:
    """Per-report counts and rates. A report with no ground truth gets ``None`` rates, never zero.

    Zero would be a claim -- "this method scored nothing here" -- about a report on which nothing
    could be scored, and it would then average into a mean that is not a mean of anything.
    """
    tp = sum(1 for v in outcomes.values() if v == "tp")
    fp = sum(1 for v in outcomes.values() if v == "fp")
    fn = sum(1 for v in outcomes.values() if v == "fn")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    elif tp + fn == 0:
        f1 = None
    else:
        f1 = 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# -- terms -------------------------------------------------------------------

def _term(terms_mod, view, code: str) -> dict:
    """Name, definition and the one hover wording every surface in the repo uses.

    ``terms.tooltip`` is reused, not reassembled so that hovering a term here says
    what hovering it in ``app/exp13_ui`` says. Baking it into the bundle is what lets the app hold
    no ontology at all: 15 MB of ``hpo.json`` against a few kilobytes of the terms this report
    actually mentions.
    """
    return {
        "label": terms_mod.label(view, code),
        "definition": terms_mod.definition(view, code),
        "tooltip": terms_mod.tooltip(view, code),
        "known": terms_mod.known(view, code),
    }


def _resolve(view, code: str) -> str:
    if view is None:
        return str(code)
    return str(view.resolve(str(code)) or code)


def _resolve_map(view, codes) -> dict:
    """``{resolved: as the method spelled it}`` -- so a panel can show both when they differ."""
    out = {}
    for code in codes:
        out.setdefault(_resolve(view, code), str(code))
    return out


def _skip_reason(ctx, report_id: str) -> str:
    if report_id not in ctx.texts:
        return "no document text under " + ctx.paths.input_dir
    if report_id not in ctx.segments:
        return "no row for this document in " + ctx.paths.segments_path
    return ("the recovered segmentation does not align to the document text, so no offset in any "
            "method's records can be placed on it")


# -- the index ---------------------------------------------------------------

def make_index(ctx, built: dict, frame=None) -> dict:
    """The bundle index: one row per built report with its cell, sizes and metrics."""
    skipped = built.get("_skipped") or {}
    reports = []
    for report_id in sorted(k for k in built if k != "_skipped"):
        bundle = built[report_id]
        reports.append({
            "report_id": report_id,
            "cell": bundle.get("cell", ""),
            "n_segments": len(bundle["segments"]),
            "n_gold": len(bundle["gold"]),
            "metrics": bundle["metrics"],
        })
    cohort = ctx.cohort
    return {
        "schema": bundles.SCHEMA_VERSION,
        "cohort": cohort.key,
        # Everything a view needs to speak about this cohort without holding a cohort table of its
        # own. The app reads bundles and nothing else, so a cohort it has never heard of must
        # still describe itself -- which is also what lets the reader tab be built once.
        "cohort_label": cohort.label,
        "cohort_blurb": cohort.blurb,
        "cohort_unit": cohort.unit,
        "cohort_units": cohort.units,
        "scored_cohort": cohort.scored_cohort,
        "gold_note": cohort.gold_note,
        "built_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "gold_dataset": getattr(ctx.gold, "name", ""),
        "frame": {
            "dir": (frame or {}).get("dir", ""),
            "generated": (frame or {}).get("generated", ""),
            "params": (frame or {}).get("params", {}),
            "inputs": (frame or {}).get("inputs", {}),
        } if frame else {},
        "methods": [{"key": m.key, "label": m.label, "short": m.short, "family": m.family,
                     "evidence": m.evidence, "evidence_note": m.note_for(cohort)}
                    for m in roster.ROSTER],
        "reports": reports,
        "skipped": skipped,
        "inputs": dict(ctx.inputs),
    }


# -- CLI ---------------------------------------------------------------------

def parse_args(argv=None):
    """Parse the command-line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", default=cohorts.DEFAULT.key, choices=list(cohorts.KEYS),
                        help="which cohort to build; one per run, each into its own "
                             "subdirectory of --bundle-root")
    parser.add_argument("--output-base", default=sources.DEFAULT_OUTPUT_BASE,
                        help="the directory holding the experiment output trees")
    parser.add_argument("--hcy-dir", default=sources.DEFAULT_HCY_DIR,
                        help="the HCY cohort directory (reports, segmentation, curated ground truth)")
    parser.add_argument("--gsc-dir", default=sources.DEFAULT_GSC_DIR,
                        help="the GSC+ corpus: Text/ and Annotations/")
    parser.add_argument("--raghpo-dir", default=sources.DEFAULT_RAGHPO_DIR,
                        help="RAG-HPO's vendored 114-document subset and its own annotation")
    parser.add_argument("--segments", default="",
                        help="override the segmentation CSV; default is the cohort's own")
    parser.add_argument("--gold", default="",
                        help="HCY only: a curated_ground_truth_<date> directory; default is the newest")
    parser.add_argument("--frame", default="",
                        help="a sampling-frame directory. Default is a search under --output-base "
                             "for HCY; GSC+ has no conventional location and must be named.")
    parser.add_argument("--bundle-root", default=sources.DEFAULT_BUNDLE_ROOT,
                        help="the parent directory of the per-cohort bundle directories")
    parser.add_argument("--out", default="",
                        help="where to write this cohort's bundles; default is "
                             "<bundle-root>/<cohort>")
    parser.add_argument("--reports", default="",
                        help="comma-separated document ids, overriding the frame's draw. For "
                             "debugging one document; the frame is what the app expects.")
    parser.add_argument("--selftest", action="store_true",
                        help="build from a synthetic fixture and check the result; no data needed")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Build one JSON bundle per selected report and the index. Returns the exit status."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    if args.selftest:
        from apps.compare_ui import selftest

        return selftest.run_builder()

    cohort = cohorts.get(args.cohort)
    out = args.out or sources.bundles_dir(args.bundle_root, cohort)
    paths = sources.Paths(output_base=args.output_base, cohort=cohort.key,
                          hcy_dir=args.hcy_dir, gsc_dir=args.gsc_dir,
                          raghpo_dir=args.raghpo_dir, segments=args.segments,
                          gold_dir=args.gold, frame_dir=args.frame)
    problems = paths.problems()
    if problems:
        for problem in problems:
            print("ERROR: " + problem, file=sys.stderr)
        return 2

    ctx = sources.build_context(paths)
    if ctx.gold is None:
        print("ERROR: no ground truth for cohort {}. HCY wants a curated_ground_truth_<date> directory under {}; "
              "GSC+ wants {}/annotations.csv.".format(cohort.key, paths.hcy_dir,
                                                      paths.raghpo_dir), file=sys.stderr)
        return 2

    frame = sources.frame_of(paths)
    if args.reports:
        report_ids = [r.strip() for r in args.reports.split(",") if r.strip()]
        cells = {r: (frame or {}).get("cell_of", {}).get(r, "") for r in report_ids}
    elif frame:
        report_ids = list(frame.get("selected") or ())
        cells = dict(frame.get("cell_of") or {})
    else:
        print("ERROR: no sampling frame found. Draw one with "
              "apps/compare_ui/{}, or pass --reports.".format(
                  "select_hcy_documents.py" if cohort.key == "hcy"
                  else "select_gsc_documents.py"), file=sys.stderr)
        return 2

    logger.info("building %d %s into %s", len(report_ids), cohort.units, out)
    built = build_all(ctx, report_ids, cells)
    skipped = built.pop("_skipped", {})

    total = 0
    for report_id, bundle in sorted(built.items()):
        path = bundles.write(out, bundle)
        total += path.stat().st_size
        logger.info("  %s  %6.1f KB  %s", report_id, path.stat().st_size / 1024,
                    bundle.get("cell", ""))

    index = make_index(ctx, dict(built, _skipped=skipped), frame)
    bundles.write_index(out, index)

    for report_id, why in sorted(skipped.items()):
        logger.warning("skipped %s: %s", report_id, why)
    logger.info("wrote %d bundle(s), %.1f KB total, %d skipped",
                len(built), total / 1024, len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
