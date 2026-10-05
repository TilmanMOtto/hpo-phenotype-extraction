"""The acceptance check, for both halves, against a synthetic cohort and never against real data.

A Dash callback that raises does not crash the process. It returns a 500 for one component and the
page renders around the hole. On an analysis app that is a missing chart. Here it is a **column** --
and a column that failed to draw is indistinguishable from a method that found nothing, which is
the exact conclusion this app exists to help somebody reach. So the panels are rendered outside the
browser and the failures are collected.

Four layers, in the order a failure is cheapest to diagnose:

1. **The build, on both cohorts.** The fixture's outcomes are arithmetic somebody worked out by
   hand, so this can assert the numbers and not merely that a number appeared: the alt id
   resolves to a true positive, the negated detection is a false negative *with a reason that
   says so*, the tree re-run's accepted set equals the pooled prediction file it is explaining,
   and the un-alignable report is skipped with a named cause rather than drawn. The GSC+ pass
   then covers what HCY cannot reach: the other decoder, all four rungs of the RAG-HPO ground truth's
   placement ladder, and a TreePhenoRAG column that has a verdict and no re-run because the
   cohort was never folded.
2. **The render matrix.** Every panel, both themes, and the degenerate states -- no bundle, a
   selection naming a term this report does not have, a method whose artifacts were missing, a
   annotated term with no segment, a reason kind no renderer knows.
3. **The layout.** No duplicate string ids, and every callback ``Input``/``State`` names a
   component that exists. ``suppress_callback_exceptions=True`` is necessary here (the views are
   toggled, not unmounted) and its cost is that a typo'd id is silently inert: the callback never
   fires and the panel it drove stays blank with nothing in the log.
4. **HTTP.** ``/``, ``/_dash-layout`` and ``/_dash-dependencies`` through the test client.

Then the guard's record is drained. A panel that failed inside ``common.guard`` returned a ``Div``,
so the render matrix saw a pass. Draining is what catches those.
"""

from __future__ import annotations

import json
import logging
import tempfile

logger = logging.getLogger(__name__)


class Report:
    """Failures are collected, not fatal. One run should name every broken thing, not the first."""

    def __init__(self, title: str):
        self.title = title
        self.passed = 0
        self.failures: list = []

    def check(self, name: str, condition, detail: str = "") -> bool:
        """Record a named check that passes when *condition* is true. Returns *condition*."""
        if condition:
            self.passed += 1
            return True
        self.failures.append("{}: {}".format(name, detail) if detail else name)
        return False

    def nonempty(self, name: str, value) -> bool:
        """A panel that returns an empty list is a failure, not a pass."""
        children = getattr(value, "children", value)
        empty = value is None or (isinstance(children, (list, tuple)) and not children)
        return self.check(name, not empty, "rendered nothing")

    def expect(self, name: str, got, want) -> bool:
        """Record a named check that passes when *got* equals *want*."""
        return self.check(name, got == want, "got {!r}, want {!r}".format(got, want))

    def finish(self) -> int:
        """Print the summary and return the exit status (0 when every check passed)."""
        total = self.passed + len(self.failures)
        if self.failures:
            print("\n{}: {} of {} checks FAILED".format(self.title, len(self.failures), total))
            for failure in self.failures:
                print("  - " + failure)
            return 1
        print("{}: {} checks passed".format(self.title, total))
        return 0


# -- layer 1: the build ------------------------------------------------------

def run_builder(report: "Report | None" = None) -> int:
    """Build bundles from the synthetic fixtures and check them. Returns the exit status."""
    from apps.compare_ui import bundles, fixture

    own = report is None
    report = report or Report("build")

    root = tempfile.mkdtemp(prefix="compare_ui_selftest_")
    written = fixture.write(root)
    ctx, built = fixture.build(written)
    run_builder_gsc(report)
    skipped = built.pop("_skipped")
    codes = fixture._codes()

    report.check("the un-alignable report is skipped", "SYN003" in skipped)
    report.check("and the skip names a cause",
                 "segmentation" in (skipped.get("SYN003") or ""), repr(skipped.get("SYN003")))
    report.check("the drawable reports are built", set(built) == {"SYN001", "SYN002"},
                 sorted(built))

    bundle = built.get("SYN001") or {}
    if not bundle:
        report.failures.append("SYN001 did not build; the remaining build checks cannot run")
        return report.finish() if own else 1

    # -- the schema holds, and the cap is respected
    try:
        bundles.validate(bundle)
        report.passed += 1
    except Exception as exc:                                     # noqa: BLE001
        report.failures.append("bundle does not validate: {}".format(exc))
    size = len(json.dumps(bundle))
    report.check("the bundle is far inside the cap", size < bundles.MAX_BUNDLE_BYTES,
                 "{:,} bytes".format(size))

    # -- latin1 survived, and a drawn span really spells its trigger word
    segment = bundle["segments"][1]["text"]
    report.check("the report decoded latin1", "Anfälle" in segment, repr(segment))
    gold = {row["hpo_id"]: row for row in bundle["gold"]}
    row = gold.get(codes["C"]) or {}
    span = row.get("span") or [0, 0]
    report.expect("the ground truth span spells its trigger", segment[span[0]:span[1]], row.get("trigger"))
    report.check("an annotated term with no trigger stays unplaced",
                 (gold.get(codes["F"]) or {}).get("segment_idx") is None)

    # -- the outcomes are the ones worked out by hand
    want = {
        "phenobert": (1, 1, 2),
        "autopcr_70b": (2, 0, 1),
        "raghpo_70b": (2, 0, 1),
        "phenojury": (3, 0, 0),
        "treephenorag": (1, 1, 2),
    }
    for key, (tp, fp, fn) in want.items():
        score = bundle["metrics"][key]
        report.expect("SYN001 " + key, (score["tp"], score["fp"], score["fn"]), (tp, fp, fn))

    # -- the alt id is a true positive, which is the whole point of resolving
    raghpo = bundle["methods"]["raghpo_70b"]
    outcome = next((m["outcome"] for m in raghpo["marks"] if m["hpo_id"] == codes["C"]), None)
    report.expect("RAG-HPO's alt id resolves to a true positive", outcome, "tp")

    # -- PhenoBERT distinguishes its three ways of losing a term
    pb = bundle["methods"]["phenobert"]["reasons"]
    report.check("a negated detection says so",
                 "negated" in (pb.get(codes["D"]) or {}).get("why", ""),
                 (pb.get(codes["D"]) or {}).get("why"))
    report.check("a term never detected says that instead",
                 "never detected" in (pb.get(codes["F"]) or {}).get("why", ""),
                 (pb.get(codes["F"]) or {}).get("why"))

    # -- AutoPCR keeps the route, which is its whole contribution
    ap = bundle["methods"]["autopcr_70b"]["reasons"]
    routes = {m["route"] for m in (ap.get(codes["C"]) or {}).get("mentions") or ()}
    report.expect("AutoPCR records the linking route", routes, {"dictionary"})
    report.check("and a losing candidate is not reported as never retrieved",
                 "candidate" in (ap.get(codes["F"]) or {}).get("why", ""),
                 (ap.get(codes["F"]) or {}).get("why"))

    # -- PhenoJury uses repetition 0's configuration, not repetition 1's
    pj = bundle["methods"]["phenojury"]["reasons"][codes["C"]]
    report.expect("PhenoJury takes k from repetition 0", pj["k"], 2)
    report.expect("and the unit from repetition 0", pj["unit"], "segment")
    report.check("the vote was recomputed from the jurors", pj["vote_recomputed"])
    report.expect("three jurors named it", pj["votes"], 3)
    report.check("the generations are stored once at block level",
                 "apertus|1" in (bundle["methods"]["phenojury"].get("generations") or {}))

    # -- the tree re-run agrees with the prediction set it is explaining
    tree = bundle["methods"]["treephenorag"]
    accepted = {code for code, reason in tree["reasons"].items() if reason.get("accepted")}
    report.expect("the re-run's accepted set is the pooled prediction set",
                  accepted, set(tree["predicted"]))
    pruned = tree["reasons"][codes["F"]]
    report.expect("a pruned annotated term names its blocking ancestor", pruned.get("lost_at"),
                  codes["E"])
    report.expect("and files it under pruning", pruned.get("bucket"), "pruning")
    scored = tree["reasons"][codes["D"]]
    report.check("a scored-but-rejected term is distinguished from a pruned one",
                 scored.get("visited") and not scored.get("accepted"),
                 "visited={} accepted={}".format(scored.get("visited"), scored.get("accepted")))
    report.check("the tree reason carries the calls behind it",
                 len(tree["reasons"][codes["C"]].get("calls") or ()) > 0)

    # -- every method has an outcome for every annotated term
    for key, block in bundle["methods"].items():
        drawn = {m["hpo_id"] for m in block["marks"]}
        report.check("{} covers every annotated term".format(key),
                     set(gold) <= drawn, sorted(set(gold) - drawn))

    # -- writing round-trips
    out = str(root) + "/bundles"
    for report_id, one in built.items():
        bundles.write(out, one)
    from apps.compare_ui import build as build_mod

    bundles.write_index(out, build_mod.make_index(ctx, dict(built, _skipped=skipped)))
    back = bundles.read(out, "SYN001")
    report.check("a written bundle reads back", back is not None)
    report.expect("and is unchanged", (back or {}).get("metrics"), bundle["metrics"])
    report.check("nothing has drifted immediately after a build",
                 bundles.drifted(bundle, written["output_base"]) == [],
                 bundles.drifted(bundle, written["output_base"]))

    return report.finish() if own else 0


def run_builder_gsc(report: "Report") -> None:
    """The same builder over the GSC+ fixture -- the paths HCY cannot exercise.

    Everything asserted here is a decision that differs between the cohorts and would
    otherwise be checked by nothing: which decoder read the corpus, which rung of
    ``gold.RagHpoGold``'s ladder placed each term, and what a column does when the artifacts
    behind its reasoning do not exist for this cohort at all.
    """
    from apps.compare_ui import bundles, fixture

    root = tempfile.mkdtemp(prefix="compare_ui_selftest_gsc_")
    written = fixture.write_gsc(root)
    ctx, built = fixture.build_gsc(written)
    skipped = built.pop("_skipped")
    codes = fixture._codes()

    report.expect("every GSC+ abstract builds", sorted(built), ["GSC001", "GSC002"])
    report.check("and none is skipped", not skipped, skipped)
    # The corpus reader drops the Windows sidecars the fixture writes beside every Text file.
    # Without that filter the cohort reads as four documents, which is invisible on the
    # cluster and wrong everywhere else.
    report.expect("the Zone.Identifier sidecars are not documents", len(ctx.texts), 2)

    bundle = built.get("GSC001") or {}
    if not bundle:
        report.failures.append("GSC001 did not build; the GSC+ checks cannot run")
        return
    try:
        bundles.validate(bundle)
        report.passed += 1
    except Exception as exc:                                     # noqa: BLE001
        report.failures.append("GSC+ bundle does not validate: {}".format(exc))

    report.expect("the bundle names its cohort", bundle.get("cohort"), "gsc")
    report.check("and the abstract decoded UTF-8",
                 "seizures" in bundle["segments"][0]["text"],
                 repr(bundle["segments"][0]["text"]))

    # -- the placement ladder, rung by rung
    gold = {row["hpo_id"]: row for row in bundle["gold"]}
    corpus_row = gold.get(codes["C"]) or {}
    span = corpus_row.get("span") or [0, 0]
    report.expect("a corpus-annotated term is drawn on the corpus's own offsets",
                  bundle["segments"][corpus_row.get("segment_idx") or 0][
                      "text"][span[0]:span[1]], "seizures")
    report.expect("and names that rung", corpus_row.get("source"), "gsc+ annotation")
    report.check("a term the corpus also annotates carries no raghpo-only qualifier",
                 not corpus_row.get("qualifiers"), corpus_row.get("qualifiers"))

    described = gold.get(codes["D"]) or {}
    report.expect("a RAG-HPO-only term falls back to its description",
                  described.get("source"), "raghpo description match")
    report.check("and is drawn where that description occurs",
                 described.get("span") is not None, described)
    report.check("and is flagged as one the corpus does not annotate",
                 "raghpo-only" in (described.get("qualifiers") or ()),
                 described.get("qualifiers"))

    unplaced = gold.get(codes["F"]) or {}
    report.check("a term nobody located stays unplaced",
                 unplaced.get("segment_idx") is None and unplaced.get("span") is None,
                 unplaced)

    second = built.get("GSC002") or {}
    mention = {row["hpo_id"]: row for row in second.get("gold") or ()}.get(codes["C"]) or {}
    span = mention.get("span") or [0, 0]
    report.check("offsets that do not spell their mention fall back to locating it",
                 mention.get("source", "").startswith("gsc+ annotation (located"),
                 mention.get("source"))
    report.expect("and the located span spells the mention",
                  second["segments"][mention.get("segment_idx") or 0][
                      "text"][span[0]:span[1]], "seizures")

    # -- the outcomes, worked out by hand against ground truth {C, D, F}
    want = {
        "phenobert": (1, 0, 2),
        "autopcr_70b": (1, 1, 2),
        "raghpo_70b": (1, 0, 2),          # The ALT id again, resolved
        "phenojury": (2, 0, 1),
        "treephenorag": (1, 0, 2),
    }
    for key, (tp, fp, fn) in want.items():
        score = bundle["metrics"][key]
        report.expect("GSC001 " + key, (score["tp"], score["fp"], score["fn"]), (tp, fp, fn))

    # -- the transfer column: a verdict, no re-run, and a note that says which
    tree = bundle["methods"]["treephenorag"]
    report.expect("TreePhenoRAG still carries its transferred verdict",
                  set(tree["predicted"]), {codes["C"]})
    report.check("and says the cohort was never folded", "not folded" in (tree.get("note") or ""),
                 tree.get("note"))
    report.check("and no reason claims a re-run",
                 not any(r.get("available", False) for r in tree["reasons"].values()))

    # -- PhenoJury found this cohort's folds beside its own prediction set
    pj = bundle["methods"]["phenojury"]["reasons"][codes["D"]]
    report.check("PhenoJury resolved a fold configuration on GSC+", pj.get("known", True),
                 pj)
    report.check("and recomputed the vote from the jurors", pj.get("vote_recomputed"), pj)

    # -- the cohort's ground truth note reaches the index, where every view reads it from
    from apps.compare_ui import build as build_mod

    index = build_mod.make_index(ctx, dict(built, _skipped=skipped))
    report.expect("the index names the scored cohort", index.get("scored_cohort"),
                  "gsc_raghpo_ann")
    report.check("and carries the ground truth note", bool(index.get("gold_note")))


# -- layer 2-4: the app ------------------------------------------------------

def run_app() -> int:
    """Build the fixtures, start the app in-process and render every view. Returns the exit status."""
    # One report for both halves: a build failure and a render failure are both reasons this app
    # is wrong, and running the app checks anyway after a build failure is deliberate -- a render
    # bug is worth finding in the same pass, not one round trip later.
    report = Report("app")
    run_builder(report)

    from apps.compare_ui import (bundles, build as build_mod, fixture,
                                registry as registry_mod, sources, state)
    from apps.compare_ui.app import VIEWS, build_app, build_layout, report_options
    from apps.compare_ui.views import common, provenance, reader, reasons, scorecard

    root = tempfile.mkdtemp(prefix="compare_ui_selftest_app_")
    written = fixture.write(root)
    ctx, built = fixture.build(written)
    skipped = built.pop("_skipped")
    bundle_root = str(root) + "/bundles"
    out = bundle_root + "/hcy"
    for one in built.values():
        bundles.write(out, one)
    bundles.write_index(out, build_mod.make_index(ctx, dict(built, _skipped=skipped)))

    # The second dataset, written beside the first as the builder lays them out.
    gsc_written = fixture.write_gsc(str(root) + "/gsc_cohort")
    gsc_ctx, gsc_built = fixture.build_gsc(gsc_written)
    gsc_skipped = gsc_built.pop("_skipped")
    gsc_out = bundle_root + "/gsc"
    for one in gsc_built.values():
        bundles.write(gsc_out, one)
    # With its frame, so the purposive-sample banner is exercised on this cohort too -- its cells
    # are defined on which methods got a term right, which is the kind of rate a reader
    # must not carry off the screen.
    gsc_frame = sources.frame_of(fixture.paths_for_gsc(gsc_written))
    bundles.write_index(gsc_out, build_mod.make_index(
        gsc_ctx, dict(gsc_built, _skipped=gsc_skipped), gsc_frame))

    registries = registry_mod.discover(bundle_root, written["output_base"])
    report.expect("both datasets are discovered", registry_mod.order(registries),
                  ["hcy", "gsc"])
    state.set_registries(registries, "hcy")
    registry = state.registry("hcy")
    report.check("the registry finds the build", registry.ok, registry.problem())
    report.expect("and lists the drawable reports", sorted(registry.report_ids()),
                  ["SYN001", "SYN002"])
    report.check("and carries the builder's skip note", "SYN003" in registry.skipped())

    gsc_registry = state.registry("gsc")
    report.expect("the GSC+ dataset lists its abstracts", sorted(gsc_registry.report_ids()),
                  ["GSC001", "GSC002"])
    report.expect("and knows which comparison row is its own", gsc_registry.scored_cohort(),
                  "gsc_raghpo_ann")
    report.expect("and calls its documents abstracts", gsc_registry.unit(1), "abstract")
    report.check("and carries the ground truth note onto every screen",
                 bool(gsc_registry.gold_note()))
    # A cohort nobody built must resolve to the default, not to None: a view handed
    # None renders an empty page that reads as a build having produced nothing.
    report.check("an unknown cohort falls back to the default",
                 state.registry("no_such_cohort") is registry)

    codes = fixture._codes()
    bundle = registry.bundle("SYN001")

    # -- layer 2: the render matrix -----------------------------------------
    selections = [
        None,
        {"method": "gold", "hpo": codes["C"]},
        {"method": "gold", "hpo": codes["F"]},                 # unplaced ground truth
        {"method": "phenobert", "hpo": codes["D"]},            # negated
        {"method": "autopcr_70b", "hpo": codes["C"]},          # a route and a menu
        {"method": "raghpo_70b", "hpo": codes["D"]},           # a menu, no route
        {"method": "phenojury", "hpo": codes["C"]},            # generations
        {"method": "treephenorag", "hpo": codes["F"]},         # pruned, with an ancestor
        {"method": "treephenorag", "hpo": codes["M"]},         # a false positive
        {"method": "phenobert", "hpo": "HP:9999999"},          # a term this report never saw
        {"method": "no_such_method", "hpo": codes["C"]},       # a column that does not exist
        {"method": "gold", "hpo": "HP:9999999"},               # not an annotated term here
    ]
    for mode in ("light", "dark"):
        for report_id in ("SYN001", "SYN002", "", "SYN999"):
            one = registry.bundle(report_id) if report_id else None
            report.nonempty("head[{} {}]".format(mode, report_id or "none"),
                            reader.render_head(registry, one))
            grid = reader.render_grid(registry, one)
            report.nonempty("grid[{} {}]".format(mode, report_id or "none"), grid)
            report.nonempty("scorecard[{} {}]".format(mode, report_id or "none"),
                            scorecard.render(registry, report_id))
            report.nonempty("provenance[{} {}]".format(mode, report_id or "none"),
                            provenance.render(registry, report_id))
        for selection in selections:
            label = "{}/{}".format((selection or {}).get("method"), (selection or {}).get("hpo"))
            report.nonempty("reason[{} {}]".format(mode, label),
                            reasons.render(bundle, selection))

    # A reason payload whose kind nobody registered must still draw, visibly unfinished.
    strange = {"kind": "not_a_kind", "why": "invented", "mystery": 7}
    report.nonempty("reason[unknown kind]",
                    reasons.RENDERERS.get("not_a_kind", reasons._unknown)(
                        bundle, bundle["methods"]["phenobert"], strange))

    # A method whose artifacts were missing greys out, not breaking the grid.
    broken = json.loads(json.dumps(bundle))
    broken["methods"]["phenobert"].update({"status": "missing", "marks": [], "reasons": {}})
    report.nonempty("grid[missing column]", reader.render_grid(registry, broken))

    # -- the same matrix over the GSC+ dataset, whose ground truth and columns differ
    gsc_bundle = gsc_registry.bundle("GSC001")
    gsc_selections = [
        None,
        {"method": "gold", "hpo": codes["C"]},
        {"method": "gold", "hpo": codes["F"]},              # unplaced, footer row
        {"method": "phenojury", "hpo": codes["D"]},         # a vote on an abstract
        {"method": "autopcr_70b", "hpo": codes["M"]},       # a false positive
        {"method": "treephenorag", "hpo": codes["C"]},      # a transferred verdict, no re-run
        {"method": "raghpo_70b", "hpo": codes["F"]},        # a miss on its authors' own ground truth
    ]
    for mode in ("light", "dark"):
        for report_id in ("GSC001", "GSC002", "", "GSC999"):
            one = gsc_registry.bundle(report_id) if report_id else None
            report.nonempty("gsc head[{} {}]".format(mode, report_id or "none"),
                            reader.render_head(gsc_registry, one))
            report.nonempty("gsc grid[{} {}]".format(mode, report_id or "none"),
                            reader.render_grid(gsc_registry, one))
            report.nonempty("gsc scorecard[{} {}]".format(mode, report_id or "none"),
                            scorecard.render(gsc_registry, report_id))
            report.nonempty("gsc provenance[{} {}]".format(mode, report_id or "none"),
                            provenance.render(gsc_registry, report_id))
        for selection in gsc_selections:
            label = "{}/{}".format((selection or {}).get("method"), (selection or {}).get("hpo"))
            report.nonempty("gsc reason[{} {}]".format(mode, label),
                            reasons.render(gsc_bundle, selection))

    # The dataset picker offers both, and switching it replaces the document list, not
    # leaving a stale id that resolves to no bundle.
    values = [option["value"] for option in report_options(gsc_registry)]
    report.expect("switching dataset re-lists the documents", values, ["GSC001", "GSC002"])
    report.check("and shares no id with the other dataset",
                 not set(values) & set(registry.report_ids()))

    # -- layer 3: the layout -------------------------------------------------
    app = build_app(registries, "hcy")
    ids = _string_ids(app.layout)
    report.check("the dataset picker is in the layout", "cohort-picker" in ids, sorted(ids))
    # A single-dataset build must still mount the picker: every render callback reads it, and
    # suppress_callback_exceptions turns a missing input into silence, not an error.
    report.check("and is mounted even with one dataset",
                 "cohort-picker" in _string_ids(build_layout({"hcy": registry}, "hcy")))
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    report.check("no duplicate component ids", not duplicates, duplicates)

    known = set(ids)
    missing = []
    for callback in app.callback_map.values():
        for item in list(callback.get("inputs") or ()) + list(callback.get("state") or ()):
            cid = item.get("id")
            if not isinstance(cid, str) or _is_pattern(cid):
                # A pattern-matching id names no single component, by design. Dash serialises it
                # into the callback map as a JSON *string*, which is why the check has to parse it
                #, not test its Python type.
                continue
            if cid not in known:
                missing.append(cid)
    report.check("every callback input names a real component", not missing, sorted(set(missing)))
    report.expect("every view is mounted", len(VIEWS), 3)

    # -- layer 4: HTTP -------------------------------------------------------
    unreadable = _unreadable_assets(app)
    if unreadable:
        # Not a failure and not a silent pass. ``assets/style.css`` and ``assets/copy.js`` are
        # symlinks into ``app/tree_ui/assets`` -- the repo-wide convention, and correct on the
        # cluster. A Windows process reaching the checkout over ``\wsl.localhost`` cannot follow
        # them, and Dash stats every asset before serving a page. Saying so beats reporting a
        # filesystem limitation as a broken app.
        print("app: HTTP layer SKIPPED -- these assets are not readable from this filesystem: "
              + ", ".join(unreadable))
        print("     (WSL symlinks through a Windows path; run the selftest on the cluster to "
              "exercise the routes)")
    else:
        client = app.server.test_client()
        for route in ("/", "/_dash-layout", "/_dash-dependencies"):
            response = client.get(route)
            report.expect("GET {}".format(route), response.status_code, 200)

    # -- drain the guard -----------------------------------------------------
    failures = common.failures()
    for key, trace in sorted(failures.items()):
        report.failures.append("guarded panel {} raised:\n{}".format(key, trace))
    common.clear_failures()

    return report.finish()


def _is_pattern(component_id: str) -> bool:
    """Whether a callback-map id is a pattern-matching id, not a component name."""
    try:
        return isinstance(json.loads(component_id), dict)
    except ValueError:
        return False


def _unreadable_assets(app) -> list:
    """Asset files Dash will stat and this filesystem cannot. See the HTTP layer for why."""
    import os

    folder = app.config.assets_folder
    out = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
        try:
            os.stat(os.path.join(folder, name))
        except OSError:
            out.append(name)
    return out


def _string_ids(component) -> list:
    """Every string id in a layout tree. Pattern-matching ids are dicts and are skipped."""
    found = []
    stack = [component]
    while stack:
        node = stack.pop()
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            found.append(cid)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found
