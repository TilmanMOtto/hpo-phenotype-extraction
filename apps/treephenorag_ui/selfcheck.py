"""``--selfcheck``, validate every discovered run, then exit.

The unit tests run against a ten-node toy ontology and a hand-written fixture. They establish that
the code is right. They cannot establish that it is right *about the cluster's files*. This does,
and it is meant to be run once on the login node before trusting anything the UI displays.

Six checks, in increasing order of how much they would ruin your day:

1. **Discovery.** Every cell, its status, its configurations, and any shard problems.
2. **Prediction coverage.** Every ``tau_*`` of a cell covers the same reports. A τ covering fewer
   is a half-written directory, and its scorecard would silently describe a subset of the cohort.
3. **Internal consistency.** ``predicted_set`` on each summary line equals the set of nodes with
   ``accepted = true`` in that τ's ``*_nodes.jsonl``. These are written by two different code
   paths from the same traversal result. A disagreement means one of the two files is stale.
4. **Ground truth agreement.** How many nodes' baked-in ``is_gold`` flag disagrees with the ground truth file the
   UI is configured to use. Non-zero is not an error, it is the expected consequence of scoring
   HCY against ``hcy_ground_truth_raw.csv`` when inference used ``hcy_ground_truth.csv``, but it
   must be *seen*, because it silently changes every TP, FP and FN on the page.
5. **The cross-check.** Recomputed micro P/R/F1 against ``result_tables``'s own ``core_quality.csv``.
   This is the strongest end-to-end evidence available: discovery, loading, ground truth joining and
   scoring all have to be right for the numbers to land, and ``result_tables`` computed its copy
   through an entirely separate code path.
6. **Evidence recovery.** Whether the Stanza segmentation resolves the ``sent_index`` values the
   call records actually reference, and whether the reconstructed prompt still matches
   ``hpo_extraction.treephenorag.verifier_prompt``.
7. **The curated ground truth.** Whether the loaded HCY ground truth is a curated dataset at all, whether its
   three files are three views of one build, the two-column ground truth, the annotation table and the
   manifest that every provenance line quotes, and how much of each HCY run its cohort covers.
   The last is the number that makes a recall on screen interpretable: the curated ground truth holds the
   reports curation reached, the rest are dropped by ``scoring.align`` in silence, and a rate over
   48 of 60 reports must say so.

Exit code is 0 when nothing failed, 1 when any check did. Warnings do not fail the run.
"""

from __future__ import annotations

import csv
import logging
import os

log = logging.getLogger("exp13_ui.selfcheck")

#: The cross-check passes if the recomputed micro F1 is within this of the result-table library's. Not zero:
#: The result-table library may have been run against a different ground truth file or before a shard merge, and the point
#: is to catch a *structural* disagreement, not floating-point noise.
F1_TOLERANCE = 1e-6


class Result:
    """Accumulates lines and a pass/fail verdict, so one printer handles every check."""

    def __init__(self):
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        """Record a passing check."""
        log.info("  PASS  %s", message)

    def warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(message)
        log.warning("  WARN  %s", message)

    def fail(self, message: str) -> None:
        """Record a failing check."""
        self.failures.append(message)
        log.error("  FAIL  %s", message)


def run(registry) -> int:
    """Run every check against ``registry``. Returns a process exit code."""
    result = Result()

    log.info("── 1. discovery ─────────────────────────────────────────────")
    cells = list(registry.cells.values())
    if not cells:
        result.fail(f"no TreePhenoRAG run directories under {registry.output_base}")
        return _finish(result)
    for cell in cells:
        line = (f"{cell.cell_id:<26} {cell.status:<9} "
                f"{len(cell.operating_points)} point(s)  "
                f"{' '.join(cell.operating_points)}")
        (result.warn if cell.status in ("partial", "missing") else result.ok)(line)
        for problem in cell.problems:
            log.info("          %s", problem)

    for problem in registry.validate():
        result.warn(problem)

    log.info("── 2-4. per-cell consistency ────────────────────────────────")
    for cell in cells:
        _check_cell(registry, cell, result)

    log.info("── 5. cross-check against the result tables' core_quality.csv ─────────")
    _cross_check(registry, cells, result)

    log.info("── 6. evidence recovery ─────────────────────────────────────")
    _check_evidence(registry, cells, result)

    log.info("── 7. PhenoBERT annotation overlay ──────────────────────────")
    _check_phenobert(registry, cells, result)

    log.info("── 8. HCY deep-dive sampling frame ──────────────────────────")
    _check_sample(registry, cells, result)

    log.info("── 9. curated HCY ground truth ──────────────────────────────────────")
    _check_curated(registry, cells, result)

    return _finish(result)


def _finish(result: Result) -> int:
    log.info("─────────────────────────────────────────────────────────────")
    if result.failures:
        log.error("%d check(s) FAILED, %d warning(s)", len(result.failures),
                  len(result.warnings))
        return 1
    log.info("all checks passed, %d warning(s)", len(result.warnings))
    return 0


def _check_cell(registry, cell, result: Result) -> None:
    from . import loaders

    points = cell.operating_points or [None]
    coverage: dict[str | None, set[str]] = {}
    # Points whose directory exists but was never written are a separate diagnosis from a
    # half-written one: the run was stopped before reaching them, which is a fact about the job,
    # not a fault in the data. They are named and then left out of the coverage comparison.
    never_written = set(cell.empty_operating_points)
    if never_written:
        result.warn(f"{cell.cell_id}: {len(never_written)} configuration(s) exist but were "
                    f"never written: {' '.join(sorted(never_written))}")

    for op in points:
        if op in never_written:
            continue
        predictions_fs = cell.files_for("predictions", op)
        if not predictions_fs.present:
            result.fail(f"{cell.cell_id} @ {op}: no predictions artifact")
            continue
        try:
            predictions = loaders.load_predictions(predictions_fs)
        except Exception as exc:
            result.fail(f"{cell.cell_id} @ {op}: predictions unreadable, {exc}")
            continue

        coverage[op] = set(predictions["predicted"])
        if predictions["reports_without_summary"]:
            result.fail(
                f"{cell.cell_id} @ {op}: {len(predictions['reports_without_summary'])} report(s) "
                f"have prediction rows but no summary line, the file is truncated "
                f"(first: {predictions['reports_without_summary'][:3]})")

        if cell.spec.is_tree:
            _check_accepted_matches_predicted(registry, cell, op, predictions, result)

    sizes = {op: len(reports) for op, reports in coverage.items()}
    if len(set(sizes.values())) > 1:
        result.fail(f"{cell.cell_id}: written configurations cover different report sets "
                    f"{sizes}, at least one directory is half-written, and its scorecard would "
                    f"describe a subset of the cohort")
    elif sizes:
        result.ok(f"{cell.cell_id}: {next(iter(sizes.values()))} reports, consistent across "
                  f"{len(sizes)} written configuration(s)")

    _check_gold(registry, cell, cell.written_operating_points[0] if cell.operating_points
                else None, result)


def _check_accepted_matches_predicted(registry, cell, op, predictions, result: Result) -> None:
    """``predicted_set`` (predictions file) must equal ``accepted`` (nodes file), report by report.

    Two files, two writers, one traversal result. If they disagree, one of them is from a previous
    run of the same directory, which resume and shard re-writes make entirely possible.
    """
    from . import loaders

    nodes_fs = cell.files_for("nodes", op)
    if not nodes_fs.present:
        result.warn(f"{cell.cell_id} @ {op}: no nodes artifact, cannot cross-check predictions")
        return
    try:
        nodes_df = loaders.load_nodes(nodes_fs)
    except Exception as exc:
        result.fail(f"{cell.cell_id} @ {op}: nodes unreadable, {exc}")
        return
    if nodes_df.empty:
        result.warn(f"{cell.cell_id} @ {op}: nodes artifact is empty")
        return

    accepted = {
        str(rid): set(group.loc[group["accepted"].to_numpy(dtype=bool), "hpo_id"].astype(str))
        # observed=True: report_id is categorical (loaders._nodes_frame), and a groupby over a
        # categorical otherwise yields an empty group for every unused category.
        for rid, group in nodes_df.groupby("report_id", observed=True, sort=False)
    }
    mismatches = []
    for report_id, predicted in predictions["predicted"].items():
        from_nodes = accepted.get(str(report_id))
        if from_nodes is None:
            mismatches.append(f"{report_id}: in predictions, absent from nodes")
        elif from_nodes != set(predicted):
            mismatches.append(
                f"{report_id}: {len(set(predicted) - from_nodes)} only in predictions, "
                f"{len(from_nodes - set(predicted))} only in nodes")

    if mismatches:
        result.fail(f"{cell.cell_id} @ {op}: predicted_set disagrees with accepted nodes in "
                    f"{len(mismatches)} report(s), e.g. {mismatches[:3]}")
    else:
        result.ok(f"{cell.cell_id} @ {op}: predicted_set == accepted nodes in every report")


def _check_gold(registry, cell, op, result: Result) -> None:
    if not registry.gold_available(cell.cohort):
        # Not configured is not broken. Point the sidebar at it or pass the flag. Skipping is the
        # only honest verdict, since nothing about this cell can be scored either way.
        result.warn(f"{cell.cell_id}: no ground truth configured for the {cell.cohort} cohort, "
                    f"skipped")
        return
    try:
        bundle = registry.bundle(cell.cell_id, op)
    except Exception as exc:
        result.fail(f"{cell.cell_id}: could not build, {exc}")
        return
    if not bundle:
        return

    counts = (bundle["report"].get("counts") or {})
    disagreements = counts.get("gold_disagreements") or 0
    n_visited = counts.get("n_visited_nodes") or 0
    gold_name = os.path.basename(registry.gold_source(cell.cohort) or "?")
    if disagreements:
        result.warn(
            f"{cell.cell_id}: {disagreements}/{n_visited} nodes' is_gold flag disagrees with "
            f"{gold_name}. Expected if inference used a different ground truth file, but every TP/FP/FN "
            f"on the page follows {gold_name}, not the flag.")
    else:
        result.ok(f"{cell.cell_id}: is_gold agrees with {gold_name} on every node")

    if not bundle["gold"]:
        result.fail(f"{cell.cell_id}: no report of this run has a ground truth entry in {gold_name}, "
                    f"the report ids in the artifacts and in the ground truth file do not match")


def _exp13_07_gold(tables_dir: str) -> dict:
    """The ground truth paths ``result_tables`` actually scored against, from its ``config_resolved.yaml``.

    Sits next to ``tables/``. Reading it is what turns the cross-check from a report about which
    ground truth file the user picked into a test of this UI's code path: the result-table library scored HCY against
    ``hcy_ground_truth_raw.csv`` while *inference* used ``hcy_ground_truth.csv``, and comparing
    across that difference can only ever fail. The app now defaults to the result-table library file, so the
    common case agrees, but the file is still read rather than assumed, because a user who
    switched the sidebar back would otherwise get a cross-check that silently means nothing.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(tables_dir or ".")),
                        "config_resolved.yaml")
    if not os.path.isfile(path):
        return {}
    wanted = {"hcy_gt_path", "gsc_dir"}
    found: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.partition(":")
            if sep and key.strip() in wanted:
                found[key.strip()] = value.strip().strip("'\"")
    return found


def _scoring_registry(registry, result: Result):
    """A registry pointed at the result-table library's ground truth files, or the original when they cannot be found.

    Returns ``(registry, note)``. The note is printed once so the reader always knows which ground truth
    the comparison used, a silently substituted ground truth file would be the mistake this
    whole module exists to catch.
    """
    from .registry import Registry

    config = _exp13_07_gold(registry.thesis_tables_dir)
    hcy = config.get("hcy_gt_path") or registry.hcy_gt_path
    gsc = config.get("gsc_dir") or registry.gsc_dir
    if (hcy, gsc) == (registry.hcy_gt_path, registry.gsc_dir):
        return registry, "using the configured ground truth files"

    if hcy and not os.path.isfile(hcy):
        result.warn(f"The result tables scored HCY against {hcy}, which is not readable, cross-checking "
                    f"against {os.path.basename(registry.hcy_gt_path)} instead, so an HCY "
                    f"mismatch below may only mean the two ground truth files differ")
        return registry, "using the configured ground truth files"

    scoped = Registry(
        registry.output_base, hcy_gt_path=hcy, hcy_input_dir=registry.hcy_input_dir,
        gsc_dir=gsc, stanza_dir=registry.stanza_dir,
        cache_dir=registry.cache_dir, thesis_tables_dir=registry.thesis_tables_dir,
    )
    return scoped, (f"scoring against the result tables' own ground truth "
                    f"(HCY: {os.path.basename(hcy)}, GSC+: {os.path.basename(gsc)}), "
                    f"which is not necessarily the file the UI displays")


def _cross_check(registry, cells, result: Result) -> None:
    path = os.path.join(registry.thesis_tables_dir or "", "core_quality.csv")
    if not os.path.isfile(path):
        result.warn(f"no result-table core_quality.csv at {path}, skipping the cross-check "
                    f"(pass --thesis-tables to point at it)")
        return

    with open(path, "r", encoding="utf-8") as f:
        published = list(csv.DictReader(f))

    registry, note = _scoring_registry(registry, result)
    log.info("  ....  %s", note)

    checked = 0
    for row in published:
        cell_id = f"{row.get('method')}/{row.get('cohort')}"
        cell = registry.cell(cell_id)
        if cell is None:
            continue
        op = row.get("operating_point")
        op = None if op in ("-", "", None) else op
        if op and op not in cell.operating_points:
            continue
        try:
            report = registry.get_report(cell_id, op)
        except Exception as exc:
            result.fail(f"{cell_id} @ {op}: could not score, {exc}")
            continue
        if not report:
            continue

        checked += 1
        for key in ("micro_precision", "micro_recall", "micro_f1"):
            theirs = _float(row.get(key))
            mine = (report.get("flat") or {}).get(key)
            if theirs is None or mine is None:
                continue
            if abs(mine - theirs) > F1_TOLERANCE:
                result.fail(
                    f"{cell_id} @ {op or '-'}: {key} {mine:.6f} != the result tables' {theirs:.6f} "
                    f"(ground truth: {os.path.basename(registry.gold_source(cell.cohort))}). Either the "
                    f"artifacts changed since the result tables were written, or this UI and they disagree "
                    f"about how the sets are compared.")
                break
        else:
            result.ok(f"{cell_id} @ {op or '-'}: micro P/R/F1 match the result tables")

    if not checked:
        result.warn("core_quality.csv named no cell that is present on disk, nothing "
                    "cross-checked")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_evidence(registry, cells, result: Result) -> None:
    from . import loaders

    tree_cells = [c for c in cells if c.spec.is_tree and c.files_for("calls").present]
    if not tree_cells:
        result.warn("no tree cell has a calls artifact, sentence recovery not checked")
        return

    cell = tree_cells[0]
    evidence = registry.evidence()
    _check_report_texts(registry, cell, result)
    segments = evidence.segments(cell.cohort)
    if not segments:
        result.warn(f"sentence text unavailable ({evidence.status(cell.cohort)}), the deep-dive "
                    f"will show indices and logits only")
        return
    # Which of the five routes answered is part of the evidence, not a detail: only the Stanza one
    # reproduces inference by design, and the ``sent_index`` check below is what certifies
    # The others. Naming the source next to that check is what makes the pair readable.
    result.ok(f"segmented {len(segments)} report(s) of {cell.cohort}, "
              f"{evidence.source(cell.cohort)}")
    if "UNVERIFIED" in evidence.source(cell.cohort):
        result.warn("the segmentation was served without being checked against the report text, "
                    "because the reports are not reachable. Set the HCY report folder: a "
                    "segmentation of a different document resolves every index and shows the "
                    "wrong sentence for all of them.")

    scan = registry.calls_scan(cell.cell_id)
    if not scan:
        return
    checked = incomplete = missing = 0
    for report_id in list(scan.get("offsets", {}))[:10]:
        calls = loaders.calls_for_report(scan, report_id)
        if calls.empty or "sent_index" not in calls.columns:
            continue
        # A report the cohort map does not hold is the *partial* case, and it is worth separating:
        # The cohort looks recovered, and this one report shows a wall of indices.
        if str(report_id) not in segments:
            missing += 1
            continue
        coverage = evidence.coverage(cell.cohort, report_id,
                                     calls["sent_index"].dropna().tolist())
        checked += 1
        if not coverage["complete"]:
            incomplete += 1
            result.fail(
                f"{cell.cell_id} report {report_id}: only {coverage['n_resolved']}/"
                f"{coverage['n_referenced']} sent_index values resolve against a "
                f"{coverage['n_sentences']}-sentence segmentation (max index "
                f"{coverage['max_index']}). The segmentation here differs from the one inference "
                f"used, so every sentence the deep-dive shows would be the wrong one.")
    if checked and not incomplete:
        result.ok(f"every sent_index resolves in {checked} sampled report(s)")
    if missing:
        result.warn(f"{missing} of the sampled report(s) have no recovered segmentation, though "
                    f"the cohort as a whole does, the deep-dive shows indices only for those")

    _check_prompt(registry, evidence, cell, scan, result)


def _check_report_texts(registry, cell, result: Result) -> None:
    """That this app reads the reports byte for byte the way inference read them.

    Not a formality. ``loaders.load_report_texts`` decoded utf-8 while ``hpo_extraction.data.loading.load_txt``
    decodes latin1, and since a recovered segmentation is verified by locating its sentences in this
    text, every German report, which is all of them, failed that check and was silently dropped.
    A decoding difference has no symptom until it is essential, so it is fixed here, not
    trusted.
    """
    from . import loaders

    if cell.cohort != "hcy" or not registry.hcy_input_dir:
        return
    try:
        loaders.assert_matches_load_txt(registry.hcy_input_dir)
    except AssertionError as exc:
        result.fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - hpo_extraction.data.loading needs sentence_transformers
        result.warn(f"could not fix the report reader against hpo_extraction.data.loading.load_txt ({exc})")
    else:
        result.ok("reports are decoded as hpo_extraction.data.loading.load_txt decodes them")


def _check_sample(registry, cells, result: Result) -> None:
    """Whether the deep-dive sample can actually be used against the runs that are here.

    Warnings only, like the PhenoBERT overlay and for the same reason: the subset filter is
    additive, and no frame simply means the control stays disabled. What must not happen silently
    is the *third* case, a frame that loads but whose report ids do not appear in any HCY run.
    That renders as a report dropdown that mysteriously does not narrow, and the cause (a draw made
    against a different cohort export, or against reports this run has not finished) is invisible
    on screen. Naming it here is the whole point of the check.
    """
    from . import sample

    frame = registry.sample
    if frame is None:
        result.warn(f"no deep-dive sampling frame at {registry.hcy_frame_dir or '(unset)'}, the "
                    f"Report subset control is disabled. Create one with "
                    f"apps/compare_ui/select_hcy_documents.py, or point --deepdive-frame at an existing "
                    f"draw.")
        return

    cell_summary = ", ".join(f"{cell}={len(frame.picks[cell])}" for cell in frame.cells)
    result.ok(f"{len(frame.selected)} report(s) over {len(frame.cells)} cell(s) from "
              f"{frame.path} (drawn {frame.generated or 'at an unrecorded date'}): {cell_summary}")

    if not frame.rows:
        result.warn(f"{frame.path}/{sample.POOLS_FILE} is missing or unreadable, so the deep dive "
                    f"can name each report's cell but not PhenoBERT's score on it")

    hcy_cells = [c for c in cells if c.cohort == sample.COHORT]
    if not hcy_cells:
        result.warn(f"a sampling frame is loaded but no {sample.COHORT.upper()} run was "
                    f"discovered, so the subset can never apply to anything on screen")
        return

    drawn = set(frame.selected)
    for cell in hcy_cells:
        try:
            ids = set(registry.report_ids(cell.cell_id))
        except Exception as exc:                                  # unreadable predictions
            result.warn(f"{cell.cell_id}: could not list report ids to check the frame against "
                        f"({exc})")
            continue
        if not ids:
            continue
        overlap = drawn & ids
        if not overlap:
            result.fail(
                f"{cell.cell_id}: none of the frame's {len(drawn)} report ids appears in this "
                f"run's {len(ids)}, the draw and the artifacts describe different report sets, so "
                f"the subset filter would silently do nothing. Check that "
                f"{frame.path}/{sample.FRAME_FILE} was drawn from the same cohort export.")
        elif len(overlap) < len(drawn):
            result.warn(
                f"{cell.cell_id}: {len(drawn) - len(overlap)} of the frame's {len(drawn)} reports "
                f"are absent from this run (it has {len(ids)}); the subset will show "
                f"{len(overlap)}. Unmerged shards do this.")
        else:
            result.ok(f"{cell.cell_id}: all {len(drawn)} sampled report(s) present")


def _check_phenobert(registry, cells, result: Result) -> None:
    """Whether the deep dive can underline the PhenoBERT baseline baseline's matches on each cohort.

    Warnings only, never a failure. The annotation layer is additive: a cohort with no PhenoBERT baseline
    run, or one whose staged text has drifted from its detections, costs the deep dive its
    underlines and nothing else. What counts is that the reason is *named* here, not
    discovered as a silently unannotated report, the two look identical on screen otherwise, and
    only one of them means "PhenoBERT found nothing".
    """
    from . import phenobert

    cohorts = sorted({c.cohort for c in cells})
    for cohort in cohorts:
        pb = registry.phenobert(cohort)
        if pb is None:
            result.warn(f"{cohort}: no PhenoBERT baseline run found, so the deep-dive report shows "
                        f"no baseline annotations. Looked in: "
                        f"{', '.join(phenobert.searched(registry, cohort)) or '(nowhere)'}")
            continue

        if pb["span_bad"]:
            result.warn(
                f"{cohort}: {len(pb['span_bad'])} of {pb['span_checked']} detection offsets do "
                f"not land on their own matched phrase in phenobert_input/, those reports are "
                f"shown unannotated rather than underlined in the wrong place. A restaged input "
                f"beside a stale {os.path.basename(pb['run_dir'])}/phenobert_detections.jsonl "
                f"does this.")

        # Alignment is the half the span check cannot see: the offsets can be perfect against the
        # staged text and still be unplaceable on *this* app's segmentation of the same report.
        evidence = registry.evidence()
        segments = evidence.segments(cohort)
        reports = [r for r in pb["report_ids"] if r in segments][:10]
        if not reports:
            result.warn(f"{cohort}: no report has both a segmentation and staged PhenoBERT text, "
                        f"so no match can be attributed to a sentence")
            continue

        aligned = sum(1 for r in reports
                      if not phenobert.annotate(pb, r, segments[r])[1])
        if aligned == len(reports):
            result.ok(f"{cohort}: {len(pb['detections'])} detection(s) from {pb['run_dir']}, "
                      f"placed onto the sentences of all {len(reports)} sampled report(s)")
        else:
            result.warn(f"{cohort}: only {aligned}/{len(reports)} sampled report(s) could have "
                        f"their PhenoBERT matches placed onto this app's segmentation "
                        f"({evidence.source(cohort)}); the rest render unannotated")


def _check_prompt(registry, evidence, cell, scan, result: Result) -> None:
    """Fix the reconstructed prompt, but only where the two sides read the same ontology.

    ``embed_symptom_in_prompt_v1`` builds its own ``HPOTree`` from ``resources/util/hpo.json``,
    while :meth:`Evidence.prompt` reads the shared view. Those are the same terms in every real
    run and are not when the view has been injected, a toy ontology has no definitions, so the
    comparison would report drift where there is none. Skipped, loudly, not asserted
    across two different ontologies.
    """
    from hpo_extraction.ontology.hpo_tree import HPOTree

    from . import loaders

    if not isinstance(registry.view.tree, HPOTree):
        result.warn("the shared ontology is not an HPOTree, so the prompt reconstruction cannot "
                    "be fixed against hpo_extraction.treephenorag.verifier_prompt (which builds its own)")
        return

    for report_id in list(scan.get("offsets", {}))[:5]:
        calls = loaders.calls_for_report(scan, report_id)
        if calls.empty:
            continue
        record = calls.iloc[0]
        sentence = evidence.sentence(cell.cohort, report_id, record.get("sent_index"))
        if not sentence:
            continue
        try:
            evidence.assert_prompt_matches(str(record["hpo_id"]), sentence)
        except AssertionError as exc:
            result.fail(f"prompt reconstruction drifted: {exc}")
            return
        except Exception as exc:
            result.warn(f"could not check the prompt reconstruction ({exc})")
            return
        result.ok("reconstructed prompt matches hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1")
        return
    result.warn("no call record with a resolvable sentence, prompt reconstruction not checked")


def _check_curated(registry, cells, result: Result) -> None:
    """Whether the curated HCY ground truth loaded, and whether it agrees with itself.

    The load itself is not the risk, an unreadable dataset is already a warning from
    ``registry.validate``. What this check is for is the three ways a curated dataset can be
    *present, readable and wrong about the run on screen*:

    **The ground truth file and the annotation table disagree.** They are two files written by one run of
    ``hcy_ground_truth``, and nothing on disk ties them together. A directory assembled by hand, or half
    overwritten by a second run, would give a scorecard whose metrics come from one policy and
    whose evidence column comes from another, and every number would still look plausible. The
    two-column file is re-read here and compared pair for pair against the ``in_gold`` rows.

    **The manifest disagrees with both.** It is what the scorecard's provenance quotes, so a
    manifest describing a different build is a provenance line that names the wrong policy.

    **The cohort does not overlap the run.** The curated ground truth covers the reports curation reached. If none of them is a report this run wrote, every metric is computed over nothing, and the
    scorecard shows a clean zero, not an error. ``_check_gold`` catches the total case per
    cell. This one reports the partial overlap, which is the normal state and the number a reader
    needs in order to know what the recall on screen is a recall over.
    """
    from . import curated as curated_mod

    directory = curated_mod.dataset_dir_of(registry.hcy_gt_path)
    if not directory:
        result.warn(f"the loaded HCY ground truth is {os.path.basename(registry.hcy_gt_path) or '(unset)'}"
                    f", not a curated dataset, the cohort panel, the trigger word behind each "
                    f"annotated term and the reason behind each excluded false positive are all off. "
                    f"Build one with experiments/03_setup/ground_truth, or pick an existing "
                    f"curated_ground_truth_<date>/ in Data sources.")
        return

    dataset = registry.curated
    if dataset is None:
        result.fail(f"{directory} looks like a curated dataset but could not be read")
        return
    result.ok(f"{dataset.describe()}")

    # 1. The two-column ground truth the app scores against, versus the dataset's own annotation table.
    try:
        scored = registry.gold("hcy")
    except Exception as exc:
        result.fail(f"the curated ground truth at {dataset.gold_path} could not be loaded, {exc}")
        return
    in_gold = {}
    for patient_id, rows in dataset.annotations.items():
        codes = {row.hpo_code for row in rows if row.in_gold}
        if codes:
            in_gold[patient_id] = codes
    if not dataset.annotations:
        result.warn(f"{dataset.name} has no {curated_mod.ANNOTATIONS_FILE}, the ground truth scores, but "
                    f"nothing can be said about where a term came from or why one was dropped")
    elif in_gold != {k: v for k, v in scored.items() if v}:
        only_gold = sum(len(v - in_gold.get(k, set())) for k, v in scored.items())
        only_table = sum(len(v - scored.get(k, set())) for k, v in in_gold.items())
        result.fail(f"{dataset.name}: {curated_mod.GOLD_FILE} and {curated_mod.ANNOTATIONS_FILE} "
                    f"disagree, {only_gold} pair(s) in the ground truth file alone, {only_table} in the "
                    f"annotation table alone. They must be two views of one build.")
    else:
        result.ok(f"{curated_mod.GOLD_FILE} matches the in_gold rows of "
                  f"{curated_mod.ANNOTATIONS_FILE} on all {dataset.n_gold_pairs} pair(s)")

    # 2. The manifest, which is what every provenance line quotes.
    counts = dataset.counts
    for key, actual in (("n_reports_in_cohort", len(dataset.gold)),
                        ("n_gold_pairs", dataset.n_gold_pairs)):
        stated = counts.get(key)
        if stated is None:
            result.warn(f"{dataset.name}: manifest states no {key}")
        elif int(stated) != actual:
            result.fail(f"{dataset.name}: manifest says {key}={stated}, the files say {actual}, "
                        f"the manifest describes a different build than the one loaded")
    if counts:
        result.ok(f"manifest counts agree with the files "
                  f"(policy: {', '.join(f'{k}={v}' for k, v in dataset.policy.items())})")

    # 3. how much of each HCY run the cohort actually covers.
    hcy_cells = [c for c in cells if c.cohort == curated_mod.COHORT]
    if not hcy_cells:
        result.warn(f"a curated {curated_mod.COHORT.upper()} ground truth is loaded but no "
                    f"{curated_mod.COHORT.upper()} run was discovered")
        return
    for cell in hcy_cells:
        try:
            ids = set(registry.report_ids(cell.cell_id))
        except Exception as exc:
            result.warn(f"{cell.cell_id}: could not list report ids to check cohort coverage "
                        f"({exc})")
            continue
        if not ids:
            continue
        outside = dataset.outside(ids)
        if not outside:
            result.ok(f"{cell.cell_id}: all {len(ids)} report(s) are in the curated cohort")
        elif len(outside) == len(ids):
            result.fail(f"{cell.cell_id}: not one of its {len(ids)} report(s) is in the curated "
                        f"cohort, every metric for this cell is computed over nothing")
        else:
            result.warn(f"{cell.cell_id}: {len(outside)} of {len(ids)} report(s) are outside the "
                        f"curated cohort and are not scored; every rate shown for this cell is "
                        f"over the remaining {len(ids) - len(outside)}")
