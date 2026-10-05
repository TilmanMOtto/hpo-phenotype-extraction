"""Discovery: which earlier cells exist, and why a blank in the results is blank.

The two failure modes this guards are the ones that would corrupt the results chapter silently:
a cohort analysed from unmerged shards (a fraction of the data, presented as all of it), and a
metric printed blank without saying whether the run is missing or the metric does not apply.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_every_method_and_cohort_is_a_cell(exp13_modules, exp13_results_dir):
    """The grid is every registered method x both base cohorts, a method that never ran still
    gets a cell. Counted from the registry rather than hardcoded: adding a method to METHODS is
    routine, and a test that has to be edited alongside it teaches nothing."""
    d = exp13_modules.discovery
    avail = d.discover(exp13_results_dir)
    assert len(avail.cells) == len(d.METHODS) * len(d.BASE_COHORTS)


def test_present_cells_list_their_operating_points_in_sweep_order(exp13_modules,
                                                                  exp13_results_dir):
    """tau dirs sort numerically, not lexicographically, ``tau_0.5`` must not precede ``tau_0.1``."""
    cell = exp13_modules.discovery.discover(exp13_results_dir).get("tree_gate_lr", "hcy")
    assert cell.status == "present"
    assert cell.operating_points == ["tau_0.1", "tau_0.5"]


def test_tau_points_parse_in_both_layouts(exp13_modules):
    """Two tau-directory layouts coexist and both must parse.

    Runs that fix one accept threshold keep the original 1-D ``tau_{prune}``, that includes
    an earlier exploratory run and every dump written before the accept sweep existed, while a real 2-D sweep adds
    ``_acc_{accept}``. Scoring old and new dumps side by side depends on reading both.
    """
    parse = exp13_modules.discovery.parse_tau_point
    assert parse("tau_0.02") == (0.02, None)
    assert parse("tau_0.02_acc_0.9") == (0.02, 0.9)
    assert parse("tau_0.00015_acc_0.95") == (0.00015, 0.95)
    assert parse("tau_1e-05") == (1e-05, None)
    with pytest.raises(ValueError):
        parse("tau_junk")


def test_two_dimensional_tau_points_sort_as_a_grid(exp13_modules, tmp_path):
    """Sorted by (tau_prune, tau_accept) so a 2-D sweep reads as a grid, and a directory that is
    not an configuration at all is ignored, not crashing the sort."""
    d = exp13_modules.discovery
    for name in ["tau_0.1", "tau_0.02_acc_0.9", "tau_0.02_acc_0.5", "tau_0.005_acc_0.9", "figures"]:
        (tmp_path / name).mkdir()
    spec = next(m for m in d.METHODS if m.key == "tree_gate_lr")
    assert d._sweep_points(tmp_path, spec) == [
        "tau_0.005_acc_0.9", "tau_0.02_acc_0.5", "tau_0.02_acc_0.9", "tau_0.1",
    ]


def test_a_method_that_never_ran_is_missing_with_the_absent_path(exp13_modules, exp13_results_dir):
    """The RAG-HPO 70B baseline is not in the fixture. The cell must name the directory it looked for."""
    cell = exp13_modules.discovery.discover(exp13_results_dir).get("raghpo_70b", "hcy")
    assert cell.status == "missing"
    assert "baseline_raghpo_70b" in cell.detail


def test_a_flat_method_has_one_unnamed_operating_point(exp13_modules, exp13_results_dir):
    cell = exp13_modules.discovery.discover(exp13_results_dir).get("raghpo_8b", "gsc")
    assert cell.status == "present"
    assert cell.operating_points == [""]


def test_ensemble_points_sort_numerically_with_plurality_last(exp13_modules, exp13_results_dir):
    """``vote_k10`` must not sort before ``vote_k2``, and plurality is not a k."""
    cell = exp13_modules.discovery.discover(exp13_results_dir).get("slm_ensemble", "hcy")
    assert cell.operating_points == ["vote_k1", "vote_k2", "agg_plurality"]


def test_unmerged_shards_make_a_cell_partial_and_name_the_merge_command(
        exp13_modules, exp13_results_dir):
    """The one failure mode that would corrupt the results invisibly: a fraction of a cohort.

    A shard with no merged counterpart means the SLURM array's outputs were never collapsed, so
    anything computed from that directory describes some of the reports while claiming to
    describe all of them. It must be refused, with the fix named.
    """
    run_dir = exp13_results_dir / "exp13_05_topm_retrieval_slm" / "hcy"
    (run_dir / "flat_topm_predictions.jsonl").unlink()
    (run_dir / "flat_topm_s0of8_predictions.jsonl").write_text('{"report_id": "r1"}\n')

    cell = exp13_modules.discovery.discover(exp13_results_dir).get("flat_topm", "hcy")
    assert cell.status == "partial"
    assert not cell.usable
    assert "merge_shards.py" in cell.detail


def test_shards_left_beside_a_merged_file_are_not_a_problem(exp13_modules, exp13_results_dir):
    """``merge_shards.py`` writes the consolidated artifact next to its inputs and leaves them
    in place. Treating any shard as unmerged would permanently refuse every array job in the
    repo, including the ones that were merged correctly."""
    import os
    import time

    run_dir = exp13_results_dir / "exp13_05_topm_retrieval_slm" / "hcy"
    merged = run_dir / "flat_topm_predictions.jsonl"
    assert merged.is_file()                                        # The merged artifact
    for shard in ("flat_topm_s0of8_predictions.jsonl", "flat_topm_s1of8_predictions.jsonl"):
        (run_dir / shard).write_text('{"report_id": "r1"}\n')
    now = time.time()
    os.utime(merged, (now, now))                                   # merged after its inputs

    cell = exp13_modules.discovery.discover(exp13_results_dir).get("flat_topm", "hcy")
    assert cell.status == "present"
    assert cell.usable


def test_a_shard_newer_than_the_merged_file_is_flagged_as_stale(exp13_modules,
                                                                exp13_results_dir):
    """A resumed session wrote more reports after the merge ran, so they exist on disk but not in
    the merged artifact, which is the same silent-undercount failure, one step later."""
    import os
    import time

    run_dir = exp13_results_dir / "exp13_05_topm_retrieval_slm" / "hcy"
    merged = run_dir / "flat_topm_predictions.jsonl"
    shard = run_dir / "flat_topm_s0of8_predictions.jsonl"
    shard.write_text('{"report_id": "r1"}\n')
    old = time.time() - 3600
    os.utime(merged, (old, old))

    cell = exp13_modules.discovery.discover(exp13_results_dir).get("flat_topm", "hcy")
    assert cell.status == "partial"
    assert "newer than the merged artifact" in cell.detail


def test_shard_detection_reaches_into_tau_subdirectories(exp13_modules, exp13_results_dir):
    """Tree runs nest their per-tau files, so a flat glob would miss the shards entirely."""
    tau_dir = exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
    (tau_dir / "tree_gate_lr_nodes.jsonl").unlink()
    (tau_dir / "tree_gate_lr_s3of8_nodes.jsonl").write_text('{"report_id": "r1"}\n')

    cell = exp13_modules.discovery.discover(exp13_results_dir).get("tree_gate_lr", "hcy")
    assert cell.status == "partial"


def test_an_operating_point_without_predictions_is_dropped_not_averaged(
        exp13_modules, exp13_results_dir):
    """A tau that ran out of wall-clock mid-sweep must not contribute a partial cohort."""
    (exp13_results_dir / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.5"
     / "tree_gate_lr_predictions.jsonl").unlink()

    cell = exp13_modules.discovery.discover(exp13_results_dir).get("tree_gate_lr", "hcy")
    assert cell.status == "partial"
    assert cell.operating_points == ["tau_0.1"]
    assert "tau_0.5" in cell.detail


def test_resolved_files_point_at_the_drivers_actual_filenames(exp13_modules, exp13_results_dir):
    cell = exp13_modules.discovery.discover(exp13_results_dir).get("tree_gate_lr", "hcy")
    files = cell.files["tau_0.1"]
    assert files["predictions"].name == "tree_gate_lr_predictions.jsonl"
    assert files["nodes"].parent.name == "tau_0.1"
    # The calls file is tau-independent, so it lives one level up.
    assert files["calls"].parent.name == "hcy"
    assert all(files[k].is_file() for k in ("predictions", "nodes", "traversal", "calls"))


def test_every_method_kind_resolves_to_files(exp13_modules, tmp_path):
    """A kind registered in METHODS but absent from ``_resolve_files`` is a hard crash.

    ``discover_cell`` returns `missing` before it resolves anything when the run directory does
    not exist, so a missing branch stays invisible until the *first* run of that method lands on
    disk, and then it takes down every caller of ``discover``, including the ground-truth build, on a method
    they were not even asking about. That is how ``typed_span`` shipped: registered in METHODS and
    in METRIC_SUPPORT, with no branch here. So resolve against a directory made to exist.
    """
    d = exp13_modules.discovery
    for kind in sorted({m.kind for m in d.METHODS}):
        spec = next(m for m in d.METHODS if m.kind == kind)
        run_dir = tmp_path / spec.exp_id / "hcy"
        run_dir.mkdir(parents=True)
        # `_resolve_files` does not validate the point name, so any stand-in exercises the branch.
        files = d._resolve_files(run_dir, spec, "" if spec.sweep is None else "a_point")
        assert "predictions" in files, f"{kind} resolves no predictions file"
        assert files["predictions"].name.startswith(spec.variant), kind


def test_a_typed_span_run_is_discovered_at_exp17s_filenames(exp13_modules, tmp_path):
    """earlier writes ``{variant}_predictions.jsonl`` straight into ``{exp_id}/{cohort}/``."""
    d = exp13_modules.discovery
    spec = d.METHODS_BY_KEY["typed_span_detect"]
    run_dir = tmp_path / spec.exp_id / "hcy"
    run_dir.mkdir(parents=True)
    (run_dir / f"{spec.variant}_predictions.jsonl").write_text("", encoding="utf-8")

    cell = d.discover_cell(tmp_path, spec, "hcy")
    assert cell.status == "present", cell.detail
    # No sweep: one unnamed configuration, files directly in the cohort directory.
    assert cell.operating_points == [""]
    assert cell.files[""]["predictions"].name == "exp17_00_predictions.jsonl"
    assert cell.files[""]["records"].name == "exp17_00_records.jsonl"


# ── The metric applicability matrix ──────────────────────────────────────────

def test_retrieval_at_the_segment_unit_does_not_apply_to_rag_hpo(exp13_modules):
    """A category error, not a missing measurement, and the reason has to say so.

    RAG-HPO retrieves ontology candidates for an extracted phrase. There is no ranked list of
    report segments for a phenotype, which is what eq. (2) is defined over.
    """
    d = exp13_modules.discovery
    spec = d.METHODS_BY_KEY["raghpo_8b"]
    assert not d.supports(spec, "segment P@S / R@S (eq. 2)")
    assert "phrase" in d.support_reason(spec, "segment P@S / R@S (eq. 2)")


def test_reachability_applies_only_to_the_traversal_methods(exp13_modules):
    d = exp13_modules.discovery
    supported = {m.key for m in d.METHODS if d.supports(m, "reachability recall (eq. 6)")}
    assert supported == {m.key for m in d.METHODS if m.kind == "tree"}
    assert "raghpo_8b" not in supported


def test_flat_metrics_apply_to_every_method(exp13_modules):
    """Eq. (1) is the common ground the whole comparison rests on."""
    d = exp13_modules.discovery
    assert all(d.supports(m, "flat P/R/F1 (eq. 1)") for m in d.METHODS)


def test_every_method_kind_has_a_verdict_for_every_metric_group(exp13_modules):
    """A missing entry would silently read as "not supported" with no reason given."""
    d = exp13_modules.discovery
    kinds = {m.kind for m in d.METHODS}
    for group, by_kind in d.METRIC_SUPPORT.items():
        assert kinds <= set(by_kind), f"{group} has no verdict for {kinds - set(by_kind)}"


def test_unsupported_metrics_always_carry_a_reason(exp13_modules):
    d = exp13_modules.discovery
    for row in d.metric_support_rows():
        if not row["supported"]:
            assert row["reason"], f"{row['metric_group']} / {row['method']} has no reason"


def test_rerun_command_names_the_real_cluster_script(exp13_modules):
    """``cluster/run_<ds>_<exp_id>.sh``, the convention CLAUDE.md mandates."""
    d = exp13_modules.discovery
    assert d.rerun_command(d.METHODS_BY_KEY["tree_gate_lr"], "gsc") == (
        "sbatch cluster/run_gsc_exp13_00_tree_gate_lr.sh")
    assert "extract" in d.rerun_command(d.METHODS_BY_KEY["slm_ensemble"], "hcy")


def test_availability_rows_cover_the_whole_grid(exp13_modules, exp13_results_dir):
    avail = exp13_modules.discovery.discover(exp13_results_dir)
    d = exp13_modules.discovery
    rows = d.availability_rows(avail)
    n_cells = len(d.METHODS) * len(d.BASE_COHORTS)
    assert len(rows) == n_cells
    assert {r["status"] for r in rows} <= {"present", "partial", "missing"}
    assert sum(avail.counts().values()) == n_cells


# ── Derived cohorts ──────────────────────────────────────────────────────────
#
# The RAG-HPO comparison is scored from the gsc artifacts under a different ground truth standard, so it
# is a *view* of a run, not a run. These fix that the view never invents a cell.

def test_a_derived_cohort_resolves_to_the_run_it_is_scored_from(exp13_modules):
    d = exp13_modules.discovery
    assert d.artifact_cohort("gsc_raghpo") == "gsc"
    assert d.artifact_cohort("gsc_raghpo_ann") == "gsc"
    assert d.artifact_cohort("gsc") == "gsc"       # a base cohort is its own source
    assert d.artifact_cohort("hcy") == "hcy"


def test_derived_cohorts_do_not_enlarge_the_availability_grid(exp13_modules, exp13_results_dir):
    """Every method x the *base* cohorts only. A derived cohort reads the same directory, so
    giving it a row would double availability.md without adding a fact."""
    d = exp13_modules.discovery
    avail = d.discover(exp13_results_dir)
    n_cells = len(d.METHODS) * len(d.BASE_COHORTS)
    assert len(avail.cells) == n_cells
    assert len(d.availability_rows(avail)) == n_cells
    assert not any(c in d.BASE_COHORTS for c in d.DERIVED_COHORTS)


def test_a_derived_cohort_reads_its_sources_cell(exp13_modules, exp13_results_dir):
    """Availability is taken over: if the gsc run is missing or unmerged, so is every view of it."""
    avail = exp13_modules.discovery.discover(exp13_results_dir)
    for method in ("tree_gate_lr", "raghpo_70b"):
        source = avail.get(method, "gsc")
        assert avail.get(method, "gsc_raghpo") is source
        assert avail.get(method, "gsc_raghpo_ann") is source


def test_the_rerun_for_a_derived_cohort_is_its_sources_script(exp13_modules):
    """There is no run_gsc_raghpo_*.sh to point anyone at."""
    d = exp13_modules.discovery
    spec = d.METHODS_BY_KEY["tree_gate_lr"]
    assert d.rerun_command(spec, "gsc_raghpo_ann") == d.rerun_command(spec, "gsc")
