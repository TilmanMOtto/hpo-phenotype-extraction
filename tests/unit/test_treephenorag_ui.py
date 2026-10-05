"""Unit tests for the earlier deep-dive UI.

The real artifacts live on the cluster, so everything here runs against the ten-node toy ontology
and ``tests/fixtures/exp13_output``'s miniature run tree, whose expected answers are hand-computed
in that module's docstring. What the tests are actually for:

**The pins.** Four things are transcribed from library code for speed, the traversal graph, the
blocking-depth DP, the scoring wrapper and the prompt template, and each is asserted equal to its
original. A transcription that drifts does not crash. It produces a plausible number, which is the
worst failure mode a diagnostic tool can have.

**The Dash wiring.** Every callback ``Input``/``Output``/``State`` must name a component that
exists in the layout. Dash does not check this: a typo yields a tab that renders nothing, with no
error anywhere, and finding it means clicking through six tabs over an SSH tunnel.

**The artifact contract.** The summary line, the shard rules, the truncated tail. These are the
places where being wrong looks like a smaller number rather than an exception.
"""

from __future__ import annotations

import json
import os
import random

import pandas as pd
import pytest
from dash import html
from dash.exceptions import PreventUpdate

from apps.treephenorag_ui import (
    calib, copyexport, curated as curated_mod, discovery, evidence as evidence_mod, loaders,
    nodes as nodes_mod, pruning, registry as registry_mod, sample as sample_mod, scoring,
    subgraph, theme,
)
from fixtures.exp13_output import (
    ACCEPTED, EXPANDED, GOLD, PRUNE_SCORE, REPORT_TEXT, SEGMENTS, build_exp13_tree,
)
from fixtures.exp13_ui_output import (
    shard, truncate_last_line, write_curated_dataset, write_extractions, write_gold_csv,
    write_node_metadata, write_report_texts, write_segments_csv,
)
from fixtures.toy_ontology import (
    A, B, C, D, E, F, G, H, HALLUCINATION, M, NAMES, OBSOLETE_C, toy_children_map,
)

pytestmark = pytest.mark.unit

TREE_CELL = "tree_gate_lr/hcy"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def output_base(tmp_path):
    """A miniature earlier output tree, plus the two sidecars the UI reads and the result-table library does not.

    ``*_node_metadata.jsonl`` supplies the labels and organ systems the tables display. The Free Listing generation run's ``llm_extractions_*.jsonl`` supplies the segmentation the deep-dive recovers its
    sentences from when Stanza is unavailable, which here it always is.
    """
    root = build_exp13_tree(tmp_path / "output")
    for exp_id, variant in (("exp13_00_tree_gate_lr", "tree_gate_lr"),
                            ("exp13_01_tree_noisyor", "tree_noisyor"),
                            ("exp13_02_tree_no_prune", "tree_no_prune"),
                            ("exp13_09_tree_lse_beta1", "tree_lse_beta1")):
        for cohort in ("hcy", "gsc"):
            write_node_metadata(root / exp_id / cohort / f"{variant}_node_metadata.jsonl")
    for cohort in ("hcy", "gsc"):
        write_extractions(root / "phenojury_generation_free_listing" / cohort, SEGMENTS)
    return root


@pytest.fixture
def report_dir(tmp_path):
    """The cohort's reports on disk, what the deep-dive shows and the recovery verifies against."""
    return write_report_texts(tmp_path / "reports", REPORT_TEXT)


@pytest.fixture
def shared_ontology(toy_view):
    """Point the registry's process-wide singletons at the toy ontology.

    ``hpo_extraction.ontology.hpo_tree`` imports nltk and stanza at module scope and the real tree is 15 MB of
    JSON. The UI is built so neither is needed to exercise it.
    """
    children_map, roots = toy_children_map()
    registry_mod.set_shared(view=toy_view,
                            graph=(children_map, roots, pruning.bfs_depths(children_map, roots)))
    yield toy_view
    registry_mod._VIEW = None
    registry_mod._GRAPH = None
    registry_mod._TREE = None


@pytest.fixture
def registry(output_base, shared_ontology, tmp_path, report_dir):
    gold_path = write_gold_csv(tmp_path / "gt.csv", GOLD)
    return registry_mod.Registry(
        str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir=str(report_dir),
        cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def cells(output_base):
    return {c.cell_id: c for c in discovery.discover(str(output_base))}


# ── discovery ────────────────────────────────────────────────────────────────

class TestDiscovery:
    def test_finds_every_written_cell_with_its_operating_points(self, cells):
        assert cells[TREE_CELL].operating_points == ["tau_0.1", "tau_0.5"]
        assert cells["tree_no_prune/hcy"].operating_points == ["tau_0.5"]
        assert cells["raghpo_8b/hcy"].operating_points == []
        assert all(c.status == "ok" for c in cells.values()), \
            {k: v.problems for k, v in cells.items() if v.status != "ok"}

    def test_operating_points_sort_numerically_not_lexicographically(self, output_base):
        """``tau_0.02`` must precede ``tau_0.1``. A lexicographic sort puts it after.

        A τ frontier plotted in lexicographic order is a scribble, and the mistake is invisible
        until a sweep happens to include a two-decimal point, which every real earlier run does.
        """
        run_dir = output_base / "exp13_00_tree_gate_lr" / "hcy"
        for tau in ("tau_0.02", "tau_0.2"):
            (run_dir / tau).mkdir()
            (run_dir / tau / "tree_gate_lr_nodes.jsonl").write_text("")
            (run_dir / tau / "tree_gate_lr_predictions.jsonl").write_text("")
        points = discovery.discover(str(output_base))
        cell = next(c for c in points if c.cell_id == TREE_CELL)
        assert cell.operating_points == ["tau_0.02", "tau_0.1", "tau_0.2", "tau_0.5"]

    def test_a_two_axis_sweep_is_discovered_and_sorted_as_a_grid(self, cells):
        """An earlier exploratory run sweeps τ_accept as well, so its points are ``tau_{p}_acc_{a}``.

        Ordered τ_prune-major: consecutive points must differ in the accept threshold only, or
        the frontier, which slices on τ_accept, reads a scrambled sweep.
        """
        assert cells["tree_lse_beta1/hcy"].operating_points == [
            "tau_0.1_acc_0.5", "tau_0.1_acc_0.9", "tau_0.5_acc_0.5", "tau_0.5_acc_0.9"]
        assert cells["tree_lse_beta1/hcy"].status == "ok"

    def test_the_primary_axis_of_a_two_axis_point_is_tau_prune_not_tau_accept(self):
        """The trailing number is τ_accept, and reading it as the configuration is the bug.

        ``op_value`` feeds the frontier's x-axis and the culprit leaderboard's "which swept τ
        would have opened this node", both questions about the *expansion* threshold.
        """
        assert discovery.op_value("tau_0.00015_acc_0.9") == 0.00015
        assert discovery.op_accept("tau_0.00015_acc_0.9") == 0.9
        assert discovery.op_value("tau_0.5") == 0.5
        assert discovery.op_accept("tau_0.5") is None
        assert discovery.op_value("vote_k3") == 3.0
        assert discovery.op_value("agg_plurality") is None

    def test_a_scientific_notation_tau_parses_as_the_number_it_is(self):
        """``%g`` writes ``tau_1e-05``. A trailing-digits regex reads that as 5."""
        assert discovery.op_value("tau_1e-05") == pytest.approx(1e-5)

    def test_the_accept_slice_is_the_prune_sweep_through_the_selected_point(self, cells):
        points = cells["tree_lse_beta1/hcy"].operating_points
        assert discovery.accept_slice(points, "tau_0.1_acc_0.9") == [
            "tau_0.1_acc_0.9", "tau_0.5_acc_0.9"]
        # A cell with no accept axis has nothing to slice on, and must not come back empty.
        assert discovery.accept_slice(["tau_0.1", "tau_0.5"], "tau_0.1") == ["tau_0.1", "tau_0.5"]

    def test_a_tau_directory_that_does_not_parse_is_dropped_not_guessed_at(self, output_base):
        (output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_scratch").mkdir()
        cell = next(c for c in discovery.discover(str(output_base)) if c.cell_id == TREE_CELL)
        assert cell.operating_points == ["tau_0.1", "tau_0.5"]

    def test_phenobert_is_discovered_from_its_own_artifact_set(self, cells):
        """The PhenoBERT baseline writes no nodes, no calls and no sweep, predictions, detections, timing."""
        cell = cells["phenobert/hcy"]
        assert cell.spec.kind == "phenobert" and cell.operating_points == []
        assert cell.status == "ok"
        assert cell.files_for("predictions").present
        assert cell.files_for("detections").present

    def test_merged_file_wins_over_the_shards_it_was_built_from(self, output_base):
        """``merge_shards.py`` leaves the shards in place. Reading both double-counts everything."""
        run_dir = output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
        merged = run_dir / "tree_gate_lr_nodes.jsonl"
        lines = merged.read_text().splitlines()
        for index in range(2):
            merged.with_name(f"tree_gate_lr_s{index}of2_nodes.jsonl").write_text(
                "".join(f"{line}\n" for line in lines))

        fileset = discovery._resolve(str(run_dir), "tree_gate_lr", "nodes")
        assert fileset.shard_state == "merged"
        assert fileset.paths == [str(merged)]
        assert len(loaders.load_nodes(fileset)) == len(lines)

    def test_unmerged_shards_are_read_and_flagged(self, output_base):
        run_dir = output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
        merged = run_dir / "tree_gate_lr_nodes.jsonl"
        expected = len(merged.read_text().splitlines())
        shard(merged, n_shards=3)

        cell = next(c for c in discovery.discover(str(output_base)) if c.cell_id == TREE_CELL)
        fileset = cell.files_for("nodes", "tau_0.1")
        assert fileset.shard_state == "shards"
        assert len(loaders.load_nodes(fileset)) == expected
        assert cell.status == "shards"
        assert any("merge_shards" in p for p in cell.problems)

    def test_an_incomplete_shard_set_is_reported_as_partial_with_the_missing_index(
            self, output_base):
        run_dir = output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
        shard(run_dir / "tree_gate_lr_nodes.jsonl", n_shards=3, drop=1)

        cell = next(c for c in discovery.discover(str(output_base)) if c.cell_id == TREE_CELL)
        assert cell.status == "partial"
        assert cell.files_for("nodes", "tau_0.1").missing_shards == [1]
        assert any("missing indices [1]" in p for p in cell.problems)


# ── loaders ──────────────────────────────────────────────────────────────────

class TestLoaders:
    def test_summary_lines_are_the_authority_and_never_become_prediction_rows(self, cells):
        """The summary line is the only record of a report that predicted nothing."""
        result = loaders.load_predictions(cells[TREE_CELL].files_for("predictions", "tau_0.1"))
        assert result["predicted"] == {rid: set(v) for rid, v in ACCEPTED.items()}
        assert result["gold"] == {rid: set(v) for rid, v in GOLD.items()}
        assert result["n_hit_rows"] == sum(len(v) for v in ACCEPTED.values())
        assert result["reports_without_summary"] == []

    def test_a_report_that_predicted_nothing_still_appears(self, output_base):
        path = (output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
                / "tree_gate_lr_predictions.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"report_id": "r9", "summary": True,
                                "predicted_set": [], "gold_set": [C]}) + "\n")
        fileset = discovery._resolve(str(path.parent), "tree_gate_lr", "predictions")
        result = loaders.load_predictions(fileset)
        assert result["predicted"]["r9"] == set()
        assert "r9" in result["report_ids"]

    def test_a_truncated_tail_raises_naming_the_file_and_line(self, output_base):
        path = (output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
                / "tree_gate_lr_nodes.jsonl")
        truncate_last_line(path)
        with pytest.raises(json.JSONDecodeError, match="tree_gate_lr_nodes.jsonl"):
            list(loaders.read_jsonl(str(path)))

    def test_calls_scan_summarises_and_indexes_without_holding_the_file(self, cells):
        fileset = cells[TREE_CELL].files_for("calls")
        scan = loaders.scan_calls(fileset)
        summary = scan["summary"]

        assert set(scan["offsets"]) == {"r1", "r2"}
        assert scan["n_rows"] == int(summary["n_calls"].sum())
        row = summary.loc[(summary["report_id"] == "r1") & (summary["hpo_id"] == C)].iloc[0]
        # The fixture gives rank 1 a margin of +2.0 and every other rank -2.0.
        assert row["n_calls"] == 3
        assert row["n_yes"] == 1
        assert row["max_margin"] == pytest.approx(2.0)
        assert row["min_margin"] == pytest.approx(-2.0)

    def test_calls_for_report_returns_that_report(self, cells):
        scan = loaders.scan_calls(cells[TREE_CELL].files_for("calls"))
        frame = loaders.calls_for_report(scan, "r1")
        assert set(frame["report_id"]) == {"r1"}
        assert len(frame) == int(
            scan["summary"].loc[scan["summary"]["report_id"] == "r1", "n_calls"].sum())


class TestNodesAreColumnarAndIndexed:
    """The two changes that made a loose tau_prune openable, and their invariants.

    A loose tau_prune writes ~200x the nodes a tight one does, 1 051 846 rows for one GSC+
    configuration, and the old reader held the whole file as a list of dicts before the frame
    existed. Both replacements here are supposed to be *purely representational*: the same rows,
    the same values, less memory. These tests are what says so.
    """

    def test_ids_are_categorical_and_the_values_survive(self, cells):
        frame = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", "tau_0.1"))
        assert isinstance(frame["report_id"].dtype, pd.CategoricalDtype)
        assert isinstance(frame["hpo_id"].dtype, pd.CategoricalDtype)
        assert set(frame["report_id"].astype(str)) == set(EXPANDED["tau_0.1"])
        assert frame["depth"].dtype == "int16"
        assert frame["expanded"].dtype == bool

    def test_an_absent_artifact_is_typed_like_a_present_one(self):
        """So no caller needs a special case for the empty frame."""
        empty = loaders.load_nodes(discovery.FileSet("nodes"))
        assert empty.empty
        assert dict(empty.dtypes.astype(str)) == dict(loaders.empty_nodes().dtypes.astype(str))

    def test_a_missing_key_is_not_the_same_claim_as_a_false_one(self):
        """``is_gold`` = -1 means "the artifact did not say"; 0 would mean "it said no"."""
        frame = loaders._nodes_frame(iter([{"report_id": "r1", "hpo_id": C}]))
        assert int(frame["is_gold"].iloc[0]) == -1
        assert int(frame["depth"].iloc[0]) == -1

    def test_chunking_does_not_change_the_frame(self, cells, monkeypatch):
        """The chunk size bounds the peak. It must not be observable in the result."""
        fileset = cells[TREE_CELL].files_for("nodes", "tau_0.1")
        whole = loaders.load_nodes(fileset)
        monkeypatch.setattr(loaders, "_NODES_CHUNK", 1)
        assert loaders.load_nodes(fileset).equals(whole)

    def test_the_index_reproduces_the_cohort_frame_report_by_report(self, cells):
        fileset = cells[TREE_CELL].files_for("nodes", "tau_0.1")
        whole = loaders.load_nodes(fileset)
        scan = loaders.scan_nodes(fileset)

        assert scan["n_rows"] == len(whole)
        assert set(scan["report_ids"]) == set(whole["report_id"].astype(str))
        for rid in scan["report_ids"]:
            one = loaders.nodes_for_report(scan, rid)
            reference = whole.loc[whole["report_id"].astype(str) == rid].reset_index(drop=True)
            assert list(one["hpo_id"].astype(str)) == list(reference["hpo_id"].astype(str))
            assert list(one["prune_score"]) == list(reference["prune_score"])
            # Same dtypes too: the deep-dive feeds this straight into build_node_table.
            assert dict(one.dtypes.astype(str)) == dict(whole.dtypes.astype(str))

    def test_the_index_spans_unmerged_shards(self, output_base):
        """A report's block can straddle two shard files, and the ranges must still cover it."""
        directory = output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
        shard(directory / "tree_gate_lr_nodes.jsonl", 2)
        fileset = discovery._resolve(str(directory), "tree_gate_lr", "nodes")
        assert len(fileset.paths) == 2

        whole = loaders.load_nodes(fileset)
        scan = loaders.scan_nodes(fileset)
        rebuilt = pd.concat([loaders.nodes_for_report(scan, r) for r in scan["report_ids"]])
        assert sorted(zip(rebuilt["report_id"].astype(str), rebuilt["hpo_id"].astype(str))) == \
            sorted(zip(whole["report_id"].astype(str), whole["hpo_id"].astype(str)))

    def test_a_re_appended_block_keeps_the_first_occurrence(self, output_base):
        """A resumed run re-writes a report's block; ``merge_shards`` keeps the first, so do we.

        Keeping the last instead would be invisible on any well-formed artifact and would silently
        disagree with the merged file that ``result_tables`` scores, the two would report different
        prune scores for the same node and nothing on screen would say which.
        """
        path = (output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
                / "tree_gate_lr_nodes.jsonl")
        original = [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()]
        replayed = [{**row, "prune_score": 0.123, "accepted": not row["accepted"]}
                    for row in original if row["report_id"] == "r1"]
        with open(path, "a", encoding="utf-8") as handle:
            for row in replayed:
                handle.write(json.dumps(row) + "\n")

        fileset = discovery._resolve(str(path.parent), "tree_gate_lr", "nodes")
        whole = loaders.load_nodes(fileset)
        one = loaders.nodes_for_report(loaders.scan_nodes(fileset), "r1")

        first = {row["hpo_id"]: row for row in reversed(original) if row["report_id"] == "r1"}
        assert len(one) == len(first)
        for frame in (one, whole.loc[whole["report_id"].astype(str) == "r1"]):
            got = dict(zip(frame["hpo_id"].astype(str), frame["prune_score"]))
            assert got == {h: pytest.approx(r["prune_score"]) for h, r in first.items()}

    def test_report_states_are_lazy_and_carry_no_accept_score(self, cells):
        """Neither consumer reads accept_score, and carrying it doubled the structure."""
        frame = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", "tau_0.1"))
        states = nodes_mod.state_by_report(frame)

        assert isinstance(states, nodes_mod.ReportStates)
        assert set(states) == set(EXPANDED["tau_0.1"])
        for rid in states:
            assert set(states[rid]) == {"visited", "expanded", "accepted", "prune_score"}
            group = frame.loc[frame["report_id"].astype(str) == rid]
            assert states[rid]["visited"] == set(group["hpo_id"].astype(str))
            assert states[rid]["expanded"] == set(EXPANDED["tau_0.1"][rid])
            assert states[rid]["prune_score"] == {
                str(h): pytest.approx(v) for h, v in
                zip(group["hpo_id"].astype(str), group["prune_score"])
            }

    def test_the_ledger_is_identical_to_the_eager_mapping(self, cells, shared_ontology):
        """The lazy mapping is a representation change, so the ledger may not move at all."""
        children_map, roots = toy_children_map()
        depths = pruning.bfs_depths(children_map, roots)
        frame = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", "tau_0.1"))
        gold = {rid: set(v) for rid, v in GOLD.items()}

        lazy = pruning.prune_ledger(nodes_mod.state_by_report(frame), gold, children_map,
                                    roots, depths)
        eager = pruning.prune_ledger(
            {rid: dict(nodes_mod.state_by_report(frame)[rid]) for rid in EXPANDED["tau_0.1"]},
            gold, children_map, roots, depths)
        assert lazy == eager

    def test_the_bundle_does_not_retain_the_raw_frame_or_the_states(self, registry):
        """Both are build-time inputs. Keeping them held three copies of the same artifact."""
        bundle = registry.bundle(TREE_CELL, "tau_0.1")
        assert "nodes_df" not in bundle
        assert "state" not in bundle

    def test_the_report_list_needs_no_bundle(self, registry):
        """Opening the deep-dive tab used to build the cohort node table to fill a dropdown."""
        assert sorted(registry.report_ids(TREE_CELL, "tau_0.1")) == sorted(ACCEPTED)
        assert not registry._bundles


class TestOneReportPath:
    """``report_bundle`` must be the cohort table's own answer, not an approximation of it."""

    #: Every analysis column the deep-dive reads off the table. Compared one by one so a failure
    #: names the column, not saying two frames differ.
    COLUMNS = ("state", "depth", "prune_score", "accept_score", "expanded", "accepted",
               "in_gold", "outcome", "subtree_has_gold", "fate", "culprit",
               "culprit_prune_score", "fp_class", "fn_class", "hpo_label")

    def test_it_equals_the_cohort_slice_column_by_column(self, registry):
        cohort = registry.bundle(TREE_CELL, "tau_0.1", heavy=True)["node_table"]
        for rid in registry.report_ids(TREE_CELL, "tau_0.1"):
            one = registry.report_bundle(TREE_CELL, "tau_0.1", rid)
            mine = one["frame"].set_index(one["frame"]["hpo_id"].astype(str)).sort_index()
            reference = cohort.loc[cohort["report_id"].astype(str) == rid]
            reference = reference.set_index(reference["hpo_id"].astype(str)).sort_index()

            assert list(mine.index) == list(reference.index), rid
            for column in self.COLUMNS:
                left = [None if pd.isna(v) else v for v in mine[column].astype(object)]
                right = [None if pd.isna(v) else v for v in reference[column].astype(object)]
                assert left == right, f"{rid}: {column}"

    def test_gold_and_predicted_match_the_cohort_bundle(self, registry):
        bundle = registry.bundle(TREE_CELL, "tau_0.1")
        for rid in registry.report_ids(TREE_CELL, "tau_0.1"):
            one = registry.report_bundle(TREE_CELL, "tau_0.1", rid)
            assert one["gold"] == set(bundle["gold"].get(rid, set()))
            assert one["predicted"] == set(bundle["predictions"]["predicted"].get(rid, set()))


class TestSample:
    """The sampling frame, and the rule it is not allowed to let a reader forget."""

    @staticmethod
    def write(directory, picks, pools=True):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "frame.json").write_text(json.dumps({
            "generated": "2026-09-09", "picks": picks, "pools": picks,
            "selected": sorted(r for v in picks.values() for r in v),
            "params": {"seed": 1},
        }), encoding="utf-8")
        if pools:
            lines = ["patient_id,selected,cell,n_gold,n_pred,tp,fp,fn,precision,recall,f1,"
                     "family,lab_value,implicit,negated"]
            for cell, members in picks.items():
                for rid in members:
                    lines.append(f"{rid},1,{cell},3,2,1,1,2,0.5,0.333,0.4,1,0,0,0")
            (directory / "pools.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    def test_it_reads_the_cells_the_scripts_draw(self, tmp_path):
        frame = sample_mod.load(str(self.write(tmp_path / "f", {"pb_worst": ["r1"],
                                                                "family": ["r2"]})))
        assert frame.cells == ("pb_worst", "family")          # CELL_ORDER, not the file's order
        assert frame.selected == ["r1", "r2"]
        assert frame.label("r1") == "pb_worst"
        assert frame.row("r1")["f1"] == pytest.approx(0.4)
        assert frame.row("r1")["family"] == 1
        assert "family" in frame.note("r2")

    def test_absence_is_an_ordinary_state(self, tmp_path):
        assert sample_mod.load("") is None
        assert sample_mod.load(str(tmp_path / "nothing-here")) is None
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "frame.json").write_text("{not json", encoding="utf-8")
        assert sample_mod.load(str(tmp_path / "bad")) is None

    def test_a_draw_with_no_reports_is_refused(self, tmp_path):
        assert sample_mod.load(str(self.write(tmp_path / "e", {"pb_worst": []}))) is None

    def test_pools_is_optional(self, tmp_path):
        frame = sample_mod.load(str(self.write(tmp_path / "n", {"pb_worst": ["r1"]}, pools=False)))
        assert frame.label("r1") == "pb_worst" and frame.row("r1") is None

    def test_an_empty_cell_is_not_applicable_rather_than_zero(self, tmp_path):
        """A report with no ground truth has no precision; 0.0 would claim it had and failed."""
        directory = self.write(tmp_path / "p", {"no_annotation": ["r1"]})
        (directory / "pools.csv").write_text(
            "patient_id,selected,cell,n_gold,n_pred,tp,fp,fn,precision,recall,f1,"
            "family,lab_value,implicit,negated\nr1,1,no_annotation,0,2,0,2,0,,,,0,0,0,0\n",
            encoding="utf-8")
        assert sample_mod.load(str(directory)).row("r1")["precision"] is None

    def test_ids_translates_every_control_state(self, tmp_path):
        frame = sample_mod.load(str(self.write(tmp_path / "f", {"pb_worst": ["r1", "r2"],
                                                                "family": ["r3"]})))
        assert frame.ids(None) is None and frame.ids("all") is None
        assert frame.ids("sample") == ["r1", "r2", "r3"]
        assert frame.ids("cell", "pb_worst") == ["r1", "r2"]
        assert frame.ids("cell", "nonexistent") == []


class TestSubsetFilter:
    @pytest.fixture
    def registry(self, output_base, shared_ontology, tmp_path, report_dir):
        gold_path = write_gold_csv(tmp_path / "gt.csv", GOLD)
        frame_dir = TestSample.write(tmp_path / "frame", {"pb_worst": ["r1"], "family": ["r2"]})
        return registry_mod.Registry(
            str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir=str(report_dir),
            cache_dir=str(tmp_path / "cache"), hcy_frame_dir=str(frame_dir))

    def test_the_frame_only_applies_to_its_own_cohort(self, registry):
        """An HCY draw intersected with a GSC+ run would empty the page with no visible cause."""
        assert registry.subset_ids(TREE_CELL, "sample") == ["r1", "r2"]
        assert registry.subset_ids("tree_gate_lr/gsc", "sample") is None

    def test_restricting_the_metrics_takes_a_second_explicit_click(self, registry):
        picked = {"mode": "cell", "cell": "pb_worst"}
        assert registry.aggregate_filter(TREE_CELL, {**picked, "aggregates": False}) is None
        assert registry.aggregate_filter(TREE_CELL, {**picked, "aggregates": True}) == ["r1"]
        assert registry.aggregate_filter(TREE_CELL, {"mode": "all", "aggregates": True}) is None

    def test_a_restricted_bundle_is_narrowed_consistently(self, registry):
        bundle = registry.bundle(TREE_CELL, "tau_0.1", report_filter=["r1"])
        assert set(bundle["predictions"]["predicted"]) == {"r1"}
        assert set(bundle["gold"]) == {"r1"}
        assert set(bundle["node_table"]["report_id"].astype(str)) <= {"r1"}
        assert {row["report_id"] for row in bundle["ledger"]["rows"]} == {"r1"}
        assert bundle["report"]["counts"]["n_reports_scored"] == 1

    def test_the_unrestricted_bundle_is_untouched_and_cached_apart(self, registry):
        full = registry.bundle(TREE_CELL, "tau_0.1")
        restricted = registry.bundle(TREE_CELL, "tau_0.1", report_filter=["r1"])
        assert restricted is not full
        assert set(full["predictions"]["predicted"]) == set(ACCEPTED)
        assert full["report"]["restricted_n"] is None
        assert restricted["report"]["restricted_n"] == 1
        # Asking again for the full bundle must not hand back the restricted one.
        assert registry.bundle(TREE_CELL, "tau_0.1") is full

    def test_a_restricted_report_says_so_in_the_provenance_line(self, registry):
        restricted = registry.get_report(TREE_CELL, "tau_0.1", report_filter=["r1"])
        line = copyexport.provenance(restricted)
        assert "RESTRICTED to 1 report(s)" in line and "purposive sample" in line
        assert "RESTRICTED" not in copyexport.provenance(
            registry.get_report(TREE_CELL, "tau_0.1"))

    def test_the_banner_carries_the_frame_s_own_sentence(self, registry):
        picked = {"mode": "cell", "cell": "pb_worst", "aggregates": True}
        note = registry.subset_note(TREE_CELL, picked)
        assert sample_mod.WARNING in flat(note)
        assert registry.subset_note(TREE_CELL, {**picked, "aggregates": False}) is None

    def test_every_aggregate_page_actually_renders_the_banner(self, registry):
        """A banner the builder accepts but never draws is the failure mode that counts here."""
        from apps.treephenorag_ui.views import calibration, pruning_view, scorecard

        picked = {"mode": "cell", "cell": "pb_worst", "aggregates": True}
        note = registry.subset_note(TREE_CELL, picked)
        report = registry.get_report(TREE_CELL, "tau_0.1", heavy=True,
                                     report_filter=registry.aggregate_filter(TREE_CELL, picked))
        for build in (scorecard._build, pruning_view._build, calibration._build):
            with_note = flat(build(report, "light", note))
            assert sample_mod.WARNING in with_note, build.__module__
            # And absent when nothing is restricted, so the warning keeps its meaning.
            assert sample_mod.WARNING not in flat(build(report, "light", None)), build.__module__


# ── pruning: the pins and the ledger ─────────────────────────────────────────

class TestBlockingCulprits:
    def test_depths_equal_thesis_metrics_on_random_cuts(self):
        """The depth is a published number. The culprit is the extra this module adds.

        Randomising the expanded set exercises the three branches of the recurrence, root,
        reachable-through-an-expanded-parent, and severed, in every combination a nine-node DAG
        with a multi-parent term admits.
        """
        children_map, roots = toy_children_map()
        rng = random.Random(20260805)
        nodes = sorted(children_map)
        for _ in range(500):
            expanded = {n for n in nodes if rng.random() < 0.6}
            pruning.assert_matches_traversal(children_map, roots, expanded)

    def test_names_the_node_that_severed_the_last_surviving_path(self):
        children_map, roots = toy_children_map()
        best = pruning.blocking_culprits(children_map, roots, {A, B, G})
        assert best[F] == (2.0, E), "E was reachable and refused to expand, so F is blocked at 2"
        assert best[C][0] == float("inf"), "B expanded, so C is reachable"

    def test_a_multi_parent_term_survives_while_any_parent_opens(self):
        """M hangs off both B and H. Closing one parent must not block it, that is the whole
        reason the DP takes a max over parents, not a min."""
        children_map, roots = toy_children_map()
        assert pruning.blocking_culprits(children_map, roots, {A, B, G})[M][0] == float("inf")
        assert pruning.blocking_culprits(children_map, roots, {A, G})[M][0] == 2.0

    @pytest.mark.slow
    def test_children_map_matches_core_tree_traversal_on_the_real_ontology(self, real_hpo_tree):
        """The transcription that avoids the nltk import chain must equal the original."""
        real_hpo_tree.buildHPOTree()
        pruning.assert_children_map_matches(real_hpo_tree)


class TestPruneLedger:
    """The fixture prunes E at tau_0.5, which makes F unreachable, the one interesting case."""

    def _ledger(self, cells, tau, toy_view):
        children_map, roots = toy_children_map()
        nodes_df = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", tau))
        state = nodes_mod.state_by_report(nodes_df)
        gold = {rid: set(v) for rid, v in GOLD.items()}
        return pruning.prune_ledger(state, gold, children_map, roots)

    def test_at_a_low_tau_the_miss_is_an_identification_failure(self, cells, toy_view):
        ledger = self._ledger(cells, "tau_0.1", toy_view)
        assert ledger["fate_counts"] == {"found": 2, "rejected": 1, "blocked": 0,
                                         "unreached": 0, "outside_graph": 0}
        assert ledger["recall_ceiling"] == 1.0, "nothing was pruned away, so nothing is unreachable"

    def test_at_a_high_tau_the_same_miss_becomes_a_traversal_failure(self, cells, toy_view):
        ledger = self._ledger(cells, "tau_0.5", toy_view)
        assert ledger["fate_counts"]["blocked"] == 1
        assert ledger["recall_ceiling"] == pytest.approx(2 / 3)
        blocked = next(r for r in ledger["rows"] if r["fate"] == "blocked")
        assert (blocked["hpo_id"], blocked["culprit"], blocked["depth"]) == (F, E, 2)
        assert blocked["culprit_prune_score"] == PRUNE_SCORE[E]

    def test_recall_is_unchanged_while_the_ceiling_collapses(self, cells, toy_view):
        """The exact situation an F1 table cannot show, and the reason this app exists."""
        low = self._ledger(cells, "tau_0.1", toy_view)
        high = self._ledger(cells, "tau_0.5", toy_view)
        assert low["recall_actual"] == high["recall_actual"]
        assert high["recall_ceiling"] < low["recall_ceiling"]

    def test_the_culprit_leaderboard_names_the_tau_that_would_have_opened_it(self, cells,
                                                                            toy_view):
        ledger = self._ledger(cells, "tau_0.5", toy_view)
        board = pruning.culprit_leaderboard(ledger, tau_sweep=[0.1, 0.5])
        assert len(board) == 1
        entry = board[0]
        assert (entry["hpo_id"], entry["n_gold_lost"], entry["n_reports"]) == (E, 1, 1)
        # E scored 0.3, so tau_prune=0.1 expands it and 0.5 does not.
        assert entry["tau_would_expand_all"] == 0.1

    def test_no_swept_tau_is_reported_as_none_rather_than_the_lowest(self, cells, toy_view):
        ledger = self._ledger(cells, "tau_0.5", toy_view)
        board = pruning.culprit_leaderboard(ledger, tau_sweep=[0.5, 0.9])
        assert board[0]["tau_would_expand_all"] is None

    def test_fp_factory_charges_the_direct_expanded_parent(self, cells, toy_view):
        """D is a false positive under B, and B was expanded, so B is the attribution point."""
        nodes_df = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", "tau_0.1"))
        state = nodes_mod.state_by_report(nodes_df)
        children_map, _ = toy_children_map()
        board = pruning.fp_factory(state, {"r1": [D]}, children_map)
        assert [(e["hpo_id"], e["n_fp"]) for e in board] == [(B, 1)]


# ── the node table ───────────────────────────────────────────────────────────

class TestNodeTable:
    def _table(self, cells, toy_view, tau="tau_0.5"):
        children_map, roots = toy_children_map()
        nodes_df = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", tau))
        gold = {rid: set(v) for rid, v in GOLD.items()}
        ledger = pruning.prune_ledger(nodes_mod.state_by_report(nodes_df), gold,
                                      children_map, roots)
        metadata = loaders.load_node_metadata(cells[TREE_CELL].files_for("node_metadata"))
        return nodes_mod.build_node_table(nodes_df, gold, toy_view, ledger, metadata=metadata)

    def test_an_annotated_term_the_traversal_never_reached_still_gets_a_row(self, cells, toy_view):
        table = self._table(cells, toy_view)
        row = table.loc[(table["report_id"] == "r1") & (table["hpo_id"] == F)].iloc[0]
        assert (row["state"], row["outcome"], row["fate"]) == ("blocked", "FN", "blocked")
        assert row["accept_score"] != row["accept_score"], \
            "a node that was never asked must have no score, not a zero"

    def test_the_culprit_survives_the_join_onto_the_node_table(self, cells, toy_view):
        """The blocked row must name the node that severed it, with that node's own score.

        This is what the deep-dive's "never reached, severed by X" note and the FN table's
        `blocked by` column read. An empty column here does not raise anywhere, it just quietly
        removes the answer the whole app exists to give.
        """
        table = self._table(cells, toy_view)
        row = table.loc[(table["report_id"] == "r1") & (table["hpo_id"] == F)].iloc[0]
        assert row["culprit"] == E
        assert row["culprit_prune_score"] == PRUNE_SCORE[E]

    def test_prune_score_is_labelled_by_subtree_presence_not_node_presence(self, cells, toy_view):
        """B is never itself annotated but C sits below it, so its subtree label is true.

        Scoring ``prune_score`` against ``in_gold`` would mark the gate wrong at every internal
        node, the mistake ``result_tables/loaders.py:340`` documents at length.
        """
        table = self._table(cells, toy_view, tau="tau_0.1")
        row = table.loc[(table["report_id"] == "r1") & (table["hpo_id"] == B)].iloc[0]
        assert bool(row["in_gold"]) is False
        assert bool(row["subtree_has_gold"]) is True

    def test_outcomes_follow_the_joined_gold_file_not_the_artifact_flag(self, cells, toy_view,
                                                                       tmp_path):
        """Change the ground-truth set and the labels must move with it, disagreements counted."""
        children_map, roots = toy_children_map()
        nodes_df = loaders.load_nodes(cells[TREE_CELL].files_for("nodes", "tau_0.1"))
        other_gold = {"r1": {D}, "r2": {D}}          # r1's ground truth is now the term it got wrong
        ledger = pruning.prune_ledger(nodes_mod.state_by_report(nodes_df), other_gold,
                                      children_map, roots)
        table = nodes_mod.build_node_table(nodes_df, other_gold, toy_view, ledger)
        row = table.loc[(table["report_id"] == "r1") & (table["hpo_id"] == D)].iloc[0]
        assert row["outcome"] == "TP"
        assert nodes_mod.gold_disagreements(table) > 0

    def test_a_label_is_resolvable_for_a_node_the_run_never_visited(self, cells, toy_view):
        """Culprits are usually absent from the node table, and a bare HPO id names nothing."""
        table = self._table(cells, toy_view)
        lookup = nodes_mod.LabelLookup(table, toy_view)
        assert lookup[C] == "Seizure"
        assert lookup["HP:0011842"] == "Abnormality of the vertebral column", \
            "H is in the ontology but not in this run's node table"

    def test_the_distance_shortcut_equals_the_library_search(self, toy_view):
        """The table measures to the reference term instead of redoing the arg-min.

        Same number by definition, half the ontology walks, but only while the two functions
        agree about which annotated term is nearest, so every node of the toy graph is checked.
        """
        from fixtures.toy_ontology import HALLUCINATION, OUT_OF_SUBTREE

        candidates = [A, B, C, D, E, F, G, M, OBSOLETE_C, OUT_OF_SUBTREE, HALLUCINATION]
        for gold in ({C}, {C, F}, {M}, set()):
            nodes_mod.assert_distance_matches_library(candidates, gold, toy_view)

    def test_false_positives_are_classified_by_the_thesis_taxonomy(self, cells, toy_view):
        table = self._table(cells, toy_view, tau="tau_0.1")
        row = table.loc[(table["report_id"] == "r1") & (table["hpo_id"] == D)].iloc[0]
        assert row["outcome"] == "FP"
        assert row["fp_class"] == "sibling", "D and C share the direct parent B"
        assert row["fp_reference"] == C


# ── scoring, calibration, evidence ───────────────────────────────────────────

class TestScoring:
    def test_matches_thesis_metrics(self, cells, toy_view):
        predictions = loaders.load_predictions(cells[TREE_CELL].files_for("predictions",
                                                                         "tau_0.1"))
        gold = {rid: set(v) for rid, v in GOLD.items()}
        scoring.assert_matches_thesis_metrics(gold, predictions["predicted"], toy_view)

    def test_reproduces_the_fixtures_hand_computed_micro_f1(self, cells, toy_view):
        """TP=2, FP=1, FN=1 over the two reports, worked out in the fixture's docstring."""
        predictions = loaders.load_predictions(cells[TREE_CELL].files_for("predictions",
                                                                         "tau_0.1"))
        gold = {rid: set(v) for rid, v in GOLD.items()}
        flat = scoring.score(gold, predictions["predicted"], toy_view)["flat"]
        assert (flat["tp"], flat["fp"], flat["fn"]) == (2, 1, 1)
        assert flat["micro_f1"] == pytest.approx(2 / 3)


class TestCalibration:
    def test_each_score_is_judged_against_its_own_label(self, registry):
        bundle = registry.bundle(TREE_CELL, "tau_0.1")
        accept = calib.sample(bundle["node_table"], "accept")
        prune = calib.sample(bundle["node_table"], "prune")
        assert accept["label_column"] == "in_gold"
        assert prune["label_column"] == "subtree_has_gold"
        assert list(accept["labels"]) != list(prune["labels"]), \
            "the two labels must differ on this fixture, or the test proves nothing"

    def test_never_scored_nodes_are_excluded_rather_than_imputed(self, registry):
        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        table = bundle["node_table"]
        assert (table["state"] != "visited").any(), "the fixture must contain a blocked term"
        assert calib.sample(table, "accept")["n"] == int((table["state"] == "visited").sum())

    def test_roc_auc_handles_perfect_separation_and_ties(self):
        assert calib.roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
        assert calib.roc_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)
        assert calib.roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


class TestEvidence:
    def test_the_reconstructed_prompt_matches_core_prompting(self, toy_tree, monkeypatch):
        """Fixed character for character, the UI transcribes it only to avoid rebuilding
        an HPOTree per prompt."""
        import hpo_extraction.treephenorag.verifier_prompt

        monkeypatch.setattr(hpo_extraction.treephenorag.verifier_prompt, "HPOTree", lambda: toy_tree)
        sentence = "The patient had seizures."
        expected = hpo_extraction.treephenorag.verifier_prompt.embed_symptom_in_prompt_v1([sentence], C)[0]
        assert evidence_mod.build_prompt(toy_tree.data[C], sentence) == expected

    def test_an_out_of_range_sentence_index_returns_none_rather_than_a_wrong_sentence(
            self, registry):
        helper = registry.evidence()
        helper._segments["hcy"] = {"r1": ["one sentence only"]}
        assert helper.sentence("hcy", "r1", 0) == "one sentence only"
        assert helper.sentence("hcy", "r1", 7) is None

    def test_partial_index_coverage_is_reported(self, registry):
        helper = registry.evidence()
        helper._segments["hcy"] = {"r1": ["a", "b"]}
        coverage = helper.coverage("hcy", "r1", [0, 1, 9])
        assert coverage["n_resolved"] == 2 and coverage["complete"] is False


class TestSegmentationRecovery:
    """The deep-dive must show sentences on a machine where Stanza will not load.

    ``stanza_dir`` is empty in every fixture, so any test here that produces sentences produced
    them without the tokenizer, which is the whole claim.
    """

    def test_the_segmentation_comes_from_exp13_06_when_stanza_is_unavailable(self, registry):
        helper = registry.evidence()
        assert helper.segments("hcy") == SEGMENTS
        assert helper._pipeline is None, "Stanza was loaded despite the artifact route succeeding"
        assert "Free Listing run" in helper.source("hcy")
        assert helper.status("hcy") is None

    def test_a_recovered_segmentation_that_is_not_in_the_report_is_rejected(
            self, registry, output_base):
        """A segmentation of a *different* document resolves every index and shows wrong text."""
        write_extractions(output_base / "phenojury_generation_free_listing" / "hcy",
                          {"r1": ["Nothing like this is in the report at all."],
                           "r2": SEGMENTS["r2"]})
        helper = registry.evidence()
        recovered = helper.segments("hcy")
        assert "r1" not in recovered and recovered["r2"] == SEGMENTS["r2"]
        assert "rejected" in helper.source("hcy")

    def test_a_hole_in_the_numbering_is_rejected_rather_than_renumbered(
            self, registry, output_base):
        """A missing sentence 1 would shift every later index by one and resolve cleanly."""
        write_extractions(output_base / "phenojury_generation_free_listing" / "hcy", SEGMENTS,
                          drop=("r1", 1))
        assert "r1" not in registry.evidence().segments("hcy")

    def test_the_recovery_survives_the_artifacts_going_away(self, registry, output_base):
        """Once recovered, the segmentation is cached by a digest of the texts, not by path."""
        assert registry.evidence().segments("hcy") == SEGMENTS
        (output_base / "phenojury_generation_free_listing" / "hcy" / "llm_extractions_toy.jsonl").unlink()

        fresh = evidence_mod.Evidence(registry)
        assert fresh.segments("hcy") == SEGMENTS
        assert fresh.source("hcy") == "cache"

    def test_an_unrecoverable_cohort_says_why_instead_of_raising(self, registry, output_base):
        for path in (output_base / "phenojury_generation_free_listing" / "hcy").glob("llm_extractions_*"):
            path.unlink()
        helper = registry.evidence()
        assert helper.segments("hcy") == {}
        assert "stanza" in (helper.status("hcy") or "").lower()

    def test_out_of_order_sentences_do_not_pass_verification(self):
        text = "One. Two. Three."
        assert evidence_mod.verify_segmentation(["One.", "Two.", "Three."], text)
        assert not evidence_mod.verify_segmentation(["Two.", "One."], text)
        assert not evidence_mod.verify_segmentation(["Four."], text)


class TestSubgraph:
    """The induced ontology subgraph behind the deep-dive's clickable diagram."""

    def test_the_ancestors_that_connect_the_terms_are_included(self, toy_view):
        built = subgraph.build({C: "TP", F: "FN"}, toy_view)
        drawn = {n["hpo_id"]: n["group"] for n in built["nodes"]}
        # C's and F's closures are {C,B,A} and {F,E,A}: the two results plus three connectors.
        assert drawn == {C: "TP", F: "FN", B: "context", E: "context", A: "context"}
        assert (A, B) in built["edges"] and (B, C) in built["edges"]

    def test_a_multi_parent_term_keeps_its_second_parent_as_an_extra_edge(self, toy_view):
        built = subgraph.build({M: "FP"}, toy_view)
        parents = {p for p, c in built["edges"] if c == M}
        extra = {p for p, c in built["extra_edges"] if c == M}
        assert len(parents) == 1 and parents | extra == {B, H}

    def test_a_code_outside_the_ontology_becomes_an_orphan_not_a_root(self, toy_view):
        built = subgraph.build({C: "TP", HALLUCINATION: "FP"}, toy_view)
        assert [o["hpo_id"] for o in built["orphans"]] == [HALLUCINATION]
        assert HALLUCINATION not in {n["hpo_id"] for n in built["nodes"]}

    def test_an_alt_id_lands_on_the_node_it_redirects_to(self, toy_view):
        built = subgraph.build({OBSOLETE_C: "FP", C: "TP"}, toy_view)
        groups = {n["hpo_id"]: n["group"] for n in built["nodes"]}
        # One node, and the worse of the two outcomes, a node drawn green must be one that worked.
        assert OBSOLETE_C not in groups and groups[C] == "FP"

    def test_thinning_drops_context_and_never_a_result(self, toy_view):
        built = subgraph.build({C: "TP", D: "FP", F: "FN"}, toy_view, max_nodes=4)
        drawn = {n["hpo_id"]: n["group"] for n in built["nodes"]}
        assert {C, D, F} <= set(drawn) and built["n_truncated"] > 0
        assert all(g != "context" for h, g in drawn.items() if h in (C, D, F))

    def test_a_parent_sits_over_its_children(self, toy_view):
        built = subgraph.build({C: "TP", D: "FP"}, toy_view)
        x = {n["hpo_id"]: n["x"] for n in built["nodes"]}
        assert min(x[C], x[D]) < x[B] < max(x[C], x[D])
        y = {n["hpo_id"]: n["y"] for n in built["nodes"]}
        assert y[A] > y[B] > y[C], "depth must increase downwards"

    def test_true_negatives_are_not_drawn(self, registry):
        frame = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        subset = frame.loc[frame["report_id"] == "r1"]
        terms = subgraph.terms_of_report(subset)
        assert set(terms.values()) <= {"TP", "FP", "FN"}
        assert terms.get(C) == "TP" and terms.get(F) == "FN"


# ── the registry and the report ──────────────────────────────────────────────

class TestRegistry:
    def test_report_is_json_serialisable(self, registry):
        json.dumps(registry.get_report(TREE_CELL, "tau_0.5"))

    def test_report_carries_the_gold_file_it_was_scored_against(self, registry):
        report = registry.get_report(TREE_CELL, "tau_0.5")
        assert report["gold_source"] == registry.hcy_gt_path

    def test_the_disk_cache_round_trips_to_an_equal_report(self, registry):
        first = registry.get_report(TREE_CELL, "tau_0.5")
        registry._bundles.clear()
        assert registry.get_report(TREE_CELL, "tau_0.5") == first

    def test_an_unknown_operating_point_falls_back_instead_of_rendering_empty(self, registry):
        """Switching methods carries a τ the new one never swept. A blank page reads as a bug."""
        assert registry.resolve_op(TREE_CELL, "tau_0.7") == "tau_0.1"
        assert registry.resolve_op("raghpo_8b/hcy", "tau_0.1") is None

    def test_gold_is_restricted_to_the_reports_the_run_processed(self, registry, output_base):
        """A shard covering an eighth of the cohort must not be blamed for the other seven."""
        write_gold_csv(registry.hcy_gt_path, {**GOLD, "r_never_run": [C]})
        registry._gold.clear()
        registry.rescan()
        bundle = registry.bundle(TREE_CELL, "tau_0.1")
        assert set(bundle["gold"]) == {"r1", "r2"}

    def test_the_tau_frontier_is_ordered_and_shows_the_ceiling_collapsing(self, registry):
        rows = registry.tau_frontier(TREE_CELL)
        assert [r["operating_point"] for r in rows] == ["tau_0.1", "tau_0.5"]
        assert rows[0]["recall_ceiling"] > rows[1]["recall_ceiling"]
        assert rows[0]["n_blocked"] == 0 and rows[1]["n_blocked"] == 1

    def test_the_frontier_of_a_two_axis_sweep_is_one_accept_slice(self, registry):
        """Over the whole grid every curve would double back, five points share each τ_prune.

        And each row is a full scoring pass, so the slice is also what keeps first open on
        an earlier exploratory run's 25 directories from costing five times what it needs to.
        """
        rows = registry.tau_frontier("tree_lse_beta1/hcy", "tau_0.1_acc_0.9")
        assert [r["operating_point"] for r in rows] == ["tau_0.1_acc_0.9", "tau_0.5_acc_0.9"]
        assert [r["op_value"] for r in rows] == [0.1, 0.5]
        assert {r["op_accept"] for r in rows} == {0.9}

    def test_the_accept_threshold_changes_what_is_kept_and_not_what_is_visited(self, registry):
        """The invariant that makes one BFS per τ_prune serve the whole accept grid."""
        loose = registry.get_report("tree_lse_beta1/hcy", "tau_0.1_acc_0.5")
        strict = registry.get_report("tree_lse_beta1/hcy", "tau_0.1_acc_0.9")
        assert strict["counts"]["n_predicted_terms"] < loose["counts"]["n_predicted_terms"]
        assert strict["counts"]["n_visited_nodes"] == loose["counts"]["n_visited_nodes"]
        assert strict["ledger"]["recall_ceiling"] == loose["ledger"]["recall_ceiling"]

    def test_phenobert_scores_through_the_flat_path(self, registry):
        report = registry.get_report("phenobert/hcy")
        assert report["is_tree"] is False and report["kind"] == "phenobert"
        assert report["flat"]["tp"] == 2
        assert report["ledger"]["fate_counts"]["blocked"] == 0

    def test_a_non_tree_method_gets_a_ledger_with_structurally_zero_pruning(self, registry):
        report = registry.get_report("raghpo_8b/hcy")
        assert report["is_tree"] is False
        assert report["ledger"]["fate_counts"]["blocked"] == 0
        assert report["ledger"]["recall_ceiling"] == 1.0
        assert report["flat"]["tp"] == 2

    def test_every_cell_builds(self, registry):
        """Including the ensemble and flat top-M, which have no nodes artifact at all."""
        for cell_id, cell in registry.cells.items():
            operating_point = registry.resolve_op(cell_id, None)
            report = registry.get_report(cell_id, operating_point)
            assert report is not None, cell_id
            if registry.gold_available(cell.cohort):
                assert report["counts"]["n_reports_scored"] >= 1, cell_id

    def test_a_cohort_without_ground_truth_degrades_instead_of_raising(self, registry):
        """An analyst looking at HCY must not get an exception page because GSC+ is unreachable."""
        assert not registry.gold_available("gsc")
        report = registry.get_report("tree_gate_lr/gsc", "tau_0.1")
        assert report is not None
        assert report["counts"]["n_reports_scored"] == 0
        assert report["status"] == "partial"
        assert any("GSC+" in p for p in report["problems"])


class TestClusterRegressions:
    """Four bugs `--selfcheck` found on the real cluster artifacts. Each fails without its fix."""

    def test_a_multi_organ_node_does_not_break_the_geometry_grouping(self, output_base,
                                                                     shared_ontology, tmp_path):
        """``layer1_organ`` is a *list*, HPO is a DAG (``core/node_metadata.py:58``).

        A list in a DataFrame column is unhashable, so ``groupby`` on it raised
        ``TypeError: unhashable type: 'list'`` and took down every tree cell on the cluster.
        """
        path = (output_base / "exp13_00_tree_gate_lr" / "hcy"
                / "tree_gate_lr_node_metadata.jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:                       # every node under two organ systems
            record["layer1_organ"] = [A, G]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(write_gold_csv(tmp_path / "gt.csv", GOLD)),
            cache_dir=str(tmp_path / "cache"))
        report = registry.get_report(TREE_CELL, "tau_0.5")
        assert report["geometry"]["by_layer1"], "the organ breakdown must still be computed"

    def test_the_metadata_loader_splits_the_organ_list(self, cells):
        frame = loaders.load_node_metadata(cells[TREE_CELL].files_for("node_metadata"))
        assert frame["layer1_organ"].map(lambda v: isinstance(v, (str, type(None)))).all()
        assert frame["layer1_organs"].map(lambda v: isinstance(v, list)).all()

    def test_scoring_compares_raw_strings_as_exp13_07_does(self, cells, toy_view):
        """``result_tables/loaders.aligned_sets`` hands raw sets to ``flat_report`` with no view.

        Resolving here instead moved micro precision by a fraction of a percent on GSC+, enough
        that every number in this UI disagreed with the thesis tables, for a reason invisible to
        a reader.
        """
        gold = {"r1": {OBSOLETE_C}}                  # An alt id that resolves to C
        predicted = {"r1": {C}}
        gold_sets, pred_sets, _ = scoring.align(gold, predicted, toy_view)
        assert gold_sets == [{OBSOLETE_C}] and pred_sets == [{C}], "no resolution by default"
        assert scoring.score(gold, predicted, toy_view)["flat"]["tp"] == 0

        canonical, _, _ = scoring.align(gold, predicted, toy_view, canonicalise=True)
        assert canonical == [{C}], "canonicalise=True is still available, and reports the change"
        assert scoring.unresolvable_counts(gold, predicted, toy_view)["gold_remapped"] == 1

    def test_the_node_table_labels_agree_with_the_scorecard(self, registry):
        """The table's TP/FP/FN must equal the flat counts, or the page contradicts itself."""
        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        table, report = bundle["node_table"], bundle["report"]
        counts = table["outcome"].value_counts()
        assert int(counts.get("TP", 0)) == report["flat"]["tp"]
        assert int(counts.get("FP", 0)) == report["flat"]["fp"]
        assert int(counts.get("FN", 0)) == report["flat"]["fn"]

    def test_an_operating_point_that_was_never_written_is_not_the_default(self, output_base,
                                                                         shared_ontology,
                                                                         tmp_path):
        """On the real gate-LR run ``tau_0.02`` sorts first and its directory is empty.

        Defaulting to the first point opened every such cell on a blank scorecard.
        """
        run_dir = output_base / "exp13_00_tree_gate_lr" / "hcy"
        for kind in ("nodes", "predictions", "traversal"):
            (run_dir / "tau_0.02").mkdir(exist_ok=True)
            (run_dir / "tau_0.02" / f"tree_gate_lr_{kind}.jsonl").write_text("")

        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(write_gold_csv(tmp_path / "gt.csv", GOLD)),
            cache_dir=str(tmp_path / "cache"))
        cell = registry.cell(TREE_CELL)
        assert cell.operating_points[0] == "tau_0.02", "it must still be offered"
        assert cell.empty_operating_points == ["tau_0.02"]
        assert registry.resolve_op(TREE_CELL, None) == "tau_0.1"
        assert any("never written" in p for p in cell.problems)
        assert registry.get_report(TREE_CELL, None)["counts"]["n_reports_scored"] == 2

    def test_an_explicitly_chosen_empty_point_is_still_honoured(self, output_base,
                                                               shared_ontology, tmp_path):
        """Offered means selectable. Silently redirecting would hide the state from the user."""
        run_dir = output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.02"
        run_dir.mkdir(exist_ok=True)
        for kind in ("nodes", "predictions"):
            (run_dir / f"tree_gate_lr_{kind}.jsonl").write_text("")
        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(write_gold_csv(tmp_path / "gt.csv", GOLD)),
            cache_dir=str(tmp_path / "cache"))
        assert registry.resolve_op(TREE_CELL, "tau_0.02") == "tau_0.02"

    def test_the_cross_check_reads_exp13_07s_own_gold_file(self, tmp_path):
        """The result-table library scored HCY against a *different* ground truth file than inference used.

        Comparing across that difference can only ever fail, and the failure says nothing about
        this UI. The paths are in ``config_resolved.yaml`` beside ``tables/``.
        """
        from apps.treephenorag_ui import selfcheck

        tables = tmp_path / "exp13_07" / "tables"
        tables.mkdir(parents=True)
        (tmp_path / "exp13_07" / "config_resolved.yaml").write_text(
            "results_dir: /somewhere\n"
            "hcy_gt_path: /data/hcy/hcy_ground_truth_raw.csv\n"
            "gsc_dir: /data/repo/resources/data/GSC_2024\n")
        assert selfcheck._exp13_07_gold(str(tables)) == {
            "hcy_gt_path": "/data/hcy/hcy_ground_truth_raw.csv",
            "gsc_dir": "/data/repo/resources/data/GSC_2024",
        }

    def test_an_unreadable_exp13_07_gold_warns_and_falls_back(self, registry, tmp_path):
        from apps.treephenorag_ui import selfcheck

        tables = tmp_path / "exp13_07" / "tables"
        tables.mkdir(parents=True)
        (tmp_path / "exp13_07" / "config_resolved.yaml").write_text(
            "hcy_gt_path: /nonexistent/raw.csv\ngsc_dir: /nonexistent/gsc\n")
        registry.thesis_tables_dir = str(tables)
        result = selfcheck.Result()
        scoped, note = selfcheck._scoring_registry(registry, result)
        assert scoped is registry
        assert any("not readable" in w for w in result.warnings)


class TestTheme:
    def test_display_vocabularies_are_fixed_to_the_metric_library(self):
        theme.assert_vocabularies_match()


# ── rendering helpers ────────────────────────────────────────────────────────

def _plain(node):
    """A Dash component tree as plain JSON-able data.

    ``str()`` on a component truncates its children with an ellipsis and ``to_plotly_json`` only
    descends one level, so neither on its own can see a whole panel. An assertion that searched a
    truncated repr would pass or fail on the panel's *size*, not on its content.
    """
    if hasattr(node, "to_plotly_json"):
        return _plain(node.to_plotly_json())
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(v) for v in node]
    return node if isinstance(node, (str, int, float, bool, type(None))) else str(node)


def flat(component) -> str:
    """The whole rendered tree as one searchable string."""
    return json.dumps(_plain(component), ensure_ascii=False)


def clipboard_content(component) -> str:
    """What a panel's copy button would put on the clipboard."""
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "Clipboard":
                yield node.get("props", {}).get("content", "")
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    return next(iter(walk(_plain(component))), "")


# ── the Dash app ─────────────────────────────────────────────────────────────

def _layout_ids(component, found=None, patterns=None):
    """``(string ids, pattern-matching id "type" values)`` present in a layout tree.

    Pattern-matching ids are dicts (``{"type": "copy-png", "graph": "sc-fate"}``) and are matched
    by their ``type``: a wildcard callback is satisfied by any component of that type, which is
    what Dash resolves at runtime.
    """
    found = set() if found is None else found
    patterns = set() if patterns is None else patterns
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        found.add(component_id)
    elif isinstance(component_id, dict) and "type" in component_id:
        patterns.add(component_id["type"])

    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _layout_ids(child, found, patterns)
    elif children is not None:
        _layout_ids(children, found, patterns)
    return found, patterns


def _referenced_ids(app):
    """Every id a registered callback names, as ``(kind, name, where)``.

    ``kind`` is ``"id"`` for a plain string component id and ``"type"`` for a pattern-matching
    one, whose ``type`` is what the layout must supply.
    """
    referenced: set[tuple[str, str, str]] = set()

    def add(identifier, where):
        # Dash stringifies pattern-matching ids in ``callback_map``, both in the output key and
        # in the input records, so a leading brace means "this is a dict, not a component name".
        if isinstance(identifier, str) and identifier.startswith("{"):
            try:
                identifier = json.loads(identifier)
            except json.JSONDecodeError:
                return
        if isinstance(identifier, str):
            referenced.add(("id", identifier, where))
        elif isinstance(identifier, dict) and "type" in identifier:
            referenced.add(("type", identifier["type"], where))

    for key, spec in app.callback_map.items():
        for part in key.split("..."):
            part = part.strip(".")
            if not part:
                continue
            if part.startswith("{"):
                try:
                    add(json.loads(part.rsplit(".", 1)[0]), f"Output of {key}")
                except json.JSONDecodeError:
                    pass
            else:
                add(part.rsplit(".", 1)[0], f"Output of {key}")
        for dependency in list(spec.get("inputs", ())) + list(spec.get("state", ())):
            add(dependency.get("id"), f"Input/State of {key}")
    return referenced


@pytest.fixture
def dash_app(registry):
    import dash

    from apps.treephenorag_ui import app as app_module, state

    state.set_registry(registry)
    application = dash.Dash(__name__, suppress_callback_exceptions=True)
    application.layout = app_module.build_layout(registry)
    app_module.register_callbacks(application)
    return application


class TestDashWiring:
    def test_every_callback_names_a_component_that_exists(self, dash_app, registry):
        """Dash does not check this, and the symptom is a tab that silently renders nothing.

        The comparison set is the static layout *plus* everything the view builders emit, since
        panels, and with them the copy buttons the clientside callback targets, are created
        inside callbacks. Rendering the builders here, not allowlisting their ids is what
        keeps the check honest: a typo in a dynamically-created id still fails.
        """
        from apps.treephenorag_ui.views import calibration, compare, deepdive, pruning_view, scorecard

        present, patterns = _layout_ids(dash_app.layout)
        report = registry.get_report(TREE_CELL, "tau_0.5")
        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        for rendered in (scorecard._build(report, "light"),
                         pruning_view._build(report, "light"),
                         pruning_view._build_frontier(TREE_CELL, report,
                                                      registry.tau_frontier(TREE_CELL), "light"),
                         calibration._build(report, "light"),
                         compare._build(TREE_CELL, "tau_0.1", TREE_CELL, "tau_0.5", "light"),
                         # The deep-dive's two halves: the calls panel creates the `sent #N`
                         # badges that point back at the report, and the report creates the
                         # sentences they point at. Both are pattern ids built inside a callback.
                         deepdive._calls_panel(registry, bundle, TREE_CELL, report, "r1", C),
                         deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), {0})):
            _layout_ids(rendered, present, patterns)

        missing = sorted({
            f"{kind}={name} ({where})"
            for kind, name, where in _referenced_ids(dash_app)
            if name not in (present if kind == "id" else patterns)
        })
        assert not missing, "callbacks reference ids that are not in the layout:\n" + \
                            "\n".join(missing)

    def test_a_tab_exists_for_every_view(self, dash_app):
        from apps.treephenorag_ui import app as app_module

        present, _ = _layout_ids(dash_app.layout)
        for view_id, _, _ in app_module.VIEWS:
            assert f"view-{view_id}" in present

    def test_the_tab_switcher_covers_every_view(self, dash_app):
        from apps.treephenorag_ui import app as app_module

        key = next(k for k in dash_app.callback_map if k.startswith("..view-"))
        assert key.count("view-") == len(app_module.VIEWS)


class _CaptureApp:
    """A stand-in for ``dash.Dash`` that records callback bodies instead of registering them.

    ``register(app)`` is the only thing standing between a view module and the browser, and the
    wiring test checks the ids it declares, not that the bodies run. Capturing the undecorated
    functions lets the bodies be called against the fixture registry, which is the difference
    between "the tab exists" and "the tab renders".
    """

    def __init__(self):
        self.captured: list[tuple[object, list]] = []

    def callback(self, *args, **kwargs):
        dependencies = [a for a in args if hasattr(a, "component_id")]

        def decorator(function):
            self.captured.append((function, dependencies))
            return function

        return decorator

    def clientside_callback(self, *args, **kwargs):  # nothing to run server-side
        return None


#: What to feed each ``(component_id, property)`` a callback asks for. Anything unlisted is None,
#: which is the state the app is genuinely in before the user touches a control.
_CALLBACK_ARGS = {
    ("run-a", "value"): TREE_CELL,
    ("op-a", "value"): "tau_0.5",
    ("run-b", "value"): "raghpo_8b/hcy",
    ("op-b", "value"): None,
    ("store-theme", "data"): "light",
    ("store-focus", "data"): None,
    ("tabs", "value"): "scorecard",   # overridden per view by _run_captured
    ("err-side", "value"): "FP",
    ("err-table", "page_current"): 0,
    ("err-table", "sort_by"): None,
    ("err-table", "selected_rows"): [],
    ("err-table", "data"): [],
    ("dd-report", "value"): "r1",
    ("dd-report", "options"): [{"label": "r1", "value": "r1"}, {"label": "r2", "value": "r2"}],
    ("dd-report-prev", "n_clicks"): 1,
    ("dd-report-next", "n_clicks"): 1,
    ("dd-node-filter", "value"): "errors",
    ("dd-node", "value"): C,
    ("src-load", "n_clicks"): 1,
    ("present-btn", "n_clicks"): 1,
    ("theme-toggle", "value"): "dark",
}


#: Which tab each view module owns. A guarded callback does nothing unless its tab is active, so
#: a harness that left ``tabs.value`` on "scorecard" would exercise one view and silently skip the
#: other five while still reporting success.
_VIEW_TABS = {
    "scorecard": "scorecard", "pruning_view": "pruning", "errors_view": "errors",
    "calibration": "calibration", "deepdive": "deepdive", "compare": "compare",
}


def _run_captured(module, registry, overrides=None, entry: str = "register"):
    """Call every server-side callback a module registers. Returns how many actually ran.

    "Actually ran" excludes the ones that raised ``PreventUpdate``, so a test asserting a positive
    count cannot be satisfied by a guard that skipped everything.
    """
    from dash.exceptions import PreventUpdate

    from apps.treephenorag_ui import state

    state.set_registry(registry)
    capture = _CaptureApp()
    getattr(module, entry)(capture)

    tab = _VIEW_TABS.get(module.__name__.rsplit(".", 1)[-1])
    values = {**_CALLBACK_ARGS, **({("tabs", "value"): tab} if tab else {}),
              **(overrides or {})}

    ran = 0
    for function, dependencies in capture.captured:
        args = [values.get((d.component_id, d.component_property))
                for d in dependencies
                if isinstance(d.component_id, str) and d.__class__.__name__ in ("Input", "State")]
        try:
            function(*args)
        except PreventUpdate:
            continue
        ran += 1
    return ran


class TestCallbackBodiesRun:
    """Every server-side callback, executed against the fixture. No browser, real code paths."""

    @pytest.mark.parametrize("module_name", ["scorecard", "pruning_view", "errors_view",
                                             "calibration", "deepdive", "compare"])
    def test_a_view_module_renders_without_raising(self, module_name, registry):
        import importlib

        module = importlib.import_module(f"apps.treephenorag_ui.views.{module_name}")
        assert _run_captured(module, registry) > 0

    def test_the_deep_dive_survives_a_report_with_no_selected_node(self, registry):
        from apps.treephenorag_ui.views import deepdive

        assert _run_captured(deepdive, registry, {("dd-node", "value"): None}) > 0

    def test_the_views_survive_an_empty_selection(self, registry):
        """The state the app is in before anything is picked, and after a failed Load."""
        import importlib

        blank = {("run-a", "value"): None, ("run-b", "value"): None,
                 ("op-a", "value"): None, ("dd-report", "value"): None}
        for module_name in ("scorecard", "pruning_view", "errors_view", "calibration",
                            "deepdive", "compare"):
            module = importlib.import_module(f"apps.treephenorag_ui.views.{module_name}")
            _run_captured(module, registry, blank)

    def test_a_non_tree_cell_renders_in_every_view(self, registry):
        """RAG-HPO has no nodes, no traversal and no scores, every view must cope."""
        import importlib

        overrides = {("run-a", "value"): "raghpo_8b/hcy", ("op-a", "value"): None}
        for module_name in ("scorecard", "pruning_view", "errors_view", "calibration",
                            "deepdive", "compare"):
            module = importlib.import_module(f"apps.treephenorag_ui.views.{module_name}")
            _run_captured(module, registry, overrides)

    def test_the_app_shells_own_callbacks_run(self, registry, tmp_path):
        from apps.treephenorag_ui import app as app_module

        overrides = {
            ("src-output", "value"): registry.output_base,
            ("src-gt", "value"): registry.hcy_gt_path,
            ("src-input", "value"): "",
            ("src-gsc", "value"): "",
            ("src-stanza", "value"): "",
            ("src-gt-preset", "value"): registry.hcy_gt_path,
        }
        assert _run_captured(app_module, registry, overrides,
                             entry="register_callbacks") > 0


class TestOnlyTheActiveTabWorks:
    """The guards are the difference between a page in a second and a page in half a minute.

    They are also invisible when they break: a missing guard costs nothing but time, and a guard
    on the wrong tab shows a blank panel. Both are asserted, not assumed.
    """

    @pytest.mark.parametrize("module_name,tab", sorted(_VIEW_TABS.items()))
    def test_a_view_does_nothing_while_another_tab_is_open(self, module_name, tab, registry):
        import importlib

        module = importlib.import_module(f"apps.treephenorag_ui.views.{module_name}")
        other = "compare" if tab != "compare" else "scorecard"
        assert _run_captured(module, registry, {("tabs", "value"): other}) == 0
        assert _run_captured(module, registry, {("tabs", "value"): tab}) > 0

    def test_every_content_callback_declares_the_tab_input(self, dash_app):
        """A callback that renders content but never reads ``tabs.value`` cannot be guarded."""
        unguarded = []
        for key, spec in dash_app.callback_map.items():
            outputs = [p.strip(".") for p in key.split("...") if p.strip(".")]
            if not any(o.split(".")[0].startswith(
                    ("scorecard-", "pr-", "err-", "cal-", "dd-", "cmp-")) for o in outputs):
                continue
            ids = {d.get("id") for d in list(spec.get("inputs", ()))
                   + list(spec.get("state", ()))}
            if "tabs" not in ids:
                unguarded.append(key)
        assert not unguarded, f"content callbacks with no tab guard: {unguarded}"


class TestHeavySectionsAreDeferred:
    """The error taxonomy and the calibration block are seconds of work read by one tab each."""

    def test_the_light_report_omits_them(self, registry):
        report = registry.get_report(TREE_CELL, "tau_0.5")
        assert report["heavy"] is False
        assert report["errors"] is None
        assert report["calibration"] is None
        # The cheap parts of the calibration family stay in the core report.
        assert report["separation"] and report["accept_curve"]

    def test_the_heavy_report_adds_them(self, registry):
        report = registry.get_report(TREE_CELL, "tau_0.5", heavy=True)
        assert report["heavy"] is True
        assert report["errors"]["fp_taxonomy"]["n_fp"] >= 1
        assert report["calibration"]["accept"] is not None

    def test_the_light_node_table_skips_error_classification(self, registry):
        """Classification walks the ontology per false positive, the run's biggest table cost."""
        light = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        heavy = registry.bundle(TREE_CELL, "tau_0.5", heavy=True)["node_table"]
        assert light["fp_class"].isna().all()
        assert heavy.loc[heavy["outcome"] == "FP", "fp_class"].notna().any()

    def test_the_two_reports_agree_on_everything_they_share(self, registry):
        """Deferring must change *when* a number is computed, never what it is."""
        light = registry.get_report(TREE_CELL, "tau_0.5")
        heavy = registry.get_report(TREE_CELL, "tau_0.5", heavy=True)
        for key in ("flat", "hierarchy", "ledger", "blocking", "culprits", "traversal", "counts"):
            assert light[key] == heavy[key], key

    def test_a_heavy_bundle_satisfies_a_light_request_but_not_the_reverse(self, registry):
        registry.bundle(TREE_CELL, "tau_0.5", heavy=True)
        assert registry.bundle(TREE_CELL, "tau_0.5")["heavy"] is True, \
            "a heavy bundle already in the LRU should be reused for a light request"
        registry._bundles.clear()
        registry.bundle(TREE_CELL, "tau_0.5")
        assert registry.bundle(TREE_CELL, "tau_0.5", heavy=True)["heavy"] is True, \
            "a light bundle must never be handed back for a heavy request"

    def test_the_two_reports_cache_to_different_files(self, registry):
        import os

        registry.get_report(TREE_CELL, "tau_0.5")
        registry.get_report(TREE_CELL, "tau_0.5", heavy=True)
        cached = os.listdir(registry.cache_dir)
        assert any(f.endswith(".report.json") for f in cached)
        assert any(f.endswith(".report.heavy.json") for f in cached)


class TestBlockingDpIsNotQuadraticInReports:
    def test_the_topological_order_is_computed_once_per_graph(self, monkeypatch):
        """It was recomputed per report though the graph is the same ontology every time."""
        children_map, roots = toy_children_map()
        calls = {"n": 0}
        original = pruning._topological_order

        def counted(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(pruning, "_topological_order", counted)
        pruning._ORDER_CACHE.clear()
        for _ in range(25):
            pruning.blocking_culprits(children_map, roots, {A, B})
        assert calls["n"] == 1

    def test_a_different_graph_is_not_served_a_stale_order(self):
        """The cache is keyed on identity, so it must verify the object it was built from."""
        first, roots = toy_children_map()
        pruning._ORDER_CACHE.clear()
        pruning.blocking_culprits(first, roots, {A})
        assert C in pruning.blocking_culprits(first, roots, {A, B, G})

        second = {node: list(children) for node, children in first.items()}
        second[B] = []          # C and D now have no route from any root at all
        result = pruning.blocking_culprits(second, roots, {A, B, G})
        # The DP only covers nodes reachable in the *unpruned* graph, so C dropping out is the
        # observable proof that the second graph's own topological order was used, not the
        # first one's, which is still in the cache under a different identity.
        assert C not in result and D not in result
        assert M in result, "M still hangs off H, so it stays reachable"


class TestFiguresRenderHeadlessly:
    """Every figure builder, called on a real report, the charts must survive the data."""

    def test_scorecard_figures(self, registry):
        from apps.treephenorag_ui.views import scorecard

        report = registry.get_report(TREE_CELL, "tau_0.5")
        for mode in ("light", "dark"):
            assert scorecard._fate_figure(report, mode).to_dict()
            assert scorecard._recall_gap_figure(report, mode).to_dict()
        assert scorecard._build(report, "light") is not None

    def test_pruning_figures(self, registry):
        from apps.treephenorag_ui.views import pruning_view

        report = registry.get_report(TREE_CELL, "tau_0.5")
        assert pruning_view._blocking_figure(report, "light").to_dict()
        assert pruning_view._depth_cost_figure(report, "light").to_dict()
        assert pruning_view._frontier_figure(registry.tau_frontier(TREE_CELL), "dark").to_dict()
        assert pruning_view._build(report, "light") is not None

    def test_error_figures_for_both_sides(self, registry):
        from apps.treephenorag_ui.views import errors_view

        report = registry.get_report(TREE_CELL, "tau_0.5")
        frame = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        for side in ("FP", "FN"):
            assert errors_view._taxonomy_figure(report, side, "light").to_dict()
            assert errors_view._score_figure(report, side, frame, "light").to_dict()
            assert errors_view._summary(report, side) is not None
            rows, pages, total, _ = errors_view._rows(frame, side, None, None, None, 0)
            assert total >= 1 and pages >= 1 and rows

    def test_calibration_and_compare_and_deepdive(self, registry):
        from apps.treephenorag_ui.views import calibration, compare, deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        assert calibration._build(report, "light") is not None
        assert compare._build(TREE_CELL, "tau_0.1", TREE_CELL, "tau_0.5", "light") is not None
        assert compare._build(TREE_CELL, "tau_0.5", "raghpo_8b/hcy", None, "light") is not None
        subset = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        subset = subset.loc[subset["report_id"] == "r1"]
        assert deepdive._walk_figure(subset, "light").to_dict()
        assert deepdive._node_options(subset, "errors")

    def test_the_pruning_view_refuses_a_non_tree_method_instead_of_drawing_zeros(self, registry):
        from apps.treephenorag_ui.views import calibration, pruning_view

        report = registry.get_report("raghpo_8b/hcy")
        for built in (pruning_view._build(report, "light"), calibration._build(report, "light")):
            assert "no ontology traversal" in str(built) or "no per-node score" in str(built)


class TestReportDeepDive:
    """The page a reader lands on to turn one number into a claim they can check."""

    def _subset(self, registry, report_id="r1"):
        frame = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        return frame.loc[frame["report_id"] == report_id]

    def test_the_report_panel_shows_the_annotated_terms_and_the_sentences(self, registry):
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        panel = flat(deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), set()))
        for i, sentence in enumerate(SEGMENTS["r1"]):
            assert f"sentence #{i}" in panel,                 "the sent_index numbering is the join to the call records"
            assert sentence in panel, "a sentence of the report is not on the page"
        assert C in panel and F in panel, "the annotated terms are not on the page"

    def test_the_retrieved_sentences_are_marked_for_the_selected_node(self, registry):
        from apps.treephenorag_ui.views import deepdive

        # The fixture retrieves sentences 0, 1 and 2 for C in r1 (``write_calls``' ranking).
        indices = deepdive.retrieved_indices(registry, TREE_CELL, "r1", C)
        assert indices == {0, 1, 2}
        assert deepdive.retrieved_indices(registry, TREE_CELL, "r1", None) == set()

        report = registry.get_report(TREE_CELL, "tau_0.5")
        marked = flat(deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), {0}))
        assert "seg-retrieved" in marked
        assert "retrieved for the selected node" in marked, "the mark does not say what it means"

    def test_the_baseline_matches_are_underlined_inside_the_numbered_sentences(self, registry):
        """The three layers coexist: sentence numbers, retrieval marks, and PhenoBERT's matches.

        The fixture's the PhenoBERT baseline cell stages the cohort's own report and detects one phrase in each
        of r1's three sentences, covering a match that became a prediction and one whose code this
        release does not have.
        """
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        panel = flat(deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), {0}, "light"))

        assert "sentence #0" in panel and "seg-retrieved" in panel,             "the sentence layers were lost"
        assert "pb-ann" in panel and "pb-tag" in panel, "no baseline annotation was drawn"
        # The tag names the term, not the code, it is read at a glance inside flowing text.
        assert "Seizure" in panel
        assert "not in this HPO release" in panel, "the unresolved match is not called out"
        assert "became a PhenoBERT prediction" in panel, "the legend is missing"

    def test_the_copy_button_hands_over_the_report_without_the_annotations(self, registry):
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        panel = deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), set(), "light")
        assert clipboard_content(panel) == REPORT_TEXT["r1"]

    def test_the_annotation_layer_degrades_without_taking_the_panel_with_it(self, registry):
        """No PhenoBERT baseline run is an ordinary state, and the page must say so, not break."""
        from apps.treephenorag_ui.views import deepdive

        registry._pb["hcy"] = None
        report = registry.get_report(TREE_CELL, "tau_0.5")
        panel = deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), {0}, "light")
        rendered = flat(panel)

        assert "no PhenoBERT baseline run was found" in rendered, "the reason is not on screen"
        assert "pb-tag" not in rendered, "annotations drawn with no baseline"
        assert "sentence #0" in rendered and "seg-retrieved" in rendered,             "the other layers were lost too"
        assert clipboard_content(panel) == REPORT_TEXT["r1"]

    def test_an_offset_that_cannot_be_honoured_is_not_drawn(self, toy_view):
        """Overlapping and out-of-range spans are skipped, never drawn approximately."""
        from apps.treephenorag_ui.views import deepdive

        positive = {"hpo_id": C, "phrase": "seiz", "negated": False, "resolved": True}
        overlapping = {"hpo_id": D, "phrase": "eizu", "negated": True, "resolved": True}
        out_of_range = {"hpo_id": F, "phrase": "xx", "negated": False, "resolved": False}

        children, codes = deepdive._weave(
            "seizures here", [(0, 4, positive), (1, 5, overlapping), (20, 30, out_of_range)],
            (), None, toy_view, "light")
        assert flat(children).count("pb-ann") == 1, "an overlap or an out-of-range span was drawn"
        assert not codes, "no gold was passed, so nothing is drawn as gold"

        # All three kinds, given room not to overlap.
        children, _codes = deepdive._weave(
            "a b c", [(0, 1, positive), (2, 3, overlapping), (4, 5, out_of_range)],
            (), None, toy_view, "light")
        assert flat(children).count("pb-ann") == 3

    def test_the_gold_outranks_a_detection_for_the_same_words(self, toy_view):
        """The ground truth is somebody asserting a phenotype about these words; PhenoBERT is a guess.

        A detection that overlaps an annotation is dropped, not drawn, because the two marks
        cannot nest, and it is the annotation that has to survive, or the report would show a
        guess where a curator wrote something down.
        """
        from apps.treephenorag_ui.curated import Annotation
        from apps.treephenorag_ui.views import deepdive

        detection = {"hpo_id": D, "phrase": "seizures", "negated": False, "resolved": True}
        annotation = Annotation(patient_id="r1", hpo_code=C, hpo_name="Seizure", in_gold=True,
                                segment_idx=0, trigger_word="seizures")

        children, codes = deepdive._weave(
            "seizures here", [(0, 8, detection)], [(0, 8, annotation)], {C}, toy_view, "light")
        rendered = flat(children)
        assert codes == {C}, "the gold was not drawn"
        assert "gold-mark" in rendered
        assert rendered.count("pb-ann") == 1, "the overlapping detection was drawn as well"

    def test_the_terms_table_names_how_each_error_is_wrong(self, registry, toy_view):
        from apps.treephenorag_ui.views import deepdive

        subset = self._subset(registry)
        gold, predicted = set(GOLD["r1"]), set(ACCEPTED["r1"])
        relations = deepdive._relations(subset, gold, predicted, toy_view)
        # D is predicted and not annotated, and it is a sibling of the annotated C under B.
        assert relations[D][0] == "sibling" and relations[D][1] == C
        # F is annotated and missed, with nothing predicted anywhere near it.
        assert relations[F][0] == "nothing_near"

        rendered = str(deepdive._terms_panel(subset, gold, predicted, toy_view, "r1"))
        assert "sibling" in rendered and "nothing_near" in rendered
        assert "TP" in rendered and "FP" in rendered and "FN" in rendered

    def test_the_organ_system_refinement_rides_on_the_thesis_bucket(self, toy_view):
        from apps.treephenorag_ui.views import deepdive

        # C and D share the layer-1 system A; F sits under the same system too, G does not.
        assert deepdive._same_system(toy_view, C, {D})
        assert not deepdive._same_system(toy_view, C, {H})

    def test_the_diagram_carries_the_hpo_id_a_click_reads_back(self, registry):
        from apps.treephenorag_ui.views import deepdive

        subset = self._subset(registry)
        built = subgraph.build(subgraph.terms_of_report(subset), registry.view)
        figure = deepdive._tree_figure(built, {}, C, "light")
        assert figure.to_dict()
        ids = {str(row[0]) for trace in figure.data
               for row in (trace.customdata if trace.customdata is not None else ())}
        assert {C, F} <= ids

        click = {"points": [{"customdata": [C, "Seizure", 3]}]}
        assert deepdive.clicked_hpo(click) == C
        assert deepdive.clicked_hpo(None) is None
        assert deepdive.clicked_hpo({"points": []}) is None

    def test_the_node_panel_opens_on_a_false_positive(self, registry):
        """The taxonomy columns are ``NA`` on a light bundle, and ``if NA`` raises.

        Which made opening any false positive, the node a reader comes here for, a TypeError,
        not a page. The classification is now done for the one node instead of read off a
        column the light bundle never fills.
        """
        from apps.treephenorag_ui.views import deepdive

        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        subset = self._subset(registry)
        panel = str(deepdive._node_panel(bundle, subset,
                                         registry.get_report(TREE_CELL, "tau_0.5"), "r1", D,
                                         TREE_CELL, set(GOLD["r1"]), set(ACCEPTED["r1"])))
        assert "sibling" in panel and "no_gold" not in panel
        assert "margin" in panel, "the SLM's margin is what the verdict was decided on"
        assert "The patient had seizures." in panel, "the retrieved sentence is not shown"

    def test_a_blocked_annotated_term_names_the_node_that_severed_it(self, registry):
        from apps.treephenorag_ui.views import deepdive

        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        subset = self._subset(registry)
        panel = str(deepdive._node_panel(bundle, subset,
                                         registry.get_report(TREE_CELL, "tau_0.5"), "r1", F,
                                         TREE_CELL, set(GOLD["r1"]), set(ACCEPTED["r1"])))
        assert "Never reached" in panel and E in panel

    def test_the_two_score_tiles_are_tinted_by_what_the_run_did_with_them(self, registry):
        """From the recorded booleans, not from a threshold comparison.

        A 1-D ``tau_0.5`` point does not carry τ_accept in its name at all, so a comparison would
        have to guess it. ``accepted``/``expanded`` are what the traversal actually did.
        """
        from apps.treephenorag_ui.views import deepdive

        subset = self._subset(registry)
        # C at tau_0.5: accepted, and not expanded.
        record = deepdive._row_for(subset, registry.view, C).iloc[0]
        tiles = flat(html.Div(deepdive._score_tiles(record, "tau_0.5_acc_0.9")))
        assert '"data-tint": "good"' in tiles, "the accepted score is not marked as accepted"
        assert '"data-tint": "warning"' in tiles, "the pruned score is not marked as pruned"
        assert "τ_accept 0.9" in tiles and "τ_prune 0.5" in tiles, "the thresholds are not named"
        assert "accepted, so reported" in tiles
        assert "pruned, so this branch stopped here" in tiles

        # A 1-D point names the one threshold it has and does not invent the other.
        one_d = flat(html.Div(deepdive._score_tiles(record, "tau_0.5")))
        assert "τ_prune 0.5" in one_d and "τ_accept 0.9" not in one_d

    def test_the_selected_term_carries_its_meaning_above_the_evidence(self, registry):
        from apps.treephenorag_ui.views import deepdive

        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        panel = flat(deepdive._node_panel(bundle, self._subset(registry),
                                          registry.get_report(TREE_CELL, "tau_0.5"), "r1", C,
                                          TREE_CELL, set(GOLD["r1"]), set(ACCEPTED["r1"]),
                                          "tau_0.5"))
        assert "definition" in panel
        # The toy ontology carries no ``Def``, so the honest fallback is what must appear.
        assert "No definition is recorded" in panel
        assert flat(deepdive._term_meaning(registry.view, HALLUCINATION)).count(
            "not in this HPO release") == 1

    def test_a_term_recorded_under_an_alt_id_is_still_found(self, registry):
        """The diagram resolves ids. The node table does not. Matching on one alone loses the row.

        A term the artifacts wrote under an alt id would otherwise be reported as never scored, which is worse than no answer, because it is a confident wrong one.
        """
        from apps.treephenorag_ui.views import deepdive

        subset = self._subset(registry)
        row = deepdive._row_for(subset, registry.view, OBSOLETE_C)
        assert row is not None and row.iloc[0]["hpo_id"] == C
        assert deepdive._row_for(subset, registry.view, HALLUCINATION) is None

    def test_an_unevaluated_node_names_the_pruning_that_stopped_the_walk(self, registry):
        """Clicking a context ancestor used to answer "not in this report's table" and stop."""
        from apps.treephenorag_ui.views import deepdive

        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        subset = self._subset(registry)

        # F sits under E, which was scored and not expanded. Drop F's own row to make it the
        # unevaluated node the diagram can produce.
        without_f = subset.loc[subset["hpo_id"] != F]
        assert deepdive._row_for(without_f, registry.view, F) is None

        culprit = deepdive.nearest_pruned_ancestor(registry.view, without_f, F)
        assert culprit["hpo_id"] == E and culprit["hops"] == 1
        assert culprit["prune_score"] == pytest.approx(0.3)

        panel = flat(deepdive._unevaluated_panel(
            registry, bundle, without_f, registry.get_report(TREE_CELL, "tau_0.5"), "r1", F,
            TREE_CELL))
        assert "not scored" in panel and "Never scored" in panel
        assert E in panel and "1 hop above" in panel
        assert "0.300" in panel, "the culprit's prune score is not shown"

    def test_a_node_with_nothing_pruned_above_it_says_so(self, registry):
        from apps.treephenorag_ui.views import deepdive

        subset = self._subset(registry)
        only_expanded = subset.loc[subset["expanded"].astype(bool)]
        assert deepdive.nearest_pruned_ancestor(registry.view, only_expanded, F) is None

        panel = flat(deepdive._unevaluated_panel(
            registry, registry.bundle(TREE_CELL, "tau_0.5"), only_expanded,
            registry.get_report(TREE_CELL, "tau_0.5"), "r1", F, TREE_CELL))
        assert "outside the walk" in panel and "Never scored" not in panel

    def test_stepping_through_reports_is_clamped_at_both_ends(self):
        from apps.treephenorag_ui.views import deepdive

        values = ["r1", "r2", "r3"]
        assert deepdive.step_to("dd-report-next", "r1", values) == "r2"
        assert deepdive.step_to("dd-report-prev", "r3", values) == "r2"
        # A value that is not in the list lands on the first, not nowhere.
        assert deepdive.step_to("dd-report-next", "gone", values) == "r1"
        for trigger, current in (("dd-report-prev", "r1"), ("dd-report-next", "r3"),
                                 (None, "r1"), ("dd-node", "r1")):
            with pytest.raises(PreventUpdate):
                deepdive.step_to(trigger, current, values)
        with pytest.raises(PreventUpdate):
            deepdive.step_to("dd-report-next", "r1", [])

    def test_a_selection_the_filter_excludes_stays_in_the_dropdown(self, registry):
        """A diagram click can select a node the Node filter does not list.

        Dash renders a value with no matching option as an empty box, which would leave the control
        blank while the panel below it kept describing the node.
        """
        from apps.treephenorag_ui.views import deepdive

        subset = self._subset(registry)
        errors_only = deepdive._node_options(subset, "errors")
        assert C not in {o["value"] for o in errors_only}, "C is a TP; the filter should drop it"

        with_c = deepdive.with_selected(errors_only, C, subset, registry.view)
        assert with_c[0]["value"] == C and len(with_c) == len(errors_only) + 1
        # Nothing to add when the option is already there, or when the table does not have it.
        assert deepdive.with_selected(errors_only, D, subset, registry.view) == errors_only
        assert deepdive.with_selected(errors_only, HALLUCINATION, subset,
                                      registry.view) == errors_only

    def test_the_metrics_are_this_reports_own(self, registry):
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        rendered = str(deepdive._summary(self._subset(registry), report, "r1",
                                         set(GOLD["r1"]), set(ACCEPTED["r1"])))
        # r1 at tau_0.5: ground truth {C, F}, predicted {C, D}, 1 TP, 1 FP, 1 FN, F blocked by the prune.
        assert "1 / 1 / 1" in rendered
        assert "0.500" in rendered, "this report's own F1 is not on the page"

    def test_a_non_tree_cell_still_gets_a_page(self, registry):
        """RAG-HPO has no traversal, no scores and no calls. Every panel must cope."""
        from apps.treephenorag_ui.views import deepdive

        bundle = registry.bundle("raghpo_8b/hcy")
        subset = bundle["node_table"]
        subset = subset.loc[subset["report_id"] == "r1"]
        assert deepdive._terms_panel(subset, set(GOLD["r1"]), set(ACCEPTED["r1"]),
                                     registry.view, "r1") is not None
        built = subgraph.build(subgraph.terms_of_report(subset), registry.view)
        assert deepdive._tree_figure(built, {}, None, "dark").to_dict()


class TestCopyExport:
    def test_every_markdown_block_carries_its_provenance(self, registry):
        from apps.treephenorag_ui import copyexport

        report = registry.get_report(TREE_CELL, "tau_0.5")
        frontier = registry.tau_frontier(TREE_CELL)
        for text in (copyexport.scorecard_md(report), copyexport.pruning_md(report),
                     copyexport.fp_md(report), copyexport.fn_md(report),
                     copyexport.calibration_md(report),
                     copyexport.frontier_md(report, frontier)):
            assert "tau_0.5" in text or "tau_0.1" in text
            assert "gold=" in text

    def test_the_findings_block_covers_the_whole_cell(self, registry):
        from apps.treephenorag_ui import copyexport

        report = registry.get_report(TREE_CELL, "tau_0.5")
        block = copyexport.findings_block(report, registry.tau_frontier(TREE_CELL))
        assert "Where every annotated term went" in block
        assert "Culprits" in block


class TestSelfcheck:
    def test_a_clean_fixture_passes(self, registry, caplog):
        from apps.treephenorag_ui import selfcheck

        assert selfcheck.run(registry) == 0

    def test_the_annotation_overlay_is_reported_but_never_fatal(self, registry, caplog):
        """The layer is additive, so its absence is a warning and never an exit code."""
        import logging

        from apps.treephenorag_ui import selfcheck

        with caplog.at_level(logging.INFO, logger="exp13_ui.selfcheck"):
            assert selfcheck.run(registry) == 0
        assert "placed onto the sentences of all" in caplog.text, caplog.text[-2000:]

        # A cohort with no baseline warns, names where it looked, and still exits clean.
        registry._pb["hcy"] = registry._pb["gsc"] = None
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="exp13_ui.selfcheck"):
            assert selfcheck.run(registry) == 0
        assert "no PhenoBERT baseline run found" in caplog.text
        assert "Looked in:" in caplog.text

    def test_an_absent_sampling_frame_warns_and_stays_clean(self, registry, caplog):
        """The subset filter is additive: no frame disables a control and nothing else."""
        import logging

        from apps.treephenorag_ui import selfcheck

        with caplog.at_level(logging.INFO, logger="exp13_ui.selfcheck"):
            assert selfcheck.run(registry) == 0
        assert "no deep-dive sampling frame" in caplog.text

    def test_a_frame_that_shares_no_report_id_is_a_failure(self, output_base, shared_ontology,
                                                           tmp_path, report_dir, caplog):
        """The one case with no visible symptom: a control that silently does not narrow."""
        import logging

        from apps.treephenorag_ui import selfcheck

        gold_path = write_gold_csv(tmp_path / "gt.csv", GOLD)
        frame_dir = TestSample.write(tmp_path / "frame", {"pb_worst": ["nobody-1", "nobody-2"]})
        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir=str(report_dir),
            cache_dir=str(tmp_path / "cache"), hcy_frame_dir=str(frame_dir))

        with caplog.at_level(logging.INFO, logger="exp13_ui.selfcheck"):
            assert selfcheck.run(registry) == 1
        assert "describe different report sets" in caplog.text

    def test_a_partially_present_frame_only_warns(self, output_base, shared_ontology,
                                                  tmp_path, report_dir, caplog):
        """Unmerged shards do this, and it is recoverable, a warning, not an exit code."""
        import logging

        from apps.treephenorag_ui import selfcheck

        gold_path = write_gold_csv(tmp_path / "gt.csv", GOLD)
        frame_dir = TestSample.write(tmp_path / "frame", {"pb_worst": ["r1", "nobody-1"]})
        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir=str(report_dir),
            cache_dir=str(tmp_path / "cache"), hcy_frame_dir=str(frame_dir))

        with caplog.at_level(logging.INFO, logger="exp13_ui.selfcheck"):
            assert selfcheck.run(registry) == 0
        assert "are absent from this run" in caplog.text

    def test_a_predictions_nodes_disagreement_fails(self, registry, output_base):
        """The two files come from one traversal result. Disagreement means one is stale."""
        from apps.treephenorag_ui import selfcheck

        path = (output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.1"
                / "tree_gate_lr_predictions.jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if record.get("summary") and record["report_id"] == "r1":
                record["predicted_set"] = [C]          # nodes.jsonl still says {C, D}
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        registry.rescan()
        registry._gold.clear()
        assert selfcheck.run(registry) == 1

    def test_a_tau_covering_fewer_reports_fails(self, registry, output_base):
        from apps.treephenorag_ui import selfcheck

        path = (output_base / "exp13_00_tree_gate_lr" / "hcy" / "tau_0.5"
                / "tree_gate_lr_predictions.jsonl")
        kept = [line for line in path.read_text().splitlines()
                if json.loads(line)["report_id"] == "r1"]
        path.write_text("".join(f"{line}\n" for line in kept))
        registry.rescan()
        assert selfcheck.run(registry) == 1


class TestCuratedGold:
    """The curated HCY dataset: how it is found, what it adds, and what it refuses to guess.

    The dataset is the project's current ground truth, and this app is where its numbers get read. Two
    kinds of failure are worth a test each. A *loud* one, the directory is absent or malformed, must leave the app as it was. That is the same additive contract the sampling frame
    and the PhenoBERT underlay hold to. A *quiet* one, an annotated term drawn on the wrong sentence,
    an excluded annotation reported as evidence for a term that is in the ground truth, produces a screen
    that looks right and is not, which is the failure mode a diagnostic tool cannot afford.
    """

    #: One report, three annotations: an annotated term with a trigger word, a second annotated term whose
    #: shipped segment text does not match the index it names, and a family finding the policy
    #: dropped. Between them they exercise every branch of the placement and evidence logic.
    ANNOTATIONS = [
        {"patient_id": "r1", "hpo_code": C, "hpo_name": "Seizure", "in_gold": 1,
         "source": "prior_annotation", "status": "confirmed", "segment_idx": 0,
         "trigger_word": "seizures", "segment_text": SEGMENTS["r1"][0], "anchored": 1,
         "anchor_how": "curated", "qualifiers": ""},
        {"patient_id": "r1", "hpo_code": F, "hpo_name": "Cerebral atrophy", "in_gold": 1,
         "source": "daphne", "status": "", "segment_idx": 99,
         "trigger_word": "atrophy", "segment_text": SEGMENTS["r1"][2], "anchored": 1,
         "anchor_how": "lexical", "qualifiers": ""},
        {"patient_id": "r1", "hpo_code": D, "hpo_name": "Anxiety", "in_gold": 0,
         "exclude_reason": "family", "source": "prior_annotation", "status": "unruled",
         "segment_idx": 1, "trigger_word": "mother", "segment_text": SEGMENTS["r1"][1],
         "anchored": 1, "anchor_how": "curated", "qualifiers": "family"},
    ]

    @pytest.fixture
    def dataset(self, tmp_path):
        return curated_mod.load(str(write_curated_dataset(
            tmp_path / "hcy", "2026-09-08", self.ANNOTATIONS)))

    # ── discovery ────────────────────────────────────────────────────────────

    def test_the_newest_dated_directory_wins(self, tmp_path):
        hcy = tmp_path / "hcy"
        for date in ("2026-07-01", "2026-09-08", "2026-08-15"):
            write_curated_dataset(hcy, date, self.ANNOTATIONS)
        found = curated_mod.discover(str(hcy))
        assert [os.path.basename(p).rsplit("_", 1)[-1] for p in found] == [
            "2026-09-08", "2026-08-15", "2026-07-01"]
        assert os.path.basename(os.path.dirname(
            curated_mod.gold_path_for(str(hcy)))) == "curated_ground_truth_2026-09-08"

    def test_a_directory_without_the_gold_file_is_not_a_dataset(self, tmp_path):
        """Half-written is worse than absent: it would appear in the dropdown and load to zero."""
        hcy = tmp_path / "hcy"
        (hcy / "curated_ground_truth_2026-09-08").mkdir(parents=True)
        (hcy / "curated_gold_old").mkdir()
        assert curated_mod.discover(str(hcy)) == []
        assert curated_mod.gold_path_for(str(hcy)) == ""

    def test_a_name_that_is_not_a_date_cannot_sort_above_every_real_one(self, tmp_path):
        hcy = tmp_path / "hcy"
        write_curated_dataset(hcy, "2026-09-08", self.ANNOTATIONS)
        stray = hcy / "curated_gold_zzz_latest"
        stray.mkdir()
        (stray / "hcy_ground_truth_curated.csv").write_text("patient_id,hpo_codes\n",
                                                            encoding="utf-8")
        assert [os.path.basename(p) for p in curated_mod.discover(str(hcy))] == [
            "curated_ground_truth_2026-09-08"]

    def test_the_dataset_is_selected_by_the_gold_path_alone(self, tmp_path):
        """One control. The annotations of one build can never sit beside the ground truth of another."""
        hcy = tmp_path / "hcy"
        directory = write_curated_dataset(hcy, "2026-09-08", self.ANNOTATIONS)
        gold_path = str(directory / "hcy_ground_truth_curated.csv")
        assert curated_mod.dataset_dir_of(gold_path) == str(directory)
        assert curated_mod.hcy_dir_of(gold_path) == str(hcy)

        raw = write_gold_csv(hcy / "hcy_ground_truth_raw.csv", GOLD)
        assert curated_mod.dataset_dir_of(str(raw)) == ""
        assert curated_mod.hcy_dir_of(str(raw)) == str(hcy), \
            "the sidebar must still be able to offer the curated datasets beside it"

    def test_absence_is_an_ordinary_state(self, tmp_path):
        assert curated_mod.load("") is None
        assert curated_mod.load(str(tmp_path / "nothing-here")) is None
        empty = tmp_path / "curated_ground_truth_2026-09-08"
        empty.mkdir()
        (empty / "hcy_ground_truth_curated.csv").write_text("patient_id,hpo_codes\n",
                                                            encoding="utf-8")
        assert curated_mod.load(str(empty)) is None, "a gold with no rows is not a dataset"

    def test_the_sidecars_are_optional(self, tmp_path):
        """The ground truth still scores when only the two-column file survived, with no evidence layer."""
        directory = write_curated_dataset(tmp_path / "hcy", "2026-09-08", self.ANNOTATIONS)
        (directory / "hcy_curated_annotations.csv").unlink()
        (directory / "manifest.json").unlink()
        dataset = curated_mod.load(str(directory))
        assert dataset.gold == {"r1": {C, F}}
        assert dataset.for_report("r1") == [] and dataset.find("r1", C) is None
        assert dataset.date == "" and dataset.describe().startswith("curated_ground_truth_2026-09-08")

    # ── the cohort ───────────────────────────────────────────────────────────

    def test_a_report_outside_the_cohort_is_named_rather_than_scored_empty(self, dataset):
        assert dataset.in_cohort("r1") and not dataset.in_cohort("r2")
        assert dataset.outside(["r1", "r2", "r3"]) == ["r2", "r3"]
        assert dataset.cohort_reason("r1") == "prior_annotation"

    def test_the_gold_is_the_applied_policy(self, dataset):
        assert dataset.gold == {"r1": {C, F}}, "the family finding must not be in the gold"
        assert dataset.n_gold_pairs == 2 and dataset.n_excluded == 1

    # ── the evidence ─────────────────────────────────────────────────────────

    def test_an_annotated_term_carries_the_words_it_was_annotated_on(self, dataset):
        row = dataset.find("r1", C)
        assert row.in_gold and row.trigger_word == "seizures" and row.segment_idx == 0
        assert "seizures" in row.describe() and "sentence 0" in row.describe()

    def test_an_excluded_annotation_explains_a_false_positive(self, dataset):
        row = dataset.excluded("r1", D)
        assert row is not None and row.exclude_reason == "family"
        assert "relative" in row.why, "the reason must be spelled out, not left as a slug"
        assert dataset.qualifiers("r1", D) == ("family",)

    def test_a_term_in_the_gold_is_never_reported_as_excluded(self, dataset):
        """A term that is both is a true positive. Naming its dropped duplicate is an accusation."""
        assert dataset.excluded("r1", C) is None
        assert dataset.excluded("r1", "HP:9999999") is None

    def test_a_placement_is_verified_against_the_recovered_segmentation(self, dataset):
        """``segment_idx`` is trusted only where the shipped text agrees with the recovered one.

        The fixture's second annotated term names sentence 99, which does not exist. Its
        ``segment_text`` does, so it lands on sentence 2, the sentence it was actually written on, not being dropped, or drawn at a plausible-looking index.
        """
        placements = dataset.placements("r1", SEGMENTS["r1"])
        assert [row.hpo_code for row in placements[0]] == [C]
        assert [row.hpo_code for row in placements[2]] == [F]
        assert 1 not in placements, "the excluded family finding must not be drawn as gold"
        assert dataset.placement_coverage("r1", SEGMENTS["r1"]) == (2, 2)

    def test_an_annotation_that_matches_nothing_is_left_unplaced(self, tmp_path):
        rows = [dict(self.ANNOTATIONS[0], segment_idx="", segment_text="nothing like this text")]
        dataset = curated_mod.load(str(write_curated_dataset(
            tmp_path / "hcy", "2026-09-08", rows)))
        assert dataset.placements("r1", SEGMENTS["r1"]) == {}
        assert dataset.placement_coverage("r1", SEGMENTS["r1"]) == (0, 1)

    def test_an_empty_segment_index_means_unplaced_not_sentence_zero(self, tmp_path):
        rows = [dict(self.ANNOTATIONS[0], segment_idx="", segment_text="")]
        dataset = curated_mod.load(str(write_curated_dataset(
            tmp_path / "hcy", "2026-09-08", rows)))
        assert dataset.find("r1", C).segment_idx is None
        assert dataset.placements("r1", SEGMENTS["r1"]) == {}

    def test_the_best_row_for_a_term_prefers_gold_then_located(self, tmp_path):
        rows = [
            {"patient_id": "r1", "hpo_code": C, "in_gold": 0, "exclude_reason": "unsure",
             "source": "suggestion", "anchored": 0},
            {"patient_id": "r1", "hpo_code": C, "in_gold": 1, "source": "prior_annotation",
             "segment_idx": 0, "trigger_word": "seizures", "segment_text": SEGMENTS["r1"][0],
             "anchored": 1, "anchor_how": "curated"},
        ]
        dataset = curated_mod.load(str(write_curated_dataset(
            tmp_path / "hcy", "2026-09-08", rows, gold={"r1": [C]})))
        assert dataset.find("r1", C).source == "prior_annotation"
        assert dataset.excluded("r1", C) is None


class TestCuratedGoldInTheUI:
    """The curated ground truth where a reader meets it: the registry, the report and the panels."""

    @pytest.fixture
    def curated_registry(self, output_base, shared_ontology, tmp_path, report_dir):
        """A registry whose HCY ground truth is a curated dataset covering **r1 only**.

        r2 is left out of the cohort. That is the situation the whole coverage story
        exists for, the run wrote two reports, the ground truth covers one, and every rate on screen is
        over that one.
        """
        directory = write_curated_dataset(tmp_path / "hcy", "2026-09-08",
                                          TestCuratedGold.ANNOTATIONS)
        return registry_mod.Registry(
            str(output_base), hcy_gt_path=str(directory / "hcy_ground_truth_curated.csv"),
            hcy_input_dir=str(report_dir), cache_dir=str(tmp_path / "cache"))

    def test_the_registry_scores_against_the_curated_gold_through_the_usual_loader(
            self, curated_registry):
        """No second implementation of the policy: the same ``HCYDataset`` path as any ground truth file."""
        assert curated_registry.gold("hcy") == {"r1": {C, F}}
        assert curated_registry.curated.name == "curated_ground_truth_2026-09-08"
        assert curated_registry.curated_for("gsc") is None, "GSC+ has no curation pass"

    def test_a_plain_gold_file_brings_no_curated_layer(self, registry):
        assert registry.curated is None and registry.curated_for("hcy") is None

    def test_the_report_names_what_the_cohort_left_out(self, curated_registry):
        report = curated_registry.get_report(TREE_CELL, "tau_0.5")
        block = report["curated"]
        assert report["counts"]["n_reports_scored"] == 1
        assert block["n_reports_in_cohort"] == 1 and block["reports_outside"] == ["r2"]
        assert block["curation_log"] == "0123456789abcdef"

    def test_a_false_positive_the_curator_had_already_dropped_is_counted_separately(
            self, curated_registry):
        """``tau_0.5`` predicts D on r1, which the curated policy dropped as a family finding.

        Against the raw ground truth that is a hit. Against this one it is a false positive. Which of the
        two it is is the most consequential difference between the ground truth files, and the scorecard
        has to be able to say so, not showing one more anonymous FP.
        """
        report = curated_registry.get_report(TREE_CELL, "tau_0.5")
        assert report["curated"]["n_excluded_fp"] == 1
        assert report["curated"]["excluded_fp_by_reason"] == {"family": 1}

    def test_the_scorecard_shows_the_cohort_and_the_policy(self, curated_registry):
        from apps.treephenorag_ui.views import scorecard

        report = curated_registry.get_report(TREE_CELL, "tau_0.5")
        rendered = flat(scorecard._build(report, "light"))
        assert "curated_ground_truth_2026-09-08" in rendered
        assert "outside the curated cohort" in rendered, "the dropped report is not named"
        assert "r2" in rendered
        assert "require_evidence" in rendered, "the policy is not on the page"

    def test_the_scorecard_is_unchanged_without_a_curated_gold(self, registry):
        from apps.treephenorag_ui.views import scorecard

        report = registry.get_report(TREE_CELL, "tau_0.5")
        assert report["curated"] is None
        assert "curated_ground_truth_" not in flat(scorecard._build(report, "light"))

    def test_the_provenance_line_names_the_dataset_not_the_filename(self, curated_registry):
        """Every curated ground truth ever built is called ``hcy_ground_truth_curated.csv``.

        Pasting that basename into a findings file names no particular ground truth, so a number carrying
        it is unreproducible. The dated directory is the identity.
        """
        report = curated_registry.get_report(TREE_CELL, "tau_0.5")
        line = copyexport.provenance(report)
        assert "curated_ground_truth_2026-09-08" in line
        assert "outside the curated cohort" in line
        assert "0123456789abcdef" in line

    def test_the_terms_panel_says_which_words_a_miss_was_missed_at(self, curated_registry):
        from apps.treephenorag_ui.views import deepdive

        frame = curated_registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        subset = frame.loc[frame["report_id"] == "r1"]
        rendered = flat(deepdive._terms_panel(subset, set(GOLD["r1"]), set(ACCEPTED["r1"]),
                                              curated_registry.view, "r1",
                                              curated_registry.curated))
        assert "curated evidence" in rendered
        assert "seizures" in rendered, "the trigger word behind the gold term is not shown"
        assert "family" in rendered, "the dropped annotation behind the FP is not explained"

    def test_the_terms_panel_is_unchanged_without_one(self, registry):
        from apps.treephenorag_ui.views import deepdive

        frame = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        subset = frame.loc[frame["report_id"] == "r1"]
        rendered = flat(deepdive._terms_panel(subset, set(GOLD["r1"]), set(ACCEPTED["r1"]),
                                              registry.view, "r1"))
        assert "curated evidence" not in rendered

    def test_the_gold_is_drawn_on_the_sentences_it_was_annotated_on(self, curated_registry):
        from apps.treephenorag_ui.views import deepdive

        report = curated_registry.get_report(TREE_CELL, "tau_0.5")
        panel = flat(deepdive._text_panel(curated_registry, report, "r1", {C, F}, set(), "light",
                                          set(ACCEPTED["r1"])))
        assert "seizures" in panel and "atrophy" in panel
        # C is predicted at tau_0.5, F is not, the marks say which, on the sentence itself.
        assert "✓" in panel and "✗" in panel
        assert "mother" not in panel, "an excluded annotation must not be drawn as gold"

    def test_the_gold_layer_is_absent_without_a_curated_gold(self, registry):
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        panel = flat(deepdive._text_panel(registry, report, "r1", set(GOLD["r1"]), set(), "light"))
        assert "annotated terms (ground truth)" in panel, "the plain chip row was lost"
        assert "✓" not in panel and "✗" not in panel

    def test_the_error_table_carries_the_curated_note_on_both_sides(self, curated_registry):
        """One column, two claims, and only for the rows on screen.

        Filling it for the whole frame would cost two dict lookups per node on a table that pages
        server-side because it can run to millions of rows.
        """
        from apps.treephenorag_ui.views import errors_view

        frame = curated_registry.bundle(TREE_CELL, "tau_0.5", heavy=True)["node_table"]
        dataset = curated_registry.curated

        fp_rows, _, _, _ = errors_view._rows(frame, "FP", None, None, None, 0, dataset)
        dropped = [r["curated_note"] for r in fp_rows if r["hpo_id"] == D]
        assert dropped and dropped[0].startswith("family"), fp_rows

        fn_rows, _, _, _ = errors_view._rows(frame, "FN", None, None, None, 0, dataset)
        missed = [r["curated_note"] for r in fn_rows if r["hpo_id"] == F]
        assert missed and "atrophy" in missed[0], fn_rows

        assert [c["id"] for c in errors_view._columns("FP", dataset)][:4] == [
            "report_id", "hpo_id", "hpo_label", "curated_note"]

    def test_the_error_table_is_unchanged_without_a_curated_gold(self, registry):
        from apps.treephenorag_ui.views import errors_view

        frame = registry.bundle(TREE_CELL, "tau_0.5", heavy=True)["node_table"]
        rows, _, _, tooltips = errors_view._rows(frame, "FP", None, None, None, 0)
        assert rows and all("curated_note" not in row for row in rows)
        assert tooltips == [], "no view was passed, so no tooltip layer is built"
        assert errors_view._columns("FP") == errors_view.FP_COLUMNS

    def test_the_selfcheck_catches_a_gold_file_and_an_annotation_table_from_two_builds(
            self, output_base, shared_ontology, tmp_path, report_dir):
        """The one failure that leaves every number on screen looking plausible."""
        from apps.treephenorag_ui import selfcheck

        directory = write_curated_dataset(
            tmp_path / "hcy", "2026-09-08", TestCuratedGold.ANNOTATIONS,
            gold={"r1": [C, F], "r2": [D]})          # a report the annotation table never mentions
        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(directory / "hcy_ground_truth_curated.csv"),
            hcy_input_dir=str(report_dir), cache_dir=str(tmp_path / "cache"))

        result = selfcheck.Result()
        selfcheck._check_curated(registry, list(registry.cells.values()), result)
        assert any("disagree" in f for f in result.failures), result.failures

    def test_the_selfcheck_passes_on_a_consistent_dataset_and_reports_the_coverage(
            self, curated_registry):
        from apps.treephenorag_ui import selfcheck

        result = selfcheck.Result()
        selfcheck._check_curated(curated_registry, list(curated_registry.cells.values()), result)
        assert not result.failures, result.failures
        assert any("outside the curated cohort" in w for w in result.warnings), result.warnings

class TestTheSegmentedReportsCsv:
    """``segmented_reports.csv`` is the route that works on a login node, and it must be exact.

    Every other route needs either Stanza (which dies there) or a Free Listing generation run (which a cohort may
    not have had). This one is a file ``experiments/03_setup/segment_reports.py`` already wrote with the *same
    three calls* the tree driver makes, so it is not an approximation of the segmentation, it is a
    copy of it, and it is the coordinate system the curated ground truth's ``segment_idx`` indexes.

    Which makes the failure mode the thing to test. A segmentation that is subtly wrong resolves
    every index and shows the wrong sentence for all of them, with nothing on screen to say so.
    """

    @pytest.fixture
    def csv_registry(self, output_base, shared_ontology, tmp_path, report_dir):
        """A registry with the csv on disk and **no Free Listing generation run artifacts**, so the route is forced."""
        for path in (output_base / "phenojury_generation_free_listing" / "hcy").glob("llm_extractions_*"):
            path.unlink()
        gold_path = write_gold_csv(tmp_path / "hcy" / "gt.csv", GOLD)
        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv", SEGMENTS)
        return registry_mod.Registry(
            str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir=str(report_dir),
            cache_dir=str(tmp_path / "cache"))

    def test_the_path_is_derived_from_the_gold_rather_than_configured_twice(self, csv_registry):
        """One control. The annotations of one build can never sit beside another build's
        segmentation, because both are found from the same cohort directory."""
        assert csv_registry.hcy_segments_path.endswith("hcy/segmented_reports.csv")
        assert os.path.isfile(csv_registry.hcy_segments_path)
        assert not [p for p in csv_registry.validate() if "segment" in p]

    def test_it_is_asked_for_by_the_same_name_the_curation_app_uses(self):
        """Fixed, because a curator's segment 4 and this app's sentence 4 must be one sentence."""
        from hpo_extraction.curation import sources

        assert evidence_mod.SEGMENTS_FILE == sources.SEGMENTS_FILE

    def test_the_csv_answers_where_stanza_cannot_and_exp13_06_never_ran(self, csv_registry):
        helper = csv_registry.evidence()
        assert helper.segments("hcy") == SEGMENTS
        assert helper._pipeline is None, "Stanza was loaded despite the csv route succeeding"
        assert "segmented_reports.csv" in helper.source("hcy")
        assert helper.status("hcy") is None

    def test_the_csv_is_preferred_over_the_exp13_06_artifacts(self, registry, tmp_path):
        """Both present: the csv wins, because it is the segmentation, not a copy of one
        that has to be checked against the report before it can be trusted."""
        write_segments_csv(tmp_path / "segmented_reports.csv", SEGMENTS)
        helper = evidence_mod.Evidence(registry)
        assert helper.segments("hcy") == SEGMENTS
        assert "segmented_reports.csv" in helper.source("hcy")
        assert "exp13_06" not in helper.source("hcy")

    def test_a_csv_describing_a_different_document_is_rejected_not_shown(
            self, csv_registry, tmp_path):
        """The quiet failure: every index resolves and every sentence shown is the wrong one."""
        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv",
                           {"r1": ["Nothing like this is in the report at all."],
                            "r2": SEGMENTS["r2"]})
        helper = evidence_mod.Evidence(csv_registry)
        recovered = helper.segments("hcy")
        assert "r1" not in recovered and recovered["r2"] == SEGMENTS["r2"]
        assert "rejected" in helper.source("hcy")

    def test_whitespace_only_differences_are_accepted_rather_than_rejected(self, csv_registry,
                                                                          tmp_path):
        """Stanza normalises whitespace, so an exact-substring check rejects good segmentations.

        ``verify_segmentation`` alone would drop this report. The report is fine and the sentences
        are the right ones. That is why :func:`aligns_to_text` falls through to the same
        whitespace-insensitive alignment the curation app places every annotation by.
        """
        spaced = {"r1": [s.replace(" ", "  ") for s in SEGMENTS["r1"]], "r2": SEGMENTS["r2"]}
        assert not evidence_mod.verify_segmentation(spaced["r1"], REPORT_TEXT["r1"])
        assert evidence_mod.aligns_to_text(spaced["r1"], REPORT_TEXT["r1"])

        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv", spaced)
        assert "r1" in evidence_mod.Evidence(csv_registry).segments("hcy")

    def test_it_is_served_unverified_when_the_reports_are_not_reachable(
            self, output_base, shared_ontology, tmp_path):
        """An unset report folder used to be fatal to the whole panel, ``segments`` returned ``{}``
        before any route ran. A deep-dive with sentences beats one without. What it must not do is
        claim they were checked."""
        gold_path = write_gold_csv(tmp_path / "hcy" / "gt.csv", GOLD)
        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv", SEGMENTS)
        registry = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir="",
            cache_dir=str(tmp_path / "cache"))

        helper = registry.evidence()
        assert helper.segments("hcy") == SEGMENTS
        assert "UNVERIFIED" in helper.source("hcy")
        # Nothing cached: the cache key is a digest of the texts, and there are none.
        assert not list((tmp_path / "cache").glob("segments.*.json"))

    def test_an_unreadable_csv_leaves_the_other_routes_alone(self, registry, tmp_path):
        (tmp_path / "segmented_reports.csv").write_text("not,a,segmentation\n", encoding="utf-8")
        helper = evidence_mod.Evidence(registry)
        assert helper.segments("hcy") == SEGMENTS, "the artifact route should still have answered"
        assert "Free Listing run" in helper.source("hcy")

    def test_a_missing_csv_is_named_rather_than_silently_degrading(self, registry, tmp_path):
        """It is the difference between a report panel and a wall of indices, so the sidebar says
        so before the reader opens one."""
        problems = " ".join(registry.validate())
        assert "segmented_reports.csv" in problems and "segment_reports.py" in problems

    def test_the_status_names_a_report_the_cohort_map_does_not_hold(self, csv_registry, tmp_path):
        """Partial recovery used to interpolate the literal string "None" into the note explaining
        why the text was missing."""
        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv", {"r2": SEGMENTS["r2"]})
        helper = evidence_mod.Evidence(csv_registry)
        assert helper.status("hcy") is None, "the cohort as a whole did resolve"
        assert "r1" in (helper.status("hcy", "r1") or ""), helper.status("hcy", "r1")
        assert helper.status("hcy", "r2") is None

    def test_a_nan_sentence_index_does_not_take_the_callback_down(self, csv_registry):
        """``calls["sent_index"].tolist()`` yields ``nan`` for a record missing the field."""
        coverage = csv_registry.evidence().coverage("hcy", "r1", [0, float("nan"), None, 2])
        assert coverage["n_referenced"] == 2 and coverage["n_resolved"] == 2

    def test_the_report_text_is_decoded_as_inference_decoded_it(self, tmp_path):
        """Every HCY report is German. Reading them as utf-8 while inference read latin1 gave a
        different string for every report with an umlaut, and since a recovered segmentation is
        verified by locating its sentences in this text, every one of them was silently dropped."""
        directory = tmp_path / "latin1"
        directory.mkdir()
        text = "Die Mutter berichtet über Anfälle."
        (directory / "r9.txt").write_bytes(text.encode("latin1"))
        (directory / "notes.md").write_text("ignored", encoding="utf-8")
        (directory / "HP_0001250.txt").write_text("ctx", encoding="latin1")

        texts = loaders.load_report_texts(str(directory))
        assert texts["r9"] == text, "the encoding does not match core.data_loading.load_txt"
        assert "notes" not in texts, "only .txt files are reports"
        assert "HP:0001250" in texts, "the underscore-to-colon keying was dropped"
        loaders.assert_matches_load_txt(str(directory))


class TestTheReportReadsAsADocument:
    """The deep-dive's report panel, once a segmentation is available.

    What is being checked is not that the text appears, it is that the *layers* over it cannot lie:
    a sentence is named by the index the call records use, an annotated term is drawn only on words a
    curator actually wrote down, and anything that could not be placed is still listed, not
    dropped.
    """

    @pytest.fixture
    def registry(self, output_base, shared_ontology, tmp_path, report_dir):
        directory = write_curated_dataset(tmp_path / "hcy", "2026-09-08",
                                          TestCuratedGold.ANNOTATIONS)
        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv", SEGMENTS)
        return registry_mod.Registry(
            str(output_base), hcy_gt_path=str(directory / "hcy_ground_truth_curated.csv"),
            hcy_input_dir=str(report_dir), cache_dir=str(tmp_path / "cache"))

    def _panel(self, registry, highlighted=frozenset(), predicted=None):
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        return deepdive._text_panel(registry, report, "r1", {C, F}, set(highlighted), "light",
                                    predicted)

    def test_the_report_keeps_its_own_line_breaks(self, registry):
        """The prose is the document, not a list of sentences. The text *between* segments, the
        newlines that make a clinical report readable, is put back from the alignment."""
        rendered = flat(self._panel(registry))
        assert "prose" in rendered
        for sentence in SEGMENTS["r1"]:
            assert sentence in rendered
        assert "\\n" in rendered, "the report's own line breaks were dropped"

    def test_every_sentence_is_hoverable_and_names_its_index(self, registry):
        rendered = flat(self._panel(registry, highlighted={1}))
        for index in range(len(SEGMENTS["r1"])):
            assert f"sentence #{index}" in rendered
        assert "seg-retrieved" in rendered and "retrieved for the selected node" in rendered

    def test_the_gold_is_drawn_on_the_words_it_was_annotated_at(self, registry):
        """C's trigger is "seizures" in sentence 0, and that is where it is drawn, green, because
        this configuration predicted it."""
        rendered = flat(self._panel(registry, predicted={C}))
        assert "gold-mark" in rendered
        assert "found by this method" in rendered and "MISSED by this method" in rendered
        # The tag names the term the way the ontology names it, not the way the annotation file
        # spells it, so a term reads identically here and in the node table above.
        assert f"{NAMES[F]} ✗" in rendered, "F is annotated and not predicted at tau_0.5"
        assert f"{NAMES[C]} ✓" in rendered

    def test_an_annotated_term_carries_the_ontology_on_hover_like_every_other_term(self, registry):
        """Until now the terms drawn on the report were the one HPO surface on the page with no
        definition, the one place a reader is deciding what the ontology means by a word.

        The toy ontology ships an empty ``Def``, so what is checked here is that the mark carries
        ``terms.tooltip``, which is the function the definition travels in. The wording itself is
        fixed in :class:`TestTermTooltips`."""
        rendered = flat(self._panel(registry, predicted={C}))
        assert f"curated ground truth → {C} · {NAMES[C]}" in rendered
        assert "annotated at" in rendered, "the curation provenance was dropped"

    def test_an_alt_id_in_the_gold_is_not_reported_as_missed(self, registry, shared_ontology):
        """A ground truth file predates the release it is scored against. Comparing the raw strings drew a
        term in red on the report while the scorecard above counted it as a true positive."""
        from apps.treephenorag_ui import terms

        view = shared_ontology[0] if isinstance(shared_ontology, tuple) else shared_ontology
        assert terms.same_term(view, C, C)
        assert not terms.same_term(view, C, D)
        assert not terms.same_term(view, C, "")

    def test_a_term_that_cannot_be_drawn_is_still_listed_with_why(self, registry, tmp_path):
        """Nothing recorded goes unshown. A trigger word absent from its own segment is not drawn
        on any words, searching the whole report for it would claim a position nobody wrote
        down, so it keeps a chip, and the chip says which of the two residues it is."""
        write_curated_dataset(
            tmp_path / "hcy", "2026-09-08",
            [{**TestCuratedGold.ANNOTATIONS[0], "trigger_word": "nowhere in this sentence"},
             *TestCuratedGold.ANNOTATIONS[1:]])
        registry.rescan()
        rendered = flat(self._panel(registry, predicted={C}))
        assert "placed on a sentence, but not on any words" in rendered
        assert C in rendered

    def test_the_panel_is_unchanged_without_a_curated_gold(self, output_base, shared_ontology,
                                                           tmp_path, report_dir):
        """The additive contract: a plain ground truth file renders the chip row it always did."""
        gold_path = write_gold_csv(tmp_path / "hcy" / "gt.csv", GOLD)
        write_segments_csv(tmp_path / "hcy" / "segmented_reports.csv", SEGMENTS)
        plain = registry_mod.Registry(
            str(output_base), hcy_gt_path=str(gold_path), hcy_input_dir=str(report_dir),
            cache_dir=str(tmp_path / "cache"))
        rendered = flat(self._panel(plain, predicted={C}))
        assert "annotated terms (ground truth)" in rendered
        assert "gold-mark" not in rendered, "there are no offsets in a plain gold to draw on"

    def test_the_calls_panel_shows_the_sentence_and_links_back_to_it(self, registry):
        from apps.treephenorag_ui.views import deepdive

        report = registry.get_report(TREE_CELL, "tau_0.5")
        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        rendered = flat(deepdive._calls_panel(registry, bundle, TREE_CELL, report, "r1", C))

        assert "sentence text unavailable" not in rendered
        assert SEGMENTS["r1"][0] in rendered, "the retrieved sentence is not on screen"
        assert "dd-call-sent" in rendered, "the badge does not point back at the report"
        assert "prompt needs the sentence text" not in rendered

    def test_the_badge_is_inert_when_the_sentence_did_not_resolve(self, registry):
        """An inert click reads as a broken control, not as missing data."""
        from apps.treephenorag_ui.views import deepdive

        registry.evidence()._segments["hcy"] = {}
        report = registry.get_report(TREE_CELL, "tau_0.5")
        bundle = registry.bundle(TREE_CELL, "tau_0.5")
        rendered = flat(deepdive._calls_panel(registry, bundle, TREE_CELL, report, "r1", C))
        assert "dd-call-sent" not in rendered and "sent #0" in rendered

    def test_every_term_surface_hovers_to_a_definition(self, registry):
        """One table-driven check over the surfaces, because the failure is per-surface and silent:
        a term with no tooltip looks like one with a tooltip nobody hovered."""
        from apps.treephenorag_ui.views import deepdive

        subset = registry.bundle(TREE_CELL, "tau_0.5")["node_table"]
        subset = subset.loc[subset["report_id"] == "r1"]
        view = registry.view
        #: What ``terms.tooltip`` leads with. On the real ontology the definition follows it. Here
        #: it is the marker that the tooltip is attached at all, which is the per-surface failure.
        hover = f"{C} · {NAMES[C]}"

        surfaces = {
            "the report panel": flat(self._panel(registry, predicted={C})),
            "terms in play": flat(deepdive._terms_panel(subset, {C, F}, {C, D}, view, "r1")),
            "the node picker": flat(deepdive._node_options(subset, "terms", view)),
        }
        for where, rendered in surfaces.items():
            assert hover in rendered, f"{where} carries no tooltip for {C}"
        assert "has-def" in surfaces["terms in play"], "no affordance on the terms table"
        # The two figures cannot carry a native title, so the definition rides in customdata.
        assert deepdive._hover_def(view, C) == "", "the toy ontology has no Def to wrap"
        assert deepdive._walk_figure(subset, "light", view).to_dict()


class TestTermTooltips:
    """The one wording every mention of a phenotype hovers to.

    Kept against a hand-built entry, not the toy ontology, which ships an empty ``Def`` on
    purpose (the "no definition is recorded" fallback is exercised by the node panel's test). What
    counts here is the order, id and name first, then the definition, then the synonyms, because
    a native tooltip is read top-down and truncated by the browser, not by us.
    """

    class _Stub:
        """The two methods :mod:`apps.treephenorag_ui.terms` needs off an ``OntologyView``."""

        def __init__(self, data, alt=None):
            self.tree = type("T", (), {"data": data})()
            self._alt = alt or {}

        def resolve(self, code):
            return self._alt.get(code, code)

    @pytest.fixture
    def view(self):
        return self._Stub(
            {C: {"Name": ["Seizure"], "Def": ["A sudden abnormal electrical discharge."],
                 "Synonym": ["Epileptic fit", "Convulsion"]},
             F: {"Name": ["Cerebral atrophy"], "Def": [], "Synonym": []}},
            alt={OBSOLETE_C: C})

    def test_the_tooltip_reads_id_name_definition_synonyms(self, view):
        from apps.treephenorag_ui import terms

        assert terms.tooltip(view, C) == (
            f"{C} · Seizure\n\n"
            "A sudden abnormal electrical discharge.\n\n"
            "Also known as: Epileptic fit, Convulsion")

    def test_a_term_with_no_definition_still_names_itself(self, view):
        from apps.treephenorag_ui import terms

        assert terms.tooltip(view, F) == f"{F} · Cerebral atrophy"

    def test_an_unknown_code_says_so_rather_than_showing_a_bare_id_as_a_label(self, view):
        from apps.treephenorag_ui import terms

        assert terms.tooltip(view, HALLUCINATION) == f"{HALLUCINATION}, not in this HPO release"
        assert not terms.known(view, HALLUCINATION)

    def test_a_prefix_gives_the_caller_its_own_context(self, view):
        from apps.treephenorag_ui import terms

        assert terms.tooltip(view, C, prefix="curated gold →").startswith(f"curated gold → {C}")

    def test_the_error_table_carries_the_definition_for_the_page_it_shows(self, registry):
        """A row there names a term and says it was wrong without saying what the term *is*.

        Built for the twenty-five rows on screen and no others: a tree cell's node table runs to
        millions of rows, and an ontology lookup over all of them is what this table pages
        server-side to avoid.
        """
        from apps.treephenorag_ui.views import errors_view

        frame = registry.bundle(TREE_CELL, "tau_0.5", heavy=True)["node_table"]
        rows, _, _, tooltips = errors_view._rows(frame, "FP", None, None, None, 0,
                                                 None, registry.view)
        assert len(tooltips) == len(rows), "the tooltip layer and the rows disagree"
        assert all("hpo_id" in t and "hpo_label" in t for t in tooltips)
        assert NAMES[D] in tooltips[0]["hpo_label"]["value"]

    def test_two_ontologies_do_not_share_a_cache_entry(self):
        """``id()`` is recycled the moment an object is collected, so a cache keyed on it alone
        answers one ontology's question with another's labels once the first has been dropped."""
        from apps.treephenorag_ui import terms

        first = self._Stub({C: {"Name": ["First"], "Def": [], "Synonym": []}})
        assert terms.label(first, C) == "First"
        second = self._Stub({C: {"Name": ["Second"], "Def": [], "Synonym": []}})
        assert terms.label(second, C) == "Second"
        assert terms.label(first, C) == "First", "the two views share a cache"

    def test_an_alt_id_resolves_to_the_same_term(self, view):
        from apps.treephenorag_ui import terms

        assert terms.label(view, OBSOLETE_C) == "Seizure"
        assert terms.same_term(view, OBSOLETE_C, C), \
            "a gold code under an alt id would be drawn as MISSED"
        assert terms.resolve_all(view, {OBSOLETE_C, F}) == {C, F}

    def test_the_definition_is_wrapped_and_bounded_for_a_figure_hover(self, view):
        from apps.treephenorag_ui.views import deepdive

        long = self._Stub({C: {"Name": ["Seizure"], "Def": ["word " * 200], "Synonym": []}})
        fragment = deepdive._hover_def(long, C)
        assert fragment.startswith("<br><br>") and fragment.endswith("…")
        assert max(len(line) for line in fragment.split("<br>")) <= 64
