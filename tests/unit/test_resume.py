"""Unit tests for the crash-safety machinery the long earlier cluster runs depend on.

These cover the two ways a 48 h job used to lose everything it had computed:

* **:mod:`hpo_extraction.utils.resume`**, a report is only "done" once its records are flushed, a resumed
  session appends instead of truncating, and a signal (or a spent time budget) is visible to the
  driver as a flag it can act on between reports.
* **:mod:`hpo_extraction.evaluation.merge_shards`**, the shards of a SLURM array (and the duplicate report block
  a hard kill leaves behind) must consolidate into one clean record set, idempotently.

No GPU, no models, no ontology: this is all file and flag logic.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time

import pytest

from hpo_extraction.utils.resume import Checkpoint, GracefulStop, flush_all, open_jsonl
from hpo_extraction.evaluation.merge_shards import merge_jsonl, merge_run_dir, unsharded_name

pytestmark = pytest.mark.unit


# ── Resume file ────────────────────────────────────────────────────────────────

def test_fresh_run_truncates_and_records(tmp_path):
    ckpt = Checkpoint(str(tmp_path), "variant_s0of2", resume=True)
    assert ckpt.done == set()
    assert ckpt.resumed is False
    assert ckpt.file_mode == "w"      # nothing to keep → outputs start clean

    ckpt.mark("report_a", n_nodes=3)
    ckpt.mark("report_b", n_nodes=5)
    ckpt.close()

    lines = (tmp_path / ".checkpoint_variant_s0of2.jsonl").read_text().splitlines()
    assert [json.loads(x)["unit_id"] for x in lines] == ["report_a", "report_b"]
    assert json.loads(lines[0])["n_nodes"] == 3


def test_resume_skips_done_and_appends(tmp_path):
    first = Checkpoint(str(tmp_path), "v", resume=True)
    first.mark("r1")
    first.mark("r2")
    first.close()

    second = Checkpoint(str(tmp_path), "v", resume=True)
    assert second.done == {"r1", "r2"}
    assert second.resumed is True
    assert second.file_mode == "a"     # outputs must be appended, never rewritten
    assert second.pending(["r1", "r2", "r3", "r4"]) == ["r3", "r4"]
    assert second.is_done("r1") and not second.is_done("r3")
    second.mark("r3")
    second.close()

    assert Checkpoint(str(tmp_path), "v").n_done == 3


def test_resume_false_starts_over(tmp_path):
    first = Checkpoint(str(tmp_path), "v")
    first.mark("r1")
    first.close()

    fresh = Checkpoint(str(tmp_path), "v", resume=False)
    assert fresh.done == set()
    assert fresh.file_mode == "w"
    fresh.close()
    assert (tmp_path / ".checkpoint_v.jsonl").read_text() == ""


def test_truncated_resume_state_line_is_ignored(tmp_path):
    """A hard kill can cut the last line in half. The rest of the progress must still count."""
    p = tmp_path / ".checkpoint_v.jsonl"
    p.write_text('{"unit_id": "r1", "ts": 1.0}\n{"unit_id": "r2", "ts":\n')
    ckpt = Checkpoint(str(tmp_path), "v", resume=True)
    assert ckpt.done == {"r1"}
    ckpt.close()


def test_shard_resume_states_are_independent(tmp_path):
    a = Checkpoint(str(tmp_path), "v_s0of8")
    b = Checkpoint(str(tmp_path), "v_s1of8")
    a.mark("r0")
    b.mark("r1")
    a.close()
    b.close()
    assert Checkpoint(str(tmp_path), "v_s0of8").done == {"r0"}
    assert Checkpoint(str(tmp_path), "v_s1of8").done == {"r1"}


def test_open_jsonl_terminates_a_partial_line_before_appending(tmp_path):
    """A killed writer can leave a line with no newline. Appending must not glue onto it."""
    p = tmp_path / "out.jsonl"
    p.write_text('{"report_id": "r1"}\n{"report_id": "r1", "hpo')   # killed mid-record
    with open_jsonl(str(p), "a") as f:
        f.write(json.dumps({"report_id": "r1", "hpo_id": "HP:1"}) + "\n")

    lines = p.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["hpo_id"] == "HP:1"   # The resumed record survived intact
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[1])                          # The stump stays behind, alone


# ── GracefulStop ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"),
    reason="SIGUSR1 is POSIX-only; the SLURM pre-emption path this asserts exists only on the "
           "cluster. GracefulStop's time-budget path (below) covers the same flag on any platform.",
)
def test_signal_sets_the_stop_flag():
    stop = GracefulStop()
    assert stop.should_stop() is False
    os.kill(os.getpid(), signal.SIGUSR1)
    assert stop.should_stop() is True
    assert "SIGUSR1" in stop.reason
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)


def test_time_budget_triggers_without_a_signal():
    stop = GracefulStop(max_runtime_s=0.01, signals=())
    assert stop.should_stop() is False
    time.sleep(0.02)
    assert stop.should_stop() is True
    assert "max_runtime_s" in stop.reason


def test_no_budget_never_stops_on_its_own():
    assert GracefulStop(max_runtime_s=None, signals=()).should_stop() is False


def test_flush_all_tolerates_closed_and_none(tmp_path):
    open_f = open(tmp_path / "a.txt", "w")
    closed_f = open(tmp_path / "b.txt", "w")
    closed_f.close()
    open_f.write("x")
    flush_all([open_f, closed_f, None])       # must not raise
    assert (tmp_path / "a.txt").read_text() == "x"   # flushed without closing
    open_f.close()


# ── merge_shards ──────────────────────────────────────────────────────────────

def test_unsharded_name():
    assert unsharded_name("tree_gate_lr_s3of8_nodes.jsonl") == "tree_gate_lr_nodes.jsonl"
    assert unsharded_name("tree_gate_lr_s12of16_calls.jsonl") == "tree_gate_lr_calls.jsonl"
    assert unsharded_name("tree_gate_lr_nodes.jsonl") == "tree_gate_lr_nodes.jsonl"


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_merge_drops_duplicate_records_of_a_redone_report(tmp_path):
    """A killed session leaves a partial block. The resumed session re-writes the report in full,
    byte-identically (greedy decoding), and the merge must keep one copy of each record."""
    partial_then_complete = tmp_path / "v_s0of2_nodes.jsonl"
    _write_jsonl(partial_then_complete, [
        {"report_id": "r1", "hpo_id": "HP:1"},          # partial block, killed here
        {"report_id": "r1", "hpo_id": "HP:1"},          # redone on resume …
        {"report_id": "r1", "hpo_id": "HP:2"},          # … and completed
        {"report_id": "r3", "hpo_id": "HP:9"},
    ])
    other = tmp_path / "v_s1of2_nodes.jsonl"
    _write_jsonl(other, [{"report_id": "r2", "hpo_id": "HP:3"}])

    out = tmp_path / "v_nodes.jsonl"
    n_in, n_out = merge_jsonl([str(partial_then_complete), str(other)], str(out))

    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert n_in == 5 and n_out == 4
    by_report = {}
    for r in recs:
        by_report.setdefault(r["report_id"], []).append(r["hpo_id"])
    assert by_report == {"r1": ["HP:1", "HP:2"], "r2": ["HP:3"], "r3": ["HP:9"]}


def test_merge_run_dir_handles_tau_subdirs_and_is_idempotent(tmp_path):
    run = tmp_path / "run"
    (run / "tau_0.5").mkdir(parents=True)
    _write_jsonl(run / "v_s0of2_calls.jsonl", [{"report_id": "r1", "hpo_id": "HP:1"}])
    _write_jsonl(run / "v_s1of2_calls.jsonl", [{"report_id": "r2", "hpo_id": "HP:2"}])
    _write_jsonl(run / "tau_0.5" / "v_s0of2_predictions.jsonl", [{"report_id": "r1", "prediction": 1}])
    _write_jsonl(run / "tau_0.5" / "v_s1of2_predictions.jsonl", [{"report_id": "r2", "prediction": 1}])
    # node metadata has no report_id, de-duplicated on hpo_id instead
    _write_jsonl(run / "v_s0of2_node_metadata.jsonl", [{"hpo_id": "HP:1"}, {"hpo_id": "HP:2"}])
    _write_jsonl(run / "v_s1of2_node_metadata.jsonl", [{"hpo_id": "HP:2"}, {"hpo_id": "HP:3"}])

    merge_run_dir(str(run), verbose=False)
    calls = (run / "v_calls.jsonl").read_text().splitlines()
    preds = (run / "tau_0.5" / "v_predictions.jsonl").read_text().splitlines()
    meta = [json.loads(x)["hpo_id"] for x in (run / "v_node_metadata.jsonl").read_text().splitlines()]
    assert len(calls) == 2 and len(preds) == 2
    assert meta == ["HP:1", "HP:2", "HP:3"]           # union, no duplicates

    merge_run_dir(str(run), verbose=False)            # re-running must not change anything
    assert (run / "v_calls.jsonl").read_text().splitlines() == calls
    assert [json.loads(x)["hpo_id"]
            for x in (run / "v_node_metadata.jsonl").read_text().splitlines()] == meta


def test_merge_run_dir_leaves_a_clean_unsharded_run_alone(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(run / "v_calls.jsonl", [{"report_id": "r1"}])
    assert merge_run_dir(str(run), verbose=False) == []


def test_merge_run_dir_cleans_a_resumed_unsharded_run(tmp_path):
    """No array, but a resumed worker: the re-done report's duplicates must still be removed."""
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(run / "v_nodes.jsonl", [
        {"report_id": "r1", "hpo_id": "HP:1"},
        {"report_id": "r1", "hpo_id": "HP:1"},
        {"report_id": "r1", "hpo_id": "HP:2"},
    ])
    changed = merge_run_dir(str(run), verbose=False)
    assert len(changed) == 1
    assert len((run / "v_nodes.jsonl").read_text().splitlines()) == 2


def test_merge_run_dir_ignores_a_superseded_unsharded_run(tmp_path):
    """A leftover whole-run artifact next to fresh shards is output, not input.

    ``unsharded_name`` is the identity on an already-unsharded name, so the stale file lands in the
    same group as the shards that replace it. Its records share report_ids with the new ones but
    differ line-for-line, the case the exact-line dedup cannot catch, so merging it in
    would show every report twice (this is what happened to an earlier exploratory run's tau_0.5 timing: 135 reports
    over 222 lines).
    """
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(run / "v_timing.jsonl", [{"report_id": "r1", "n_slm_calls": 565}])  # superseded
    _write_jsonl(run / "v_s0of2_timing.jsonl", [{"report_id": "r1", "n_slm_calls": 490}])
    _write_jsonl(run / "v_s1of2_timing.jsonl", [{"report_id": "r2", "n_slm_calls": 300}])

    merge_run_dir(str(run), verbose=False)
    recs = [json.loads(x) for x in (run / "v_timing.jsonl").read_text().splitlines()]
    assert [(r["report_id"], r["n_slm_calls"]) for r in recs] == [("r1", 490), ("r2", 300)]


# ── end-to-end: the crash-safety contract with the real tree writers ──────────

def test_killed_report_is_repaired_by_resume_and_merge(tmp_path):
    """Session 1 dies mid-report. Session 2 resumes. The merge leaves one clean record set.

    This exercises the real ``_write_tau`` writer, ``open_jsonl`` append handling and the
    resume file together, the sequence that lost 48 h of an earlier exploratory run to a wall-clock kill.
    """
    from hpo_extraction.utils.resume import Checkpoint, open_jsonl
    from hpo_extraction.treephenorag.score_store import _write_tau
    from hpo_extraction.treephenorag.traversal import NodeVisit, TraversalResult

    def fake_result(nodes):
        visits = {h: NodeVisit(h, 1, 0.9, 0.9, True, True, 5) for h in nodes}
        return TraversalResult(order=list(nodes), visits=visits, calls=[], accepted=set(nodes),
                               total_frontier_insertions=len(nodes), n_slm_calls=5 * len(nodes))

    tau_dir = tmp_path / "tau_0.5"
    tau_dir.mkdir()

    def open_set(mode):
        return {k: open_jsonl(str(tau_dir / f"v_{name}.jsonl"), mode) for k, name in
                (("nodes", "nodes"), ("pred", "predictions"),
                 ("trav", "traversal"), ("time", "timing"))}

    # ── session 1: r1 completes and is recorded. R2 is written but killed before the mark
    ckpt = Checkpoint(str(tmp_path), "v", resume=True)
    files = open_set(ckpt.file_mode)
    _write_tau(files, "v", "r1", fake_result(["HP:1", "HP:2"]), {"HP:1"})
    for fh in files.values():
        fh.flush()
    ckpt.mark("r1")
    _write_tau(files, "v", "r2", fake_result(["HP:3"]), set())
    for fh in files.values():
        fh.flush()
    # …killed here: r2's records are on disk but unmarked, and the last line is cut in half
    for fh in files.values():
        fh.close()
    ckpt.close()
    nodes_path = tau_dir / "v_nodes.jsonl"
    nodes_path.write_text(nodes_path.read_text()[:-8])

    # ── session 2: r1 is skipped, r2 is redone in full
    ckpt2 = Checkpoint(str(tmp_path), "v", resume=True)
    assert ckpt2.pending(["r1", "r2"]) == ["r2"]
    files2 = open_set(ckpt2.file_mode)
    _write_tau(files2, "v", "r2", fake_result(["HP:3"]), set())
    for fh in files2.values():
        fh.flush()
    ckpt2.mark("r2")
    for fh in files2.values():
        fh.close()
    ckpt2.close()

    merge_run_dir(str(tmp_path), verbose=False)

    nodes = [json.loads(x) for x in nodes_path.read_text().splitlines()]
    assert [(n["report_id"], n["hpo_id"]) for n in nodes] == [
        ("r1", "HP:1"), ("r1", "HP:2"), ("r2", "HP:3"),
    ]
    # one traversal/timing line per report, despite r2 having been written twice
    for name in ("traversal", "timing"):
        recs = [json.loads(x) for x in (tau_dir / f"v_{name}.jsonl").read_text().splitlines()]
        assert [r["report_id"] for r in recs] == ["r1", "r2"]


# ── end-to-end: the Free Listing generation run extraction resumes instead of regenerating ───────────

class _StopAfter:
    """Stop request that fires once ``n`` units are done."""

    def __init__(self, n):
        self.n = n
        self.calls = 0
        self.reason = "test stop"

    def should_stop(self):
        self.calls += 1
        return self.calls >= self.n


class _FakeSLM:
    supports_batching = True

    def __init__(self):
        self.prompts_seen = []

    def generate_many(self, _system, prompts, _max_new_tokens):
        self.prompts_seen.extend(prompts)
        return [f"reply to {p[-6:]}" for p in prompts]

    def unload(self):
        pass


def test_extraction_resumes_from_partial_cache(monkeypatch, tmp_path):
    """A killed extract task must continue where it stopped, not regenerate the cohort."""
    from omegaconf import OmegaConf

    import hpo_extraction.phenojury.generation as se

    fakes = []

    def fake_load_slm(*_a, **_kw):
        fakes.append(_FakeSLM())
        return fakes[-1]

    monkeypatch.setattr(se, "load_slm", fake_load_slm)

    cfg = OmegaConf.create({"reuse_extractions": True, "gen_batch_size": 8, "max_new_tokens": 32})
    sent_dict = {"r1": ["sentence one", "sentence two"], "r2": ["sentence three"]}
    report_ids = ["r1", "r2"]
    args = (cfg, "phi4", "/models/phi4", sent_dict, report_ids, str(tmp_path))
    logger = logging.getLogger("test")

    # session 1, stops after the first report
    records, complete = se._run_extraction(*args, logger, stop=_StopAfter(1))
    assert complete is False
    assert [r["patient_id"] for r in records] == ["r1", "r1"]
    assert len(fakes[0].prompts_seen) == 2

    path = tmp_path / "llm_extractions_phi4.jsonl"
    assert len(path.read_text().splitlines()) == 2   # flushed, not held in memory

    # session 2, regenerates nothing already on disk
    records, complete = se._run_extraction(*args, logger, stop=_StopAfter(99))
    assert complete is True
    assert [(r["patient_id"], r["sentence_number"]) for r in records] == [
        ("r1", 0), ("r1", 1), ("r2", 0),
    ]
    assert len(fakes[1].prompts_seen) == 1          # only the one missing sentence
    assert len(path.read_text().splitlines()) == 3

    # session 3, nothing to do at all, so the model is never loaded
    records, complete = se._run_extraction(*args, logger, stop=_StopAfter(99))
    assert complete is True and len(records) == 3
    assert len(fakes) == 2


def test_cached_records_survive_a_tear_in_the_middle(tmp_path):
    """A torn line mid-file must cost one sentence, not every record written after it.

    Real case: hcy ``llm_extractions_deepseek.jsonl`` was SIGKILLed at the 12 h wall mid-write,
    resubmitted, and appended 170 more records, so the damage sat at line 2551 of 2721. Stopping
    at the first bad line would have silently thrown that tail away and regenerated it on GPU.
    """
    import hpo_extraction.phenojury.generation as se

    path = tmp_path / "llm_extractions_deepseek.jsonl"
    good = [{"patient_id": "r1", "sentence_number": i, "raw": f"out {i}"} for i in range(4)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(good[0]) + "\n")
        f.write(json.dumps(good[1]) + "\n")
        f.write('{"patient_id": "r1", "sentence_num')      # killed here
        f.write("\n")
        f.write(json.dumps(good[2]) + "\n")                 # appended by the resubmission
        f.write(json.dumps(good[3]) + "\n")

    records = se._load_cached_records(str(path), logging.getLogger("test"))
    assert [r["sentence_number"] for r in records] == [0, 1, 2, 3]


def test_read_detections_tolerates_a_torn_line(tmp_path):
    """The same tear in a detections dump drops its row, not the aggregation."""
    import hpo_extraction.phenojury.generation as se

    path = tmp_path / "detections_phi4.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"report_id": "r1", "sentence_number": 0,
                            "hpo_id": "HP:0001250", "count": 2}) + "\n")
        f.write('{"report_id": "r1", "sentence_numb' + "\n")
        f.write(json.dumps({"report_id": "r2", "sentence_number": 1,
                            "hpo_id": "HP:0002160", "count": 1}) + "\n")

    out = se._read_detections("phi4", str(tmp_path))
    assert out == {"r1": {0: {"HP:0001250": 2}}, "r2": {1: {"HP:0002160": 1}}}
