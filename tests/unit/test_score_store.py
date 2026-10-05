"""Unit tests for the earlier runs TreePhenoRAG traversal components.

Covers the pure logic that would silently corrupt a run if wrong, none of which needs a GPU,
model weights, or the real ontology:

* **pruning gates**, the LR gate must reproduce the committed spec arithmetic (including
  the missing-depth branch), and noisyOR / accept-confidence must match hand-computed values;
* **UnionScorer traversal methods**, the lazy column-max, own-column, and per-HPO reductions the
  live traversal and the flat baseline rely on;
* **traverse**, BFS order, DAG deduplication, the two independent thresholds, the depth table,
  frontier-insertion accounting, the node cap, and the tau-sweep monotonicity + memoisation the
  sweep's correctness rests on;
* **GSC loader**, parsing and the WSL/macOS junk-file filter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hpo_extraction.retrieval.descendant_closure import UnionScorer
from hpo_extraction.treephenorag.pooling import (
    LRGate,
    accept_confidence,
    lse_beta,
    lse_beta_prob,
    margin_max,
    noisy_or,
)
from hpo_extraction.treephenorag.score_store import (_accept_sweep, _load_report_allowlist,
                                 _tau_dir_name)
from hpo_extraction.treephenorag.traversal import NodeEval, traverse

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = pytest.mark.unit


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ── pruning gates ─────────────────────────────────────────────────────────────

def _z(gate, depth_long, subtree_size, margin, missing=0.0):
    """The gate's linear predictor, recomputed from the loaded spec.

    reads the weights off the artifact instead of hardcoding them: the constants are
    a *fit*, refit whenever the ground truth changes (an earlier exploratory run -> an earlier exploratory run), and a test that pins them
    fails on every legitimate refit while proving nothing about the arithmetic. Provenance is
    fixed separately, by ``test_committed_gate_is_the_exp12_01_refit``.
    """
    w = gate.weights
    return (
        gate.intercept
        + w["depth_long"] * depth_long
        + w["log1p_subtree_size"] * np.log1p(subtree_size)
        + w["margin_max"] * margin
        + w["depth_long_missing"] * missing
    )


def test_lr_gate_matches_spec_arithmetic():
    gate = LRGate()  # committed resources/util/minimal_gate.json
    # depth_long=5, subtree_size=0 (log1p=0), margin_max=0, not missing.
    p = gate.p_expand({"depth_long": 5.0, "subtree_size": 0}, np.array([0.0]))
    assert p == pytest.approx(_sigmoid(_z(gate, 5.0, 0, 0.0)), abs=1e-9)


def test_lr_gate_missing_depth_branch():
    gate = LRGate()
    margins = np.array([1.0, -2.0, 0.5])  # margin_max = 1.0
    p = gate.p_expand({"depth_long": None, "subtree_size": 100}, margins)
    # depth_long is imputed to the spec's median and the missing flag is set.
    expect = _z(gate, gate.impute_median["depth_long"], 100, 1.0, missing=1.0)
    assert p == pytest.approx(_sigmoid(expect), abs=1e-9)


def test_committed_gate_is_the_exp12_01_refit():
    """The shipped gate must be the fit made against the ground truth earlier is scored against.

    An earlier exploratory run's gate was fit on the 9-target HCY projection (base rate 0.0105) while earlier scores
    against the full 326-term annotation set. Deploying the projection fit is the confound
    an earlier exploratory run exists to remove, and the two files are schema-identical, swapping them back would
    raise no error and silently produce a plausible, wrong result. Hence this guard.
    """
    import json

    spec = json.loads((_REPO_ROOT / "resources" / "util" / "minimal_gate.json").read_text())
    assert spec["metrics"]["base_rate"] == pytest.approx(0.0365, abs=5e-4), \
        "gate was not fit on the full-annotation gold (base rate should be ~0.0365, not 0.0105)"
    assert LRGate().intercept == pytest.approx(-3.3771, abs=1e-3)


def test_lr_gate_uses_margin_max_not_mean():
    gate = LRGate()
    meta = {"depth_long": 3.0, "subtree_size": 10}
    # Same max, different mean → identical gate output (only the max feeds the gate).
    a = gate.p_expand(meta, [2.0, -5.0, -5.0])
    b = gate.p_expand(meta, [2.0, 1.0, 1.0])
    assert a == pytest.approx(b)


def test_noisy_or_hand_value():
    m = [2.0, -3.0]
    expect = 1.0 - (1 - _sigmoid(2.0)) * (1 - _sigmoid(-3.0))
    assert noisy_or(m) == pytest.approx(expect)
    assert noisy_or([]) == 0.0


def test_accept_confidence_and_margin_max():
    assert accept_confidence([2.0, -3.0]) == pytest.approx(_sigmoid(2.0))
    assert accept_confidence([]) == 0.0
    assert margin_max([2.0, -3.0, 1.0]) == 2.0


def test_lse_beta_hand_value_and_empty():
    # logsumexp([0, 0]) = log 2. Dividing by beta=1 leaves it.
    assert lse_beta([0.0, 0.0]) == pytest.approx(np.log(2.0))
    # A single margin is returned unchanged for any beta.
    assert lse_beta([-3.0], beta=4.0) == pytest.approx(-3.0)
    assert lse_beta([]) == 0.0
    assert lse_beta_prob([]) == 0.0


def test_lse_beta_prob_is_sigmoid_of_lse_and_monotone():
    m = [-2.0, -7.0, 0.5]
    assert lse_beta_prob(m) == pytest.approx(_sigmoid(lse_beta(m)))
    # Monotone in every margin, the property that makes the sigmoid squash ranking-preserving,
    # and therefore lets an earlier exploratory run's lse_beta1 leaderboard describe this rule as deployed.
    scores = [lse_beta_prob([x, -5.0]) for x in (-10.0, -5.0, 0.0, 5.0)]
    assert scores == sorted(scores)


def test_lse_beta_approaches_margin_max_as_beta_grows():
    """Large beta collapses the pooling onto the max, the documented beta -> inf limit."""
    m = [-1.0, -6.0, -9.0]
    assert lse_beta(m, beta=200.0) == pytest.approx(margin_max(m), abs=1e-6)




# ── UnionScorer traversal methods ─────────────────────────────────────────────

@pytest.fixture
def tiny_scorer():
    # Three orthogonal unit context vectors, one sentence each → clean cosines.
    ctx = {
        "HP:0000001": np.array([[1.0, 0.0, 0.0]]),
        "HP:0000002": np.array([[0.0, 1.0, 0.0]]),
        "HP:0000003": np.array([[0.0, 0.0, 1.0]]),
    }
    scorer = UnionScorer(ctx, {h: [h] for h in ctx})
    scorer.prepare()
    # seg0 aligns with HP1. Seg1 is a 0.6/0.8 mix of HP1/HP2.
    segs = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
    scorer.score_patient(segs)
    return scorer


def test_own_sims(tiny_scorer):
    assert tiny_scorer.own_sims("HP:0000001") == pytest.approx([1.0, 0.6])
    assert tiny_scorer.own_sims("HP:0000002") == pytest.approx([0.0, 0.8])
    # A node with no context column scores 0 on its own presence, one entry per segment.
    assert tiny_scorer.own_sims("HP:9999999") == pytest.approx([0.0, 0.0])


def test_union_over_columns_is_elementwise_max(tiny_scorer):
    cols = tiny_scorer.columns_for(["HP:0000001", "HP:0000002", "HP:9999999"])  # junk dropped
    assert len(cols) == 2
    union = tiny_scorer.union_over_columns(cols)
    assert union == pytest.approx([1.0, 0.8])  # max(own1, own2) per segment
    assert tiny_scorer.union_over_columns(np.array([], dtype=np.int64)) == pytest.approx([0.0, 0.0])


def test_per_hpo_max_ranks_ontology(tiny_scorer):
    per = tiny_scorer.per_hpo_max()  # aligned to sorted universe [HP1, HP2, HP3]
    assert per == pytest.approx([1.0, 0.8, 0.0])


# ── traverse ──────────────────────────────────────────────────────────────────

@pytest.fixture
def toy_dag():
    # A,B roots. D has two parents (A,B) → must be scored once. C→F chain.
    children = {"A": ["C", "D"], "B": ["D", "E"], "C": ["F"], "D": [], "E": [], "F": []}
    return children, ["A", "B"]


def test_traverse_bfs_dedup_and_thresholds(toy_dag):
    children, roots = toy_dag
    prune = {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.1, "E": 0.1, "F": 0.1}
    accept = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.8, "F": 0.8}
    scored = []

    def ev(h):
        scored.append(h)
        return NodeEval(prune[h], accept[h], n_slm_calls=5, calls=[{"h": h}])

    res = traverse(children, roots, ev, tau_prune=0.5, tau_accept=0.5)
    assert set(res.visits) == {"A", "B", "C", "D", "E", "F"}
    assert scored.count("D") == 1                       # DAG dedup: scored once
    assert res.accepted == {"E", "F"}
    assert res.n_unique_nodes_visited == 6
    assert res.total_frontier_insertions == 7           # 2 roots + 2(A) + 2(B) + 1(C)
    assert res.n_slm_calls == 30
    assert len(res.calls) == 6


def test_traverse_depth_table(toy_dag):
    children, roots = toy_dag
    def ev(h):
        return NodeEval(0.9, 0.0, 1, [])
    res = traverse(children, roots, ev, 0.5, 0.5)
    rows = {r["depth"]: r for r in res.depth_table()}
    assert rows[1]["n_visited"] == 2 and rows[1]["cumulative_visited"] == 2
    assert rows[2]["n_visited"] == 3 and rows[2]["cumulative_visited"] == 5
    assert rows[3]["n_visited"] == 1 and rows[3]["cumulative_visited"] == 6


def test_traverse_max_nodes_cap(toy_dag):
    children, roots = toy_dag
    def ev(h):
        return NodeEval(0.9, 0.9, 1, [])
    res = traverse(children, roots, ev, 0.5, 0.5, max_nodes=3)
    assert res.n_unique_nodes_visited == 3


def test_tau_sweep_monotone_and_memoised(toy_dag):
    children, roots = toy_dag
    prune = {"A": 0.9, "B": 0.6, "C": 0.7, "D": 0.1, "E": 0.1, "F": 0.1}
    real_calls = []
    cache = {}

    def real(h):
        real_calls.append(h)
        return NodeEval(prune[h], 0.9, 5, [])

    def ev(h):
        if h not in cache:
            cache[h] = real(h)
        return cache[h]

    visited_sets = []
    for t in [0.5, 0.65, 0.8, 0.95]:
        visited_sets.append(set(traverse(children, roots, ev, t, 0.5).visits))
    # visited set shrinks monotonically as tau rises
    for hi, lo in zip(visited_sets, visited_sets[1:]):
        assert lo <= hi
    # every unique node is really evaluated at most once across the whole sweep
    assert len(real_calls) == len(set(real_calls)) == len(visited_sets[0])


# ── tau_accept sweep ──────────────────────────────────────────────────────────

def test_at_accept_is_free_and_nested(toy_dag):
    """The accept sweep must cost nothing and shrink monotonically.

    Expansion depends only on tau_prune, so re-deriving the accept decisions is exact
    post-processing. This is the property the whole (tau_prune x tau_accept) grid rests on: 5x5
    output sets for 5 traversals. If ``at_accept`` ever needed a second BFS, the grid would cost
    25x the SLM calls and the sweep would stop being free.
    """
    children, roots = toy_dag
    accept = {"A": 0.95, "B": 0.85, "C": 0.75, "D": 0.55, "E": 0.3, "F": 0.05}
    n_eval = []

    def ev(h):
        n_eval.append(h)
        return NodeEval(0.9, accept[h], 5, [{"h": h}])

    res = traverse(children, roots, ev, tau_prune=0.5, tau_accept=0.5)
    calls_after_traversal = len(n_eval)

    prev = None
    for tau_a in [0.0, 0.3, 0.55, 0.75, 0.85, 0.95, 1.0]:
        view = res.at_accept(tau_a)
        assert view.accepted == {h for h, a in accept.items() if a >= tau_a}
        if prev is not None:
            assert view.accepted <= prev            # nested as the threshold rises
        prev = view.accepted
        # Traversal-invariant quantities must survive untouched, they are the cost axis.
        assert set(view.visits) == set(res.visits)
        assert view.n_slm_calls == res.n_slm_calls
        assert view.total_frontier_insertions == res.total_frontier_insertions
        # The per-node accepted flags must agree with the derived set (depth_table reads them).
        assert {h for h, v in view.visits.items() if v.accepted} == view.accepted
        assert sum(r["n_accepted"] for r in view.depth_table()) == len(view.accepted)

    assert len(n_eval) == calls_after_traversal, "at_accept must not trigger re-evaluation"
    assert res.accepted == {h for h, a in accept.items() if a >= 0.5}, "original left untouched"


def test_accept_sweep_config_and_dir_naming():
    """Scalar ``tau_accept`` keeps the old behaviour *and* the old directory names.

    An earlier exploratory run is not being re-run, so its existing ``tau_{prune}`` dumps must stay discoverable. Only a real 2-D sweep may introduce the ``_acc_`` suffix.
    """
    assert _accept_sweep({"tau_accept": 0.5}) == [0.5]
    # A sweep overrides the scalar and is deduplicated + sorted, so the grid order is stable.
    assert _accept_sweep({"tau_accept": 0.5, "tau_accept_sweep": [0.9, 0.5, 0.9]}) == [0.5, 0.9]

    assert _tau_dir_name(0.0045, 0.5, False) == "tau_0.0045"
    assert _tau_dir_name(0.0045, 0.9, True) == "tau_0.0045_acc_0.9"


# ── GSC loader ────────────────────────────────────────────────────────────────

def _write_gsc(tmp_path):
    (tmp_path / "Text").mkdir()
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "Text" / "doc1").write_text("Patient has seizures and short stature.")
    (tmp_path / "Annotations" / "doc1").write_text(
        "12:20\tHP:0001250\tseizures\n"
        "25:38\tHP:0004322\tshort stature\n"
        "25:38\tHP:0001250\tseizures\n"        # duplicate term → deduplicated
        "0:0\tNOTACODE\tjunk\n"                 # malformed HPO field → skipped
    )
    # WSL/macOS junk that must be ignored by both loaders:
    (tmp_path / "Text" / "doc1:Zone.Identifier").write_text("junk")
    (tmp_path / "Text" / "._doc1").write_text("junk")
    (tmp_path / "Annotations" / "doc1:Zone.Identifier").write_text("junk")
    return tmp_path


def test_gsc_loader_parses_and_filters_junk(tmp_path):
    from hpo_extraction.evaluation.datasets.gsc import load_gsc_ground_truth, load_gsc_reports

    d = _write_gsc(tmp_path)
    reports = load_gsc_reports(d)
    gt = load_gsc_ground_truth(d)
    assert list(reports) == ["doc1"]                       # junk twins filtered
    assert gt["doc1"] == ["HP:0001250", "HP:0004322"]      # deduped, malformed dropped, order kept


class _FakeSTModel:
    """Minimal SentenceTransformer stand-in with a fixed embedding width."""

    def __init__(self, dim: int):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, sents):
        return np.ones((len(sents), self._dim), dtype=np.float32)


def test_lazycontextdict_invalidates_wrong_dimension_cache(tmp_path):
    """A cached .npy written by a different-width model must be re-encoded, not served.

    This is the failure that crashed the earlier runs: an 8→32 dim mismatch surfaced only when
    UnionScorer.prepare vstacked the pool. The loader now treats it as a cache miss.
    """
    from hpo_extraction.data.loading import LazyContextDict

    (tmp_path / "HP_0000001.txt").write_text("line one\nline two\n", encoding="latin1")

    model = _FakeSTModel(dim=8)
    d = LazyContextDict(str(tmp_path), model)
    emb = d["HP:0000001"]
    assert emb.shape == (2, 8)  # fresh encode, cached

    # Poison the cache with a wrong-width array, newer than the source .txt.
    import os
    npy = tmp_path / ".cache" / "HP_0000001.npy"
    np.save(npy, np.zeros((2, 32), dtype=np.float32))
    os.utime(npy, None)
    d2 = LazyContextDict(str(tmp_path), model)  # fresh instance → no in-memory shortcut
    healed = d2["HP:0000001"]
    assert healed.shape == (2, 8)  # detected 32≠8, re-encoded
    assert np.load(npy).shape == (2, 8)  # stale cache overwritten with the correct width


def test_lazycontextdict_reencodes_corrupt_cache(tmp_path):
    """A truncated/corrupt .npy (concurrent-write race) must be re-encoded, not crash the run.

    This is the failure that killed the earlier array: 12 jobs wrote the shared cache at once and
    produced partial .npy files that np.load could not read.
    """
    import os

    from hpo_extraction.data.loading import LazyContextDict

    (tmp_path / "HP_0000001.txt").write_text("line one\nline two\n", encoding="latin1")
    model = _FakeSTModel(dim=8)
    LazyContextDict(str(tmp_path), model)["HP:0000001"]  # write a valid cache

    npy = tmp_path / ".cache" / "HP_0000001.npy"
    with open(npy, "r+b") as fh:            # truncate to half → unreadable
        data = fh.read()
        fh.seek(0); fh.truncate(len(data) // 2)
    os.utime(npy, None)                     # keep it "newer" than the .txt

    healed = LazyContextDict(str(tmp_path), model)["HP:0000001"]
    assert healed.shape == (2, 8)           # re-encoded rather than raising
    assert np.load(npy).shape == (2, 8)     # corrupt cache overwritten atomically


def _write_raghpo_subset(tmp_path):
    """The shape of ``resources/data/GSC_RAGHPO``: an id list plus a separate annotation file."""
    d = tmp_path / "raghpo"
    d.mkdir()
    (d / "document_ids.txt").write_text("doc1\ndoc2\n")
    (d / "annotations.csv").write_text(
        "doc_id,hpo_id,hpo_description\n"
        "doc1,HP:0001250,seizure\n"
        "doc1,HP:0001250,seizures\n"      # same code twice → deduplicated
        "doc1,HP:0000924,skeletal\n"      # a term the corpus does not annotate
        "doc2,NOTACODE,junk\n"            # malformed → skipped, doc2 keeps an empty ground-truth set
    )
    return d


def test_raghpo_subset_loaders(tmp_path):
    from hpo_extraction.evaluation.datasets.gsc import load_raghpo_ground_truth, load_raghpo_ids

    d = _write_raghpo_subset(tmp_path)
    assert load_raghpo_ids(d) == ["doc1", "doc2"]

    gt = load_raghpo_ground_truth(d)
    assert gt["doc1"] == ["HP:0001250", "HP:0000924"]   # deduped, first-seen order
    # doc2's only row is malformed, but the document must survive: dropping it would shrink the
    # cohort, not record that the document has no annotated terms.
    assert gt["doc2"] == []


def test_restrict_to_ids_refuses_to_shrink_a_cohort_silently(tmp_path):
    from hpo_extraction.evaluation.datasets.gsc import load_gsc_ground_truth, restrict_to_ids

    gt = load_gsc_ground_truth(_write_gsc(tmp_path))
    assert list(restrict_to_ids(gt, ["doc1"])) == ["doc1"]
    with pytest.raises(KeyError, match="absent from the ground truth"):
        restrict_to_ids(gt, ["doc1", "nosuchdoc"])


@pytest.mark.skipif(
    not (_REPO_ROOT / "resources" / "data" / "GSC_RAGHPO" / "document_ids.txt").is_file()
    or not (_REPO_ROOT / "resources" / "data" / "GSC_2024" / "Text").is_dir(),
    reason="the vendored GSC+ corpus / RAG-HPO subset is not present",
)
def test_the_vendored_raghpo_subset_matches_the_corpus_and_the_paper():
    """The three counts ``resources/data/GSC_RAGHPO/PROVENANCE.md`` documents.

    RAG-HPO reports 114 documents and 1013 terms (415 unique). We recover 1011 from their
    workbook, and 1322 for the same documents under the corpus annotations against their 1323.
    Those two gaps are recorded in PROVENANCE.md, this test pins them so they cannot drift
    unnoticed into something larger.
    """
    from hpo_extraction.evaluation.datasets.gsc import (load_gsc_ground_truth, load_raghpo_ground_truth,
                                         load_raghpo_ids, restrict_to_ids)

    gsc_dir = _REPO_ROOT / "resources" / "data" / "GSC_2024"
    subset_dir = _REPO_ROOT / "resources" / "data" / "GSC_RAGHPO"

    ids = load_raghpo_ids(subset_dir)
    assert len(ids) == len(set(ids)) == 114

    # Every id must exist in the corpus, or the derived cohorts describe fewer documents than
    # they claim to.
    corpus = restrict_to_ids(load_gsc_ground_truth(gsc_dir), ids)
    assert sum(len(v) for v in corpus.values()) == 1322

    theirs = load_raghpo_ground_truth(subset_dir)
    assert len(theirs) == 114
    assert sum(len(v) for v in theirs.values()) == 1011
    assert len({c for v in theirs.values() for c in v}) == 415


def test_gsc_dataset_evaluate_flat(tmp_path):
    from hpo_extraction.evaluation.datasets.gsc import GSCDataset

    d = _write_gsc(tmp_path)
    ds = GSCDataset(d)
    # Predict one correct, one wrong term for doc1.
    response = {"doc1": {"HP:0001250": [{"response": "Yes"}], "HP:9999999": [{"response": "Yes"}]}}
    metrics = ds.evaluate(response, target_symptoms=[])
    # 1 TP (HP:0001250), 1 FP (HP:9999999), 1 FN (HP:0004322) → P=R=0.5, F1=0.5
    assert metrics["micro_precision"] == pytest.approx(0.5)
    assert metrics["micro_recall"] == pytest.approx(0.5)
    assert metrics["micro_f1"] == pytest.approx(0.5)


# ─────────────────────────────── cohort allowlist ────────────────────────────────────────────
# HCY's input_dir holds 135 .txt files. The curated ground truth scores 118 of them. Every extra report
# is ~8 GPU-hours of work no table can quote, so the synthetic-sentence score store reads the cohort from the ground truth file
# itself - which also means the run and the scorer cannot disagree about what the cohort is.


class TestReportAllowlist:
    def test_reads_the_curated_gold_two_column_file(self, tmp_path):
        p = tmp_path / 'hcy_ground_truth_curated.csv'
        p.write_text(
            '"patient_id","hpo_codes"' + chr(10) +
            '"SYN004","HP:0000252;HP:0001250"' + chr(10) +
            '"SYN005","HP:0000252"' + chr(10),
            encoding='utf-8',
        )
        assert _load_report_allowlist(str(p)) == ['SYN004', 'SYN005']

    def test_reads_the_raw_gold_header_too(self, tmp_path):
        p = tmp_path / 'hcy_ground_truth_raw.csv'
        p.write_text('ID,Codes' + chr(10) + 'SYN004,x' + chr(10), encoding='utf-8')
        assert _load_report_allowlist(str(p)) == ['SYN004']

    def test_reads_a_bare_id_per_line(self, tmp_path):
        p = tmp_path / 'ids.txt'
        p.write_text(chr(10).join(['SYN004', 'SYN005', '', 'SYN006', '']), encoding='utf-8')
        assert _load_report_allowlist(str(p)) == ['SYN004', 'SYN005', 'SYN006']

    def test_only_the_first_row_can_be_a_header(self, tmp_path):
        """A later cell that happens to read ``id`` is data, not a second header."""
        p = tmp_path / 'ids.txt'
        p.write_text(chr(10).join(['patient_id', 'SYN004', 'id', '']), encoding='utf-8')
        assert _load_report_allowlist(str(p)) == ['SYN004', 'id']

    def test_filtering_keeps_sorted_order_and_names_what_is_missing(self, tmp_path):
        """The filter runs over the sorted ids, so the round-robin shard slices stay stable."""
        p = tmp_path / 'ids.txt'
        p.write_text(chr(10).join(
            ['patient_id', 'SYN005', 'SYN009', 'SYN999', '']), encoding='utf-8')
        present = ['SYN004', 'SYN005', 'SYN006', 'SYN009']
        keep = set(_load_report_allowlist(str(p)))
        assert [r for r in present if r in keep] == ['SYN005', 'SYN009']
        assert sorted(keep - set(present)) == ['SYN999']
