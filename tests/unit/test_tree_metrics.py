"""Unit tests for hpo_extraction.evaluation.tree_metrics (graded HPO scoring, §1).

Builds a small but *real* HPOTree from a synthetic ontology so Wu-Palmer
similarity, ancestor closures and depths are exercised end-to-end (no mocks).

Synthetic ontology (depths from BFS via `Son` from the root HP:0000118):

    HP:0000118 phenotypic abnormality              depth 0  (root)
    ├─ HP:0000707 Nervous system                   depth 1  (layer 1)
    │   ├─ HP:0012638 Abnormal NS physiology        depth 2
    │   │   └─ HP:0001250 Seizure                    depth 3
    │   │       ├─ HP:0007359 Focal-onset seizure    depth 4
    │   │       └─ HP:0011146 Dialeptic seizure      depth 4
    │   └─ HP:0002011 Morphological CNS abnormality  depth 2
    └─ HP:0001626 Cardiovascular                    depth 1  (layer 1)
        └─ HP:0001627 Abnormal heart morphology      depth 2
            └─ HP:0001631 Atrial septal defect        depth 3

HP:0001250 carries alt id HP:0099999 to exercise obsolete-id normalization.
"""

from __future__ import annotations

import json

import pytest

from hpo_extraction.evaluation import tree_metrics as tm
from hpo_extraction.ontology.hpo_tree import HPOTree

ROOT = "HP:0000118"
NERV = "HP:0000707"
NSPHYS = "HP:0012638"
SEIZ = "HP:0001250"
FOCAL = "HP:0007359"
DIAL = "HP:0011146"
MORPH = "HP:0002011"
CARD = "HP:0001626"
HEART = "HP:0001627"
ASD = "HP:0001631"


def _node(hid, name, is_a, fathers, sons, children, alt_id=None):
    return {
        "Id": hid,
        "Name": [name],
        "Alt_id": alt_id or [],
        "Def": [],
        "Comment": [],
        "Synonym": [],
        "Xref": [],
        "Is_a": is_a,
        "Father": {f: True for f in fathers},
        "Child": {c: True for c in children},
        "Son": {s: True for s in sons},
    }


def _synthetic_hpo() -> dict:
    return {
        ROOT: _node(ROOT, "Phenotypic abnormality", [], [], [NERV, CARD],
                    [NERV, CARD, NSPHYS, SEIZ, FOCAL, DIAL, MORPH, HEART, ASD]),
        NERV: _node(NERV, "Abnormality of the nervous system", [ROOT], [ROOT],
                    [NSPHYS, MORPH], [NSPHYS, SEIZ, FOCAL, DIAL, MORPH]),
        CARD: _node(CARD, "Abnormality of the cardiovascular system", [ROOT], [ROOT],
                    [HEART], [HEART, ASD]),
        NSPHYS: _node(NSPHYS, "Abnormal nervous system physiology", [NERV], [NERV, ROOT],
                      [SEIZ], [SEIZ, FOCAL, DIAL]),
        MORPH: _node(MORPH, "Morphological CNS abnormality", [NERV], [NERV, ROOT], [], []),
        SEIZ: _node(SEIZ, "Seizure", [NSPHYS], [NSPHYS, NERV, ROOT],
                    [FOCAL, DIAL], [FOCAL, DIAL], alt_id=["HP:0099999"]),
        FOCAL: _node(FOCAL, "Focal-onset seizure", [SEIZ], [SEIZ, NSPHYS, NERV, ROOT], [], []),
        DIAL: _node(DIAL, "Dialeptic seizure", [SEIZ], [SEIZ, NSPHYS, NERV, ROOT], [], []),
        HEART: _node(HEART, "Abnormal heart morphology", [CARD], [CARD, ROOT], [ASD], [ASD]),
        ASD: _node(ASD, "Atrial septal defect", [HEART], [HEART, CARD, ROOT], [], []),
    }


@pytest.fixture
def tree(tmp_path):
    path = tmp_path / "hpo.json"
    path.write_text(json.dumps(_synthetic_hpo()))
    t = HPOTree(hpo_json_path=str(path))
    t.buildHPOTree()
    return t


# ---------------------------------------------------------------------------
# depth sanity, guards the fixture itself
# ---------------------------------------------------------------------------

def test_synthetic_depths(tree):
    assert tree.depth_dict[ROOT] == 0
    assert tree.depth_dict[NERV] == 1 and tree.depth_dict[CARD] == 1
    assert tree.depth_dict[SEIZ] == 3
    assert tree.depth_dict[FOCAL] == 4 and tree.depth_dict[DIAL] == 4


# ---------------------------------------------------------------------------
# node_similarity, graded by tree distance
# ---------------------------------------------------------------------------

def test_similarity_identity(tree):
    assert tree_metrics_sim(tree, FOCAL, FOCAL) == 1.0


def tree_metrics_sim(tree, a, b):
    return tm.node_similarity(tree, a, b)


def test_similarity_ordering(tree):
    parent = tm.node_similarity(tree, FOCAL, SEIZ)       # child vs parent
    sibling = tm.node_similarity(tree, FOCAL, DIAL)      # siblings
    same_system = tm.node_similarity(tree, FOCAL, MORPH)  # share only nervous system
    cross = tm.node_similarity(tree, FOCAL, ASD)         # different organ system
    assert parent == pytest.approx(6 / 7)
    assert sibling == pytest.approx(0.75)
    assert same_system == pytest.approx(1 / 3)
    assert cross == 0.0
    assert parent > sibling > same_system > cross


def test_similarity_unknown_code_is_zero(tree):
    assert tm.node_similarity(tree, FOCAL, "HP:9999999") == 0.0


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def test_normalize_maps_alt_id(tree):
    valid, dropped = tm.normalize_set({"HP:0099999"}, tree)
    assert valid == {SEIZ}
    assert dropped == 0


def test_normalize_drops_unscorable(tree):
    valid, dropped = tm.normalize_set({FOCAL, "HP:9999999", ""}, tree)
    assert valid == {FOCAL}
    assert dropped == 2


def test_resolve_root_is_unscorable_as_leaf(tree):
    # Root has no ancestors and is dropped from father-based scoring paths.
    assert tm.resolve_code(tree, "  HP:0007359 ") == FOCAL


# ---------------------------------------------------------------------------
# soft precision / recall / f1
# ---------------------------------------------------------------------------

def test_soft_rewards_near_miss(tree):
    # Predict the parent of the true term: exact F1 would be 0, soft is high.
    p, r, f1 = tm.soft_precision_recall_f1({FOCAL}, {SEIZ}, tree)
    assert f1 == pytest.approx(6 / 7)
    assert p == pytest.approx(6 / 7) and r == pytest.approx(6 / 7)


def test_soft_exact_match_is_one(tree):
    assert tm.soft_precision_recall_f1({FOCAL, ASD}, {FOCAL, ASD}, tree) == (1.0, 1.0, 1.0)


def test_soft_empty_cases(tree):
    assert tm.soft_precision_recall_f1(set(), set(), tree) == (1.0, 1.0, 1.0)
    assert tm.soft_precision_recall_f1({FOCAL}, set(), tree) == (0.0, 0.0, 0.0)
    assert tm.soft_precision_recall_f1(set(), {FOCAL}, tree) == (0.0, 0.0, 0.0)


def test_soft_cross_system_is_zero(tree):
    p, r, f1 = tm.soft_precision_recall_f1({FOCAL}, {ASD}, tree)
    assert (p, r, f1) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# ancestor-closure metrics
# ---------------------------------------------------------------------------

def test_closure_set_includes_ancestors(tree):
    assert tm.closure_set({FOCAL}, tree) == {FOCAL, SEIZ, NSPHYS, NERV, ROOT}


def test_closure_rewards_ancestor_prediction(tree):
    # Predict parent SEIZ for truth FOCAL: closure precision is perfect (SEIZ's
    # closure is a subset of FOCAL's), recall < 1 (misses FOCAL itself).
    p, r, f1 = tm.closure_precision_recall_f1({FOCAL}, {SEIZ}, tree)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(4 / 5)


# ---------------------------------------------------------------------------
# aggregate driver
# ---------------------------------------------------------------------------

def test_evaluate_graded_reports_positive_tax(tree):
    gt_sets = [{FOCAL}, {ASD}]
    pred_sets = [{SEIZ}, {ASD}]  # one near-miss, one exact
    m = tm.evaluate_graded(gt_sets, pred_sets, tree)
    assert m["macro_exact_f1"] == pytest.approx(0.5)          # only ASD is exact
    assert m["macro_soft_f1"] > m["macro_exact_f1"]           # near-miss recovered
    assert m["granularity_tax_f1"] == pytest.approx(
        m["macro_soft_f1"] - m["macro_exact_f1"]
    )
    assert m["macro_soft_f1"] == pytest.approx((6 / 7 + 1.0) / 2)


def test_empty_case_audit_partitions(tree):
    audit = tm.empty_case_audit(
        [set(), set(), {FOCAL}, {FOCAL}],
        [set(), {ASD}, set(), {FOCAL}],
    )
    assert audit["both_empty"] == 1
    assert audit["gt_empty_pred_nonempty"] == 1
    assert audit["gt_nonempty_pred_empty"] == 1
    assert audit["both_nonempty"] == 1
    assert audit["n_samples"] == 4
