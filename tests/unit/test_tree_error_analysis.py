"""Unit tests for hpo_extraction.evaluation.tree_error_analysis (tree-geometry diagnostics, §2).

Reuses the same synthetic ontology as test_tree_metrics (see that module's
docstring for the tree shape and node ids).
"""

from __future__ import annotations

import json

import pytest

from hpo_extraction.evaluation import tree_error_analysis as tea
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
        "Id": hid, "Name": [name], "Alt_id": alt_id or [], "Def": [], "Comment": [],
        "Synonym": [], "Xref": [], "Is_a": is_a,
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
# lowest common subsumer
# ---------------------------------------------------------------------------

def test_lcs_siblings(tree):
    lcs, depth = tea.lowest_common_subsumer(tree, FOCAL, DIAL)
    assert lcs == SEIZ and depth == 3


def test_lcs_cross_system_is_root(tree):
    lcs, depth = tea.lowest_common_subsumer(tree, FOCAL, ASD)
    assert lcs == ROOT and depth == 0


# ---------------------------------------------------------------------------
# classify_error, the core relationship labels
# ---------------------------------------------------------------------------

def test_classify_exact(tree):
    assert tea.classify_error(FOCAL, FOCAL, tree) == "exact"


def test_classify_ancestor(tree):
    # Predicted the parent for a specific truth -> too general.
    assert tea.classify_error(SEIZ, FOCAL, tree) == "ancestor"


def test_classify_descendant(tree):
    # Predicted a specific child for a general truth -> too specific.
    assert tea.classify_error(FOCAL, SEIZ, tree) == "descendant"


def test_classify_sibling(tree):
    assert tea.classify_error(FOCAL, DIAL, tree) == "sibling"


def test_classify_same_system(tree):
    # Share only the nervous-system layer-1 ancestor.
    assert tea.classify_error(FOCAL, MORPH, tree) == "same_system"


def test_classify_unrelated(tree):
    assert tea.classify_error(FOCAL, ASD, tree) == "unrelated"


def test_classify_handles_alt_id(tree):
    # HP:0099999 is an obsolete id for SEIZ.
    assert tea.classify_error("HP:0099999", FOCAL, tree) == "ancestor"


# ---------------------------------------------------------------------------
# nearest_gt & layer1
# ---------------------------------------------------------------------------

def test_nearest_gt_picks_closest(tree):
    best, sim = tea.nearest_gt(FOCAL, {ASD, SEIZ, MORPH}, tree)
    assert best == SEIZ
    assert sim == pytest.approx(6 / 7)


def test_primary_layer1(tree):
    assert tea.primary_layer1(tree, FOCAL) == NERV
    assert tea.primary_layer1(tree, ASD) == CARD
    assert tea.primary_layer1(tree, "HP:9999999") == "None"


# ---------------------------------------------------------------------------
# granularity_vector
# ---------------------------------------------------------------------------

def test_granularity_vector_labels_and_counts(tree):
    gt_sets = [{FOCAL}, {FOCAL}, set()]
    pred_sets = [
        {SEIZ},   # ancestor of FOCAL
        {ASD},    # unrelated to FOCAL
        {DIAL},   # no GT -> no_gt (pure hallucination)
    ]
    gv = tea.granularity_vector(gt_sets, pred_sets, tree)
    assert gv["n_fp"] == 3
    assert gv["counts"]["ancestor"] == 1
    assert gv["counts"]["unrelated"] == 1
    assert gv["counts"]["no_gt"] == 1
    assert gv["fractions"]["ancestor"] == pytest.approx(1 / 3)


def test_granularity_vector_excludes_true_positives(tree):
    gv = tea.granularity_vector([{FOCAL, ASD}], [{FOCAL, ASD}], tree)
    assert gv["n_fp"] == 0
    assert gv["counts"] == {}


# ---------------------------------------------------------------------------
# layer1_performance
# ---------------------------------------------------------------------------

def test_layer1_performance_splits_by_system(tree):
    # Nervous: TP=FOCAL, FP=DIAL, FN=none. Cardio: FN=ASD (missed).
    gt_sets = [{FOCAL, ASD}]
    pred_sets = [{FOCAL, DIAL}]
    perf = tea.layer1_performance(gt_sets, pred_sets, tree)
    assert perf[NERV]["tp"] == 1
    assert perf[NERV]["fp"] == 1
    assert perf[NERV]["precision"] == pytest.approx(0.5)
    assert perf[NERV]["recall"] == pytest.approx(1.0)
    assert perf[CARD]["fn"] == 1
    assert perf[CARD]["recall"] == 0.0


# ---------------------------------------------------------------------------
# depth_stratified_recall
# ---------------------------------------------------------------------------

def test_depth_stratified_recall(tree):
    # GT: FOCAL (depth 4, found), SEIZ (depth 3, missed), ASD (depth 3, found)
    gt_sets = [{FOCAL, SEIZ, ASD}]
    pred_sets = [{FOCAL, ASD}]
    ds = tea.depth_stratified_recall(gt_sets, pred_sets, tree)
    assert ds[4]["recall"] == pytest.approx(1.0)
    assert ds[3]["found"] == 1 and ds[3]["total"] == 2
    assert ds[3]["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# miss_similarity_histogram
# ---------------------------------------------------------------------------

def test_miss_similarity_histogram(tree):
    # FP SEIZ vs GT FOCAL -> sim 6/7 ~0.857 -> [0.8,1.0) bin.
    # FP ASD vs GT FOCAL  -> sim 0.0        -> [0.0,0.2) bin.
    gt_sets = [{FOCAL}, {FOCAL}]
    pred_sets = [{SEIZ}, {ASD}]
    hist = tea.miss_similarity_histogram(gt_sets, pred_sets, tree)
    assert hist["n_fp"] == 2
    assert hist["bins"]["[0.8,1.0)"] == 1
    assert hist["bins"]["[0.0,0.2)"] == 1


def test_miss_similarity_histogram_counts_no_gt(tree):
    hist = tea.miss_similarity_histogram([set()], [{FOCAL}], tree)
    assert hist["n_no_gt"] == 1
    assert hist["bins"]["[0.0,0.2)"] == 1
