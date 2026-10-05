"""Tree-geometry characterization of HPO prediction errors, the §2 "where/why" lens.

Where :mod:`hpo_extraction.evaluation.tree_metrics` re-scores predictions on a graded scale, this
module *classifies* each error by its position in the ontology, turning a scalar F1
into an actionable diagnosis: are we too general, too specific, in the wrong
neighborhood, or genuinely lost?

Pure functions, no IO. Inputs mirror
:func:`hpo_extraction.evaluation.set_metrics.evaluate_micro_macro`: two aligned ``list[set]`` of HPO
ids (ground truth and predictions, one set per patient).

Key primitives:

* :func:`classify_error`, one (prediction, truth) pair -> relationship label
* :func:`granularity_vector`, distribution of relationship labels over all FPs
* :func:`layer1_performance`, per organ-system precision/recall
* :func:`depth_stratified_recall`, recall as a function of GT term depth
* :func:`miss_similarity_histogram`, near-miss vs hallucination spectrum
"""

from __future__ import annotations

from collections import Counter, defaultdict

from hpo_extraction.evaluation.tree_metrics import _ensure_depths, node_similarity, resolve_code

# Relationship labels emitted by classify_error, in "closeness" order.
ERROR_LABELS = ("exact", "ancestor", "descendant", "sibling", "same_system", "unrelated")


# ---------------------------------------------------------------------------
# Pairwise relationship
# ---------------------------------------------------------------------------

def lowest_common_subsumer(tree, a: str, b: str) -> tuple[str | None, int]:
    """Return the deepest shared ancestor of two nodes and its depth.

    Returns ``(None, -1)`` if either code is unresolvable or they share no ancestor.
    Because every phenotypic-abnormality node descends from the root, resolvable
    pairs always share at least the root (depth 0).
    """
    _ensure_depths(tree)
    ra, rb = resolve_code(tree, a), resolve_code(tree, b)
    if ra is None or rb is None:
        return None, -1
    fa = tree.getAllFatherHPOByHPO(ra) | {ra}
    fb = tree.getAllFatherHPOByHPO(rb) | {rb}
    common = [c for c in (fa & fb) if c in tree.depth_dict]
    if not common:
        return None, -1
    lcs = max(common, key=lambda c: tree.depth_dict[c])
    return lcs, tree.depth_dict[lcs]


def classify_error(pred: str, gt: str, tree) -> str:
    """Classify a predicted code's relationship to a ground-truth code.

    * ``exact``, identical term
    * ``ancestor``, prediction is an ancestor of the truth (too general)
    * ``descendant``, prediction is a descendant of the truth (too specific)
    * ``sibling``, shares an ancestor *below* the organ-system level (near cousin)
    * ``same_system``, shares only the layer-1 organ system
    * ``unrelated``, shares only the root (or unresolvable)

    Sibling vs same_system vs unrelated is decided by the depth of the lowest
    common subsumer, since the trivially-shared root makes plain ancestor overlap
    uninformative.
    """
    _ensure_depths(tree)
    rp, rg = resolve_code(tree, pred), resolve_code(tree, gt)
    if rp is None or rg is None:
        return "unrelated"
    if rp == rg:
        return "exact"
    if rp in tree.getAllFatherHPOByHPO(rg):
        return "ancestor"
    if rg in tree.getAllFatherHPOByHPO(rp):
        return "descendant"
    _, depth = lowest_common_subsumer(tree, rp, rg)
    if depth >= 2:
        return "sibling"
    if depth == 1:
        return "same_system"
    return "unrelated"


def nearest_gt(pred: str, gt_set, tree) -> tuple[str | None, float]:
    """Return the GT term most similar to ``pred`` and that Wu-Palmer similarity."""
    best_code, best_sim = None, -1.0
    for g in gt_set:
        s = node_similarity(tree, pred, g)
        if s > best_sim:
            best_code, best_sim = g, s
    return best_code, (best_sim if best_sim >= 0 else 0.0)


# ---------------------------------------------------------------------------
# Organ-system (layer 1) helpers
# ---------------------------------------------------------------------------

def primary_layer1(tree, code: str) -> str:
    """Return a single layer-1 (organ-system) id for a code, or ``"None"``.

    ``HPOTree.getLayer1HPOByHPO`` may return several ancestors. We take the
    lexicographically-first for a stable single bucket.
    """
    r = resolve_code(tree, code)
    if r is None:
        return "None"
    l1 = tree.getLayer1HPOByHPO(r)
    if not l1 or l1 == ["None"]:
        return "None"
    return sorted(l1)[0]


# ---------------------------------------------------------------------------
# Aggregate diagnostics over a corpus
# ---------------------------------------------------------------------------

def _false_positives(gt_set, pred_set, tree) -> list[str]:
    """Predicted codes not present in GT (after normalization)."""
    gt_norm = {resolve_code(tree, g) for g in gt_set}
    return [p for p in pred_set if resolve_code(tree, p) not in gt_norm]


def granularity_vector(gt_sets, pred_sets, tree) -> dict:
    """Distribution of error relationships over every false-positive prediction.

    Each FP is matched to its nearest GT term and classified. FPs for patients
    with no GT terms are bucketed as ``no_gt`` (pure hallucinations). Returns
    ``{"counts": {...}, "fractions": {...}, "n_fp": int}``, the single most
    decision-relevant chart, since each bucket implies a different fix.
    """
    _ensure_depths(tree)
    counts: Counter = Counter()
    for gt_set, pred_set in zip(gt_sets, pred_sets):
        for fp in _false_positives(gt_set, pred_set, tree):
            if not gt_set:
                counts["no_gt"] += 1
                continue
            best_gt, _ = nearest_gt(fp, gt_set, tree)
            counts[classify_error(fp, best_gt, tree) if best_gt else "no_gt"] += 1
    n_fp = sum(counts.values())
    fractions = {k: v / n_fp for k, v in counts.items()} if n_fp else {}
    return {"counts": dict(counts), "fractions": fractions, "n_fp": n_fp}


def layer1_performance(gt_sets, pred_sets, tree) -> dict:
    """Per organ-system (layer-1) exact TP/FP/FN with precision and recall.

    Reveals systematic blind spots, organ systems where the pipeline is strong
    vs where it hemorrhages precision or recall.
    """
    _ensure_depths(tree)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for gt_set, pred_set in zip(gt_sets, pred_sets):
        gt_norm = {resolve_code(tree, g) for g in gt_set} - {None}
        pred_norm = {resolve_code(tree, p) for p in pred_set} - {None}
        for code in gt_norm & pred_norm:
            stats[primary_layer1(tree, code)]["tp"] += 1
        for code in pred_norm - gt_norm:
            stats[primary_layer1(tree, code)]["fp"] += 1
        for code in gt_norm - pred_norm:
            stats[primary_layer1(tree, code)]["fn"] += 1

    out: dict[str, dict] = {}
    for l1, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        label = tree.getNameByHPO(l1) if l1 != "None" else "None"
        out[l1] = {**s, "precision": precision, "recall": recall, "name": label}
    return out


def depth_stratified_recall(gt_sets, pred_sets, tree) -> dict:
    """Micro exact recall of GT terms grouped by their depth in the ontology.

    Tests the "recall dies on specific (deep) terms" hypothesis directly: if
    recall falls with depth while shallow recall is high, the problem is
    specificity, not detection.
    """
    _ensure_depths(tree)
    by_depth: dict[int, dict[str, int]] = defaultdict(lambda: {"found": 0, "total": 0})
    for gt_set, pred_set in zip(gt_sets, pred_sets):
        pred_norm = {resolve_code(tree, p) for p in pred_set} - {None}
        for g in gt_set:
            rg = resolve_code(tree, g)
            if rg is None:
                continue
            depth = tree.depth_dict[rg]
            by_depth[depth]["total"] += 1
            if rg in pred_norm:
                by_depth[depth]["found"] += 1
    return {
        d: {**v, "recall": (v["found"] / v["total"] if v["total"] else 0.0)}
        for d, v in sorted(by_depth.items())
    }


def miss_similarity_histogram(gt_sets, pred_sets, tree, bin_edges=None) -> dict:
    """Histogram of nearest-GT similarity for every false positive.

    A bimodal shape (a hump near 1.0 + a hump near 0.0) means two distinct FP
    populations, near-misses and hallucinations, that warrant different fixes.
    Returns ``{"bins": {label: count}, "n_fp": int, "n_no_gt": int}``.
    """
    _ensure_depths(tree)
    if bin_edges is None:
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = [f"[{bin_edges[i]:.1f},{bin_edges[i + 1]:.1f})" for i in range(len(bin_edges) - 1)]
    bins: Counter = Counter()
    n_fp = n_no_gt = 0
    for gt_set, pred_set in zip(gt_sets, pred_sets):
        for fp in _false_positives(gt_set, pred_set, tree):
            n_fp += 1
            if not gt_set:
                n_no_gt += 1
                sim = 0.0
            else:
                _, sim = nearest_gt(fp, gt_set, tree)
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= sim < bin_edges[i + 1]:
                    bins[labels[i]] += 1
                    break
    return {"bins": {lbl: bins.get(lbl, 0) for lbl in labels}, "n_fp": n_fp, "n_no_gt": n_no_gt}
