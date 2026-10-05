"""Graded (tree-aware) scoring for HPO prediction sets, the §1 "measurement lens".

Exact-set F1 (see ``hpo_extraction.evaluation.set_metrics.calc_metric``) treats a prediction one node
off in the ontology identically to a wild hallucination. These functions re-score
the same ``(ground_truth, prediction)`` HPO sets through *graded* lenses so a
near-miss (parent/child of the truth) is distinguished from a genuine miss.

Pure functions with no IO. Every public function takes an ``HPOTree`` instance. Depth information is built lazily (``buildHPOTree`` is *not* called by the tree's
own ``__init__``). All functions operate on sets of HPO id strings, e.g.
``{"HP:0001250", "HP:0007359"}``.

Two complementary lenses are provided:

* **Soft (best-match-average)**, each term is credited by its *most similar*
  counterpart via Wu-Palmer node similarity. Degrades gracefully with tree
  distance. See :func:`soft_precision_recall_f1`.
* **Ancestor-closure**, expand both sets to their ancestor closures and score
 . The gap ``closure_f1 - exact_f1`` is the pure specificity-mismatch
  fraction. See :func:`closure_precision_recall_f1`.
"""

from __future__ import annotations

from statistics import mean

from hpo_extraction.evaluation.set_metrics import calc_metric


# ---------------------------------------------------------------------------
# Tree helpers (normalization + safe similarity)
# ---------------------------------------------------------------------------

def _ensure_depths(tree) -> None:
    """Populate ``tree.depth_dict`` if the caller hasn't run ``buildHPOTree`` yet.

    Idempotent and cheap after the first call, guarded by ``hasattr`` so it runs
    at most once per tree instance even when invoked per-patient in a loop.
    """
    if not hasattr(tree, "depth_dict"):
        tree.buildHPOTree()


def resolve_code(tree, code: str) -> str | None:
    """Normalize an HPO id to a scorable canonical node, or ``None`` if unscorable.

    Handles: whitespace, alt/obsolete ids (mapped via ``tree.alt_id_dict``), and
    filters out anything outside the phenotypic-abnormality subtree or missing a
    depth (which would silently score as similarity 0.0 otherwise).
    """
    if not code:
        return None
    code = code.strip()
    if code not in tree.data and code in getattr(tree, "alt_id_dict", {}):
        code = tree.alt_id_dict[code]
    if code not in tree.phenotypic_abnormality:
        return None
    if code not in tree.depth_dict:
        return None
    return code


def normalize_set(codes, tree) -> tuple[set[str], int]:
    """Resolve a set of raw codes to canonical scorable nodes.

    Returns ``(valid_set, n_dropped)`` so callers can *report* how many codes were
    obsolete/out-of-subtree rather than silently absorbing them as total misses.
    """
    _ensure_depths(tree)
    valid: set[str] = set()
    dropped = 0
    for c in codes:
        r = resolve_code(tree, c)
        if r is None:
            dropped += 1
        else:
            valid.add(r)
    return valid, dropped


def node_similarity(tree, a: str, b: str) -> float:
    """Wu-Palmer node similarity in ``[0, 1]``, reliable to invalid/obsolete codes.

    Wraps ``HPOTree.getNodeSimilarityByID`` with normalization and guards against
    the ``IndexError``/``KeyError`` that method can raise on nodes with no common
    ancestor or missing depth.
    """
    _ensure_depths(tree)
    ra, rb = resolve_code(tree, a), resolve_code(tree, b)
    if ra is None or rb is None:
        return 0.0
    if ra == rb:
        return 1.0
    try:
        return tree.getNodeSimilarityByID(ra, rb)
    except (KeyError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# Soft (best-match-average) precision / recall / F1
# ---------------------------------------------------------------------------

def soft_precision_recall_f1(gt, pred, tree) -> tuple[float, float, float]:
    """Best-match-average graded precision, recall, F1 for two HPO sets.

    * ``recall``    = mean over GT terms of the similarity to the nearest prediction
    * ``precision`` = mean over predictions of the similarity to the nearest GT term
    * ``f1``        = harmonic mean of the two

    Empty-case semantics mirror :func:`hpo_extraction.evaluation.set_metrics.calc_metric`:
    both empty -> ``(1, 1, 1)``. One empty -> ``(0, 0, 0)``.

    Codes are normalized first. Unresolved codes are dropped (they cannot be
    scored on the tree). Use :func:`normalize_set` if you need the drop counts.
    """
    _ensure_depths(tree)
    gt_set, _ = normalize_set(gt, tree)
    pred_set, _ = normalize_set(pred, tree)
    return _soft_prf_from_normalized(gt_set, pred_set, tree)


def _soft_prf_from_normalized(gt_set, pred_set, tree) -> tuple[float, float, float]:
    """Best-match-average P/R/F1 on already-normalized (scorable) node sets.

    Split out from :func:`soft_precision_recall_f1` so callers that normalize once
    (e.g. :func:`evaluate_graded`, which also needs the raw per-term sims for the
    micro pool) don't re-resolve every code.
    """
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gt_set or not pred_set:
        return 0.0, 0.0, 0.0

    recall = mean(max(node_similarity(tree, g, p) for p in pred_set) for g in gt_set)
    precision = mean(max(node_similarity(tree, p, g) for g in gt_set) for p in pred_set)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Ancestor-closure exact metrics
# ---------------------------------------------------------------------------

def closure_set(codes, tree) -> set[str]:
    """Expand a set of HPO ids to the union of each term and all its ancestors."""
    _ensure_depths(tree)
    out: set[str] = set()
    for c in codes:
        r = resolve_code(tree, c)
        if r is None:
            continue
        out.add(r)
        out |= tree.getAllFatherHPOByHPO(r)
    return out


def closure_precision_recall_f1(gt, pred, tree) -> tuple[float, float, float]:
    """Exact precision/recall/F1 computed on the ancestor closures of both sets.

    A prediction that is the parent of a true term now overlaps its closure, so
    this lens rewards "right concept, wrong granularity". The gap vs plain exact
    F1 isolates the specificity-mismatch component of the error.
    """
    return calc_metric(closure_set(gt, tree), closure_set(pred, tree))


# ---------------------------------------------------------------------------
# Aggregate driver + empty-case audit
# ---------------------------------------------------------------------------

def empty_case_audit(gt_sets, pred_sets) -> dict:
    """Count how patients partition by (GT empty?, prediction empty?).

    ``both_empty`` cases score a free ``(1,1,1)`` under exact and soft metrics and
    can silently inflate macro scores, this surfaces how many there are.
    """
    both_empty = gt_only_empty = pred_only_empty = both_nonempty = 0
    for gt, pred in zip(gt_sets, pred_sets):
        g, p = bool(gt), bool(pred)
        if not g and not p:
            both_empty += 1
        elif not g and p:
            gt_only_empty += 1
        elif g and not p:
            pred_only_empty += 1
        else:
            both_nonempty += 1
    return {
        "both_empty": both_empty,
        "gt_empty_pred_nonempty": gt_only_empty,
        "gt_nonempty_pred_empty": pred_only_empty,
        "both_nonempty": both_nonempty,
        "n_samples": len(gt_sets),
    }


def _micro_prf(tp: float, pred_count: float, actual_count: float) -> tuple[float, float, float]:
    """Micro P/R/F1 from pooled counts, guarding empty denominators.

    ``tp`` is a float so the soft lens can pool graded best-match sims (in ``[0, 1]``),
    not integer hits. Precision divides by the prediction total, recall by the
    GT total. A zero denominator yields 0.0 (no predictions / no truth), not
    raising.
    """
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / actual_count if actual_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate_graded(gt_sets, pred_sets, tree) -> dict:
    """Macro- and micro-averaged exact / soft / closure metrics over a list of patients.

    Returns a flat dict of ``{macro,micro}_{exact,soft,closure}_{precision,recall,f1}``,
    the main ``granularity_tax_f1`` (soft macro F1 minus exact macro F1, how
    much apparent failure is really near-miss), the empty-case audit, and
    per-sample lists for drill-down. Inputs mirror
    :func:`hpo_extraction.evaluation.set_metrics.evaluate_micro_macro`: two aligned lists of sets.

    Macro averages per-patient scores (equal weight per patient). Micro pools across
    all terms (equal weight per term). For exact/closure the micro pool is TP / pred /
    GT term counts. For soft it pools per-term best-match Wu-Palmer sims, so a patient
    with no predictions still drags recall down (its GT terms score 0). Both-empty
    patients contribute no terms to the micro pool, they cannot inflate it the way
    they hand macro a free 1.0.
    """
    _ensure_depths(tree)
    per_sample: dict[str, list] = {
        "exact_f1": [], "soft_f1": [], "closure_f1": [],
    }
    acc = {k: 0.0 for k in (
        "exact_precision", "exact_recall", "exact_f1",
        "soft_precision", "soft_recall", "soft_f1",
        "closure_precision", "closure_recall", "closure_f1",
    )}
    # Exact/closure micro pools: [tp, pred_total, gt_total] (precision & recall share tp).
    hard_pool = {lens: [0, 0, 0] for lens in ("exact", "closure")}
    # Soft micro pool: precision and recall have *different* numerators (a best-match
    # sim summed over predictions vs over GT terms), so it needs its own four counters.
    soft_prec_sum = 0.0; soft_pred_total = 0
    soft_recall_sum = 0.0; soft_gt_total = 0

    for gt, pred in zip(gt_sets, pred_sets):
        gt_raw, pred_raw = set(gt), set(pred)
        ep, er, ef = calc_metric(gt_raw, pred_raw)
        acc["exact_precision"] += ep; acc["exact_recall"] += er; acc["exact_f1"] += ef
        hard_pool["exact"][0] += len(gt_raw & pred_raw)
        hard_pool["exact"][1] += len(pred_raw)
        hard_pool["exact"][2] += len(gt_raw)

        # Soft: normalize once, then derive both the per-patient macro score and the
        # per-term sims that feed the micro pool.
        gt_n, _ = normalize_set(gt, tree)
        pred_n, _ = normalize_set(pred, tree)
        sp, sr, sf = _soft_prf_from_normalized(gt_n, pred_n, tree)
        acc["soft_precision"] += sp; acc["soft_recall"] += sr; acc["soft_f1"] += sf
        for g in gt_n:
            soft_recall_sum += max((node_similarity(tree, g, p) for p in pred_n), default=0.0)
        soft_gt_total += len(gt_n)
        for p in pred_n:
            soft_prec_sum += max((node_similarity(tree, p, g) for g in gt_n), default=0.0)
        soft_pred_total += len(pred_n)

        gc, pc = closure_set(gt, tree), closure_set(pred, tree)
        cp, cr, cf = calc_metric(gc, pc)
        acc["closure_precision"] += cp; acc["closure_recall"] += cr; acc["closure_f1"] += cf
        hard_pool["closure"][0] += len(gc & pc)
        hard_pool["closure"][1] += len(pc)
        hard_pool["closure"][2] += len(gc)

        per_sample["exact_f1"].append(ef)
        per_sample["soft_f1"].append(sf)
        per_sample["closure_f1"].append(cf)

    n = len(gt_sets) or 1
    metrics = {f"macro_{k}": v / n for k, v in acc.items()}
    for lens in ("exact", "closure"):
        tp, pred_total, gt_total = hard_pool[lens]
        mp, mr, mf = _micro_prf(tp, pred_total, gt_total)
        metrics[f"micro_{lens}_precision"] = mp
        metrics[f"micro_{lens}_recall"] = mr
        metrics[f"micro_{lens}_f1"] = mf

    # Soft micro: pooled precision and recall sims each over their own denominator.
    s_prec = soft_prec_sum / soft_pred_total if soft_pred_total else 0.0
    s_recall = soft_recall_sum / soft_gt_total if soft_gt_total else 0.0
    s_f1 = (2 * s_prec * s_recall / (s_prec + s_recall)) if s_prec + s_recall > 0 else 0.0
    metrics["micro_soft_precision"] = s_prec
    metrics["micro_soft_recall"] = s_recall
    metrics["micro_soft_f1"] = s_f1

    metrics["granularity_tax_f1"] = metrics["macro_soft_f1"] - metrics["macro_exact_f1"]
    metrics["empty_cases"] = empty_case_audit(gt_sets, pred_sets)
    metrics["per_sample"] = per_sample
    return metrics
