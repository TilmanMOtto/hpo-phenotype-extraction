"""Shared HPO-tree pruning primitives and confidence-threshold sweeping.

These helpers are lifted verbatim (behaviour-preserving) from
``experiments/an earlier exploratory run/run.py`` and
``experiments/an earlier exploratory run/run.py`` so that the threshold-sweep
tooling (``scripts/sweep_tau.py``) and the new group-10 experiments agree bit-for-bit.

Existing experiments keep their own local copies. Only new code imports this module.

Key property (unit-tested): ``apply_pruning_verdict`` re-runs the topological pruning walk
from a *full, unpruned* per-node confidence table and produces the same node verdicts
that an earlier exploratory run's inline pruning would, because a node that satisfies ``should_prune_confident``
is marked ``"pruned"`` during the walk before any descendant consults it, and non-pruned nodes
carry identical confidences in both the inline and post-hoc paths. This is what makes a
post-hoc (tau_accept, tau_prune) sweep over cached margins valid without re-running inference.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

# NOTE: ``hpo_extraction.ontology.hpo_tree`` transitively imports nltk. The pure pruning/sweep functions
# below must stay importable without it (for CPU-only tooling and unit tests), so HPOTree/
# HPO_class are imported lazily inside the three functions that build the ontology graph.
# ``from __future__ import annotations`` keeps the ``HPOTree`` type hints as lazy strings.


# ── Target set / graph construction ───────────────────────────────────────────

def build_expanded_target_set(
    gt_dict: dict,
    original_targets: Iterable[str],
    hpo_tree: "HPOTree",
    logger=None,
) -> tuple[set[str], int, int]:
    """Expand ground-truth HPOs upward to their ancestors, union the original targets.

    Returns (target_set, n_valid_gt_hpos, n_skipped_gt).
    """
    from hpo_extraction.ontology.hpo_tree import HPO_class

    all_gt_hpos: set[str] = {hpo for codes in gt_dict.values() for hpo in codes}
    missing = {h for h in all_gt_hpos if h not in hpo_tree.data}
    if missing and logger is not None:
        logger.warning("GT HPOs absent from hpo.json (skipped): %s", missing)
    valid_gt_hpos = all_gt_hpos - missing

    target_set: set[str] = set()
    for hpo_id in valid_gt_hpos:
        target_set.add(hpo_id)
        target_set.update(HPO_class(hpo_tree.data[hpo_id]).father)
    target_set.add(hpo_tree.root)

    for hpo_id in original_targets:
        if hpo_id in hpo_tree.data:
            target_set.add(hpo_id)

    return target_set, len(valid_gt_hpos), len(missing)


def build_direct_parents_map(target_set: set[str], hpo_tree: "HPOTree") -> dict[str, list[str]]:
    """Map each node to the subset of its direct ``is_a`` parents that are in target_set."""
    from hpo_extraction.ontology.hpo_tree import HPO_class

    return {
        n: [p for p in HPO_class(hpo_tree.data[n]).is_a if p in target_set]
        for n in target_set
    }


def topo_sort(target_set: set[str], hpo_tree: "HPOTree") -> tuple[list[str], dict[str, list[str]]]:
    """Kahn topological sort over target_set (parents before children). Returns (order, children)."""
    from hpo_extraction.ontology.hpo_tree import HPO_class

    in_degree: dict[str, int] = {n: 0 for n in target_set}
    children: dict[str, list[str]] = {n: [] for n in target_set}

    for n in target_set:
        direct_parents = [p for p in HPO_class(hpo_tree.data[n]).is_a if p in target_set]
        in_degree[n] = len(direct_parents)
        for p in direct_parents:
            children[p].append(n)

    queue = deque(n for n in target_set if in_degree[n] == 0)
    order: list[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for child in children[n]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return order, children


# ── Pruning rules ─────────────────────────────────────────────────────────────

def should_prune_confident(
    hpo_id: str,
    direct_parents_map: dict[str, list[str]],
    node_conf: dict[str, float],
    node_pred: dict[str, str],
    tau_prune: float,
) -> bool:
    """Confidence analog of two-consecutive-No (an earlier exploratory run).

    A node counts as CONFIDENTLY absent if it was pruned, or it was evaluated with
    ``node_conf < tau_prune``. Prune ``hpo_id`` if a direct parent was pruned, or if a parent
    AND one of its parents are both confidently absent. A parent only *weakly* absent
    (conf in [tau_prune, tau_accept)) does NOT trigger pruning.
    """
    def low(n: str) -> bool:
        return node_pred.get(n) == "pruned" or (n in node_conf and node_conf[n] < tau_prune)

    for p in direct_parents_map[hpo_id]:
        if node_pred.get(p) == "pruned":
            return True
        if low(p):
            for gp in direct_parents_map[p]:
                if low(gp):
                    return True
    return False


def should_prune_two_no(
    hpo_id: str,
    direct_parents_map: dict[str, list[str]],
    node_pred: dict[str, str],
) -> bool:
    """Hard two-consecutive-No pruning (an earlier exploratory run/an earlier exploratory run)."""
    for p in direct_parents_map[hpo_id]:
        pred_p = node_pred.get(p)
        if pred_p == "pruned":
            return True
        if pred_p == "no":
            for gp in direct_parents_map[p]:
                if node_pred.get(gp) in ("no", "pruned"):
                    return True
    return False


# ── Post-hoc verdict/pruning re-run + threshold sweep ─────────────────────────

def apply_pruning_verdict(
    topo_order: list[str],
    direct_parents_map: dict[str, list[str]],
    depth_skip_set: set[str],
    node_conf: dict[str, float],
    tau_accept: float,
    tau_prune: float,
) -> tuple[dict, dict]:
    """Re-run one patient's topo order from a full (unpruned) confidence table.

    ``node_conf`` holds a confidence for every *scored* node (depth-skipped nodes absent).
    Returns (response_dict_for_patient, node_pred) where response_dict_for_patient is
    ``{hpo: {0: {"response": "Yes"|"No"|"PRUNED"|"SKIPPED"}}}``, the shape HCYDataset consumes.
    """
    node_pred: dict[str, str] = {}
    resp: dict[str, dict] = {}
    for hpo_id in topo_order:
        if hpo_id in depth_skip_set:
            node_pred[hpo_id] = "skipped"
            resp[hpo_id] = {0: {"response": "SKIPPED"}}
            continue
        if should_prune_confident(hpo_id, direct_parents_map, node_conf, node_pred, tau_prune):
            node_pred[hpo_id] = "pruned"
            resp[hpo_id] = {0: {"response": "PRUNED"}}
            continue
        conf = node_conf.get(hpo_id, 0.0)
        positive = conf >= tau_accept
        node_pred[hpo_id] = "yes" if positive else "no"
        resp[hpo_id] = {0: {"response": "Yes" if positive else "No"}}
    return resp, node_pred


def sweep_thresholds(
    node_conf_by_patient: dict[str, dict[str, float]],
    topo_order: list[str],
    direct_parents_map: dict[str, list[str]],
    depth_skip_set: set[str],
    dataset,
    original_targets: list[str],
    expanded_targets: list[str] | None,
    tau_accept_grid: Iterable[float],
    tau_prune_grid: Iterable[float],
) -> list[dict]:
    """Evaluate every (tau_accept, tau_prune) on the grid via HCYDataset.evaluate.

    Uses the *same* evaluate path the experiments report, so a sweep row at (ta, tp)
    reproduces the metrics an experiment run with those thresholds would produce on the same
    confidences. Returns one row dict per grid point (original-target metrics, optional expanded
    metrics prefixed ``expanded_``, plus pruned/positive counts).
    """
    rows: list[dict] = []
    for ta in tau_accept_grid:
        for tp in tau_prune_grid:
            response_dict: dict = {}
            n_pruned = n_positive = n_evaluated = 0
            for patient, node_conf in node_conf_by_patient.items():
                resp, node_pred = apply_pruning_verdict(
                    topo_order, direct_parents_map, depth_skip_set, node_conf, ta, tp
                )
                response_dict[patient] = resp
                n_pruned += sum(1 for v in node_pred.values() if v == "pruned")
                n_positive += sum(1 for v in node_pred.values() if v == "yes")
                n_evaluated += sum(1 for v in node_pred.values() if v in ("yes", "no"))

            m_orig = dataset.evaluate(
                response_dict, original_targets, output_dir=None, filter_unannotated=True,
            )
            row = {
                "tau_accept": round(float(ta), 4),
                "tau_prune": round(float(tp), 4),
                "n_pruned": n_pruned,
                "n_positive": n_positive,
                "n_evaluated": n_evaluated,
                **{k: v for k, v in m_orig.items()},
            }
            if expanded_targets:
                m_exp = dataset.evaluate(
                    response_dict, expanded_targets, output_dir=None, filter_unannotated=True,
                )
                row.update({f"expanded_{k}": v for k, v in m_exp.items()})
            rows.append(row)
    return rows


def summarize_sweep(rows: list[dict], precision_floor: float | None = None) -> dict:
    """Pick the F1-optimal grid point and (optionally) the max-recall point meeting a
    ``micro_precision >= precision_floor`` constraint."""
    best_f1 = max(rows, key=lambda r: r["micro_f1"])
    out = {"best_f1": best_f1}
    if precision_floor is not None:
        eligible = [r for r in rows if r["micro_precision"] >= precision_floor]
        out["precision_floor"] = precision_floor
        out["best_recall_at_floor"] = (
            max(eligible, key=lambda r: r["micro_recall"]) if eligible else None
        )
    return out
