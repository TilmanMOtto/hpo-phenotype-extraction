"""What pruning actually cost, the recall side of the speed/recall trade.

Tree pruning buys LLM calls by cutting subtrees, and every experiment logs how many calls
it saved. Nothing on disk records what that saving *cost* in recall. This module answers it
by giving every ground-truth term a cause of death:

    found          the run predicted it                                     (TP)
    rejected       the model judged the node and said No                    (a model error)
    pruned         an ancestor was confidently absent, so it was never asked (a pruning cost)
    skipped        below the depth cutoff, never asked                       (a config cost)
    not_traversed  never in the target set at all, never asked               (a coverage gap)

Only ``rejected`` is a model failure. The other three are failures of the *traversal*, and
each has a different fix, which is what a single recall number hides.
"""

from __future__ import annotations

import pandas as pd

from hpo_extraction.evaluation.tree_metrics import resolve_code

# Fates in the order they read as a funnel, best to worst.
GT_FATES = ("found", "rejected", "pruned", "skipped", "not_traversed")


def _pruned_by_patient(nodes: pd.DataFrame) -> dict[str, set[str]]:
    pruned = nodes[nodes["state"] == "pruned"]
    return {p: set(g["hpo_id"]) for p, g in pruned.groupby("patient_id")}


def _cut_point(tree, hpo_id: str, pruned: set[str]) -> str | None:
    """The topmost pruned ancestor-or-self of a term, where the subtree was actually cut.

    A lost GT term may sit under several nested pruned nodes. Blaming the deepest one would
    scatter the audit across dozens of nodes that were themselves collateral. The shallowest
    is the decision that did the damage.
    """
    candidates = {hpo_id} & pruned
    ancestors = tree.getAllFatherHPOByHPO(hpo_id) if hpo_id in tree.phenotypic_abnormalityNT else set()
    candidates |= ancestors & pruned
    if not candidates:
        return None
    return min(candidates, key=lambda c: tree.depth_dict.get(c, 999))


def audit(nodes: pd.DataFrame, gt_dict: dict[str, list[str]], tree) -> dict:
    """Attribute every GT term's fate, and name the prune decisions that cost recall."""
    if nodes.empty:
        return {"n_gt_terms": 0, "gt_fate": {f: 0 for f in GT_FATES}, "cut_points": []}

    pruned_by_patient = _pruned_by_patient(nodes)
    state_by_key = {
        (r.patient_id, r.hpo_id): r.state
        for r in nodes.itertuples(index=False)
    }

    fate_counts = {f: 0 for f in GT_FATES}
    losses: dict[tuple[str, str], list[dict]] = {}  # (patient, cut node) → GT terms lost under it
    found = 0

    gt_rows = nodes[nodes["in_gt"]]
    for row in gt_rows.itertuples(index=False):
        if row.predicted:
            fate_counts["found"] += 1
            found += 1
            continue
        if row.state == "evaluated" or row.state == "assumed_yes":
            fate_counts["rejected"] += 1
        elif row.state == "skipped":
            fate_counts["skipped"] += 1
        elif row.state == "not_traversed":
            fate_counts["not_traversed"] += 1
        elif row.state == "pruned":
            fate_counts["pruned"] += 1
            cut = _cut_point(tree, row.hpo_id, pruned_by_patient.get(row.patient_id, set()))
            key = (row.patient_id, cut or row.hpo_id)
            losses.setdefault(key, []).append({
                "hpo_id": row.hpo_id, "hpo_label": row.hpo_label,
            })

    n_gt = sum(fate_counts.values())
    n_pruned_nodes = int((nodes["state"] == "pruned").sum())
    n_evaluated = int((nodes["state"] == "evaluated").sum())
    n_asked = n_evaluated + n_pruned_nodes

    cut_points = [
        {
            "patient_id": patient,
            "hpo_id": cut,
            "hpo_label": _label(tree, cut),
            "n_gt_lost": len(lost),
            "gt_lost": lost,
        }
        for (patient, cut), lost in losses.items()
    ]
    cut_points.sort(key=lambda c: -c["n_gt_lost"])

    recall_actual = found / n_gt if n_gt else 0.0
    # The most pruning could possibly have cost: assume every pruned GT term would have been
    # found had it been asked. An upper bound on the recall a no-pruning rerun could recover,
    # not a prediction of it, the model might well have rejected them too.
    recall_ceiling = (found + fate_counts["pruned"]) / n_gt if n_gt else 0.0

    return {
        "n_gt_terms": n_gt,
        "gt_fate": fate_counts,
        "recall_actual": recall_actual,
        "recall_ceiling_no_pruning": recall_ceiling,
        "recall_lost_to_pruning": recall_ceiling - recall_actual,
        "n_pruned_nodes": n_pruned_nodes,
        "n_evaluated_nodes": n_evaluated,
        "n_skipped_nodes": int((nodes["state"] == "skipped").sum()),
        "llm_call_savings_pct": (100.0 * n_pruned_nodes / n_asked) if n_asked else 0.0,
        "n_cut_points": len(cut_points),
        "n_costly_cut_points": sum(1 for c in cut_points if c["n_gt_lost"]),
        "cut_points": cut_points[:50],
    }


def _label(tree, hpo_id: str) -> str:
    try:
        return tree.getNameByHPO(hpo_id)
    except (KeyError, IndexError):
        return hpo_id


def pruned_subtrees_with_gt(nodes: pd.DataFrame, gt_dict, tree) -> dict[str, set[str]]:
    """{patient: {pruned node ids whose subtree contained a GT term}}, for the tree view.

    These are the nodes to outline in red: the cut that demonstrably threw away truth.
    """
    out: dict[str, set[str]] = {}
    pruned_by_patient = _pruned_by_patient(nodes)
    for patient, pruned in pruned_by_patient.items():
        gt_norm = {resolve_code(tree, g) for g in gt_dict.get(patient, [])} - {None}
        guilty = set()
        for g in gt_norm:
            cut = _cut_point(tree, g, pruned)
            if cut:
                guilty.add(cut)
        out[patient] = guilty
    return out
