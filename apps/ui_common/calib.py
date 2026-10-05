"""Confidence calibration and post-hoc τ re-run.

Two things live here:

**Calibration diagnostics**, is ``sigmoid(logit_margin)`` a trustworthy probability?
ECE and the reliability bins are lifted from ``scripts/analyse_calibration.py`` (same binning,
so the numbers agree with what that script prints), plus Brier and the ROC-AUC of confidence
against ground truth.

**τ re-run**, re-deciding a run at a different (τ_accept, τ_prune) without touching a GPU,
by re-running ``tree_pruning.apply_pruning_verdict`` over the cached per-node confidences.

A soundness constraint governs the second, and the UI enforces it rather than papering over it:

    apply_pruning_verdict re-runs from a *full, unpruned* confidence table. A run that pruned
    has no confidence for the nodes it cut, they were never asked. Lowering τ_prune below the
    run's own value would ask us to score those nodes, and ``node_conf.get(hpo, 0.0)`` would
    quietly answer 0.0 for every one of them: a fabricated wall of confident absence.

So τ_prune is only a free variable when the run carries a full unpruned confidence table, ``node_conf_by_patient.pkl`` (an earlier exploratory run's whole purpose) or a run that pruned nothing. On any
other run, τ_prune is fixed to the value the run actually used, where the re-run *is* exact
(the same nodes prune, the same nodes are evaluated), and only τ_accept moves.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hpo_extraction.evaluation.tree_pruning import apply_pruning_verdict, build_direct_parents_map, topo_sort

from . import scoring


def sigmoid(x: float) -> float:
    """1 / (1 + exp(-x)), a value between 0 and 1."""
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


# ── calibration diagnostics ──────────────────────────────────────────────────

def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error, equal-width bins. Same binning as analyse_calibration.py."""
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if hi == 1.0:
            mask |= confidences == 1.0
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(confidences[mask].mean() - labels[mask].mean())
    return float(ece)


def reliability_bins(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per-bin (centre, empirical positive rate, count) for the reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if hi == 1.0:
            mask |= confidences == 1.0
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "centre": float((lo + hi) / 2),
            "accuracy": float(labels[mask].mean()),
            "confidence": float(confidences[mask].mean()),
            "count": n,
        })
    return out


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney ROC-AUC, P(a positive scores above a negative). Ties count as half."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks within ties so ties contribute 0.5, not an arbitrary order.
    values = np.concatenate([pos, neg])
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum_pos = ranks[: len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibration_summary(nodes: pd.DataFrame, n_bins: int = 10) -> dict | None:
    """ECE / Brier / AUC / reliability over every *evaluated* node that carries a confidence.

    Scored at node level against the raw GT membership (``in_gt``), the same unit the tree
    metrics use, and the unit τ_accept actually thresholds.
    """
    if nodes.empty or "confidence" not in nodes:
        return None
    scored = nodes[(nodes["state"] == "evaluated") & nodes["confidence"].notna()]
    if scored.empty:
        return None

    conf = scored["confidence"].to_numpy(dtype=float)
    labels = scored["in_gt"].to_numpy(dtype=float)

    return {
        "n_samples": int(len(conf)),
        "prevalence": float(labels.mean()),
        "brier": float(np.mean((conf - labels) ** 2)),
        "ece": compute_ece(conf, labels, n_bins),
        "auc": roc_auc(conf, labels),
        "reliability_bins": reliability_bins(conf, labels, n_bins),
        "confidence_positive": conf[labels == 1].tolist(),
        "confidence_negative": conf[labels == 0].tolist(),
        "margin_positive": scored.loc[labels == 1, "max_margin"].dropna().tolist(),
        "margin_negative": scored.loc[labels == 0, "max_margin"].dropna().tolist(),
    }


# ── τ re-run ─────────────────────────────────────────────────────────────────

def build_graph(nodes: pd.DataFrame, tree) -> dict:
    """The traversal graph the run actually used, recovered from the nodes it touched.

    Reconstructed from the run's own artifacts, not re-deriving it from the config, so
    the re-run walks the node set the run walked.
    """
    target_set = set(nodes.loc[nodes["state"] != "not_traversed", "hpo_id"]) & set(tree.data)
    order, _ = topo_sort(target_set, tree)
    return {
        "target_set": target_set,
        "topo_order": order,
        "direct_parents_map": build_direct_parents_map(target_set, tree),
        "depth_skip_set": set(nodes.loc[nodes["state"] == "skipped", "hpo_id"]),
    }


def node_conf_table(raw: dict, nodes: pd.DataFrame) -> dict[str, dict[str, float]]:
    """{patient: {hpo: confidence}}, preferring the run's own unpruned table when it has one."""
    if raw.get("node_conf"):
        return raw["node_conf"]
    scored = nodes[nodes["confidence"].notna()]
    table: dict[str, dict[str, float]] = {}
    for row in scored.itertuples(index=False):
        table.setdefault(row.patient_id, {})[row.hpo_id] = float(row.confidence)
    return table


def conf_table_is_unpruned(raw: dict, nodes: pd.DataFrame) -> bool:
    """Whether τ_prune can be varied without fabricating confidences for un-asked nodes."""
    if raw.get("node_conf"):
        return True
    return not nodes.empty and not (nodes["state"] == "pruned").any()


def replay(
    node_conf: dict[str, dict[str, float]],
    graph: dict,
    gt_dict: dict[str, list[str]],
    target_symptoms: list[str],
    tree,
    tau_accept: float,
    tau_prune: float,
) -> dict:
    """Re-decide every patient at (τ_accept, τ_prune) and re-score. No GPU, no re-inference.

    Returns the target-level metrics plus the node counts, so the scorecard and the
    traversal panel can both be driven from one call.
    """
    response_dict: dict = {}
    n_pruned = n_positive = n_evaluated = 0
    for patient, conf in node_conf.items():
        resp, node_pred = apply_pruning_verdict(
            graph["topo_order"], graph["direct_parents_map"], graph["depth_skip_set"],
            conf, tau_accept, tau_prune,
        )
        response_dict[patient] = resp
        n_pruned += sum(1 for v in node_pred.values() if v == "pruned")
        n_positive += sum(1 for v in node_pred.values() if v == "yes")
        n_evaluated += sum(1 for v in node_pred.values() if v in ("yes", "no"))

    metrics = scoring.evaluate_targets(response_dict, target_symptoms, gt_dict, tree)
    n_asked = n_evaluated + n_pruned
    return {
        "tau_accept": round(float(tau_accept), 4),
        "tau_prune": round(float(tau_prune), 4),
        "n_pruned": n_pruned,
        "n_positive": n_positive,
        "n_evaluated": n_evaluated,
        "llm_call_savings_pct": (100.0 * n_pruned / n_asked) if n_asked else 0.0,
        **metrics,
    }


def offline_accept_only(
    nodes: pd.DataFrame,
    gt_dict: dict[str, list[str]],
    target_symptoms: list[str],
    tree,
    tau_accept: float,
) -> dict:
    """Re-threshold τ_accept while holding the run's actual prune decisions fixed.

    This is the sound move for a run that pruned. ``should_prune_confident`` consults only
    τ_prune and the confidences, never τ_accept, so varying τ_accept alone cannot change which
    nodes were cut. Every node the run evaluated keeps its confidence and is simply re-decided,
    and every node it pruned or skipped stays pruned or skipped. No confidence is invented for a
    node that was never asked, which is the trap a full re-run would fall into here.
    """
    response_dict: dict = {}
    n_pruned = n_positive = n_evaluated = 0
    for row in nodes.itertuples(index=False):
        entry = response_dict.setdefault(row.patient_id, {})
        if row.state == "pruned":
            entry[row.hpo_id] = {0: {"response": "PRUNED"}}
            n_pruned += 1
        elif row.state == "skipped":
            entry[row.hpo_id] = {0: {"response": "SKIPPED"}}
        elif row.state == "not_traversed":
            continue
        else:
            conf = row.confidence if row.confidence is not None else 0.0
            positive = (conf >= tau_accept) if pd.notna(conf) else False
            entry[row.hpo_id] = {0: {"response": "Yes" if positive else "No"}}
            n_evaluated += 1
            n_positive += int(positive)

    metrics = scoring.evaluate_targets(response_dict, target_symptoms, gt_dict, tree)
    n_asked = n_evaluated + n_pruned
    return {
        "tau_accept": round(float(tau_accept), 4),
        "tau_prune": None,
        "n_pruned": n_pruned,
        "n_positive": n_positive,
        "n_evaluated": n_evaluated,
        "llm_call_savings_pct": (100.0 * n_pruned / n_asked) if n_asked else 0.0,
        **metrics,
    }


def offline_grid(
    node_conf, graph, gt_dict, target_symptoms, tree,
    tau_accept_grid, tau_prune_grid,
) -> list[dict]:
    """The (τ_accept, τ_prune) landscape, computed on demand for runs with no tau_sweep.csv."""
    return [
        replay(node_conf, graph, gt_dict, target_symptoms, tree, ta, tp)
        for ta in tau_accept_grid
        for tp in tau_prune_grid
    ]


def summarize_grid(rows: list[dict], precision_floor: float = 0.9) -> dict:
    """The F1-optimal point and the max-recall point that still clears a precision floor."""
    if not rows:
        return {}
    best_f1 = max(rows, key=lambda r: r.get("micro_f1", 0.0))
    eligible = [r for r in rows if r.get("micro_precision", 0.0) >= precision_floor]
    return {
        "best_f1": best_f1,
        "precision_floor": precision_floor,
        "best_recall_at_floor": max(eligible, key=lambda r: r["micro_recall"]) if eligible else None,
    }
