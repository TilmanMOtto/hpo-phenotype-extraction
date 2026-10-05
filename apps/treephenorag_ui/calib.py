"""Are the two scores worth thresholding? Calibration and risk--coverage, one per score.

The earlier runs'tree runs carry **two** node scores with two thresholds, and they answer different
questions, so they need two different labels:

``accept_score`` → **node presence**
    "Is this term itself present in the report?" Label: the node is in the report's ground-truth set.
    Thresholded by τ_accept, which decides what is reported.

``prune_score`` → **subtree presence**
    "Is anything worth finding below this node?" Label: some annotated term of the report lies in the
    subtree rooted here. Thresholded by τ_prune, which decides what is explored.

Scoring ``prune_score`` against node presence is the mistake ``result_tables/loaders.py:340`` warns
about at length, and it is not a small one: an internal node like *Abnormality of the nervous
system* is almost never itself annotated, so a gate that correctly assigns it high subtree
probability would be marked confidently wrong at every one of those nodes, and the reliability
diagram would show a systematic overconfidence that does not exist. :func:`sample` takes the
label as an argument for this reason, and :func:`summary` names which one it used.

Nodes that were never scored, the synthetic rows :mod:`~apps.treephenorag_ui.nodes` adds for annotated terms
the traversal never reached, are excluded. They have no confidence, and imputing 0.0 for them
would enter them as confidently-absent predictions that the model never actually made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ``(score column, label column, what the score is predicting)``.
SCORES: dict[str, tuple[str, str, str]] = {
    "accept": ("accept_score", "in_gold", "node presence, is this term in the report?"),
    "prune": ("prune_score", "subtree_has_gold",
              "subtree presence, is any annotated term below this node?"),
}


def sample(frame: pd.DataFrame, score: str) -> dict:
    """Aligned ``confidences``/``labels`` arrays for one score, over scored nodes only."""
    if score not in SCORES:
        raise ValueError(f"score must be one of {sorted(SCORES)}, got {score!r}")
    score_column, label_column, meaning = SCORES[score]

    if frame is None or frame.empty or score_column not in frame.columns:
        return {"confidences": np.array([]), "labels": np.array([]), "n": 0,
                "score": score, "score_column": score_column,
                "label_column": label_column, "meaning": meaning, "index": pd.Index([])}

    mask = frame[score_column].notna() & (frame["state"] == "visited")
    subset = frame.loc[mask]
    return {
        "confidences": subset[score_column].to_numpy(dtype=float),
        "labels": subset[label_column].fillna(False).to_numpy(dtype=bool).astype(int),
        "n": int(len(subset)),
        "score": score,
        "score_column": score_column,
        "label_column": label_column,
        "meaning": meaning,
        "index": subset.index,
    }


def roc_auc(scores, labels) -> float | None:
    """Mann--Whitney AUROC, ties counted at 0.5.

    ``thesis_metrics`` reports AURC (risk--coverage) but not AUROC, and the two answer different
    questions: AURC folds calibration and ranking together, AUROC isolates ranking. On the earlier tree runs the interesting failure is a ranking one, the score barely separates the classes, so the scorecard needs the isolated number.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    # Average the ranks inside each tie group, which is what makes ties score 0.5.
    values = np.concatenate([positive, negative])[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or values[index] != values[start]:
            ranks[order[start:index]] = (start + index + 1) / 2.0
            start = index
    rank_sum = ranks[:positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2.0)
                 / (positive.size * negative.size))


def summary(frame: pd.DataFrame, score: str, *, n_bins: int = 15) -> dict | None:
    """ECE, Brier, AUROC, reliability bins and the risk--coverage scalars for one score.

    ``None`` when the sample has no positives or no negatives: every one of these metrics is
    undefined there, and a table of zeros would read as "perfectly calibrated" rather than as
    "not measurable on this cell".
    """
    from hpo_extraction.evaluation.metrics import calibration, selective

    data = sample(frame, score)
    labels = data["labels"]
    if data["n"] == 0 or labels.min() == labels.max():
        return None

    confidences = data["confidences"]
    report = dict(calibration.calibration_report(confidences, labels, n_bins=n_bins))
    # ``calibration_report`` and ``selective_report`` both emit ``n`` and ``prevalence`` with the
    # same values, so the overwrite is a no-op. Nothing else overlaps.
    report.update(selective.selective_report(confidences, labels))
    report.update({
        "auroc": roc_auc(confidences, labels),
        "score": score,
        "score_column": data["score_column"],
        "label_column": data["label_column"],
        "meaning": data["meaning"],
        "n": data["n"],
        "prevalence": float(labels.mean()),
        "bins": calibration.bin_summary(confidences, labels, n_bins=n_bins,
                                        binning="equal_mass"),
        "mean_confidence_positive": float(confidences[labels == 1].mean()),
        "mean_confidence_negative": float(confidences[labels == 0].mean()),
    })
    return report


def threshold_curve(frame: pd.DataFrame, score: str, grid=None) -> list[dict]:
    """How many nodes clear each candidate threshold, and how many of them are labelled positive.

    This is the "would a different τ_accept have helped" question in its most direct form: at each
    threshold, the positives above it are the terms that would be reported correctly and the
    negatives above it the ones that would be reported wrongly. It is exact for τ_accept, that
    threshold never influenced which nodes were scored, and it is *not* exact for τ_prune, which
    is why only the accept curve is offered as a live control. Changing τ_prune changes which
    nodes exist at all, and the run holds no score for the ones it never visited. The τ_prune
    frontier is read from the configurations the sweep actually materialised on disk instead.
    """
    data = sample(frame, score)
    if data["n"] == 0:
        return []
    if grid is None:
        grid = np.round(np.arange(0.0, 1.001, 0.05), 3)

    confidences, labels = data["confidences"], data["labels"]
    total_positive = int(labels.sum())
    rows = []
    for tau in grid:
        above = confidences >= tau
        n_above = int(above.sum())
        tp = int(labels[above].sum())
        rows.append({
            "tau": float(tau),
            "n_above": n_above,
            "tp": tp,
            "fp": n_above - tp,
            "precision": tp / n_above if n_above else None,
            "recall": tp / total_positive if total_positive else None,
        })
    return rows


def score_separation(frame: pd.DataFrame, score: str, *, n_bins: int = 20) -> dict:
    """Two histograms of the same score, split by label, the readable form of AUROC.

    A reliability diagram says whether the number means what it claims. This says whether the two
    populations are separable at all. On the earlier tree runs they are largely not, and that is a
    more actionable finding than any single scalar.
    """
    data = sample(frame, score)
    if data["n"] == 0:
        return {"edges": [], "positive": [], "negative": [], "n": 0}
    confidences, labels = data["confidences"], data["labels"]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    positive, _ = np.histogram(confidences[labels == 1], bins=edges)
    negative, _ = np.histogram(confidences[labels == 0], bins=edges)
    return {
        "edges": [float(e) for e in edges],
        "positive": [int(v) for v in positive],
        "negative": [int(v) for v in negative],
        "n": data["n"],
    }
