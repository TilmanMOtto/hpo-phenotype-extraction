"""Pruning / acceptance score functions for the TreePhenoRAG live traversal (the earlier runs).

A traversal node carries two decisions, on two *different* scores (earlier design):

* **prune_score**, P(the subtree rooted here contains an annotated term). Gates whether the
  traversal expands the node's children (``>= tau_prune``). This is a *subtree-present*
  quantity: at an internal node on the path to a ground truth leaf, the node's own label is usually
  never stated, so ``node_present`` is ~0 while ``subtree_present`` is 1.
* **accept_score**, P(this exact term is present). Gates whether the node is reported
  (``>= tau_accept``). A *node-present* quantity, the AnyYes-soft aggregation prior work used.

Three prune-score rules are provided. All AUPRCs below are from **an earlier exploratory run**, which re-fit the
proxies on the full 326-term HCY ground truth (base rate 0.0365) rather than earlier exploratory run's 9-target
projection (base rate 0.0105). The an earlier exploratory run numbers this docstring used to quote are not
comparable across that label change.

* :class:`LRGate`, the calibrated 4-feature logistic gate (per-cell AUPRC 0.456, node-held-out
  0.451). Combines the SLM signal (``margin_max``) with two static tree features (``depth_long``,
  ``log1p(subtree_size)``). Loads ``minimal_gate.json``, now the an earlier exploratory run refit. **Its edge over
  the training-free rules is only ~1.13× under the corrected ground truth** (it was ~1.8× under the
  projection): an earlier exploratory run showed the tree-geometry half was fitting the target list, not the
  ontology, ``depth_long`` lost 97 % of its permutation importance.
* :func:`lse_beta_prob`, ``sigmoid(lse_beta(margins))``, the **best training-free rule**
  (AUPRC 0.398 at beta=1). The high-recall end of the fixed-rule tradeoff, which is what counts
  when reachability binds.
* :func:`noisy_or`, the previous best training-free rule (AUPRC 0.381). Plateaus at ~0.85 re-run
  recall, so it trades recall for cost against ``lse_beta_prob``.

Neither fixed rule is calibrated, so ``tau_prune`` against them is an empirical configuration
read off the recall-vs-cost curve, not a probability.

All functions are pure and operate on plain numpy / floats so they unit-test in isolation and
re-run over a dump at zero GPU cost. ``margins`` is always ``logit_yes - logit_no`` per retrieved
sentence; ``S`` varies per cell (a report with fewer than ``top_n`` distinct sentences yields a
short array), so every function reduces over whatever length it is given.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Repo-default gate artifact (committed copy of the earlier analysis output).
from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

DEFAULT_GATE_PATH = _REPO_ROOT / "resources" / "util" / "minimal_gate.json"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def noisy_or(margins) -> float:
    """``1 - Π_s (1 - σ(margin_s))``, soft "at least one sentence says Yes".

    Uses all S sentences (unlike AnyYes/``margin_max``, which uses only the best). Not
    calibrated, the score is pushed toward 1 as S grows, so ``tau_prune`` against it is an
    empirical configuration read off the recall-vs-cost curve, not a probability.
    """
    m = np.asarray(margins, dtype=np.float64)
    if m.size == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - _sigmoid(m)))


def lse_beta(margins, beta: float = 1.0) -> float:
    """``logsumexp(beta * margins) / beta``, a smooth max whose sharpness is set by ``beta``.

    Sits in log-odds space, not ``[0, 1]``: it interpolates between the sum of the margins
    (``beta -> 0``) and their max (``beta -> inf``), so at ``beta = 1`` it pools evidence from all
    S sentences while staying dominated by the strongest. That pooling is why it keeps rising
    where :func:`noisy_or` plateaus.

    Kept numerically identical to ``an earlier exploratory run/proxies.py:lse_beta``, the live
    rule and the offline proxy must not drift, or the an earlier exploratory run leaderboard stops describing what
    the traversal actually does.
    """
    m = np.asarray(margins, dtype=np.float64) * float(beta)
    if m.size == 0:
        return 0.0
    mx = m.max()
    return float((mx + np.log(np.exp(m - mx).sum())) / beta)


def lse_beta_prob(margins, beta: float = 1.0) -> float:
    """``sigmoid(lse_beta(margins, beta))``, the same rule squashed into ``[0, 1]``.

    ``sigmoid`` is monotone, so the ranking, and every AUPRC / recall number an earlier exploratory run reports for
    ``lse_beta1``, is unchanged. The squash exists so ``tau_prune`` lives on the same scale as the
    other prune scores and the calibration analysis can treat every ``prune_score`` column
    uniformly. The result is *not* calibrated. See the module docstring.
    """
    m = np.asarray(margins, dtype=np.float64)
    if m.size == 0:
        return 0.0
    return float(_sigmoid(lse_beta(m, beta)))


def accept_confidence(margins) -> float:
    """``max_s σ(margin_s)``, node-present confidence (AnyYes-soft), the accept score.

    P(Yes | {Yes, No}) of the single most-confident retrieved sentence. Matches the
    ``conf = max sigmoid(margin)`` aggregation used by an earlier exploratory run's node verdict.
    """
    m = np.asarray(margins, dtype=np.float64)
    if m.size == 0:
        return 0.0
    return float(np.max(_sigmoid(m)))


def noisy_or_kappa(margins, kappa: float = 1.0) -> float:
    """``1 - prod_s (1 - sigma(margin_s))^(1/kappa)``, noisy-OR with a tunable independence count.

    :func:`noisy_or` is the ``kappa = 1`` case. The exponent turns the hidden independence
    assumption into a parameter: ``S/kappa`` acts as an *effective* number of independent segments,
    so ``kappa = S`` treats all S retrieved sentences as one observation and ``kappa = 1`` treats
    them as S separate ones. Since clinical records repeat content across segments, the truth is
    somewhere between, and the thesis sweeps it, not assuming either end.

    Decreasing in ``kappa``: ``(1-p)^(1/kappa)`` rises with ``kappa``, so the product rises and the
    complement falls. ``kappa = 1`` is therefore the **maximum** over the family, which is what
    makes it the right rule to build a re-run cache with.
    """
    m = np.asarray(margins, dtype=np.float64)
    if m.size == 0:
        return 0.0
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    return float(1.0 - np.prod((1.0 - _sigmoid(m)) ** (1.0 / float(kappa))))


def second_largest_confidence(margins) -> float:
    """``sigma(margin)`` of the *second* most confident segment, 0 when only one was retrieved.

    The thesis's P2. Where P1 (:func:`accept_confidence`) accepts on a single affirmative segment,
    this requires two, so one spuriously confident sentence cannot carry a term on its own. The
    cost is recall on phenotypes a report mentions once, which on these cohorts is most of
    them, it is included as the reliability end of the trade-off, not as a candidate to win.
    """
    m = np.asarray(margins, dtype=np.float64)
    if m.size < 2:
        return 0.0
    return float(np.sort(_sigmoid(m))[-2])


def mean_confidence(margins) -> float:
    """``mean_s sigma(margin_s)``, the thesis's P4, included as a negative control.

    Averaging is the wrong shape for this problem: evidence for a phenotype in one segment is not
    weakened by ten other segments failing to mention it, but the mean says it is, and the
    penalty grows with report length. It is reported so that the pooling table contains a rule
    expected to fail for a stated reason.
    """
    m = np.asarray(margins, dtype=np.float64)
    if m.size == 0:
        return 0.0
    return float(np.mean(_sigmoid(m)))


def margin_max(margins) -> float:
    """``max_s (logit_yes - logit_no)``, the SLM feature the LR gate consumes."""
    m = np.asarray(margins, dtype=np.float64)
    return float(np.max(m)) if m.size else 0.0


class LRGate:
    """Calibrated 4-feature logistic pruning gate (the earlier runs ``minimal_gate.json``).

    ``p_expand(node_meta, margins) = σ(intercept + Σ_f w_f · x_f)`` with features

        depth_long          longest is-a path from HP:0000118 (imputed to the median with
                            ``depth_long_missing = 1`` when the node is outside the
                            phenotypic-abnormality subtree, i.e. ``node_meta['depth_long']`` is
                            None),
        log1p_subtree_size  log1p(transitive descendant count),
        margin_max          max_s(logit_yes - logit_no) over the S retrieved sentences,
        depth_long_missing  1 if depth_long was imputed else 0.

    The three static features are read from a ``node_metadata``-style dict. Only ``margin_max``
    is per-report. ``calibrated=True`` uses the Platt-folded weight set (the default and what
    makes ``tau_prune`` a probability). The raw set is available for the ablation.
    """

    def __init__(self, gate_json_path: str | Path = DEFAULT_GATE_PATH, calibrated: bool = True):
        spec = json.loads(Path(gate_json_path).read_text(encoding="utf-8"))
        block = "deployable_calibrated" if calibrated else "deployable_uncalibrated"
        if block not in spec:  # older/partial artifact: fall back to whichever set exists
            block = "deployable_calibrated" if "deployable_calibrated" in spec else "deployable_uncalibrated"
        self.features: list[str] = list(spec["features"])
        self.weights: dict[str, float] = {k: float(v) for k, v in spec[block]["weights"].items()}
        self.intercept: float = float(spec[block]["intercept"])
        self.impute_median: dict[str, float] = {
            k: float(v) for k, v in spec.get("impute_median", {}).items()
        }
        self.gate_json_path = str(gate_json_path)
        self.calibrated = calibrated

    def features_for(self, node_meta: dict, margins) -> dict[str, float]:
        """Assemble the raw feature vector for one (node, report) cell.

        ``node_meta`` is a :func:`hpo_extraction.treephenorag.node_metadata.node_metadata` dict; ``depth_long`` is None
        for nodes outside the phenotypic-abnormality subtree, which triggers imputation and the
        missing flag.
        """
        depth_long = node_meta.get("depth_long")
        missing = depth_long is None
        if missing:
            depth_long = self.impute_median.get("depth_long", 5.0)
        subtree_size = node_meta.get("subtree_size") or 0
        return {
            "depth_long": float(depth_long),
            "log1p_subtree_size": float(np.log1p(subtree_size)),
            "margin_max": margin_max(margins),
            "depth_long_missing": 1.0 if missing else 0.0,
        }

    def p_expand(self, node_meta: dict, margins) -> float:
        """Calibrated P(subtree contains an annotated term) for one (node, report) cell."""
        feats = self.features_for(node_meta, margins)
        z = self.intercept + sum(self.weights.get(f, 0.0) * feats[f] for f in self.features)
        return float(_sigmoid(z))
