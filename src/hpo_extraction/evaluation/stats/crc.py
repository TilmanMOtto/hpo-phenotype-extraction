r"""Conformal risk control for the pruning threshold, a guarantee instead of an oracle.

Today :math:`\tau_{\text{prune}}` is read off the sweep at "best micro :math:`F_1` on this cohort",
which is an oracle: the threshold is chosen by looking at the answer. A results chapter cannot
report that as a finding, because nothing says the same threshold would be chosen, or would work, on a patient the method has not seen.

Conformal risk control replaces it. Pruning is a **constraint**, not an objective: what the
traversal owes is coverage of the annotated terms, and among the thresholds that deliver it we want the
cheapest. So fix a tolerance :math:`\alpha` on the per-report miss rate, and pick

.. math::
    \hat{\lambda} = \inf\Big\{ \lambda : \tfrac{n}{n+1}\hat{R}_n(\lambda) + \tfrac{1}{n+1}
    \le \alpha \Big\},

where :math:`\lambda = 1 - \tau_{\text{prune}}` and :math:`\hat{R}_n` is the mean loss over the
:math:`n` calibration reports. This guarantees :math:`\mathbb{E}[L_{n+1}(\hat{\lambda})] \le \alpha`
for a new report exchangeable with them, distribution-free, with no assumption about the score.

The loss is the fraction of a report's annotated terms the traversal never **scored**:

.. math::
    L_i(\tau) = \frac{|Y_i \setminus \mathcal{C}_i(\tau)|}{|Y_i|}, \qquad L_i = 0 \text{ if }
    Y_i = \emptyset .

Four things a caller has to keep in view, three of them limits on what the guarantee says:

* **It bounds the expected per-report miss fraction**, not the micro-averaged miss rate over terms.
  A cohort where a few heavily annotated patients fail badly can satisfy the bound while losing more
  terms than it suggests.
* **Exchangeability is an assumption.** Thirty years of documentation from one hospital is not
  obviously exchangeable with a new referral.
* **Monotonicity is required**, and it holds here: :math:`\mathcal{C}_i` only grows as
  :math:`\tau` falls, so :math:`L_i` is non-increasing in :math:`\lambda`. Any node budget, or an
  :math:`S` that varied per term, would break it.
* **The bound can be unreachable.** If even the most permissive cached threshold leaves more than
  :math:`\alpha` of the ground truth unscored, no valid :math:`\hat{\lambda}` exists, the ceiling is set by
  retrieval and the verifier, not by the pruning rule. :func:`crc_lambda` reports that case rather
  than returning a threshold that does not hold, and the attained ceiling is the honest number to
  print beside it.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def miss_rate(gold: Iterable[str], scored: Iterable[str]) -> float:
    """:math:`L_i` for one report: the share of its annotated terms that were never scored.

    A report with no annotated terms contributes ``0.0``, it cannot lose what it does not have, and
    excluding it instead would change the calibration-set size in a way the bound's ``1/(n+1)``
    term is not accounting for.
    """
    gold = set(gold)
    if not gold:
        return 0.0
    return len(gold - set(scored)) / len(gold)


def loss_curve(
    gold_by_report: Mapping[str, set],
    scored_by_report_and_tau: Mapping[float, Mapping[str, set]],
    report_ids: Sequence[str] | None = None,
) -> tuple[list[float], np.ndarray]:
    """``(taus, losses)`` with ``losses[i, t]`` the loss of report *i* at the *t*-th threshold.

    Thresholds come back sorted **ascending**, so that moving right along the array is moving to a
    stricter threshold and therefore to a non-decreasing loss, the orientation
    :func:`crc_lambda` scans in.
    """
    taus = sorted(scored_by_report_and_tau)
    rids = list(report_ids if report_ids is not None else sorted(gold_by_report))
    losses = np.empty((len(rids), len(taus)), dtype=np.float64)
    for t, tau in enumerate(taus):
        scored = scored_by_report_and_tau[tau]
        for i, rid in enumerate(rids):
            losses[i, t] = miss_rate(gold_by_report.get(rid, set()), scored.get(rid, set()))
    return taus, losses


def crc_lambda(
    taus: Sequence[float],
    losses: np.ndarray,
    alpha: float,
) -> dict:
    """The largest (cheapest) threshold whose conformal bound holds, or the attained ceiling.

    Args:
        taus: candidate thresholds, ascending. Drawn from the cache's own distinct expansion
            scores, which is what makes the grid fine enough to invert a risk bound at all, a
            five-point sweep is not.
        losses: ``(n_reports, n_taus)``, from :func:`loss_curve`.
        alpha: the tolerance on the expected per-report miss rate.

    Returns:
        ``tau`` (``None`` when the bound is unreachable), ``bound`` at that threshold, ``feasible``,
        ``empirical_risk``, and, always, ``attained_ceiling``, the smallest bound available
        anywhere in the grid. When ``feasible`` is False, that ceiling is the number to report:
        it says what the pipeline can actually promise, and the gap to ``alpha`` is a statement
        about retrieval and the verifier, not about the threshold.
    """
    n = losses.shape[0]
    if n == 0:
        raise ValueError("no calibration reports")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    empirical = losses.mean(axis=0)
    bounds = (n / (n + 1)) * empirical + 1 / (n + 1)

    feasible_idx = [t for t in range(len(taus)) if bounds[t] <= alpha]
    ceiling_idx = int(np.argmin(bounds))

    if not feasible_idx:
        return {
            "tau": None,
            "feasible": False,
            "bound": float("nan"),
            "empirical_risk": float("nan"),
            "alpha": float(alpha),
            "attained_ceiling": float(bounds[ceiling_idx]),
            "attained_ceiling_tau": float(taus[ceiling_idx]),
            "n_calibration": n,
        }

    # Cheapest feasible threshold = the largest tau, since a stricter threshold scores fewer nodes.
    chosen = max(feasible_idx)
    return {
        "tau": float(taus[chosen]),
        "feasible": True,
        "bound": float(bounds[chosen]),
        "empirical_risk": float(empirical[chosen]),
        "alpha": float(alpha),
        "attained_ceiling": float(bounds[ceiling_idx]),
        "attained_ceiling_tau": float(taus[ceiling_idx]),
        "n_calibration": n,
    }


def held_out_risk(
    gold_by_report: Mapping[str, set],
    scored_by_report: Mapping[str, set],
    report_ids: Sequence[str],
) -> dict:
    """The empirical miss rate on reports the threshold was *not* chosen on.

    This is the number that makes the guarantee checkable, not merely asserted: the bound
    promises ``alpha``. This says what actually happened on the outer fold. Both belong in the
    table, side by side.
    """
    values = [miss_rate(gold_by_report.get(r, set()), scored_by_report.get(r, set()))
              for r in report_ids]
    if not values:
        return {"mean_loss": float("nan"), "n_reports": 0, "n_reports_with_gold": 0}
    with_gold = [r for r in report_ids if gold_by_report.get(r)]
    return {
        "mean_loss": float(np.mean(values)),
        "n_reports": len(values),
        "n_reports_with_gold": len(with_gold),
    }
