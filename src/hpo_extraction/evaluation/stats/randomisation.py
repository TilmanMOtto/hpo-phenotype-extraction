r"""Paired approximate randomisation, the significance test for a difference between two systems.

The question every results table raises and none of them currently answer: PhenoBERT scores 0.653
micro-F1 on GSC and the ensemble 0.637. Is 0.016 a difference?

The test is nonparametric and makes no distributional assumption. Under the null, the two systems
are interchangeable, so *for each report* their two outputs could equally well have been swapped.
Draw many such swap patterns, recompute the metric difference under each, and ask how often the
shuffled difference is at least as extreme as the observed one.

Properties of this setting the test relies on:

* **The report is the unit**, for the same reason as in the bootstrap: cells within a report are
  correlated, and swapping whole reports respects that.
* **It is exact for the metric as defined**, including micro-F1, which is a ratio of pooled counts
  and therefore not a mean of per-report quantities. A paired t-test would be wrong here for
  that reason. Randomisation does not care what shape the statistic has.
* **The p-value adds one to both counts**, ``(1 + b) / (1 + B)``, which keeps it strictly
  positive and is the standard correction for estimating a tail by sampling. With B = 10 000 the
  floor is 1e-4, so a table should print ``< 0.001`` rather than ``0``.

Systems must be **aligned on the same report list in the same order**. A silent misalignment does
not raise, it produces a plausible-looking p-value about nothing, so the alignment is asserted here,
not trusted.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


def paired_randomisation(
    gold_sets: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric_fn: Callable[[Sequence, Sequence], Mapping[str, float]],
    *,
    statistic: str = "micro_f1",
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided paired randomisation test on one statistic.

    Args:
        gold_sets: per-report ground truth, shared by both systems by design.
        pred_a, pred_b: per-report predictions, aligned to ``gold_sets``.
        metric_fn: ``(gold_sets, pred_sets) -> {name: value}``.
        statistic: which key of ``metric_fn``'s output the test is about. The thesis restricts
            testing to the pre-specified primary metric. Everything else is reported with intervals
            and read descriptively, which is what keeps the family size honest.

    Returns:
        ``delta`` (A minus B, positive means A is ahead), ``p_value``, ``n_permutations``,
        ``n_reports``, and both systems' point values.
    """
    n = len(gold_sets)
    if not (len(pred_a) == len(pred_b) == n):
        raise ValueError(
            f"systems are not aligned: {n} gold, {len(pred_a)} A, {len(pred_b)} B"
        )
    if n == 0:
        raise ValueError("no reports to test")

    value_a = float(metric_fn(gold_sets, pred_a)[statistic])
    value_b = float(metric_fn(gold_sets, pred_b)[statistic])
    delta = value_a - value_b

    rng = np.random.default_rng(seed)
    swaps = rng.random((n_permutations, n)) < 0.5

    at_least_as_extreme = 0
    observed = abs(delta)
    for row in swaps:
        shuffled_a = [pred_b[i] if row[i] else pred_a[i] for i in range(n)]
        shuffled_b = [pred_a[i] if row[i] else pred_b[i] for i in range(n)]
        d = float(metric_fn(gold_sets, shuffled_a)[statistic]) - float(
            metric_fn(gold_sets, shuffled_b)[statistic]
        )
        if abs(d) >= observed - 1e-12:
            at_least_as_extreme += 1

    return {
        "value_a": value_a,
        "value_b": value_b,
        "delta": delta,
        "p_value": (1 + at_least_as_extreme) / (1 + n_permutations),
        "n_permutations": int(n_permutations),
        "n_reports": n,
        "statistic": statistic,
    }
