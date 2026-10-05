r"""Report-level nonparametric bootstrap, the interval every number in the thesis carries.

The sampling unit is the **report**, never the (report, term) cell. Terms within one report are
strongly correlated: a patient with a severe neurological presentation supplies a dozen annotated terms
that stand or fall together, so treating each cell as independent would claim thousands of
independent observations where there are 118 patients. Resampling whole reports is what makes the
interval reflect the cohort's real size.

**Every metric is recomputed on every resample**, from the resampled per-report units rather than
from pooled counts. That is not a stylistic choice: label-macro averages over the *set of terms
present in the resample*, which changes from draw to draw, so it cannot be recovered from a pooled
(TP, FP, FN) triple. Micro could be. Doing both the same way keeps one code path.

**Draw the indices once.** :class:`ReportResampler` holds an ``(n_resamples, n_reports)`` index
matrix, and the same instance should be reused across every method and every metric on a cohort.
Two reasons, and the second counts more than the speed:

* the 10 000 draws are paid once instead of once per (method, metric) cell;
* every system's interval then comes from the *identical* resamples, which is what makes a
  difference of intervals meaningful and is required for the paired comparisons in
  :mod:`~hpo_extraction.evaluation.stats.randomisation` to line up with the intervals reported beside them.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

#: What a metric function must look like: a sequence of per-report units in, a flat mapping of
#: metric name to scalar out. ``flat_report``-shaped callables satisfy this after a thin adapter.
MetricFn = Callable[[Sequence], Mapping[str, float]]


class ReportResampler:
    """A fixed matrix of report indices, shared across every cell scored on one cohort.

    Args:
        n_reports: the cohort size. Every unit sequence passed to :meth:`resample` must match it,
            which is checked, a silent length mismatch would resample a different cohort than the
            one the caller believes it is scoring.
        n_resamples: number of bootstrap draws.
        seed: fixes the matrix. Two runs with the same seed produce byte-identical intervals.
    """

    def __init__(self, n_reports: int, n_resamples: int = 10_000, seed: int = 0):
        if n_reports <= 0:
            raise ValueError(f"n_reports must be positive, got {n_reports}")
        if n_resamples <= 0:
            raise ValueError(f"n_resamples must be positive, got {n_resamples}")
        self.n_reports = int(n_reports)
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.indices: np.ndarray = rng.integers(
            0, self.n_reports, size=(self.n_resamples, self.n_reports), dtype=np.int64
        )

    def resample(self, units: Sequence) -> list:
        """Yield each draw's resampled unit list, in matrix order."""
        if len(units) != self.n_reports:
            raise ValueError(
                f"resampler built for {self.n_reports} reports, got {len(units)} units"
            )
        for row in self.indices:
            yield [units[i] for i in row]


def percentile_ci(values: Sequence[float], ci: float = 0.95) -> tuple[float, float]:
    """The percentile interval, ignoring draws where the metric was undefined (NaN).

    The percentile method, not BCa: it is what the thesis specifies, it needs no jackknife
    over 118 reports per cell, and with 10 000 draws the difference is well inside the reporting
    precision.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    half = (1.0 - ci) / 2.0
    lo, hi = np.percentile(arr, [100 * half, 100 * (1 - half)])
    return float(lo), float(hi)


def bootstrap_reports(
    metric_fn: MetricFn,
    units: Sequence,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    resampler: ReportResampler | None = None,
) -> dict[str, dict[str, float]]:
    """``{metric: {point, lo, hi, n_valid}}`` for one (method, cohort) cell.

    Args:
        metric_fn: maps a sequence of per-report units to a mapping of metric name to value. For
            extraction quality a unit is a ``(gold_set, pred_set)`` pair.
        units: the per-report units, one entry per report, in a fixed order.
        resampler: reuse across cells on the same cohort. When ``None`` one is built from
            ``n_resamples`` and ``seed``, which is convenient for a single cell and wasteful for a
            table.

    Only metrics present in the point estimate are carried. A metric a resample invents is ignored,
    and one it omits contributes NaN to that draw, not aborting the cell.
    """
    point = dict(metric_fn(units))
    names = list(point)
    sampler = resampler or ReportResampler(len(units), n_resamples=n_resamples, seed=seed)

    draws = np.full((sampler.n_resamples, len(names)), np.nan, dtype=np.float64)
    for d, resampled in enumerate(sampler.resample(units)):
        got = metric_fn(resampled)
        for j, name in enumerate(names):
            value = got.get(name)
            if value is not None:
                draws[d, j] = float(value)

    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(names):
        lo, hi = percentile_ci(draws[:, j], ci)
        out[name] = {
            "point": float(point[name]),
            "lo": lo,
            "hi": hi,
            "n_valid": int(np.count_nonzero(~np.isnan(draws[:, j]))),
        }
    return out


def flatten_ci(prefix_free: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """``{metric: {...}}`` to the flat ``metric``/``metric_lo``/``metric_hi`` columns of a table."""
    flat: dict[str, float] = {}
    for name, stats in prefix_free.items():
        flat[name] = stats["point"]
        flat[f"{name}_lo"] = stats["lo"]
        flat[f"{name}_hi"] = stats["hi"]
    return flat
