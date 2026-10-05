"""Statistical machinery shared by both results chapters: intervals, tests, and folds.

separate from :mod:`hpo_extraction.evaluation.metrics`, which computes *what* a number is.
This package answers *how sure we are of it*, and it is metric-agnostic: everything takes a metric
function plus a sequence of per-report units, so it applies unchanged to flat quality,
hierarchy-aware quality, retrieval and the traversal diagnostics.

One invariant runs through all of it: **the report is the sampling unit.** Terms within a report
are strongly correlated, a severely affected patient supplies a dozen annotated terms that stand or
fall together, so any procedure treating cells as independent overstates the evidence. That rules
out a bootstrap over cells, a permutation over cells, and any split that puts one patient on both
sides of a fold boundary.

Entry points::

    from hpo_extraction.evaluation.stats import ReportResampler, bootstrap_reports, paired_randomisation, holm

    sampler = ReportResampler(n_reports=118, n_resamples=10_000, seed=0)   # once per cohort
    ci = bootstrap_reports(metric_fn, units, resampler=sampler)            # per method
    test = paired_randomisation(ground truth, pred_a, pred_b, metric_fn)           # per pair
    adjusted = holm({name: t["p_value"] for name, t in tests.items()})     # per cohort
"""

from __future__ import annotations

from .bootstrap import MetricFn, ReportResampler, bootstrap_reports, flatten_ci, percentile_ci
from .crc import crc_lambda, held_out_risk, loss_curve, miss_rate
from .folds import (
    FOLD_COLUMNS,
    nested_folds,
    read_folds,
    size_tertiles,
    stratified_report_folds,
    write_folds,
)
from .multiplicity import holm
from .randomisation import paired_randomisation

__all__ = [
    # bootstrap
    "MetricFn", "ReportResampler", "bootstrap_reports", "flatten_ci", "percentile_ci",
    # conformal risk control
    "miss_rate", "loss_curve", "crc_lambda", "held_out_risk",
    # significance
    "paired_randomisation", "holm",
    # folds
    "FOLD_COLUMNS", "size_tertiles", "stratified_report_folds", "nested_folds",
    "write_folds", "read_folds",
]
