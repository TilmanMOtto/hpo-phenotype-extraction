"""§Calibration, the estimators the two-threshold design rests on.

":math:`\\tau_{\\text{prune}}` and :math:`\\tau_{\\text{accept}}` are transferable across ontology
regions and cohorts only if a score of :math:`0.3` denotes the same empirical presence frequency at
depth two as at depth six. Calibration is therefore not a cosmetic property of the system but the
assumption its thresholds rest on."

The skeleton reports ECE "because it is the field's default" and then states three reasons it is
inadequate here, binning sensitivity and discontinuity, equal-width bias, and cancellation across
bins while averaging over a confidence range in which only a narrow tail counts. Each objection has
a corresponding estimator in this module, so the critique in the text is backed by a number:

=================================  =========================================================
:func:`ece_equal_width`            the field default, eq. (7), reported for comparability
:func:`ece_equal_mass`             answers the equal-width bias objection
:func:`roelofs_monotonic_sweep`    answers the "how many bins" objection
:func:`smooth_ece`                 answers the binning-and-discontinuity objection (bin-free)
:func:`discrete_score_table`       the binning-free limit when the score support is enumerable
:func:`signed_gap_bins`            answers the sign-cancellation objection
:func:`conditional_calibration`    answers the group-conditional objection
:func:`tail_calibration_error`     answers the "wrong region of the confidence range" objection
=================================  =========================================================

**Which score against which label.** The pipeline carries two scores per node
(``hpo_extraction.treephenorag.pooling``) and they are calibrated against *different* labels, which the skeleton
leaves as an open TODO but the implementation settles:

* ``accept_score`` is a *node-present* quantity, calibrate it against
  :math:`\\mathbf{1}[p_i \\in Y_r]`.
* ``prune_score`` is a *subtree-present* quantity, calibrate it against
  :math:`\\mathbf{1}[\\text{subtree}(p_i) \\cap Y_r \\neq \\emptyset]`.

Mixing them produces a number that means nothing. Every function here takes ``confidences`` and
``labels`` as caller-supplied arrays. Pairing them correctly is the caller's job and is stated in
this docstring so the pairing is checkable.

All functions take 1-D array-likes of confidences in :math:`[0, 1]` and binary labels in
:math:`\\{0, 1\\}`, and return plain floats/dicts.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Binning = Literal["equal_width", "equal_mass"]

# Set the first time relplot is found missing, so the warning fires once per process rather than
# once per (method, cohort, score) cell.
_SMOOTH_ECE_WARNED = False


def _as_arrays(confidences, labels) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    lab = np.asarray(labels, dtype=np.float64).ravel()
    if conf.shape != lab.shape:
        raise ValueError(f"confidences and labels must align: {conf.shape} vs {lab.shape}")
    if conf.size and (conf.min() < 0.0 or conf.max() > 1.0):
        raise ValueError(
            f"confidences must lie in [0, 1]; got [{conf.min():.4f}, {conf.max():.4f}]. "
            "Pass sigmoid(margin), not the raw logit margin."
        )
    bad = set(np.unique(lab).tolist()) - {0.0, 1.0}
    if bad:
        raise ValueError(f"labels must be binary 0/1; found {sorted(bad)[:5]}")
    return conf, lab


# ── Binning ──────────────────────────────────────────────────────────────────

def bin_edges(confidences, n_bins: int, binning: Binning = "equal_mass") -> np.ndarray:
    """Bin boundaries for the two schemes the skeleton contrasts.

    ``equal_width`` splits :math:`[0, 1]` into ``n_bins`` equal intervals. ``equal_mass`` places
    the edges at the empirical quantiles, so each bin holds (as nearly as ties allow) the same
    number of predictions. Because :math:`c(p_i)` aggregates :math:`S` binary SLM responses its
    support is discrete with at most :math:`S+1` atoms, so equal-mass edges will frequently
    coincide and produce fewer non-empty bins than requested, which is the phenomenon
    :func:`discrete_score_table` sidesteps.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if binning == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    if conf.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    quantiles = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
    quantiles[0], quantiles[-1] = min(quantiles[0], 0.0), max(quantiles[-1], 1.0)
    return quantiles


def bin_summary(
    confidences,
    labels,
    n_bins: int = 15,
    binning: Binning = "equal_mass",
) -> list[dict]:
    """Per-bin statistics: the raw material of eq. (7) and of every reliability diagram.

    Returns one dict per **non-empty** bin with ``lo``, ``hi``, ``count``, ``weight`` (share of all
    predictions), ``confidence`` (:math:`\\mathrm{conf}(B_m)`, the mean predicted probability),
    ``accuracy`` (:math:`\\mathrm{acc}(B_m)`, the empirical presence frequency) and ``gap``
    (:math:`\\mathrm{acc} - \\mathrm{conf}`, **signed**).
    """
    conf, lab = _as_arrays(confidences, labels)
    if conf.size == 0:
        return []
    edges = bin_edges(conf, n_bins, binning)
    n = conf.size
    out: list[dict] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # Half-open bins, with the final bin closed so a confidence of 1.0 is never lost.
        mask = (conf >= lo) & (conf < hi)
        if i == len(edges) - 2:
            mask |= conf == hi
        count = int(mask.sum())
        if count == 0:
            continue
        c_mean = float(conf[mask].mean())
        a_mean = float(lab[mask].mean())
        out.append({
            "lo": float(lo), "hi": float(hi), "count": count, "weight": count / n,
            "confidence": c_mean, "accuracy": a_mean, "gap": a_mean - c_mean,
        })
    return out


# ── Eq. (7): expected calibration error ──────────────────────────────────────

def expected_calibration_error(
    confidences,
    labels,
    n_bins: int = 15,
    binning: Binning = "equal_mass",
) -> float:
    """Skeleton eq. (7):
    :math:`\\mathrm{ECE} = \\sum_m \\frac{|B_m|}{n}\\,|\\mathrm{acc}(B_m) - \\mathrm{conf}(B_m)|`.

    ``nan`` for an empty input. The absolute value inside the sum is the source of the
    cancellation the skeleton objects to; :func:`signed_gap_bins` exposes the same bins without it.
    """
    bins = bin_summary(confidences, labels, n_bins, binning)
    if not bins:
        return float("nan")
    return float(sum(b["weight"] * abs(b["gap"]) for b in bins))


def ece_equal_width(confidences, labels, n_bins: int = 10) -> float:
    """Eq. (7) with equal-**width** bins, the field's default, reported for comparability.

    Kept as a named function because it is the convention the repo's existing
    ``app/tree_ui/calib.compute_ece`` uses, so the two can be cross-checked against each other.
    """
    return expected_calibration_error(confidences, labels, n_bins, "equal_width")


def ece_equal_mass(confidences, labels, n_bins: int = 15) -> float:
    """Eq. (7) with equal-**mass** bins, "equal-width bins are biased relative to equal-mass bins"."""
    return expected_calibration_error(confidences, labels, n_bins, "equal_mass")


def roelofs_monotonic_sweep(
    confidences,
    labels,
    max_bins: int | None = None,
) -> dict:
    """Choose the bin count by the monotonicity-preserving sweep of Roelofs et al. (2022).

    The rule as implemented: sweep :math:`M = 2, 3, \\dots` over equal-mass binnings and keep the
    **largest** :math:`M` whose sequence of bin accuracies :math:`\\mathrm{acc}(B_1), \\dots,
    \\mathrm{acc}(B_M)` is still non-decreasing in confidence. Finer bins reduce the estimator's
    bias but raise its variance. The point at which the binned accuracy stops being monotone is
    where the variance has started to manufacture structure, so it is where the sweep stops.

    Args:
        max_bins: upper end of the sweep. Defaults to the number of distinct confidence values,
            capped at 100, going beyond the score's own support cannot add information.

    Returns:
        ``{"n_bins", "ece", "n_distinct_scores", "max_bins_tried", "monotone_at_max"}``.
        ``n_bins`` is 1 when even two bins are non-monotone, in which case ECE degenerates to the
        single-bin gap ``|mean(label) - mean(confidence)|``.
    """
    conf, lab = _as_arrays(confidences, labels)
    n_distinct = int(np.unique(conf).size)
    if conf.size == 0:
        return {"n_bins": 0, "ece": float("nan"), "n_distinct_scores": 0,
                "max_bins_tried": 0, "monotone_at_max": False}

    upper = max_bins if max_bins is not None else min(max(n_distinct, 1), 100)
    upper = max(1, min(upper, conf.size))

    best_m = 1
    monotone_at_max = False
    for m in range(2, upper + 1):
        bins = bin_summary(conf, lab, m, "equal_mass")
        accuracies = [b["accuracy"] for b in bins]
        if all(a <= b + 1e-12 for a, b in zip(accuracies, accuracies[1:])):
            best_m = m
            monotone_at_max = m == upper
        else:
            break

    return {
        "n_bins": best_m,
        "ece": expected_calibration_error(conf, lab, best_m, "equal_mass"),
        "n_distinct_scores": n_distinct,
        "max_bins_tried": upper,
        "monotone_at_max": monotone_at_max,
    }


def smooth_ece(confidences, labels) -> float:
    """SmoothECE, the bin-free kernel-smoothed estimator of Błasiok & Nakkiran (2024).

    "A bin-free kernel-smoothed estimator that is continuous in the predictor and admits a
    principled reliability diagram." Unlike eq. (7) it does not depend on a binning scheme and is
    not a discontinuous functional of the predictor, which is the skeleton's central objection to
    ECE.

    This delegates to the authors' own ``relplot`` package, not reimplementing the adaptive
    bandwidth search, so the reported number is by design the published estimator. Verified
    against **relplot 1.0.3**, whose entry point is ``relplot.smECE(f, y)`` taking predicted
    probabilities first and binary labels second, the same argument order as every other function
    in this package. ``relplot.metrics.smECE`` is accepted as a fallback in case a future release
    moves the top-level alias.

    The import is lazy so the rest of the package stays usable without the optional dependency;
    :func:`calibration_report` reports ``None`` for this row, not failing when it is absent.

    Install with ``pip install relplot`` (it is listed in ``environment.yaml``).
    """
    conf, lab = _as_arrays(confidences, labels)
    if conf.size == 0:
        return float("nan")
    try:
        import relplot  # noqa: PLC0415  (optional dependency, resolved lazily)
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "smooth_ece requires the 'relplot' package (Błasiok & Nakkiran 2024). "
            "Install it with `pip install relplot`; it is listed in environment.yaml."
        ) from exc

    fn = getattr(relplot, "smECE", None) or getattr(
        getattr(relplot, "metrics", None), "smECE", None
    )
    if not callable(fn):  # pragma: no cover - guards against a future relplot API change
        raise AttributeError(
            "the installed relplot exposes neither relplot.smECE nor relplot.metrics.smECE; "
            "check its API and update smooth_ece accordingly."
        )
    return float(fn(conf, lab))


def brier(confidences, labels) -> float:
    """Mean squared error of the probability, a strictly proper score, used here as a control.

    Not requested by the skeleton, but it decomposes into calibration plus refinement, so a Brier
    that improves while ECE worsens (or vice versa) is a signal that one of the two numbers is
    being driven by binning, not by the predictor.
    """
    conf, lab = _as_arrays(confidences, labels)
    if conf.size == 0:
        return float("nan")
    return float(np.mean((conf - lab) ** 2))


# ── The binning-free limit: one row per attainable score ─────────────────────

def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default: 95 %).

    Preferred over the normal approximation because the presence frequency at an individual score
    atom is often estimated from few observations and often near 0 or 1, where the Wald interval
    leaves the unit interval outright.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def discrete_score_table(confidences, labels, decimals: int = 6) -> list[dict]:
    """Empirical presence frequency and Wilson interval at each attainable score.

    "Where :math:`S` is small enough that the score support is enumerable, we additionally tabulate
    the empirical presence frequency and Wilson interval at each attainable score, which is the
    binning-free limit of a reliability diagram for a discrete score."

    Since :math:`c(p_i)` aggregates :math:`S` binary responses its support has at most :math:`S+1`
    atoms, so this table *is* the reliability diagram, no binning choice enters it at all.

    Args:
        decimals: scores are rounded to this many decimals before grouping, so floating-point
            noise does not split one atom into several near-identical rows.

    Returns:
        One row per distinct score, ascending: ``score``, ``count``, ``n_positive``, ``frequency``,
        ``wilson_lo``, ``wilson_hi``, ``gap`` (frequency − score).
    """
    conf, lab = _as_arrays(confidences, labels)
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])  # [count, n_positive]
    for c, y in zip(conf, lab):
        cell = grouped[round(float(c), decimals)]
        cell[0] += 1
        cell[1] += int(y)

    rows: list[dict] = []
    for score in sorted(grouped):
        count, positives = grouped[score]
        lo, hi = wilson_interval(positives, count)
        freq = positives / count
        rows.append({
            "score": score, "count": count, "n_positive": positives, "frequency": freq,
            "wilson_lo": lo, "wilson_hi": hi, "gap": freq - score,
        })
    return rows


# ── Signed and conditional calibration ───────────────────────────────────────

def signed_gap_bins(
    confidences,
    labels,
    n_bins: int = 15,
    binning: Binning = "equal_mass",
) -> list[dict]:
    """:math:`\\mathrm{acc}(B_m) - \\mathrm{conf}(B_m)` **with its sign**, plus bin populations.

    "Because the direction of miscalibration determines the correct threshold adjustment, we plot
    the signed gap with bin populations overlaid, not its absolute value." A positive gap
    means the score is under-confident in that bin (the threshold can be lowered), a negative gap
    over-confident.

    Identical to :func:`bin_summary`. Named separately because it is what the reliability figure
    plots and because the distinction from eq. (7) is the point being made.
    """
    return bin_summary(confidences, labels, n_bins, binning)


def conditional_calibration(
    confidences,
    labels,
    groups: Sequence,
    n_bins: int = 15,
    binning: Binning = "equal_mass",
    min_count: int = 30,
) -> dict:
    """Calibration computed separately within each group, "our argument is regional by design".

    "Aggregate ECE cannot detect group-conditional miscalibration." The skeleton names three
    groupings. Build them with :func:`depth_groups`, :func:`layer1_groups` and
    :func:`frequency_buckets` and pass the result here.

    Args:
        groups: one group key per prediction, aligned with ``confidences``.
        min_count: groups smaller than this are computed but flagged ``sufficient=False``, since an
            ECE over a handful of points is noise, not evidence.

    Returns:
        ``{"groups": {key: {...}}, "worst_group", "worst_ece", "max_minus_aggregate_ece",
        "aggregate_ece"}``. ``max_minus_aggregate_ece`` is the main: how much miscalibration
        the aggregate number hides.
    """
    conf, lab = _as_arrays(confidences, labels)
    groups = list(groups)
    if len(groups) != conf.size:
        raise ValueError(f"groups must align with confidences: {len(groups)} vs {conf.size}")

    by_group: dict[object, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)

    aggregate = expected_calibration_error(conf, lab, n_bins, binning)
    out: dict[object, dict] = {}
    for key, idx in by_group.items():
        sub_conf, sub_lab = conf[idx], lab[idx]
        out[key] = {
            "n": len(idx),
            "prevalence": float(sub_lab.mean()),
            "mean_confidence": float(sub_conf.mean()),
            "ece": expected_calibration_error(sub_conf, sub_lab, n_bins, binning),
            "brier": brier(sub_conf, sub_lab),
            "sufficient": len(idx) >= min_count,
        }

    usable = {k: v for k, v in out.items() if v["sufficient"] and not math.isnan(v["ece"])}
    worst_key = max(usable, key=lambda k: usable[k]["ece"]) if usable else None
    worst_ece = usable[worst_key]["ece"] if worst_key is not None else float("nan")
    return {
        "groups": out,
        "aggregate_ece": aggregate,
        "worst_group": worst_key,
        "worst_ece": worst_ece,
        "max_minus_aggregate_ece": (
            worst_ece - aggregate if worst_key is not None else float("nan")
        ),
        "n_groups": len(out),
        "n_groups_sufficient": len(usable),
    }


def depth_groups(hpo_ids: Iterable[str], view) -> list:
    """Group key per prediction = the term's ontology depth (§"separately by ontology depth")."""
    return [view.depth(h) for h in hpo_ids]


def layer1_groups(hpo_ids: Iterable[str], view) -> list:
    """Group key per prediction = the term's top-level subtree (§"by top-level subtree (organ system)")."""
    return [view.layer1(h) for h in hpo_ids]


def frequency_buckets(
    hpo_ids: Iterable[str],
    gold_sets: Iterable[Iterable[str]],
    n_buckets: int = 4,
) -> list[str]:
    """Group key per prediction = a gold-frequency quantile bucket (§"by gold-set term frequency").

    A term's frequency is the number of reports whose ground-truth set contains it. Terms are ranked by
    that count and split into ``n_buckets`` roughly equal-sized groups, labelled ``"q1"`` (rarest)
    upward. Terms absent from every ground-truth set get ``"unseen"``, which is its own bucket because
    "never ground truth" is a categorically different condition from "rarely ground truth".
    """
    counts: Counter = Counter()
    for gold in gold_sets:
        counts.update(set(gold))
    if not counts:
        return ["unseen"] * len(list(hpo_ids))

    ranked = sorted(counts, key=lambda t: (counts[t], t))
    bucket_of: dict[str, str] = {}
    n = len(ranked)
    for i, term in enumerate(ranked):
        q = min(n_buckets, int(i * n_buckets / n) + 1)
        bucket_of[term] = f"q{q}"
    return [bucket_of.get(h, "unseen") for h in hpo_ids]


def tail_calibration_error(
    confidences,
    labels,
    upper: float,
    n_bins: int = 10,
    binning: Binning = "equal_mass",
) -> dict:
    """Calibration error restricted to the low-confidence tail :math:`[0, \\tau_{\\text{prune}} + \\delta]`.

    "Since :math:`\\tau_{\\text{prune}}` operates in the low-confidence tail while ECE is dominated
    by the densely populated high-confidence region, we additionally report calibration error
    restricted to :math:`[0, \\tau_{\\text{prune}} + \\delta]`."

    Args:
        upper: the right edge, pass :math:`\\tau_{\\text{prune}} + \\delta` already summed.
            :math:`\\delta` is not defaulted here: it is an operating-point choice
            that belongs in the experiment config, not in the metric.

    Returns:
        ``{"upper", "n", "coverage", "ece", "brier", "prevalence", "mean_confidence"}``.
        ``coverage`` is the share of all predictions that fall in the tail, the number that says
        whether the aggregate ECE could have seen this region at all.
    """
    conf, lab = _as_arrays(confidences, labels)
    if not 0.0 < upper <= 1.0:
        raise ValueError(f"upper must lie in (0, 1], got {upper}")
    mask = conf <= upper
    n_tail = int(mask.sum())
    if n_tail == 0:
        return {"upper": upper, "n": 0, "coverage": 0.0, "ece": float("nan"),
                "brier": float("nan"), "prevalence": float("nan"),
                "mean_confidence": float("nan")}
    sub_conf, sub_lab = conf[mask], lab[mask]
    return {
        "upper": upper,
        "n": n_tail,
        "coverage": n_tail / conf.size,
        "ece": expected_calibration_error(sub_conf, sub_lab, n_bins, binning),
        "brier": brier(sub_conf, sub_lab),
        "prevalence": float(sub_lab.mean()),
        "mean_confidence": float(sub_conf.mean()),
    }


def calibration_report(
    confidences,
    labels,
    tail_upper: float | None = None,
    n_bins: int = 15,
) -> dict:
    """Every §Calibration scalar for one score/label pairing, in one dict.

    ``smooth_ece`` is included when ``relplot`` is importable and reported as ``None`` when it is
    not, so a missing optional dependency degrades one row, not failing the whole report.
    The absence is **logged once per process** at WARNING: a silent ``None`` in a results table is
    indistinguishable from "this metric does not apply to this method", and the skeleton names
    SmoothECE as one of the two replacements for equal-width ECE, so it must not go missing
    quietly.
    """
    conf, lab = _as_arrays(confidences, labels)
    sweep = roelofs_monotonic_sweep(conf, lab)
    try:
        smece: float | None = smooth_ece(conf, lab)
    except (ImportError, AttributeError) as exc:
        smece = None
        global _SMOOTH_ECE_WARNED
        if not _SMOOTH_ECE_WARNED:
            _SMOOTH_ECE_WARNED = True
            logger.warning(
                "SmoothECE unavailable — every smooth_ece cell will read n/a. %s "
                "Fix with `pip install relplot==1.0.3` (already pinned in environment.yaml); "
                "the conda env in use predates that pin.", exc)
    out = {
        "n": int(conf.size),
        "prevalence": float(lab.mean()) if conf.size else float("nan"),
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
        "ece_equal_width": ece_equal_width(conf, lab),
        "ece_equal_mass": ece_equal_mass(conf, lab, n_bins),
        "ece_roelofs": sweep["ece"],
        "roelofs_n_bins": sweep["n_bins"],
        "smooth_ece": smece,
        "brier": brier(conf, lab),
        "n_distinct_scores": sweep["n_distinct_scores"],
    }
    if tail_upper is not None:
        out["tail"] = tail_calibration_error(conf, lab, tail_upper)
    return out


# ── §Threshold transferability ───────────────────────────────────────────────

def fit_thresholds(
    evaluate_fn: Callable[[float, float], Mapping[str, float]],
    tau_prune_grid: Sequence[float],
    tau_accept_grid: Sequence[float],
    objective: str = "micro_f1",
) -> dict:
    """Grid-search :math:`(\\tau_{\\text{prune}}, \\tau_{\\text{accept}})` on one split.

    The module owns the *protocol*. The caller owns the re-run. ``evaluate_fn(tau_prune,
    tau_accept)`` must return a metrics mapping containing ``objective`` for the split being fit, for the tree experiments that is a re-run of the cached per-node confidences, which is why it
    cannot live in a metrics package that does no IO.

    Returns ``{"tau_prune", "tau_accept", "objective", "value", "metrics"}``.
    """
    best: dict | None = None
    for tp in tau_prune_grid:
        for ta in tau_accept_grid:
            metrics = evaluate_fn(tp, ta)
            if objective not in metrics:
                raise KeyError(
                    f"evaluate_fn returned no {objective!r}; got {sorted(metrics)[:8]}"
                )
            value = float(metrics[objective])
            if best is None or value > best["value"]:
                best = {"tau_prune": float(tp), "tau_accept": float(ta),
                        "objective": objective, "value": value, "metrics": dict(metrics)}
    if best is None:
        raise ValueError("empty threshold grid")
    return best


def transfer_gap(
    fit_evaluate: Callable[[float, float], Mapping[str, float]],
    eval_evaluate: Callable[[float, float], Mapping[str, float]],
    tau_prune_grid: Sequence[float],
    tau_accept_grid: Sequence[float],
    objective: str = "micro_f1",
) -> dict:
    """Fit a threshold pair on one split, apply it to a held-out one, and price the transfer.

    "We test this directly by fitting :math:`(\\tau_{\\text{prune}}, \\tau_{\\text{accept}})` on one
    subtree or cohort and evaluating on a held-out one."

    The claim being tested is *narrow*: monotone recalibration cannot reorder scores and therefore
    cannot improve discrimination, so the only thing calibration can buy is that a single threshold
    pair is portable. ``gap`` is what portability costs, the held-out objective under the
    transferred thresholds versus under thresholds fitted on the held-out split itself (the oracle
    a portable threshold can never beat). A gap near zero is the evidence the claim needs.
    """
    fitted = fit_thresholds(fit_evaluate, tau_prune_grid, tau_accept_grid, objective)
    transferred = eval_evaluate(fitted["tau_prune"], fitted["tau_accept"])
    oracle = fit_thresholds(eval_evaluate, tau_prune_grid, tau_accept_grid, objective)
    transferred_value = float(transferred[objective])
    return {
        "objective": objective,
        "fitted_tau_prune": fitted["tau_prune"],
        "fitted_tau_accept": fitted["tau_accept"],
        "fit_value": fitted["value"],
        "transferred_value": transferred_value,
        "oracle_tau_prune": oracle["tau_prune"],
        "oracle_tau_accept": oracle["tau_accept"],
        "oracle_value": oracle["value"],
        "gap": oracle["value"] - transferred_value,
        "transferred_metrics": dict(transferred),
    }


def cohort_transfer(
    evaluate_by_cohort: Mapping[str, Callable[[float, float], Mapping[str, float]]],
    tau_prune_grid: Sequence[float],
    tau_accept_grid: Sequence[float],
    objective: str = "micro_f1",
) -> dict:
    """Every ordered pair of cohorts: fit on one, evaluate on the other (HCY :math:`\\leftrightarrow` GSC+).

    Args:
        evaluate_by_cohort: ``{"hcy": evaluate_fn, "gsc": evaluate_fn}``.

    Returns ``{(fit_cohort, eval_cohort): transfer_gap(...)}`` keyed by ``"fit→eval"`` strings, plus
    ``"max_gap"`` and ``"mean_gap"`` over the off-diagonal pairs.
    """
    results: dict[str, dict] = {}
    for fit_name, fit_fn in evaluate_by_cohort.items():
        for eval_name, eval_fn in evaluate_by_cohort.items():
            if fit_name == eval_name:
                continue
            results[f"{fit_name}->{eval_name}"] = transfer_gap(
                fit_fn, eval_fn, tau_prune_grid, tau_accept_grid, objective
            )
    gaps = [r["gap"] for r in results.values()]
    return {
        "pairs": results,
        "max_gap": max(gaps) if gaps else float("nan"),
        "mean_gap": sum(gaps) / len(gaps) if gaps else float("nan"),
    }


def kfold_report_split(
    report_ids: Sequence[str],
    k: int = 5,
    seed: int = 0,
) -> list[tuple[list[str], list[str]]]:
    """``k`` deterministic ``(fit_ids, eval_ids)`` splits over reports.

    The weakest of the transferability tests, same cohort, same ontology regions, and it is
    included for that reason: it isolates plain overfitting of the threshold pair to the
    tuning sample from the harder cross-cohort question. Reports, never individual cells, are the
    unit, since cells within a report are strongly correlated.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    ids = list(report_ids)
    if len(ids) < k:
        raise ValueError(f"cannot make {k} folds from {len(ids)} reports")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(ids)))
    folds: list[list[str]] = [[] for _ in range(k)]
    for position, index in enumerate(order):
        folds[position % k].append(ids[index])
    return [
        ([rid for j, fold in enumerate(folds) if j != i for rid in fold], folds[i])
        for i in range(k)
    ]


def compare_raw_vs_calibrated(
    calibrated: Mapping[str, float],
    raw: Mapping[str, float],
) -> dict:
    """Put a calibrated-score transfer next to a raw-score transfer under the same protocol.

    "We test this directly … comparing the calibrated score against the raw aggregated score under
    the same protocol." Both arguments are :func:`transfer_gap` results. A smaller ``gap`` for the
    calibrated score is the evidence for portability. A comparable ``transferred_value`` at both is
    the expected control, since monotone recalibration cannot change discrimination.
    """
    return {
        "calibrated_gap": calibrated["gap"],
        "raw_gap": raw["gap"],
        "gap_reduction": raw["gap"] - calibrated["gap"],
        "calibrated_transferred_value": calibrated["transferred_value"],
        "raw_transferred_value": raw["transferred_value"],
        "calibrated_oracle_value": calibrated["oracle_value"],
        "raw_oracle_value": raw["oracle_value"],
    }


# ── The calibration map h ────────────────────────────────────────────────────
#
# Everything above *measures* calibration. This fits it. ``h`` is the non-decreasing map of
# Algorithm 1: a term is accepted iff ``h(pool_acc(sigma)) >= tau_accept``.
#
# **What h can and cannot do, stated once so no table overclaims.** h is monotone, so
# ``h(a) >= t`` is ``a >= h^{-1}(t)``: within one cohort it is a *reparametrisation* of the
# acceptance axis and cannot change which terms are accepted at a threshold tuned on the same data.
# It cannot improve discrimination either, AUC is invariant under a monotone transform. So an
# "omit h" ablation on the development cohort has nothing to omit, and that is the expected result
#, not a disappointment.
#
# What h does buy is that a *single threshold transfers*. Fit h and tau_accept on HCY, apply both
# unchanged to GSC+, and the question is whether the same number still means the same empirical
# frequency. That is why the cross-cohort transfer (:func:`transfer_gap`, :func:`cohort_transfer`)
# is the evidence for calibration being a contribution, and the within-cohort reliability diagram is
# only a diagnostic.


class CalibrationMap:
    """A fitted, non-decreasing ``h: [0, 1] -> [0, 1]``, callable and picklable.

    Picklable counts: a map is fitted per outer fold and has to survive being carried to the GSC
    evaluation, which happens in a different stage of the run.
    """

    def __init__(self, method: str, predict, params: Mapping[str, object] | None = None):
        self.method = method
        self._predict = predict
        self.params = dict(params or {})

    def __call__(self, scores) -> np.ndarray:
        out = np.asarray(self._predict(np.asarray(scores, dtype=np.float64).ravel()),
                         dtype=np.float64)
        return np.clip(out, 0.0, 1.0)

    def inverse_threshold(self, tau: float, grid: int = 20001) -> float:
        """The raw score at which ``h`` first reaches ``tau``, ``h^{-1}(tau)``.

        The acceptance rule is applied in calibrated space, but the *re-run* thresholds raw pooled
        scores, so the two have to be relatable. Monotonicity makes this well defined up to the
        flat stretches isotonic regression produces, where the smallest raw score attaining ``tau``
        is the right choice: it is the one that reproduces ``h(a) >= tau``.
        """
        xs = np.linspace(0.0, 1.0, int(grid))
        ys = self(xs)
        hit = np.flatnonzero(ys >= tau)
        return float(xs[hit[0]]) if hit.size else float("inf")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CalibrationMap({self.method!r}, {self.params})"


def _identity_map() -> CalibrationMap:
    return CalibrationMap("identity", lambda x: x, {})


def fit_calibration_map(
    confidences,
    labels,
    method: str = "isotonic",
    min_positives: int = 5,
) -> CalibrationMap:
    """Fit ``h`` on scored pairs. ``method`` is ``identity``, ``platt`` or ``isotonic``.

    The thesis leaves the family as a ``\\todo{isotonic regression or Platt scaling}``, so it is
    selected by inner cross-validation like every other hyper-parameter, not asserted here;
    ``identity`` is in the menu so "no calibration" competes on the same footing.

    Degenerate fits fall back to ``identity``, not raising: an inner fold can easily contain
    too few positives to fit anything, and a fold that cannot fit ``h`` should contribute its
    uncalibrated score to the selection, not kill the run. The fallback is recorded in ``params``
    so it is visible in the selection table instead of being invisible.

    Only ``sklearn`` is used, the compute nodes are offline with fixed versions, so no
    calibration-specific dependency is available.
    """
    conf, lab = _as_arrays(confidences, labels)
    if method == "identity":
        return _identity_map()
    n_pos = int(lab.sum())
    if conf.size == 0 or n_pos < min_positives or n_pos == lab.size:
        return CalibrationMap("identity", lambda x: x,
                              {"fallback_from": method, "n": int(conf.size),
                               "n_positive": n_pos})
    if method == "platt":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        model = LogisticRegression(max_iter=5000)
        model.fit(conf.reshape(-1, 1), lab)
        # A negative coefficient would make h decreasing, which Algorithm 1 forbids. It means the
        # score is anti-correlated with presence on this fold, a real signal, but not one a
        # calibration map may encode, so fall back, not silently invert the ranking.
        if float(model.coef_.ravel()[0]) < 0:
            return CalibrationMap("identity", lambda x: x,
                                  {"fallback_from": "platt", "reason": "negative slope"})
        return CalibrationMap(
            "platt", lambda x: model.predict_proba(x.reshape(-1, 1))[:, 1],
            {"coef": float(model.coef_.ravel()[0]), "intercept": float(model.intercept_[0]),
             "n": int(conf.size), "n_positive": n_pos},
        )
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(conf, lab)
        return CalibrationMap(
            "isotonic", model.predict,
            {"n": int(conf.size), "n_positive": n_pos,
             "n_steps": int(np.unique(model.predict(conf)).size)},
        )
    raise ValueError(f"unknown calibration method {method!r} "
                     "(expected 'identity', 'platt' or 'isotonic')")


#: The families the inner fold selects between. ``identity`` first so ties prefer no calibration.
CALIBRATION_METHODS: tuple[str, ...] = ("identity", "platt", "isotonic")


def f1_optimal_threshold_diagnostic(confidences, labels, h: CalibrationMap | None = None) -> dict:
    """The chosen threshold against half the optimal pairwise F1, §4.2's calibration diagnostic.

    "As a diagnostic of calibration, we compare the chosen threshold with half the optimal pairwise
    :math:`F_1`, which is the :math:`F_1`-optimal threshold of a calibrated score" (Lipton et al.,
    2014). For a perfectly calibrated score the two coincide. The distance between them is a
    calibration statement that does not depend on any binning.
    """
    conf, lab = _as_arrays(confidences, labels)
    if h is not None:
        conf = h(conf)
    if conf.size == 0 or lab.sum() == 0:
        return {"best_threshold": float("nan"), "best_f1": float("nan"),
                "half_optimal_f1": float("nan"), "gap": float("nan")}
    order = np.argsort(-conf, kind="stable")
    sorted_conf = conf[order]
    tp = np.cumsum(lab[order])
    predicted = np.arange(1, conf.size + 1)
    f1 = 2 * tp / (predicted + lab.sum())
    # A threshold cannot split a run of tied scores: ``conf >= t`` admits the whole run. So F1 is
    # only defined at the END of each tied block. Scoring it inside a block made the answer depend
    # on how the ties happened to be ordered -- on an isotonic map, which is all plateaus, the
    # "optimal" threshold changed between two runs over identical scores.
    block_end = np.r_[sorted_conf[1:] != sorted_conf[:-1], True]
    best = int(np.argmax(np.where(block_end, f1, -np.inf)))
    best_f1 = float(f1[best])
    return {
        "best_threshold": float(sorted_conf[best]),
        "best_f1": best_f1,
        "half_optimal_f1": best_f1 / 2.0,
        "gap": float(conf[order][best]) - best_f1 / 2.0,
    }
