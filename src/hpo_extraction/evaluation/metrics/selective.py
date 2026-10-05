"""§Risk--coverage analysis, selective classification, AURC and AUGRC (skeleton eq. 8).

"The accept decision is an instance of selective classification, whose natural evaluation is the
trade-off between coverage and selective risk rather than an aggregate calibration statistic."

**The unit and the loss.** Eq. (8) writes :math:`\\phi(\\tau)` and :math:`R(\\tau)` without saying
what a sample :math:`i` is or what :math:`\\ell(\\hat{y}_i, y_i)` measures. In a multi-label,
ontology-scale setting that choice changes the number completely. This package fixes it as:

* :math:`i` ranges over every **evaluated (report, term) cell**, the unit
  :math:`\\tau_{\\text{accept}}` actually thresholds;
* :math:`c_i` is that cell's ``accept_score``;
* :math:`\\ell_i = \\mathbf{1}[(c_i \\geq \\tau) \\neq (\\text{term} \\in Y_r)]`, the 0/1 error of
  the accept decision.

**A consequence worth knowing before reading the curve.** A *covered* cell is by design one
the system reports (:math:`c_i \\geq \\tau`), so on covered cells the loss reduces to
:math:`\\mathbf{1}[y_i = 0]` and

.. math:: R(\\tau) = 1 - \\text{micro-precision}(\\tau).

The risk--coverage curve is therefore the precision--coverage curve read upside down. That is not a
degeneracy, it is what selective risk *means* under a report-or-abstain decision, but it means the
curve says nothing about recall, and it is fixed as a unit test so the identity cannot silently
break.

**AURC and AUGRC.** AURC "is known to over-weight high-confidence failures in a manner that can
invert method rankings relative to accuracy and failure detection", so AUGRC is reported alongside:
it omits the division by coverage, which is what makes it interpretable as the *average risk of
undetected failure*. Both conflate discrimination with calibration quality, since methods differ in
accuracy at full coverage, state that explicitly wherever the single- and two-threshold variants
are compared.

**Ties.** :math:`c(p_i)` aggregates :math:`S` binary responses, so its support has at most
:math:`S+1` atoms and tie groups are enormous. A top-:math:`k` AURC would then depend on an
arbitrary ordering within a tie group, so the default (``tie_handling="expected"``) replaces each
cell's loss by its tie-group mean before accumulating, the exact expectation over random orderings
within ties, and permutation-invariant. :func:`risk_coverage_curve` sweeps distinct thresholds and
is tie-free by design. It is the object to plot.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

TieHandling = Literal["expected", "ordered"]


def _as_arrays(confidences, labels) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    lab = np.asarray(labels, dtype=np.float64).ravel()
    if conf.shape != lab.shape:
        raise ValueError(f"confidences and labels must align: {conf.shape} vs {lab.shape}")
    bad = set(np.unique(lab).tolist()) - {0.0, 1.0}
    if bad:
        raise ValueError(f"labels must be binary 0/1; found {sorted(bad)[:5]}")
    return conf, lab


# ── The two quantities of eq. (8) ────────────────────────────────────────────

def coverage(confidences, tau: float) -> float:
    """:math:`\\phi(\\tau) = \\frac{1}{n}\\sum_i \\mathbf{1}[c_i \\geq \\tau]`, the reported fraction."""
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    if conf.size == 0:
        return 0.0
    return float((conf >= tau).mean())


def selective_loss(confidences, labels, tau: float) -> np.ndarray:
    """Per-cell :math:`\\ell_i = \\mathbf{1}[(c_i \\geq \\tau) \\neq y_i]` at one threshold."""
    conf, lab = _as_arrays(confidences, labels)
    return ((conf >= tau).astype(np.float64) != lab).astype(np.float64)


def selective_risk(confidences, labels, tau: float) -> float:
    """:math:`R(\\tau)` of eq. (8), mean loss over the covered cells.

    ``nan`` at a threshold that covers nothing: the risk of an empty selection is undefined, and
    substituting 0 would let a method look perfect by abstaining everywhere.
    """
    conf, lab = _as_arrays(confidences, labels)
    covered = conf >= tau
    n_covered = int(covered.sum())
    if n_covered == 0:
        return float("nan")
    return float(selective_loss(conf, lab, tau)[covered].sum() / n_covered)


def generalized_risk(confidences, labels, tau: float) -> float:
    """:math:`\\frac{1}{n}\\sum_i \\ell_i \\mathbf{1}[c_i \\geq \\tau]`, risk *not* renormalised by coverage.

    The integrand of AUGRC (Traub et al. 2024). Unlike :math:`R(\\tau)` it stays finite as coverage
    goes to zero and it does not reward abstention, which is the ranking pathology AUGRC exists to
    remove.
    """
    conf, lab = _as_arrays(confidences, labels)
    if conf.size == 0:
        return float("nan")
    covered = conf >= tau
    return float((selective_loss(conf, lab, tau) * covered).sum() / conf.size)


def risk_coverage_curve(confidences, labels) -> list[dict]:
    """The full curve, one row per distinct threshold, the object to plot.

    Thresholds run in descending order, a point above the maximum confidence, where coverage is 0,
    then each distinct confidence value. Coverage therefore increases monotonically down the rows,
    which is the order a risk--coverage plot reads in. Every row is exact at its threshold, with no
    tie convention involved.

    Returns:
        rows of ``tau``, ``coverage``, ``n_covered``, ``selective_risk``, ``generalized_risk`` and
        ``precision`` (:math:`1 - R(\\tau)`, included so the identity noted in the module docstring
        is visible in the table itself).
    """
    conf, lab = _as_arrays(confidences, labels)
    if conf.size == 0:
        return []
    rows: list[dict] = [{
        "tau": float(np.nextafter(conf.max(), np.inf)),
        "coverage": 0.0, "n_covered": 0,
        "selective_risk": float("nan"), "generalized_risk": 0.0, "precision": float("nan"),
    }]
    for tau in np.unique(conf)[::-1]:
        n_covered = int((conf >= tau).sum())
        risk = selective_risk(conf, lab, float(tau))
        rows.append({
            "tau": float(tau),
            "coverage": n_covered / conf.size,
            "n_covered": n_covered,
            "selective_risk": risk,
            "generalized_risk": generalized_risk(conf, lab, float(tau)),
            "precision": (1.0 - risk) if risk == risk else float("nan"),
        })
    return rows


# ── Areas ────────────────────────────────────────────────────────────────────

def _ordered_losses(
    confidences,
    labels,
    tie_handling: TieHandling,
) -> np.ndarray:
    """Per-cell losses sorted by descending confidence, with ties resolved.

    On covered cells the loss is :math:`\\mathbf{1}[y_i = 0]` regardless of :math:`\\tau` (see the
    module docstring), so the sweep needs only that vector, the loss does not have to be
    recomputed at every threshold.
    """
    conf, lab = _as_arrays(confidences, labels)
    loss = 1.0 - lab  # covered ⇒ predicted positive ⇒ loss = 1[label == 0]

    if tie_handling == "expected":
        loss = _tie_group_means(conf, loss)
    elif tie_handling != "ordered":
        raise ValueError(f"tie_handling must be 'expected' or 'ordered', got {tie_handling!r}")

    order = np.argsort(-conf, kind="stable")
    return loss[order]


def _tie_group_means(conf: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Replace each cell's loss by the mean over the cells sharing its confidence.

    That mean is the exact expectation over random orderings within a tie group, which is what
    makes the sweep independent of the caller's input order.

    Computed by sorting once and reducing over the run boundaries, :math:`O(n \\log n)`. The
    obvious loop over ``np.unique(conf)`` with a boolean mask per value is :math:`O(U \\cdot n)`,
    and on continuous scores :math:`U \\approx n`: a calibration sample of 540k tree nodes with
    six-decimal scores has ~540k distinct values, which made a single AURC take hours. The two
    agree (``tests/unit/test_thesis_metrics_selective.py``). This is a speed fix, not a
    definition change.
    """
    if conf.size == 0:
        return loss
    order = np.argsort(conf, kind="stable")
    sorted_conf = conf[order]
    # Start index of each run of equal confidences.
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_conf[1:] != sorted_conf[:-1]))
    )
    sums = np.add.reduceat(loss[order], starts)
    counts = np.diff(np.concatenate((starts, [conf.size])))
    means = np.repeat(sums / counts, counts)

    out = np.empty_like(loss, dtype=float)
    out[order] = means
    return out


def aurc(confidences, labels, tie_handling: TieHandling = "expected") -> float:
    """Area under the risk--coverage curve (Geifman et al. 2018).

    Computed in the standard discrete form
    :math:`\\mathrm{AURC} = \\frac{1}{n}\\sum_{k=1}^{n} r(k)`, where :math:`r(k)` is the mean loss
    over the :math:`k` most confident cells, i.e. The curve sampled at every attainable coverage
    :math:`k/n`. **Lower is better.**

    Note the confound the skeleton flags: AURC mixes discrimination with calibration quality,
    because methods differ in accuracy at full coverage (:math:`r(n)` is just the aggregate error
    rate). Two systems can be ranked differently by AURC and by accuracy without either number
    being wrong.
    """
    conf, _ = _as_arrays(confidences, labels)
    if conf.size == 0:
        return float("nan")
    losses = _ordered_losses(confidences, labels, tie_handling)
    k = np.arange(1, losses.size + 1)
    return float((np.cumsum(losses) / k).mean())


def augrc(confidences, labels, tie_handling: TieHandling = "expected") -> float:
    """Area under the **generalised** risk--coverage curve (Traub et al. 2024).

    :math:`\\mathrm{AUGRC} = \\frac{1}{n}\\sum_{k=1}^{n} \\frac{k}{n} r(k)`, the same sweep as
    :func:`aurc` but without renormalising by coverage, so it "admits interpretation as the average
    risk of undetected failure". **Lower is better.**

    Reported alongside AURC because AURC "is known to over-weight high-confidence failures in a
    manner that can invert method rankings relative to accuracy and failure detection".
    """
    conf, _ = _as_arrays(confidences, labels)
    if conf.size == 0:
        return float("nan")
    losses = _ordered_losses(confidences, labels, tie_handling)
    n = losses.size
    return float((np.cumsum(losses) / n).mean())


def selective_report(
    confidences,
    labels,
    tie_handling: TieHandling = "expected",
) -> dict:
    """Every §Risk--coverage scalar for one score/label pairing.

    ``risk_at_full_coverage`` is the aggregate error rate, the evidence location that makes the
    discrimination/calibration confound in AURC readable, not hidden.
    """
    conf, lab = _as_arrays(confidences, labels)
    if conf.size == 0:
        return {"n": 0, "aurc": float("nan"), "augrc": float("nan"),
                "risk_at_full_coverage": float("nan"), "prevalence": float("nan")}
    lowest = float(conf.min())
    return {
        "n": int(conf.size),
        "prevalence": float(lab.mean()),
        "aurc": aurc(conf, lab, tie_handling),
        "augrc": augrc(conf, lab, tie_handling),
        "risk_at_full_coverage": selective_risk(conf, lab, lowest),
        "coverage_at_lowest_tau": coverage(conf, lowest),
    }
