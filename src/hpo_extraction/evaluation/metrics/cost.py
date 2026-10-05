"""§Scalability, measured cost, in the three regimes the skeleton insists are not interchangeable.

"Asymptotic statements of the form :math:`\\mathcal{O}(K)` with :math:`K \\ll N` are shared by every
tree-based method and do not by themselves establish a cost advantage, since the per-node constant
differs by orders of magnitude between a linear classifier and :math:`S` SLM forward passes. We
therefore report measured cost."

The three regimes:

1. **Per-report inference cost**, :func:`slm_call_stats`, :func:`hardware_profile`. The primary
   unit is the number of SLM invocations per report, which is hardware-independent and
   reproducible. Wall-clock, memory and model size are tabulated beside it *in the main results
   table*, following the extreme-classification convention, not exiled to an appendix.
2. **Deployment cost**, :func:`deployment_cost`, :func:`break_even_corpus_size`. Contextual
   database generation is a substantial one-time expense per ontology version, and "reporting
   inference cost alone would misstate the system's economics".
3. **Scaling in ontology size**, :func:`scaling_curve`. "This is the measurement that distinguishes
   a scalability claim from a runtime report."

(Traversal cost by depth, the fourth item of §Scalability, lives in
:func:`~hpo_extraction.evaluation.metrics.traversal.depth_cost_table`, next to the reachability recall it
trades against.)

.. warning::
   **Two inputs are not currently recorded by any run and must be instrumented before they can be
   reported.**

   * *Tokens processed per report.* ``hpo_extraction.treephenorag.score_store._call_records`` persists logits, not
     prompt text, and no driver logs a token count. Populating it requires adding a counter to the
     SLM wrappers.
   * *Deployment GPU-hours.* Must be read off the earlier context-generation job records. Nothing in
     the pipeline aggregates them.

   Both are accepted here as caller-supplied arguments and are reported as ``None`` when omitted,
   rather than being silently defaulted to a number that would look measured.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Slack around beta = 1 in :func:`scaling_curve`. A least-squares fit to perfectly linear data
#: lands a few ULPs below 1. Without this, exact linearity would be reported as sublinear.
_LINEARITY_TOLERANCE = 1e-9


def _summary(values: Sequence[float]) -> dict:
    """``n``/``mean``/``median``/``min``/``max``/``total`` for a sample, ``nan`` when empty."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"), "total": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }


# ── 1. Per-report inference cost ─────────────────────────────────────────────

def slm_call_stats(traversal_records: Iterable[Mapping]) -> dict:
    """Per-report SLM invocations, with the DAG deduplication kept auditable.

    "Let :math:`K` denote the number of unique ontology terms evaluated. Total SLM calls are
    :math:`S \\cdot K`. Because :math:`\\mathcal{H}` is a DAG, a term may be inserted into the
    frontier along several paths, and we report unique nodes evaluated and total frontier insertions
    separately so that deduplication is auditable."

    Args:
        traversal_records: one mapping per report with ``n_slm_calls``,
            ``n_unique_nodes_visited`` and ``total_frontier_insertions``, the
            ``*_traversal.jsonl`` schema written by ``hpo_extraction.treephenorag.score_store._write_tau``.

    Returns:
        A ``{quantity: summary}`` dict plus ``dedup_ratio`` (frontier insertions per unique node, 1.0 would mean the ontology behaved as a tree) and ``calls_per_node`` (the effective
        :math:`S`, which is 2, not 1 in the ``union_prune_own_accept`` retrieval mode
        because that mode issues two retrieval passes per node).
    """
    records = [dict(r) for r in traversal_records]
    calls = [float(r.get("n_slm_calls", 0) or 0) for r in records]
    nodes = [float(r.get("n_unique_nodes_visited", 0) or 0) for r in records]
    insertions = [float(r.get("total_frontier_insertions", 0) or 0) for r in records]

    total_nodes = sum(nodes)
    total_insertions = sum(insertions)
    total_calls = sum(calls)
    return {
        "n_reports": len(records),
        "slm_calls_per_report": _summary(calls),
        "unique_nodes_per_report": _summary(nodes),
        "frontier_insertions_per_report": _summary(insertions),
        "dedup_ratio": (total_insertions / total_nodes) if total_nodes else float("nan"),
        "calls_per_node": (total_calls / total_nodes) if total_nodes else float("nan"),
    }


def hardware_profile(
    n_reports: int,
    wall_clock_seconds: float,
    peak_gpu_bytes: int | None = None,
    model_parameters: int | None = None,
    model_bytes: int | None = None,
    tokens_per_report: float | None = None,
    device: str | None = None,
    quantisation: str | None = None,
    serving_stack: str | None = None,
    batch_size: int | None = None,
    context_length: int | None = None,
) -> dict:
    """The cost columns of the main results table, with the hardware spec attached to them.

    "…each with full hardware specification (device, quantisation, serving stack, batch size,
    context length)". The specification fields are carried through unchanged so a cost row can never
    be quoted without the configuration that produced it.

    ``tokens_per_report`` is ``None`` until the run drivers are instrumented, see the module
    warning. It is left ``None``, not estimated, because an estimate would be
    indistinguishable from a measurement once it is in a table.
    """
    if n_reports <= 0:
        raise ValueError(f"n_reports must be positive, got {n_reports}")
    return {
        "n_reports": n_reports,
        "wall_clock_seconds": wall_clock_seconds,
        "seconds_per_report": wall_clock_seconds / n_reports,
        "peak_gpu_bytes": peak_gpu_bytes,
        "peak_gpu_gb": (peak_gpu_bytes / 1024 ** 3) if peak_gpu_bytes is not None else None,
        "model_parameters": model_parameters,
        "model_bytes": model_bytes,
        "model_gb": (model_bytes / 1024 ** 3) if model_bytes is not None else None,
        "tokens_per_report": tokens_per_report,
        "hardware": {
            "device": device, "quantisation": quantisation, "serving_stack": serving_stack,
            "batch_size": batch_size, "context_length": context_length,
        },
    }


# ── 2. Deployment cost ───────────────────────────────────────────────────────

def deployment_cost(
    gpu_hours: float | None,
    index_bytes: int | None = None,
    n_terms_indexed: int | None = None,
    n_context_sentences: int | None = None,
) -> dict:
    """One-time cost of preparing the method for a new or updated ontology.

    Adopts the *deployment time* of Tao et al.: the total time required to prepare a method for
    inference on a new or updated ontology, including index construction and any retraining,
    reported in GPU-hours together with the storage footprint of the resulting index.

    The per-term figures make deployment cost comparable across methods whose ontology coverage
    differs, a system that indexes 9 target phenotypes and one that indexes all ~19k are not
    otherwise on the same axis.
    """
    return {
        "gpu_hours": gpu_hours,
        "index_bytes": index_bytes,
        "index_gb": (index_bytes / 1024 ** 3) if index_bytes is not None else None,
        "n_terms_indexed": n_terms_indexed,
        "n_context_sentences": n_context_sentences,
        "gpu_hours_per_term": (
            gpu_hours / n_terms_indexed
            if gpu_hours is not None and n_terms_indexed else None
        ),
        "bytes_per_term": (
            index_bytes / n_terms_indexed
            if index_bytes is not None and n_terms_indexed else None
        ),
    }


def break_even_corpus_size(
    deployment_gpu_hours: float,
    baseline_seconds_per_report: float,
    method_seconds_per_report: float,
) -> dict:
    """Corpus size at which the amortised deployment cost is offset by the per-report saving.

    "…the quantity that determines whether the approach is economical for a given archive."

    .. math:: n^{*} = \\frac{3600 \\cdot \\text{deployment GPU-hours}}
                           {t_{\\text{baseline}} - t_{\\text{method}}}

    Returns ``n_reports`` as ``inf`` when the method is not actually faster per report, in that
    case there is no corpus size that pays the deployment cost back, and reporting a finite
    break-even would invert the conclusion.
    """
    saving = baseline_seconds_per_report - method_seconds_per_report
    if saving <= 0:
        return {
            "n_reports": math.inf,
            "seconds_saved_per_report": saving,
            "deployment_seconds": deployment_gpu_hours * 3600.0,
            "economical": False,
        }
    deployment_seconds = deployment_gpu_hours * 3600.0
    return {
        "n_reports": deployment_seconds / saving,
        "seconds_saved_per_report": saving,
        "deployment_seconds": deployment_seconds,
        "economical": True,
    }


# ── 3. Scaling in ontology size ──────────────────────────────────────────────

def scaling_curve(points: Iterable[tuple[float, float]]) -> dict:
    """Empirical sublinearity: SLM calls per report against label-space size :math:`N`.

    "To establish sublinearity empirically, not by design, we measure SLM calls per
    report against label-space size :math:`N` by restricting traversal to subtrees of increasing
    size, and compare the resulting curve to a linear reference."

    A least-squares fit of :math:`\\log(\\text{calls}) = \\alpha + \\beta \\log N` is performed;
    :math:`\\beta` is the scaling exponent, and :math:`\\beta < 1` is the sublinearity claim. The
    linear reference is located at the **smallest** measured :math:`N`, so it answers "what would
    the largest subtree have cost had scaling been linear from the smallest one".

    Args:
        points: ``(N, mean_slm_calls_per_report)`` pairs, one per subtree restriction. Produced by
            runs with ``cfg.restrict_to_subtree`` set (``hpo_extraction.treephenorag.score_store._build_universe``).

    ``sublinear`` applies a small tolerance around :math:`\\beta = 1`: a least-squares fit to
    perfectly linear data lands a few ULPs below 1, and reporting that as evidence of sublinearity
    would be an artifact of floating-point arithmetic, not a measurement.

    Returns:
        ``exponent`` (:math:`\\beta`), ``intercept``, ``r_squared``, ``sublinear``, ``points`` and
        ``linear_reference`` (the calls a linear system would have needed at each :math:`N`).
    """
    data = [(float(n), float(c)) for n, c in points]
    if len(data) < 2:
        raise ValueError(f"need at least two (N, calls) points to fit a slope, got {len(data)}")
    if any(n <= 0 or c <= 0 for n, c in data):
        raise ValueError("N and calls must be positive for a log-log fit")

    data.sort()
    log_n = np.log(np.array([n for n, _ in data]))
    log_c = np.log(np.array([c for _, c in data]))
    beta, alpha = np.polyfit(log_n, log_c, 1)

    predicted = alpha + beta * log_n
    ss_res = float(((log_c - predicted) ** 2).sum())
    ss_tot = float(((log_c - log_c.mean()) ** 2).sum())

    n0, c0 = data[0]
    return {
        "exponent": float(beta),
        "intercept": float(alpha),
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "sublinear": bool(beta < 1.0 - _LINEARITY_TOLERANCE),
        "points": [{"n_terms": n, "slm_calls_per_report": c} for n, c in data],
        "linear_reference": [
            {"n_terms": n, "slm_calls_per_report": c0 * n / n0} for n, _ in data
        ],
    }
