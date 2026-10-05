r"""The chapter-4 numbers the prose quotes inline, as LaTeX macros.

A number a paragraph asserts is as likely to go stale as a number in a table, and far
harder to notice when it does -- nobody re-reads a sentence to check a percentage. So every inline
figure chapter 4 needs is defined here, computed from the same CSVs the tables read, and used in
the prose as a macro:

    \input{thesis_figures_latex/num_ch4_inline}
    ... a median of \chFourCallsMedian{} verifier calls per report ...

Two outputs. ``num_ch4_inline.tex`` is the macro file LaTeX inputs; ``num_ch4_inline.md`` is the
same content as a table a human can read at a glance, which is what you actually want when
writing the paragraph.

A number whose source does not exist yet is NOT silently omitted and NOT guessed. Its macro is
defined to expand to a bold ``??``, which is LaTeX's own convention for an unresolved reference and
is impossible to miss in a draft, and the reason is listed at the top of both files and in
docs/thesis_map.md. That is the whole point of routing inline numbers through a build: a gap becomes
visible instead of becoming a plausible sentence.
"""
from __future__ import annotations

import os

import common as C

SCRIPT = "num_ch4_inline.py"


class Gap(Exception):
    """This number cannot be computed yet. The message says what would produce it."""


def thousands(value: float, places: int = 0) -> str:
    """*value* with LaTeX thin spaces between thousands."""
    return f"{value:,.{places}f}".replace(",", r"\,")


# -- one function per number --------------------------------------------------
# Each returns (rendered value, one-line provenance). Raising Gap is a legitimate outcome.

def _calls(statistic: str):
    if not C.has_table("calls_per_report"):
        raise Gap("needs calls_per_report.csv from exp13_22 (CPU; stage writes the per-report "
                  "call distribution, of which only the mean is currently kept)")
    rows = {r["statistic"]: C.f(r, "value") for r in C.table("calls_per_report")}
    if statistic not in rows:
        raise Gap(f"calls_per_report.csv has no statistic={statistic}")
    return thousands(rows[statistic]), f"calls_per_report.csv, statistic={statistic}"


def calls_median():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    return _calls("median")


def calls_mean():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    return _calls("mean")


def calls_p95():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    return _calls("p95")


def captured_mass_floor():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    if not C.has_table("captured_mass"):
        raise Gap("needs captured_mass.csv from exp13_22 (CPU; the cache already carries the "
                  "captured_mass column, ingest_score_cache keeps it)")
    row = C.table("captured_mass")[0]
    return f"{C.f(row, 'floor'):.3f}", "captured_mass.csv, floor"


def captured_mass_share():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    if not C.has_table("captured_mass"):
        raise Gap("needs captured_mass.csv from exp13_22 (CPU)")
    row = C.table("captured_mass")[0]
    return (f"{C.f(row, 'share_calls_below_floor') * 100:.1f}",
            "captured_mass.csv, share_calls_below_floor, as a percentage")


def replay_fidelity():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    if not C.has_table("replay_fidelity"):
        raise Gap("needs replay_fidelity.csv from exp13_22 (CPU; the cohort-scale version of "
                  "what tests/unit/test_tree_replay.py asserts on random DAGs)")
    worst = max(C.f(r, "max_abs_score_discrepancy") for r in C.table("replay_fidelity"))
    return f"{worst:.2e}", "replay_fidelity.csv, max over retrieval indices"


def containment_misses():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    rows = C.table("containment")
    return (str(int(max(C.f(r, "n_containment_misses") for r in rows))),
            "containment.csv, max over (index, pooling, S) cells")


def containment_cells():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    return str(len(C.table("containment"))), "containment.csv, row count"


def cached_nodes_max():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    rows = C.table("containment")
    return (thousands(max(C.f(r, "n_cached_nodes_max") for r in rows)),
            "containment.csv, n_cached_nodes_max")


def unreachable_below_floor():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    rows = C.table("containment")
    return (str(int(max(C.f(r, "n_unreachable_below_floor", 0.0) for r in rows))),
            "containment.csv, n_unreachable_below_floor")


def smooth_ece_raw():
    """The raw acceptance score's SmoothECE, as fig_ch4_reliability's note prints it."""
    if not C.has_table("calibration_crossfit"):
        raise Gap("calibration_crossfit.csv is missing -- exp13_22 `calibration` stage")
    row = {r["curve"]: r for r in C.table("calibration_crossfit")}.get("raw")
    if row is None:
        raise Gap("calibration_crossfit.csv has no `raw` row -- exp13_22 `calibration` stage")
    point, lo, hi = (C.f(row, k) for k in ("smooth_ece_bc", "smooth_ece_bc_lo", "smooth_ece_bc_hi"))
    return (f"{point:.3f} [{lo:.3f}, {hi:.3f}]",
            "calibration_crossfit.csv, curve=raw: bias-corrected SmoothECE, 95% basic bootstrap "
            "over reports (the note of fig_ch4_reliability)")


# -- the recall decomposition's reliability numbers (check Q20) ------------------
# The delta_m sweep runs at the modal configuration on every report, the decomposition itself on
# The pooled out-of-fold predictions. The prose says which is which.

def _sweep():
    if not C.has_table("delta_m_sensitivity"):
        raise Gap("delta_m_sensitivity.csv is missing -- exp13_22 `diagnostics` stage")
    return sorted(C.table("delta_m_sensitivity"), key=lambda r: C.f(r, "delta_m"))


def _sweep_value(column, end):
    rows = _sweep()
    row = rows[0] if end == "low" else rows[-1]
    return (str(int(C.f(row, column))),
            f"delta_m_sensitivity.csv, {column} at delta_m={C.f(row, 'delta_m'):g}")


def delta_m_max():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    row = _sweep()[-1]
    return f"{C.f(row, 'delta_m'):g}", "delta_m_sensitivity.csv, largest delta_m swept"


def delta_m_false_negatives():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    row = _sweep()[0]
    return (str(int(C.f(row, "n_false_negative"))),
            "delta_m_sensitivity.csv, n_false_negative (modal configuration)")


def residual_max():
    """Largest residual count anywhere: the decomposition and every delta_m of the sweep."""
    values = [C.f(r, "residual") for r in _sweep()]
    values += [C.f(r, "count") for r in C.table("recall_decomposition") if r["bucket"] == "residual"]
    return str(int(max(values))), "max residual over recall_decomposition.csv and the sweep"


def _blocking():
    if not C.has_table("blocking_causes") or not C.has_table("blocking_depth"):
        raise Gap("blocking_causes.csv / blocking_depth.csv missing -- exp13_22 `diagnostics`")
    causes = {r["cause"]: int(C.f(r, "count")) for r in C.table("blocking_causes")}
    depths = {int(C.f(r, "depth")): int(C.f(r, "n_missed_gold")) for r in C.table("blocking_depth")}
    return causes, depths


def blocking_total():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    causes, depths = _blocking()
    if sum(causes.values()) != sum(depths.values()):
        raise Gap("blocking_causes.csv and blocking_depth.csv count different pruning losses")
    return str(sum(causes.values())), "blocking_causes.csv, all causes"


def blocking_judgement():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    causes, _ = _blocking()
    return str(causes.get("judgement", 0)), "blocking_causes.csv, cause=judgement"


def blocking_shallow():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    _, depths = _blocking()
    return (str(sum(n for d, n in depths.items() if d <= 3)),
            "blocking_depth.csv, blocking ancestor at depth <= 3")


def _near_miss_share(max_distance):
    if not C.has_table("near_miss"):
        raise Gap("near_miss.csv is missing -- exp13_22 `diagnostics` stage")
    rows = {int(C.f(r, "distance")): C.f(r, "n_false_positive") for r in C.table("near_miss")}
    total = sum(rows.values())
    return (f"{100 * sum(n for d, n in rows.items() if d <= max_distance) / total:.0f}",
            f"near_miss.csv, share of false positives within {max_distance} edge(s) (percent)")


# -- the synthetic-sentence index (check X31) -----------------------------------
INDEX_STATS = "output/synthetic_sentence_statistics/index_statistics.csv"


def _index(statistic, render=lambda v: thousands(v)):
    import csv
    path = C.result_file(INDEX_STATS)
    if not path.is_file():
        raise Gap(f"{INDEX_STATS} is missing -- run scripts/index_statistics.py (no cluster)")
    with path.open(encoding="utf-8") as fh:
        rows = {r["statistic"]: float(r["value"]) for r in csv.DictReader(fh)}
    if statistic not in rows:
        raise Gap(f"{INDEX_STATS} has no {statistic}")
    return render(rows[statistic]), f"{INDEX_STATS}, {statistic}"


def peak_gpu_gb():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    raise Gap("GPU-BLOCKED: peak GPU memory was not recorded by exp13_21. Recoverable from SLURM "
              "job accounting (sacct MaxRSS / nvidia-smi logs) without a rerun; otherwise needs a "
              "GPU run with memory instrumentation. See figures/GAPS.md")


def deploy_hours_ont():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    raise Gap("GPU-BLOCKED: one encoder pass over 18 354 terms was not timed. Needs the exp13_24 "
              "(R3u) index build to record wall clock. See figures/GAPS.md")


def deploy_hours_ex():
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    raise Gap("GPU-BLOCKED: 70B example-sentence generation over 18 354 terms was not timed. Needs the "
              "exp13_21 (R1u) index build to record wall clock. See figures/GAPS.md")


# macro name, prose description, producer
NUMBERS = [
    (r"\chFourCallsMedian", "verifier calls per report, median", calls_median),
    (r"\chFourCallsMean", "verifier calls per report, mean", calls_mean),
    (r"\chFourCallsPNinetyFive", "verifier calls per report, p95", calls_p95),
    (r"\chFourCapturedMassFloor", "captured-mass floor", captured_mass_floor),
    (r"\chFourCapturedMassShare", "share of calls below the floor (percent)",
     captured_mass_share),
    (r"\chFourReplayDiscrepancy", "replay fidelity, max absolute score discrepancy",
     replay_fidelity),
    (r"\chFourContainmentMisses", "containment misses (must be 0)", containment_misses),
    (r"\chFourContainmentCells", "(index, pooling, S) cells gated", containment_cells),
    (r"\chFourCachedNodesMax", "nodes in the cache, maximum over cells", cached_nodes_max),
    (r"\chFourUnreachableBelowFloor", "(report, node) pairs unreachable below the floor",
     unreachable_below_floor),
    (r"\chFourSmoothECERaw", "SmoothECE of the raw acceptance score, with 95% CI",
     smooth_ece_raw),
    (r"\chFourDeltaMMax", "largest judgement margin swept", delta_m_max),
    (r"\chFourDeltaMFalseNegatives", "false negatives at the modal configuration",
     delta_m_false_negatives),
    (r"\chFourDeltaMJudgementLow", "judgement losses at delta_m = 0",
     lambda: _sweep_value("judgement", "low")),
    (r"\chFourDeltaMJudgementHigh", "judgement losses at the largest delta_m",
     lambda: _sweep_value("judgement", "high")),
    (r"\chFourDeltaMPoolingLow", "pooling losses at delta_m = 0",
     lambda: _sweep_value("pooling", "low")),
    (r"\chFourDeltaMPoolingHigh", "pooling losses at the largest delta_m",
     lambda: _sweep_value("pooling", "high")),
    (r"\chFourResidualMax", "largest residual count (decomposition and sweep)", residual_max),
    (r"\chFourBlockingTotal", "pruning losses with a blocking ancestor", blocking_total),
    (r"\chFourBlockingJudgement", "of which the verifier said No at the ancestor",
     blocking_judgement),
    (r"\chFourBlockingShallow", "of which the blocking ancestor is at depth <= 3",
     blocking_shallow),
    (r"\chFourNearMissOne", "false positives adjacent to an annotated term (percent)",
     lambda: _near_miss_share(1)),
    (r"\chFourNearMissTwo", "false positives within two edges of an annotated term (percent)",
     lambda: _near_miss_share(2)),
    (r"\chFourIndexSentences", "synthetic sentences over the terms below v0",
     lambda: _index("n_sentences")),
    (r"\chFourIndexTerms", "terms below v0 with a sentence file",
     lambda: _index("n_terms_with_file")),
    (r"\chFourIndexNoFile", "terms below v0 without a sentence file",
     lambda: _index("n_terms_without_file")),
    (r"\chFourIndexMedian", "sentences per term, median", lambda: _index("sentences_median")),
    (r"\chFourIndexQOne", "sentences per term, 25th percentile", lambda: _index("sentences_q1")),
    (r"\chFourIndexQThree", "sentences per term, 75th percentile",
     lambda: _index("sentences_q3")),
    (r"\chFourIndexMin", "fewest sentences for one term", lambda: _index("sentences_min")),
    (r"\chFourPeakGpuGb", "peak GPU memory, GB", peak_gpu_gb),
    (r"\chFourDeployHoursOnt", "deployment GPU-hours, ontology index", deploy_hours_ont),
    (r"\chFourDeployHoursEx", "deployment GPU-hours, example-sentence index", deploy_hours_ex),
]

MISSING = r"\textbf{??}"


def main() -> None:
    """Write the chapter 4 inline-number macros and their Markdown twin."""
    resolved, gaps = [], []
    for macro, description, producer in NUMBERS:
        try:
            value, source = producer()
            resolved.append((macro, description, value, source))
        except Gap as exc:
            gaps.append((macro, description, str(exc)))
        except SystemExit as exc:
            gaps.append((macro, description, str(exc)))

    C.TEX_DIR.mkdir(parents=True, exist_ok=True)
    tex = [
        "%% GENERATED FILE -- do not edit by hand.",
        f"%% Regenerate with: python figures/thesis_figures_scripts/{SCRIPT}",
        "%% Source: output/treephenorag_protocol/tables/",
        "%%",
        "%% Input this once in the preamble, then quote the macros in the prose:",
        r"%%     \input{thesis_figures_latex/num_ch4_inline}",
        r"%% A macro expanding to \textbf{??} has no source yet -- see figures/GAPS.md.",
        "",
    ]
    for macro, description, value, source in resolved:
        tex.append(f"%% {description} -- {source}")
        tex.append(rf"\newcommand{{{macro}}}{{{value}}}")
    if gaps:
        tex.append("")
        tex.append("%% ---- no source yet; these expand to ?? on purpose ----")
        for macro, description, reason in gaps:
            tex.append(f"%% {description}")
            tex.append(f"%%   {reason}")
            tex.append(rf"\newcommand{{{macro}}}{{{MISSING}}}")
    out = C.TEX_DIR / "num_ch4_inline.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  latex  -> {os.path.relpath(out, C.REPO)}  "
          f"({len(resolved)} resolved, {len(gaps)} awaiting a source)")

    md = [
        "# Chapter 4 inline numbers",
        "",
        f"Generated by `figures/thesis_figures_scripts/{SCRIPT}`. Do not edit.",
        "Quote these in the prose as macros, never as typed digits:",
        "`\\input{thesis_figures_latex/num_ch4_inline}`.",
        "",
        "| macro | number | value | source |",
        "|---|---|---|---|",
    ]
    for macro, description, value, source in resolved:
        md.append(f"| `{macro}` | {description} | {value.replace(chr(92) + ',', ' ')} | {source} |")
    if gaps:
        md += ["", "## Awaiting a source", "",
               "These expand to a bold `??` in the PDF, so a draft cannot quietly ship without",
               "them. See `figures/GAPS.md`.", "",
               "| macro | number | what would produce it |", "|---|---|---|"]
        for macro, description, reason in gaps:
            md.append(f"| `{macro}` | {description} | {reason} |")
    md_out = C.TEX_DIR / "num_ch4_inline.md"
    md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  notes  -> {os.path.relpath(md_out, C.REPO)}")


if __name__ == "__main__":
    main()
