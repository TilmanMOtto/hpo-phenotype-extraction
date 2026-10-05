#!/usr/bin/env python
r"""CH6 -- quality against cost on HCY: micro F1 against wall-clock seconds per report.

    python experiments/figures/fig_ch6_cost_quality.py [--preview]

Reads ``t1_overall.csv`` (F1 and its report-level bootstrap interval) and ``t4_cost.csv`` (seconds
per report), both the comparison's, for the HCY cohort only: the discussion's cost argument is about
clinical reports, and GSC+'s timings are a different corpus on different cards.

## Every note of the cost table is shown, not footnoted

A seconds figure is only as good as what the clock covered, and three rows of the cost table say
theirs covered something other than "this method, end to end". Each is stated where a reader cannot
miss, decided from the table's own columns rather than from method names:

* **lower bound** (``note`` says so -- PhenoJury's full pool, where two jurors hit the wall clock
  and have no single timing) and **partial scope** (``scope_partial`` -- PhenoBERT, whose CNN
  output was reused so only its linking pass was timed): named in the legend and the caption,
  since the true cost is larger;
* **off scale** (more than ``OFF_SCALE_FACTOR`` times every other system -- TreePhenoRAG, whose
  seconds are those of the exhaustive score cache): drawn past an axis break and labelled with its
  real seconds, so the linear axis keeps the rest readable;
* **no timing at all** (RAG-HPO): left out, and named in the caption.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C                                       # noqa: E402

import matplotlib.pyplot as plt                              # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch6_cost_quality.py"
COHORT = "hcy"
#: Drawn at this share of the text width and included at the same share -- never rescaled.
FRACTION = 0.8
#: A system costing more than this many times the dearest other one is drawn past an axis break.
OFF_SCALE_FACTOR = 20


def _flag(row, key) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def lower_bound(row) -> bool:
    """True when a cost row's seconds are a lower bound."""
    return "lower bound" in str(row.get("note", "")).lower() or _flag(row, "seconds_is_lower_bound")


def partial(row) -> bool:
    """True when a cost row covers only part of the pipeline."""
    return _flag(row, "scope_partial")


def exhaustive(row) -> bool:
    """True when a cost row is the run that scored every term."""
    return "exhaustive" in str(row.get("scope", "")).lower()


def _number(row, key):
    try:
        v = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _thin(v) -> str:
    return f"{v:,.0f}".replace(",", r"\,")


def seconds(row):
    """Seconds per report of a cost row, or None."""
    v = row.get("seconds_per_report")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def off_scale(drawn, cost) -> list:
    """Methods whose seconds dwarf every other system's, so a linear axis would flatten the rest.

    Decided from the numbers, not from names: a point is off scale if it costs more than
    ``OFF_SCALE_FACTOR`` times the dearest system that is not.
    """
    out = []
    for m in drawn:
        rest = [seconds(cost[o]) for o in drawn if o != m]
        if rest and seconds(cost[m]) > OFF_SCALE_FACTOR * max(rest):
            out.append(m)
    return out


def _break_mark(ax, x):
    """Two slanted strokes across the bottom spine at *x*: the axis is not continuous here."""
    trans = ax.get_xaxis_transform()
    ax.axvline(x, color=C.GREY_LIGHT, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for dx in (-2.2, 2.2):
        ax.plot([x + dx - 2.2, x + dx + 2.2], [-0.03, 0.03], transform=trans, color=C.INK,
                lw=0.8, clip_on=False, zorder=5)
    ax.plot([x - 2.4, x + 2.4], [0, 0], transform=trans, color="white", lw=2.5,
            clip_on=False, zorder=4.5)


def build(results):
    """Draw micro F1 against seconds per report on HCY and write the figure and its LaTeX."""
    quality = C.by_method(C.table(results, "t1_overall"), COHORT)
    cost_df = C.table(results, "t4_cost")
    cost = {r["method"]: r for _, r in cost_df[cost_df["cohort"] == COHORT].iterrows()}

    methods = [m for m in C.METHOD_ORDER if m in quality]
    drawn = [m for m in methods if m in cost and seconds(cost[m]) is not None]
    omitted = [m for m in methods if m not in drawn]
    if not drawn:
        raise SystemExit(C.MISSING_SOURCE_EXIT)
    far = off_scale(drawn, cost)

    # A linear axis over the systems that share one. An off-scale system sits past a break, at
    # a position that carries no magnitude and is labelled with its real seconds instead.
    on_max = max(seconds(cost[m]) for m in drawn if m not in far)
    step = 50 if on_max > 100 else 10
    last_tick = step * int(on_max // step)
    scale_end = on_max * 1.08
    brk = scale_end + 0.05 * scale_end
    far_x = {m: brk + (0.10 + 0.07 * i) * scale_end for i, m in enumerate(far)}

    fig, ax = plt.subplots(figsize=(C.WIDTH_FULL * FRACTION, 3.0))
    handles = []
    for m in drawn:
        q, c = quality[m], cost[m]
        x = far_x.get(m, seconds(c))
        y = float(q["micro_f1"])
        lo, hi = float(q["micro_f1_lo"]), float(q["micro_f1_hi"])
        col = C.METHOD_COLOR[m]
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col, elinewidth=0.9,
                    capsize=2.0, zorder=2)
        ax.plot([x], [y], marker="o", ms=6.5, color=col, mec="white", mew=0.6, ls="none",
                zorder=4)
        label = C.METHOD_LABEL[m]
        if partial(c):
            label += " (linking pass)"
        elif lower_bound(c):
            label += ", lower bound"
        if m in far:
            label += f" (off scale: {seconds(c) / 3600:.1f} h)"
        handles.append(plt.Line2D([], [], marker="o", color=col, ls="none", ms=6,
                                  mec="white", mew=0.5, label=label))

    ax.set_xlabel("seconds per report")
    ax.set_ylabel("micro $F_1$ on HCY")
    ax.set_ylim(0, max(float(quality[m]["micro_f1_hi"]) for m in drawn) + 0.1)
    ax.set_xlim(-0.03 * scale_end, max(far_x.values(), default=scale_end) + 0.06 * scale_end)
    ticks = list(range(0, last_tick + 1, step))
    ax.set_xticks(ticks + [far_x[m] for m in far],
                  [str(t) for t in ticks] + [f"{seconds(cost[m]):,.0f}".replace(",", r"$\,$")
                                             for m in far])
    C.style.light_grid(ax, axis="y")
    C.style.despine(ax)
    if far:
        _break_mark(ax, brk)
    C.legend_below(fig, handles, ncol=2)
    return fig, drawn, omitted, cost, far


def caption(drawn, omitted, cost, far) -> "tuple[str, str]":
    """The caption (what is shown) and the note below the figure (everything else)."""
    title = (r"Performance against cost on HCY. $F_{1,\mu}$ against wall-clock seconds per report "
             r"for every system with recorded cost.")
    parts = [r"$F_{1,\mu}$ from \Cref{tab:ch6-overall-comparison}, seconds from "
             r"\Cref{tab:ch6-cost-detail}."]
    if far:
        names = " and ".join(C.METHOD_LABEL_TEX[m] for m in far)
        secs = " and ".join(f"{seconds(cost[m]):,.0f}".replace(",", r"\,") for m in far)
        parts.append(names + r" is drawn past the axis break at " + secs + r" seconds" +
                     (r", the time of its exhaustive scoring run" if any(exhaustive(cost[m]) for m in far) else "")
                     + ".")
    elif any(exhaustive(cost[m]) for m in drawn):
        parts.append(r"TreePhenoRAG's seconds are those of the exhaustive scoring run.")
    # The exhaustive run OVERSTATES the selected setting: say by how much, in calls, since the
    # selected setting's own seconds were never measured (it is a CPU re-run of that run).
    for m in drawn:
        run, op = _number(cost[m], "llm_calls_per_report"), _number(cost[m], "operating_point_calls_per_report")
        if exhaustive(cost[m]) and run and op:
            parts.append(r"At the selected setting it makes " + f"{100 * op / run:.0f}"
                         + r"\,\% of that run's verifier calls (" + _thin(op) + " of " + _thin(run)
                         + r" per report). The selected setting's wall-clock time was not measured.")
    # Each understated cost says so in its own sentence; "for both" only when both are drawn,
    # so it can never be read as covering the exhaustive run above.
    bounded, scoped = any(lower_bound(cost[m]) for m in drawn), any(partial(cost[m]) for m in drawn)
    higher = r" true cost is higher than drawn."
    if bounded:
        parts.append(r"PhenoJury's full-pool seconds are a lower bound summed over jurors on "
                     r"separate GPUs" + ("." if scoped else r", so its" + higher))
    if scoped:
        parts.append(r"PhenoBERT's seconds cover its linking pass only"
                     + (r". For both, the" + higher if bounded else r", so its" + higher))
    if omitted:
        names = sorted({C.METHOD_LABEL_TEX[m].rsplit(" ", 1)[0] for m in omitted})
        parts.append(" and ".join(names) + r" is omitted: no time was recorded for its runs.")
    return title, " ".join(parts)


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    C.prepare(args)
    fig, drawn, omitted, cost, far = build(args.results)
    pdf = os.path.join(args.figdir, "fig_ch6_cost_quality.pdf")
    C.style.save(fig, pdf, preview_png=pdf.replace(".pdf", ".png") if args.preview else None)
    title, note = caption(drawn, omitted, cost, far)
    C.style.write_tex(os.path.join(args.texdir, "fig_ch6_cost_quality.tex"),
                      "fig_ch6_cost_quality.pdf", title, "fig:ch6-cost-quality",
                      width=f"{FRACTION:g}\\textwidth", generator=GENERATOR, note=note)


if __name__ == "__main__":
    main()
