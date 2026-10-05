#!/usr/bin/env python
r"""CH6-F2 -- the precision/recall plane with iso-$F_1$ contours, both cohorts.

    python experiments/figures/fig_ch6_pr_plane.py [--preview]

## Why this figure exists

A single $F_1$ column collapses the axis a reader actually needs. The roster's real structure is not
a ranking but a trade: PhenoBERT sits in the precision corner, AutoPCR in the recall corner, and
\textsc{PhenoJury} between them. Those are different systems for different jobs -- in a curation
workflow a false positive costs a reviewer's minute and a miss costs a phenotype -- and the choice
between them cannot be made from $F_1$.

The iso-$F_1$ contours are what make the trade legible rather than merely visible: two systems on
the same contour are equivalent under the metric the rest of the chapter reports, so the *distance
between contours* is the only part of the vertical spread that $F_1$ can see. Everything else is
the trade the number throws away.

Drawn with a light interval cross on each point (the micro-precision and micro-recall intervals),
because a position in this plane is no more certain than the numbers behind it.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C                                       # noqa: E402

import numpy as np                                           # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch6_pr_plane.py"

ISO_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def iso_f1(ax, levels=ISO_LEVELS):
    """Contours of constant F1 over the (recall, precision) plane.

    F1 = 2PR/(P+R), so the contour at level f is P = fR / (2R - f): a hyperbola that leaves the
    unit square for R <= f/2. Masking there, not clipping keeps the curve from drawing a
    spurious vertical wall at the left edge.
    """
    r = np.linspace(0.01, 1.0, 400)
    for f in levels:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = f * r / (2 * r - f)
        p[(2 * r - f) <= 0] = np.nan
        p[(p < 0) | (p > 1.02)] = np.nan
        ax.plot(r, p, color=C.GREY_LIGHT, lw=0.55, zorder=0)
        # Label where the contour LEAVES the drawn window -- the right edge for low levels, the
        # top edge for high ones. Searching the sampled curve instead lands outside the axes and
        # The label silently never appears, which is what happened first time round.
        (rx0, rx1), (py0, py1) = ax.get_xlim(), ax.get_ylim()
        p_right = f * rx1 / (2 * rx1 - f) if (2 * rx1 - f) > 0 else None
        if p_right is not None and py0 <= p_right <= py1:
            xy, ha, va, off = (rx1, p_right), "right", "bottom", (-1.5, 1.0)
        else:
            r_top = py1 * f / (2 * py1 - f) if (2 * py1 - f) > 0 else None
            if r_top is None or not (rx0 <= r_top <= rx1):
                continue
            xy, ha, va, off = (r_top, py1), "left", "top", (1.5, -1.0)
        ax.annotate((r"$F_1$=" if f == levels[0] else "") + f"{f:g}", xy=xy, color=C.GREY,
                    fontsize=5.8, ha=ha, va=va, xytext=off,
                    textcoords="offset points", zorder=0)


def build(results):
    """Draw every system in the precision-recall plane and write the figure and its LaTeX."""
    df = C.table(results, "t1_overall")
    cohorts = C.cohorts_present(df)
    methods = C.methods_present(df)

    fig, axes = plt.subplots(1, len(cohorts), figsize=(C.WIDTH_FULL, 3.0),
                             sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08))
    axes = [axes] if len(cohorts) == 1 else list(axes)

    for ax, cohort in zip(axes, cohorts):
        # Limits FIRST: `iso_f1` places each label where the contour leaves the window, so it has
        # to be able to read the real window. Drawing the contours before setting the limits left
        # The first panel labelling against matplotlib's default (0, 1) and putting every label
        # outside the final axes, where it silently vanished, while `sharex` made the second
        # panel look correct.
        ax.set_xlim(0.25, 0.90)
        ax.set_ylim(0.25, 0.90)
        iso_f1(ax)
        rows = C.by_method(df, cohort)
        absent = []
        for method in methods:
            row = rows.get(method)
            if row is None:
                absent.append(C.METHOD_LABEL[method])
                continue
            p, r = float(row["micro_precision"]), float(row["micro_recall"])
            colour = C.METHOD_COLOR[method]
            # Interval cross: the same report-level bootstrap the forest plot shows, on both axes.
            for key_lo, key_hi, horizontal in (("micro_recall_lo", "micro_recall_hi", True),
                                               ("micro_precision_lo", "micro_precision_hi", False)):
                if key_lo not in row or key_hi not in row:
                    continue
                lo, hi = float(row[key_lo]), float(row[key_hi])
                if lo != lo or hi != hi:
                    continue
                if horizontal:
                    ax.plot([lo, hi], [p, p], color=colour, lw=0.7, alpha=0.55, zorder=1)
                else:
                    ax.plot([r, r], [lo, hi], color=colour, lw=0.7, alpha=0.55, zorder=1)
            ax.plot([r], [p], marker=C.METHOD_MARKER[method], color=colour,
                    markersize=5.2, markeredgecolor="white", markeredgewidth=0.5, zorder=3)

        if absent:
            ax.annotate(f"{C.MISSING_TXT} {', '.join(absent)}", xy=(0.03, 0.03),
                        xycoords="axes fraction", fontsize=6.5, color=C.GREY,
                        ha="left", va="bottom")

        ax.set_title(C.COHORT_LABEL[cohort], pad=5)
        ax.set_xlabel("micro-recall")
        # Equal ranges on both axes (set above) so `aspect="equal"` yields a square panel.
        ax.set_aspect("equal", adjustable="box")
        C.style.light_grid(ax)
        C.style.despine(ax)

    axes[0].set_ylabel("micro-precision")

    handles = [plt.Line2D([], [], marker=C.METHOD_MARKER[m], color=C.METHOD_COLOR[m], ls="none",
                          markersize=5.0, label=C.METHOD_LABEL[m]) for m in methods]
    C.legend_below(fig, handles, ncol=4)
    return fig


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    C.prepare(args)
    fig = build(args.results)
    C.emit(fig, args, "fig_ch6_pr_plane.pdf", "fig_ch6_pr_plane.tex",
           caption=(r"Micro precision against micro recall of all systems on each cohort. Grey "
                    r"curves are iso-$F_1$ contours."),
           label="fig:ch6-pr-plane", generator=GENERATOR)


if __name__ == "__main__":
    main()
