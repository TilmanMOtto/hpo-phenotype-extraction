#!/usr/bin/env python
r"""CH6-F1 -- the main forest plot: micro-F1 with 95\% intervals, both cohorts.

    python experiments/figures/fig_ch6_headline.py [--preview]

## Why a forest plot and not a bar chart

The discussion's one essential claim is a statement about *separation*, not about height. On HCY
PhenoJury's lower bound clears every baseline's upper bound. On GSC+ it does not, and PhenoBERT
leads. A bar chart encodes the point estimate and hides the interval, which is the part of
the comparison that decides whether a sentence may be written. A forest plot makes "the intervals
do not overlap" the visual primitive.

Three things the figure carries that a table row cannot:

* **the estimator**, as marker shape. Three different quantities share the axis -- a plain cohort
  estimate for methods with nothing to tune, a pooled out-of-fold estimate for the two whose
  configuration is chosen inside nested CV, and a pure transfer for TreePhenoRAG on GSC+ where
  nothing at all is fitted on the cohort. The last is the strictest and would read as the weakest
  unmarked.
* **the best baseline's interval**, as a shaded band spanning the panel, so "clears every baseline"
  is checkable by eye rather than by reading twelve numbers.
* **the gap**, as an explicit em dash where TreePhenoRAG's GSC+ row will go.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C                                       # noqa: E402

import matplotlib.pyplot as plt                              # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch6_headline.py"

#: Methods that are ours. The band is drawn against the best of the *others*, so this is the
#: partition the claim is about.
OURS = {"phenojury", "treephenorag"}


def build(results):
    """Draw micro F1 of every system per cohort and write the figure and its LaTeX."""
    df = C.table(results, "t1_overall")
    cohorts = C.cohorts_present(df)
    methods = C.methods_present(df)

    fig, axes = plt.subplots(
        1, len(cohorts), figsize=(C.WIDTH_FULL, 3.1),
        sharey=True, gridspec_kw=dict(wspace=0.08))
    axes = [axes] if len(cohorts) == 1 else list(axes)

    ypos = {m: len(methods) - 1 - i for i, m in enumerate(methods)}
    seen_estimators = []

    for ax, cohort in zip(axes, cohorts):
        rows = C.by_method(df, cohort)

        # The comparator band: the best EXTERNAL baseline's interval. Everything to its right at
        # non-overlapping distance is a system that beats every published method on this cohort.
        external = [r for k, r in rows.items() if k not in OURS]
        if external:
            best = max(external, key=lambda r: float(r["micro_f1"]))
            ax.axvspan(float(best["micro_f1_lo"]), float(best["micro_f1_hi"]),
                       color=C.GREY_LIGHT, alpha=0.45, lw=0, zorder=0)
            ax.axvline(float(best["micro_f1"]), color=C.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)

        for method in methods:
            y = ypos[method]
            row = rows.get(method)
            if row is None:
                C.missing_marker(ax, 0.5, y)
                continue
            est = C.estimator_short(row.get("estimator"))
            if est not in seen_estimators:
                seen_estimators.append(est)
            lo, hi, point = (float(row["micro_f1_lo"]), float(row["micro_f1_hi"]),
                             float(row["micro_f1"]))
            ax.plot([lo, hi], [y, y], color=C.METHOD_COLOR[method], lw=1.4,
                    solid_capstyle="butt", zorder=2)
            for edge in (lo, hi):                     # interval caps, so the ends are unambiguous
                ax.plot([edge, edge], [y - 0.16, y + 0.16],
                        color=C.METHOD_COLOR[method], lw=1.0, zorder=2)
            ax.plot([point], [y], marker=C.ESTIMATOR_MARKER.get(est, "o"),
                    color=C.METHOD_COLOR[method], markersize=4.6,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)

        ax.set_title(C.COHORT_LABEL[cohort], pad=5)
        ax.set_xlabel(r"micro-$F_1$")
        ax.set_xlim(0.20, 0.90)
        ax.set_ylim(-0.7, len(methods) - 0.3)
        C.style.light_grid(ax, axis="x")
        C.style.despine(ax)

    axes[0].set_yticks([ypos[m] for m in methods])
    axes[0].set_yticklabels([C.METHOD_LABEL[m] for m in methods])

    # One legend for the estimator, which is the figure's second dimension.
    handles = [plt.Line2D([], [], marker=C.ESTIMATOR_MARKER[e], color=C.INK, ls="none",
                          markersize=4.6, label=e)
               for e in C.ESTIMATOR_ORDER if e in seen_estimators]
    handles.append(plt.Line2D([], [], color=C.GREY_LIGHT, lw=6, alpha=0.7,
                              label="best baseline 95% CI"))
    # Upper right of the LEFT panel: the only region empty in every row on both cohorts,
    # checked against the data, not assumed (the top rows are RAG-HPO, whose
    # intervals end well short of it).
    axes[0].legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.5,
                   borderaxespad=0.4, labelspacing=0.3, fontsize=6.8)
    return fig


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    C.prepare(args)
    fig = build(args.results)
    C.emit(fig, args, "fig_ch6_headline.pdf", "fig_ch6_headline.tex",
           caption=(r"\textbf{$F_{1,\mu}$ of every system on each cohort.} "
                    r"Marker shape: scored once with nothing tuned (circle), pooled out-of-fold "
                    r"under nested cross-validation (square), or HCY-selected configuration applied "
                    r"unchanged (diamond). Shaded band: the interval of the best external system on "
                    r"that cohort. " + C.MISSING_NOTE),
           label="fig:ch6-headline", generator=GENERATOR)


if __name__ == "__main__":
    main()
