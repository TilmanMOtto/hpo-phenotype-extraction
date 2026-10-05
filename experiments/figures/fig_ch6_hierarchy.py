#!/usr/bin/env python
r"""CH6-F4 -- flat $F_1$ against CoPHE-$F$: how much of each system's error is a near miss.

    python experiments/figures/fig_ch6_hierarchy.py [--preview]

## What the slope encodes, and the objection it answers

Flat $F_1$ gives zero credit for predicting *Abnormality of the nervous system* when the annotated term
is *Seizure*. CoPHE propagates counts through the ontology, so an ancestor earns partial credit --
but, unlike plain ancestor closure, it does not let over-prediction vanish: several predicted
descendants of one ground truth ancestor survive as false positives rather than collapsing onto it.

So the **rise** from flat to CoPHE is the share of a system's errors that are ontologically near
misses, and the figure is a slope chart because that rise is the quantity, not either endpoint.

Two readings the discussion needs:

* \textsc{PhenoJury}'s lead **survives** the change of metric. That is the answer to the obvious
  objection -- that an ensemble wins flat $F_1$ merely by emitting more terms -- because CoPHE is
  the metric designed to punish that.
* RAG-HPO rises furthest while PhenoBERT rises least: RAG-HPO's errors are disproportionately
  wrong-level, not wrong-branch, which is a different failure and suggests a different fix.

CoPHE-$F$ is the figure quoted in the text and $hF$ is reported beside it in the table for
comparability with the wider literature. Plotting the pair here would make a three-way slope that
says less than the two-way one.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C                                       # noqa: E402

import matplotlib.pyplot as plt                              # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch6_hierarchy.py"

LEFT, RIGHT = 0.0, 1.0
YLIM = (0.25, 0.95)
AXIS_LABEL = [r"flat micro-$F_1$", r"CoPHE-$F$"]


def build(results):
    """Draw flat against hierarchy-aware F scores and write the figure and its LaTeX."""
    flat = C.table(results, "t1_overall")
    hier = C.table(results, "t2_hierarchy")
    cohorts = C.cohorts_present(flat)
    methods = C.methods_present(flat)

    fig, axes = plt.subplots(1, len(cohorts), figsize=(C.WIDTH_FULL, 3.2),
                             sharey=True, gridspec_kw=dict(wspace=0.10))
    axes = [axes] if len(cohorts) == 1 else list(axes)

    for ax, cohort in zip(axes, cohorts):
        flat_rows, hier_rows = C.by_method(flat, cohort), C.by_method(hier, cohort)
        absent, drawn = [], []
        for method in methods:
            a, b = flat_rows.get(method), hier_rows.get(method)
            if a is None or b is None:
                absent.append(C.METHOD_LABEL[method])
                continue
            lo, hi = float(a["micro_f1"]), float(b["micro_cophe_f1"])
            colour = C.METHOD_COLOR[method]
            ax.plot([LEFT, RIGHT], [lo, hi], color=colour, lw=1.2, zorder=2,
                    marker=C.METHOD_MARKER[method], markersize=4.6,
                    markeredgecolor="white", markeredgewidth=0.5)
            drawn.append((hi, hi - lo, colour))

        # The rise is the figure's content, so it is written on the line. Where systems converge
        # The labels would stack into an unreadable blur, so their TEXT is spread apart while the
        # markers stay put. A leader tick keeps each label attached to its own line.
        if drawn:
            ys = C.spread([d[0] for d in drawn], min_gap=0.032,
                          bounds=(YLIM[0] + 0.02, YLIM[1] - 0.02))
            for (hi, delta, colour), y in zip(drawn, ys):
                ax.annotate(f"+{delta:.3f}", xy=(RIGHT, hi), xytext=(RIGHT + 0.10, y),
                            fontsize=6.2, color=colour, ha="left", va="center",
                            arrowprops=dict(arrowstyle="-", color=colour, lw=0.45,
                                            shrinkA=0.5, shrinkB=1.5))
        if absent:
            ax.annotate(f"{C.MISSING_TXT} {', '.join(absent)}", xy=(0.5, 0.02),
                        xycoords="axes fraction", fontsize=6.5, color=C.GREY,
                        ha="center", va="bottom")

        ax.set_title(C.COHORT_LABEL[cohort], pad=5)
        ax.set_xticks([LEFT, RIGHT])
        ax.set_xticklabels(AXIS_LABEL, fontsize=7.5)
        ax.set_xlim(-0.22, 1.42)
        ax.set_ylim(*YLIM)
        C.style.light_grid(ax, axis="y")
        C.style.despine(ax)

    axes[0].set_ylabel("score")
    handles = [plt.Line2D([], [], marker=C.METHOD_MARKER[m], color=C.METHOD_COLOR[m],
                          lw=1.2, markersize=4.6, label=C.METHOD_LABEL[m]) for m in methods]
    C.legend_below(fig, handles, ncol=4)
    return fig


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    C.prepare(args)
    fig = build(args.results)
    C.emit(fig, args, "fig_ch6_hierarchy.pdf", "fig_ch6_hierarchy.tex",
           caption=(r"Flat $F_{1,\mu}$ and hierarchy-aware CoPHE-$F$ of all systems on each "
                    r"cohort. Each line joins a system's $F_{1,\mu}$ to its CoPHE-$F$."),
           label="fig:ch6-hierarchy", generator=GENERATOR)


if __name__ == "__main__":
    main()
