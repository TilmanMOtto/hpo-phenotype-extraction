#!/usr/bin/env python
r"""Figure: what PhenoJury's false positives are, one stacked bar per cohort.

    python experiments/figures/fig_ch5_error_analysis.py

Reads ``s9_error_taxonomy.csv`` for each cohort -- the selected jury's pooled out-of-fold
predictions, filed by the same function and the same precedence order as chapter 4's
``fig_ch4_error_analysis`` (a term goes to the first bucket it matches), and drawn in the same
colours so the two chapters' bars can be read against each other.

``invalid`` and ``no_gold`` are not drawn: neither relates a false positive to an annotated term.
A non-empty one is named in the caption with its count, so a false positive left out of the bar
is never left out silently. The shares are of the false positives drawn, and each bar's n says
how many that is.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C                       # noqa: E402
from common import FP_RELATION_SEGMENTS, stacked_share_bars  # noqa: E402
import matplotlib.pyplot as plt              # noqa: E402

FIGNAME = "fig_ch5_error_analysis.pdf"
TEXNAME = "fig_ch5_error_analysis.tex"
LABEL = "fig:ch5-error-analysis"
GENERATOR = "figures/thesis_figures_scripts/fig_ch5_error_analysis.py"

# The same segments object fig_ch4_error_analysis draws: one colour per bucket across both chapters.
SEGMENTS = FP_RELATION_SEGMENTS
#: Buckets that relate a false positive to no annotated term: the caption's words for (1, many).
UNDRAWN = {
    "no_gold": ("falls on a report with 0 annotations", "fall on reports with 0 annotations"),
    "invalid": ("is not an HPO identifier", "are not HPO identifiers"),
}
NUMBER_WORD = {1: "One", 2: "Two", 3: "Three"}


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    C.prepare(args)

    bars, undrawn, unrelated = [], [], {}
    for cohort in C.COHORTS:
        if not C.has_cohort(args.results, cohort):
            continue
        tax = C.table(args.results, cohort, "s9_error_taxonomy")
        counts = {str(r["bucket"]): int(r["count"]) for _, r in tax.iterrows()}
        drawn = sum(counts.get(k, 0) for k, _, _ in SEGMENTS)
        unrelated[cohort] = counts.get("unrelated", 0) / drawn if drawn else 1.0
        bars.append((f"{C.COHORT_SHORT[cohort]} ({drawn} FP)", counts))
        for bucket, (one, many) in UNDRAWN.items():
            n = counts.get(bucket, 0)
            if n:
                undrawn.append(f" {NUMBER_WORD.get(n, n)} {C.COHORT_SHORT[cohort]} "
                               + (f"false positive {one} and is not drawn." if n == 1 else
                                  f"false positives {many} and are not drawn."))
    if not bars:
        raise SystemExit(C.MISSING_SOURCE_EXIT)

    fig, ax = plt.subplots(figsize=(C.style.WIDTH_FULL, 1.75))
    handles = stacked_share_bars(ax, bars, SEGMENTS)
    ax.set_xlabel("share of false positives (%)")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=C.style.TICK_FONT_SIZE)

    caption = (r"Relation of PhenoJury's false positives to the annotated terms of the report, per "
               r"cohort.")
    note = (r"Pooled out-of-fold predictions, categories of \Cref{sec:metrics} in the order shown, "
            r"with the buckets and colours of \Cref{fig:ch4-error-analysis}. Ancestor or "
            r"descendant: on an annotated term's path. Sibling: shares a direct parent with an "
            r"annotated term. Same branch: shares a first-level organ system. Unrelated: none of "
            r"these." + "".join(undrawn))
    C.emit(fig, args, FIGNAME, TEXNAME, caption, LABEL, GENERATOR, note=note)


if __name__ == "__main__":
    main()
