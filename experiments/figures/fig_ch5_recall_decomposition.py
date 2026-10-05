#!/usr/bin/env python
r"""Figure: where PhenoJury's missed annotated terms are, one stacked bar per cohort.

    python experiments/figures/fig_ch5_recall_decomposition.py

Reads ``s9_recall_decomposition.csv`` for each cohort. Three buckets, exhaustive over the selected
jury's false negatives:

  specificity  a juror named a term on the annotated term's own path, but not the annotated term
  vote         the annotated term was in some juror's candidates and did not reach k
  naming       no juror's wording resolved to the annotated term

Shares of each cohort's own false negatives, so the two bars are comparable although HCY and GSC+
have different numbers of misses. Each bar's n is in its label. When the PhenoJury protocol reports the last two
buckets merged for a cohort (``merged``), the merged bucket is drawn as one segment and named so.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C                       # noqa: E402
from common import stacked_share_bars        # noqa: E402
import palette as P                          # noqa: E402
import matplotlib.pyplot as plt              # noqa: E402

FIGNAME = "fig_ch5_recall_decomposition.pdf"
TEXNAME = "fig_ch5_recall_decomposition.tex"
LABEL = "fig:ch5-recall-decomposition"
GENERATOR = "figures/thesis_figures_scripts/fig_ch5_recall_decomposition.py"

# Errors, so the palette's error ramp, light to dark in drawn order. The merged bucket only appears
# when vote and naming cannot be told apart, and takes the step between them. Colour alone
# separates the segments (no hatching). Each is also labelled with its share.
_RAMP = P.error_ramp(4)
SEGMENTS = [
    ("specificity", "specificity (wrong ontology level)", dict(color=_RAMP[0])),
    ("vote", "vote (proposed, outvoted)", dict(color=_RAMP[1])),
    ("naming", "naming or normalisation", dict(color=_RAMP[3])),
    ("naming_normalisation", "vote or naming (merged)", dict(color=_RAMP[2])),
]


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    C.prepare(args)

    bars, used, n_fn, shares = [], set(), [], {}
    for cohort in C.COHORTS:
        if not C.has_cohort(args.results, cohort):
            continue
        dec = C.table(args.results, cohort, "s9_recall_decomposition")
        counts = {str(r["bucket"]): int(r["count"]) for _, r in dec.iterrows()}
        used |= set(counts)
        shares[cohort] = {k: v / sum(counts.values()) for k, v in counts.items()}
        bars.append((f"{C.COHORT_SHORT[cohort]} ({sum(counts.values())} FN)", counts))
        # GSC+ carries its document count, as in the chapter's text: "GSC+ (206)".
        name = C.COHORT_SHORT[cohort]
        if cohort == "gsc206":
            name += f" ({C.manifest(args.results, cohort)['n_reports']})"
        n_fn.append(f"{name}: {sum(counts.values())}")
    if not bars:
        raise SystemExit(C.MISSING_SOURCE_EXIT)
    segments = [s for s in SEGMENTS if s[0] in used]

    fig, ax = plt.subplots(figsize=(C.style.WIDTH_FULL, 1.75))
    handles = stacked_share_bars(ax, bars, segments)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=C.style.TICK_FONT_SIZE)

    caption = r"Cause of each false negative of PhenoJury, as a share per cohort."
    note = (r"Pooled out-of-fold predictions. " + n_fn[0] + " false negatives"
            + "".join(", " + x for x in n_fn[1:]) + r". Specificity: a juror named a term on the "
            r"annotated term's path, but not the term itself. Vote: the annotated term was among "
            r"some juror's candidates but received fewer than $k$ votes. Naming or normalisation: "
            r"no juror's wording was resolved to the annotated term.")
    C.emit(fig, args, FIGNAME, TEXNAME, caption, LABEL, GENERATOR, note=note)


if __name__ == "__main__":
    main()
