#!/usr/bin/env python
r"""F5.1 -- the aggregation curves: micro F1 against k, by vote scope and matching rule.

THE OPERATING-POINT FIGURE of chapter 5. Three things have to be chosen together before a jury can
answer at all, and this figure is where the reader sees that they interact:

  k       how many of the eight jurors must agree                   -- the x axis
  scope   what counts as "the same finding" when jurors are compared -- the three lines
  rule    whether a match must be exact or may use ancestor closure  -- the two panels

Prompt and normaliser are held at the point the development split selected, and the jury is the
FULL POOL: the PhenoJury protocol's `stage_curves` votes over every juror of the prompt, because the jury itself
is selected per fold afterwards and has no single membership to draw. That is deliberate. Varying a fourth axis here would make the figure a grid rather than an
argument, and the chapter already has a table for every axis separately.

Why scope counts more than it sounds: a report-level vote lets two jurors agree on a term they
found in different sentences, which is a weaker kind of agreement than segment-level and inflates
recall at k > 1. Segment-level is the strict reading. If the curves order differently under
CLOSURE than under EXACT, then the matching rule and the scope cannot be chosen independently --
which is the finding H5.3 is about.

If s3_curves.csv predates the rule/scope sweep it will carry only the single (exact, report)
series. The figure then draws that one line and says so, in the axes and in the caption,
not quietly re-labelling a one-line plot as a three-line comparison. See docs/thesis_map.md.

    python experiments/figures/fig_ch5_aggregation_curves.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C                       # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402

FIGNAME = "fig_ch5_aggregation_curves.pdf"
TEXNAME = "fig_ch5_aggregation_curves.tex"
LABEL = "fig:ch5-aggregation-curves"
GENERATOR = "figures/thesis_figures_scripts/fig_ch5_aggregation_curves.py"

# The cohort whose configuration the figure reports. HCY is primary throughout the chapter;
# GSC+ is a transfer check and gets its own panel only where the comparison is the point.
PRIMARY = "hcy"


def series(df, rule, scope):
    """The (rule, scope) curve, tolerating a CSV that predates either column."""
    sub = df
    if "rule" in df.columns:
        sub = sub[sub["rule"] == rule]
    elif rule != C.RULE_ORDER[0]:
        return None                    # only the default rule exists in this CSV
    if "unit" in df.columns:
        sub = sub[sub["unit"] == scope]
    elif scope != "report":
        return None                    # only the report scope exists in this CSV
    return sub.sort_values("k") if not sub.empty else None


def panel(ax, df, rule, show_ylabel, partial):
    """Draw F1 against k for every vote scope of one matching rule."""
    drawn = 0
    for scope in C.SCOPE_ORDER:
        sub = series(df, rule, scope)
        if sub is None or sub.empty:
            continue
        drawn += 1
        col, mk = C.SCOPE_COLOR[scope], C.SCOPE_MARKER[scope]
        ax.plot(sub["k"], sub["micro_f1"], color=col, lw=1.6, marker=mk, ms=4.2,
                label=C.SCOPE_LABEL[scope], zorder=3)
        # The selected k is a choice, so mark where each curve would choose to sit.
        best = sub.loc[sub["micro_f1"].idxmax()]
        ax.plot(best["k"], best["micro_f1"], marker=mk, color=col, ms=8.0, mec="white",
                mew=1.0, zorder=4)

    ax.set_xlabel("$k$ (jurors that must agree)")
    if show_ylabel:
        ax.set_ylabel("micro F1")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0, 1.0)
    ax.set_title(C.RULE_LABEL[rule], fontsize=C.style.BASE_FONT_SIZE, pad=3)
    C.style.light_grid(ax, axis="y")
    C.style.despine(ax)

    if not drawn:
        # An empty axes with a title is worse than an empty axes that says why it is empty.
        ax.text(0.5, 0.5, "not yet computed\n(see GAPS.md)", transform=ax.transAxes,
                ha="center", va="center", fontsize=C.style.LEGEND_FONT_SIZE, color=C.GREY,
                style="italic")
    elif partial:
        ax.text(0.5, 0.06, "report scope only", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=C.style.LEGEND_FONT_SIZE, color=C.GREY, style="italic")


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    C.prepare(args)

    cohort = PRIMARY if C.has_cohort(args.results, PRIMARY) else next(
        (c for c in C.COHORTS if C.has_cohort(args.results, c)), None)
    if cohort is None:
        raise SystemExit(f"no cohort tables under {args.results}")

    df = C.table(args.results, cohort, "s3_curves")
    man = C.manifest(args.results, cohort)
    sel_norm = man.get("selected_normaliser", "phenobert_candidates")
    sel_prompt = man.get("selected_prompt", "p0_baseline")
    if "normaliser" in df.columns:
        held = df[df["normaliser"] == sel_norm]
        df = held if not held.empty else df
    if "prompt" in df.columns:
        held = df[df["prompt"] == sel_prompt]
        df = held if not held.empty else df

    # The curves vote over every juror of the prompt. Say how many, from the CSV itself.
    n_pool = int(df["n_jurors"].max()) if "n_jurors" in df.columns else 8

    # Which of the two specified axes the CSV actually carries. This decides both the figure and
    # what the caption is allowed to claim.
    have_rule = "rule" in df.columns
    have_scope = "unit" in df.columns
    partial = not have_scope
    missing = [name for name, ok in (("matching rule", have_rule), ("vote scope", have_scope))
               if not ok]

    fig, axes = plt.subplots(1, 2, figsize=(C.style.WIDTH_FULL, 2.95), sharey=True)
    for i, (ax, rule) in enumerate(zip(axes, C.RULE_ORDER)):
        panel(ax, df, rule, show_ylabel=(i == 0), partial=partial)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False,
                   fontsize=C.style.LEGEND_FONT_SIZE, handletextpad=0.4, columnspacing=1.2)

    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    pool = words.get(n_pool, str(n_pool))
    caption = (r"$F_{1,\mu}$ of the " + pool + r"-juror pool against the vote threshold $k$ on the "
               + C.COHORT_SHORT[cohort] + r" development split. Left: exact rule. Right: closure "
               r"rule.")
    note = (
        r"$F_{1,\mu}$ of the " + pool + r"-juror pool as the number of jurors that must agree, $k$, "
        r"rises from 1 (any juror) to " + str(n_pool) + r" (all). \emph{"
        + C.tex_escape(C.PROMPT_LABEL.get(sel_prompt, sel_prompt)) + r"} prompt, "
        + C.NORMALISER_SHORT.get(sel_norm, C.tex_escape(sel_norm)) + r" normaliser. Left: exact "
        r"rule (a juror supports only the terms it named). Right: closure rule (a juror also "
        r"supports every ancestor of a term it named). Lines: the unit within which jurors must "
        r"agree, the report, a window of $\pm1$ sentence, or a segment. Enlarged markers: the best "
        r"$k$ of each curve.")
    if missing:
        note += (r" \emph{Incomplete:} \texttt{s3\_curves.csv} does not yet carry the "
                 + " or ".join(missing) + r" axis, so only the series it holds is drawn. "
                 r"This figure is not the specified comparison until exp14\_04 sweeps "
                 r"$\{$exact, closure$\}\times\{$report, window, segment$\}$, see "
                 r"\texttt{figures/GAPS.md}.")
        print(f"[fig] partial: s3_curves.csv lacks {', '.join(missing)}")

    C.emit(fig, args, FIGNAME, TEXNAME, caption, LABEL, GENERATOR, note=note)


if __name__ == "__main__":
    main()
