#!/usr/bin/env python
r"""CH6-F3 -- the decisive HCY slices, as three panels.

    python experiments/figures/fig_ch6_subgroups.py [--preview]

## Why this is the figure the experiment exists for

"Recall $= 0.48$" hides every reason a term was missed. The curation pass recorded *why* each
annotation was hard, and two of its qualifiers name slices where no string match to the term
exists at all: a **lab value** (the evidence is a number with a unit) and an **implicit**
description (named nowhere. Only a reader who understands the sentence finds it).

The claim the discussion wants to make is not "\textsc{PhenoJury} is better" -- the aggregate
already says that, by 0.08 over PhenoBERT. It is that the advantage is **concentrated where
reading the sentence is the only route**, which is a claim about a *difference of differences* and
is invisible in any table of subgroup recalls on its own.

So each bar in the first two panels is drawn against that system's own aggregate recall on the full
cohort, as a caret on the same axis. A bar far above its caret is a system that finds these hard
terms at better than its usual rate. A bar below is one for which the slice is harder than the
cohort. That comparison is the figure's whole content, and it is why the aggregate is plotted
rather than mentioned.

## The third panel is a count, and is drawn so it cannot be read as a rate

Family attribution is the mirror image of the other two: a finding the report attributes to a
*relative*, which the curation policy excludes from the ground truth, so every one a system emits is
already a false positive in Table~\ref{tab:ch6-overall-comparison}. Twelve such pairs exist, in
five reports.

Twelve events support no rate, and this module used to refuse the panel on that ground.
The panel is here now because the *count* is the finding -- it says which system is reading "her
mother has..." as a statement about the patient -- but the refusal still stands, so the panel is
drawn as a count and nothing about it invites a proportion: its own 0-12 axis with integer ticks,
a hatched fill that no other panel uses, no confidence interval, no aggregate caret, and a label
saying lower is better. It is the one panel where down is good.

HCY only: GSC+ has no curation pass and therefore no qualifiers. Saying so in the caption is
cheaper than a reader wondering where the second cohort went.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C                                       # noqa: E402

import numpy as np                                           # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch6_subgroups.py"

# Titles and descriptions are kept narrower than their panel. Three panels across 5.5in leaves
# about 1.6in each, and matplotlib does NOT widen a panel to fit its title -- it clips it.
SUBGROUP_LABEL = {
    "lab_value": "Lab value",
    "implicit": "Implicit description",
    "family": "Relative's finding",
}
#: The two recall slices, in the order the discussion introduces them. `family` is handled
#: separately below because it is a count on its own axis, not a third bar of the same kind.
RECALL_ORDER = ["lab_value", "implicit"]
COUNT_SUBGROUP = "family"

SUBGROUP_COHORT = "hcy"

#: Shared y-limit for the two recall panels, so a bar in one is the same height as an equal bar in
#: The other. Panels that share a quantity must share a scale or the eye compares nothing.
#: A little headroom above the tallest interval (0.62) and above the count panel's tallest bar,
#: so the count label on a bar is not clipped.
HEADROOM = 1.08
RECALL_YMAX = 0.70


def _rows(sub, group):
    return {r["method"]: r for _, r in sub[sub["subgroup"] == group].iterrows()
            if str(r.get("status", "ok")) == "ok"}


def _recall_panel(ax, sub, group, methods, overall):
    rows = _rows(sub, group)
    n_gold = n_reports = None
    for i, method in enumerate(methods):
        row = rows.get(method)
        if row is None:
            C.missing_marker(ax, i, 0.05)
            continue
        n_gold = int(row["n_gold"])
        n_reports = int(row["n_reports"])
        ax.bar(i, float(row["recall"]), width=0.66, color=C.METHOD_COLOR[method],
               edgecolor="white", linewidth=0.4, zorder=2)
        lo, hi = float(row["recall_lo"]), float(row["recall_hi"])
        ax.plot([i, i], [lo, hi], color=C.INK, lw=0.8, zorder=3)
        for edge in (lo, hi):
            ax.plot([i - 0.12, i + 0.12], [edge, edge], color=C.INK, lw=0.8, zorder=3)

        # The system's own aggregate recall on the full cohort: the reference the bar is read
        # against. Drawn as a caret, not a line so it cannot be mistaken for a CI cap.
        agg = overall.get(method)
        if agg is not None:
            ax.plot([i], [float(agg["micro_recall"])], marker="_", color=C.INK,
                    markersize=11, markeredgewidth=1.1, zorder=4)

    title = SUBGROUP_LABEL.get(group, group)
    if n_gold:
        title += f"\n{n_gold} annotated terms, {n_reports} reports"
    ax.set_title(title, pad=5, fontsize=8)
    ax.set_ylim(0, RECALL_YMAX)
    ax.set_yticks([t / 10 for t in range(0, 8)])
    C.style.light_grid(ax, axis="y")
    C.style.despine(ax)


def _count_panel(ax, sub, methods):
    """Family attribution: a count out of twelve, on its own axis, and lower is better."""
    rows = _rows(sub, COUNT_SUBGROUP)
    total = None
    for i, method in enumerate(methods):
        row = rows.get(method)
        if row is None or row.get("n_emitted") != row.get("n_emitted"):   # NaN
            C.missing_marker(ax, i, 0.5)
            continue
        total = int(row["n_attribution_pairs"])
        emitted = int(row["n_emitted"])
        # Hatched and outlined, not solid: nothing else in the figure looks like this, so
        # The panel cannot be skimmed as a third recall.
        ax.bar(i, emitted, width=0.66, facecolor="white",
               edgecolor=C.METHOD_COLOR[method], linewidth=1.0, hatch="///", zorder=2)
        # A count of zero draws nothing, and on this panel zero is the BEST result -- the system
        # never attributed a relative's finding to the patient. Unlabelled it is indistinguishable
        # from a missing bar, which is the one reading the figure must not permit.
        if emitted == 0:
            ax.annotate("0", xy=(i, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5,
                        color=C.INK, fontweight="bold", zorder=3)

    n_reports = next((int(r["n_reports"]) for r in rows.values()), 0)
    ax.set_title(f"{SUBGROUP_LABEL[COUNT_SUBGROUP]}\n"
                 f"{total or 0} excluded pairs, {n_reports} rep.",
                 pad=5, fontsize=8)
    # Integer ticks: a count of 12 with a tick at 2.5 invites the reading the note refuses.
    ax.set_yticks(range(0, (total or 12) + 1, 2))
    ax.set_ylim(0, (total or 12) * HEADROOM)
    ax.set_ylabel("terms emitted (lower is better)", fontsize=7.5)
    C.style.light_grid(ax, axis="y")
    C.style.despine(ax)


def build(results):
    """Draw recall on the laboratory-value and implicit subgroups and write the figure and its LaTeX."""
    sub = C.table(results, "t3_subgroups")
    overall = C.by_method(C.table(results, "t1_overall"), SUBGROUP_COHORT)

    present = set(sub["subgroup"])
    groups = [g for g in RECALL_ORDER if g in present]
    has_count = COUNT_SUBGROUP in present
    methods = [m for m in C.METHOD_ORDER if m in set(sub["method"])]

    n_panels = len(groups) + (1 if has_count else 0)
    if not n_panels:
        raise SystemExit(C.MISSING_SOURCE_EXIT)
    # The count panel does NOT share the y-axis: it is a different quantity. The two recall panels
    # share theirs with each other, which is what `sharey` would have given for free and has to be
    # done by hand now that a third panel is in the row.
    fig, axes = plt.subplots(1, n_panels, figsize=(C.WIDTH_FULL, 3.3),
                             gridspec_kw=dict(wspace=0.28))
    axes = [axes] if n_panels == 1 else list(axes)

    for ax, group in zip(axes, groups):
        _recall_panel(ax, sub, group, methods, overall)
    for ax in axes[1:len(groups)]:
        ax.set_yticklabels([])
    if has_count:
        _count_panel(axes[len(groups)], sub, methods)

    for ax in axes:
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([C.METHOD_LABEL[m] for m in methods], rotation=38,
                           ha="right", fontsize=7)
    if groups:
        axes[0].set_ylabel("recall within slice")

    handles = [plt.Line2D([], [], marker="_", color=C.INK, ls="none", markersize=11,
                          markeredgewidth=1.1, label="same system's aggregate recall"),
               plt.Line2D([], [], color=C.INK, lw=0.8, label="95% CI (report bootstrap)")]
    if has_count:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C.INK,
                                     hatch="///", lw=1.0,
                                     label="count, not a rate (right panel)"))
    # Below the panels, so no legend sits over a bar.
    C.legend_below(fig, handles, ncol=3)
    return fig, _caption(sub, overall)




def _caption(sub, overall):
    """05_PhenoJury.tex's caption and note for this figure, every number read from t3/t1.

    """
    rows = {g: _rows(sub, g) for g in RECALL_ORDER + [COUNT_SUBGROUP]}
    headline = (r"Recall on laboratory-value and implicitly described HCY terms, and relatives' "
                r"findings emitted, per system.")

    def size(group):
        r = next(iter(rows[group].values()))
        return int(r["n_gold"]), int(r["n_reports"])

    (lab_n, lab_r), (imp_n, imp_r) = size("lab_value"), size("implicit")
    fam = next(iter(rows[COUNT_SUBGROUP].values()))
    agg = next(iter(overall.values()))
    n_all = int(agg["tp"]) + int(agg["fn"])
    note = (r"Left and centre: recall on HCY terms that follow from a laboratory value against its "
            r"reference range (" + f"{lab_n} terms, {lab_r} reports) and on implicitly described "
            r"terms, where neither label nor synonym occurs (" + f"{imp_n} terms, {imp_r} reports)"
            r". Horizontal tick: the same system's recall on all "
            + f"{n_all:,}".replace(",", r"\,") + r" annotated terms. Right: number of the "
            + str(int(fam["n_attribution_pairs"])) + r" excluded relatives' findings ("
            + str(int(fam["n_reports"])) + r" reports) that each system emitted (lower is better).")
    return headline, note


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_common_args(ap)
    args = ap.parse_args()
    C.prepare(args)
    fig, (caption, note) = build(args.results)
    C.emit(fig, args, "fig_ch6_subgroups.pdf", "fig_ch6_subgroups.tex", caption=caption,
           label="fig:ch6-subgroups", generator=GENERATOR, note=note)


if __name__ == "__main__":
    main()
