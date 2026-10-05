#!/usr/bin/env python
r"""Figure: the evaluation protocol -- design, selection and estimation, selection stability.

    python experiments/figures/fig_ch3_protocol.py

A schematic of ``sec:protocol`` in ``03_experimental_setup.tex``. It reads no results. The fold
structure is computed by the repo's own ``hpo_extraction.evaluation.stats.folds.nested_folds``, the function that
wrote the fold files every tuned run reads, on placeholder report ids with rank tertiles as the
strata. The assignment depends only on the stratum sizes, the sort order of the ids and the seed,
so every fold and inner-fold size drawn here is the real one. Only which report sits in which cell
is unknown, and the figure draws no individual report. No HCY data is read.

Three sections, one per level of the protocol, one colour per role:

* **Design** (orange). One bar over all HCY reports: what the configurations are built from was
  fixed informally on the whole cohort, outside any split.
* **Selection & estimation** (nested CV). Repetition 0's outer 5-fold as a staircase: held-out
  fold blue, training split grey. Each row ends in a purple diamond, the configuration its inner
  CV selected. The inner 5-fold is drawn once, framed in purple (selection), because it runs in
  *every* training split. The parameters it selects are listed under it. The held-out blocks
  drop into the pooled bar (estimation). There is no development split: since PhenoJury's prompt
  and normaliser are selected in the folds too, no training split has a role outside them.
* **Selection stability** (purple). 5 folds x 10 repetitions of diamonds. Repetition 0's column is
  solid and framed in blue: its five configurations produce the reported performance. Repetitions 1-9 are only counted.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))  # style.py
import style  # noqa: E402
import palette as P  # noqa: E402

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hpo_extraction.evaluation.stats.folds import nested_folds, size_tertiles  # noqa: E402
from hpo_extraction.paths import thesis_dir  # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch3_protocol.py"
FIG_DIR = os.path.join(str(thesis_dir()), "thesis_figures")
TEX_DIR = os.path.join(str(thesis_dir()), "thesis_figures_latex")
NAME = "fig_ch3_protocol"

#: Cohort sizes of tab:ch3-datasets. GSC+ (206) only enters the caption.
N_HCY, N_GSC = 118, 206
K_OUTER, K_INNER, REPETITIONS = 5, 5, 10

#: What each level decides, as sec:protocol lists it.
DESIGNED = "verifier prompt, encoder, juror pool,\nfour prompts, value grids"
SELECTED = (r"TreePhenoRAG: index, operators, $\tau_{\mathrm{accept}}$, $\tau_{\mathrm{prune}}$"
            "\nPhenoJury: prompt, normaliser, jury, rule, scope, $k$")

W, H = style.WIDTH_FULL, 4.2
SMALL = style.TICK_FONT_SIZE - 1.6
# One colour per role (see the module docstring). Chapter 3 is setup, not a method, so these are
# chosen for this figure alone and are not the methods' hues in disguise (palette.py).
EVAL, VAL, DES = P.BLUE, P.PURPLE, P.AMBER
TRAIN, GREY, FRAME = P.tint(P.MID, 0.78), P.DARK, P.tint(P.MID, 0.2)
VAL_CELL, VAL_CELL_REP0 = P.tint(P.PURPLE, 0.84), P.tint(P.PURPLE, 0.68)


def structure(n: int) -> list[dict]:
    """``nested_folds`` on placeholder ids. Sizes are exact (see the module docstring)."""
    ids = [f"r{i:03d}" for i in range(n)]
    strata = size_tertiles({rid: i for i, rid in enumerate(ids)})
    return nested_folds(ids, strata, k_outer=K_OUTER, k_inner=K_INNER, repetitions=REPETITIONS)


def rect(ax, x, y, w, h, fc, ec="none", lw=0.0):
    """Draw a rectangle."""
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))


def diamond(ax, x, y, s=0.045, alpha=1.0):
    """Draw a diamond marker (a held-out fold)."""
    ax.add_patch(mpatches.RegularPolygon((x, y), 4, radius=s, fc=VAL, ec="none", alpha=alpha))


def section(ax, y, title, color):
    """Draw a section rule and title at height *y*."""
    ax.plot([0.06, W - 0.06], [y, y], color=P.LIGHT, lw=0.5)
    ax.text(0.06, y - 0.05, title, ha="left", va="top", fontsize=style.TICK_FONT_SIZE - 1.2,
            weight="bold", color=color)


def train_sizes(n: int) -> list[int]:
    """Sizes of the outer training splits for a cohort of *n* reports."""
    return sorted({len(r["train_ids"]) for r in structure(n) if r["repetition"] == 0})


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    # --results is accepted because make_all_ch3.py passes it to every script. A schematic has
    # no results folder, so it is ignored.
    ap.add_argument("--results", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--figdir", default=FIG_DIR)
    ap.add_argument("--texdir", default=TEX_DIR)
    args = ap.parse_args()

    rep0 = [r for r in structure(N_HCY) if r["repetition"] == 0]
    sizes = [len(r["eval_ids"]) for r in rep0]
    inner = [len(f) for f in rep0[0]["inner_folds"]]
    starts = np.cumsum([0] + sizes)

    style.apply()
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")         # hand-placed inch coordinates
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    x0, bw, rh, dy = 0.75, 2.1, 0.16, 0.235       # staircase origin, width, row height, pitch
    unit = bw / N_HCY
    xd = x0 + bw + 0.16                           # The column of selected configurations
    xi, wi = 3.6, 1.3                             # The inner-CV close-up
    last = (K_OUTER - 1) * dy

    # ---- design ----
    s0 = H - 0.04
    section(ax, s0, "DESIGN", DES)
    yb = s0 - 0.42
    rect(ax, x0, yb, bw, rh, DES)
    ax.text(x0 + bw / 2, yb + rh / 2, "all HCY reports, informally", ha="center", va="center",
            fontsize=SMALL, color="white", weight="bold")
    ax.text(x0 - 0.1, yb + rh / 2, "HCY", ha="right", va="center", fontsize=SMALL)
    ax.text(xi - 0.07, yb + rh / 2 + 0.01, DESIGNED, ha="left", va="center", fontsize=SMALL,
            linespacing=1.2)

    # ---- selection & estimation ----
    s1 = yb - 0.2
    section(ax, s1, "SELECTION & ESTIMATION  ·  NESTED CV", P.INK)
    top = s1 - 0.5
    ax.text(x0 + bw / 2, top + rh + 0.05, f"outer {K_OUTER}-fold, repetition 0", ha="center",
            va="bottom", fontsize=SMALL)
    for f in range(K_OUTER):
        y = top - f * dy
        rect(ax, x0, y, bw, rh, TRAIN, ec="white", lw=0.6)
        rect(ax, x0 + starts[f] * unit, y, sizes[f] * unit, rh, EVAL, ec="white", lw=0.6)
        ax.text(x0 - 0.1, y + rh / 2, f"fold {f}", ha="right", va="center", fontsize=SMALL)
        diamond(ax, xd, y + rh / 2)
    ax.text(xd, top - last - 0.04, "selected", ha="center", va="top", fontsize=SMALL - 0.6,
            color=VAL)

    # The inner CV runs in every training split. Fold 0's sizes are drawn.
    ui = wi / sum(inner)
    ist = np.cumsum([0] + inner)
    block_h = (K_INNER - 1) * dy + rh
    rect(ax, xi - 0.07, top - (K_INNER - 1) * dy - 0.07, wi + 0.14, block_h + 0.14, "white",
         ec=VAL, lw=0.9)
    for j, s in enumerate(inner):
        y = top - j * dy
        rect(ax, xi, y, wi, rh, TRAIN, ec="white", lw=0.6)    # same edge as the purple block
        rect(ax, xi + ist[j] * ui, y, s * ui, rh, VAL, ec="white", lw=0.6)
    ax.plot([x0 + bw, xi - 0.07], [top + rh, top + rh + 0.07], color=FRAME, lw=0.5, ls=":")
    ax.plot([x0 + bw, xi - 0.07], [top, top - (K_INNER - 1) * dy - 0.07], color=FRAME, lw=0.5,
            ls=":")
    ax.text(xi - 0.07, top + rh + 0.14, "SELECTION", ha="left", va="bottom", fontsize=SMALL,
            weight="bold", color=VAL)
    ax.text(xi + 0.62, top + rh + 0.14, f"inner {K_INNER}-fold, every training split",
            ha="left", va="bottom", fontsize=SMALL - 0.6, color=GREY)
    ax.text(xi - 0.07, top - (K_INNER - 1) * dy - 0.13, SELECTED, ha="left", va="top",
            fontsize=SMALL - 0.6, color=P.INK, linespacing=1.25)

    yp = top - last - 0.36
    for f in range(K_OUTER):
        rect(ax, x0 + starts[f] * unit, yp, sizes[f] * unit, rh, EVAL, ec="white", lw=0.6)
        xc = x0 + (starts[f] + sizes[f] / 2) * unit
        ax.plot([xc, xc], [top - f * dy - 0.01, yp + rh + 0.01], color=EVAL, lw=0.4, ls=":",
                alpha=0.7)
    ax.text(x0 - 0.1, yp + rh / 2, "pooled", ha="right", va="center", fontsize=SMALL,
            weight="bold", color=EVAL)
    ax.text(x0, yp - 0.07, "ESTIMATION", ha="left", va="top", fontsize=SMALL, weight="bold",
            color=EVAL)
    ax.text(x0 + 0.72, yp - 0.075,
            "held-out folds, pooled: point estimates, intervals, tests", ha="left",
            va="top", fontsize=SMALL - 0.6, color=GREY)

    # ---- selection stability ----
    s2 = yp - 0.42
    section(ax, s2, "SELECTION STABILITY", VAL)
    gtop = s2 - 0.42
    cw, ch = 0.2, 0.13
    for rep in range(REPETITIONS):
        for f in range(K_OUTER):
            x, y = x0 + rep * cw, gtop - f * ch
            rect(ax, x + 0.01, y - ch / 2 + 0.01, cw - 0.02, ch - 0.02,
                 VAL_CELL_REP0 if rep == 0 else VAL_CELL)
            diamond(ax, x + cw / 2, y, s=0.035, alpha=1.0 if rep == 0 else 0.5)
        ax.text(x0 + rep * cw + cw / 2, gtop + ch / 2 + 0.03, str(rep), ha="center",
                va="bottom", fontsize=SMALL - 0.6, color=GREY)
    for f in range(K_OUTER):
        ax.text(x0 - 0.1, gtop - f * ch, f"fold {f}", ha="right", va="center",
                fontsize=SMALL - 0.6, color=GREY)
    ax.text(x0 + REPETITIONS * cw + 0.08, gtop + ch / 2 + 0.03, "repetition", ha="left",
            va="bottom", fontsize=SMALL - 0.6, color=GREY)
    ax.add_patch(mpatches.Rectangle((x0, gtop - (K_OUTER - 0.5) * ch), cw, K_OUTER * ch,
                                    fill=False, ec=EVAL, lw=0.9))
    ax.text(x0 + cw / 2, gtop - (K_OUTER - 0.5) * ch - 0.04, "reported\nperformance",
            ha="center", va="top", fontsize=SMALL - 0.4, color=EVAL, linespacing=1.1)
    n_sel = K_OUTER * REPETITIONS
    ax.text(xi - 0.07, gtop - 2 * ch,
            f"{n_sel} selections: how often each\nparameter component is chosen",
            ha="left", va="center", fontsize=SMALL, color=P.INK, linespacing=1.2)

    os.makedirs(args.figdir, exist_ok=True)
    preview = os.environ.get("THESIS_FIG_PREVIEW")
    style.save(fig, os.path.join(args.figdir, f"{NAME}.pdf"),
               preview_png=os.path.join(preview, f"{NAME}.png") if preview else None)

    def either(v):
        return " or ".join(map(str, v))

    caption = (r"Evaluation protocol based on nested cross-validation. All reported numbers are "
               r"computed on the pooled held-out outer folds.")
    note = (
        r"\emph{Design} (orange): the components the configurations are built from were developed "
        r"informally on all HCY reports. \emph{Selection and estimation}: repetition~0 of the outer "
        + str(K_OUTER) + r"-fold split of the " + str(N_HCY) + r" HCY reports (fold sizes "
        + ", ".join(map(str, sizes)) + r"). In every training split (grey, "
        + either(train_sizes(N_HCY)) + r" reports), " + str(K_INNER) + r"-fold inner "
        r"cross-validation selects a configuration (diamond, with the inner split of fold~0 in "
        r"purple on the right), which is applied to the held-out fold (blue). \emph{Selection "
        r"stability}: the outer split is drawn " + str(REPETITIONS) + r" times with new fold "
        r"assignments. Only the configurations of repetition~0 (framed) are scored, and all "
        + str(n_sel) + r" selections measure how often each value is chosen. GSC+ (206) follows "
        r"the same design, with training splits of " + either(train_sizes(N_GSC)) + r" documents.")
    os.makedirs(args.texdir, exist_ok=True)
    style.write_tex(os.path.join(args.texdir, f"{NAME}.tex"), f"{NAME}.pdf", caption,
                    "fig:ch3-protocol", generator=GENERATOR, note=note)


if __name__ == "__main__":
    main()
