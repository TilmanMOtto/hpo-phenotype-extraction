#!/usr/bin/env python
r"""Figure: the HPO is a DAG -- one annotated HCY term reached through two organ systems.

    python experiments/figures/fig_ch3_hpo_fragment.py

Reads only the fixed ontology, ``resources/util/hpo.json``. Every node, edge and depth in the
figure is computed from its ``Is_a`` relations. Nothing about the graph is typed here.

## The term, and why this one

The placeholder in ``03_experimental_setup.tex`` asks for one multi-parent term of the HCY ground
truth, drawn from *Phenotypic abnormality* ($v_0$) down with all of its ancestors, three to four
levels deep. *Inflammatory abnormality of the skin* (HP:0011123) is the HCY annotated term that fits
best, chosen by this rule over the annotated terms available off the cluster: two parents that lie in
**different organ systems** (first-level branches), and the smallest ancestor set. Its two paths
are four and five steps long (integument. Immune system), so the one figure also shows that a
term's depth depends on the path it is reached by. No three-to-four-level term with parents in two
organ systems exists among them. The next candidates (*Otitis media*, *Pneumonia*) are deeper.

Its ground truth membership is checked, not assumed, against the TreePhenoRAG protocol's ``recall_attribution.csv`` -- an
aggregate of (report, term) codes with no report text, so reading it here crosses no patient-data
line. That file lists only the annotated terms TreePhenoRAG missed, which is enough to *confirm* a term
is annotated but not to show that one is not, so only the chosen term is shaded: shading any
other node as "not annotated" would be a claim this machine cannot check.

## Drawing

One axes in inch coordinates with equal aspect, as in ``fig_ch4_pipeline``: box sizes, arrow
heads and text are then physical measurements and cannot drift apart when the layout changes.
Rows are depths (longest path from $v_0$), columns are the two organ-system branches.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))  # style.py
import style  # noqa: E402
import palette as P  # noqa: E402

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from hpo_extraction.paths import results_dir, thesis_dir  # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/fig_ch3_hpo_fragment.py"
HPO_JSON = os.path.join(_REPO, "resources", "util", "hpo.json")
GOLD_EVIDENCE = os.path.join(str(results_dir()), "treephenorag_protocol", "tables",
                             "recall_attribution.csv")
FIG_DIR = os.path.join(str(thesis_dir()), "thesis_figures")
TEX_DIR = os.path.join(str(thesis_dir()), "thesis_figures_latex")
NAME = "fig_ch3_hpo_fragment"

V0 = "HP:0000118"          # Phenotypic abnormality
ROOT = "HP:0000001"        # All -- above v0, never drawn
TERM = "HP:0011123"        # Inflammatory abnormality of the skin (see the module docstring)

W, ROW = style.WIDTH_FULL, 0.44          # inches; ROW is the vertical pitch of one depth
BOX_W, BOX_H = 1.78, 0.30
FS = style.TICK_FONT_SIZE                # 8 pt node labels
TINY = 6.8

# Annotated is green: the positive outcome's colour in every pipeline figure (palette.py, rule 2).
ANNOTATED = dict(fc=P.GREEN, ec=P.shade(P.GREEN, 0.47), tc=P.WHITE, lw=1.0)
ANCESTOR = dict(fc=P.WHITE, ec=P.DARK, tc=P.INK, lw=0.8)
EDGE = P.DARK


def load():
    """The ontology file as a dict."""
    with open(HPO_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def parents(hpo, t):
    """Direct is-a parents, in the file's own order -- the second one is drawn dashed."""
    return [p for p in hpo[t].get("Is_a", []) if p in hpo and p != ROOT]


def depth(hpo, t, memo=None):
    """Longest path from v0 -- the row a term is drawn in."""
    memo = {} if memo is None else memo
    if t == V0:
        return 0
    if t not in memo:
        memo[t] = 1 + max(depth(hpo, p, memo) for p in parents(hpo, t))
    return memo[t]


def chain(hpo, start):
    """The path from ``start`` up to v0. Raises if it forks: the layout draws each branch as one
    column, so a second parent above the term would silently be left out rather than drawn."""
    path = [start]
    while path[-1] != V0:
        ps = parents(hpo, path[-1])
        if len(ps) != 1:
            raise SystemExit(f"{path[-1]} has {len(ps)} parents; the two-column layout needs a "
                             f"single chain above each parent of {TERM}")
        path.append(ps[0])
    return path


def check_gold() -> bool | None:
    """Is TERM in the HCY ground truth? True/False when the evidence file exists, None when it does not."""
    if not os.path.isfile(GOLD_EVIDENCE):
        return None
    with open(GOLD_EVIDENCE, encoding="utf-8") as fh:
        return any(r["hpo_id"] == TERM for r in csv.DictReader(fh))


def label(hpo, t):
    """Display label of term *t*."""
    name = hpo[t]["Name"][0]
    if t == V0:
        name += " ($v_0$)"
    return "\n".join(textwrap.wrap(name, 26))


def box(ax, x, y, text, st):
    """Draw one term box with style *st* centred at (*x*, *y*)."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, facecolor=st["fc"], edgecolor=st["ec"],
        lw=st["lw"], boxstyle="round,pad=0,rounding_size=0.045", zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, color=st["tc"], zorder=3,
            linespacing=1.15)


def edge(ax, a, b, dashed=False):
    """Draw an is-a arrow from box *a* to box *b*."""
    ax.add_patch(mpatches.FancyArrowPatch(
        (a[0], a[1] - BOX_H / 2), (b[0], b[1] + BOX_H / 2), arrowstyle="-|>", mutation_scale=6.5,
        color=EDGE, lw=0.8, linestyle=(0, (3.0, 2.0)) if dashed else "solid",
        shrinkA=0.5, shrinkB=0.5, zorder=1))


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    # --results is accepted because make_all_ch3.py passes it to every script. The ontology file,
    # not a results folder, is this figure's source, so it is ignored.
    ap.add_argument("--results", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--figdir", default=FIG_DIR)
    ap.add_argument("--texdir", default=TEX_DIR)
    args = ap.parse_args()

    hpo = load()
    ps = parents(hpo, TERM)
    if len(ps) < 2:
        raise SystemExit(f"{TERM} has {len(ps)} parent(s); the figure is about a multi-parent term")
    branches = [chain(hpo, p) for p in ps]          # each: parent ... v0
    systems = [b[-2] for b in branches]             # The first-level term of each branch
    if len(set(systems)) < 2:
        raise SystemExit(f"{TERM}'s parents share one organ system; the figure needs two")
    gold = check_gold()
    if gold is False:
        raise SystemExit(f"{TERM} is not in the HCY gold evidence {GOLD_EVIDENCE}")
    if gold is None:
        print(f"  note: {GOLD_EVIDENCE} absent; HCY gold membership of {TERM} not re-checked")

    memo: dict = {}
    d_term = depth(hpo, TERM, memo)
    steps = [len(b) for b in branches]              # path length v0 -> term along each parent
    n_rows = d_term + 1
    H = n_rows * ROW + 0.30

    style.apply()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()

    def y_of(t):
        return H - 0.28 - depth(hpo, t, memo) * ROW

    col_x = [1.55, 3.95]                            # one column per branch
    mid = (col_x[0] + col_x[1]) / 2
    pos = {V0: (mid, y_of(V0)), TERM: (mid, y_of(TERM))}
    for x, b in zip(col_x, branches):
        for t in b[:-1]:                            # v0 is shared, drawn once in the middle
            pos[t] = (x, y_of(t))

    # Edges parent -> child. Only the term's second parent is dashed.
    for b in branches:
        for child, parent in zip(b[:-1], b[1:]):
            edge(ax, pos[parent], pos[child])
    for i, p in enumerate(ps):
        edge(ax, pos[p], pos[TERM], dashed=i > 0)

    for t, (x, y) in pos.items():
        box(ax, x, y, label(hpo, t), ANNOTATED if t == TERM else ANCESTOR)

    # Depth ruler: the rows ARE depths, and the term's two path lengths are the point.
    for dd in range(n_rows):
        ax.text(0.10, H - 0.28 - dd * ROW, f"depth {dd}", ha="left", va="center",
                fontsize=TINY, color=P.MID)
    tx, ty = pos[TERM]
    ax.text(tx + BOX_W / 2 + 0.08, ty,
            "\n".join(f"{s} steps via {hpo[sys_]['Name'][0].replace('Abnormality of the ', '')}"
                      for s, sys_ in zip(steps, systems)),
            ha="left", va="center", fontsize=TINY, color=P.DARK, style="italic")

    # Hand-rolled legend, top right beside v0 (the one empty corner), in the same inch coordinates.
    lx, ly = 4.30, H - 0.12
    ax.add_patch(mpatches.Rectangle((lx, ly - 0.05), 0.16, 0.10, facecolor=ANNOTATED["fc"],
                                    edgecolor=ANNOTATED["ec"], lw=0.8))
    ax.text(lx + 0.22, ly, "annotated term", ha="left", va="center", fontsize=TINY)
    ax.add_patch(mpatches.Rectangle((lx, ly - 0.23), 0.16, 0.10, facecolor="white",
                                    edgecolor=ANCESTOR["ec"], lw=0.8))
    ax.text(lx + 0.22, ly - 0.18, "ancestor", ha="left", va="center", fontsize=TINY)
    ax.plot([lx, lx + 0.16], [ly - 0.36, ly - 0.36], color=EDGE, lw=0.8, ls=(0, (3.0, 2.0)))
    ax.text(lx + 0.22, ly - 0.36, "second parent", ha="left", va="center", fontsize=TINY)

    os.makedirs(args.figdir, exist_ok=True)
    preview = os.environ.get("THESIS_FIG_PREVIEW")
    style.save(fig, os.path.join(args.figdir, f"{NAME}.pdf"),
               preview_png=os.path.join(preview, f"{NAME}.png") if preview else None)

    names = [hpo[s]["Name"][0][0].lower() + hpo[s]["Name"][0][1:] for s in systems]
    caption = (r"Fragment of the HPO below \emph{Phenotypic abnormality}. A term with two parents "
               r"is reachable along two paths through different organ systems.")
    note = (
        r"Fragment of the ontology below \emph{Phenotypic abnormality} ($v_0$, " + V0 + r"). Edges "
        r"point from a term to its more specific children. Shaded: a term annotated in an HCY "
        r"report. White: its ancestors. \emph{" + hpo[TERM]["Name"][0] + r"} (" + TERM
        + r") is reached in " + str(steps[0]) + r" steps through "
        + names[0].replace("abnormality of the ", "the ") + r" and in " + str(steps[1])
        + r" through " + names[1].replace("abnormality of the ", "the ")
        + r". Dashed: its second parent.")
    os.makedirs(args.texdir, exist_ok=True)
    style.write_tex(os.path.join(args.texdir, f"{NAME}.tex"), f"{NAME}.pdf", caption,
                    "fig:ch3-hpo-fragment", generator=GENERATOR, note=note)


if __name__ == "__main__":
    main()
