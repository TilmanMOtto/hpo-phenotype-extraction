r"""Figure: the TreePhenoRAG pipeline schematic.

This closes the \todo{} at thesis/sections/04_TreePhenoRAG.tex and is the only figure in the
chapter that plots nothing -- it is a diagram, and its content is the method as
``src/hpo_extraction/treephenorag/traversal.py`` and ``src/hpo_extraction/treephenorag/selection.py`` actually implement it.

Drawn in matplotlib rather than TikZ,: it then takes over style.py's geometry and
Times-metric serif like every other figure in the thesis, renders at 5.5in so LaTeX never
rescales its text, and needs no preamble package the template does not already load.

ONE axes in INCH coordinates, with equal aspect. That is the whole trick to a readable matplotlib
diagram: box padding, corner radii, arrow heads and circle radii are all specified in points, so
in any other coordinate system they scale independently of the layout and the result is a pile of
overlapping blobs. Here 1.0 of x is 1.0 of y is one inch of paper, and every size below is a
physical measurement.

Three panels, matching the three things a reader has to hold at once:

  A  what one report's traversal produces on an ontology excerpt: scored, expanded, pruned,
     never-scored and accepted terms, including one term reached through two parents;
  B  what happens at one term -- the S retrieval calls, the S verifier margins, and the TWO
     poolings, which is the part every other method here collapses into one decision;
  C  the two retrieval indices, built once per HPO release and therefore not per-report cost.

Panel A carries the load. The bottleneck rule -- a term is visited iff some parent path to it
stayed above tau_prune -- is what makes the whole tau axis collapse to one value per node, and it
is invisible in prose. The middle term of the third row is there to show it: its left parent was
expanded and its right parent was never scored, and it is visited anyway.

The 18 354 in panel C is a property of the fixed ontology. It is typed here because a schematic's
labels are not a measurement. Anything a reader would quote lives in the tables.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import common as C
import palette as P

W, H = C.WIDTH_FULL, 3.25           # inches. The axes IS the page area
SMALL = C.style.TICK_FONT_SIZE      # 8 pt -- box text
TINY = 6.8                          # captions inside the diagram, legend, footnotes

# The four states a term can be in after one report's traversal. Colour is never the only cue:
# each state also differs in outline weight and fill, and "accepted" additionally carries a
# dashed ring, so the panel survives greyscale printing.
# Colours are palette.py's pipeline roles, the same in every chapter-4 figure: expansion purple,
# acceptance green, what is cut vermillion, what was never touched neutral.
STATE = {
    "expanded": dict(P.box(P.EXPANSION), lw=1.0),
    "pruned":   dict(P.box(P.NEGATIVE), lw=1.0),
    "unscored": dict(P.MUTED_BOX, lw=0.6),
    "accepted": dict(P.box(P.ACCEPTANCE), lw=1.5),
}
# Fill encodes the expansion decision, the ring encodes acceptance. They are independent, so the
# legend lists three fills and one ring, not four mutually exclusive states.
STATE_ORDER = [("expanded", "scored, expanded"), ("pruned", "scored, not expanded"),
               ("unscored", "never scored"), ("accepted", "accepted")]

# The two retrieval indices, matching common.INDEX_STYLE so the schematic and every plotted figure
# colour the same index the same way. The two scoring roles take the two decisions' colours:
# expansion purple, acceptance green, as the traversal panel draws them.
EX = P.box(P.AMBER)
ONT = P.box(P.EVIDENCE)
PLAIN = P.PLAIN_BOX
GREY = P.NEUTRAL_BOX
ACC = P.box(P.ACCEPTANCE)
PR = P.box(P.EXPANSION)


def box(ax, x, y, w, h, text, style=PLAIN, lw=0.8, fs=SMALL, ls="solid"):
    """A rounded box centred at (x, y), sized in inches. A dashed outline marks a side output."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, facecolor=style["fc"], edgecolor=style["ec"], lw=lw,
        linestyle=ls, boxstyle="round,pad=0,rounding_size=0.045", mutation_aspect=1.0, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=style["tc"], zorder=3,
            linespacing=1.30)


def arrow(ax, xy_from, xy_to, color=P.DARK, lw=0.8, ls="solid", rad=0.0):
    """Draw an arrow."""
    ax.add_patch(mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=6.5, color=color, lw=lw,
        linestyle=ls, shrinkA=0.5, shrinkB=0.5, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


def note(ax, x, y, text):
    """Write a small centred note."""
    ax.text(x, y, text, ha="center", va="center", fontsize=TINY, color=P.MID, style="italic")


def title(ax, x, text):
    """Write a panel title."""
    ax.text(x, H - 0.09, text, ha="left", va="center", fontsize=C.style.BASE_FONT_SIZE,
            color=P.INK, weight="bold")


# -- C: what is built once per HPO release ------------------------------------

def offline(ax):
    """Draw panel C, the retrieval indices built once per ontology release."""
    title(ax, 0.06, "C  Retrieval indices")
    cx = 0.95
    box(ax, cx, 2.76, 1.66, 0.34, "HPO release\n18 354 terms", GREY)

    box(ax, 0.50, 2.12, 0.76, 0.40, "synthetic\nsentences\nfrom a 70B LM", EX)
    box(ax, 1.42, 2.12, 0.76, 0.40, "label and\ndefinition\n(no LM)", ONT)
    arrow(ax, (0.72, 2.59), (0.50, 2.33))
    arrow(ax, (1.18, 2.59), (1.42, 2.33))

    box(ax, cx, 1.50, 1.44, 0.26, "sentence encoder", PLAIN)
    arrow(ax, (0.50, 1.92), (0.76, 1.64))
    arrow(ax, (1.42, 1.92), (1.14, 1.64))

    box(ax, 0.50, 0.87, 0.76, 0.44, "synthetic-\nsentence\nindex", EX, lw=1.1)
    box(ax, 1.42, 0.87, 0.76, 0.44, "term-\ninformation\nindex", ONT, lw=1.1)
    arrow(ax, (0.76, 1.37), (0.50, 1.10))
    arrow(ax, (1.14, 1.37), (1.42, 1.10))

    note(ax, cx, 0.40, "built once per HPO release,")
    note(ax, cx, 0.26, "one index is selected")


# -- B: one term's decision ---------------------------------------------------

def per_term(ax):
    """Draw panel B, retrieval, verification and pooling at one term."""
    title(ax, 1.98, "B  At one term $v$")
    cx = 2.80
    box(ax, cx, 2.90, 1.66, 0.24, "report $x$, segmented", GREY)
    # The query set is v's term information AND its descendants' (descendant closure) -- the
    # placeholder in 04_TreePhenoRAG.tex asks for it to be visible, not left to the caption.
    box(ax, cx, 2.52, 1.66, 0.24, "top-$S$ retrieval for $v$ + descendants", PLAIN, fs=TINY)
    arrow(ax, (cx, 2.78), (cx, 2.64))

    # The S verifier calls are the only per-report model cost in the whole method, so each gets its
    # own row: segment -> verifier -> score bar. The retrieved segments the two poolings read are then drawn as
    # literally that, S bars against the 1/2 line (greedy decoding answers Yes iff past it). The
    # scores are illustrative. The segments are grey strokes, so no report wording is quoted.
    xs, xv, xb0, bw = 2.28, 2.66, 2.78, 0.78
    ax.text(xs, 2.30, "segment", ha="center", va="center", fontsize=5.8, color=P.MID,
            style="italic")
    ax.text(xb0 + bw / 2, 2.30,
            "$\\sigma_j=\\mathrm{sig}(\\log p(\\mathrm{Yes})-\\log p(\\mathrm{No}))$",
            ha="center", va="center", fontsize=5.8, color=P.MID)
    for y, lab, s in ((2.16, "$s_1$", 0.93), (1.99, "$s_2$", 0.21), (1.76, "$s_S$", 0.64)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs - 0.30, y - 0.06), 0.60, 0.12, facecolor=P.WHITE, edgecolor=P.MID, lw=0.5,
            boxstyle="round,pad=0,rounding_size=0.02", zorder=2))
        ax.text(xs - 0.26, y, lab, ha="left", va="center", fontsize=5.8, color=P.INK, zorder=3)
        ax.plot([xs - 0.11, xs + 0.25], [y, y], color=P.LIGHT, lw=1.1,
                solid_capstyle="round", zorder=3)
        ax.add_patch(mpatches.Circle((xv, y), 0.045, facecolor=P.WHITE, edgecolor=P.DARK,
                                     lw=0.6, zorder=3))
        ax.text(xv, y, "V", ha="center", va="center", fontsize=4.8, color=P.INK, zorder=4)
        ax.plot([xs + 0.30, xv - 0.045], [y, y], color=P.DARK, lw=0.5, zorder=1)
        ax.plot([xv + 0.045, xb0], [y, y], color=P.DARK, lw=0.5, zorder=1)
        ax.add_patch(mpatches.Rectangle((xb0, y - 0.04), bw, 0.08, facecolor=P.FILL,
                                        edgecolor="none", zorder=2))
        ax.add_patch(mpatches.Rectangle((xb0, y - 0.04), bw * s, 0.08,
                                        facecolor=P.EVIDENCE if s > 0.5 else P.LIGHT,
                                        edgecolor="none", alpha=0.85, zorder=3))
        ax.text(xb0 + bw + 0.03, y, f"{s:.2f}", ha="left", va="center", fontsize=5.6,
                color=P.INK)
    ax.text(xs, 1.885, "$\\vdots$", ha="center", va="center", fontsize=TINY, color=P.DARK)
    ax.plot([xb0 + bw / 2] * 2, [1.68, 2.23], color=P.DARK, lw=0.6, ls=(0, (2, 1.5)), zorder=4)
    ax.text(xb0 + bw / 2, 1.62, "1/2", ha="center", va="center", fontsize=5.6, color=P.DARK)
    ax.text(xv, 1.62, "verifier", ha="center", va="center", fontsize=5.6, color=P.MID,
            style="italic")

    box(ax, 2.36, 1.22, 0.84, 0.30, "$\\Pi_{\\mathrm{pr}}$\nexpansion", PR)
    box(ax, 3.24, 1.22, 0.84, 0.30, "$\\Pi_{\\mathrm{acc}}$\nacceptance", ACC)
    arrow(ax, (2.60, 1.54), (2.36, 1.39))
    arrow(ax, (3.00, 1.54), (3.24, 1.39))

    box(ax, 2.36, 0.68, 0.84, 0.40,
        "expand children\niff $\\Pi_{\\mathrm{pr}}(v)$\n$\\geq \\tau_{\\mathrm{prune}}$",
        dict(PR, fc=P.WHITE), lw=1.0, fs=TINY)
    box(ax, 3.24, 0.68, 0.84, 0.40,
        "accept $v$ iff\n$\\Pi_{\\mathrm{acc}}(v)$\n$\\geq \\tau_{\\mathrm{accept}}$",
        dict(ACC, fc=P.WHITE), lw=1.0, fs=TINY)
    arrow(ax, (2.36, 1.06), (2.36, 0.89))
    arrow(ax, (3.24, 1.06), (3.24, 0.89))

    note(ax, 2.80, 0.34, "one set of margins,")
    note(ax, 2.80, 0.20, "two thresholds")


# -- A: the traversal on an ontology excerpt ----------------------------------

# (x, y, expansion state, accepted, label), in inches. y is depth: larger is shallower, so the DAG
# reads top-down.
#
# The fill is the EXPANSION outcome and the ring is the ACCEPTANCE outcome, because those are two
# independent decisions taken from two separately pooled scores -- which is the chapter's whole
# structural claim. Drawing acceptance as a third fill colour would quietly assert they are one
# decision with three outcomes. So term b is expanded AND accepted, and term d is accepted but not
# expanded: it answered, and the search stopped there.
NODES = {
    "r1": (4.22, 2.82, "expanded", False, ""),
    "r2": (5.16, 2.82, "pruned", False, ""),
    "a":  (4.00, 2.24, "expanded", False, ""),
    "b":  (4.54, 2.24, "expanded", True, ""),
    "c":  (5.16, 2.24, "unscored", False, ""),
    "d":  (4.00, 1.66, "pruned", True, ""),
    "e":  (4.60, 1.66, "expanded", True, ""),
    "f":  (5.20, 1.66, "unscored", False, ""),
    "g":  (4.60, 1.08, "pruned", False, ""),
}
EDGES = [("r1", "a"), ("r1", "b"), ("r2", "c"), ("a", "d"),
         ("b", "e"), ("c", "e"), ("e", "g"), ("c", "f")]
RADIUS = 0.105
RING = P.ACCEPTANCE


def traversal(ax):
    """Draw panel A, one report's traversal of the ontology."""
    title(ax, 3.92, "A  One report's traversal")
    # The two top terms are children of v0 (Phenotypic abnormality): the traversal seeds them
    # unconditionally, which is why they carry no incoming edge.
    note(ax, 4.69, 3.00, "start: children of $v_0$")

    for src, dst in EDGES:
        x0, y0, s0, _, _ = NODES[src]
        x1, y1, _, _, _ = NODES[dst]
        # An edge out of a term that was never scored, or scored and not expanded, carries no
        # traversal: draw it as ontology structure (faint, dotted), not as a path that was taken.
        live = s0 == "expanded"
        arrow(ax, (x0, y0 - RADIUS), (x1, y1 + RADIUS),
              color=P.DARK if live else P.LIGHT, lw=0.8 if live else 0.6,
              ls="solid" if live else (0, (1.3, 1.5)))

    for x, y, state, accepted, label in NODES.values():
        st = STATE[state]
        if accepted:
            ax.add_patch(mpatches.Circle((x, y), RADIUS + 0.042, facecolor="none",
                                         edgecolor=RING, lw=0.9, ls=(0, (1.5, 1.3)), zorder=3))
        ax.add_patch(mpatches.Circle((x, y), RADIUS, facecolor=st["fc"], edgecolor=st["ec"],
                                     lw=st["lw"], zorder=4))
        if label:
            ax.text(x, y, label, ha="center", va="center", fontsize=TINY, color=st["tc"],
                    zorder=5)

    # Hand-rolled legend: ax.legend would place its handles in axes fractions of a figure-wide
    # axes, which is the wrong coordinate system for a three-panel diagram sharing one canvas.
    for i, (state, text) in enumerate(STATE_ORDER):
        y = 0.70 - i * 0.155
        if state == "accepted":
            ax.add_patch(mpatches.Circle((4.02, y), 0.075, facecolor="none", edgecolor=RING,
                                         lw=0.9, ls=(0, (1.5, 1.3)), zorder=3))
            ax.add_patch(mpatches.Circle((4.02, y), 0.052, facecolor=P.FILL,
                                         edgecolor=P.LIGHT, lw=0.6, zorder=4))
        else:
            st = STATE[state]
            ax.add_patch(mpatches.Circle((4.02, y), 0.052, facecolor=st["fc"],
                                         edgecolor=st["ec"], lw=st["lw"], zorder=4))
        ax.text(4.14, y, text, ha="left", va="center", fontsize=TINY, color=P.INK)


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    C.style.apply()
    fig = plt.figure(figsize=(W, H))
    # Each panel is drawn in the inch coordinates above, inside its own axes whose width in
    # inches equals its x-span -- so it stays equal-aspect and a panel moves as a unit. Order on
    # The page: traversal (A), one term (B), offline (C).
    #   (draw, x-span in panel coordinates, left edge on the page)
    panels = [(traversal, (3.76, W), 0.0),
              (per_term, (1.88, 3.76), W - 3.76),
              (offline, (0.0, 1.88), W - 1.88)]
    for draw, (x0, x1), left in panels:
        ax = fig.add_axes([left / W, 0, (x1 - x0) / W, 1])
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_axis_off()
        draw(ax)

    # Thin rules between the panels: three stories on one canvas need a seam, and a 0.4pt line is
    # quieter than the whitespace it would otherwise take to separate them.
    for x in (W - 3.76, W - 1.88):
        fig.add_artist(plt.Line2D([x / W, x / W], [0.14 / H, (H - 0.20) / H],
                                  color=P.LIGHT, lw=0.5, zorder=0))

    C.save(fig, "fig_ch4_pipeline")
    C.write_figure_tex(
        "fig_ch4_pipeline", label="ch4-pipeline", script="fig_ch4_pipeline.py", placement="t",
        caption=(
            r"Schematic of TreePhenoRAG. \emph{A}: path of one report through part of the "
            r"ontology. \emph{B}: the two decisions at one term. \emph{C}: the two retrieval "
            r"indices."),
        note=(
            r"\emph{A}: a term is visited if at least one path to it runs only through terms whose "
            r"expansion score reached $\tau_{\mathrm{prune}}$. The middle term of the third row has "
            r"two parents, one expanded and one never scored, so it is visited through the first. "
            r"Dotted grey edges were not followed. \emph{B}: the verifier gives each retrieved "
            r"segment $s_1,\dots,s_S$ a score $\sigma_j$ in $[0,1]$. $\Pi_{\mathrm{pr}}$ pools these "
            r"into the expansion score, compared with $\tau_{\mathrm{prune}}$, and "
            r"$\Pi_{\mathrm{acc}}$ into the acceptance score, compared with "
            r"$\tau_{\mathrm{accept}}$, from the same verifier calls. \emph{C}: both indices cover "
            r"all 18\,354 terms and are built once per HPO release. The synthetic-sentence index "
            r"holds sentences a 70B model wrote for each term, the term-information index each "
            r"term's label, definition and synonyms. A run uses one of the two, chosen by "
            r"cross-validation."))


if __name__ == "__main__":
    main()
