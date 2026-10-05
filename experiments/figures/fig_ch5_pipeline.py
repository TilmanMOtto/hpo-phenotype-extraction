#!/usr/bin/env python
r"""Figure: the PhenoJury pipeline schematic.

Redrawn in matplotlib, in the same idiom as ``fig_ch4_pipeline``, so the two method chapters open
on diagrams that read as one set: the same Times-metric serif, the same Okabe-Ito fills, the same
rounded boxes, and panel titles in the same place. It replaces the copied PNG of the author's slide
diagram (thesis/graphs/HPO Jury Architecture-selection.png), whose content it keeps -- report,
eight jurors, normalisation, per-juror HPO candidates, vote against a threshold -- but whose type
shrank to a few points once a 3566 px slide was set at the 5.5 in text width.

ONE axes in INCH coordinates with equal aspect, as in ``fig_ch4_pipeline``: box padding, arrow heads
and marker sizes are all in points, so in any other coordinate system they scale independently of
the layout. Here 1.0 of x is 1.0 of y is one inch of paper. The figure is drawn at 5.5 in,
so LaTeX never rescales it, and no text is smaller than 7.5 pt.

Four panels follow one sentence through the pipeline:

  A  the sentence, read alone -- a juror never sees its neighbours;
  B  the eight jurors, each writing the findings it names in its own words;
  C  the parser and normaliser, which turn each juror's strings into HPO identifiers;
  D  the vote: for each identifier, how many jurors support it, against the threshold k;

The sentence is the chapter's own worked example (05_PhenoJury.tex, "Pipeline"), and the chapter
says every juror named the developmental delay. The juror OUTPUTS are illustrative, as is k = 3: a
schematic's labels are not a measurement, and anything a reader would quote lives in the tables.
They are chosen so the one panel carries both rules. Two jurors write *Mild global developmental
delay* (HP:0011342), a child of *Global developmental delay* (HP:0001263) in the fixed ontology. Under the exact rule they do not support HP:0001263, under the closure rule they do. That is the
only difference between the rules, and prose leaves it abstract. One juror adds *Intellectual
disability* (HP:0001249), which the vote removes.

    python experiments/figures/fig_ch5_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C                       # noqa: E402
import palette as P                          # noqa: E402

FIGNAME = "fig_ch5_pipeline.pdf"
TEXNAME = "fig_ch5_pipeline.tex"
LABEL = "fig:ch5-pipeline"
GENERATOR = "figures/thesis_figures_scripts/fig_ch5_pipeline.py"

W, H = C.style.WIDTH_FULL, 3.28     # inches. The axes IS the page area
TITLE = C.style.BASE_FONT_SIZE      # 9 pt -- panel titles
BODY = C.style.TICK_FONT_SIZE       # 8 pt -- box text
SMALL = 7.5                         # The floor: nothing in this figure is set smaller

INK = P.INK
MUTED = P.tint(P.DARK, 0.13)
RULE = P.LIGHT

# Fills shared with fig_ch4_pipeline, so a colour means the same thing in both schematics: green is
# The positive outcome, and in this figure it marks one thing -- the predicted term.
GREY = P.NEUTRAL_BOX
PLAIN = dict(fc=P.WHITE, ec=P.LIGHT, tc=INK)
ACC = P.box(P.POSITIVE)
# The report sentence is amber, and so is every juror's spine: each juror reads that sentence.
SENT = dict(fc=P.tint(P.AMBER, 0.80), ec=P.AMBER, tc=P.shade(P.AMBER, 0.60))
JUROR_ACCENT = P.AMBER

# The three HPO identifiers the example produces. Blue is the answer and a lighter blue its child,
# so the family relation the closure rule uses is visible before it is read. Vermillion is the
# stray term the vote removes. Every chip and every vote row also prints the identifier, so colour
# is never the only cue.
TERMS = {
    "HP:0001263": dict(name="Global developmental delay", **P.box(P.BLUE), mark=P.BLUE),
    "HP:0011342": dict(name="Mild global developmental delay", fc=P.tint(P.SKY, 0.84), ec=P.SKY,
                       mark=P.SKY, tc=P.shade(P.SKY, 0.55)),
    "HP:0001249": dict(name="Intellectual disability", **P.box(P.VERMILLION),
                       mark=P.VERMILLION),
}
PARENT = {"HP:0011342": "HP:0001263"}      # The one is_a edge the example needs
K = 3

# (juror, [(string it wrote, identifier the normaliser maps it to)]). Illustrative -- see docstring.
JURY = [
    ("Apertus", [("Global developmental delay", "HP:0001263")]),
    ("DeepSeek", [("Developmental delay", "HP:0001263")]),
    ("Intelligent-Internet", [("Mild global developmental delay", "HP:0011342")]),
    ("Llama", [("Delayed development", "HP:0001263")]),
    ("MedGemma", [("Global developmental delay", "HP:0001263")]),
    ("MedPsy", [("Mild global developmental delay", "HP:0011342")]),
    ("OpenBioLLM", [("Developmental delay", "HP:0001263"),
                    ("Intellectual disability", "HP:0001249")]),
    ("Phi-4", [("Global developmental delay", "HP:0001263")]),
]

# Column edges, in inches.
A_X0, A_X1 = 0.05, 0.90
B_X0, B_X1 = 1.06, 2.78
N_X0, N_X1 = 2.88, 3.44          # The normaliser band
K_X0, K_X1 = 3.53, 4.21          # The per-juror candidate chips
D_X0, D_X1 = 4.35, 5.46

TOP = H - 0.30                   # first row's top edge
ROW_H, LINE_H, ROW_GAP = 0.30, 0.125, 0.052


def box(ax, x0, y0, x1, y1, style, lw=0.8, r=0.045, z=2, ls="solid"):
    """Draw a rounded box."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, facecolor=style["fc"], edgecolor=style["ec"], lw=lw,
        linestyle=ls, boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=1.0, zorder=z))


def text(ax, x, y, s, fs=BODY, color=INK, ha="center", va="center", z=5, **kw):
    """Write text."""
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, linespacing=1.25, **kw)


def arrow(ax, xy_from, xy_to, color=P.DARK, lw=0.7, head=True):
    """Draw a line, with an arrow head when *head* is true."""
    ax.add_patch(mpatches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>" if head else "-", mutation_scale=6.0, color=color,
        lw=lw, shrinkA=0, shrinkB=0.4, zorder=1))


def title(ax, x, y, s):
    """Write a panel title on its baseline."""
    # Baseline, not centre: a centred title with a descender sits higher than one without.
    text(ax, x, y, s, fs=TITLE, color=P.INK, ha="left", va="baseline", weight="bold")


def row_geometry():
    """Top and bottom edge of each juror's row. A juror that wrote two strings gets a taller row."""
    rows, y = [], TOP
    for _, outputs in JURY:
        h = ROW_H + LINE_H * (len(outputs) - 1)
        rows.append((y - h, y))
        y -= h + ROW_GAP
    return rows


# -- A: one sentence, read alone ----------------------------------------------

def report(ax, rows):
    """Draw panel A, the report's segments."""
    title(ax, A_X0, H - 0.16, "A  Report")
    y_bot = rows[-1][0]
    box(ax, A_X0, y_bot, A_X1, TOP, PLAIN, lw=0.6)

    # The rest of the report is drawn as placeholder lines: it exists, and no juror reads it.
    def filler(y, frac):
        ax.plot([A_X0 + 0.09, A_X0 + 0.09 + frac * (A_X1 - A_X0 - 0.18)], [y, y],
                color=P.tint(P.LIGHT, 0.47), lw=2.2, solid_capstyle="round", zorder=3)
    for i, frac in enumerate((0.95, 0.70, 0.88, 0.55)):
        filler(TOP - 0.16 - i * 0.13, frac)

    sy0, sy1 = TOP - 1.52, TOP - 0.68
    box(ax, A_X0 + 0.06, sy0, A_X1 - 0.06, sy1, SENT, lw=0.9, r=0.03, z=3)
    text(ax, (A_X0 + A_X1) / 2, (sy0 + sy1) / 2,
         "At the age of\n11 months his\ndevelopment\nis mildly\nretarded", fs=SMALL,
         color=SENT["tc"], style="italic")

    for i, frac in enumerate((0.92, 0.62, 0.84, 0.74, 0.50, 0.90)):
        filler(sy0 - 0.20 - i * 0.13, frac)

    # One bus from the sentence to every juror: all eight receive the same input, in parallel.
    bus_x = (A_X1 + B_X0) / 2
    ymid = (sy0 + sy1) / 2
    ax.plot([A_X1 - 0.06, bus_x], [ymid, ymid], color=P.DARK, lw=0.7, zorder=1)
    centres = [(lo + hi) / 2 for lo, hi in rows]
    ax.plot([bus_x, bus_x], [min(centres), max(centres)], color=P.DARK, lw=0.7, zorder=1)
    for c in centres:
        arrow(ax, (bus_x, c), (B_X0, c))


# -- B: the jurors ------------------------------------------------------------

def jurors(ax, rows):
    """Draw panel B, the eight jurors' outputs."""
    title(ax, B_X0, H - 0.16, "B  Eight jurors, one prompt")
    for (name, outputs), (lo, hi) in zip(JURY, rows):
        box(ax, B_X0, lo, B_X1, hi, PLAIN, lw=0.6, r=0.03)
        # A coloured spine marks the model, as in the slide version. It carries no other meaning.
        ax.add_patch(mpatches.Rectangle((B_X0, lo + 0.012), 0.028, hi - lo - 0.024,
                                        facecolor=JUROR_ACCENT, edgecolor="none", zorder=3))
        text(ax, B_X0 + 0.09, hi - 0.085, name, fs=SMALL, ha="left", weight="bold")
        for j, (s, _) in enumerate(outputs):
            text(ax, B_X0 + 0.09, hi - 0.085 - LINE_H * (j + 1), f"“{s}”", fs=SMALL,
                 ha="left", color=P.DARK, style="italic")


# -- C: parse and normalise ---------------------------------------------------

def normalise(ax, rows):
    """Draw panel C, normalisation to HPO terms."""
    title(ax, N_X0, H - 0.16, "C  Normalise")
    y_bot, y_top = rows[-1][0], TOP
    box(ax, N_X0, y_bot, N_X1, y_top, GREY, lw=0.8)
    cx = (N_X0 + N_X1) / 2
    text(ax, cx, (y_bot + y_top) / 2 + 0.34, "parser", fs=BODY, color=GREY["tc"], weight="bold")
    text(ax, cx, (y_bot + y_top) / 2 + 0.18, "$\\downarrow$", fs=BODY, color=GREY["tc"])
    text(ax, cx, (y_bot + y_top) / 2 + 0.02, "normaliser", fs=BODY, color=GREY["tc"],
         weight="bold")
    text(ax, cx, (y_bot + y_top) / 2 - 0.30, "PhenoBERT\nDictionary\nSapBERT", fs=SMALL,
         color=MUTED)

    chip_h = 0.105
    for (_, outputs), (lo, hi) in zip(JURY, rows):
        c = (lo + hi) / 2
        arrow(ax, (B_X1, c), (N_X0, c))
        n = len(outputs)
        for j, (_, hpo) in enumerate(outputs):
            # Chips stack when a juror produced more than one identifier.
            yc = c + (n - 1) * (LINE_H / 2) - j * LINE_H
            arrow(ax, (N_X1, yc), (K_X0, yc))
            t = TERMS[hpo]
            box(ax, K_X0, yc - chip_h / 2 - 0.004, K_X1, yc + chip_h / 2 + 0.004,
                dict(fc=t["fc"], ec=t["ec"]), lw=0.7, r=0.025, z=3)
            text(ax, (K_X0 + K_X1) / 2, yc, hpo, fs=SMALL, color=t["tc"])

    # Every juror's identifiers are pooled into one vote.
    bus_x = (K_X1 + D_X0) / 2
    ys = [(lo + hi) / 2 for lo, hi in rows]
    for (_, outputs), c in zip(JURY, ys):
        n = len(outputs)
        for j in range(n):
            yc = c + (n - 1) * (LINE_H / 2) - j * LINE_H
            ax.plot([K_X1, bus_x], [yc, yc], color=P.DARK, lw=0.7, zorder=1)
    ax.plot([bus_x, bus_x], [min(ys) - LINE_H / 2, max(ys)], color=P.DARK, lw=0.7, zorder=1)
    return bus_x


# -- D: the vote --------------------------------------------------------------

def supports(hpo: str, closure: bool) -> list[str]:
    """Per juror: 'named' if it produced ``hpo``, 'desc' if (closure only) a descendant, else ''."""
    out = []
    for _, outputs in JURY:
        ids = {h for _, h in outputs}
        if hpo in ids:
            out.append("named")
        elif closure and any(PARENT.get(h) == hpo for h in ids):
            out.append("desc")
        else:
            out.append("")
    # Solid first, then the closure-only support, so the count reads left to right.
    return sorted(out, key=lambda s: {"named": 0, "desc": 1, "": 2}[s])


def vote(ax, bus_x):
    """Draw panel D, the vote."""
    title(ax, D_X0, H - 0.16, "D  Vote")
    sq, gap, k_gap = 0.100, 0.022, 0.050
    x0 = D_X0 + 0.02

    def cell_x(i):
        # The squares open a wider gap after the k-th, so the threshold falls visibly between two.
        return x0 + i * (sq + gap) + (k_gap - gap if i >= K else 0.0)
    k_x = cell_x(K) - k_gap / 2
    k_colour = P.VERMILLION

    # One row per identifier: its label, then one square per juror. The threshold is drawn beside
    # The squares only, so it never crosses a label.
    y = TOP - 0.06
    square_rows = []
    for hpo, t in TERMS.items():
        cells = supports(hpo, closure=True)
        n_exact = sum(c == "named" for c in cells)
        n_closure = sum(c != "" for c in cells)
        # "exact · closure" when the rules disagree -- the same order the prediction box uses.
        count = f"{n_exact}/8" if n_exact == n_closure else f"{n_exact}/8 \u00b7 {n_closure}/8"
        text(ax, x0, y, hpo, fs=SMALL, ha="left", color=t["tc"], weight="bold")
        text(ax, D_X1, y, count, fs=SMALL, ha="right", color=t["tc"])
        y -= 0.135
        square_rows.append(y)
        for i, c in enumerate(cells):
            cx = cell_x(i)
            if c == "named":
                kw = dict(facecolor=t["mark"], edgecolor=t["mark"], lw=0.6)
            elif c == "desc":
                kw = dict(facecolor="white", edgecolor=t["mark"], lw=0.7, hatch="//////")
            else:
                kw = dict(facecolor="white", edgecolor=RULE, lw=0.6)
            ax.add_patch(mpatches.Rectangle((cx, y - sq / 2), sq, sq, zorder=3, **kw))
        ax.plot([k_x, k_x], [y - sq / 2 - 0.03, y + sq / 2 + 0.03], color=k_colour, lw=0.9,
                ls=(0, (2.0, 1.4)), zorder=4)
        y -= 0.20
    text(ax, k_x, y + 0.075, f"threshold $k={K}$", fs=SMALL, color=k_colour)

    # All candidates enter the vote together.
    arrow(ax, (bus_x, square_rows[0]), (x0 - 0.015, square_rows[0]))

    # Legend for the two fills, each entry centred on its swatch.
    lx, ly = x0, y - 0.10
    ax.add_patch(mpatches.Rectangle((lx, ly - 0.045), 0.09, 0.09, facecolor=P.BLUE,
                                    edgecolor=P.BLUE, lw=0.6, zorder=3))
    text(ax, lx + 0.14, ly, "juror named $v$", fs=SMALL, ha="left", color=P.INK)
    ly -= 0.23
    ax.add_patch(mpatches.Rectangle((lx, ly - 0.045), 0.09, 0.09, facecolor="white",
                                    edgecolor=P.BLUE, lw=0.7, hatch="//////", zorder=3))
    text(ax, lx + 0.14, ly, "juror named a child\nof $v$ (closure only)", fs=SMALL, ha="left",
         color=P.INK)

    # The prediction, and what it carries.
    oy1 = ly - 0.20
    oy0 = oy1 - 0.80
    box(ax, D_X0, oy0, D_X1, oy1, ACC, lw=1.0)
    cx = (D_X0 + D_X1) / 2
    text(ax, cx, oy1 - 0.12, "predicted", fs=SMALL, color=ACC["tc"], weight="bold")
    text(ax, cx, oy1 - 0.28, "HP:0001263", fs=BODY, color=ACC["tc"], weight="bold")
    text(ax, cx, oy1 - 0.47, "Global\ndevelopmental delay", fs=SMALL, color=ACC["tc"])
    text(ax, cx, oy0 + 0.11, "exact 6/8 \u00b7 closure 8/8", fs=SMALL, color=ACC["tc"])
    text(ax, cx, oy0 - 0.11, "the other two fall below $k$", fs=SMALL, color=MUTED,
         style="italic")


def draw():
    """Draw the whole PhenoJury schematic and return the figure."""
    C.style.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()

    rows = row_geometry()
    report(ax, rows)
    jurors(ax, rows)
    bus_x = normalise(ax, rows)
    vote(ax, bus_x)
    return fig


CAPTION = (r"Schematic of PhenoJury. Each sentence is read alone (\emph{A}), every juror names findings "
           r"(\emph{B}), a normaliser maps the names to HPO identifiers (\emph{C}), and a vote "
           r"decides (\emph{D}).")
NOTE = (r"\emph{C}: a prompt-specific parser extracts the candidate strings before normalisation. "
        r"\emph{D}: a juror supports a term $v$ when it produced $v$ (exact rule) or $v$ or a "
        r"descendant of $v$ (closure rule), and $v$ is predicted when at least $k$ jurors support "
        r"it within one unit of text. The sentence is from HCY, the juror outputs and $k=3$ are "
        r"illustrative.")


def main():
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figdir", default=C.DEFAULT_FIGDIR)
    ap.add_argument("--texdir", default=C.DEFAULT_TEXDIR)
    ap.add_argument("--preview", default=None, help="also write a PNG here, for review")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.texdir, exist_ok=True)

    fig = draw()
    out = os.path.join(args.figdir, FIGNAME)
    fig.savefig(out)
    if args.preview:
        fig.savefig(args.preview, dpi=220, format="png")
    plt.close(fig)
    print(f"[fig] {FIGNAME}  ({W:.1f} x {H:.2f} in)")

    C.style.write_tex(os.path.join(args.texdir, TEXNAME), FIGNAME, CAPTION, LABEL,
                      placement="t", generator=GENERATOR, note=NOTE)


if __name__ == "__main__":
    main()
