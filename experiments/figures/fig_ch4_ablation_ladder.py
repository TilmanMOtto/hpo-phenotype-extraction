r"""Figure: the V0 -> V3 ablation ladder as a schematic -- what each rung changes.

    python experiments/figures/fig_ch4_ablation_ladder.py

Plots nothing. Its content is ``LADDER`` in ``experiments/04_treephenorag/protocol/stages.py``, read
out of that file's source with ``ast`` (importing it would pull in the whole re-run stack), so the
schematic cannot describe a ladder the experiment did not run. The numbers for each rung are
``tab:tpr-ladder``'s job, not this figure's.

Each rung is drawn as panel B of ``fig_ch4_pipeline`` in miniature: the scores of the retrieved segments, two
pooling boxes with the operator written out, the threshold(s), and the two decisions. A box is
highlighted when it differs from the same box one rung to the left. Two things the placeholder in
``04_TreePhenoRAG.tex`` got wrong are drawn as the code has them rather than as it said:

* **One threshold is one box.** V0 and V1 hand the same number to both decisions, so their
  threshold box spans both branches; V2 splits it. "One threshold against two" is then the shape of
  the column, not a word in a cell.
* **V0 -> V1 changes two boxes, not one.** P0 is an indicator, so there is no threshold to tune
  until the pooling becomes P1. Making the score continuous and tuning its one threshold are one
  step, and the figure shows both boxes of it.

The retrieval index is selected by inner CV in every rung (``run_ladder`` hands each one
``space.indices``), so it is not something the ladder varies and is not drawn.
"""
from __future__ import annotations

import ast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import common as C
import palette as P

SCRIPT = "fig_ch4_ablation_ladder.py"
STAGES = C.REPO / "experiments" / "04_treephenorag" / "protocol" / "stages.py"
SOURCE = "experiments/04_treephenorag/protocol/stages.py (LADDER)"

W, H = C.WIDTH_FULL, 2.02
FS = C.style.TICK_FONT_SIZE
TINY = 6.8
BOX_FS = 6.2

# What a rung changes is amber, what it keeps is neutral. The segment scores are the verifier's
# evidence and are the same in every rung, so they take the evidence blue -- never the "changed"
# colour, which would claim they differ between rungs (palette.py, rule 2).
CHANGED = dict(P.box(P.CHANGED), lw=1.1)
SAME = dict(fc=P.WHITE, ec=P.LIGHT, tc=P.INK, lw=0.6)
SUBTITLE = {0: "PhenoRAG's rule", 3: "TreePhenoRAG"}

# The pooled score each rung's operator computes, as tab:tpr-operators defines it. None is a rung
# whose operators are selected, so the box names the selection rule instead of one formula.
POOLING = {"P0": r"$\mathbf{1}[\sigma_{(1)}>\frac{1}{2}]$", "P1": r"$\sigma_{(1)}$"}
SELECTED = {"pr": r"$\Pi_{\mathrm{pr}}$" "\nfewest calls",
            "acc": r"$\Pi_{\mathrm{acc}}$" "\ninner CV"}
TAU_PRUNE = r"$\tau_{\mathrm{prune}}$" "\nrisk control"
TAU_ACCEPT = r"$\tau_{\mathrm{accept}}$" "\ninner CV"

# The illustrative retrieved-segment scores every column pools: the same three scores as fig_ch4_pipeline's panel B, so
# The two figures show one term. A schematic, not a measurement.
SIGMA = [0.93, 0.21, 0.64]


def read_ladder() -> tuple:
    """``LADDER`` from stages.py, parsed, not imported."""
    tree = ast.parse(STAGES.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "LADDER" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no LADDER assignment in {STAGES}")


def threshold(shared: bool, fixed_tau):
    """One string for a threshold both decisions share, a (prune, accept) pair otherwise."""
    if fixed_tau is not None:
        # P0 is in {0, 1}: every threshold in (0, 1] gives the same decisions (stages.py's note).
        return f"one $\\tau={fixed_tau:g}$, fixed" if fixed_tau != 0.5 else \
            r"one $\tau=\frac{1}{2}$, fixed"
    if shared:
        return r"one shared $\tau$, inner CV"
    return (TAU_PRUNE, TAU_ACCEPT)


def specs(ladder) -> list[dict]:
    """Column specifications of the variants V0 to V3, from the ladder the protocol ran."""
    out = []
    for _label, pool_pr, pool_acc, shared, fixed_tau in ladder:
        out.append(dict(pr=POOLING[pool_pr] if pool_pr else SELECTED["pr"],
                        acc=POOLING[pool_acc] if pool_acc else SELECTED["acc"],
                        thr=threshold(shared, fixed_tau)))
    for i, sp in enumerate(out):
        prev = out[i - 1] if i else sp
        sp["c_pool"] = (sp["pr"], sp["acc"]) != (prev["pr"], prev["acc"])
        sp["c_thr"] = sp["thr"] != prev["thr"]
    return out


def box(ax, x, y, w, h, text, st):
    """Draw one box with style *st*."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, facecolor=st["fc"], edgecolor=st["ec"], lw=st["lw"],
        boxstyle="round,pad=0,rounding_size=0.04", zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=BOX_FS, color=st["tc"],
            linespacing=1.25, zorder=3)


def arrow(ax, a, b):
    """Draw an arrow from *a* to *b*."""
    ax.add_patch(mpatches.FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5.5,
                                          color=P.DARK, lw=0.6, shrinkA=0.5, shrinkB=0.5,
                                          zorder=1))


def rung(ax, x, y_top, cw, sp) -> tuple[float, float, float]:
    """One column: retrieved-segment scores -> two poolings -> threshold(s) -> expand / accept.
    Returns the y of the pooling, threshold and decision rows, for the row labels."""
    bw, bh = cw * 0.62, 0.075
    bx0 = x - bw / 2 + 0.05
    for k, s in enumerate(SIGMA):
        y = y_top - k * 0.11
        ax.text(bx0 - 0.06, y, f"$\\sigma_{k + 1 if k < 2 else 'S'}$", ha="right", va="center",
                fontsize=5.6, color=P.DARK)
        ax.add_patch(mpatches.Rectangle((bx0, y - bh / 2), bw, bh, fc=P.FILL, ec="none"))
        ax.add_patch(mpatches.Rectangle((bx0, y - bh / 2), bw * s, bh,
                                        fc=P.EVIDENCE if s > 0.5 else P.LIGHT, ec="none",
                                        alpha=0.85))
    # The 1/2 line: greedy decoding answers Yes iff a bar crosses it (P0 reads nothing else).
    ax.plot([bx0 + bw / 2] * 2, [y_top + 0.07, y_top - 0.29], color=P.DARK, lw=0.5,
            ls=(0, (2, 1.5)))

    xl, xr, pw = x - cw * 0.25, x + cw * 0.25, cw * 0.48
    yb = y_top - 0.36
    yp = yb - 0.26
    arrow(ax, (x - 0.05, yb), (xl, yp + 0.14))
    arrow(ax, (x + 0.05, yb), (xr, yp + 0.14))
    pool = CHANGED if sp["c_pool"] else SAME
    box(ax, xl, yp, pw, 0.26, sp["pr"], pool)
    box(ax, xr, yp, pw, 0.26, sp["acc"], pool)

    yt = yp - 0.40
    thr = CHANGED if sp["c_thr"] else SAME
    if isinstance(sp["thr"], str):            # one threshold, one box across both branches
        box(ax, x, yt, cw - 0.04, 0.24, sp["thr"], thr)
    else:
        box(ax, xl, yt, pw, 0.24, sp["thr"][0], thr)
        box(ax, xr, yt, pw, 0.24, sp["thr"][1], thr)
    yd = yt - 0.27
    for xx in (xl, xr):
        arrow(ax, (xx, yp - 0.13), (xx, yt + 0.12))
        arrow(ax, (xx, yt - 0.12), (xx, yd + 0.06))
    ax.text(xl, yd, "expand", ha="center", va="center", fontsize=BOX_FS,
            color=P.shade(P.EXPANSION, 0.35))
    ax.text(xr, yd, "accept", ha="center", va="center", fontsize=BOX_FS,
            color=P.shade(P.ACCEPTANCE, 0.35))
    return yp, yt, yd


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    cols = specs(read_ladder())

    C.style.apply()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()

    x0, cw, gap = 0.80, 1.12, 0.06          # first column's left edge, column width, gutter
    xs = [x0 + 0.5 * cw + i * (cw + gap) for i in range(len(cols))]
    y_head, y_top = H - 0.10, H - 0.42

    for i, (x, sp) in enumerate(zip(xs, cols)):
        ax.text(x, y_head, f"V{i}", ha="center", va="center", fontsize=FS + 0.5,
                weight="bold", color=P.INK)
        if i in SUBTITLE:
            ax.text(x, y_head - 0.12, SUBTITLE[i], ha="center", va="center", fontsize=TINY,
                    color=P.MID, style="italic")
        if i:                                 # The ladder's direction, between column headers
            ax.annotate("", xy=(x - cw / 2 + 0.02, y_head),
                        xytext=(xs[i - 1] + cw / 2 - 0.02, y_head),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=P.MID,
                                        mutation_scale=6.5))
        yp, yt, yd = rung(ax, x, y_top, cw, sp)

    for y, text in ((y_top - 0.11, "segment\nscores"), (yp, "pooling"), (yt, "threshold"),
                    (yd, "decision")):
        ax.text(0.08, y, text, ha="left", va="center", fontsize=TINY + 0.4, color=P.INK,
                linespacing=1.15)

    ax.add_patch(mpatches.Rectangle((x0, 0.04), 0.16, 0.10, facecolor=CHANGED["fc"],
                                    edgecolor=CHANGED["ec"], lw=0.8))
    ax.text(x0 + 0.22, 0.09, "changed from the variant to its left", ha="left", va="center",
            fontsize=TINY, color=P.INK)

    C.save(fig, "fig_ch4_ablation_ladder")
    C.write_figure_tex(
        "fig_ch4_ablation_ladder", label="ch4-ablation-ladder", script=SCRIPT, source=SOURCE,
        caption=(
            r"Ablation ladder from PhenoRAG's decision rule (V0) to the full TreePhenoRAG (V3). "
            r"Each variant changes the highlighted components of the one to its left."),
        note=(
            r"Each variant is the step for one term (\Cref{fig:ch4-pipeline}\,B) on the same "
            r"segment scores. A threshold box spanning both decisions is one value used for both."))


if __name__ == "__main__":
    main()
