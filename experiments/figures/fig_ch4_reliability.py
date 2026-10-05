r"""Figure: reliability of the raw acceptance score, on a logit score axis.

    python experiments/figures/fig_ch4_reliability.py

Reads ``score_logit_acceptance.csv`` (per-class counts on one equal-width logit grid) and
``calibration_crossfit.csv`` (SmoothECE), both from the TreePhenoRAG protocol's ``calibration_sample`` -- the same
467k scored HCY pairs under the most frequent configuration. Design, and why:

* **Logit x, linear y.** 89% of pairs score below 0.025 and the threshold sits at 0.99, so a linear
  score axis puts every bin but one in the corner and a log axis squeezes 0.9 / 0.99 / 0.999
  together. The logit axis stretches both ends. Its ticks are printed as scores. The y axis stays a
  plain rate so it can be read off, which is also the scale SmoothECE measures the error on.
  Perfect calibration is therefore the logistic curve, not a diagonal.
* **Raw score only, no intervals.** The cross-fitted Platt curve lies on the dashed line and the
  Wilson bars dominated the old log-log version. Only the raw SmoothECE is quoted (the maps are not used).
* **Sparse bins are merged**, adjacent bins pooled left to right until each holds at least
  ``MIN_PAIRS`` pairs (a short remainder joins the last group). The top of the score range holds a
  handful of pairs per bin, and unmerged its rate jumps between 0.3 and 0.6.
* **x is the bin midpoint** (pair-weighted over merged bins): the table carries counts, not the mean
  score per bin. On a 60-bin grid 0.43 logits wide that is a small error.
* **Two dot strips** under the axis: pairs per bin, all and annotated, each scaled to its own
  largest bin so the shapes compare, not the areas.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import common as C
import palette as P

SCRIPT = "fig_ch4_reliability.py"
#: Drawn at this share of the text width, and included at the same share -- never rescaled.
FRACTION = 0.62
MIN_PAIRS = 50
X_MIN = -12.5          # below this the grid holds ~150 pairs in all, every one unannotated
PROB_TICKS = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
RAW = C.RELIABILITY_STYLE["raw"]["color"]
GREY = P.LIGHT            # all pairs: context
GOLD = RAW                # annotated pairs: the same colour as the curve they are the numerator of


def logit(p: float) -> float:
    """log(p / (1 - p)) for p strictly between 0 and 1."""
    return math.log(p / (1 - p))


def merged_bins(rows: list[dict]) -> list[dict]:
    """Adjacent bins pooled until each holds at least MIN_PAIRS pairs."""
    groups, cur = [], []
    for r in rows:
        cur.append(r)
        if sum(g["n"] for g in cur) >= MIN_PAIRS:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    out = []
    for g in groups:
        n = sum(r["n"] for r in g)
        pos = sum(r["pos"] for r in g)
        out.append(dict(x=sum(r["x"] * r["n"] for r in g) / n, n=n, pos=pos, rate=pos / n))
    return out


def _tick(p: float) -> str:
    return f"$10^{{{int(round(math.log10(p)))}}}$" if p < 0.01 else f"{p:g}"


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    raw = [dict(x=(C.f(r, "lo") + C.f(r, "hi")) / 2, pos=C.f(r, "n_positive"),
                n=C.f(r, "n_positive") + C.f(r, "n_negative"))
           for r in C.table("score_logit_acceptance")]
    shown = [r for r in raw if r["x"] >= X_MIN and r["n"] > 0]
    bins = merged_bins(shown)
    summary = {r["curve"]: r for r in C.table("calibration_crossfit")}
    tau = C.f(next(r for r in C.table("score_distribution_summary")
                   if r["decision"] == "acceptance"), "threshold")
    x_max = max(r["x"] for r in raw) + 0.4

    C.style.apply()
    w = C.WIDTH_FULL * FRACTION
    fig = plt.figure(figsize=(w, w * 0.95))
    fig.set_layout_engine("none")           # hand-placed axes: the strip must align with the plot
    ax = fig.add_axes([0.14, 0.34, 0.80, 0.60])
    sx = fig.add_axes([0.14, 0.12, 0.80, 0.12], sharex=ax)
    fs = C.style.TICK_FONT_SIZE

    grid = np.linspace(X_MIN, x_max, 400)
    ax.plot(grid, 1 / (1 + np.exp(-grid)), color=C.NEUTRAL, lw=0.8, ls=(0, (2, 2)), zorder=1,
            label="perfect calibration")
    xs, ys = [b["x"] for b in bins], [b["rate"] for b in bins]
    ax.plot(xs, ys, color=RAW, lw=0.8, zorder=2)
    ax.scatter(xs, ys, s=10, color=RAW, lw=0, zorder=3, label="raw acceptance score")
    ax.axvline(logit(tau), color=P.DARK, lw=0.6, ls=":", zorder=1)
    ax.text(logit(tau) + 0.25, 0.75, rf"$\tau_{{\mathrm{{accept}}}}={tau:g}$", ha="left",
            va="center", fontsize=fs - 1, color=P.DARK)
    ax.set_xlim(X_MIN, x_max)
    ax.set_ylim(0, 1)
    ax.set_ylabel("observed annotated rate")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", frameon=False, fontsize=fs - 1)
    C.style.light_grid(ax, axis="both")
    C.style.despine(ax)

    # Density: the unmerged grid, so the strips show where pairs are, not where bins were pooled.
    dx = [r["x"] for r in shown]
    n_max, p_max = max(r["n"] for r in shown), max(r["pos"] for r in shown)
    sx.scatter(dx, [0] * len(dx), s=[60 * r["n"] / n_max for r in shown], color=GREY, lw=0,
               alpha=0.8, clip_on=False)
    sx.scatter(dx, [-0.7] * len(dx), s=[60 * r["pos"] / p_max for r in shown], color=GOLD,
               lw=0, alpha=0.8, clip_on=False)
    sx.set_ylim(-1.1, 0.4)
    sx.set_yticks([])
    for side in ("left", "right", "top"):
        sx.spines[side].set_visible(False)
    sx.set_xticks([logit(p) for p in PROB_TICKS])
    sx.set_xticklabels([_tick(p) for p in PROB_TICKS], fontsize=fs - 1)
    sx.set_xlabel("raw acceptance score (logit scale)")
    for y, text in ((0, "all pairs"), (-0.7, "annotated")):
        sx.text(-0.01, y, text, ha="right", va="center", fontsize=fs - 1, color=P.DARK,
                transform=sx.get_yaxis_transform())
    C.save(fig, "fig_ch4_reliability")


    def ece(curve: str) -> str:
        s = summary[curve]
        vals = [C.f(s, k) for k in ("smooth_ece_bc", "smooth_ece_bc_lo", "smooth_ece_bc_hi")]
        fmt = (lambda v: f"{v:.3f}") if vals[0] >= 1e-3 else (
            lambda v: f"{v / 10 ** math.floor(math.log10(v)):.1f}"
                      rf"\times10^{{{math.floor(math.log10(v))}}}" if v > 0 else "0")
        return rf"${fmt(vals[0])}$ [${fmt(vals[1])}$, ${fmt(vals[2])}$]"

    s0 = summary["raw"]
    C.write_figure_tex(
        "fig_ch4_reliability", label="ch4-reliability", script=SCRIPT, fraction=FRACTION,
        caption=(
            r"Reliability diagram of the raw acceptance score on HCY."),
        note=(
            r"All " + f"{int(C.f(s0, 'n')):,}".replace(",", r"\,") + r" scored HCY pairs ("
            + f"{int(C.f(s0, 'n_positive')):,}".replace(",", r"\,") + r" annotated) under the "
            r"most frequent configuration. Score axis on the logit scale, so perfect calibration "
            r"is the dashed logistic curve. Bins are equal-width in the logit, with adjacent bins "
            r"merged until each holds at least " + str(MIN_PAIRS) + r" pairs. Dots below the axis: "
            r"pairs per bin, all and annotated, each scaled to its own largest bin. Bias-corrected "
            r"SmoothECE of the raw score: " + ece("raw") + r"."))


if __name__ == "__main__":
    main()
