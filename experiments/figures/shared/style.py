r"""Shared publication style for the thesis figures.

Every figure script imports this module and nothing else visual, so each ablation axis ->
colour / linestyle / marker / hatch mapping is defined once and is identical across all
figures. Two axes live here: tau_prune (TreePhenoRAG, TAU_STYLE) and the retrieval gate family
(an earlier exploratory run, GATE_STYLE).

Design constraints (thesis-imposed):
  * vector PDF only -- no rasterised artists, no bitmap fallback, no Type-3 fonts;
  * serif body font matching the NeurIPS/ETH template (which loads Times), 9pt base, 8pt ticks;
  * explicit figure widths in inches -- never rescaled with \includegraphics[scale=], so the
    rendered text size is identical in every figure;
  * no figure titles (captions live in LaTeX);
  * colourblind-safe palette (Okabe-Ito) and every colour paired with a second, redundant cue.
"""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.font_manager import FontProperties, findfont

# The thesis template loads Times. Liberation Serif and Nimbus Roman are metric-compatible clones
# and are what a Linux box actually has. Register them explicitly rather than relying on
# matplotlib's font cache, which on this machine had gone stale and silently fell back to STIX.
_SYSTEM_SERIF_GLOBS = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-*.ttf",
    "/usr/share/fonts/**/NimbusRoman-*.otf",
    "/usr/share/fonts/**/*imes*New*Roman*.ttf",
    os.path.expanduser("~/.fonts/**/*.ttf"),
]
for _pat in _SYSTEM_SERIF_GLOBS:
    for _p in glob.glob(_pat, recursive=True):
        try:
            _fm.fontManager.addfont(_p)
        except Exception:
            pass

# --------------------------------------------------------------------------------------------
# Figure geometry.  Set the width explicitly. Do not rescale in LaTeX.
# --------------------------------------------------------------------------------------------
WIDTH_FULL = 5.5   # inches -- full text width
WIDTH_HALF = 3.3   # inches -- two-column subfigure

BASE_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8

# Times-metric serifs first, then matplotlib's bundled STIX (Times-compatible), then DejaVu.
_SERIF_STACK = [
    "Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif",
    "STIXGeneral", "FreeSerif", "Times", "DejaVu Serif",
]

RC = {
    # --- vector output, no rasterisation, no Type-3 ------------------------------------------
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "image.composite_image": False,
    "agg.path.chunksize": 0,
    "path.simplify": False,          # keep every vertex. No lossy path decimation
    "savefig.dpi": 600,              # only relevant if something ever rasterises. It must not
    "figure.dpi": 600,
    "savefig.format": "pdf",
    # NOT "tight": a tight bbox crops to the drawn content, so every figure would come out a
    # different width and \includegraphics[width=...] would rescale the text by a different factor
    # in each one. The figsize IS the deliverable width. Constrained layout fits the content to it.
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "savefig.transparent": False,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.010,
    "figure.constrained_layout.w_pad": 0.010,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,

    # --- typography ---------------------------------------------------------------------------
    "font.family": "serif",
    "font.serif": _SERIF_STACK,
    "mathtext.fontset": "stix",
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": BASE_FONT_SIZE,

    # --- axes -----------------------------------------------------------------------------------
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelpad": 2.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.4,
    "lines.markeredgewidth": 0.8,
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.5,

    # --- ticks ------------------------------------------------------------------------------------
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,

    # --- grid: light, and only where a script switches it on --------------------------------------
    "axes.grid": False,
    "grid.color": "#B8B8B8",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.55,

    # --- legend: inside the axes, frameless, 8pt --------------------------------------------------
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.30,
    "legend.borderpad": 0.2,
    "legend.borderaxespad": 0.4,
    "legend.columnspacing": 1.0,
}


def apply() -> None:
    """Install the style. Call once at the top of every figure script."""
    plt.rcParams.update(RC)


def resolved_serif() -> str:
    """The font file matplotlib actually resolved for the serif stack (printed by each script)."""
    return findfont(FontProperties(family="serif"))


# --------------------------------------------------------------------------------------------
# tau_prune encoding -- IDENTICAL IN EVERY FIGURE.
#
# Okabe-Ito colourblind-safe palette. Colour is never the only cue: lines also differ in dash
# pattern, points in marker shape, bars in hatch.
# --------------------------------------------------------------------------------------------
TAU_PRUNE_ORDER = [0.0045, 0.016, 0.075, 0.21, 0.62]

TAU_STYLE = {
    0.0045: dict(color="#0072B2", ls="solid",                        marker="o", hatch=""),
    0.016:  dict(color="#009E73", ls=(0, (5.0, 1.6)),                marker="s", hatch="///"),
    0.075:  dict(color="#E69F00", ls=(0, (1.2, 1.2)),                marker="^", hatch="..."),
    0.21:   dict(color="#CC79A7", ls=(0, (6.0, 1.6, 1.2, 1.6)),      marker="D", hatch="xxx"),
    0.62:   dict(color="#333333", ls=(0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)), marker="v", hatch="\\\\\\"),
}

# The naive default a practitioner would reach for before seeing any of this: both thresholds at
# 0.5. Marked identically in every figure, in a shape and colour used for nothing else.
INTUITIVE_TAU = 0.5
INTUITIVE_STYLE = dict(color="#000000", marker="*", markersize=8.5, markeredgewidth=0.7)
INTUITIVE_LABEL = (r"naive default: $\tau_{\mathrm{prune}}=\tau_{\mathrm{accept}}=%g$"
                   % INTUITIVE_TAU)

# Baselines drawn as horizontal reference lines. Colour + dash pattern again both carry identity.
BASELINE_STYLE = {
    "phenobert":          dict(color="#7F7F7F", ls=(0, (4.0, 2.0)),           label="PhenoBERT"),
    "raghpo_llama70b":    dict(color="#7F7F7F", ls=(0, (1.0, 1.6)),           label="RAG-HPO (Llama-70B)"),
    "raghpo_llama8b":     dict(color="#7F7F7F", ls=(0, (5.0, 1.4, 1.0, 1.4)), label="RAG-HPO (Llama-8B)"),
    "raghpo_paper70b":    dict(color="#A9A9A9", ls=(0, (2.0, 1.4)),           label="RAG-HPO (paper, 70B)"),
    "topm_retrieval_slm": dict(color="#A9A9A9", ls=(0, (6.0, 1.2, 1.0, 1.2, 1.0, 1.2)),
                               label="Top-$M$ retrieval + SLM"),
    "tree_no_prune":      dict(color="#A9A9A9", ls=(0, (3.0, 3.0)),
                               label=r"Single-threshold tree ($\tau_{\mathrm{prune}}{=}\tau_{\mathrm{accept}}$)"),
}


def tau_label(tau: float) -> str:
    """Consistent inline/legend text for one tau_prune value."""
    return rf"$\tau_{{\mathrm{{prune}}}}={tau:g}$"


def sty(tau: float) -> dict:
    """Style dict of one threshold."""
    return TAU_STYLE[tau]


def line_kw(tau: float, **over) -> dict:
    """Line keyword arguments for one threshold."""
    s = TAU_STYLE[tau]
    kw = dict(color=s["color"], linestyle=s["ls"], marker=s["marker"])
    kw.update(over)
    return kw


def bar_kw(tau: float, **over) -> dict:
    """Bar keyword arguments for one threshold."""
    s = TAU_STYLE[tau]
    kw = dict(facecolor=s["color"], hatch=s["hatch"], edgecolor="white")
    kw.update(over)
    return kw


def despine(ax, left=True, bottom=True) -> None:
    """Hide the top and right spines (and optionally the left and bottom)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def light_grid(ax, axis="both") -> None:
    """Draw a light grid on *axis*."""
    ax.grid(True, axis=axis, color=RC["grid.color"], lw=RC["grid.linewidth"],
            alpha=RC["grid.alpha"], zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------------
# Label placement.
#
# These figures have five curves, seven iso-F1 contours, five endpoint annotations and a legend
# competing for one axes. Hand-tuned offsets break the moment a number moves, so labels are placed
# by clearance search instead: propose candidate positions, score each by how far it sits from
# everything already on the axes, take the best. Everything is done in AXES FRACTION coordinates so
# The two axis scales are commensurate.
# --------------------------------------------------------------------------------------------
class Occupancy:
    """What is already drawn on an axes, for clearance-based label placement."""

    def __init__(self, ax):
        self.ax = ax
        self.pts = []        # [(x, y)] in axes fraction
        self.rects = []      # [(x0, y0, x1, y1)] in axes fraction

    def _to_frac(self, x, y):
        import numpy as _np
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        return _np.asarray(x, float) - x0, _np.asarray(y, float) - y0, (x1 - x0), (y1 - y0)

    def add_data_points(self, xs, ys, stride=1):
        """Mark the axes area covered by data points, every *stride*-th point."""
        import numpy as _np
        dx, dy, w, h = self._to_frac(xs, ys)
        f = _np.column_stack([dx / w, dy / h])[::stride]
        self.pts.extend(map(tuple, f))

    def add_axes_rect(self, x0, y0, x1, y1):
        """Mark a rectangle in axes coordinates as occupied."""
        self.rects.append((x0, y0, x1, y1))

    def add_axes_point(self, x, y):
        """Mark a point in axes coordinates as occupied."""
        self.pts.append((x, y))

    def clearance(self, fx, fy):
        """Distance from an axes-fraction point to the nearest occupied thing; 0 inside a rect."""
        import numpy as _np
        for (x0, y0, x1, y1) in self.rects:
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                return 0.0
        best = min(abs(fx), abs(1 - fx), abs(fy), abs(1 - fy))   # keep clear of the axes edges too
        if self.pts:
            p = _np.asarray(self.pts)
            d = _np.hypot(p[:, 0] - fx, p[:, 1] - fy).min()
            best = min(best, float(d))
        return best

    def best(self, candidates_data):
        """Pick the (x, y) data-coordinate candidate with the greatest clearance."""
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        scored = []
        for (cx, cy) in candidates_data:
            fx, fy = (cx - x0) / w, (cy - y0) / h
            if not (0.01 <= fx <= 0.99 and 0.01 <= fy <= 0.99):
                continue
            scored.append((self.clearance(fx, fy), cx, cy))
        if not scored:
            return candidates_data[0]
        scored.sort(reverse=True)
        _, cx, cy = scored[0]
        self.add_data_points([cx], [cy])
        return cx, cy

    def _hits(self, bb, pad=0.0):
        """True if a drawn box leaves the axes, overlaps a reserved rectangle, or covers a point."""
        x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
        if x0 < 0.004 or x1 > 0.996 or y0 < 0.004 or y1 > 0.996:
            return True                       # a label outside the axes would change the PDF width
        for (a0, b0, a1, b1) in self.rects:
            if x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1:
                return True
        for (px, py) in self.pts:
            if x0 <= px <= x1 and y0 <= py <= y1:
                return True
        return False

    def _overlap(self, bb):
        """How badly a drawn box collides: overlap area with reserved rects, plus penalties for
        leaving the axes and for covering occupied points. 0.0 means completely clear."""
        cost = 0.0
        for (a0, b0, a1, b1) in self.rects:
            w = min(bb.x1, a1) - max(bb.x0, a0)
            h = min(bb.y1, b1) - max(bb.y0, b0)
            if w > 0 and h > 0:
                cost += w * h
        out = (max(0.0, 0.004 - bb.x0) + max(0.0, bb.x1 - 0.996)
               + max(0.0, 0.004 - bb.y0) + max(0.0, bb.y1 - 0.996))
        cost += 4.0 * out                     # leaving the axes is worse than overlapping a label
        cost += 2e-4 * sum(1 for (px, py) in self.pts
                           if bb.x0 <= px <= bb.x1 and bb.y0 <= py <= bb.y1)
        return cost

    def place(self, fig, candidates_data, make_artist, pad=0.010, tries=48):
        """Place a label by trial: candidates best-clearance-first, keep the first that lands clear.

        Scoring an evidence location point is not enough -- a two-line label is wide, so the evidence location can be
        clear while the text is not. Each candidate is therefore actually drawn and measured. If
        none is completely clear, the one with the SMALLEST overlap wins. Taking the best evidence location
        instead is what let two labels land on top of each other.
        """
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        scored = []
        for (cx, cy) in candidates_data:
            fx, fy = (cx - x0) / w, (cy - y0) / h
            if not (0.015 <= fx <= 0.985 and 0.015 <= fy <= 0.985):
                continue
            scored.append((self.clearance(fx, fy), cx, cy))
        if not scored:
            scored = [(0.0, candidates_data[0][0], candidates_data[0][1])]
        scored.sort(key=lambda t: -t[0])

        best = None
        for _, cx, cy in scored[:tries]:
            art = make_artist(cx, cy)
            fig.canvas.draw()
            bb = art.get_window_extent().transformed(self.ax.transAxes.inverted())
            cost = self._overlap(bb)
            if cost <= 0.0:
                self.reserve(fig, art, pad)
                return art
            if best is None or cost < best[0]:
                best = (cost, cx, cy)
            art.remove()
        art = make_artist(best[1], best[2])
        self.reserve(fig, art, pad)
        return art

    def reserve(self, fig, artist, pad=0.012):
        """Register a drawn artist's real extent (legend, text, annotation) as occupied.

        Reserving the measured bounding box -- not just the evidence location point -- is what stops two
        multi-line labels from being placed on top of each other.
        """
        fig.canvas.draw()
        try:
            bb = artist.get_window_extent()
        except TypeError:                                  # some artists need the renderer
            bb = artist.get_window_extent(fig.canvas.get_renderer())
        bb = bb.transformed(self.ax.transAxes.inverted())
        self.add_axes_rect(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
        # sample the rectangle so `clearance` pushes candidates away from it, not merely out of it
        import numpy as _np
        gx, gy = _np.meshgrid(_np.linspace(bb.x0, bb.x1, 7), _np.linspace(bb.y0, bb.y1, 4))
        self.pts.extend(zip(gx.ravel(), gy.ravel()))

    # backwards-compatible alias
    legend_rect = reserve


def save(fig, path, preview_png=None) -> None:
    """Write a pure-vector PDF, asserting nothing in it was rasterised.

    Any missing-glyph warning is promoted to an error: a silently dropped character in a thesis
    figure is worse than a crash. `preview_png` is a review convenience only -- the deliverable in
    figures/ is always the PDF.
    """
    import warnings

    for ax in fig.get_axes():
        for art in list(ax.get_children()):
            if getattr(art, "get_rasterized", None) and art.get_rasterized():
                raise RuntimeError(f"rasterised artist would break vector output: {art!r}")
    w, h = fig.get_size_inches()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*[Gg]lyph.*missing from font.*")
        fig.savefig(path, format="pdf", bbox_inches=None)
        if preview_png:
            fig.savefig(preview_png, format="png", dpi=300, bbox_inches=None)
    plt.close(fig)
    print("[style] %s  %.2f x %.2f in (vector PDF, fonttype 42, no rasterised artists)"
          % (os.path.basename(path), w, h))


# --------------------------------------------------------------------------------------------
# Retrieval-gate encoding -- for the an earlier exploratory run retrieval figures.
#
# A "gate" is a rule that decides which retrieved sentences are forwarded to the SLM. The three
# families below are the ablation axis of the retrieval step, and this mapping is what makes them
# comparable across figures. Same Okabe-Ito palette as TAU_STYLE, disjoint assignment: colour is
# again never the only cue -- each family also owns a dash pattern and a marker shape.
# --------------------------------------------------------------------------------------------
GATE_ORDER = ["topk", "tau"]

GATE_STYLE = {
    # The shipped gate: keep the k best-ranked sentences, k fixed for every query
    "topk": dict(color="#0072B2", ls="solid",         marker="o", hatch=""),
    # absolute cosine cutoff, one global value for every query
    "tau":  dict(color="#E69F00", ls=(0, (5.0, 1.6)), marker="s", hatch="..."),
}

GATE_LABEL = {
    "topk": r"top-$k$ (fixed rank cutoff)",
    "tau":  r"global $\tau$ (absolute cosine cutoff)",
}

# The configuration the pipeline actually ships (top_n=5). Marked identically wherever it appears,
# in a shape used for nothing else -- the same role INTUITIVE_STYLE plays in the tau figures.
SHIPPED_TOP_N = 5
SHIPPED_STYLE = dict(color="#000000", marker="*", markersize=9.0, markeredgewidth=0.7)


def gate_kw(gate: str, **over) -> dict:
    """Line keyword arguments for one retrieval condition."""
    s = GATE_STYLE[gate]
    kw = dict(color=s["color"], linestyle=s["ls"], marker=s["marker"])
    kw.update(over)
    return kw


# --------------------------------------------------------------------------------------------
# LaTeX include snippets.
#
# The caption of a results figure quotes numbers, and a number typed by hand into a .tex file
# drifts away from the PDF beside it the first time the analysis is re-run. So the snippet is
# generated by the same script that draws the figure, from the same variables, and is overwritten
# on every run. Edit the figure script, never the generated .tex.
# --------------------------------------------------------------------------------------------
TEX_PREAMBLE_NOTE = r"""% Requires in the preamble (neurips_2025.sty does NOT load graphicx):
%     \usepackage{graphicx}
%     \graphicspath{{thesis_figures/}}
% The PDF is drawn at exactly \textwidth = 5.5in, so width=\textwidth does not rescale it and
% the figure text renders at the same 9pt/8pt as every other figure. Do not use [scale=]."""


def note_tex(note) -> list:
    r"""The note below a float: the definitions a caption no longer carries.

    Captions say what is shown and the takeaway. Every legend term, interval type and abbreviation
    goes here, in the thesis's own convention (hand-typed floats in the sections use the same two
    lines), so a generated float and a typed one look alike. Empty for no note.
    """
    if not note or not note.strip():
        return []
    return [r"  \par\vspace{3pt}",
            r"  \begin{minipage}{\linewidth}\footnotesize %s\end{minipage}" % note.strip()]


def _split_title(caption: str):
    r"""``(title, rest)`` of a caption opening with ``\textbf{title}``, or ``None`` if it does not."""
    caption = caption.strip()
    head = r"\textbf{"
    if not caption.startswith(head):
        return None
    depth = 0
    for i in range(len(head) - 1, len(caption)):
        depth += {"{": 1, "}": -1}.get(caption[i], 0)
        if depth == 0:
            return caption[len(head):i].strip(), caption[i + 1:]
    raise ValueError(f"unbalanced braces in caption: {caption[:80]!r}")


def plain_caption(caption: str) -> str:
    r"""The thesis's caption for a figure or table: the same text, its leading ``\textbf{...}``
    title unbolded. Figures drop the note below them (see :func:`write_tex`), not the caption."""
    split = _split_title(caption)
    return (split[0] + split[1]).strip() if split else caption.strip()


def write_tex(path, pdf_name, caption, label, width=r"\textwidth", placement="tb",
              generator=None, note=None) -> None:
    r"""Write a standalone \begin{figure} snippet next to the PDF, and say where it came from.

    Figures carry their caption only (:func:`plain_caption`), so *note* is accepted and not printed.
    """
    gen = generator or "scripts/figures/"
    body = "\n".join([
        "% GENERATED FILE -- do not edit by hand.",
        "%% Regenerate with: %s" % gen,
        TEX_PREAMBLE_NOTE,
        r"\begin{figure}[%s]" % placement,
        r"  \centering",
        r"  \includegraphics[width=%s]{%s}" % (width, pdf_name),
        r"  \caption{%s}" % plain_caption(caption),
        r"  \label{%s}" % label,
        r"\end{figure}",
        "",
    ])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print("[style] %s  (LaTeX include snippet)" % os.path.basename(path))
