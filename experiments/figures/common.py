r"""Shared plumbing for the chapter-4 (the TreePhenoRAG protocol) thesis figures and tables.

Three things live here and nowhere else:

* **where the numbers come from** -- every script reads ``output/treephenorag_protocol/tables``
  and nothing else, so a figure can never disagree with the table beside it;
* **the visual system** -- imported from ``experiments/figures/style.py`` rather than redefined, so
  chapter 4 renders identically to the figures already in the thesis (Times-metric serif at 9pt,
  vector-only PDF, no Type-3, explicit widths that LaTeX must not rescale);
* **the chapter-4 encodings** -- the ladder rungs and the two retrieval indices get one
  colour/marker/hatch mapping, defined once and used by every figure that shows them.

Colour is never the only cue. Every series also differs in marker, dash pattern or hatch, because
the thesis has to survive greyscale printing and because roughly one reader in twelve cannot
separate the palette by hue alone.
"""
from __future__ import annotations

import csv
import re
import os
import sys
from pathlib import Path
from hpo_extraction.paths import results_dir, thesis_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TABLES = results_dir() / "treephenorag_protocol" / "tables"
RESULTS = results_dir() / "treephenorag_protocol"
# Both output directories live INSIDE the thesis subtree: Overleaf compiles thesis/ as its
# project root, so a float it cannot reach by a relative path might as well not exist.
FIG_DIR = thesis_dir() / "thesis_figures"
TEX_DIR = thesis_dir() / "thesis_figures_latex"

# The existing figure style module is the single source of typography and geometry for the whole
# thesis. Importing it (rather than copying it) is what keeps chapter 4 from drifting visually.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))  # style.py
import style  # noqa: E402

import palette as P  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

WIDTH_FULL = style.WIDTH_FULL
WIDTH_HALF = style.WIDTH_HALF

# ── The chapter-4 encodings ──────────────────────────────────────────────────
# Colours come from palette.py. Chapter 4 is TreePhenoRAG's, so its ordered variants are
# lightness steps of TreePhenoRAG's purple, darker for the later rung. V0 is PhenoRAG's rule
#, not TreePhenoRAG's, so it takes the baseline grey. The oracle is not a configuration
# anyone could run, so it is drawn in the light context grey, outlined by its hatch.
RUNG_STYLE = {
    "V0": dict(color=P.BASELINE, hatch="",     marker="v", label=r"V0 \-- $P_0$ both"),
    "V1": dict(color=P.TREE_RAMP[0], hatch="///",  marker="s", label=r"V1 \-- one shared threshold"),
    "V2": dict(color=P.TREE_RAMP[1], hatch="...",  marker="o", label=r"V2 \-- risk control"),
    "V3": dict(color=P.TREE_RAMP[2], hatch="xxx",  marker="D", label=r"V3 \-- selected poolings"),
    "oracle": dict(color=P.LIGHT, hatch="\\\\\\", marker="*",
                   label="oracle expansion"),
}

# The names match fig_ch4_retrieval's legend word for word: one index, one name, in every float.
# Colours match fig_ch4_pipeline's index boxes: the ontology index is verifier evidence (blue),
# The synthetic-sentence index amber.
INDEX_STYLE = {
    "ontology_r3": dict(color=P.EVIDENCE, marker="o", ls="solid", hatch="",
                        label="label + definition + synonyms + descendant closure"),
    "exemplar": dict(color=P.AMBER, marker="s", ls=(0, (5.0, 1.6)), hatch="///",
                     label="synthetic sentences + descendant closure"),
}

# The prose name of each index, taken from the plotting style so a table and a figure legend
# cannot end up calling the same index two different things.
INDEX_LABEL = {k: v["label"] for k, v in INDEX_STYLE.items()}

# Buckets of the recall decomposition, in the order 04_TreePhenoRAG.tex defines them: the loss
# moves down the pipeline left to right, so the reader's eye follows the term's journey. They are
# errors, so they take palette.error_ramp -- in the order fig_ch4_recall_decomposition draws them
# (retrieval first), so the bar darkens monotonically left to right. Colour alone separates the
# segments (no hatching): each is also labelled with its share. The residual is unattributed and
# stays neutral.
BUCKET_ORDER = ["pruning", "retrieval", "judgement", "pooling", "residual"]
_BUCKET_RAMP = P.error_ramp(4)
BUCKET_STYLE = {
    "pruning":   dict(color=_BUCKET_RAMP[1]),
    "retrieval": dict(color=_BUCKET_RAMP[0]),
    "judgement": dict(color=_BUCKET_RAMP[2]),
    "pooling":   dict(color=_BUCKET_RAMP[3]),
    "residual":  dict(color=P.LIGHT),
}

# False positives filed by their relation to the report's annotated terms, nearest first. Shared
# verbatim by fig_ch4_error_analysis and fig_ch5_error_analysis, so the two chapters' bars read
# against each other: one ordered error ramp, light for a near miss on the annotated term's path,
# darkest for a term unrelated to anything annotated. Colour alone separates the segments (no
# hatching). Each is also labelled with its share.
_FP_RAMP = P.error_ramp(5)
FP_RELATION_SEGMENTS = [
    ("ancestor", "ancestor", dict(color=_FP_RAMP[0])),
    ("descendant", "descendant", dict(color=_FP_RAMP[1])),
    ("sibling", "sibling", dict(color=_FP_RAMP[2])),
    ("same_branch", "same branch", dict(color=_FP_RAMP[3])),
    ("unrelated", "unrelated", dict(color=_FP_RAMP[4])),
]

#: The reliability figure's two curves: the method's raw score in its hue, the recalibration grey.
RELIABILITY_STYLE = {
    "raw": dict(color=P.TREE, marker="o", ls=(0, (1.2, 1.2)), label="raw score"),
    "platt_crossfit": dict(color=P.BASELINE, marker="s", ls="solid",
                           label="cross-fitted Platt"),
}

CALIB_STYLE = {
    "identity": dict(color=P.TREE, marker="o", ls=(0, (1.2, 1.2)), label="raw (identity)"),
    "platt":    dict(color=P.SKY, marker="s", ls=(0, (5.0, 1.6)), label="Platt"),
    "isotonic": dict(color=P.BLUE, marker="D", ls="solid", label="isotonic"),
}

NEUTRAL = P.INK


# ── Reading the results ──────────────────────────────────────────────────────

def table(name: str) -> list[dict]:
    """One results CSV as a list of dicts, numbers left as strings for the caller to cast."""
    path = TABLES / f"{name}.csv"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}.\nRun the experiment first, or pull its tables from the cluster:\n"
            f"  rsync -rlt <LEOMED_HOST>:.../output/treephenorag_protocol/tables/ {TABLES}/")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def has_table(name: str) -> bool:
    """True when the TreePhenoRAG protocol wrote ``<name>.csv``."""
    return (TABLES / f"{name}.csv").is_file()


def result_file(relative: str) -> Path:
    """*relative* resolved for reading: ``output/...`` is the results folder (``results_dir`` of the path
    file), anything else is relative to the repository. The ``output/...`` form is what the
    generated LaTeX names as its source, so it stays the same on every machine."""
    prefix = "output/"
    return results_dir() / relative[len(prefix):] if relative.startswith(prefix) else REPO / relative


def external_table(relative: str, *, rsync_hint: str = "") -> list[dict]:
    """A results CSV from an experiment other than TreePhenoRAG protocol, by repo-relative path.

    one chapter-4 figure needs this: the retrieval sweep is the retrieval-curve analysis's, because measuring
    a gate over the whole query set needs the retrieval *scores*, which the TreePhenoRAG protocol's cache does not
    carry -- it holds only what the traversal actually reached. Everything else in the chapter
    reads `table()` and must keep doing so. This is a named exception, not a second loading path.

    Exits 3 (not 1) when the file is absent, which is how every driver here distinguishes "this
    artifact has no source yet" from a crash.
    """
    path = result_file(relative)
    if not path.is_file():
        hint = ("\n  pull it with: " + rsync_hint) if rsync_hint else ""
        print(f"  skipped: missing {path}{hint}", file=sys.stderr)
        raise SystemExit(MISSING_SOURCE_EXIT)
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict, key: str, default: float = float("nan")) -> float:
    """A float from a CSV cell, tolerating the empty strings DictWriter leaves for absent keys."""
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


# ── Marking the winner ───────────────────────────────────────────────────────

def bold_best(rows: list, columns, *, higher_is_better: bool = True,
              skip: "set[int] | None" = None) -> None:
    r"""Wrap the winning cell of each named column in ``\textbf{}``, **in place, per block**.

    "Per block" is the whole subtlety. A results table here is several cohorts stacked under
    ``\multicolumn`` headers, and the comparison is *within* a cohort: bolding the best cell of the
    whole column would mark GSC+'s winner and leave the HCY block with none, which reads as HCY
    having no best system. So a non-list row (a group header, a ``\midrule``) **closes the current
    block**, and each block gets its own winner.

    ``skip`` is a set of row indices excluded from the contest but still printed -- the oracle rung
    of the ablation ladder, and the marked reference rows that are not competitors. Bolding a row
    the text calls "not an configuration" would make it the answer to a question it is not in.

    A cell is compared by the first float appearing in it, so ``0.309 \ [0.273, 0.343]`` competes
    on its point estimate and a cell with no number (``--``) never wins. Ties are left unbolded on
    purpose: two bold cells in a column say "these are equal", which is true and is what the
    reader should see, but a table that silently picks the first of two equal rows says something
    false about ordering.
    """
    skip = skip or set()
    if isinstance(columns, int):
        columns = [columns]

    block: list[int] = []

    def close(block_rows):
        if not block_rows:
            return
        for col in columns:
            best, winners = None, []
            for i in block_rows:
                value = _leading_float(rows[i][col])
                if value != value:      # NaN: no number in the cell
                    continue
                if best is None or (value > best if higher_is_better else value < best):
                    best, winners = value, [i]
                elif value == best:
                    winners.append(i)
            if best is not None and len(winners) == 1:
                rows[winners[0]][col] = _embolden(rows[winners[0]][col])

    for i, row in enumerate(rows):
        if not isinstance(row, list):
            close(block)
            block = []
            continue
        if i not in skip:
            block.append(i)
    close(block)


def mark_best_second(rows: list, columns, *, higher_is_better: bool = True,
                     skip: "set[int] | None" = None, whole_cell: bool = False,
                     second: bool = True) -> None:
    r"""Bold every cell holding the best PRINTED value and underline the second, per block.

    The thesis's own convention, and it differs from :func:`bold_best` in two ways: ties at the
    printed precision are all bold (two rows showing 0.75 are equally best to the reader), and the
    second-best printed value is underlined (``second=False`` marks the best only). Blocks are
    closed by any non-list row, as in ``bold_best``, and ``skip`` excludes rows from the
    contest -- the oracle rung of the ablation ladder -- while still printing them.

    A cell competes on its leading token, so ``0.31 [0.27, 0.34]`` competes on 0.31 and
    ``30\,858`` on 30858. A cell with no number (``--``) never wins.
    """
    skip = skip or set()
    columns = [columns] if isinstance(columns, int) else columns

    def close(block):
        for col in columns:
            vals = {}
            for i in block:
                token = str(rows[i][col]).split(" ")[0].replace(r"\,", "").strip("$")
                try:
                    vals[i] = float(token)
                except ValueError:
                    continue
            distinct = sorted(set(vals.values()), reverse=higher_is_better)
            if len(distinct) < 2 and len(vals) < 2:
                continue
            marks = (("textbf", 0), ("underline", 1)) if second else (("textbf", 0),)
            for cmd, rank in marks:
                if rank < len(distinct):
                    for i, v in vals.items():
                        if v == distinct[rank]:
                            rows[i][col] = _wrap_leading(rows[i][col], cmd, whole_cell)

    block = []
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            close(block)
            block = []
        elif i not in skip:
            block.append(i)
    close(block)


def _wrap_leading(cell: str, cmd: str, whole: bool) -> str:
    """Wrap the cell's leading token (the point estimate) -- or the whole cell -- in ``cmd``."""
    if whole:
        return rf"\{cmd}{{{cell}}}"
    head, sep, tail = str(cell).partition(" ")
    return rf"\{cmd}{{{head}}}" + (sep + tail if sep else "")


_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _leading_float(cell) -> float:
    """The first number in a cell, or NaN. ``0.309 \\ [0.273, 0.343]`` -> 0.309."""
    match = _NUMBER.search(str(cell))
    return float(match.group()) if match else float("nan")


def _embolden(cell: str) -> str:
    r"""Bold a cell without bolding its interval.

    Only the point estimate is emphasised: bolding ``[0.273, 0.343]`` as well makes the cell twice
    as heavy as its neighbours for no extra meaning, and the interval is not what won.
    """
    text = str(cell)
    match = _NUMBER.search(text)
    if not match:
        return text
    a, b = match.span()
    return f"{text[:a]}\\textbf{{{text[a:b]}}}{text[b:]}"


# ── One shared drawing: shares of a whole as stacked horizontal bars ──────────

def stacked_share_bars(ax, bars, segments, *, min_label_share=0.04, fontsize=7.5):
    """Draw each bar as its segments' shares of 100%, left to right in ``segments`` order.

    ``bars`` is ``[(bar_label, {segment_key: count})]``, top to bottom; ``segments`` is
    ``[(segment_key, label, style)]`` where ``style`` carries ``color`` and ``hatch``. Every segment
    is labelled inside with its percentage when it is wide enough to hold one, so the figure is read
    without a lookup, and hatch + colour keep it legible in greyscale. Returns legend handles.
    """
    from matplotlib.patches import Patch  # noqa: PLC0415

    y_positions = list(range(len(bars)))[::-1]
    for y, (_, counts) in zip(y_positions, bars):
        total = sum(counts.get(k, 0) for k, _, _ in segments)
        left = 0.0
        for key, _, st in segments:
            share = counts.get(key, 0) / total * 100 if total else 0.0
            if share <= 0:
                continue
            ax.barh(y, share, left=left, height=0.62, color=st["color"], hatch=st.get("hatch", ""),
                    edgecolor="white", linewidth=0.6)
            if share / 100 >= min_label_share:
                ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                        fontsize=fontsize, color=P.text_on(st["color"]),
                        bbox=dict(boxstyle="round,pad=0.12", fc=st["color"], ec="none"))
            left += share
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in bars])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of false negatives (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return [Patch(facecolor=st["color"], hatch=st.get("hatch", ""), edgecolor="white",
                  label=label) for _, label, st in segments]


# ── Writing the deliverables ─────────────────────────────────────────────────

def start(width: float = WIDTH_FULL, height: float = 2.6, **kw):
    """A figure at an exact width in inches. The width IS the deliverable -- never rescale it."""
    style.apply()
    return plt.subplots(figsize=(width, height), **kw)


def save(fig, name: str) -> Path:
    """Save a figure as ``<name>.pdf`` (and a preview PNG when ``THESIS_FIG_PREVIEW`` is set)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    # Opt-in raster twin for reviewing a build without a PDF viewer. Never written into the thesis.
    preview = __import__("os").environ.get("THESIS_FIG_PREVIEW")
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview) / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  figure -> {os.path.relpath(out, REPO)}  ({out.stat().st_size / 1024:.0f} KB)")
    return out


_HEADER = (
    "%% GENERATED FILE -- do not edit by hand.\n"
    "%% Regenerate with: python figures/thesis_figures_scripts/{script}\n"
    "%% Source: {source}\n"
)

#: Where a chapter-4 artifact's numbers come from, unless it names another experiment.
DEFAULT_SOURCE = "output/treephenorag_protocol/tables/"

#: "this artifact has no source yet", matching ch5_common/ch6_common so all three drivers
#: treat a skip identically. A genuine crash still exits 1 and still fails the build.
MISSING_SOURCE_EXIT = 3

_FIGURE_NOTE = r"""%% Requires in the preamble (neurips_2025.sty does NOT load graphicx):
%%     \usepackage{graphicx}
%%     \graphicspath{{thesis_figures/}}
%% One entry: every generated figure lives there and is referenced by bare filename.
%% The PDF is drawn at exactly \textwidth = 5.5in, so width=\textwidth does not rescale it and
%% its text renders at the same 9pt/8pt as every other figure. Never use [scale=].
"""


def write_figure_tex(name: str, caption: str, label: str, script: str,
                     placement: str = "tb", source: str = DEFAULT_SOURCE,
                     fraction: float = 1.0, note: str = "") -> Path:
    """``fraction`` < 1 for a figure drawn at ``fraction * WIDTH_FULL``: the include width then
    matches the drawn width, so the text still renders at its design size. The caption is
    unbolded (``style.plain_caption``) and *note* is not printed."""
    width = r"\textwidth" if fraction == 1.0 else f"{fraction:g}\\textwidth"
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    out = TEX_DIR / f"{name}.tex"
    body = (
        _HEADER.format(script=script, source=source) + _FIGURE_NOTE
        + f"\\begin{{figure}}[{placement}]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={width}]{{{name}.pdf}}\n"
        f"  \\caption{{{style.plain_caption(caption)}}}\n"
        f"  \\label{{fig:{label}}}\n"
        "\\end{figure}\n"
    )
    out.write_text(body, encoding="utf-8")
    print(f"  latex  -> {os.path.relpath(out, REPO)}")
    return out


def labels_tex(label: "str | list[str]") -> str:
    r"""``\label{tab:a}`` -- or several, for a float the prose knows by more than one name.

    A float the thesis has renamed keeps its old label as an alias until the last ``\Cref`` to it
    is gone, so the rename cannot leave a "??" anywhere in the PDF. Each name is given without the
    ``tab:`` prefix.
    """
    names = [label] if isinstance(label, str) else list(label)
    return "".join(f"\\label{{tab:{n}}}" for n in names)


def write_table_tex(name: str, caption: str, label: "str | list[str]", script: str,
                    column_spec: str, header: "list[str] | None", rows: list,
                    notes: str = "", placement: str = "tb", rules: list[int] | None = None,
                    compact: bool = False, size: str = r"\small",
                    tabcolsep: "str | None" = None, source: str = DEFAULT_SOURCE) -> Path:
    """A standalone ``\\begin{table}`` file. ``rules`` gives row indices to precede with a midrule.

    booktabs only -- no ``\\hline``. That is the thesis template's convention and also simply the
    right way to set a table.

    A row is either a list of cells or a **string**, emitted verbatim: ``"\\midrule"`` draws a
    rule, anything else (a ``\\multicolumn`` block header, a header row inside the body) is written
    as one line. A string row also closes a marking block in ``bold_best``/``mark_best_second``, so
    a table with two blocks gets one winner per block. ``header=None`` omits the header row, for a
    table whose headers sit inside the body.

    ``compact`` halves the inter-column padding for a table that would otherwise run past the
    5.5 in text width (``tabcolsep`` sets it explicitly instead). It is an opt-in, not the
    default: at five or more columns the default padding alone is 60 pt or more, so this recovers
    real width without touching a number, but applying it everywhere would make most tables
    needlessly cramped. ``size`` is the type size of the body. Both are scoped inside the ``table``
    environment and leak nothing.
    """
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    out = TEX_DIR / f"{name}.tex"
    rules = set(rules or [])
    padding = tabcolsep or ("3.5pt" if compact else None)
    lines = [
        _HEADER.format(script=script, source=source),
        "%% Requires \\usepackage{booktabs} in the preamble.",
        f"\\begin{{table}}[{placement}]",
        "  \\centering",
        f"  \\caption{{{style.plain_caption(caption)}}}",
        f"  {labels_tex(label)}",
        f"  {size}",
        *([f"  \\setlength{{\\tabcolsep}}{{{padding}}}"] if padding else []),
        f"  \\begin{{tabular}}{{{column_spec}}}",
        "    \\toprule",
    ]
    if header is not None:
        lines += ["    " + " & ".join(header) + r" \\", "    \\midrule"]
    for i, row in enumerate(rows):
        if i in rules:
            lines.append("    \\midrule")
        if row == r"\midrule":
            lines.append("    \\midrule")
        elif isinstance(row, str):
            lines.append("    " + row + r" \\")
        else:
            lines.append("    " + " & ".join(row) + r" \\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    lines += style.note_tex(notes)
    lines += [r"\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  latex  -> {os.path.relpath(out, REPO)}")
    return out


def tex_escape(text: str) -> str:
    """Escape the characters that bite in a results label (``_`` in every pooling name)."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$")):
        text = text.replace(a, b)
    return text
