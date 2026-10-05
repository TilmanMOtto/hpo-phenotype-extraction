"""Shared loading, ordering and palette for the comparison (final comparison) discussion figures.

Every figure and table in the ch6 family reads `output/comparison/tables` and
nothing else, so a figure can never disagree with the table beside it or with the chapter text.
Nothing is recomputed here. The only arithmetic is presentational (sorting, or a difference between
two columns that are both already in the CSV), and it is marked where it happens.

House style comes from `experiments/figures/style.py`, and the three generic LaTeX helpers are imported
from `ch5_common` rather than copied, a third implementation of `write_table_tex` would be a third
thing to keep in step.

## Two encodings that carry an argument, not a preference

**Estimator is a marker shape, not a footnote.** Three different things are being compared: methods
with nothing to tune (a plain cohort estimate), methods whose configuration is chosen inside
nested CV (pooled out-of-fold), and TreePhenoRAG on GSC+, where the configuration selected on all of
HCY is applied unchanged and *nothing* is fitted on the cohort at all. The third is the strictest
and reads as the weakest if it is not marked, so it is marked in every figure that shows a point.

**Our two systems are drawn last and in the accent hues.** The roster order is fixed, external
baselines, then ours, so a reader who learns a colour in one figure keeps it in the next.

## Missing cells

TreePhenoRAG has no GSC+ row until `treephenorag_protocol` runs (it is queued behind the `treephenorag_scores_terminfo`
cache array). A missing cell is drawn as an explicit em dash at the axis, never omitted: a silently
shorter panel would read as "not applicable" when the truth is "not yet computed".
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))  # style.py
import style  # noqa: E402

# Generic LaTeX plumbing, reused, not re-implemented. These three carry no ch5 semantics.
from ch5_common import fmt, render_table_tex, tex_escape, write_table_tex  # noqa: E402,F401

# `bold_best` lives in chapter 4's `common` because that is where the first table needed it. It is
# chapter-agnostic -- it knows only about rows, columns and blocks -- so it is imported, not
# copied. A second implementation would be a second set of rules about what counts as a block.
from common import bold_best  # noqa: E402,F401
from hpo_extraction.paths import results_dir, thesis_dir  # noqa: E402

# Geometry is the style module's, never re-chosen: the figsize IS the LaTeX width, and a
# \includegraphics that rescales it would change the type size on the page.
WIDTH_FULL = style.WIDTH_FULL
WIDTH_HALF = style.WIDTH_HALF

DEFAULT_RESULTS = os.path.join(str(results_dir()), "comparison")
DEFAULT_FIGDIR = os.path.join(str(thesis_dir()), "thesis_figures")
DEFAULT_TEXDIR = os.path.join(str(thesis_dir()), "thesis_figures_latex")

COHORTS = ("hcy", "gsc_raghpo_ann", "gsc_2024_eval_206")
# Panel titles are kept short enough to fit a half-width panel at 9pt: the ground truth standard and the
# document filter are stated in every caption, so repeating them here only clips the title.
COHORT_LABEL = {
    "hcy": "HCY (118 clinical reports)",
    "gsc_raghpo_ann": "GSC+ (114 abstracts)",
    "gsc_2024_eval_206": "GSC+ (206 abstracts)",
}
#: Table block headers. The two GSC+ frames differ in who filtered the corpus (and, for RAG-HPO's,
#: who re-annotated it), so a table says so where the block starts. The figures' panel titles keep
#: The short form above, and every figure caption names the frame.
COHORT_HEADER = {
    "hcy": "HCY (118 clinical reports)",
    "gsc_raghpo_ann": "GSC+ (114 abstracts, filtered and re-annotated by RAG-HPO)",
    "gsc_2024_eval_206": "GSC+ (206 abstracts, filtered by AutoPCR)",
}
# The two GSC+ views must never both be called "GSC+" in the same artifact: they are different
# document sets under different ground truth standards, and a reader who conflates them will read the
# 114-document row against AutoPCR's published number, which is the one mistake this cohort was
# added to prevent.
COHORT_SHORT = {"hcy": "HCY", "gsc_raghpo_ann": "GSC+ (114)",
                "gsc_2024_eval_206": "GSC+ (206)"}

# ── the roster, in one fixed order ──────────────────────────────────────────────────────────
# External baselines first in the order they enter the thesis, ours last. Never sorted by score:
# a bar chart whose order changes between cohorts makes the two panels incomparable at a glance.
METHOD_ORDER = ["raghpo_8b", "raghpo_70b", "autopcr_8b", "autopcr_70b",
                "phenobert", "treephenorag", "phenojury", "phenojury_core3"]

METHOD_LABEL = {
    "raghpo_8b": "RAG-HPO 8B",
    "raghpo_70b": "RAG-HPO 70B",
    "autopcr_8b": "AutoPCR 8B",
    "autopcr_70b": "AutoPCR 70B",
    "phenobert": "PhenoBERT",
    "treephenorag": "TreePhenoRAG",
    "phenojury": "PhenoJury (8)",
    "phenojury_core3": "PhenoJury (core-3)",
}

#: LaTeX forms. Plain text, as the chapters set every system name (no small caps).
METHOD_LABEL_TEX = {
    "raghpo_8b": "RAG-HPO 8B",
    "raghpo_70b": "RAG-HPO 70B",
    "autopcr_8b": "AutoPCR 8B",
    "autopcr_70b": "AutoPCR 70B",
    "phenobert": "PhenoBERT",
    "treephenorag": "TreePhenoRAG",
    "phenojury": "PhenoJury (full pool, 8)",
    "phenojury_core3": "PhenoJury (core-3)",
}

# Colours come from palette.py, the thesis-wide system. This chapter's method assignment IS the
# system's rule 1, so every other chapter draws a method in the hue it has here.
import palette as P  # noqa: E402

GREY = P.MID
GREY_LIGHT = P.LIGHT
INK = P.INK

# Baselines in muted hues, ours in the two strongest. The 8B/70B pairs share a hue and differ in
# marker, because the pair IS the comparison, same pipeline, one model apart.
METHOD_COLOR = {
    "raghpo_8b": P.RAGHPO_LIGHT, "raghpo_70b": P.RAGHPO,
    "autopcr_8b": P.AUTOPCR_LIGHT, "autopcr_70b": P.AUTOPCR,
    "phenobert": P.PHENOBERT,
    "treephenorag": P.TREE,
    # The two PhenoJury rows share a hue and differ in lightness, for the same reason the 8B/70B
    # pairs share one: the pair IS a comparison -- one jury, five members removed -- and a second
    # hue would read as a second method.
    "phenojury": P.JURY, "phenojury_core3": P.JURY_LIGHT,
}
METHOD_MARKER = {
    "raghpo_8b": "v", "raghpo_70b": "^",
    "autopcr_8b": "s", "autopcr_70b": "D",
    "phenobert": "o",
    "treephenorag": "P",
    "phenojury": "*", "phenojury_core3": "X",
}

# ── estimators ──────────────────────────────────────────────────────────────────────────────
# Matched by prefix against the comparison's own `estimator` column, so the strings live in one place
# (roster.py) and this is only a shortening.
ESTIMATOR_SHORT = {
    "cohort": "fixed",
    "pooled": "out-of-fold",
    "transfer": "transfer",
}
ESTIMATOR_ORDER = ["fixed", "out-of-fold", "transfer"]
ESTIMATOR_MARKER = {"fixed": "o", "out-of-fold": "s", "transfer": "D"}
ESTIMATOR_TEX = {
    "fixed": "single operating point, nothing tuned",
    "out-of-fold": "pooled out-of-fold (nested CV)",
    "transfer": "HCY-selected configuration, nothing fitted on this cohort",
}


def estimator_short(value) -> str:
    """`fixed` | `out-of-fold` | `transfer` from the comparison's long estimator string."""
    head = str(value or "").split(" ", 1)[0].strip().lower()
    return ESTIMATOR_SHORT.get(head, head or "")


#: What a missing cell looks like. Spelled differently in the two media on purpose: LaTeX needs
#: three hyphens for an em dash, matplotlib needs the character.
MISSING_TEX = "---"
MISSING_TXT = "—"

#: Why the one missing cell is missing. Printed in every caption that has a gap, so a reader never
#: has to guess whether it is "not applicable" or "not yet run".
#: Both experiment ids need their _ character escaped -- this string lands in a LaTeX caption, where
#: a bare `_` is a subscript outside maths and a hard error inside text. It went unnoticed because
#: only one of the two was escaped, and TeX's recovery from the first one swallowed the rest.
#:
#: It is appended UNCONDITIONALLY by every caption below, so it has to describe the gap that is
#: actually still there. It used to say the GSC+ transfer was "queued behind the earlier runs\_24 score-cache
#: array". That landed on 2026-09-17 and TreePhenoRAG now has a GSC+ (114) row, at which point the
#: sentence was asserting the absence of a number printed two lines above it. The remaining absence
#: is a different statement and the weaker one: "not applicable", not "not yet run".
MISSING_NOTE = (r"TreePhenoRAG is not run on the 206-abstract frame.")


# ── loading ─────────────────────────────────────────────────────────────────────────────────

#: Exit code for "this artifact's source does not exist yet", see
#: :data:`ch5_common.MISSING_SOURCE_EXIT`, which this matches so the two drivers can
#: treat a skip identically. A genuine crash still exits 1 and still fails the build.
MISSING_SOURCE_EXIT = 3


#: Where each system's model details are DECLARED. No artifact records a model name or its
#: precision, so the comparison's config is the source of truth and `t4_cost.csv`'s `checkpoint` column
#: only echoes it as of the last cluster run. Reading the config directly means a corrected label
#: reaches the thesis without a rerun whose only output change would be that string.
COMPARISON_CONFIG = os.path.join(_REPO, "configs", "experiments", "06_comparison", "comparison.yaml")


def declared_model_versions(path: str = COMPARISON_CONFIG) -> dict[str, str]:
    """``{method: checkpoint label}`` from the config's ``cost`` block; ``{}`` if unreadable."""
    import yaml
    try:
        with open(path, encoding="utf-8") as fh:
            cost = (yaml.safe_load(fh) or {}).get("cost") or {}
    except OSError:
        return {}
    return {m: str(v["checkpoint"]) for m, v in cost.items()
            if isinstance(v, dict) and v.get("checkpoint")}


def table(results_dir: str, name: str) -> pd.DataFrame:
    """One table by name. Exits 3 if absent, a figure built from a silently empty frame is worse
    than no figure, and a missing source is not a crash."""
    path = os.path.join(results_dir, "tables", f"{name}.csv")
    if not os.path.exists(path):
        print(f"  skipped: missing table {path}\n"
              f"    run experiments/06_comparison/comparison/run.py, or pass --results "
              f"(see docs/thesis_map.md)", file=sys.stderr)
        raise SystemExit(MISSING_SOURCE_EXIT)
    df = pd.read_csv(path)
    if df.empty:
        print(f"  skipped: empty table {path}", file=sys.stderr)
        raise SystemExit(MISSING_SOURCE_EXIT)
    return df


def manifest(results_dir: str) -> dict:
    """The comparison's ``manifest.json``."""
    import json
    with open(os.path.join(results_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def by_method(df: pd.DataFrame, cohort: str) -> dict:
    """``{method_key: row}`` for one cohort, scored rows only.

    A row whose `status` is not `ok` carries no numbers, it is the named absence the comparison writes
    instead of dropping the method, so it is excluded here and the caller draws the gap.
    """
    sub = df[df["cohort"] == cohort] if "cohort" in df.columns else df
    out = {}
    for _, row in sub.iterrows():
        if str(row.get("status", "ok")) != "ok":
            continue
        out[row["method"]] = row
    return out


def methods_present(df: pd.DataFrame) -> list:
    """The roster order, restricted to methods this table knows about (unknown keys last)."""
    seen = set(df["method"])
    known = [m for m in METHOD_ORDER if m in seen]
    return known + sorted(seen - set(METHOD_ORDER))


def cohorts_present(df: pd.DataFrame) -> list:
    """Cohorts that occur in a comparison table, in the thesis order."""
    if "cohort" not in df.columns:
        return []
    seen = set(df["cohort"])
    return [c for c in COHORTS if c in seen] + sorted(seen - set(COHORTS))


# ── argument plumbing ───────────────────────────────────────────────────────────────────────

def add_common_args(ap):
    """Add ``--results``, ``--figdir`` and ``--texdir`` to an argument parser and return it."""
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="output/comparison (default: repo copy)")
    ap.add_argument("--figdir", default=DEFAULT_FIGDIR, help="where the PDF goes")
    ap.add_argument("--texdir", default=DEFAULT_TEXDIR,
                    help="where the \\includegraphics snippet goes")
    ap.add_argument("--preview", action="store_true", help="also write a PNG for quick review")
    return ap


def prepare(args):
    """Create the output folders named in *args*."""
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.texdir, exist_ok=True)
    style.apply()


def emit(fig, args, figname, texname, caption, label, generator, note=None):
    """Write the PDF into figdir and the \\begin{figure} snippet into texdir."""
    pdf = os.path.join(args.figdir, figname)
    png = pdf.replace(".pdf", ".png") if args.preview else None
    style.save(fig, pdf, preview_png=png)
    style.write_tex(os.path.join(args.texdir, texname), figname, caption, label,
                    generator=generator, note=note)


def legend_below(fig, handles, ncol=4, fontsize=6.8):
    """A figure legend under the panels, placed so constrained layout reserves room for it.

    `style.py` turns on `figure.constrained_layout.use` globally, which **overrides both
    `tight_layout` and `subplots_adjust`**, anything this module set by hand was silently
    discarded, which is what clipped the tick labels and dropped the legend on top of them. The
    supported way to add a figure-level legend under a constrained layout is the `"outside ..."`
    location family, which the engine treats as an element to allocate space for, not as an
    overlay. So: never re-adjust the margins here. State where the legend goes and let the engine
    do the arithmetic, which it does correctly for rotated tick labels too.
    """
    return fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False,
                      fontsize=fontsize, handletextpad=0.4, columnspacing=1.1)


def spread(values, min_gap, bounds=None):
    """Nudge *values* apart to at least *min_gap*, preserving order and staying near the originals.

    For the label at the end of each slope in the hierarchy figure: where two systems converge the
    annotations overlap into an unreadable stack, and moving them is honest as long as the marker
    they belong to does not move. One upward pass then one downward pass is enough for the handful
    of series here and cannot oscillate.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] - out[lo] < min_gap:
            out[hi] = out[lo] + min_gap
    for k in range(len(order) - 2, -1, -1):
        lo, hi = order[k], order[k + 1]
        if out[hi] - out[lo] < min_gap:
            out[lo] = out[hi] - min_gap
    if bounds:
        # Spreading can push the stack past the axis. Slide it back as a block so the ORDER and
        # The spacing both survive. Clamping each value individually would re-collide them.
        low, high = bounds
        if max(out) > high:
            out = [v - (max(out) - high) for v in out]
        if min(out) < low:
            out = [v + (low - min(out)) for v in out]
    return out


def missing_marker(ax, x, y, label="not yet computed", **kw):
    """Draw the em dash that stands in for a cell with no number yet.

    The dash alone is ambiguous, at a glance it reads as a very tight interval, so it carries its
    own caption. A reader must never have to infer whether a gap means "not applicable" or "not
    finished".
    """
    ax.annotate(f"{MISSING_TXT} {label}" if label else MISSING_TXT, xy=(x, y),
                ha="center", va="center", color=GREY,
                fontsize=kw.pop("fontsize", 6.5), **kw)
