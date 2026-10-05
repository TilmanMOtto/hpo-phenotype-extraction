"""Shared loading, ordering and palette for the PhenoJury protocol (PhenoJury) thesis figures.

Every figure in this folder reads the tables `phenojury_protocol` wrote and nothing
else. Nothing is recomputed here: if a number is in a figure it came out of a CSV, so a figure and
the chapter text cannot drift apart. An exception is arithmetic that is *presentational*, sorting, or a difference between two columns already present, and those are marked where they
happen.

House style comes from `experiments/figures/style.py`, the module `fig1`–`fig6` already use, so these
figures share its geometry (5.5 in = \\textwidth, 9 pt/8 pt type, pure-vector PDF) and its
colour-blind-safe Okabe–Ito palette. Do not set a width in LaTeX. The figsize IS the width.

HCY is the primary cohort throughout and GSC+ is the second panel, the layout is the argument,
since GSC+ is a transfer check on a decision taken on HCY, not an equal partner.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# experiments/figures/style.py, the same module fig1..fig6 use.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))  # style.py
import style  # noqa: E402
from hpo_extraction.paths import results_dir, thesis_dir  # noqa: E402

DEFAULT_RESULTS = os.path.join(str(results_dir()), "phenojury_protocol")


def result_file(relative: str) -> str:
    """*relative* resolved for reading: ``output/...`` is the results folder (``results_dir`` of the path
    file), anything else is relative to the repository."""
    prefix = "output/"
    if relative.startswith(prefix):
        return os.path.join(str(results_dir()), relative[len(prefix):])
    return os.path.join(_REPO, relative)
DEFAULT_FIGDIR = os.path.join(str(thesis_dir()), "thesis_figures")
DEFAULT_TEXDIR = os.path.join(str(thesis_dir()), "thesis_figures_latex")

# `gsc206` is the PhenoJury protocol's output directory for AutoPCR's 206-abstract evaluation split
# (cohort `gsc_2024_eval_206`), scored from the 228-document `gsc` cache with its own ground truth and folds.
# The full 228-document corpus is not a reported frame: GSC+ appears only in the two frames a
# published baseline can be read against, and chapter 5 uses the one with the corpus's own ground truth.
COHORTS = ("hcy", "gsc206")
COHORT_LABEL = {"hcy": "HCY (clinical reports)", "gsc206": "GSC+ (206 abstracts)"}
COHORT_SHORT = {"hcy": "HCY", "gsc206": "GSC+"}

# ── fixed orders ────────────────────────────────────────────────────────────────────────────
# Assigned once and never cycled: a reader who learns a colour in one figure keeps it in every
# other. Order is the experiment's own (p0 is the control. Q-prompts in index order).

PROMPT_ORDER = ["p0_baseline", "q2_sentence_last", "q4_span_json", "q7_recall"]
# The thesis's names for the four prompts. One name per prompt in every float and in the prose.
PROMPT_LABEL = {
    "p0_baseline": "Free Listing",
    "q2_sentence_last": "HPO-Guided",
    "q4_span_json": "Evidence Spans",
    "q7_recall": "Recall-First",
}

# phenobert_raw is the REFERENCE reader, it reads the whole generation while the other three read
# parsed candidates, so it is excluded from selection. It is drawn in grey everywhere to keep that
# visible: it is context, never a competitor.
NORMALISER_ORDER = ["phenobert_candidates", "dictionary", "sapbert", "phenobert_raw"]
NORMALISER_LABEL = {
    "phenobert_candidates": "PhenoBERT (candidates)",
    "dictionary": "Dictionary",
    "sapbert": "SapBERT",
    "phenobert_raw": "PhenoBERT (raw)*",
}
REFERENCE_NORMALISER = "phenobert_raw"
# The names a thesis table prints. The candidate reader is simply "PhenoBERT": the raw reader is a
# reference that is never selected, so no printed row needs the two told apart.
NORMALISER_SHORT = {"phenobert_candidates": "PhenoBERT", "dictionary": "Dictionary",
                    "sapbert": "SapBERT", "phenobert_raw": "PhenoBERT (raw)"}

# The vote SCOPE (what counts as "the same finding" when two jurors are compared) and the matching
# RULE. Both are selection axes, so both need a fixed encoding: a reader who learns that segment is
# green in one figure keeps it in the next.
SCOPE_ORDER = ["report", "window", "segment"]
SCOPE_LABEL = {"report": "report", "window": "window ($\\pm$1 sentence)", "segment": "segment"}

RULE_ORDER = ["exact", "closure_reduced"]
RULE_LABEL = {"exact": "EXACT", "closure_reduced": "CLOSURE"}

MODEL_ORDER = ["apertus", "deepseek", "intelligent_internet", "llama",
               "medgemma", "medpsy", "openbiollm", "phi4"]
MODEL_LABEL = {
    "apertus": "Apertus", "deepseek": "DeepSeek", "intelligent_internet": "Intelligent-Internet",
    "llama": "Llama", "medgemma": "MedGemma", "medpsy": "MedPsy", "openbiollm": "OpenBioLLM",
    "phi4": "Phi-4",
}

# Colours come from palette.py, the thesis-wide system. Chapter 5 is PhenoJury's, so its variants
# are steps of PhenoJury's green, and a marker repeats the identity. A control or reference condition is
# The baseline grey, never a competitor hue.
_H = os.path.dirname(os.path.abspath(__file__))
if _H not in sys.path:
    sys.path.insert(0, _H)
import palette as P  # noqa: E402

GREY = P.MID
GREY_LIGHT = P.LIGHT
INK = P.INK

_JURY_STEPS = (P.JURY_LIGHT, P.JURY, P.shade(P.JURY, 0.35))
# p0 is the control prompt, so it is grey. The three q-prompts take the steps in index order.
PROMPT_COLOR = {"p0_baseline": P.BASELINE, "q2_sentence_last": _JURY_STEPS[0],
                "q4_span_json": _JURY_STEPS[1], "q7_recall": _JURY_STEPS[2]}
NORMALISER_COLOR = {
    "phenobert_candidates": _JURY_STEPS[2],
    "dictionary": _JURY_STEPS[1],
    "sapbert": _JURY_STEPS[0],
    REFERENCE_NORMALISER: P.BASELINE,   # reference reader: grey, never a competitor hue
}
# Eight jurors cannot be eight steps of one hue and must not borrow the other methods' hues, so a
# juror is identified by its label, never by colour.
MODEL_COLOR = {m: P.JURY for m in MODEL_ORDER}
# The scope axis is ORDERED -- report is coarsest, segment finest -- so it takes the palette's
# three-step ordered scale (green -> teal -> blue): one progression, three lines far enough apart
# to tell apart at a glance.
SCOPE_COLOR = dict(zip(["report", "window", "segment"], P.ORDERED_3))
SCOPE_MARKER = {"report": "o", "window": "s", "segment": "^"}

# Marker shapes repeat the identity a colour carries, so the figures survive greyscale printing.
PROMPT_MARKER = {"p0_baseline": "o", "q2_sentence_last": "s",
                 "q4_span_json": "^", "q7_recall": "D"}
NORMALISER_MARKER = {"phenobert_candidates": "o", "dictionary": "s",
                     "sapbert": "^", REFERENCE_NORMALISER: "x"}

# ── loading ─────────────────────────────────────────────────────────────────────────────────


#: Exit code for "this artifact's source does not exist yet".
#:
#: Distinguished from 1 on purpose. Several specified artifacts wait on an experiment stage that
#: has not run (see docs/thesis_map.md), and that is an ordinary state of the world rather than a
#: build failure, but a script that just returned 0 would be indistinguishable from one that
#: produced something. So the script exits 3, `make_all_ch5.py` reports it as *skipped*, and a
#: genuine crash still exits 1 and still fails the build.
MISSING_SOURCE_EXIT = 3


def table(results_dir: str, cohort: str, name: str) -> pd.DataFrame:
    """One table for one cohort. Exits 3 if it is missing, a figure built from a silently-empty
    frame is worse than no figure, and a missing source is not a crash."""
    path = os.path.join(results_dir, cohort, "tables", f"{name}.csv")
    if not os.path.exists(path):
        print(f"  skipped: missing table {path}\n"
              f"    run experiments/05_phenojury/protocol/run.py for cohort={cohort}, or pass --results "
              f"(see docs/thesis_map.md)", file=sys.stderr)
        raise SystemExit(MISSING_SOURCE_EXIT)
    df = pd.read_csv(path)
    if df.empty:
        print(f"  skipped: empty table {path}", file=sys.stderr)
        raise SystemExit(MISSING_SOURCE_EXIT)
    return df


def manifest(results_dir: str, cohort: str) -> dict:
    """The PhenoJury protocol's ``manifest.json`` of one cohort."""
    import json
    path = os.path.join(results_dir, cohort, "manifest.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reported_pair(man: dict, arm: str = "phenojury") -> tuple[str, str]:
    """The (prompt, normaliser) *condition*'s out-of-fold rows were scored at, from a manifest.

    Since the PhenoJury protocol selects the pair inside each outer training split, that is the pair the folds
    chose most often (`in_fold_modal_pair`), not `selected_*` -- which is only the dev-split pair
    the descriptive stages hold fixed. An older manifest has no in-fold entry, and there the dev
    pair WAS what every row was scored at, so it is the right fallback.
    """
    modal = (man.get("in_fold_modal_pair") or {}).get(arm)
    if modal:
        return modal[0], modal[1]
    return man.get("selected_prompt", ""), man.get("selected_normaliser", "")


def has_cohort(results_dir: str, cohort: str) -> bool:
    """True when the PhenoJury protocol wrote tables for *cohort*."""
    return os.path.isdir(os.path.join(results_dir, cohort, "tables"))


def order_categorical(df: pd.DataFrame, column: str, order: list) -> pd.DataFrame:
    """Sort by a fixed roster, keeping anything unexpected at the end, not dropping it."""
    known = [v for v in order if v in set(df[column])]
    extra = [v for v in df[column].unique() if v not in set(order)]
    cats = known + sorted(extra)
    out = df.copy()
    out[column] = pd.Categorical(out[column], categories=cats, ordered=True)
    return out.sort_values(column)


# ── argument plumbing ───────────────────────────────────────────────────────────────────────


def add_common_args(ap):
    """Add ``--results``, ``--figdir`` and ``--texdir`` to an argument parser and return it."""
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="output/phenojury_protocol (default: repo copy)")
    ap.add_argument("--figdir", default=DEFAULT_FIGDIR, help="where the PDF goes")
    ap.add_argument("--texdir", default=DEFAULT_TEXDIR, help="where the \\includegraphics snippet goes")
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


# ── LaTeX table helpers ─────────────────────────────────────────────────────────────────────

def tex_escape(text) -> str:
    """Escape a cell for LaTeX. Model and prompt keys carry _ character characters, which are maths-mode
    subscripts if left alone and silently mangle every label in the table."""
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


# `bold_best` lives in chapter 4's `common` because that is where the first table needed it. It
# is chapter-agnostic -- it knows only about rows, columns and blocks -- so it is imported rather
# than copied. A second implementation would be a second set of rules about what a block is.
from common import bold_best, mark_best_second  # noqa: E402,F401


def write_table_tex(path, *, caption, label, header, rows, colspec, notes=None,
                    generator="figures/thesis_figures_scripts/", placement="tb",
                    compact=False, preamble=None):
    r"""Write a standalone `table` environment (see :func:`render_table_tex`)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_table_tex(caption=caption, label=label, header=header, rows=rows,
                                  colspec=colspec, notes=notes, generator=generator,
                                  placement=placement, compact=compact, preamble=preamble))
    print(f"[tables] {os.path.basename(path)}")


def render_table_tex(*, caption, label, header, rows, colspec, notes=None,
                     generator="figures/thesis_figures_scripts/", placement="tb",
                     compact=False, preamble=None, file_header=True) -> str:
    r"""A standalone `table` environment as a string. Booktabs. No vertical rules.

    ``label`` may be empty, for the continuation of a float split over pages -- only the first
    part carries the label the prose cites. ``file_header=False`` drops the three comment lines,
    for a second environment appended to a file that already has them.

    ``compact`` shrinks the body to ``\small`` and halves the inter-column padding, for a table
    that would otherwise run past the 5.5 in text width. It is a deliberate opt-in, not the
    default: at seven columns the default padding alone is 96 pt, so this recovers about 1.1 in
    without touching a single number, but applying it everywhere would make most tables needlessly
    cramped. Both changes are scoped inside the ``table`` environment and leak nothing.

    A row may be:

    * a **list** of cells, joined with ``&`` and terminated with ``\\``;
    * the sentinel ``"\midrule"``, drawn as a rule between groups, or a ``"\cmidrule..."``
      string, a thinner rule inside a group -- both emitted without a row terminator;
    * any other **string**, emitted verbatim (plus ``\\``). This is what group headers use:
      a ``\multicolumn{n}{l}{...}`` already consumes *n* columns, so padding it out to the
      column count as a list would emit n extra ``&`` and LaTeX would abort with "extra
      alignment tab". Passing it as a string is the only correct way to write one.
    """
    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        f"%% Regenerate with: {generator}",
        r"% Requires \usepackage{booktabs} in the preamble.",
    ] if file_header else []
    lines += [
        r"\begin{table}[%s]" % placement,
        r"  \centering",
        r"  \caption{%s}" % style.plain_caption(caption),
    ]
    # A list of labels: a float the prose knows by more than one name (see common.labels_tex).
    labels = [label] if isinstance(label, str) else list(label or ())
    if any(labels):
        lines.append("  " + "".join(r"\label{%s}" % name for name in labels if name))
    if compact:
        lines += [r"  \small", r"  \setlength{\tabcolsep}{3.5pt}"]
    # Verbatim lines between \label and the tabular, for a table whose type size and padding are
    # set to match hand-written neighbours in the same chapter (e.g. ``\small`` + 4pt).
    lines += ["  " + line for line in (preamble or [])]
    # A header is a list of cells, or a string written verbatim -- for a two-line header whose
    # first line is a row of \multicolumn spans (the side-by-side cohort blocks of a table).
    head = header if isinstance(header, str) else " & ".join(header)
    lines += [
        r"  \begin{tabular}{%s}" % colspec,
        r"    \toprule",
        "    " + head + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        if row == r"\midrule" or (isinstance(row, str) and row.startswith(r"\cmidrule")):
            lines.append("    " + row)
            continue
        if isinstance(row, str):
            lines.append("    " + row + r" \\")
            continue
        lines.append("    " + " & ".join(str(c) for c in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    lines += style.note_tex(notes)
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def fmt(x, nd=2):
    """Fixed-width numeric formatting; ``--`` for missing so a column stays readable.

    Two decimals by default: the report-level intervals on these cohorts are about +/-0.03 wide,
    so a third decimal on P/R/F1 would be noise."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if v != v:
        return "--"
    return f"{v:.{nd}f}"
