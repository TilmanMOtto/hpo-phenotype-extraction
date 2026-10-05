#!/usr/bin/env python
r"""The chapter-5 numbers the prose quotes inline, as LaTeX macros.

Same contract as num_ch4_inline.py: a number a paragraph asserts is as likely to go stale as a
number in a table and far harder to notice when it does, so every inline figure chapter 5 needs is
computed here and quoted as a macro.

    \input{thesis_figures_latex/num_ch5_inline}
    ... \chFiveJurors{} jurors, \chFiveReportsHcy{} reports ...

Two outputs: the macro file LaTeX inputs, and a Markdown twin for reading while writing the
paragraph.

One number crosses result directories on purpose. Wall clock per report is measured by the comparison's
cost collection rather than by the PhenoJury protocol, which is an analysis-only re-run and times nothing. Rather
than leave it a gap or re-measure it, it is read from there and its provenance line says so --
that is what provenance lines are for, and an inline number is the place where a
cross-chapter reference is normal.

A number whose source does not exist yet is NOT guessed. Its macro expands to a bold ``??``, which
is impossible to miss in a draft, and the reason is listed in both files and in docs/thesis_map.md.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C                       # noqa: E402

SCRIPT = "num_ch5_inline.py"
MISSING = r"\textbf{??}"
# Forward slashes on purpose: this string is joined onto the repo root (which works on every
# platform) and is also printed verbatim as a provenance line, where Windows separators would
# read as LaTeX escapes.
COST_TABLE = "output/comparison/tables/t4_cost.csv"


class Gap(Exception):
    """This number cannot be computed yet. The message says what would produce it."""


def optional(results, cohort, name):
    """A PhenoJury protocol table of one cohort, or None when it does not exist."""
    path = os.path.join(results, cohort, "tables", f"{name}.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    return None if df.empty else df


# -- one function per number --------------------------------------------------
# Each takes (results, cohort) and returns (rendered value, one-line provenance), or raises Gap.

def jurors(results, cohort):
    """Readers per cohort: every (model, prompt, normaliser) cell a jury can be drawn from.

    Not the manifest's ``n_jurors``, which also counts the raw PhenoBERT reader -- a reference that
    reads the whole generation and is never selected -- and so printed 128 against the 8 x 4 x 3 =
    96 readers the chapter describes.
    """
    inv = optional(results, cohort, "s0_cell_inventory")
    if inv is None:
        raise Gap("needs s0_cell_inventory.csv from exp14_04")
    cells = inv[(inv["normaliser"] != C.REFERENCE_NORMALISER) & inv["found"].astype(bool)]
    n = len(cells[["model", "prompt", "normaliser"]].drop_duplicates())
    return str(n), ("s0_cell_inventory.csv, distinct (model, prompt, normaliser) cells found, "
                    "reference reader excluded")


def reports(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    man = C.manifest(results, cohort)
    return str(int(man["n_reports"])), "manifest.json, n_reports"


def selected_prompt(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    prompt, _ = C.reported_pair(C.manifest(results, cohort), "phenojury")
    return (rf"\texttt{{{C.tex_escape(prompt)}}}",
            "manifest.json, in_fold_modal_pair.phenojury (else selected_prompt)")


def selected_normaliser(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    _, normaliser = C.reported_pair(C.manifest(results, cohort), "phenojury")
    return (rf"\texttt{{{C.tex_escape(normaliser)}}}",
            "manifest.json, in_fold_modal_pair.phenojury (else selected_normaliser)")


def selected_k(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    freq = optional(results, cohort, "s5_selection_frequency")
    if freq is None:
        raise Gap("needs s5_selection_frequency.csv")
    sub = freq[freq["axis"] == "k"].sort_values("n_folds", ascending=False)
    if sub.empty:
        raise Gap("s5_selection_frequency.csv has no k axis")
    row = sub.iloc[0]
    return (f"{int(row['value'])}", f"s5_selection_frequency.csv, modal k "
                                    f"({int(row['n_folds'])}/50 fold-rows)")


def _cost_row(cohort, column):
    """One cell of the comparison's cost table for PhenoJury, or None when it is empty."""
    path = C.result_file(COST_TABLE)
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    # The comparison names the GSC+ cohort by the frame it scores. Chapter 5's is AutoPCR's 206.
    want = "hcy" if cohort == "hcy" else "gsc_2024_eval_206"
    row = df[(df["method"] == "phenojury") & (df["cohort"] == want)]
    if row.empty:
        return None
    value = row.iloc[0][column]
    return None if value != value else (float(value), want)


# Generation calls and sentences per report are NOT quoted from here. The PhenoJury protocol's s0_call_budget
# counts the segments of the jurors' normalised payload, which has an entry only where a juror's
# output linked to a term -- 3.3 of the 6.9 sentences of a GSC+ abstract (check X25). The prose
# quotes the comparison's cost table, which counts the generations that were run.


def phenobert_proposed_share(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    grid = optional(results, cohort, "s2_gold_restricted")
    if grid is None:
        raise Gap("needs s2_gold_restricted.csv from exp14_04 (CPU, IF the curated annotation "
                  "sidecar records each gold pair's provenance -- verify on the cluster before "
                  "assuming it does)")
    row = grid.iloc[0]
    # What the PhenoJury protocol counts (run.py, the restricted ground truth): the pairs whose ONLY source is a curator
    # suggestion written with PhenoBERT's output on screen. No pair entered the ground truth on
    # PhenoBERT's proposal alone (appendix, curation), so "from a PhenoBERT proposal" was wrong.
    return (f"{float(row['share_gold_from_phenobert']) * 100:.1f}",
            "s2_gold_restricted.csv, share_gold_from_phenobert (pairs whose only source is a "
            "curator suggestion made with PhenoBERT on screen), as a percentage")


def wall_clock_s(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    got = _cost_row(cohort, "seconds_per_report")
    if got is None:
        raise Gap(f"seconds_per_report is EMPTY for phenojury in {COST_TABLE} -- the column "
                  "exists and was never filled, because exp14_04 is an analysis-only replay and "
                  "times nothing, and the exp13_06 generation jobs wrote no timing artifact. "
                  "Recoverable from SLURM job accounting for those jobs without a rerun. See "
                  "figures/GAPS.md")
    value, want = got
    return (f"{value:,.0f}".replace(",", r"\,"),
            f"{COST_TABLE}, phenojury/{want}, seconds_per_report (measured by exp13_25)")


def peak_mem_sequential(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    raise Gap("GPU-BLOCKED: peak memory under sequential serving was not recorded. peak_gpu_gb is "
              "empty for every row of t4_cost.csv; recoverable from SLURM job accounting for the "
              "exp13_06 generation jobs, otherwise needs a GPU run with memory instrumentation. "
              "See figures/GAPS.md")


def peak_mem_parallel(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    raise Gap("GPU-BLOCKED: peak memory under parallel serving was never measured -- the jurors "
              "were run sequentially, so this is a new GPU experiment rather than a missing "
              "column. See figures/GAPS.md")


def sapbert_index_build_s(results, cohort):
    """``(value, source)`` of one inline thesis number. Raises ``Gap`` when no source exists."""
    raise Gap("GPU-BLOCKED: the SapBERT index build was not timed. Needs exp14_03's index build "
              "to record wall clock. See figures/GAPS.md")


# macro stem, prose description, producer, per-cohort?
def pool_prompt_count(results, cohort):
    """``n of total``: how many outer splits chose the full pool's modal prompt.

    The design-space analyses run on one prompt per cohort, and the setup sentence justifies it by
    this count. It is the FULL POOL's count, which differs from the selected jury's
    (tab:app-pj-stability) -- check X21 was the two being read as one.
    """
    import json
    head = optional(results, cohort, "s7_headline")
    if head is None:
        raise Gap("needs s7_headline.csv from exp14_04")
    head = head.set_index("config")
    if "full_pool" not in head.index:
        raise Gap("s7_headline.csv has no full_pool row")
    try:
        modes = json.loads(head.loc["full_pool"].get("pair_modes"))
    except (TypeError, ValueError):
        raise Gap("s7_headline.csv full_pool row has no pair_modes")
    per_prompt = {}
    for pair, n in modes:
        prompt = str(pair).split("|", 1)[0]
        per_prompt[prompt] = per_prompt.get(prompt, 0) + int(n)
    prompt, n = max(per_prompt.items(), key=lambda kv: kv[1])
    total = sum(per_prompt.values())
    return (f"{n} of {total}",
            f"s7_headline.csv, full_pool pair_modes, modal prompt "
            f"{C.PROMPT_LABEL.get(prompt, prompt)}")


def sel_vs_pool_delta(results, cohort):
    """Selected jury minus full pool, micro F1, with its paired 95 % bootstrap interval.

    The discussion calls the two "similar". A non-significant test alone cannot carry that, the
    interval can (check Q21). Written by the PhenoJury protocol's S7 beside the test's p-values.
    """
    sig = optional(results, cohort, "s7_significance")
    if sig is None:
        raise Gap("needs s7_significance.csv from exp14_04")
    row = sig[sig["comparison"] == "phenojury vs full_pool"]
    if row.empty:
        raise Gap("s7_significance.csv has no 'phenojury vs full_pool' row")
    if "delta_lo" not in sig.columns:
        raise Gap("s7_significance.csv predates the paired interval -- rerun exp14_04 "
                  "stages=[grid,selection,headline]")
    r = row.iloc[0]
    fmt = lambda v: f"${float(v):+.2f}$"
    return (f"{fmt(r['delta_micro_f1'])} [{fmt(r['delta_lo'])}, {fmt(r['delta_hi'])}]",
            "s7_significance.csv, phenojury vs full_pool, paired report bootstrap")


# -- the core-3 member swap (check X24) -------------------------------------------
# fixed:core3 (MedGemma, MedPsy, Phi-4) against fixed:core3swap (Intelligent-Internet in place of
# MedGemma), both exploratory fixed juries of the PhenoJury protocol's `headline` stage.
CORE3, SWAP = "fixed:core3", "fixed:core3swap"


def _fixed_row(results, cohort, config, column):
    head = optional(results, cohort, "s7_headline")
    if head is None:
        raise Gap("needs s7_headline.csv from exp14_04")
    row = head[head["config"] == config]
    if row.empty:
        raise Gap(f"s7_headline.csv has no {config} row -- run exp14_04 with CORE3_SWAP set "
                  "(the default of its cluster scripts)")
    return f"{float(row.iloc[0][column]):.2f}", f"s7_headline.csv, {config}, {column}"


def swap_delta(results, cohort):
    """core3swap minus core3, micro F1, with its paired 95 % bootstrap interval."""
    exp = optional(results, cohort, "s7_exploratory_fixed_jury")
    if exp is None or "delta_lo" not in exp.columns:
        raise Gap("needs s7_exploratory_fixed_jury.csv with delta_lo/delta_hi -- rerun exp14_04 "
                  "stages=[grid,selection,headline]")
    fmt = lambda v: f"${float(v):+.2f}$"
    for comparison, sign in ((f"{SWAP} vs {CORE3}", 1), (f"{CORE3} vs {SWAP}", -1)):
        row = exp[exp["comparison"] == comparison]
        if not row.empty:
            r = row.iloc[0]
            d, lo, hi = (sign * float(r[k]) for k in ("delta_micro_f1", "delta_lo", "delta_hi"))
            lo, hi = min(lo, hi), max(lo, hi)
            return (f"{fmt(d)} [{fmt(lo)}, {fmt(hi)}]",
                    f"s7_exploratory_fixed_jury.csv, {comparison} (as swap minus core-3)")
    raise Gap(f"s7_exploratory_fixed_jury.csv has no {SWAP} vs {CORE3} row")


NUMBERS = [
    ("Jurors", "readers per cohort (models x prompts x normalisers, reference reader "
                "excluded)", jurors, False),
    ("Reports", "documents scored", reports, True),
    ("Prompt", "selected prompt", selected_prompt, True),
    ("Normaliser", "selected normaliser", selected_normaliser, True),
    ("K", "modal vote threshold $k$", selected_k, True),
    ("PoolPromptCount", "outer splits in which the full pool chose its modal prompt",
     pool_prompt_count, True),
    ("SelVsPoolDelta", "selected jury minus full pool, micro F1 [95 % paired CI]",
     sel_vs_pool_delta, True),
    ("CoreThreeP", "core-3 jury, micro precision",
     lambda r, c: _fixed_row(r, c, CORE3, "micro_precision"), True),
    ("CoreThreeR", "core-3 jury, micro recall",
     lambda r, c: _fixed_row(r, c, CORE3, "micro_recall"), True),
    ("CoreThreeF", "core-3 jury, micro F1", lambda r, c: _fixed_row(r, c, CORE3, "micro_f1"),
     True),
    ("SwapP", "core-3 with Intelligent-Internet for MedGemma, micro precision",
     lambda r, c: _fixed_row(r, c, SWAP, "micro_precision"), True),
    ("SwapR", "core-3 with Intelligent-Internet for MedGemma, micro recall",
     lambda r, c: _fixed_row(r, c, SWAP, "micro_recall"), True),
    ("SwapF", "core-3 with Intelligent-Internet for MedGemma, micro F1",
     lambda r, c: _fixed_row(r, c, SWAP, "micro_f1"), True),
    ("SwapDelta", "swap minus core-3, micro F1 [95 % paired CI]", swap_delta, True),
    ("PhenobertProposedShare", "share of annotated pairs whose only source is a curator "
                               "suggestion made with PhenoBERT's output on screen (percent)",
     phenobert_proposed_share, True),
    ("WallClockSeconds", "wall clock per report, seconds", wall_clock_s, True),
    ("PeakMemSequential", "peak memory, sequential serving (GB)", peak_mem_sequential, False),
    ("PeakMemParallel", "peak memory, parallel serving (GB)", peak_mem_parallel, False),
    ("SapbertIndexBuild", "SapBERT index build time, seconds", sapbert_index_build_s, False),
]

# LaTeX macro names cannot carry digits or _ character characters, so the cohort rides as a word. The GSC+
# macros keep the suffix `Gsc` across the move from the 228- to the 206-document frame, so the
# prose that quotes them does not change.
COHORT_SUFFIX = {"hcy": "Hcy", "gsc206": "Gsc"}


def main():
    """Write the chapter 5 inline-number macros and their Markdown twin."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.DEFAULT_RESULTS)
    ap.add_argument("--texdir", default=C.DEFAULT_TEXDIR)
    args = ap.parse_args()
    os.makedirs(args.texdir, exist_ok=True)

    cohorts = [c for c in C.COHORTS if C.has_cohort(args.results, c)]
    if not cohorts:
        raise SystemExit(f"no cohort tables under {args.results}")

    resolved, gaps = [], []
    for stem, description, producer, per_cohort in NUMBERS:
        targets = cohorts if per_cohort else cohorts[:1]
        for cohort in targets:
            macro = rf"\chFive{stem}" + (COHORT_SUFFIX[cohort] if per_cohort else "")
            label = f"{description}" + (f" ({C.COHORT_SHORT[cohort]})" if per_cohort else "")
            try:
                value, source = producer(args.results, cohort)
                resolved.append((macro, label, value, source))
            except Gap as exc:
                gaps.append((macro, label, str(exc)))
            except (KeyError, SystemExit) as exc:
                gaps.append((macro, label, str(exc)))

    tex = [
        "%% GENERATED FILE -- do not edit by hand.",
        f"%% Regenerate with: python figures/thesis_figures_scripts/{SCRIPT}",
        "%% Source: output/phenojury_protocol/{hcy,gsc206}/",
        "%%",
        "%% Input this once in the preamble, then quote the macros in the prose:",
        r"%%     \input{thesis_figures_latex/num_ch5_inline}",
        r"%% A macro expanding to \textbf{??} has no source yet -- see figures/GAPS.md.",
        "",
    ]
    for macro, label, value, source in resolved:
        tex.append(f"%% {label} -- {source}")
        tex.append(rf"\newcommand{{{macro}}}{{{value}}}")
    if gaps:
        tex.append("")
        tex.append("%% ---- no source yet; these expand to ?? on purpose ----")
        for macro, label, reason in gaps:
            tex.append(f"%% {label}")
            tex.append(f"%%   {reason}")
            tex.append(rf"\newcommand{{{macro}}}{{{MISSING}}}")
    out = os.path.join(args.texdir, "num_ch5_inline.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"[nums] num_ch5_inline.tex  ({len(resolved)} resolved, "
          f"{len(gaps)} awaiting a source)")

    md = [
        "# Chapter 5 inline numbers",
        "",
        f"Generated by `figures/thesis_figures_scripts/{SCRIPT}`. Do not edit.",
        "Quote these in the prose as macros, never as typed digits:",
        "`\\input{thesis_figures_latex/num_ch5_inline}`.",
        "",
        "| macro | number | value | source |",
        "|---|---|---|---|",
    ]
    for macro, label, value, source in resolved:
        md.append(f"| `{macro}` | {label} | {value.replace(chr(92) + ',', ' ')} | {source} |")
    if gaps:
        md += ["", "## Awaiting a source", "",
               "These expand to a bold `??` in the PDF, so a draft cannot quietly ship without",
               "them. See `figures/GAPS.md`.", "",
               "| macro | number | what would produce it |", "|---|---|---|"]
        for macro, label, reason in gaps:
            md.append(f"| `{macro}` | {label} | {reason} |")
    md_out = os.path.join(args.texdir, "num_ch5_inline.md")
    with open(md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("[nums] num_ch5_inline.md")


if __name__ == "__main__":
    main()
