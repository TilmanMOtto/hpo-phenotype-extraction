#!/usr/bin/env python
r"""The chapter-3 numbers the prose quotes inline, as LaTeX macros.

Same contract as num_ch4_inline.py and num_ch5_inline.py: a number a paragraph asserts is computed
here and quoted as a macro, never typed.

    \input{thesis_figures_latex/num_ch3_inline}
    ... 958 pairs inside the subontology, over \chThreeGscSubsetTerms{} unique terms ...

Source: ``output/dataset_statistics/dataset_statistics.csv`` (``experiments/03_setup/dataset_statistics.py``),
the same file ``tab_ch3_datasets`` is built from, so the paragraph and the table cannot disagree.
A number whose cohort has not been computed expands to a bold ``??``.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_common as C                       # noqa: E402

SCRIPT = "num_ch3_inline.py"
MISSING = r"\textbf{??}"


def count(v) -> str:
    r"""A count as the prose writes it, ``1\,649`` -- the thin space of every typed number and of
    the chapter-4/5 macros. ``count``'s ``1{,}649`` is the dataset table's convention."""
    return MISSING if v is None else f"{int(round(v)):,}".replace(",", r"\,")


# LaTeX macro names cannot carry digits or _ character characters, so the cohort rides as a word.
COHORT_WORD = {"hcy": "Hcy", "gsc_raghpo_ann": "GscSubset", "gsc_2024_eval_206": "GscEval"}

# macro stem, prose description, statistic, formatter[, cohorts it exists for]
NUMBERS = [
    ("Reports", "reports", "n_reports", count),
    ("Pairs", "gold (report, term) pairs as scored", "n_gold_pairs", count),
    ("Terms", "unique gold terms as scored", "n_unique_terms", count),
    ("EmptyReports", "reports with an empty gold set", "n_reports_empty_gold", count),
    ("OutOfSubtree", "gold pairs dropped as outside the subontology", "n_gold_out_of_subtree",
     count),
    ("ParentChild", "(report, term, direct parent) with both in the gold", "n_parent_child_pairs",
     count, ("hcy",)),
    ("RawPairs", "gold (report, term) pairs as loaded", "n_gold_pairs_raw", count),
    # RAG-HPO's re-annotation against the corpus annotation of the same 114 documents.
    ("ReannOriginal", "corpus annotation of the same documents (raw pairs)",
     "reann_pairs_original", count, ("gsc_raghpo_ann",)),
    ("ReannKept", "raw pairs in both annotations", "reann_kept", count, ("gsc_raghpo_ann",)),
    ("ReannAdded", "raw pairs only in the re-annotation", "reann_added", count,
     ("gsc_raghpo_ann",)),
    ("ReannRemoved", "raw pairs only in the corpus annotation", "reann_removed", count,
     ("gsc_raghpo_ann",)),
    ("ReannUnion", "raw pairs in either annotation", "reann_union", count, ("gsc_raghpo_ann",)),
    ("ReannJaccard", "Jaccard similarity of the two annotations", "reann_jaccard",
     lambda v: f"{v:.2f}", ("gsc_raghpo_ann",)),
    ("ReannFewerShare", "percent fewer raw pairs in the re-annotation", "reann_fewer_share",
     lambda v: f"{v:.1f}", ("gsc_raghpo_ann",)),
    # The curation record of the HCY ground truth, copied from its manifest by dataset_statistics.py.
    ("CurationCandidates", "candidate annotations considered", "curation_candidates", count,
     ("hcy",)),
    ("CurationAdmitted", "candidate annotations admitted", "curation_admitted", count, ("hcy",)),
    ("CurationDropped", "candidate annotations dropped", "curation_dropped", count, ("hcy",)),
    ("CurationDroppedDeletion", "dropped: deletion proposal", "curation_dropped_deletion",
     count, ("hcy",)),
    ("CurationDroppedNoEvidence", "dropped: no evidence segment", "curation_dropped_no_evidence",
     count, ("hcy",)),
    ("CurationDroppedUncertain", "dropped: labelled uncertain", "curation_dropped_uncertain",
     count, ("hcy",)),
    ("CurationDroppedFamily", "dropped: attributed to a relative", "curation_dropped_family",
     count, ("hcy",)),
    ("CurationDroppedUnfinished", "dropped: marked as needing work",
     "curation_dropped_unfinished", count, ("hcy",)),
]


def main():
    """Write the chapter 3 inline-number macros and their Markdown twin."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.DEFAULT_RESULTS)
    ap.add_argument("--texdir", default=C.DEFAULT_TEXDIR)
    args = ap.parse_args()
    os.makedirs(args.texdir, exist_ok=True)
    stats = C.load(args.results)

    resolved, gaps = [], []
    for cohort in C.COHORTS:
        for stem, description, statistic, render, *only in NUMBERS:
            if only and cohort not in only[0]:
                continue
            macro = rf"\chThree{COHORT_WORD[cohort]}{stem}"
            label = f"{description} ({C.COHORT_HEADER[cohort]})"
            if cohort in stats and statistic in stats[cohort]:
                resolved.append((macro, label, render(stats[cohort][statistic]),
                                 f"dataset_statistics.csv, {cohort}/{statistic}"))
            else:
                gaps.append((macro, label, C.PRODUCER[cohort]))

    tex = [
        "%% GENERATED FILE -- do not edit by hand.",
        f"%% Regenerate with: python figures/thesis_figures_scripts/{SCRIPT}",
        "%% Source: output/dataset_statistics/dataset_statistics.csv",
        "%%",
        "%% Input this once in the preamble, then quote the macros in the prose:",
        r"%%     \input{thesis_figures_latex/num_ch3_inline}",
        r"%% A macro expanding to \textbf{??} has no source yet.",
        "",
    ]
    for macro, label, value, source in resolved:
        tex.append(f"%% {label} -- {source}")
        tex.append(rf"\newcommand{{{macro}}}{{{value}}}")
    if gaps:
        tex += ["", "%% ---- no source yet; these expand to ?? on purpose ----"]
        for macro, label, producer in gaps:
            tex.append(f"%% {label} -- run: {producer}")
            tex.append(rf"\newcommand{{{macro}}}{{{MISSING}}}")
    with open(os.path.join(args.texdir, "num_ch3_inline.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"[nums] num_ch3_inline.tex  ({len(resolved)} resolved, {len(gaps)} awaiting a source)")

    md = [
        "# Chapter 3 inline numbers",
        "",
        f"Generated by `figures/thesis_figures_scripts/{SCRIPT}`. Do not edit.",
        "Quote these in the prose as macros, never as typed digits:",
        "`\\input{thesis_figures_latex/num_ch3_inline}`.",
        "",
        "| macro | number | value | source |",
        "|---|---|---|---|",
    ]
    for macro, label, value, source in resolved:
        md.append(f"| `{macro}` | {label} | {value.replace(chr(92) + ',', ',')} | {source} |")
    if gaps:
        md += ["", "## Awaiting a source", "", "| macro | number | what would produce it |",
               "|---|---|---|"]
        md += [f"| `{m}` | {lab} | `{p}` |" for m, lab, p in gaps]
    with open(os.path.join(args.texdir, "num_ch3_inline.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("[nums] num_ch3_inline.md")


if __name__ == "__main__":
    main()
