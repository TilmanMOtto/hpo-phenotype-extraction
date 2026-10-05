#!/usr/bin/env python
r"""The chapter-3 dataset table (`tab:datasets`), as a standalone booktabs `table` environment.

Every number is read from ``output/dataset_statistics/dataset_statistics.csv`` at build time --
the ground truth as scored, through the same loaders and normalisation as the comparison -- so the table cannot
drift from the ground truth the results chapters divide by. Re-run after any change of the curated ground truth.

Three rows are descriptive facts about the corpora rather than counts (text type, languages, who
wrote the ground truth), and are constants below.

    python experiments/figures/tables_ch3.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_common as C                                       # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/tables_ch3.py"

#: Not computed: what the corpora are and who wrote their ground truth. The GSC+ (114) ground truth is RAG-HPO's
#: re-annotation (resources/data/GSC_RAGHPO/annotations.csv). The GSC+ (206) ground truth is the corpus's
#: own annotation over AutoPCR's evaluation split. The HCY ground truth is the curated set of this work.
DESCRIPTIVE = [
    ("Text", {"hcy": "clinical records", "gsc_raghpo_ann": "abstracts",
              "gsc_2024_eval_206": "abstracts"}),
    ("Original languages", {"hcy": "DE, EN, FR, IT", "gsc_raghpo_ann": "EN",
                            "gsc_2024_eval_206": "EN"}),
]
GROUND_TRUTH = {"hcy": "this work", "gsc_raghpo_ann": "RAG-HPO authors",
                "gsc_2024_eval_206": "GSC+"}


def datasets(results: str, texdir: str) -> None:
    """Write ``tab_ch3_datasets.tex``, the three ground truths as scored."""
    stats = C.load(results)
    present = [c for c in C.COHORTS if c in stats]
    absent = [c for c in C.COHORTS if c not in stats]

    def cell(cohort, fn):
        return fn(stats[cohort]) if cohort in stats else C.MISSING_TEX

    def median(prefix, nd):
        return lambda s: C.num(s[prefix + "_median"], nd)

    def computed_rows(spec):
        return [[label] + [cell(c, fn) for c in C.COHORTS] for label, fn in spec]

    rows = [[label] + [by[c] for c in C.COHORTS] for label, by in DESCRIPTIVE]
    rows += computed_rows([("Documents", lambda s: C.count(s["n_reports"]))])
    rows.append(["Ground truth"] + [GROUND_TRUTH[c] for c in C.COHORTS])
    rows += computed_rows([
        ("Pairs", lambda s: C.count(s["n_gold_pairs_raw"])),
        (r"Pairs below $v_0$", lambda s: C.count(s["n_gold_pairs"])),
        (r"Distinct terms below $v_0$", lambda s: C.count(s["n_unique_terms"])),
    ])
    rows.append(r"\midrule")
    rows += computed_rows([
        ("Documents with 0 annotations", lambda s: C.count(s["n_reports_empty_gold"])),
        ("Words per document, median", median("words", 0)),
        ("Terms per document, mean", lambda s: f"{s['terms_per_report_mean']:.1f}"),
        (r"Term depth $\delta$, median", median("depth", 1)),
        ("Terms in one document only (\\%)", lambda s: f"{s['singleton_term_share']:.1f}"),
    ])

    # Caption: what is shown. Every definition goes in the note below the float.
    notes = (
        r"Pairs: unique (document, term) pairs of the ground truth. Below $v_0$: pairs inside the "
        r"label space $\mathcal{V}$, on which every system is scored. The pairs outside it are "
        r"mostly modes of inheritance. Term statistics are over the pairs below $v_0$, and the mean "
        r"number of terms per document includes documents with 0 annotations. Depth $\delta$: "
        r"shortest path from $v_0$. Words: whitespace-delimited, counted in the text every system "
        r"receives, for HCY the English translation."
    )
    nonexistent = {c: stats[c]["n_gold_nonexistent"] for c in present
                   if stats[c]["n_gold_nonexistent"]}
    if nonexistent:
        notes += " Pairs dropped as absent from the ontology file: " + ", ".join(
            f"{C.COHORT_SHORT[c]} {C.count(v)}" for c, v in nonexistent.items()) + "."
    if absent:
        notes += " " + (
            r"\textbf{Not yet computed:} " + ", ".join(C.COHORT_SHORT[c] for c in absent)
            + " -- " + ", ".join(sorted({C.tex_escape(C.PRODUCER[c]) for c in absent})) + ".")

    caption = (r"\textbf{Evaluation datasets}: size, ground truth and term statistics of the "
               r"three corpora.")
    C.write_table_tex(
        os.path.join(texdir, "tab_ch3_datasets.tex"),
        caption=caption, label="tab:datasets",
        header=[""] + [C.COHORT_HEADER[c] for c in C.COHORTS],
        rows=rows, colspec="@{}lccc@{}", notes=notes, generator=GENERATOR, placement="t",
        compact=True,
    )


ALL = [datasets]


def main():
    """Build the chapter 3 tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.DEFAULT_RESULTS)
    ap.add_argument("--texdir", default=C.DEFAULT_TEXDIR)
    args = ap.parse_args()
    os.makedirs(args.texdir, exist_ok=True)
    for fn in ALL:
        fn(args.results, args.texdir)


if __name__ == "__main__":
    main()
