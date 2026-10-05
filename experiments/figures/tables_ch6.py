#!/usr/bin/env python
r"""Every chapter-6 (final comparison) table, as a standalone booktabs `table` environment.

Generated rather than typed so the chapter and the results cannot drift: every number is read out of
`output/comparison/tables/` at build time. Re-run after any re-run of the comparison.

The three specified artifacts:

    T6.1  overall_comparison   quality AND cost, 7 methods x 2 cohorts, out-of-envelope rows marked
    T6.2  decisive_subset      THE DECISIVE TABLE: the annotated pairs with no string match to the term
    T6.3  complementarity      how much of what one architecture misses the other recovers

and the supporting tables they refer out to, so no column had to be dropped to make room:

    overall_macro   both macro conventions (mean of per-document F1, and F1 of the two means)
    hierarchy       ancestor-closure hP/hR/hF and CoPHE
    literature      published figures, our replications and PhenoJury, per GSC+ frame
    scoring         appendix: the two scoring conventions reconciled
    cost_detail     runtime, calls, memory, and which run actually paid it
    qualitative_*   appendix: the five methods side by side on hand-picked segments (per cohort)

Requires \usepackage{booktabs}, and \usepackage{pifont} for the side-by-side examples' marks;
\usepackage{siunitx} is *not* needed -- numbers are pre-formatted.

    python experiments/figures/tables_ch6.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C                                       # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/tables_ch6.py"


def _ci(row, key, lo_key=None, hi_key=None, nd=2):
    """`0.64 [0.61, 0.67]`, or an em dash when the row has no number."""
    if row is None or key not in row:
        return C.MISSING_TEX
    point = C.fmt(row[key], nd)
    if point == "--":
        return C.MISSING_TEX
    lo, hi = row.get(lo_key or f"{key}_lo"), row.get(hi_key or f"{key}_hi")
    lo_s, hi_s = C.fmt(lo, nd), C.fmt(hi, nd)
    if "--" in (lo_s, hi_s):
        return point
    return rf"{point} \footnotesize[{lo_s}, {hi_s}]"


def _val(row, key, nd=2):
    if row is None or key not in row:
        return C.MISSING_TEX
    out = C.fmt(row[key], nd)
    return C.MISSING_TEX if out == "--" else out


def _breakable(text) -> str:
    r"""Escape for LaTeX, then allow a line break after each _ character and hyphen.

    Experiment ids (``baseline_raghpo_8b``) and model names
    (``Llama-3.3-70B-Instruct (8-bit)``) are single unbreakable words to TeX, so in a narrow ``p``
    column one of them overflows its own cell no matter how the column is sized -- narrowing the
    column makes it worse, not better. _ character characters and hyphens are where a reader would
    break these strings anyway, so ``\allowbreak`` is placed after each. It adds no glue and
    changes nothing when the string already fits.
    """
    escaped = C.tex_escape(str(text))
    return escaped.replace(r"\_", r"\_\allowbreak{}").replace("-", r"-\allowbreak{}")


# ── T6.1 ────────────────────────────────────────────────────────────────────────────────────
#: Our systems start here in every block. A little vertical space separates them from the others.
FIRST_OURS = "treephenorag"


def ours_gap(label: str, method: str) -> str:
    r"""Prefix ``\addlinespace`` to the first of our rows, which booktabs reads after ``\\``."""
    return (r"\addlinespace " + label) if method == FIRST_OURS else label


# ── Thesis-table conventions ─────────────────────────────────────────────────────────────────
# The four tables the chapter prints (tab:ch6-overall-comparison, tab:ch6-literature, tab:ch6-hierarchy,
# tab:ch6-decisive-subset) are \input from here, never typed. They follow the chapter's own convention,
# which differs from `bold_best` in two ways: ties at the PRINTED precision are all bold (two rows
# showing 0.75 are equally best to the reader), and the second-best printed value is underlined.

#: Block headers as the chapter words them (the figures keep C.COHORT_HEADER's shorter form).
THESIS_COHORT_HEADER = {
    "hcy": "HCY (118 clinical reports)",
    "gsc_raghpo_ann": "GSC+ (114 abstracts, selected and re-annotated by the RAG-HPO authors)",
    "gsc_2024_eval_206": "GSC+ (206 abstracts, AutoPCR's evaluation split)",
}
THESIS_COHORT_SHORT_HEADER = {
    "hcy": "HCY (118 clinical reports)",
    "gsc_raghpo_ann": "GSC+ (114 abstracts)",
    "gsc_2024_eval_206": "GSC+ (206 abstracts)",
}
#: Row labels in the chapter: plain text, no small caps.
THESIS_METHOD_LABEL = {
    "raghpo_8b": "RAG-HPO 8B", "raghpo_70b": "RAG-HPO 70B",
    "autopcr_8b": "AutoPCR 8B", "autopcr_70b": "AutoPCR 70B",
    "phenobert": "PhenoBERT", "treephenorag": "TreePhenoRAG",
    "phenojury": "PhenoJury (full pool, 8)", "phenojury_core3": "PhenoJury (core-3)",
}
#: The system a caption names when a row leads: sizes of one system are one system.
FAMILY = {"raghpo_8b": "RAG-HPO", "raghpo_70b": "RAG-HPO", "autopcr_8b": "AutoPCR",
          "autopcr_70b": "AutoPCR", "phenobert": "PhenoBERT", "treephenorag": "TreePhenoRAG",
          "phenojury": "PhenoJury", "phenojury_core3": "PhenoJury"}
THESIS_MISSING = "--"


def _tv(row, key, nd=2):
    """A number as the chapter prints it, or `--`."""
    if row is None or key not in row:
        return THESIS_MISSING
    out = C.fmt(row[key], nd)
    return THESIS_MISSING if out == "--" else out


def _tci(row, key, lo_key=None, hi_key=None, nd=2, inline=False):
    """`[0.61, 0.67]` as its own cell, or `0.52 [0.43, 0.62]` when ``inline``."""
    point = _tv(row, key, nd)
    if row is None:
        return THESIS_MISSING
    lo, hi = C.fmt(row.get(lo_key or f"{key}_lo"), nd), C.fmt(row.get(hi_key or f"{key}_hi"), nd)
    if inline:
        return point if "--" in (lo, hi) or point == THESIS_MISSING else f"{point} [{lo}, {hi}]"
    return THESIS_MISSING if "--" in (lo, hi) else f"[{lo}, {hi}]"


# The marking rule lives in chapter 4's `common`, beside `bold_best`: it knows only about rows,
# columns and blocks, and chapters 4 and 5 print the same bold/underline convention.
from common import mark_best_second  # noqa: E402,F401


def _join(names: list) -> str:
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _leaders(df, cohort) -> list:
    """Systems whose micro F1 is the best PRINTED value on the cohort (ties at 2 dp all lead)."""
    by = C.by_method(df, cohort)
    printed = {m: C.fmt(r.get("micro_f1")) for m, r in by.items() if r is not None}
    printed = {m: float(v) for m, v in printed.items() if v != "--"}
    if not printed:
        return []
    top = max(printed.values())
    return [FAMILY[m] for m in C.METHOD_ORDER if printed.get(m) == top]


def main_result_sentence(df) -> str:
    """The caption's claim, read off the numbers: who leads on the reports, who on the abstracts."""
    hcy = _leaders(df, "hcy")
    abstracts = _leaders(df, "gsc_raghpo_ann") + _leaders(df, "gsc_2024_eval_206")
    if hcy and abstracts:
        return (f"{_join(hcy)} is the most accurate system on the clinical reports, and "
                f"{_join(abstracts)} on the abstracts.")
    if hcy:
        return f"{_join(hcy)} is the most accurate system on the clinical reports."
    if abstracts:
        return f"{_join(abstracts)} is the most accurate system on the abstracts."
    return ""


#: The overall-comparison layout, shared by tab:ch6-overall-comparison and the two chapter-level
#: comparisons that reuse it (tables_ch4.main_comparison, tables_ch5.t50_main_comparison), so the
#: three cannot drift apart in columns, marking or type size.
OVERALL_HEADER = ["System", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$", r"95\,\% CI",
                  "$P_M$", "$R_M$", "$F_{1,M}$"]
OVERALL_COLSPEC = "@{}lccccccc@{}"
OVERALL_MARKED = [1, 2, 3, 5, 6, 7]


def overall_rows(df, cohorts=None, methods=None) -> list:
    """One block per cohort, one row per method, best and second best marked within each block.

    *cohorts* and *methods* restrict the table and keep the roster order; ``None`` keeps all. A
    method with no scored row on a cohort is printed as a row of dashes, never dropped.
    """
    cohorts = [c for c in C.cohorts_present(df) if cohorts is None or c in cohorts]
    methods = [m for m in C.methods_present(df) if methods is None or m in methods]
    rows = []
    for ci, cohort in enumerate(cohorts):
        by = C.by_method(df, cohort)
        if ci:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{8}{@{}l}{\emph{%s}}" % THESIS_COHORT_HEADER[cohort])
        for method in methods:
            row = by.get(method)
            rows.append([THESIS_METHOD_LABEL[method],
                         _tv(row, "micro_precision"), _tv(row, "micro_recall"),
                         _tv(row, "micro_f1"), _tci(row, "micro_f1"),
                         _tv(row, "macro_precision"), _tv(row, "macro_recall"),
                         _tv(row, "macro_f1")])
    mark_best_second(rows, OVERALL_MARKED)
    return rows


def write_overall_table(path, *, rows, caption, label, generator, placement="t", notes=None):
    """Write *rows* (from :func:`overall_rows`) in the overall-comparison layout."""
    C.write_table_tex(
        path, caption=caption, label=label, placement=placement, header=OVERALL_HEADER,
        rows=rows, colspec=OVERALL_COLSPEC, generator=generator, notes=notes,
        preamble=[r"\small", r"\setlength{\tabcolsep}{4pt}"])


def _corpus(cohort: str) -> str:
    return "HCY" if cohort == "hcy" else "GSC+"


def t61_overall_comparison(results, texdir):
    r"""\Cref{tab:ch6-overall-comparison}: micro and macro P/R/$F_1$ of every system on each corpus.

    The caption is derived from the rows (``headline_sentence``), so it cannot outlive the numbers
    it summarises.
    """
    df = C.table(results, "t1_overall")
    rows = overall_rows(df)
    caption = r"\textbf{" + main_result_sentence(df) + r"}"
    notes = (
        r"PhenoJury: pooled out-of-fold "
        r"predictions of nested cross-validation. TreePhenoRAG: pooled out-of-fold on HCY, the "
        r"HCY-selected configuration on GSC+ (114), not run on GSC+ (206).")
    write_overall_table(os.path.join(texdir, "tab_ch6_overall_comparison.tex"),
                        rows=rows, caption=caption, notes=notes, label="tab:ch6-overall-comparison",
                        generator=GENERATOR)


def _estimator_sentence(estimators: dict) -> str:
    """How each row's configuration was arrived at, as prose, not three superscripts.

    Three different things are being compared and the difference counts: a method with nothing to
    tune, a method whose configuration is chosen inside nested CV, and one where a configuration
    selected on another cohort is applied unchanged. The third is the strictest and reads as the
    weakest if it is not said. Saying it once here costs a sentence. Saying it as a mark cost the
    reader a lookup on every row.
    """
    if not estimators:
        return ""
    parts = []
    for name in C.ESTIMATOR_ORDER:
        methods = estimators.get(name)
        if not methods:
            continue
        unique = sorted(set(methods))
        parts.append(C.tex_escape(", ".join(unique)) + " --- " + C.ESTIMATOR_TEX[name])
    return (r"Operating points: " + ". ".join(parts) + ". ") \
        if parts else ""


def t6_overall_macro(results, texdir):
    r"""APPENDIX. The two macro conventions side by side.

    Macro P/R/$F_1$ moved into Table~\ref{tab:ch6-overall-comparison} when that table dropped its
    cost columns, so what is left here is the column that could not go there: the harmonic mean of
    the two macro averages, beside the mean of per-document $F_1$. Garcia et al.'s Table 5 is NOT
    the former: their Methods average the per-case $F_1$ (the workbook they release reproduces
    0.71 that way and 0.73 as the harmonic mean), so the literature table compares micro against
    micro instead, recomputed from their printed counts -- see t4_literature.

    One column and a system name is a thin table, and that is the correct size for it. Folding it
    into the main table as an eighth column would put a quantity used once beside six used
    throughout, and a reader would have to be told to ignore it.
    """
    df = C.table(results, "t1_overall")
    rows = []
    for ci, cohort in enumerate(C.cohorts_present(df)):
        by = C.by_method(df, cohort)
        if ci:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{3}{l}{\emph{%s}}" % C.COHORT_HEADER[cohort])
        for method in C.methods_present(df):
            row = by.get(method)
            rows.append([
                C.METHOD_LABEL_TEX[method],
                _val(row, "macro_f1"),
                _val(row, "macro_f1_of_means"),
            ])
    C.bold_best(rows, [1, 2])
    notes = (
        r"$F_1^{\mu\text{-of-means}}$: harmonic mean of "
        r"macro precision and macro recall. "
        + C.MISSING_NOTE)
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_overall_macro.tex"),
        caption=(r"\textbf{Two macro averages of $F_1$}, for every system on each cohort."),
        label="tab:ch6-overall-macro",
        header=["System", r"$F_{1,M}$", r"$F_1^{\mu\text{-of-means}}$"],
        rows=rows, colspec="l r r", notes=notes, generator=GENERATOR)


# ── T2 ──────────────────────────────────────────────────────────────────────────────────────
def t2_hierarchy(results, texdir):
    r"""\Cref{tab:ch6-hierarchy}: ancestor-closure and CoPHE P/R/F of every system on each corpus."""
    df = C.table(results, "t2_hierarchy")
    rows = []
    for ci, cohort in enumerate(C.cohorts_present(df)):
        by = C.by_method(df, cohort)
        if ci:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{7}{@{}l}{\emph{%s}}" % THESIS_COHORT_SHORT_HEADER[cohort])
        for method in C.methods_present(df):
            row = by.get(method)
            rows.append([THESIS_METHOD_LABEL[method],
                         _tv(row, "micro_hp"), _tv(row, "micro_hr"), _tv(row, "micro_hf"),
                         _tv(row, "micro_cophe_precision"), _tv(row, "micro_cophe_recall"),
                         _tv(row, "micro_cophe_f1")])
    mark_best_second(rows, [1, 2, 3, 4, 5, 6])
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_hierarchy.tex"),
        caption=(r"\textbf{Hierarchy-aware quality.} Ancestor-closure and CoPHE precision, recall "
                 r"and $F$ of every system on each corpus."),
        label="tab:ch6-hierarchy", placement="h",
        header=["System", "$hP$", "$hR$", "$hF$", r"CoPHE-$P$", r"CoPHE-$R$", r"CoPHE-$F$"],
        rows=rows, colspec="@{}lcccccc@{}", generator=GENERATOR,
        preamble=[r"\small", r"\setlength{\tabcolsep}{4pt}"],
        notes=r"TreePhenoRAG is not run on GSC+ (206).")


# ── T3 ──────────────────────────────────────────────────────────────────────────────────────
def t62_decisive_subset(results, texdir):
    r"""\Cref{tab:ch6-decisive-subset}: recall on the HCY terms without a matching phrase.

    Every count in the caption (slice sizes, their overlap, their union's share of the ground truth) is
    read from the comparison's CSVs. The union comes from ``t3_label_audit.csv`` because the two slices
    are separate qualifier columns, not a partition, and their sum over-counts.
    """
    df = C.table(results, "t3_subgroups")
    recall = df[df["kind"] == "recall"] if "kind" in df.columns else df
    attrib = df[df["kind"] == "attribution"] if "kind" in df.columns else df.iloc[0:0]
    groups = [g for g in ("lab_value", "implicit") if g in set(recall["subgroup"])]
    methods = [m for m in C.METHOD_ORDER if m in set(df["method"])]
    by = {(r["method"], r["subgroup"]): r for _, r in recall.iterrows()
          if str(r.get("status", "ok")) == "ok"}

    size = {}
    for g in groups:
        sample = next((by[(m, g)] for m in methods if (m, g) in by), None)
        size[g] = (int(sample["n_gold"]), int(sample["n_reports"])) if sample is not None else (0, 0)
    overall = C.table(results, "t1_overall")
    hcy = overall[overall["cohort"] == "hcy"]
    n_gold_total = int(hcy.iloc[0]["n_gold_pairs"]) if not hcy.empty else 0
    audit = C.table(results, "t3_label_audit")
    hit = audit[audit["label"] == "|".join(groups)]
    if hit.empty:
        raise SystemExit("t3_label_audit.csv has no union row for " + "|".join(groups)
                         + " -- the caption's overlap and share cannot be computed")
    union = int(hit.iloc[0]["n_pairs_in_gold"])
    overlap = sum(n for n, _ in size.values()) - union
    share = f"{union / n_gold_total * 100:.1f}" if n_gold_total else "--"

    rows = []
    for method in methods:
        cells = [THESIS_METHOD_LABEL[method]]
        for g in groups:
            cells.append(_tci(by.get((method, g)), "recall", "recall_lo", "recall_hi", inline=True))
        att = next((r for _, r in attrib.iterrows() if r["method"] == method), None)
        cells.append(THESIS_MISSING if att is None or str(att.get("status", "ok")) != "ok"
                     else f"{int(att['n_emitted'])} / {int(att['n_attribution_pairs'])}")
        rows.append(cells)
    mark_best_second(rows, list(range(1, 1 + len(groups))))
    # Relatives' findings are false positives by design, so fewer is better.
    mark_best_second(rows, 1 + len(groups), higher_is_better=False, whole_cell=True)

    n_rel = next((int(r["n_attribution_pairs"]) for _, r in attrib.iterrows()
                  if str(r.get("status", "ok")) == "ok"), 0)
    lab, imp = size.get("lab_value", (0, 0)), size.get("implicit", (0, 0))
    caption = (r"\textbf{Recall on HCY terms without a matching phrase, and relatives' findings "
               r"emitted.} The values of \Cref{fig:ch6-subgroups}.")
    notes = (
        rf"Laboratory values: the term follows from a value against its reference range ({lab[0]} "
        rf"terms in {lab[1]} reports). Implicit descriptions: neither label nor synonym occurs "
        rf"({imp[0]} terms in {imp[1]} reports). The two overlap in {overlap} terms and together "
        rf"hold {union} of the " + f"{n_gold_total:,}".replace(",", r"\,")
        + rf" annotated terms ({share}\,\%). Recall only, since restricting the ground truth does "
        r"not restrict the predictions. "
        rf"Relatives: excluded relatives' findings emitted (count out of {n_rel}).")
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_decisive_subset.tex"),
        caption=caption, notes=notes, label="tab:ch6-decisive-subset", placement="h",
        header=["System", "Laboratory value", "Implicit description", "Relatives"],
        rows=rows, colspec="@{}lccc@{}", generator=GENERATOR, preamble=[r"\small"])


# ── T4 ──────────────────────────────────────────────────────────────────────────────────────
#: The two GSC+ frames of \Cref{tab:ch6-literature}: the chapter's header, the cohort key in both CSVs,
#: and the systems we reproduced there.
LITERATURE_SECTIONS = [
    dict(header="GSC+ (114), selected and re-annotated by the RAG-HPO authors",
         cohort="gsc_raghpo_ann", replicated=["raghpo_8b", "raghpo_70b"],
         pairs=[("RAG-HPO + LLaMA-3 70B", "raghpo_70b")]),
    dict(header="GSC+ (206), AutoPCR's evaluation split", cohort="gsc_2024_eval_206",
         replicated=["autopcr_8b", "autopcr_70b", "phenobert"],
         pairs=[("AutoPCR (published, document-level)", "autopcr_70b")]),
]
#: Each section opens with its pairs: a published row (the comparison's label) directly above our
#: reproduction of the same system, so the gap between them reads at a glance. The remaining
#: reproductions and this work follow, and the published rows without a reproduction sit below a
#: thin rule as context.
LITERATURE_OURS = ["phenojury", "phenojury_core3"]

#: Published rows whose printed numbers are NOT micro over documents, and the mark they carry. A
#: per-case row is printed as micro recomputed from its own TP/FP/FN (`_published_micro`), and the
#: dagger says so. A mention-level row cannot be, and is printed as published.
CONVENTION_MARK = {"per-case": r"$^\dagger$", "mention": r"$^\ddagger$"}

#: The comparison's labels for published rows -> the chapter's wording. A published row missing here
#: aborts the build, not printing an unreviewed label.
LITERATURE_LABEL = {
    "AutoPCR (published, mention-level)": "AutoPCR",
    "AutoPCR (published, document-level)": "AutoPCR",
    # The Source column names the table each row is transcribed from, so the label need not.
    "PhenoBERT (in AutoPCR's Table 1)": "PhenoBERT",
    "FastHPOCR (in AutoPCR's Table 1)": "FastHPOCR",
}
#: First word of the `source` column -> bib key.
CITE_KEY = {"Garcia": "garcia2025", "Tao": "tao2026a"}


def _convention(row) -> str:
    """'' for a published row defined as ours are, else the key of its mark."""
    if str(row.get("unit", "")) == "mention":
        return "mention"
    if "per-case" in str(row.get("averaging", "")):
        return "per-case"
    return ""


def _cite(row) -> str:
    r"""``\citep[Table~N]{key}``: the paper AND the table the row is transcribed from (check R5).

    Both come from the row's own `source` string in the comparison's config, so the table number is
    written down once, where the number itself was transcribed.
    """
    source = str(row.get("source", "")).strip()
    first = source.split()[0] if source else ""
    if first not in CITE_KEY:
        raise SystemExit(f"t1_literature.csv: no bib key for source {source!r} -- add it "
                         "to CITE_KEY in tables_ch6.py")
    table = re.search(r"Table\s+(\d+)", source)
    if table is None:
        raise SystemExit(f"t1_literature.csv: source {source!r} names no table -- add it in "
                         "exp13_25's config, where the row is transcribed")
    return rf"\citep[Table~{table.group(1)}]{{{CITE_KEY[first]}}}"


def _published_micro(row) -> dict:
    """Micro P/R/F1 of a published row, from the TP/FP/FN counts the paper prints beside it.

    Garcia et al.'s Table 5 prints per-case means (their Methods: P, R and F1 per case, then the
    average), which no column of ours is defined as, but it also prints the pooled counts. Those
    give the micro scores every computed row of the table carries, from the paper's own numbers.
    """
    try:
        tp, fp, fn = (float(row[k]) for k in ("tp", "fp", "fn"))
    except (KeyError, TypeError, ValueError):
        tp = fp = fn = float("nan")
    if any(x != x for x in (tp, fp, fn)):
        raise SystemExit(f"t1_literature.csv: per-case row {row.get('label')!r} prints no TP/FP/FN, "
                         "so it cannot be recomputed as micro -- add the counts in exp13_25's config")
    return {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}


def _published_as_printed(lit) -> str:
    """'0.69, 0.77 and 0.71': the per-case means the paper prints for its RAG-HPO + LLaMA-3 70B row."""
    pub = lit[lit["label"] == PUBLISHED_RAGHPO]
    if pub.empty:
        raise SystemExit(f"t1_literature.csv has no {PUBLISHED_RAGHPO!r} row")
    pub = pub.iloc[0]
    return f"{_tv(pub, 'precision')}, {_tv(pub, 'recall')} and {_tv(pub, 'f1')}"


def t4_literature(results, texdir):
    r"""\Cref{tab:ch6-literature}: published figures, our reproductions, and PhenoJury per GSC+ frame.

    Nothing is marked best: a cited number comes from a paper, and a mention-level score beside a
    micro one is not a contest. The table is read along the pairs instead. Garcia et al.'s rows are
    micro, recomputed from the counts they print (`_published_micro`), so that pair IS like for like.
    """
    lit = C.table(results, "t1_literature")
    overall = C.table(results, "t1_overall")
    rows = []
    for si, section in enumerate(LITERATURE_SECTIONS):
        if si:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{7}{@{}l}{\emph{%s}}" % section["header"])
        ours = C.by_method(overall, section["cohort"])
        n_docs = next((int(r["n_reports"]) for r in ours.values() if r is not None), "")
        published = {str(r["label"]): r for _, r in lit[lit["cohort"] == section["cohort"]].iterrows()}

        def cited(label):
            row = published[label]
            shown = LITERATURE_LABEL.get(label, label)
            values = _published_micro(row) if _convention(row) == "per-case" else row
            return [C.tex_escape(shown) + CONVENTION_MARK.get(_convention(row), ""), _cite(row),
                    C.tex_escape(str(row.get("unit", ""))), str(int(row["n_documents"])),
                    _tv(values, "precision"), _tv(values, "recall"), _tv(values, "f1")]

        def computed(method, kind):
            row = ours.get(method)
            return None if row is None else [
                THESIS_METHOD_LABEL[method], kind, "document", str(n_docs),
                _tv(row, "micro_precision"), _tv(row, "micro_recall"), _tv(row, "micro_f1")]

        paired_labels, paired_methods = set(), set()
        for label, method in section["pairs"]:
            if label not in published:
                raise SystemExit(f"t1_literature.csv has no published row {label!r} to pair with "
                                 f"{method} -- update LITERATURE_SECTIONS in tables_ch6.py")
            rows.append(cited(label))
            reproduced = computed(method, "reproduced")
            if reproduced is not None:
                rows.append(reproduced)
            paired_labels.add(label)
            paired_methods.add(method)
        for kind, methods in (("reproduced", section["replicated"]), ("this work", LITERATURE_OURS)):
            for method in methods:
                if method not in paired_methods and (row := computed(method, kind)) is not None:
                    rows.append(row)
        context = [label for label in published if label not in paired_labels]
        if context:
            rows.append(r"\cmidrule{1-7}")
            rows += [cited(label) for label in context]
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_literature.tex"),
        caption=r"\textbf{Published and reproduced results on GSC+.}",
        label="tab:ch6-literature", placement="t",
        header=["System", "Source", "Unit", "Documents", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$"],
        rows=rows, colspec="@{}lllcccc@{}", generator=GENERATOR,
        preamble=[r"\small", r"\setlength{\tabcolsep}{4pt}"],
        # The paper's own per-document means sit in the replication table's note, and the local
        # AutoPCR linker is named in the text, so neither is repeated here.
        notes=(r"$^\dagger$~Recomputed from the counts the paper prints. "
               r"$^\ddagger$~Mention level, as published."))


# ── T5 ──────────────────────────────────────────────────────────────────────────────────────
def t5_scoring(results, texdir):
    """The appendix reconciliation: why this chapter's numbers differ from the ground-truth build's."""
    df = C.table(results, "t1_scoring_policy")
    rows = []
    for ci, cohort in enumerate(C.cohorts_present(df)):
        sub = df[df["cohort"] == cohort]
        by = {r["method"]: r for _, r in sub.iterrows()}
        if ci:
            rows.append(r"\midrule")
        n_pairs = int(sub.iloc[0]["n_gold_pairs"])
        n_scored = int(sub.iloc[0]["n_gold_scored"])
        rows.append(r"\multicolumn{5}{l}{\emph{%s} \footnotesize(%d gold pairs, %d scored)}"
                    % (C.COHORT_HEADER[cohort], n_pairs, n_scored))
        for method in C.methods_present(df):
            row = by.get(method)
            rows.append([
                C.METHOD_LABEL_TEX[method],
                _val(row, "micro_f1_normalised"),
                _val(row, "micro_f1_raw_strings"),
                (rf"$+{C.fmt(row['delta'])}$" if row is not None
                 and C.fmt(row["delta"]) != "--" else C.MISSING_TEX),
                (str(int(row["n_pred_out_of_subtree"])) if row is not None else C.MISSING_TEX),
            ])
    notes = (
        r"Normalised: both gold and predictions restricted to the phenotypic-abnormality subtree, "
        r"the convention of every other table in this chapter. Raw: no restriction, the "
        r"convention of the earlier exp13\_07 / exp13\_18 tables. $\Delta$: normalised minus raw. "
        r"Last column: predicted terms outside the subtree, which count as false positives only "
        r"under raw scoring. " + C.MISSING_NOTE)
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_scoring.tex"),
        caption=(r"\textbf{$F_{1,\mu}$ under the two scoring conventions}, for every system on "
                 r"each cohort."),
        label="tab:ch6-scoring",
        header=["System", r"$F_{1,\mu}$ norm.", r"$F_{1,\mu}$ raw", r"$\Delta$",
                r"out-of-subtree pred."],
        rows=rows, colspec="l rr r r", notes=notes, generator=GENERATOR)


#: A cost the run did not record.
COST_MISSING = "n/a"


# ── T6 ──────────────────────────────────────────────────────────────────────────────────────
def t6_cost(results, texdir):
    r"""Runtime, model calls, peak memory and setup.

    Two things this table must not let a reader do.

    **Compare a cost to the experiment that scored it.** PhenoJury and TreePhenoRAG are scored from
    CPU re-runs of minutes. What they cost is the generation and cache-building runs, which is why
    `cost_experiment` is printed, not assumed.

    The tree's calls look implausibly alike on HCY and GSC+ (about 128 000 per document) and are
    not: the run that stored the scores visits every one of the 18 354 terms and scores each on
    min(S, segments) segments, so a document's count is 18 354 x its capped segment count. The
    medians are that -- 18 354 x 7 = 128 478 on GSC+, 18 354 x 9 = 165 186 on HCY -- and
    the dagger note says so, not leaving the coincidence to the reader.

    **Read a number without its scope.** A row whose run reused a completed CNN output (PhenoBERT's
    scored HCY run did) times the linking pass alone -- detected from its log, not assumed -- and
    is marked as such. HCY's PhenoBERT cost now comes from a dedicated uncached rerun, so both of
    its rows are the full pipeline. The mark stays for any future reused run.
    """
    df = C.table(results, "t4_cost")
    declared = C.declared_model_versions()
    # A method with no run on a cohort gets no cost row there. t4_cost.csv carries one row per
    # (method, cohort) regardless, and the tree's 206-abstract row printed the 114-abstract cache
    # build's seconds under a frame the tree was never evaluated on (tab:ch6-overall says so).
    overall = C.table(results, "t1_overall")
    not_run = {(r["method"], r["cohort"]) for _, r in overall.iterrows()
               if str(r.get("status", "")).startswith("not applicable")}
    rows, flagged = [], []
    # Whether the jury's summed wall clock survived the comparison's guards. It does not always: the
    # sum is withheld when a juror resumed from cached generations, because its timing file then
    # covers only the reports it had left. The note has to say which of the two happened rather
    # than assert one, or the table can end up explaining a number that is not printed.
    jury = [r for _, r in df.iterrows() if str(r.get("method", "")).startswith("phenojury")]
    sequential = any(C.fmt(r.get("seconds_per_report"), 2) != "--" for r in jury)
    bounded = any(str(r.get("seconds_is_lower_bound", "")).lower() in ("true", "1") for r in jury)
    # The tree's row is the cache build's seconds AND calls (one run, as the note promises). The
    # re-run configuration's calls are a different run's number and go in a footnote.
    has_op_calls = ("operating_point_calls_per_report" in df.columns
                    and df["operating_point_calls_per_report"].notna().any())
    tree_op_calls = None
    for ci, cohort in enumerate(C.cohorts_present(df)):
        by = {r["method"]: r for _, r in df[df["cohort"] == cohort].iterrows()}
        if ci:
            rows.append(r"\midrule")
        rows.append(r"\multicolumn{5}{l}{\emph{%s}}" % C.COHORT_HEADER[cohort])
        for method in C.methods_present(df):
            if (method, cohort) in not_run:
                continue
            row = by.get(method)
            # The flag is the collector's own boolean, never a substring of the scope prose:
            # matching on "only" flagged every row whose scope merely said "CPU-only".
            partial = bool(row is not None and str(row.get("scope_partial", "")).lower()
                           in ("true", "1"))
            if partial:
                flagged.append(f"{C.COHORT_SHORT[cohort]}/{C.METHOD_LABEL[method]}")
            # Bare numbers: the lower-bound and approximation flags are named in the note, not
            # printed as $\geq$ / $\approx$ beside the value.
            seconds = _val(row, "seconds_per_report", 2)
            op_calls = (row.get("operating_point_calls_per_report")
                        if row is not None and "operating_point_calls_per_report" in row else None)
            if op_calls is not None and op_calls == op_calls and op_calls != "":
                tree_op_calls = float(op_calls)
            cells = [
                ours_gap(THESIS_METHOD_LABEL[method], method)
                + (r"$^{\ddagger}$" if partial else "")
                + (r"$^{\dagger}$" if method == "treephenorag" and has_op_calls else ""),
                seconds,
                _val(row, "llm_calls_per_report", 1),
                _val(row, "peak_gpu_gb", 1),
                _breakable(declared.get(method)
                           or (row.get("checkpoint", "") if row is not None else "")),
            ]
            # "n/a", not the em dash the other tables use: in this table's narrow columns a dash
            # reads as a rule, and the thesis's style bans it as a placeholder.
            rows.append([COST_MISSING if c == C.MISSING_TEX else c for c in cells])
    notes = (
        r"s / rep.: wall-clock seconds per report, with model loading included only for PhenoBERT "
        r"and PhenoJury. calls: model generations per report. GPU: peak GPU memory in GB. "
        + (r"PhenoJury's seconds are summed over its jurors"
           + (r" and are a lower bound for the full pool on HCY. " if bounded else r". ")
           if sequential else
           r"PhenoJury's seconds are withheld because a juror resumed from cached generations. ")
        + ((r"$^{\dagger}$ Run that scores every term. The selected setting makes "
            + f"{tree_op_calls:,.0f}".replace(",", r"\,")
            + r" verifier calls per HCY report (\Cref{tab:tpr-ladder}). ")
           if tree_op_calls is not None else "")
        + COST_MISSING + r": not recorded."
        + (r" $^{\ddagger}$ Linking pass only (" + C.tex_escape(", ".join(flagged))
           + r")." if flagged else ""))
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_cost_detail.tex"),
        caption=r"\textbf{Cost per report} on each corpus.",
        label="tab:ch6-cost-detail",
        header=["System", "s / rep.", "calls", "GPU", "Model details"],
        rows=rows, colspec=r"l rrr p{2.1in}", notes=notes, generator=GENERATOR,
        compact=True)


#: Row labels of T6.3, keyed on the comparison's machine-readable ``buckets`` column. The prose
#: ``bucket`` column of runs before 2026-09-24 called the first group "never scored", which is
#: false of its retrieval half. Keying on the bucket ids keeps older CSVs rendering correctly.
COMPLEMENTARITY_LABEL = {
    "pruning+retrieval": "Evidence never read (pruning, retrieval)",
    "judgement+pooling": "Evidence read, term rejected (judgement, pooling)",
}

#: Which PhenoJury configuration the recovery is counted for, as the caption names it. The comparison
#: intersects with the ``phenojury`` roster row (run.py, T6.3), the full pool.
RECOVERED_BY_PHRASE = {"phenojury": "the full pool of eight jurors",
                       "phenojury_core3": "the core-3 jury"}


# ── T6.3 ────────────────────────────────────────────────────────────────────────────────────
def t63_complementarity(results, texdir):
    r"""The appendix's tab:app-complementarity: do the two architectures fail on the same terms?

    Two numbers, and they are not symmetric. TreePhenoRAG's recall decomposition
    (chapter 4) splits its false negatives by the stage that lost them, and the split counts here:

      * the terms it lost to **pruning or retrieval** are ones whose evidence the verifier never
        read -- a pruned term was never scored, a retrieval miss was scored on segments without the
        evidence -- so no threshold anywhere in its configuration space could have recovered them;
      * the terms it lost to **judgement or pooling** are ones where it read the evidence and
        rejected the term.

    A method that recovers the first kind supplies evidence the traversal cannot see. One that
    recovers the second disagrees with its verdict. Those are different claims about
    complementarity, and collapsing them into one "overlap" figure would hide which is true. The
    row shares have different denominators and are not each other's complement.

    Set algebra over two prediction files plus chapter 4's per-pair attribution, so pure CPU. A
    cohort with no missed terms at all has not been measured -- the per-pair attribution exists on
    HCY only -- and is named in the caption, not printed as two rows of zeros that read as
    "recovered nothing".
    """
    df = optional_table(results, "t5_complementarity")
    header = ["TreePhenoRAG's losses", "Missed", "Found by PhenoJury", "Share"]
    if df is None:
        C.write_table_tex(
            os.path.join(texdir, "tab_ch6_complementarity.tex"),
            caption=(r"\textbf{Complementarity of PhenoJury and TreePhenoRAG: not yet computed.} "
                     r"Needs \texttt{t5\_complementarity.csv} from exp13\_25 "
                     r"(\texttt{figures/GAPS.md})."),
            label="tab:app-complementarity", placement="h", header=header,
            rows=[["not yet computed", COST_MISSING, COST_MISSING, COST_MISSING]],
            colspec="@{}lccc@{}", preamble=[r"\small"], generator=GENERATOR)
        return

    cohorts_present = [c for c in C.COHORTS if c in set(df["cohort"])] if "cohort" in df \
        else [None]
    measured, unmeasured = [], []
    for cohort in cohorts_present:
        sub = df if cohort is None else df[df["cohort"] == cohort]
        (measured if sub["n_missed"].astype(float).sum() > 0 else unmeasured).append(
            (cohort, sub))

    rows, shares, n_missed = [], [], 0
    for cohort, sub in measured:
        if cohort is not None and len(measured) > 1:
            rows.append(r"\multicolumn{4}{@{}l}{\emph{%s}}" % THESIS_COHORT_SHORT_HEADER[cohort])
        for _, row in sub.iterrows():
            rows.append([
                COMPLEMENTARITY_LABEL.get(str(row.get("buckets", "")),
                                          C.tex_escape(str(row.get("bucket", "")))),
                str(int(row["n_missed"])), str(int(row["n_recovered"])),
                _val(row, "share_recovered"),
            ])
            shares.append(round(float(row["share_recovered"]), 2))
            n_missed += int(row["n_missed"])

    method = (str(df["recovered_by_method"].dropna().iloc[0])
              if "recovered_by_method" in df and df["recovered_by_method"].notna().any()
              else "phenojury")
    where = _join([THESIS_COHORT_SHORT_HEADER[c].split(" (")[0] for c, _ in measured if c]) or "HCY"
    # The claim, read off the printed shares: "about half" only while every share is within 0.10
    # of one half, and "whatever stage lost them" only while the stages agree to within 0.05.
    if shares and all(abs(s - 0.5) <= 0.10 for s in shares):
        amount = "about half of"
    elif shares:
        amount = f"{min(shares):.2f} to {max(shares):.2f} of"
    else:
        amount = "part of"
    same = shares and max(shares) - min(shares) <= 0.05
    title = (f"PhenoJury finds {amount} the annotated {where} terms that TreePhenoRAG misses"
             + (", whatever stage lost them." if same else ", depending on the stage that lost them."))
    unmeasured_names = sorted({THESIS_COHORT_SHORT_HEADER[c].split(" (")[0]
                               for c, _ in unmeasured if c})
    caption = r"\textbf{" + title + r"}"
    who = ("the full PhenoJury pool" if method == "phenojury"
           else RECOVERED_BY_PHRASE.get(method, C.tex_escape(method)))
    notes = (
        r"TreePhenoRAG's " + f"{n_missed:,}".replace(",", r"\,") + r" false negatives on " + where
        + r", split by the stage of \Cref{fig:ch4-recall-decomposition} at which they were lost, "
        r"and how many " + who + r" predicts. Evidence never read: pruning and retrieval losses. "
        r"Evidence read: judgement and pooling losses, where the verifier saw the evidence and "
        r"rejected the term."
        + ((r" Not measured on " + _join(unmeasured_names) + r", where TreePhenoRAG's losses are "
            r"not attributed to stages.") if unmeasured_names else ""))
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_complementarity.tex"),
        caption=caption, notes=notes, label="tab:app-complementarity", placement="h",
        header=header, rows=rows, colspec="@{}lccc@{}", preamble=[r"\small"], generator=GENERATOR)


def optional_table(results, name):
    """The table, or None when the comparison has not written it yet."""
    import pandas as pd
    path = os.path.join(results, "tables", f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


# The three specified artifacts first, then the supporting tables they refer out to.
# ── T7 ──────────────────────────────────────────────────────────────────────────────────────
#: comparison/replication.py's conditions -> the row label, the one thing each changes from the row above.
REPLICATION_LABEL = {
    "current": "Our run (\\Cref{tab:ch6-overall-comparison})",
    "published_code": "$\\to$ old code",
    "published_backend": "$\\to$ original model",
    "published_prompt": "$\\to$ paper prompt",
}
REPLICATION_BACKEND = {
    "Llama-3.3-70B-Instruct": "Llama-3.3 70B",
    "Llama-3-Groq-70B-Tool-Use": "Llama-3 70B (tool use)",
    "Meta-Llama-3-70B-Instruct": "Llama-3 70B",
}
PUBLISHED_RAGHPO = "RAG-HPO + LLaMA-3 70B"


def t7_raghpo_replication(results, texdir):
    r"""\Cref{tab:app-raghpo-replication}: the published RAG-HPO numbers, approached one change at a
    time (check Q16).

    The published row is read from t1_literature.csv, ours from t6_raghpo_replication.csv. Every
    column is micro: the paper's row is recomputed from the counts it prints (`_published_micro`),
    since its own per-case means match no column of ours -- the F1 it prints is the mean of the
    per-case F1s, not the harmonic mean of the two means this table used to set it against.
    Nothing is marked best: the table is read down the steps.
    """
    rep = C.table(results, "t6_raghpo_replication")
    lit = C.table(results, "t1_literature")
    pub = lit[lit["label"] == PUBLISHED_RAGHPO]
    if pub.empty:
        raise SystemExit(f"t1_literature.csv has no {PUBLISHED_RAGHPO!r} row")
    pub = pub.iloc[0]

    # The first condition IS chapter 6's RAG-HPO 70B row. If they disagree, one of the two is stale.
    overall = C.by_method(C.table(results, "t1_overall"), "gsc_raghpo_ann")
    ours = overall.get("raghpo_70b")
    current = rep[rep["arm"] == "current"]
    if ours is not None and not current.empty and \
            abs(float(ours["micro_f1"]) - float(current.iloc[0]["micro_f1"])) > 1e-4:
        raise SystemExit(f"t6_raghpo_replication.csv's current arm (F1 "
                         f"{float(current.iloc[0]['micro_f1']):.4f}) is not t1_overall.csv's "
                         f"RAG-HPO 70B on GSC+ (114) ({float(ours['micro_f1']):.4f}) -- rerun "
                         f"slurm/comparison.sbatch")

    table_no = re.search(r"Table\s+(\d+)", str(pub["source"]))
    cite = r"\citep[Table~%s]{garcia2025}" % (table_no.group(1) if table_no else "5")
    paper = _published_micro(pub)
    rows = [["Paper", "Llama-3 70B", "--",
             _tv(paper, "precision"), _tv(paper, "recall"), _tv(paper, "f1")],
            r"\midrule"]
    for _, r in rep.iterrows():
        if r["arm"] not in REPLICATION_LABEL:
            raise SystemExit(f"t6_raghpo_replication.csv: unknown arm {r['arm']!r} -- add it to "
                             f"REPLICATION_LABEL in tables_ch6.py")
        rows.append([REPLICATION_LABEL[r["arm"]],
                     REPLICATION_BACKEND.get(str(r["backend"]), C.tex_escape(str(r["backend"]))),
                     "paper" if str(r["prompt"]).startswith("paper") else "repository",
                     _tv(r, "micro_precision"), _tv(r, "micro_recall"),
                     _tci(r, "micro_f1", inline=True)])
    n_gold = int(rep["n_gold_scored"].iloc[0])
    C.write_table_tex(
        os.path.join(texdir, "tab_ch6_raghpo_replication.tex"),
        caption=(r"\textbf{RAG-HPO's published recall is approached only with the published code, "
                 r"backend and prompt, and then at lower precision.} RAG-HPO 70B on GSC+ (114) "
                 r"against the published values~" + cite + r". Each $\to$ step changes one setting "
                 r"and keeps those above it."),
        label="tab:app-raghpo-replication", placement="h",
        header=["Configuration", "Backend", "Prompt", r"$P_\mu$", r"$R_\mu$",
                r"$F_{1,\mu}$ [95\,\% CI]"],
        rows=rows, colspec="@{}lllccc@{}", generator=GENERATOR,
        preamble=[r"\small", r"\setlength{\tabcolsep}{3pt}"],
        notes=(r"Paper: recomputed from the counts in its Table~5, which reports means of "
               r"per-document precision, recall and $F_1$ (" + _published_as_printed(lit) + r"). "
               r"The paper scores all " + f"{int(pub['n_gold_terms']):,}".replace(",", r"\,")
               + r" annotated terms, our runs the " + f"{n_gold:,}".replace(",", r"\,")
               + r" below $v_0$. Prompt: the extraction prompt in the "
               r"repository or the one printed in the paper's supplement. Backends: the "
               r"instruction-tuned checkpoints. Tool use: Groq's Llama-3 70B fine-tune, which "
               r"upstream's client names. All runs local in 8-bit "
               r"precision. The current code decodes greedily, the old code samples at "
               r"temperature 0.2 with a fixed seed."))


# ── T8 ──────────────────────────────────────────────────────────────────────────────────────
# The side-by-side examples. Rows come from apps/compare_ui/export_examples.py, which reads the
# compare_ui bundles on the cluster. The definitions of every cell are in its docstring.

QUAL_METHODS = ("phenobert", "autopcr_70b", "raghpo_70b", "phenojury", "treephenorag")
#: The column heads are chapter 6's own row labels, so a column names the configuration it shows.
QUAL_HEADER = {m: THESIS_METHOD_LABEL[m] for m in QUAL_METHODS}
#: One sign per meaning, the same in both tables: a find whose evidence is in another segment and
#: one whose evidence has no segment are both "found, but not from this segment" to a reader.
QUAL_MARK = {"tp": r"\ding{51}", "tp_elsewhere": r"\ding{51}$^{*}$",
             "tp_unplaced": r"\ding{51}$^{*}$", "fn": "--", "fp": r"\ding{55}", "na": "n/a",
             "": ""}
#: ... and one row marker for a term the document is annotated with, but not on this segment,
#: whether the ground truth evidence locations it on another segment or on none.
QUAL_ELSEWHERE = r"$^{\dagger}$"
QUAL_LEGEND = (r"\ding{51}~found; \ding{51}$^{*}$~found, but not from this segment; --~missed; "
               r"\ding{55}~spurious; $^{\dagger}$~annotated elsewhere in the document.")
QUAL_BLOCK = {"hcy_phenojury": "Chosen while reviewing PhenoJury's errors",
              "hcy_treephenorag": "Chosen while reviewing TreePhenoRAG's errors"}
QUAL_COHORT = {
    "hcy": dict(stem="tab_ch6_qualitative_hcy.tex", label="tab:app-qualitative-hcy",
                name="HCY", unit="report",
                gold=r"the curated HCY gold (\Cref{app:hcy-curation})"),
    "gsc": dict(stem="tab_ch6_qualitative_gsc.tex", label="tab:app-qualitative-gsc",
                name="GSC+ (114)", unit="abstract",
                gold=r"RAG-HPO's re-annotation of 114 abstracts (\Cref{app:gsc})"),
}
#: A float is closed before its estimated body passes this many \small lines, and the rest
#: continues in a second float -- a table that runs off the page is silently clipped by LaTeX.
#: A 9 in text block holds ~59 \small lines. The rotated header (its longest label is
#: "PhenoJury (full pool, 8)") takes ~10 and the rules ~2. The first float also carries the
#: legend, so its budget is this minus the legend's estimated height.
QUAL_LINES_PER_FLOAT = 45
QUAL_TERM_CHARS = 64          # characters per \small line across the term column
QUAL_ARRAYSTRETCH = 1.15      # a little air between rows. The height estimate scales with it
QUAL_NOTE_CHARS = 110         # characters per \footnotesize line of the notes
#: The five mark columns are this wide, so the sentence row can span \linewidth.
QUAL_MARK_WIDTH = "1.4em"
QUAL_COLSPEC = (r"@{}p{\dimexpr\linewidth-7em-10\tabcolsep\relax}" + "c" * 5 + "@{}")

_TEX_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
                "#": r"\#", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
_TEX_UNICODE = {"°": r"\textdegree{}", "µ": r"\textmu{}", "μ": r"\textmu{}",
                "±": r"\textpm{}", "≤": r"$\leq$", "≥": r"$\geq$",
                "×": r"$\times$", "–": "--", "—": "---", "‘": "`",
                "’": "'", "“": "``", "”": "''", " ": "~",
                "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}",
                "→": r"$\rightarrow$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
                "½": r"\textonehalf{}", "…": r"\ldots{}"}


def _tex_text(text: str, where: str) -> str:
    """Report text, escaped for a T1/utf8 preamble. Aborts on a character it cannot set.

    Aborting, not dropping: the HCY sentences are printed verbatim and are not proofread
    by whoever runs this (patient text is not read by a model), so a character LaTeX would choke
    on -- or worse, silently omit -- must stop the build. The message names the code point and the
    document, never the text.
    """
    import unicodedata

    out, bad = [], set()
    for ch in re.sub(r"\s+", " ", str(text)).strip():
        if ch in _TEX_SPECIAL:
            out.append(_TEX_SPECIAL[ch])
        elif ch in _TEX_UNICODE:
            out.append(_TEX_UNICODE[ch])
        elif ord(ch) < 128 or (ord(ch) < 0x180 and unicodedata.category(ch).startswith("L")):
            out.append(ch)
        else:
            bad.add(ch)
    if bad:
        raise SystemExit("{}: no LaTeX mapping for {} -- add it to _TEX_UNICODE in "
                         "tables_ch6.py".format(where, ", ".join(
                             "U+{:04X}".format(ord(c)) for c in sorted(bad))))
    return "".join(out)


def _qual_examples(path):
    """``[(head, [term rows])]`` in the export's order, or None when it has not been exported."""
    import csv

    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    missing = [c for c in ("example_no", "block", "doc_id", "segment_idx", "sentence", "hpo_id",
                           "label", "row_kind", "trigger") + QUAL_METHODS if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing} -- re-export with "
                         f"app/compare_ui/export_examples.py")
    examples = {}
    for r in rows:
        bad = [m for m in QUAL_METHODS if r[m] not in QUAL_MARK]
        if bad:
            raise SystemExit(f"{path}: example {r['example_no']} ({r['doc_id']}) carries marks "
                             f"outside the vocabulary in {bad}")
        head, terms = examples.setdefault(int(r["example_no"]), (r, []))
        if r["hpo_id"]:
            terms.append(r)
    return [examples[n] for n in sorted(examples)]


def _qual_lines(head, terms) -> float:
    """Estimated height of one example in \\small lines: the sentence, its terms, the rule."""
    import math

    # The sentence is set in the term column only, so it wraps at that column's width.
    lines = 1 + math.ceil((len(head["sentence"]) + 2) / QUAL_TERM_CHARS)   # source line + text
    for t in terms or [None]:
        width = 50 if t is None else len(t["label"]) + 12      # \quad label HP:...
        lines += math.ceil(width / QUAL_TERM_CHARS)
    return (lines + 0.7) * QUAL_ARRAYSTRETCH


def _qual_rows(n_shown, head, terms):
    where = f"{head['doc_id']} segment {head['segment_idx']}"
    sentence = _tex_text(head["sentence"], where)
    box = lambda s: r"\makebox[%s]{%s}" % (QUAL_MARK_WIDTH, s)       # noqa: E731
    # The sentence sits in the term column alone -- spanning all six would run it across the
    # mark columns -- with the mark cells left empty beside it.
    # Where it comes from on a line of its own, then the sentence: a reader finds the example by
    # its number and reads the text without a document id breaking into it.
    rows = [[rf"\textbf{{({n_shown})}}\enspace{{\footnotesize\textsf{{"
             rf"{C.tex_escape(head['doc_id'])}}}, segment {int(head['segment_idx'])}}}"
             rf"\newline ``{sentence}''"] + [""] * len(QUAL_METHODS)]
    if not terms:
        rows.append([r"\emph{no term annotated in, or predicted from, this segment}"]
                    + [box("")] * len(QUAL_METHODS))
    for i, t in enumerate(terms):
        # The term alone (its trigger word is in the export, not the table), indented as a block
        # so a wrapped label stays under itself, with the HPO id pushed to the right in grey --
        # The labels read as a list and the ids stay out of the way until they are wanted.
        marker = QUAL_ELSEWHERE if t["row_kind"] in ("gold_elsewhere", "gold_unplaced") else ""
        cell = (r"\leftskip=1em\relax " + C.tex_escape(t["label"]) + marker
                + r"\hfill{\scriptsize\textcolor{gray}{" + C.tex_escape(t["hpo_id"]) + "}}")
        if i == 0:
            cell = r"\addlinespace[2pt]" + cell       # booktabs reads it after the row break
        rows.append([cell] + [box(QUAL_MARK[t[m]]) for m in QUAL_METHODS])
    return rows


def _qual_notes(cohort, examples) -> str:
    """The legend: the same five signs under both tables, so the two read alike."""
    used = {t[m] for _h, terms in examples for t in terms for m in QUAL_METHODS}
    return QUAL_LEGEND + (" n/a~no output." if "na" in used else "")


def t8_qualitative_examples(results, texdir):
    r"""\Cref{tab:app-qualitative-hcy,tab:app-qualitative-gsc}: the five methods side by side on
    single segments chosen during manual error analysis.

    One table per cohort, never merged: the two have different ground truths and different documents.
    The export lives beside the compare_ui bundles (``output/compare_ui_bundles/qualitative/``);
    ``--results``'s parent is the ``output/`` it is looked up under.
    """
    source = os.path.join(os.path.dirname(os.path.abspath(results)), "compare_ui_bundles",
                          "qualitative")
    header = ([r"Segment, then its terms"]
              + [r"\makebox[%s]{\rotatebox{90}{%s}}" % (QUAL_MARK_WIDTH, QUAL_HEADER[m])
                 for m in QUAL_METHODS])
    preamble = [r"\small", r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{%s}" % QUAL_ARRAYSTRETCH]
    for cohort, spec in QUAL_COHORT.items():
        path = os.path.join(source, f"examples_{cohort}.csv")
        examples = _qual_examples(path)
        out = os.path.join(texdir, spec["stem"])
        if examples is None:
            C.write_table_tex(
                out, label=spec["label"], placement="h", header=header, colspec=QUAL_COLSPEC,
                caption=(rf"\textbf{{The five methods on single {spec['name']} segments: not yet "
                         rf"computed.}} Needs \texttt{{examples\_{cohort}.csv}} from "
                         r"\texttt{app/compare\_ui/export\_examples.py} "
                         r"(\texttt{figures/GAPS.md})."),
                rows=[["not yet computed"] + [""] * len(QUAL_METHODS)],
                preamble=preamble, generator=GENERATOR)
            continue

        # Pack whole examples into floats by estimated height. A block header opens each block
        # and is repeated when a float starts in the middle of one.
        import math

        notes = _qual_notes(cohort, examples)
        budget = QUAL_LINES_PER_FLOAT - 3 - math.ceil(len(notes) / QUAL_NOTE_CHARS)
        floats, current, height, last_block = [], [], 0.0, None
        for n, (head, terms) in enumerate(examples, start=1):
            need = _qual_lines(head, terms)
            if current and height + need > budget:
                budget = QUAL_LINES_PER_FLOAT - 2
                floats.append((current, height))
                current, height, last_block = [], 0.0, None
            block = QUAL_BLOCK.get(head["block"])
            if block and block != last_block:
                if current:
                    current.append(r"\midrule")
                current.append(r"\multicolumn{6}{@{}l}{\emph{%s}}" % block)
                height += 1
            elif current:
                current.append(r"\midrule")
            last_block = block
            current += _qual_rows(n, head, terms)
            height += need
        floats.append((current, height))

        title = rf"The five methods on hand-picked {spec['name']} segments"
        parts = []
        for i, (rows, lines) in enumerate(floats):
            first = i == 0
            parts.append(C.render_table_tex(
                caption=(rf"\textbf{{{title}.}}" if first else
                         rf"\textbf{{{title} (continued).}}"),
                # A float over half a page cannot sit "here". It goes to a page of its own.
                label=spec["label"] if first else "",
                placement="p" if lines > QUAL_LINES_PER_FLOAT / 2 - 5 else "htbp",
                header=header, rows=rows, colspec=QUAL_COLSPEC,
                notes=notes if first else None,
                preamble=preamble, generator=GENERATOR, file_header=first))
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(parts))
        print(f"[tables] {spec['stem']}  ({len(examples)} examples, {len(floats)} float(s))")


ALL = [
    t61_overall_comparison,     # T6.1  micro and macro, one row per method per cohort
    t62_decisive_subset,        # T6.2  the decisive table
    t63_complementarity,        # T6.3  do they fail on the same terms?
    t6_overall_macro,           #       macro conventions, split out of T6.1
    t2_hierarchy,               #       hF and CoPHE
    t4_literature,              #       published figures and their comparability
    t5_scoring,                 #       appendix: the two scoring conventions reconciled
    t6_cost,                    #       cost in detail, with the run that paid it
    t7_raghpo_replication,      #       appendix: the published RAG-HPO numbers, one change at a time
    t8_qualitative_examples,    #       appendix: the five methods side by side on single segments
]


def main():
    """Build every chapter 6 table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.DEFAULT_RESULTS)
    ap.add_argument("--texdir", default=C.DEFAULT_TEXDIR)
    ap.add_argument("--only", action="append", choices=[fn.__name__ for fn in ALL],
                    help="build only this table (repeatable); default is all of them")
    args = ap.parse_args()
    os.makedirs(args.texdir, exist_ok=True)
    for fn in ALL:
        if args.only and fn.__name__ not in args.only:
            continue
        try:
            fn(args.results, args.texdir)
        except SystemExit as exc:
            # ch{5,6}_common.table() already printed the path and the reason to stderr and exits
            # with MISSING_SOURCE_EXIT. Anything else carries its own message.
            detail = ("source not available -- see the line above"
                      if exc.code == C.MISSING_SOURCE_EXIT else exc)
            print(f"[tables] skipped {fn.__name__}: {detail}")


if __name__ == "__main__":
    main()
