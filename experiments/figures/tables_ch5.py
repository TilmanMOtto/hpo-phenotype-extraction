#!/usr/bin/env python
r"""Every chapter-5 (PhenoJury) table, as a standalone booktabs `table` environment.

One function per specified artifact, named for what the table *carries*:

    T5.0  main_comparison             PhenoJury against the 8B baselines and PhenoBERT
    T5.1  union_recall_gate           the gate: can the pool reach the annotated term at all
    T5.2  jury_ablation               the pre-specified comparison and its test family
    T5.3  selection_frequency         whether "the protocol selected this jury" is a finding
    T5.4  model_prompt_interaction    the 8x4 matrix, and why one global prompt is defensible
    T5.5  prompt_sensitivity          the full pool under each prompt: how much the vote absorbs
    T5.6  normaliser_exchange         the same generations read by three normalisers
    T5.7  leave_one_out               per-juror quality and what removing each juror costs

The recall decomposition is a figure (fig_ch5_recall_decomposition.py), not a table.

Generated rather than typed so the chapter and the results cannot drift: every number here is read
out of `output/phenojury_protocol/` at build time. Re-run after any re-run of the PhenoJury protocol.
T5.0 also reads the baselines' rows from `output/comparison/` (`--comparison`).

A table whose CSV does not exist yet prints the rows it *can* and states, in its own notes, which
block is missing and what would produce it -- never a blank cell a reader would take for a zero.
See docs/thesis_map.md.

Requires \usepackage{booktabs}.

    python experiments/figures/tables_ch5.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C                       # noqa: E402

GENERATOR = "figures/thesis_figures_scripts/tables_ch5.py"

NUMBER_WORDS = {3: "three", 8: "eight"}

CONFIG_LABEL = {
    "phenobert": r"PhenoBERT alone ($J_0$)",
    "full_pool": f"Full pool ({len(C.MODEL_ORDER)} jurors)",
    "phenojury": "PhenoJury (selected jury)",
    "fixed:core3": "Core-3 jury (exploratory)",
}

THESIS_NORMALISER_LABEL = C.NORMALISER_SHORT

#: The block header of tab:pj-main, which names the report count. The others use COHORT_LABEL.
MAIN_COHORT_LABEL = {"hcy": "HCY ({n} clinical reports)", "gsc206": "GSC+ ({n} abstracts)"}


#: The cost-constrained condition, in `s7_headline.csv`'s own spelling. It is reported here because the
#: deployment claim -- three of the eight jurors carry nearly the whole effect -- is invisible in a
#: table that shows only the pool and the selected jury.
#:
#: It is NOT in the fixed in advance test family and must not be moved into it: the three models were
#: named after reading the selection frequencies in S5, so a Holm adjustment computed over a family
#: it belongs to would be arithmetic dressed as inference. The PhenoJury protocol scores it in
#: `s7_exploratory_fixed_jury.csv` with uncorrected p-values, and the note says so.
CORE3_CONFIG = "fixed:core3"

GAP_NOTE = r"\emph{Awaiting a source} (\texttt{figures/GAPS.md}): "


def _escape_prose(text: str) -> str:
    r"""Escape _ character characters in gap prose, leaving any ``$...$`` span alone.

    Gap reasons name CSV files and experiment ids -- ``s6_lomo.csv``, ``phenojury_protocol`` -- and a bare
    ``_`` in text mode is a hard LaTeX error, not a cosmetic one: it aborts the compile and TeX's
    recovery swallows the rest of the note. A blanket escape is wrong too, because some reasons
    carry real maths (``$J_0$``), so the maths spans are stepped over.
    """
    out = []
    for span in re.split(r"(\$[^$]*\$)", text):
        out.append(span if span.startswith("$") else span.replace("_", r"\_"))
    return "".join(out)


def gap_sentence(gaps: list[str]) -> str:
    r"""The gap notes as one sentence, with identical per-cohort reasons collapsed.

    Both cohorts usually lack the same CSV for the same reason, and printing that reason twice
    verbatim makes one missing table look like two different problems. Each entry is expected to
    read ``"<cohort>: <reason>"``.
    """
    if not gaps:
        return ""
    grouped: dict[str, list[str]] = {}
    for gap in gaps:
        cohort, _, reason = gap.partition(": ")
        if cohort not in grouped.setdefault(reason, []):
            grouped[reason].append(cohort)
    clauses = [f"{', '.join(cohorts)}: {_escape_prose(reason)}"
               for reason, cohorts in grouped.items()]
    return r" \\ " + GAP_NOTE + ". ".join(clauses) + "."


def gap_note(gaps: list[str]) -> str:
    """The gap sentence as a table's only note: without the line break that follows a caption."""
    return gap_sentence(gaps)[len(r" \\ "):] if gaps else ""


def optional(results, cohort, name):
    """The table, or None when the experiment has not written it yet."""
    path = os.path.join(results, cohort, "tables", f"{name}.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    return None if df.empty else df


def cohort_header(rows, cohort, ncols, first, label=None):
    """Append a cohort block header to the table rows."""
    if not first:
        rows.append(r"\midrule")
    rows.append(r"\multicolumn{%d}{@{}l}{\emph{%s}}" % (ncols, label or C.COHORT_LABEL[cohort]))


def interval(row, key="micro_f1"):
    """``[lo, hi]`` of a metric, at the printed precision."""
    return f"[{C.fmt(row[f'{key}_lo'])}, {C.fmt(row[f'{key}_hi'])}]"


# -- T5.1 the gate ------------------------------------------------------------

#: In the gate table the candidate reader is simply "PhenoBERT": the raw reader is not listed there,
#: so there is nothing to disambiguate. Other tables keep ch5_common's own labels.
UNION_NORMALISER_LABEL = {"phenobert_candidates": "PhenoBERT"}

def t51_union_recall_gate(results, cohorts, texdir):
    """Write ``tab_ch5_union_recall_gate.tex``."""
    rows = []
    for ci, cohort in enumerate(cohorts):
        ceil = C.table(results, cohort, "s1_ceilings")
        by = {str(r["ceiling"]): r for _, r in ceil.iterrows()}
        cohort_header(rows, cohort, 5, ci == 0)

        # Block 1 -- one row per normaliser, union over every prompt and model. The reference
        # reader (PhenoBERT raw) is not a selectable normaliser and is not listed.
        rows.append(r"\multicolumn{5}{l}{\quad per normaliser, union over all 32 readers}")
        for norm in C.NORMALISER_ORDER:
            if norm == C.REFERENCE_NORMALISER:
                continue
            key = next((k for k in by if k.endswith(norm)), None)
            if key is None:
                continue
            r = by[key]
            rows.append([r"\quad " + C.tex_escape(UNION_NORMALISER_LABEL.get(
                             norm, C.NORMALISER_LABEL[norm])),
                         C.fmt(r["micro_recall"]), C.fmt(r["micro_precision"]),
                         C.fmt(r["micro_f1"]), C.fmt(r["preds_per_report"], 1)])

        # Block 2 -- one row per prompt, union over every model and reader.
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{5}{l}{\quad per prompt, union over all 24 readers}")
        for prompt in C.PROMPT_ORDER:
            key = next((k for k in by if f"prompt {prompt}," in k), None)
            if key is None:
                continue
            r = by[key]
            rows.append([r"\quad " + C.tex_escape(C.PROMPT_LABEL[prompt]),
                         C.fmt(r["micro_recall"]), C.fmt(r["micro_precision"]),
                         C.fmt(r["micro_f1"]), C.fmt(r["preds_per_report"], 1)])

        # Block 3 -- one juror against eight.
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{5}{l}{\quad best single juror against the 8-model union}")
        head = C.table(results, cohort, "s7_headline")
        singles = [(k, v) for k, v in
                   ((str(r["config"]), r) for _, r in head.iterrows()) if k.startswith("single:")]
        if singles:
            name, r = max(singles, key=lambda kv: float(kv[1]["micro_recall"]))
            model = name.split(":", 1)[1]
            rows.append([r"\quad best single (" +
                         C.tex_escape(C.MODEL_LABEL.get(model, model)) + ")",
                         C.fmt(r["micro_recall"]), C.fmt(r["micro_precision"]),
                         C.fmt(r["micro_f1"]), C.fmt(r["preds_per_report"], 1)])
        curves = C.table(results, cohort, "s3_curves")
        k1 = curves[curves["k"] == 1]
        if not k1.empty:
            r = k1.iloc[0]
            rows.append([r"\quad 8-model union, $k=1$",
                         C.fmt(r["micro_recall"]), C.fmt(r["micro_precision"]),
                         C.fmt(r["micro_f1"]), C.fmt(r["preds_per_report"], 1)])

    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_union_recall_gate.tex"),
        caption=(r"\textbf{Union recall at $k=1$: the highest recall any vote over the pool can "
                 r"reach.} Micro recall, precision and $F_1$ of the union of all jurors' "
                 r"candidates (a term counts if any one juror proposes it), grouped by normaliser "
                 r"and by prompt, and for the best single juror against the 8-model union. "
                 r"Preds/rep.: predicted terms per report."),
        label="tab:ch5-union-recall-gate",
        header=["Arm", r"$R_\mu$", r"$P_\mu$", r"$F_{1,\mu}$", "preds/rep."],
        colspec="lcccc",
        rows=rows, generator=GENERATOR)


# -- T5.2 the main comparison -------------------------------------------------

# ── T5.0 the chapter's comparison with the baselines ───────────────────────────────────────

#: The PhenoJury protocol's cohort directories, under the names the comparison scores them as.
COMPARISON_COHORT = {"hcy": "hcy", "gsc206": "gsc_2024_eval_206"}
#: The two PhenoJury rows, from the PhenoJury protocol's `s7_headline.csv`, under the comparison's method keys.
JURY_ROWS = {"full_pool": "phenojury", CORE3_CONFIG: "phenojury_core3"}
#: The baselines, from the comparison: the smaller size of each LLM baseline, and PhenoBERT.
CH5_BASELINES = ["raghpo_8b", "autopcr_8b", "phenobert"]
#: Two runs that score the same predictions against the same ground truth agree to rounding.
AGREEMENT_TOL = 5e-4
SCORE_COLUMNS = ("micro_precision", "micro_recall", "micro_f1", "micro_f1_lo", "micro_f1_hi",
                 "macro_precision", "macro_recall", "macro_f1")


def t50_main_comparison(results, cohorts, texdir, comparison=None):
    r"""\Cref{tab:ch5-main-comparison}: PhenoJury against the baselines, in the ch6 layout.

    Two sources, joined on one row. The PhenoJury rows are the PhenoJury protocol's own (`s7_headline.csv`, the
    prompt selected per corpus), so they are the numbers the jury ablation below this table
    prints. The baselines come from the comparison, which scores every system against the same ground truth.
    Both runs score PhenoBERT, and that row must agree or the two are not the same frame -- a
    disagreement is a crash, never a skip.

    The comparison's own PhenoJury rows are compared too. They are what tab:ch6-overall-comparison
    prints, and a disagreement means chapter 6 is quoting a different system: it is reported,
    not fatal, because the fix is a cluster re-run (slurm/comparison_inputs.sbatch, then
    the comparison), not a change here.
    """
    import pandas as pd
    import ch6_common as C6
    import tables_ch6 as T6

    overall = C6.table(comparison or C6.DEFAULT_RESULTS, "t1_overall")
    frames, stale, core3 = [], [], []
    for cohort in cohorts:
        key = COMPARISON_COHORT[cohort]
        head = C.table(results, cohort, "s7_headline").set_index("config")
        scored = C6.by_method(overall, key)

        pb = scored.get("phenobert")
        if pb is None or "phenobert" not in head.index:
            raise SystemExit(f"{key}: PhenoBERT is not scored by both exp13_25 and exp14_04, so "
                             f"the two sources cannot be shown to share a frame")
        if abs(float(pb["micro_f1"]) - float(head.loc["phenobert", "micro_f1"])) > AGREEMENT_TOL:
            raise RuntimeError(
                f"{key}: PhenoBERT micro F1 is {float(pb['micro_f1']):.4f} in exp13_25 but "
                f"{float(head.loc['phenobert', 'micro_f1']):.4f} in exp14_04 -- different gold "
                f"or documents; the baselines cannot be printed beside the jury")

        sub = overall[(overall["cohort"] == key) & overall["method"].isin(CH5_BASELINES)]
        frames.append(sub)
        for config, method in JURY_ROWS.items():
            if config not in head.index:
                continue
            r = head.loc[config]
            frames.append(pd.DataFrame([{"method": method, "cohort": key, "status": "ok",
                                         **{c: r[c] for c in SCORE_COLUMNS}}]))
            theirs = scored.get(method)
            if theirs is not None and abs(float(theirs["micro_f1"]) - float(r["micro_f1"])) \
                    > AGREEMENT_TOL:
                stale.append(f"{key}/{method}: exp13_25 {float(theirs['micro_f1']):.4f}, "
                             f"exp14_04 {float(r['micro_f1']):.4f}")
        if CORE3_CONFIG in head.index and not core3:
            found = re.search(r"\(([a-z0-9_+]+)\)", str(head.loc[CORE3_CONFIG].get("note", "")))
            core3 = found.group(1).split("+") if found else []

    if stale:
        print("[tables] WARNING: tab:ch6-overall-comparison quotes a different PhenoJury than "
              "chapter 5 (" + "; ".join(stale) + "). Re-run slurm/comparison_inputs.sbatch "
              "and slurm/comparison.sbatch, then tables_ch6.py.", file=sys.stderr)

    df = pd.concat(frames, ignore_index=True)
    rows = T6.overall_rows(df, cohorts=[COMPARISON_COHORT[c] for c in cohorts],
                           methods=CH5_BASELINES + list(JURY_ROWS.values()))
    members = T6._join([C.MODEL_LABEL.get(m, m) for m in core3]) if core3 else "three jurors"
    caption = r"\textbf{" + T6.main_result_sentence(df) + r"}"
    notes = (
        r"PhenoJury: pooled "
        r"out-of-fold predictions of nested cross-validation. Core-3: " + members + r".")
    T6.write_overall_table(os.path.join(texdir, "tab_ch5_main_comparison.tex"), rows=rows,
                           caption=caption, notes=notes, label="tab:ch5-main-comparison",
                           generator=GENERATOR)


def t52_jury_ablation(results, cohorts, texdir):
    r"""The chapter's tab:pj-main: no language model, one model, three models, eight.

    ## What moved out of the body, and where it went

    The pre-specified test family used to be printed as two extra rows per cohort, with the
    p-values sitting in the columns that hold $F_1$ and its interval everywhere else. Four rows
    whose numbers mean something different from the column they are under is the density
    this table was asked to lose, so the family is now a **sentence in the caption**. Nothing is
    dropped: both comparisons, both p-values and the pre-specification are still stated.

    ## What moved in

    **Macro P/R/$F_1$**, beside micro. HCY is imbalanced enough that the two averagings can
    disagree about an ordering, and the overall comparison in chapter 6 reports both -- a condition
    table that reports only micro cannot be read against it.

    **The core-3 condition.** The ladder the discussion wants is PhenoBERT -> best single juror -> a
    small fixed jury -> the whole pool, because that is the sequence a reader deciding what to
    deploy walks down. Without the third rung the table answers "is a jury worth it" and not "how
    much of a jury is worth it", and the second question is the one with a cost attached.
    """
    rows, gaps, printed = [], [], {}
    for ci, cohort in enumerate(cohorts):
        head = C.table(results, cohort, "s7_headline")
        by = {str(r["config"]): r for _, r in head.iterrows()}
        n_reports = C.manifest(results, cohort).get("n_reports")
        cohort_header(rows, cohort, 8, ci == 0, label=MAIN_COHORT_LABEL[cohort].format(n=n_reports))
        block_start = len(rows)
        printed[cohort] = {}

        def arm(key, label, r):
            """One scored row, or a stated absence with the same number of columns."""
            if r is None:
                rows.append([label] + ["--"] * 7)
                return
            printed[cohort][key] = float(C.fmt(r["micro_f1"]))
            rows.append([label, C.fmt(r["micro_precision"]), C.fmt(r["micro_recall"]),
                         C.fmt(r["micro_f1"]), interval(r),
                         C.fmt(r.get("macro_precision")), C.fmt(r.get("macro_recall")),
                         C.fmt(r.get("macro_f1"))])

        # J0 is PhenoBERT read off the report text itself, with no language model in the loop.
        # It is the row that decides whether the whole chapter bought anything.
        arm("phenobert", CONFIG_LABEL["phenobert"], by.get("phenobert"))
        if "phenobert" not in by:
            gaps.append(f"{C.COHORT_SHORT[cohort]}: $J_0$ (PhenoBERT on the report text) is not "
                        "scored in s7_headline.csv; exp14_04 must score exp13_08's predictions on "
                        "the same folds")

        singles = [(k, v) for k, v in by.items() if k.startswith("single:")]
        if singles:
            name, r = max(singles, key=lambda kv: float(kv[1]["micro_f1"]))
            model = name.split(":", 1)[1]
            arm("single", f"Best single juror ({C.tex_escape(C.MODEL_LABEL.get(model, model))})", r)
        # Smallest jury first: the row order IS the ladder, and a reader should be able to read
        # down it and watch what each added model buys.
        for key in (CORE3_CONFIG, "full_pool", "phenojury"):
            if key in by:
                arm(key, CONFIG_LABEL[key], by[key])
        if CORE3_CONFIG not in by:
            gaps.append(f"{C.COHORT_SHORT[cohort]}: the fixed core-3 jury is not in "
                        "s7_headline.csv; exp14_04 must be run with "
                        "+fixed_juries.core3=[medgemma,medpsy,phi4]")

        # Marked within the cohort block only -- HCY and GSC+ are separate contests.
        block = rows[block_start:]
        C.mark_best_second(block, [1, 2, 3, 5, 6, 7])
        rows[block_start:] = block

    caption = r"\textbf{" + _ablation_title(printed, cohorts) + r"}"
    notes = (
        _family_sentence(results, cohorts).rstrip()
        + (" " + gap_note(gaps) if gaps else ""))
    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_jury_ablation.tex"),
        caption=caption, label="tab:pj-main", placement="t",
        header=["Configuration", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$", r"95\,\% CI",
                "$P_M$", "$R_M$", "$F_{1,M}$"],
        colspec="@{}lccccccc@{}", preamble=[r"\small"],
        rows=rows, notes=notes, generator=GENERATOR)


#: What each cohort's documents are called in a caption's claim.
COHORT_READING = {"hcy": "the clinical reports", "gsc206": "the abstracts"}


def _ablation_title(printed: dict, cohorts) -> str:
    """The caption's claim, read off the printed micro F1 of each cohort's block.

    Two comparisons, the two the chapter argues from: the selected jury against its best single
    juror (the jury effect), and against PhenoBERT alone (whether the language model pays).
    """
    def beats(ref):
        return [c for c in cohorts
                if "phenojury" in printed[c] and ref in printed[c]
                and printed[c]["phenojury"] > printed[c][ref]]

    def clause(subject, ref_phrase, wins):
        if len(wins) == len(cohorts):
            both = "both cohorts" if len(cohorts) == 2 else "every cohort"
            return f"{subject} beats {ref_phrase} on {both}."
        if not wins:
            either = "either cohort" if len(cohorts) == 2 else "any cohort"
            return f"{subject} does not beat {ref_phrase} on {either}."
        losses = [c for c in cohorts if c not in wins]
        return (f"{subject} beats {ref_phrase} on "
                + " and ".join(COHORT_READING[c] for c in wins) + " but not on "
                + " and ".join(COHORT_READING[c] for c in losses) + ".")

    jury, bert = beats("single"), beats("phenobert")
    # One sentence where the jury wins everywhere, which is the claim the chapter argues from.
    if len(jury) == len(cohorts) and bert and len(bert) < len(cohorts):
        both = "both cohorts" if len(cohorts) == 2 else "every cohort"
        return (f"The jury beats its best single juror on {both}, and PhenoBERT alone only on "
                + " and ".join(COHORT_READING[c] for c in bert) + ".")
    return " ".join([clause("The jury", "its best single juror", jury),
                     clause("It", "PhenoBERT alone", bert)])


#: The pre-specified comparisons as the caption words them. The first cohort names each one. The
#: others give the p-values in the same order.
COMPARISON_TEXT = {"phenojury vs best_single_juror": "selected against best single juror",
                   "phenojury vs full_pool": "against full pool"}


def _family_sentence(results, cohorts) -> str:
    r"""The pre-specified test family, as prose, not as rows in the table body.

    It used to be two rows per cohort whose last two columns held p-values while the same columns
    held $F_1$ everywhere else. That is a different quantity under the same heading, which is the
    one thing a results table must not do.
    """
    def p_text(p: float) -> str:
        return r"<0.001" if p < 0.001 else f"={p:.3f}"

    parts = []
    for cohort in cohorts:
        try:
            sig = C.table(results, cohort, "s7_significance")
        except SystemExit:
            continue
        tests = [(str(r["comparison"]), float(r["p_holm"])) for _, r in sig.iterrows()]
        if not parts:
            named = [f"{COMPARISON_TEXT.get(name, C.tex_escape(name))} "
                     rf"$p_{{\mathrm{{Holm}}}}{p_text(p)}$" for name, p in tests]
            parts.append(f"{C.COHORT_SHORT[cohort]}, " + ", ".join(named) + ".")
        else:
            values = [rf"$p_{{\mathrm{{Holm}}}}{p_text(tests[0][1])}$"] + \
                     [f"${p_text(p).lstrip('=')}$" for _, p in tests[1:]]
            parts.append(f"{C.COHORT_SHORT[cohort]}, "
                         + (", ".join(values[:-1]) + " and " + values[-1]
                            if len(values) > 1 else values[0]) + ".")
    if not parts:
        return ""
    return (r"Paired randomisation tests, Holm-corrected within cohort: " + " ".join(parts) + " ")


# -- T5.3 is the jury a finding? ----------------------------------------------

#: The appendix's tab:app-pj-stability, as groups of axes separated by a rule, in its row order.
STABILITY_GROUPS = [
    [("prompt", "Prompt", lambda v: C.PROMPT_LABEL.get(v, v)),
     ("normaliser", "Normaliser", lambda v: THESIS_NORMALISER_LABEL.get(v, v))],
    [("juror", "Juror", lambda v: C.MODEL_LABEL.get(v, v))],
    [("rule", "Rule", str), ("unit", "Scope", str), ("k", "$k$", str)],
]
#: A value chosen in at least this many of the outer evaluations is printed bold.
STABLE_AT = 47


def t53_selection_frequency(results, cohorts, texdir):
    r"""The appendix's tab:app-pj-stability: the cohorts side by side, one parameter per line group.

    Each axis's values are sorted by how often they were chosen, and the shorter cohort's column is
    padded with blanks, so a line pairs the two cohorts' *ranks*, not their values.
    """
    freqs, gaps, totals = {}, [], set()
    for cohort in cohorts:
        freq = C.table(results, cohort, "s5_selection_frequency")
        freqs[cohort] = freq
        # Every outer evaluation picks one prompt, so the prompt axis sums to their number.
        # The juror axis does not: one evaluation selects several jurors.
        totals.add(int(freq[freq["axis"] == "prompt"]["n_folds"].sum()))
    total = max(totals) if totals else 0

    def lines_of(cohort, axis, title, label_of):
        freq = freqs[cohort]
        sub = freq[freq["axis"] == axis]
        if sub.empty:
            gaps.append(f"{C.COHORT_SHORT[cohort]}: axis {axis} is not recorded in "
                        "s5_selection_frequency.csv")
            return []
        # Ties in reverse name order, which is the order the appendix has always printed them.
        sub = sub.assign(_v=sub["value"].astype(str)).sort_values(
            ["n_folds", "_v"], ascending=[False, False], kind="stable")
        out = []
        for i, (_, r) in enumerate(sub.iterrows()):
            n = int(r["n_folds"])
            chosen = rf"\textbf{{{n}/{total}}}" if n >= STABLE_AT else f"{n}/{total}"
            out.append([title if i == 0 else "", C.tex_escape(label_of(str(r["value"]))),
                        chosen, C.fmt(r["share"], 2)])
        return out

    rows = []
    for g, group in enumerate(STABILITY_GROUPS):
        if g:
            rows.append(r"\midrule")
        for axis, title, label_of in group:
            sides = [lines_of(cohort, axis, title, label_of) for cohort in cohorts]
            depth = max(len(side) for side in sides)
            for i in range(depth):
                rows.append([cell for side in sides
                             for cell in (side[i] if i < len(side) else ["", "", "", ""])])

    # The fold design, read from the selection log itself: one row per outer evaluation.
    sel = optional(results, cohorts[0], "s5_selection")
    design = ""
    if sel is not None and {"outer_fold", "repetition"} <= set(sel.columns):
        n_outer, n_reps = sel["outer_fold"].nunique(), sel["repetition"].nunique()
        if n_outer * n_reps == total:
            design = rf" ({n_outer} outer folds $\times$ {n_reps} repetitions)"
    spans = " & ".join(
        r"\multicolumn{4}{c%s}{%s}" % ("|" if i < len(cohorts) - 1 else "", C.COHORT_LABEL[c])
        for i, c in enumerate(cohorts))
    header = (spans + r" \\" + "\n    "
              + " & ".join(["Parameter", "Value", "Chosen", "Share"] * len(cohorts)))
    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_selection_frequency.tex"),
        caption=(r"\textbf{PhenoJury selection stability.} How often each value of each selected "
                 r"parameter was chosen for the selected jury."),
        label="tab:app-pj-stability", placement="h",
        header=header, colspec="@{}" + "|".join(["llcc"] * len(cohorts)) + "@{}",
        preamble=[r"\small"], rows=rows,
        notes=(rf"Counts over the {total} outer evaluations{design} of nested cross-validation. "
               rf"Bold: chosen in at least {STABLE_AT} of {total}. " + gap_note(gaps)),
        generator=GENERATOR)


# -- T5.4 model x prompt ------------------------------------------------------

#: The cohort as the appendix's variance sentence names it.
VARIANCE_COHORT = {"hcy": "HCY", "gsc206": "GSC+ (206)"}


def _variance_sentence(variance) -> str:
    r"""The sum-of-squares decomposition of the same matrix, as prose.

    Three shares per cohort. They answer the question the caption poses -- can prompt and model be
    chosen sequentially -- and they are the only numbers in this artifact that are not an $F_1$, so
    printing them as a table row put a different quantity under an $F_1$ column heading.
    """
    if not variance:
        return ""
    bits = [f"{VARIANCE_COHORT.get(cohort, C.COHORT_SHORT[cohort])}: "
            f"model {C.fmt(dec['share_model'])}, prompt {C.fmt(dec['share_prompt'])}, "
            f"interaction {C.fmt(dec['share_interaction'])} "
            f"(grand mean {C.fmt(dec['grand_mean'])})" for cohort, dec in variance]
    return r"Variance shares of the matrix (sum of squares): " + ". ".join(bits) + ". "


def t54_model_prompt_interaction(results, cohorts, texdir):
    """Write ``tab_ch5_model_prompt_interaction.tex``."""
    rows, variance = [], []
    for ci, cohort in enumerate(cohorts):
        grid = C.table(results, cohort, "s4_interaction")
        cell = {(str(r["model"]), str(r["prompt"])): r for _, r in grid.iterrows()}
        cohort_header(rows, cohort, 5, ci == 0)
        for model in C.MODEL_ORDER:
            cells = []
            for prompt in C.PROMPT_ORDER:
                r = cell.get((model, prompt))
                cells.append(C.fmt(r["micro_f1"]) if r is not None else "--")
            # Marked along the ROW: bold is the model's best prompt and underline its second, so
            # each model is read against itself. The cells go in as a column to reuse the rule.
            column = [[c] for c in cells]
            C.mark_best_second(column, 0)
            rows.append([C.tex_escape(C.MODEL_LABEL[model])] + [c[0] for c in column])
        variance.append((cohort, C.table(results, cohort, "s4_decomposition").iloc[0]))

    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_model_prompt_interaction.tex"),
        caption=(r"\textbf{The best prompt differs by model and by cohort.} Single-juror "
                 r"$F_{1,\mu}$ for each model and prompt on the development split."),
        label="tab:app-interaction", placement="h",
        header=["Model"] + [C.tex_escape(C.PROMPT_LABEL[p]) for p in C.PROMPT_ORDER],
        colspec="@{}lcccc@{}", preamble=[r"\small"], rows=rows, generator=GENERATOR,
        notes=(r"At the selected normaliser. Bold: each model's best prompt. Underlined: its second "
               r"best. " + _variance_sentence(variance)
               + r"No intervals: descriptive, development split only."))


# -- T5.5 the full pool under each prompt ------------------------------------

def t55_prompt_sensitivity(results, cohorts, texdir):
    """Write ``tab_ch5_prompt_sensitivity.tex``."""
    rows, spread, single_spread, no_macro = [], {}, {}, []
    for ci, cohort in enumerate(cohorts):
        sens = C.table(results, cohort, "s8_prompt_sensitivity")
        if "macro_f1" not in sens.columns:
            no_macro.append(cohort)
        cohort_header(rows, cohort, 6, ci == 0)
        block_start = len(rows)
        shown = []
        for prompt in C.PROMPT_ORDER:
            sub = sens[sens["prompt"] == prompt]
            if sub.empty:
                continue
            r = sub.iloc[0]
            shown.append(float(C.fmt(r["micro_f1"])))
            rows.append([C.tex_escape(C.PROMPT_LABEL[prompt]),
                         C.fmt(r.get("micro_precision")), C.fmt(r.get("micro_recall")),
                         C.fmt(r["micro_f1"]), interval(r), C.fmt(r.get("macro_f1"))])
        block = rows[block_start:]
        C.mark_best_second(block, [1, 2, 3, 5])
        rows[block_start:] = block
        spread[cohort] = max(shown) - min(shown) if shown else float("nan")
        # The single jurors' prompt sensitivity, from the model x prompt matrix beside it: the
        # widest spread any one model shows across the four prompts.
        grid = C.table(results, cohort, "s4_interaction")
        single_spread[cohort] = max(
            float(g["micro_f1"].max() - g["micro_f1"].min()) for _, g in grid.groupby("model"))
    if no_macro:
        print("  WARNING tab_ch5_prompt_sensitivity: s8_prompt_sensitivity.csv has no macro_f1 "
              f"on {', '.join(no_macro)}, so F1_M prints '--'. Rerun exp14_04's sensitivity "
              "stage and pull its tables.", file=sys.stderr)

    # "Much less sensitive": the pool's widest spread across prompts is under half the widest
    # single juror's, on every cohort. The claim is read off the numbers, not assumed.
    absorbed = all(spread[c] < single_spread[c] / 2 for c in cohorts)
    title = (r"The eight-juror pool is much less sensitive to the prompt than single jurors."
             if absorbed else r"The eight-juror pool under each prompt.")
    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_prompt_sensitivity.tex"),
        caption=(r"\textbf{" + title + r"} Out-of-fold scores of the full pool under each prompt."),
        label="tab:pj-prompts", placement="t",
        header=["Prompt", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$", r"95\,\% CI", r"$F_{1,M}$"],
        colspec="@{}lccccc@{}", preamble=[r"\small"], rows=rows, generator=GENERATOR,
        notes=(r"At the selected normaliser, with rule, scope and $k$ selected inside the folds."))


# -- T5.6 the normaliser exchange ---------------------------------------------

#: Dictionary first, then the two learned readers. The raw PhenoBERT reader is not a selectable
#: normaliser and is not listed; "PhenoBERT" here is always the candidate reader.
EXCHANGE_ORDER = ["dictionary", "phenobert_candidates", "sapbert"]
EXCHANGE_LABEL = THESIS_NORMALISER_LABEL


def t56_normaliser_exchange(results, cohorts, texdir):
    r"""The chapter's tab:pj-normaliser: the same generations read by three normalisers.

    The re-scoring without the annotations first proposed with PhenoBERT on screen is no longer
    printed here: it is a statement about how the HCY ground truth was built, not about the
    normalisers, and the text reports it where the curation is described.
    """
    rows, gaps, dev_prompt, best = [], [], {}, {}
    for ci, cohort in enumerate(cohorts):
        full = C.table(results, cohort, "s2_prompt_normaliser_grid")
        sel_prompt = C.manifest(results, cohort)["selected_prompt"]
        dev_prompt[cohort] = sel_prompt

        cohort_header(rows, cohort, 6, ci == 0)
        by_full = {str(r["normaliser"]): r for _, r in full.iterrows()
                   if str(r["prompt"]) == sel_prompt}
        block_start = len(rows)
        for norm in EXCHANGE_ORDER:
            r = by_full.get(norm)
            if r is None:
                continue
            if not ("micro_f1_lo" in r and r["micro_f1_lo"] == r["micro_f1_lo"]):
                gaps.append(f"{C.COHORT_SHORT[cohort]}: no interval in "
                            "s2_prompt_normaliser_grid.csv; re-run exp14_04")
            rows.append([EXCHANGE_LABEL[norm], str(int(r["k"])),
                         C.fmt(r["micro_precision"]), C.fmt(r["micro_recall"]),
                         C.fmt(r["micro_f1"]), interval(r) if r["micro_f1_lo"] == r["micro_f1_lo"]
                         else "--"])
        block = rows[block_start:]
        f1 = {row[0]: float(row[4]) for row in block if row[4] != "--"}
        top = max(f1.values()) if f1 else None
        best[cohort] = sorted(k for k, v in f1.items() if v == top)
        C.mark_best_second(block, [2, 3, 4])
        rows[block_start:] = block

    winners = {tuple(v) for v in best.values()}
    if len(winners) == 1 and len(next(iter(winners))) == 1:
        both = "both cohorts" if len(cohorts) == 2 else "every cohort"
        title = f"{next(iter(winners))[0]} is the best normaliser on {both}."
    else:
        title = "The best normaliser differs by cohort."
    # "\emph{HPO-Guided} prompt on HCY, \emph{Free Listing} on GSC+": the noun once, on the first.
    prompts = ", ".join(f"\\emph{{{C.PROMPT_LABEL.get(p, C.tex_escape(p))}}}{' prompt' if i == 0 else ''}"
                        f" on {C.COHORT_SHORT[c]}" for i, (c, p) in enumerate(dev_prompt.items()))
    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_normaliser_exchange.tex"),
        caption=(r"\textbf{" + title + r"} Eight-juror pool on the development split with only the "
                 r"normaliser changed."),
        label="tab:pj-normaliser", placement="t",
        header=["Normaliser", "$k$", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$", r"95\,\% CI"],
        colspec="@{}lccccc@{}", preamble=[r"\small"], rows=rows,
        notes=(prompts + r", each normaliser at its own best vote threshold $k$. "
               + gap_note(sorted(set(gaps)))), generator=GENERATOR)


# -- T5.7 leave one model out -------------------------------------------------

def t57_leave_one_out(results, cohorts, texdir):
    """Write ``tab_ch5_leave_one_out.tex``."""
    rows, gaps = [], []
    for ci, cohort in enumerate(cohorts):
        lomo = optional(results, cohort, "s6_lomo")
        diag = optional(results, cohort, "s3_prompt_diagnostics")
        grid = C.table(results, cohort, "s4_interaction")
        man = C.manifest(results, cohort)
        sel_prompt = man["selected_prompt"]
        if lomo is None:
            gaps.append(f"{C.COHORT_SHORT[cohort]}: no s6_lomo.csv, so the "
                        "leave-one-model-out column is absent; exp14_04 must re-score the "
                        "selected jury eight times, once without each member")

        by_lomo = {str(r["model"]): r for _, r in lomo.iterrows()} if lomo is not None else {}
        by_grid = {str(r["model"]): r for _, r in grid.iterrows()
                   if str(r["prompt"]) == sel_prompt}
        by_diag = ({str(r["model"]): r for _, r in diag.iterrows()
                    if str(r["prompt"]) == sel_prompt} if diag is not None else {})

        cohort_header(rows, cohort, 7, ci == 0)
        for model in C.MODEL_ORDER:
            g, d, l = by_grid.get(model), by_diag.get(model), by_lomo.get(model)
            rows.append([
                C.tex_escape(C.MODEL_LABEL[model]),
                (r"$%+.2f$" % float(l["delta_micro_f1"])) if l is not None else "--",
                C.fmt(g["micro_precision"]) if g is not None else "--",
                C.fmt(g["micro_recall"]) if g is not None else "--",
                C.fmt(g["micro_f1"]) if g is not None else "--",
                C.fmt(g["findings_per_segment"], 2) if g is not None else "--",
                C.fmt(1.0 - float(d["empty_report_rate"]), 2) if d is not None else "--",
            ])

        # Three summary numbers stand in for the 8x8 matrix. A heatmap of 28 pairs would carry
        # less than these do: what counts is whether the jurors fail together.
        div = C.table(results, cohort, "s6_error_diversity")
        pairs = div[(div["kind"] == "model_diverse")
                    & div["a"].astype(str).str.contains(sel_prompt, regex=False)]
        pairs = pairs if not pairs.empty else div[div["kind"] == "model_diverse"]
        j = [float(v) for v in pairs["fp_jaccard"]]
        rows.append(r"\midrule")
        rows.append([r"\emph{pairwise FP Jaccard}", "",
                     r"mean " + C.fmt(sum(j) / len(j)) if j else "--",
                     r"min " + C.fmt(min(j)) if j else "--",
                     r"max " + C.fmt(max(j)) if j else "--",
                     r"%d pairs" % len(j), ""])

    notes = (r"$\Delta F_{1,\mu}$: change in the selected jury's $F_{1,\mu}$ when that juror is "
             r"removed. $P_\mu$, $R_\mu$, $F_{1,\mu}$: the juror alone at the selected prompt. "
             r"cand./seg.: candidates per segment. compl.: share of documents with at least one "
             r"parseable finding. Last row: pairwise false-positive Jaccard similarity over all "
             r"juror pairs.")
    notes += gap_sentence(gaps)

    C.write_table_tex(
        os.path.join(texdir, "tab_ch5_leave_one_out.tex"),
        caption=(r"\textbf{Leave one model out.} For each juror: the change in the selected "
                 r"jury's score when it is removed, and its own single-model quality."),
        label="tab:ch5-leave-one-out",
        header=["Juror", r"$\Delta F_{1,\mu}$", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$",
                "cand./seg.", "compl."],
        colspec="lcccccc",
        # Seven columns: over the 5.5in text width at default padding.
        rows=rows, notes=notes, generator=GENERATOR, compact=True)


ALL = [t50_main_comparison, t51_union_recall_gate, t52_jury_ablation, t53_selection_frequency,
       t54_model_prompt_interaction, t55_prompt_sensitivity, t56_normaliser_exchange,
       t57_leave_one_out]


def main():
    """Build every chapter 5 table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=C.DEFAULT_RESULTS)
    ap.add_argument("--texdir", default=C.DEFAULT_TEXDIR)
    ap.add_argument("--comparison", default=None,
                    help="output/comparison, for T5.0's baseline rows")
    args = ap.parse_args()
    os.makedirs(args.texdir, exist_ok=True)

    cohorts = [c for c in C.COHORTS if C.has_cohort(args.results, c)]
    if not cohorts:
        raise SystemExit(f"no cohort tables under {args.results}")

    for fn in ALL:
        try:
            extra = {"comparison": args.comparison} if fn is t50_main_comparison else {}
            fn(args.results, cohorts, args.texdir, **extra)
        except SystemExit as exc:
            # ch{5,6}_common.table() already printed the path and the reason to stderr and exits
            # with MISSING_SOURCE_EXIT. Anything else carries its own message.
            detail = ("source not available -- see the line above"
                      if exc.code == C.MISSING_SOURCE_EXIT else exc)
            print(f"[tables] skipped {fn.__name__}: {detail}")


if __name__ == "__main__":
    main()
