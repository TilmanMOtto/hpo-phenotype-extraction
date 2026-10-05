r"""Every chapter-4 table, as a standalone booktabs ``\begin{table}`` file.

One function per specified artifact, named for what the table *carries* rather than for which CSV
it happens to read:

    T4.0  main_comparison         TreePhenoRAG against the 8B baselines and PhenoBERT on HCY
    T4.1  retrieval_sufficiency   does retrieval reach the annotated evidence at all
    T4.2  ablation_ladder         which component the result comes from
    T4.3  pooling_operators       both pooling axes at the selected configuration
    T4.4  alpha_sensitivity       what the risk tolerance buys
    T4.6  segment_ablation        segments verified per visited term
    T4.8  selection_frequency     parameter stability across the folds

Each table carries the label the thesis prose already cites (``tab:tpr-ladder``,
``tab:app-retrieval``, ...), and its caption is the thesis's own wording with every number read
from the CSV, so the generated file can replace the float in the chapter without touching a line
of prose.

The recall decomposition and the false-positive taxonomy are shares of a whole, so they are
figures (``fig_ch4_recall_decomposition``, ``fig_ch4_error_analysis``), not tables.

Caption rules, applied to every float in chapters 4-6: a bold title stating the finding (or, for a
purely descriptive table, what it shows), then what each column is, the cohort, n and the split,
then what the intervals are -- and stop. Argument, method and notes belong in the text.

Precision: P, R and F1 are printed to **2 decimals**. The report-level intervals on ~118 reports
are about +/-0.03 wide, so a third decimal would be noise. A difference column is computed from
the *printed* values, so it always equals the subtraction a reader does by eye.

A table whose CSV does not exist yet raises ``SystemExit`` naming the CSV it wanted -- see
``docs/thesis_map.md`` --, not printing a plausible blank.
"""
from __future__ import annotations

import collections
import json

import common as C

SCRIPT = "tables_ch4.py"

#: P/R/F1 precision -- see the module docstring.
PRF = 2

# The five variants the ladder reports. `nested CV (pooled out-of-fold)` is absent: it
# is the same run as V3, and printing one result twice invites a reader to compare it with itself.
LADDER_ROWS = [
    ("V0 traversal, P0 both decisions", r"V0 (P0, fixed at $\nicefrac{1}{2}$)"),
    ("V1 P1 both, one shared threshold", "V1 (P1, one threshold)"),
    ("V2 P1, tau_prune by risk control", "V2 (P1, two thresholds)"),
    ("V3 selected poolings", "V3 (selected, two thresholds)"),
    ("oracle expansion (not an operating point)", "Upper bound"),
]

#: The upper-bound row is printed but never competes for a bold mark: it reads the answer.
LADDER_ORACLE_ROW = len(LADDER_ROWS) - 1

# Both pooling axes, in the order 04_TreePhenoRAG.tex (tab:tpr-operators) defines the operators.
# The keys must be spelled as `hpo_extraction.treephenorag.stored_scores.POOLINGS` spells them -- `P3_S`, not `P3S`;
# `main` checks the CSV's operators against these, not printing a short table.
#
# The descriptions are the definitions in hpo_extraction.treephenorag.stored_scores, NOT paraphrases: P2 is the second
# largest segment score (not a mean), P3_kappa the tempered noisy-OR, and P4 the mean -- an earlier
# version of this table had those three the wrong way round.
POOL_ORDER = ["P1", "P2", "P3_1", "P3_2", "P3_S", "P4", "lse_beta1"]
POOL_LABEL = {
    "P0": r"P0 indicator $\sigma_{(1)}>\nicefrac{1}{2}$",
    "P1": "P1 maximum",
    "P2": "P2 second largest",
    "P3_1": r"P3$_1$ noisy-OR, $\kappa=1$",
    "P3_2": r"P3$_2$ noisy-OR, $\kappa=2$",
    "P3_S": r"P3$_S$ noisy-OR, $\kappa=S$",
    "P4": "P4 mean",
    "lse_beta1": "LSE log-sum-exp",
}

#: The two retrieval indices as the appendix names them: the table's column header already says
#: "(with descendant closure)", so the rows do not repeat it.
INDEX_SHORT = {"exemplar": "synthetic sentences", "ontology_r3": "label + definition + synonyms"}


def num(value: float, places: int = PRF) -> str:
    """*value* to *places* decimals, or ``--`` for NaN."""
    return "--" if value != value else f"{value:.{places}f}"


def thousands(value: float) -> str:
    """*value* with LaTeX thin spaces between thousands, or ``--`` for NaN."""
    return "--" if value != value else f"{value:,.0f}".replace(",", r"\,")


def sci(value: float) -> str:
    r"""``\num{5.58e-05}`` -- siunitx sets it as the score-distribution caption quotes thresholds."""
    return "--" if value != value else rf"\num{{{value:.2e}}}"


def ci(row: dict, key: str, places: int = PRF, signed: bool = False) -> str:
    r"""``0.31 \ [0.27, 0.34]``, or the bare point estimate when no interval was written.

    The whole cell goes inside one ``$...$`` when ``signed``, because a paired difference can be
    negative and a bare ``-`` in text mode sets a hyphen, not a minus sign.
    """
    point = C.f(row, key)
    lo, hi = C.f(row, f"{key}_lo"), C.f(row, f"{key}_hi")
    if signed:
        if lo != lo or hi != hi:
            return "--" if point != point else f"${point:+.{places}f}$"
        return (f"${point:+.{places}f}$ \\ "
                f"$[{num(lo, places)}, {num(hi, places)}]$")
    if lo != lo or hi != hi:
        return num(point, places)
    return rf"{num(point, places)} [{num(lo, places)}, {num(hi, places)}]"


def p_value(p: float) -> str:
    """A p-value at the precision a permutation test with 10 000 draws supports."""
    if p != p:
        return "--"
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def require_columns(name: str, rows: list[dict], columns: tuple[str, ...]) -> None:
    """Skip a table whose CSV predates the pooled-out-of-fold rewrite of the TreePhenoRAG protocol (2026-09-24).

    Those CSVs measured the same quantities on a different population (one fold's configuration
    re-run in-sample on every report, or the CRC log averaged over both retrieval indices), so
    rendering them under the new captions would put old numbers under a claim they do not satisfy.
    Skipping keeps the previous .tex on disk and names the rerun that fixes it.
    """
    missing = [c for c in columns if not rows or c not in rows[0]]
    if missing:
        raise SystemExit(
            f"{name}.csv predates the pooled out-of-fold rewrite (no column {missing[0]!r}); "
            f"rerun the TreePhenoRAG protocol (slurm/treephenorag_protocol.sbatch) and copy its tables")


def operating_point() -> dict:
    """The run's own ``selected_configuration.json``: the index and S the chapter runs at."""
    import json

    path = C.RESULTS / "selected_configuration.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} -- pull it with the tables (README, 'Pull the tables')")
    return json.loads(path.read_text(encoding="utf-8"))


def chapter_alpha() -> float:
    """The risk tolerance the chapter runs at, from the run's own resolved config.

    ``selected_configuration.json`` does not carry it: alpha is fixed, not selected.
    """
    path = C.RESULTS / "config_resolved.yaml"
    if not path.is_file():
        raise SystemExit(f"missing {path} -- pull it with the tables (README, 'Pull the tables')")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("alpha:"):
            return float(line.split(":", 1)[1])
    raise SystemExit(f"{path} has no top-level alpha")


def pct(value: float) -> str:
    r"""A percentage as the chapter prints it: ``97\,\%``."""
    return f"{value * 100:.0f}" + r"\,\%"


# -- T4.1 does retrieval reach the annotated evidence --------------------------

def retrieval_sufficiency() -> None:
    r"""The appendix's tab:app-retrieval.

    No bold row: the setting the chapter runs at is named in the caption instead, read from the
    run, not typed -- this was a constant (ontology_r3, S = 5) while the traversal ran at
    S = 10, so the marked row named a setting no other table in the chapter uses.
    """
    selected = operating_point()
    rows = sorted(C.table("retrieval_sufficiency"),
                  key=lambda r: (int(float(r["S"])), r["retrieval_index"] != "exemplar"))
    out = [[str(int(float(r["S"]))),
            INDEX_SHORT.get(r["retrieval_index"], C.tex_escape(r["retrieval_index"])),
            ci(r, "hit_at_s"), ci(r, "path_hit_rate")] for r in rows]

    n_gold = int(C.f(rows[0], "n_gold")) if rows else 0
    n_reports = int(C.f(rows[0], "n_reports")) if rows else 0
    index = INDEX_SHORT.get(selected["retrieval_index"], C.tex_escape(selected["retrieval_index"]))
    C.write_table_tex(
        "tab_ch4_retrieval_sufficiency", label="app-retrieval", script=SCRIPT,
        placement="h", column_spec="@{}clcc@{}",
        header=["$S$", "Retrieval index (with descendant closure)", r"Hit@$S$",
                r"Path-wise Hit@$S$"],
        rows=out,
        caption=(
            r"\textbf{Both retrieval indices retrieve the evidence of at least "
            + f"{min(C.f(r, 'hit_at_s') for r in rows if int(float(r['S'])) == int(selected['S'])):.2f}"
            + r" of the annotated HCY terms at " + rf"$S={int(selected['S'])}$.}}"),
        notes=(
            str(n_reports) + r" reports with non-empty ground truth, " + thousands(n_gold)
            + r" annotated terms. Hit@$S$: share of annotated terms for which at least one of the "
            r"$S$ segments retrieved for the term contains the evidence named by the curator. "
            r"Path-wise Hit@$S$: the same, required at every term on the path from the top to the "
            r"annotated term. Main results "
            rf"use $S={int(selected['S'])}$ with {index}."))


# -- T4.2 the ablation ladder -------------------------------------------------

def ablation_ladder() -> None:
    """Write ``tab_ch4_ablation_ladder.tex``."""
    rows = {r["config"]: r for r in C.table("core_quality")}
    tests = {t["comparison"]: t for t in C.table("ablation_tests")}
    by_higher = {k.split(" vs ")[0]: v for k, v in tests.items()}

    out, shown_f1, oracle_at = [], [], None
    for i, (key, label) in enumerate(LADDER_ROWS):
        if key not in rows:
            continue
        r = rows[key]
        test = by_higher.get(key)
        f1 = round(C.f(r, "micro_f1"), PRF)
        # Delta is the difference of the two PRINTED F1 values, so it can never disagree with
        # The subtraction a reader does by eye (the test's own delta is unrounded).
        delta = (f"${f1 - shown_f1[-1]:+.{PRF}f}$"
                 if test and shown_f1 and i != LADDER_ORACLE_ROW else "--")
        if i == LADDER_ORACLE_ROW:
            oracle_at = len(out)
        else:
            shown_f1.append(f1)
        out.append([
            label,
            num(C.f(r, "micro_precision")), num(C.f(r, "micro_recall")), ci(r, "micro_f1"),
            num(C.f(r, "macro_precision")), num(C.f(r, "macro_recall")), num(C.f(r, "macro_f1")),
            thousands(C.f(r, "calls_per_report")) if i != LADDER_ORACLE_ROW else "--",
            delta,
            p_value(float(test["p_holm"])) if test else "--",
        ])
    # The upper bound reads the answer, so it is printed but never competes for a mark.
    C.mark_best_second(out, [1, 2, 3, 4, 5, 6], skip={oracle_at} if oracle_at is not None else None)
    n_reports = int(C.f(rows.get("nested CV (pooled out-of-fold)", {}), "n_reports", 118))
    selected = operating_point()

    one, two, v3 = (rows.get(LADDER_ROWS[k][0]) for k in (1, 2, 3))
    if one and two and v3:
        f1 = {k: round(C.f(r, "micro_f1"), PRF) for k, r in (("V1", one), ("V2", two), ("V3", v3))}
        gain = f1["V3"] - f1["V2"]
        tail = ("adds nothing" if abs(gain) < 0.005 else
                rf"changes $F_{{1,\mu}}$ by ${gain:+.{PRF}f}$")
        title = (r"\textbf{Separating the two thresholds raises $F_{1,\mu}$ from "
                 f"{f1['V1']:.{PRF}f} to {f1['V2']:.{PRF}f}, and selecting the pooling "
                 f"operators {tail}.}}")
    else:
        title = r"\textbf{Ablation ladder on HCY.}"
    C.write_table_tex(
        "tab_ch4_ablation_ladder", label=["tpr-ladder", "ch4-ablation-ladder"], script=SCRIPT,
        placement="t", column_spec="@{}lccccccrcc@{}", size=r"\footnotesize", tabcolsep="3pt",
        header=["Variant", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$ [95\,\% CI]",
                "$P_M$", "$R_M$", "$F_{1,M}$", "Calls", r"$\Delta F_{1,\mu}$",
                r"$p_{\mathrm{Holm}}$"],
        rows=out, rules=[oracle_at] if oracle_at is not None else None,
        caption=(
            title + r" Ablation ladder on HCY (" + str(n_reports) + r" reports)."),
        notes=(
            r"Pooled out-of-fold predictions of nested cross-validation at "
            rf"$S={int(selected['S'])}$, $\alpha={chapter_alpha():.2f}$. "
            r"Upper bound: expands only the terms with an annotated term below them and "
            r"reads the ground truth."))


# -- T4.3 both pooling axes at the selected configuration ---------------------

def pooling_operators() -> None:
    """Write ``tab_ch4_pooling_operators.tex``."""
    rows = C.table("pooling_operators")
    require_columns("pooling_operators", rows, ("tau_min", "attained_coverage_macro"))
    blocks: dict[str, dict] = collections.defaultdict(dict)
    for r in rows:
        blocks[r["block"]][r["operator"]] = r

    expansion, acceptance = blocks.get("expansion", {}), blocks.get("acceptance", {})
    unknown = (set(expansion) | set(acceptance)) - set(POOL_ORDER) - {"P0"}
    if unknown:
        print(f"  WARNING tab_ch4_pooling_operators: pooling_operators.csv carries "
              f"{sorted(unknown)}, which POOL_ORDER does not name — those rows are NOT in the "
              f"table. Add them to POOL_ORDER/POOL_LABEL.")
    expand = [[POOL_LABEL.get(op, C.tex_escape(op)), sci(C.f(r, "tau")),
               thousands(C.f(r, "calls_per_report")), num(C.f(r, "attained_coverage")),
               num(C.f(r, "attained_coverage_macro"))]
              for op, r in ((op, expansion.get(op)) for op in POOL_ORDER) if r is not None]
    # Expansion competes on cost alone: every operator is held to the same risk, so coverage is
    # The bound's doing and marking its ties would single out nothing.
    C.mark_best_second(expand, 2, higher_is_better=False, second=False)
    # Average precision per acceptance operator over the same scored pairs: the threshold-free
    # ranking the single-threshold columns cannot show (check Q4). Written by the TreePhenoRAG protocol's `ranking`
    # stage. While it is absent the column stays empty, not the table failing.
    ranking = {}
    if (C.TABLES / "acceptance_ranking.csv").is_file():
        ranking = {r["operator"]: r for r in C.table("acceptance_ranking")}
    accept, tau_accept = [], None
    for op in ["P0"] + POOL_ORDER:
        r = acceptance.get(op)
        if r is None:
            continue
        tau_accept = C.f(r, "tau")
        ap = num(C.f(ranking[op], "average_precision")) if op in ranking else ""
        accept.append([POOL_LABEL.get(op, C.tex_escape(op)), num(C.f(r, "micro_precision")),
                       num(C.f(r, "micro_recall")), num(C.f(r, "micro_f1")), ap])
    C.mark_best_second(accept, [1, 2, 3, 4] if ranking else [1, 2, 3])

    alpha = chapter_alpha()
    tau_text = f"{tau_accept:g}" if tau_accept else "--"
    coverage_m = [round(C.f(r, "attained_coverage_macro"), PRF) for r in expansion.values()]
    same = coverage_m and max(coverage_m) - min(coverage_m) <= 0.01 + 1e-9
    if not same:
        print("  WARNING tab_ch4_pooling_operators: Coverage_M differs by more than 0.01 across "
              "expansion operators -- the title no longer says 'the same coverage'.")
    title = (r"\textbf{All expansion operators reach the same coverage at equal risk, and at the "
             r"selected acceptance threshold only maximum-type operators keep recall.}"
             if same else r"\textbf{Pooling operators for expansion and acceptance on HCY.}")
    out = [rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{Expansion, at equal risk $\alpha={alpha:.2f}$}}}}",
           r"Operator & $\hat\tau_{\mathrm{prune}}$ & Calls & Coverage$_\mu$ & Coverage$_M$",
           r"\midrule", *expand, r"\midrule",
           rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{Acceptance, at $\tau_{{\mathrm{{accept}}}}={tau_text}$}}}}",
           r"Operator & $P_\mu$ & $R_\mu$ & $F_{1,\mu}$ & " + ("AP" if ranking else ""),
           r"\midrule", *accept]
    C.write_table_tex(
        "tab_ch4_pooling_operators", label="tpr-pooling", script=SCRIPT,
        placement="t", column_spec="@{}lcrcc@{}", header=None, rows=out,
        caption=(
            title),
        notes=(
            r"HCY, pooled out-of-fold. Top: each expansion operator at its own "
            rf"conformal-risk-control threshold at $\alpha={alpha:.2f}$ (median over folds), with "
            r"verifier calls per report and coverage, the share of annotated terms the traversal "
            r"visits. Bottom: each acceptance operator at "
            rf"$\tau_{{\mathrm{{accept}}}}={tau_text}$, the value selected for LSE, with expansion "
            r"fixed at the selected configuration"
            + (r". AP: average precision of the operator's score over the same scored pairs, "
               r"which needs no threshold" if ranking else "")
            + r". Operators as in \Cref{tab:tpr-operators}."))


# -- T4.4 what the risk tolerance buys ----------------------------------------

def alpha_sensitivity() -> None:
    """Write ``tab_ch4_alpha_sensitivity.tex``."""
    rows = sorted(C.table("alpha_sensitivity"), key=lambda r: float(r["alpha"]))
    require_columns("alpha_sensitivity", rows, ("attained_coverage_micro",
                                                "attained_coverage_macro", "held_out_miss_rate",
                                                "calls_per_report"))
    out = [[
        f"{C.f(r, 'alpha'):.2f}",
        num(C.f(r, "attained_coverage_micro")),
        num(C.f(r, "attained_coverage_macro")),
        num(C.f(r, "held_out_miss_rate")),
        thousands(C.f(r, "calls_per_report")),
        num(C.f(r, "micro_f1")),
    ] for r in rows]
    # Feasible is dropped only while it carries no variation. The moment one tolerance is
    # infeasible it is information again, so it comes back, not silently vanishing.
    feasible = {(r["feasible_folds"], r["n_folds"]) for r in rows}
    all_feasible = all(f == n for f, n in feasible)
    header = [r"$\alpha$", r"Coverage$_\mu$", r"Coverage$_M$", "Held-out miss rate", "Calls",
              r"$F_{1,\mu}$"]
    if not all_feasible:
        for row, r in zip(out, rows):
            row.insert(1, f"{int(float(r['feasible_folds']))}/{int(float(r['n_folds']))}")
        header.insert(1, "Feasible")

    lo, hi = rows[0], rows[-1]
    cut = 1 - C.f(hi, "calls_per_report") / C.f(lo, "calls_per_report")
    # Every number in the title is read off the printed cells, so it is the comparison a reader
    # makes by eye: 0.989 to 0.825 prints as 0.99 to 0.83.
    f1s = [round(C.f(r, "micro_f1"), PRF) for r in rows]
    C.write_table_tex(
        "tab_ch4_alpha_sensitivity", label="app-alpha", script=SCRIPT,
        placement="h", column_spec="@{}" + "c" * len(header) + "@{}",
        header=header, rows=out,
        caption=(
            rf"\textbf{{Loosening $\alpha$ from {C.f(lo, 'alpha'):.2f} to {C.f(hi, 'alpha'):.2f} "
            rf"cuts verifier calls by {pct(cut)} and lowers coverage from "
            rf"{num(C.f(lo, 'attained_coverage_micro'))} to "
            rf"{num(C.f(hi, 'attained_coverage_micro'))}, with $F_{{1,\mu}}$ between "
            rf"{min(f1s):.{PRF}f} and {max(f1s):.{PRF}f}.}}"),
        notes=(
            r"The full protocol on HCY at each risk tolerance $\alpha$. Coverage: share of "
            r"annotated terms the traversal visits. Held-out miss rate: mean per-report share of annotated terms not "
            r"visited, on the outer folds. Calls: verifier calls per report. Main results use "
            rf"$\alpha={chapter_alpha():.2f}$."))


# -- T4.6 segments verified per visited term ----------------------------------

#: The knee's S as the title's first word.
NUMBER_WORD = {1: "One segment", 2: "Two segments", 3: "Three segments", 4: "Four segments",
               5: "Five segments"}


def segment_ablation() -> None:
    """Write ``tab_ch4_segment_ablation.tex``."""
    rows = sorted(C.table("s_ablation"), key=lambda r: int(float(r["S"])))
    require_columns("s_ablation", rows, ("attained_coverage",))
    if not rows or "macro_f1" not in rows[0]:
        print("  WARNING tab_ch4_segment_ablation: s_ablation.csv has no macro_f1 column, so "
              "F1_M prints '--'. Rerun the TreePhenoRAG protocol's ablations stage "
              "(slurm/treephenorag_protocol.sbatch) and copy its tables.")
    chapter_s = int(operating_point()["S"])
    best = max(C.f(r, "micro_f1") for r in rows)
    top_cost = max(C.f(r, "calls_per_report") for r in rows)
    out = [[
        str(int(float(r["S"]))),
        num(C.f(r, "micro_precision")),
        num(C.f(r, "micro_recall")),
        num(C.f(r, "micro_f1")),
        num(C.f(r, "macro_f1")),
        f"{C.f(r, 'micro_f1') / best * 100:.0f}",
        num(C.f(r, "attained_coverage")),
        thousands(C.f(r, "calls_per_report")),
        f"{C.f(r, 'calls_per_report') / top_cost * 100:.0f}",
    ] for r in rows]
    C.mark_best_second(out, [1, 2, 3, 4, 6])

    s = lambda r: int(float(r["S"]))  # noqa: E731
    knee, top = rows[1], rows[-1]
    C.write_table_tex(
        "tab_ch4_segment_ablation", label="tpr-segments", script=SCRIPT,
        placement="t", column_spec="@{}ccccccccc@{}",
        header=["$S$", r"$P_\mu$", r"$R_\mu$", r"$F_{1,\mu}$", r"$F_{1,M}$",
                r"\% of best $F_{1,\mu}$", "Coverage", "Calls", r"\% of calls"],
        rows=out,
        caption=(
            rf"\textbf{{{NUMBER_WORD.get(s(knee), f'S={s(knee)}')} per term keep "
            rf"{pct(C.f(knee, 'micro_f1') / best)} of the best $F_{{1,\mu}}$ at "
            rf"{pct(C.f(knee, 'calls_per_report') / top_cost)} of the verifier calls.}}"),
        notes=(
            r"$S$: segments verified per visited term, at the selected configuration on HCY, "
            r"with $\tau_{\mathrm{prune}}$ refitted for each $S$, pooled out-of-fold. "
            r"Coverage: share of annotated terms the traversal visits. Calls: verifier calls per report. Percentages relative "
            rf"to the best $F_{{1,\mu}}$ and to the calls at $S={s(top)}$. The $S={chapter_s}$ row "
            r"is V3 of \Cref{tab:tpr-ladder}."))


# T4.7 (tab_ch4_calibration_transfer: SmoothECE of the raw score and of Platt and isotonic maps) was
# dropped on 2026-09-28. The maps are not used -- a monotone h cannot change what is accepted -- and
# The raw SmoothECE it also carried is quoted in fig_ch4_reliability's note.


# -- T4.0 the chapter's comparison with the baselines -----------------------------

#: The systems chapter 4 is compared against: the smaller size of each LLM baseline (the one closest
#: to TreePhenoRAG's 8B verifier) and PhenoBERT. The 70B rows and PhenoJury wait for chapter 6.
CH4_COMPARISON = ["raghpo_8b", "autopcr_8b", "phenobert", "treephenorag"]


def main_comparison() -> None:
    r"""\Cref{tab:ch4-main-comparison}: TreePhenoRAG against the baselines on HCY.

    Same layout, marking and source as tab:ch6-overall-comparison (``tables_ch6.overall_rows``,
    the comparison's ``t1_overall.csv``), restricted to HCY and four systems, so a number here is the
    number chapter 6 prints for the same system. TreePhenoRAG's row is the pooled out-of-fold V3 of
    the ablation ladder, scored by the comparison, not TreePhenoRAG protocol.
    """
    import ch6_common as C6
    import tables_ch6 as T6

    df = C6.table(C6.DEFAULT_RESULTS, "t1_overall")
    rows = T6.overall_rows(df, cohorts=["hcy"], methods=CH4_COMPARISON)

    # The finding, read off the printed micro F1: which baselines TreePhenoRAG falls below.
    by = C6.by_method(df, "hcy")
    printed = {m: float(C6.fmt(by[m]["micro_f1"])) for m in CH4_COMPARISON if m in by}
    if "treephenorag" not in printed:
        raise SystemExit("t1_overall.csv has no scored TreePhenoRAG row on HCY")
    tree = printed.pop("treephenorag")
    above = [T6.THESIS_METHOD_LABEL[m] for m in CH4_COMPARISON if printed.get(m, -1) > tree]
    if len(above) == len(printed):
        count = {2: "two", 3: "three", 4: "four"}.get(len(printed), str(len(printed)))
        finding = f"TreePhenoRAG is below all {count} baselines on HCY."
    elif above:
        finding = f"TreePhenoRAG is below {T6._join(above)} on HCY."
    else:
        finding = "TreePhenoRAG is the most accurate of these systems on HCY."

    caption = (r"\textbf{" + finding + r"} Scores on the " + str(int(by["treephenorag"]["n_reports"]))
               + r" HCY reports against the same ground truth.")
    T6.write_overall_table(str(C.TEX_DIR / "tab_ch4_main_comparison.tex"), rows=rows,
                           caption=caption, label="tab:ch4-main-comparison",
                           generator=f"figures/thesis_figures_scripts/{SCRIPT}")


# -- T4.8 parameter stability --------------------------------------------------

#: The names the chapter uses for a selected value. A value missing here is printed raw in \texttt.
STABILITY_VALUE = {
    "ontology_r3": "label + definition + synonyms",
    "P1": "P1 (max)",
    "P2": "P2 (second largest)",
    "P3_1": r"P3$_1$ (noisy-OR, $\kappa=1$)",
    "P3_2": r"P3$_2$ (noisy-OR, $\kappa=2$)",
    "P3_S": r"P3$_S$ (noisy-OR, $\kappa=S$)",
    "P4": "P4 (mean)",
    "lse_beta1": "LSE",
}


def selection_frequency() -> None:
    """The chapter's tab:tpr-stability. A value tied with the mode gets its own line under it: the
    expansion operator split 25/25 between P4 and P3_S, and printing only the mode would name one of
    two equally chosen operators as "the" choice."""
    stability = {r["field"]: r for r in C.table("selection_stability")}
    label = {
        "retrieval_index": "Retrieval index",
        "pool_pr": r"Expansion operator $\Pi_{\mathrm{pr}}$",
        "pool_acc": r"Acceptance operator $\Pi_{\mathrm{acc}}$",
        "tau_accept": r"$\tau_{\mathrm{accept}}$",
    }

    def counts_of(stat) -> dict[str, int]:
        return {k: int(v) for k, v in json.loads(stat["counts"]).items()} if stat else {}

    def value(v: str) -> str:
        try:
            return f"{float(v):g}"                  # a threshold prints as a number
        except ValueError:
            return STABILITY_VALUE.get(v, rf"\texttt{{{C.tex_escape(v)}}}")

    # The value printed is the folds' MODE, read from the same CSV as its count. An earlier version
    # read `selected_configuration.json`, which can be a stale file from an earlier run -- and was:
    # it named P1, which no fold of the current run chose.
    out, tied = [], []
    for field in ("retrieval_index", "pool_pr", "pool_acc", "tau_accept"):
        stat = stability.get(field)
        if stat is None:
            continue
        counts = counts_of(stat)
        total, top = sum(counts.values()), counts.get(stat["mode"], 0)
        out.append([label[field], value(stat["mode"]), f"{top}/{total}",
                    str(int(float(stat["n_distinct"])))])
        ties = sorted(v for v, n in counts.items() if n == top and v != stat["mode"])
        out += [["", value(v), f"{top}/{total}", ""] for v in ties]
        if ties:
            tied.append((label[field][0].lower() + label[field][1:], 1 + len(ties)))

    choices = C.table("selection_choices")
    n_outer = len({r["outer_fold"] for r in choices})
    n_reps = len({r["repetition"] for r in choices})
    if len(tied) == 1 and tied[0][1] == 2:
        tie_note = f" The {tied[0][0]} tied between two values, so both are listed."
    elif tied:
        tie_note = " Values tied for most frequent are each listed on their own line."
    else:
        tie_note = ""
    C.write_table_tex(
        "tab_ch4_selection_frequency", label="tpr-stability", script=SCRIPT,
        placement="t", column_spec="@{}llcc@{}",
        header=["Parameter", "Most frequent value", "Chosen in", "Distinct values"],
        rows=out,
        caption=(
            r"\textbf{The TreePhenoRAG protocol chooses the same configuration in almost every "
            r"outer evaluation.}"),
        notes=(
            r"For each parameter selected inside nested cross-validation on HCY: the most frequent "
            rf"value, in how many of the {len(choices)} outer evaluations ({n_outer} folds "
            rf"$\times$ {n_reps} repetitions) it was chosen, and the number of distinct values "
            rf"chosen.{tie_note} $\tau_{{\mathrm{{prune}}}}$ is refitted in every fold by "
            r"conformal risk control and is not listed."))


ALL = [main_comparison, retrieval_sufficiency, ablation_ladder, pooling_operators, alpha_sensitivity,
       segment_ablation, selection_frequency]


def main() -> None:
    """Build every chapter 4 table. Exits 3 when a source table is missing."""
    for fn in ALL:
        try:
            fn()
        except SystemExit as exc:
            print(f"  skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
