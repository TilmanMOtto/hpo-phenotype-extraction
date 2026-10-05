r"""Figure: the cost-coverage curve, and where the bound stops being reachable.

Conformal risk control is not a hyperparameter to be tuned but a dial with a stated meaning: alpha
is the expected share of a report's annotated terms the traversal never scores. Two panels, for the two
things that make it worth having.

Left is the cost-coverage frontier proper -- verifier calls per report against attained coverage.
That pairing, rather than calls against micro F1, is the one the dial actually controls: coverage is
what alpha bounds, and it is a hard ceiling on recall no downstream threshold can lift. Cost
against quality is a different question and is answered by tab_ch4_segment_ablation.

Right is the diagnostic that the guarantee is real: the held-out per-report miss rate against the
bound it was asked to honour, measured on the outer split the threshold was never chosen on. The
infeasible region is shaded, not omitted. Where alpha sits below the traversal's own
attained ceiling no threshold in the cache satisfies it -- a fact about retrieval and the verifier,
not about the pruning rule, and one the chapter should show, not bury.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import common as C
import palette as P


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    rows = sorted(C.table("alpha_sensitivity"), key=lambda r: float(r["alpha"]))
    if not rows or "attained_coverage_micro" not in rows[0]:
        # Pre-2026-09-24 CSV: its calls and coverage averaged the CRC log over BOTH retrieval
        # indices and over training reports ("100 fold rows"), which is not the traversal the
        # chapter scores. Keep the previous PDF, not redraw it under the new caption.
        raise SystemExit("alpha_sensitivity.csv predates the pooled out-of-fold rewrite; rerun "
                         "exp13_22 and pull its tables")
    alpha = [C.f(r, "alpha") for r in rows]
    calls = [C.f(r, "calls_per_report") for r in rows]
    miss = [C.f(r, "held_out_miss_rate") for r in rows]
    feasible = [int(float(r["feasible_folds"])) for r in rows]
    n_folds = [int(float(r["n_folds"])) for r in rows]

    # Coverage comes from the experiment where it writes it, and is otherwise derived from the
    # out-of-fold miss rate -- which is the same quantity measured on the outer split. Which of
    # The two was used goes into the axis label and the caption, so the figure never implies a
    # column it did not read.
    measured = all(C.f(r, "attained_coverage_micro") == C.f(r, "attained_coverage_micro")
                   for r in rows)
    if measured:
        coverage = [C.f(r, "attained_coverage_micro") for r in rows]
        # "coverage" is the thesis's word for this quantity (Cov, 04_TreePhenoRAG.tex).
        cov_axis = "coverage"
        cov_source = r"share of annotated terms the traversal visits, pooled over reports"
    else:
        coverage = [1.0 - m for m in miss]
        cov_axis = "coverage\n$1-$ held-out miss rate"
        cov_source = r"one minus the held-out per-report miss rate"

    C.style.apply()
    fig, (left, right) = plt.subplots(1, 2, figsize=(C.WIDTH_FULL, 2.55))

    # -- left: the cost-coverage frontier ------------------------------------
    left.plot(calls, coverage, color=C.NEUTRAL, ls=(0, (4.0, 1.6)), lw=1.0, zorder=2)
    for i, a in enumerate(alpha):
        ok = feasible[i] == n_folds[i]
        left.plot(calls[i], coverage[i], marker="o" if ok else "X", ms=5.0 if ok else 5.6,
                  mfc=P.TREE if ok else "white", mec=P.TREE if ok else P.ERROR,
                  mew=1.0, ls="none", zorder=3)
        # The frontier rises left to right, so a label placed up-and-left of its marker
        # clears the trend line. One placed directly above is struck through by it.
        left.annotate(rf"$\alpha={a:g}$", (calls[i], coverage[i]), textcoords="offset points",
                      xytext=(-5, 7), ha="right",
                      fontsize=C.style.TICK_FONT_SIZE, color=C.NEUTRAL)
    left.set_xscale("log")
    left.set_xlabel("verifier calls per report (log)")
    left.set_ylabel(cov_axis)
    left.set_xlim(min(calls) * 0.62, max(calls) * 1.55)
    span = max(coverage) - min(coverage)
    # Clamp at 1.0: coverage is a share, and an axis running past it invites a reader to look for
    # headroom that cannot exist.
    left.set_ylim(min(coverage) - span * 0.45, min(1.0, max(coverage) + span * 1.35))
    # The default log formatter sets these as 2x10^4, 3x10^4, 4x10^4, which collide at this
    # width. Verifier calls are counted in thousands. Say so.
    left.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    left.xaxis.set_minor_formatter(plt.NullFormatter())
    left.set_xticks(sorted(calls))
    left.set_axisbelow(True)
    left.grid(True, which="major")
    # A legend only when there is something to tell apart: with every tolerance feasible all
    # markers are one kind, and a two-entry legend would name a state no point is in.
    if any(f != n for f, n in zip(feasible, n_folds)):
        left.plot([], [], marker="o", color=P.TREE, ls="none", label="bound reachable")
        left.plot([], [], marker="X", mfc="white", mec=P.ERROR, mew=1.0, ls="none",
                  label="bound unreachable")
        left.legend(loc="lower right")

    # -- right: is the bound honoured out of fold? ---------------------------
    lim = max(max(alpha), max(miss)) * 1.12
    right.fill_between([0, lim], [0, lim], [lim, lim], color=P.ERROR, alpha=0.07, lw=0,
                       zorder=1)
    right.plot([0, lim], [0, lim], color=C.NEUTRAL, lw=0.9, ls=(0, (1.0, 1.6)), zorder=2,
               label=r"bound (miss rate $=\alpha$)")
    right.annotate("bound violated", (lim * 0.16, lim * 0.88), ha="left", va="center",
                   fontsize=C.style.TICK_FONT_SIZE, color=P.ERROR)
    right.plot(alpha, miss, color=P.TREE, ls="solid", marker="o", ms=4.0, lw=1.0,
               zorder=4, label="held-out miss rate")
    for i, a in enumerate(alpha):
        if feasible[i] != n_folds[i]:
            right.annotate(f"{feasible[i]}/{n_folds[i]} folds\nfeasible", (a, miss[i]),
                           textcoords="offset points", xytext=(8, -4), ha="left", va="top",
                           fontsize=C.style.TICK_FONT_SIZE, color=P.ERROR)
    right.set_xlabel(r"risk tolerance $\alpha$")
    right.set_ylabel("per-report miss rate")
    right.set_xlim(0, lim)
    right.set_ylim(0, lim)
    right.set_axisbelow(True)
    right.grid(True)
    right.legend(loc="lower right")

    C.save(fig, "fig_ch4_cost_coverage")

    cheap, dear = rows[-1], rows[0]
    # Both endpoints are quoted at two decimals, the same rounding the alpha table prints, so the
    # caption and the table cannot be read as disagreeing by a rounding step.

    C.write_figure_tex(
        "fig_ch4_cost_coverage", label="ch4-cost-coverage", script="fig_ch4_cost_coverage.py",
        caption=(
            r"Coverage, verifier calls and held-out miss rate of TreePhenoRAG on HCY for $\alpha$ "
            r"from " + f"{C.f(dear, 'alpha'):.2f}" + r" to " + f"{C.f(cheap, 'alpha'):.2f}"
            + r". Left: coverage against verifier calls per report. Right: held-out miss rate "
            r"against $\alpha$."),
        note=(
            r"Left: coverage (" + cov_source + r") against verifier calls per report on HCY "
            r"(log scale), one point per risk tolerance $\alpha$. Right: per-report miss rate on "
            r"the outer folds against $\alpha$. Points below the diagonal satisfy the bound. "
            r"Every point reruns the full protocol at that $\alpha$ and is measured on its pooled "
            r"out-of-fold traversal over the " + f"{int(C.f(rows[0], 'n_reports'))}"
            + r" reports."))


if __name__ == "__main__":
    main()
