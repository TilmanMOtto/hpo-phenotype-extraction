r"""Figure: what the retrieval gate buys, as one curve per index.

    python experiments/figures/fig_ch4_retrieval.py

One message: **on the query set the traversal actually issues, pooling a term's descendants is
what buys recall. How the term itself is represented does not matter.** Everything that does not
serve that message is out -- in particular the second gate.

## What was removed from the retrieval-curve analysis's own figure, and why

`treephenorag_retrieval_curves`'s `figC_all_queries` draws five conditions times two gates: `top-k`, and a single global
cosine cutoff `tau`. Ten curves crossing each other is several messages at once, and the second
one costs a paragraph to set up -- each condition needs its *own* tau grid, because R1, R1u and R3 have
unrelated score distributions, so "the same tau" is not a shared setting and the dashed curves are
not comparable with each other at a fixed x. Here only `top-k` is drawn, whose parameter does mean
the same thing in every condition. The tau finding is stated once in the caption.

The chance condition (R5, uniform sampling) is dropped for a different reason: it sits near 0.42 where
the four real conditions sit near 0.92, so including it forces a y-axis on which the comparison the
figure exists for is four overlapping lines in the top tenth. It is quoted as a number in the
caption instead. The full five-condition, two-gate sweep stays in the appendix as the retrieval-curve analysis's own figure.

## The facet counts, and the two design decisions pay off on different ones

This is facet **C**: every (patient, term) pair the pipeline issues -- the annotated terms *and* every
ancestor they have to be reached through. That is the set that bounds an end-to-end run, because a
term whose ancestor was never reached is unreachable however well it scores itself.

On facet C the closure pays (+0.035 and +0.047 at k=5, both p=0.0001) and the representation does
not (-0.003, p=0.57). On facet **A** -- the annotated terms alone, no ancestors -- it is the other way
round: the label+definition+synonyms index beats the synthetic-sentence index by +0.024 (p=0.0004) and the
closure buys nothing (+0.001, p=0.77). Neither result contradicts the other. Pooling a subtree
helps where a query is an interior node standing in for its descendants, which is what an
ancestor query is and what a terminal annotated term is not.

The caption states facet C's numbers because facet C is what the traversal issues. Every one of
them is read out of the CSVs rather than typed.
"""
from __future__ import annotations

import common as C
import palette as P

SCRIPT = "fig_ch4_retrieval.py"

POINTS = "output/treephenorag_retrieval_curves/operating_points.csv"
DELTAS = "output/treephenorag_retrieval_curves/paired_deltas.csv"
RSYNC = ("rsync -rlt '<LEOMED_HOST>:.../output/treephenorag_retrieval_curves/"
         "{operating_points,paired_deltas}.csv' output/treephenorag_retrieval_curves/")

#: Facet C: every (patient, term) pair the pipeline would actually issue.
FACET = "gold + ancestors"

def shipped_k() -> float:
    """The S the traversal runs at, from the TreePhenoRAG protocol's own selected_configuration.json.

    The star marks the chapter's ONE setting. It once sat at k = 5, the point an earlier draft's
    prose quoted, which gave the chapter three different "settings" (S = 10 for every result, S = 3
    recommended for deployment, k = 5 here) -- check X8. Read, never typed.
    """
    import json

    path = C.RESULTS / "selected_configuration.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} -- pull it with exp13_22's tables")
    return float(json.loads(path.read_text(encoding="utf-8"))["S"])


#: Uniform sampling. Not drawn -- see the module docstring -- but quoted in the caption, so the
#: reader still knows what the floor is.
CHANCE_CONDITION = "R5"

# Colour carries the representation, fill carries the closure: the two synthetic-sentence conditions share a hue
# and the two ontology conditions share another, so the 2x2 reads as two pairs, not four
# unrelated lines. They are the index colours of common.INDEX_STYLE and fig_ch4_pipeline. The closed variants are solid and filled, the open ones dashed and hollow --
# which is also the distinction the figure's message is about, so it is the one carried by shape
#, not by hue alone.
CONDITION_STYLE = {
    "R1": dict(color=P.AMBER, ls=(0, (5.0, 1.6)), marker="s", mfc="white",
               label="synthetic sentences"),
    "R1u": dict(color=P.AMBER, ls="solid", marker="s", mfc=P.AMBER,
                label="synthetic sentences + descendant closure"),
    "R3": dict(color=P.EVIDENCE, ls=(0, (5.0, 1.6)), marker="o", mfc="white",
               label="label + definition + synonyms"),
    "R3u": dict(color=P.EVIDENCE, ls="solid", marker="o", mfc=P.EVIDENCE,
                label="label + definition + synonyms + descendant closure"),
}
CONDITION_ORDER = ["R1", "R1u", "R3", "R3u"]

#: The retrieval the chapter runs at: every fold of the TreePhenoRAG protocol's nested CV selected `ontology_r3`
#: (tab_ch4_selection_frequency), i.e. label + definition + synonyms under descendant closure.
#: The star sits on this condition.
TRAVERSAL_CONDITION = "R3u"

X_MAX = 20.0
Y_MIN = 0.55


def _series(rows, arm):
    """``(n_seg, hit, lo, hi)`` for one condition's top-k curve, in increasing cost order."""
    got = [r for r in rows if r["arm"] == arm]
    got.sort(key=lambda r: C.f(r, "mean_bag"))
    return ([C.f(r, "mean_bag") for r in got], [C.f(r, "hit_rate") for r in got],
            [C.f(r, "hit_lo") for r in got], [C.f(r, "hit_hi") for r in got])


def _at_k(rows, arm, k):
    for r in rows:
        if r["arm"] == arm and abs(C.f(r, "param") - k) < 1e-9:
            return r
    return None


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    all_points = C.external_table(POINTS, rsync_hint=RSYNC)
    rows = [r for r in all_points
            if r["figure"] == "C" and r["facet"] == FACET and r["gate"] == "topk"]
    if not rows:
        raise SystemExit(f"{POINTS} carries no figure C / {FACET} / topk rows")

    fig, ax = C.start(width=C.WIDTH_FULL, height=3.2)

    for arm in CONDITION_ORDER:
        n_seg, hit, lo, hi = _series(rows, arm)
        if not n_seg:
            continue
        s = CONDITION_STYLE[arm]
        # The band is the 95% report-clustered bootstrap interval. Light enough that four of them
        # overlapping stays readable. The lines carry the comparison.
        ax.fill_between(n_seg, lo, hi, color=s["color"], alpha=0.10, lw=0, zorder=2)
        # Markers only on the drawn points: k is an integer, so those are the only settings the
        # pipeline can be put into and the joining line is a guide, not a claim about k = 7.3.
        ax.plot(n_seg, hit, color=s["color"], ls=s["ls"], lw=1.2, marker=s["marker"], ms=3.4,
                mfc=s["mfc"], mec=s["color"], mew=0.9, zorder=3, label=s["label"])

    STAR_K = shipped_k()
    star = _at_k(rows, TRAVERSAL_CONDITION, STAR_K)
    if star is None:
        raise SystemExit(f"{POINTS} has no {TRAVERSAL_CONDITION} point at k = {STAR_K:g}")
    star_x, star_y = C.f(star, "mean_bag"), C.f(star, "hit_rate")
    ax.plot([star_x], [star_y], marker="*", ms=12, color=C.NEUTRAL, mfc="white", mew=1.1,
            ls="none", zorder=5)
    # Three lines, centred in the empty band between the curves and the legend (on one line the
    # label ran past the right spine).
    ax.annotate(f"S = {STAR_K:.0f} (used):\n{star_x:.2f} mean segments per pair,\n"
                f"{star_y * 100:.0f} % of pairs with evidence",
                xy=(star_x, star_y), xytext=(0.62 * X_MAX, star_y - 0.18),
                fontsize=C.style.TICK_FONT_SIZE, color=C.NEUTRAL, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C.NEUTRAL, shrinkA=4, shrinkB=6))

    n_pairs = max(int(C.f(r, "n_pairs")) for r in rows)
    ax.set_xlim(0, X_MAX)
    # The axis is cut at 0.55 and says so in the caption. The whole content of this figure is the
    # ordering of four curves that all sit above 0.57 at k=1. On a 0-1 axis they are one band.
    ax.set_ylim(Y_MIN, 1.005)
    ax.set_xlabel("mean segments retrieved per (patient, term) pair")
    ax.set_ylabel("share of pairs with the evidence retrieved")
    ax.set_axisbelow(True)
    C.style.light_grid(ax, axis="both")
    C.style.despine(ax)
    ax.legend(loc="lower right", fontsize=C.style.TICK_FONT_SIZE, frameon=False)

    C.save(fig, "fig_ch4_retrieval")

    C.write_figure_tex(
        "fig_ch4_retrieval", label="ch4-retrieval", script=SCRIPT, source=POINTS,
        caption=(
            r"Retrieval of the evidence segment on HCY for four term representations. Share of "
            r"queried pairs whose evidence is retrieved, against the number of segments retrieved "
            r"per pair."),
        placement="tb",
        note=(
            r"The " + f"{n_pairs:,}".replace(",", r"\,") + r" HCY (patient, term) pairs the "
            r"pipeline queries: every annotated term and each of its ancestors. A pair counts if at "
            r"least one retrieved segment contains the evidence named by the curator. The rank "
            r"cutoff runs over its integer values. Four representations: synthetic sentences or "
            r"label + definition + synonyms, each with and without descendant closure. Star: $S=" + f"{STAR_K:.0f}"
            + r"$ with closure (" + f"{star_x:.2f}" + r" segments per pair, since short reports "
            r"have fewer, evidence retrieved for " + f"{star_y:.2f}" + r" of pairs). The vertical "
            r"axis starts at " + f"{Y_MIN:.2f}" + r"."))


if __name__ == "__main__":
    main()
