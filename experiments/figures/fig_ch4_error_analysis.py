r"""Figure: what TreePhenoRAG's false positives are, as one stacked bar.

    python experiments/figures/fig_ch4_error_analysis.py

Reads ``error_taxonomy.csv``. Buckets follow the precedence order of the metrics section -- a term
is filed in the first bucket it matches. ``no_gold`` and ``invalid`` are not drawn: the output space
is the ontology, so both are empty by design. If either is ever non-empty the script says so
on stdout and the shares stay computed over every false positive, so the bar falls short of 100%
instead of renormalising the problem away.
"""
from __future__ import annotations

import common as C

SCRIPT = "fig_ch4_error_analysis.py"

SEGMENTS = C.FP_RELATION_SEGMENTS


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    tax = {r["bucket"]: int(float(r["count"])) for r in C.table("error_taxonomy")}
    total = sum(tax.values())
    for bucket in ("no_gold", "invalid"):
        if tax.get(bucket, 0):
            print(f"  WARNING: {bucket} holds {tax[bucket]} false positives and is not drawn.")

    fig, ax = C.start(width=C.WIDTH_FULL, height=1.45)
    handles = C.stacked_share_bars(ax, [(f"HCY ({total} FP)", tax)], SEGMENTS)
    ax.set_xlabel("share of false positives (%)")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=C.style.TICK_FONT_SIZE)
    C.save(fig, "fig_ch4_error_analysis")

    C.write_figure_tex(
        "fig_ch4_error_analysis", label="ch4-error-analysis", script=SCRIPT,
        caption=(
            r"Relation of TreePhenoRAG's " + str(total) + r" false positives on HCY to the "
            r"annotated terms of the report."),
        note=(
            r"Pooled out-of-fold predictions, categories of \Cref{sec:metrics} in the order shown. "
            r"Ancestor or descendant: on an annotated term's path. Sibling: shares a direct parent "
            r"with an annotated term. Same branch: shares a first-level organ system. Unrelated: "
            r"none of these."))


if __name__ == "__main__":
    main()
