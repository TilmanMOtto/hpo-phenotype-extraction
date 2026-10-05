r"""Figure: where TreePhenoRAG's missed annotated terms were lost, as one stacked bar.

    python experiments/figures/fig_ch4_recall_decomposition.py

Reads ``recall_decomposition.csv`` -- the decomposition at the default margin cut (delta_m = 0,
"the verifier answered No on every segment it saw"). The delta_m sweep is not drawn: it moves only
the judgement/pooling boundary and belongs in the text, where the convention can be argued.

The residual bucket is empty on every run so far and is then left out. If it is ever non-empty it
is drawn as a fifth segment and the script says so on stdout -- never dropped, which would
renormalise the other four over a set that is not every false negative.
"""
from __future__ import annotations

import common as C

SCRIPT = "fig_ch4_recall_decomposition.py"

#: The four stages, in the order the reader is asked to see them.
SEGMENTS = [
    ("retrieval", "retrieval", C.BUCKET_STYLE["retrieval"]),
    ("pruning", "pruning", C.BUCKET_STYLE["pruning"]),
    ("judgement", "judgement", C.BUCKET_STYLE["judgement"]),
    ("pooling", "pooling", C.BUCKET_STYLE["pooling"]),
]


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    dec = {r["bucket"]: int(float(r["count"])) for r in C.table("recall_decomposition")}
    total = sum(dec.values())
    if dec.get("residual", 0):
        print(f"  WARNING: residual bucket holds {dec['residual']} of {total} false negatives; "
              f"it is drawn as a fifth segment -- mention it in the caption.")
    counts = dict(dec)
    counts_for_bar = {k: counts.get(k, 0) for k, _, _ in SEGMENTS}
    counts_for_bar["_residual"] = counts.get("residual", 0)

    fig, ax = C.start(width=C.WIDTH_FULL, height=1.45)
    handles = C.stacked_share_bars(
        ax, [(f"HCY ({total} FN)", counts_for_bar)],
        SEGMENTS + ([("_residual", "residual", C.BUCKET_STYLE["residual"])]
                    if counts.get("residual", 0) else []))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=C.style.TICK_FONT_SIZE)
    C.save(fig, "fig_ch4_recall_decomposition")

    C.write_figure_tex(
        "fig_ch4_recall_decomposition", label="ch4-recall-decomposition", script=SCRIPT,
        caption=(
            rf"Stage at which each of TreePhenoRAG's {total} false negatives on HCY was lost."),
        note=(
            r"Pooled out-of-fold predictions. Pruning: no path to the term was expanded. Retrieval: "
            r"the term was visited but none of its retrieved segments is the evidence segment named "
            r"by the curator. Judgement: the evidence was retrieved and the verifier answered "
            r"\emph{No} on every segment. Pooling: at least one segment was answered \emph{Yes}, but "
            r"the pooled score fell below $\tau_{\mathrm{accept}}$."))


if __name__ == "__main__":
    main()
