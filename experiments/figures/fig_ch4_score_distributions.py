r"""Figure: do the two pooled scores separate the cases they are thresholding? -- three views.

    python experiments/figures/fig_ch4_score_distributions.py

Two decisions, one panel each, in the order a term meets them during a traversal:

* **Expansion** -- the pooled prune score of every node the traversal scores, split by whether a
  annotated term lies at or below the node. The selected expansion pooling is ``P4`` (mean segment
  confidence) at the modal ``tau_prune`` -- read from ``score_distribution_summary.csv``'s own
  ``pooling``/``threshold`` columns, never assumed.
* **Acceptance** -- the pooled acceptance score of the same nodes, split by whether the node is
  itself an annotated term. The selected acceptance pooling is ``lse_beta1`` at ``tau_accept = 0.99``.

Both scores lie in (0, 1) and pile up against its ends, so every view is on the **logit** of the
score. For ``lse_beta1`` that is the pooled margin (the score is sigmoid(logsumexp(m))). For
``P4`` it is the logit of a mean of sigmoids. The TreePhenoRAG protocol clips at 1e-12 before the transform and
records how many scores it clipped (``n_clipped_to_logit_range``).

Three artifacts, for the author to choose between -- each is complete on its own:

``fig_ch4_score_distributions``    class-normalised histograms on shared logit bins
``fig_ch4_score_distributions_2``  survival curves: share of each class scoring >= t
``fig_ch4_score_distributions_3``  precision-recall curves with the configuration
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

import common as C
import palette as P

SCRIPT = "fig_ch4_score_distributions.py"

DECISIONS = ["expansion", "acceptance"]

# The operator names of tab:tpr-operators (P4, P3$_S$), so a caption and the table agree.
POOLING_TEX = {"P4": "P4, mean", "lse_beta1": "LSE", "P1": "P1, max",
               "P3_S": r"P3$_S$, noisy-OR"}
POOLING_TXT = {"P4": "P4, mean", "lse_beta1": "LSE", "P1": "P1, max", "P3_S": "P3,S, noisy-OR"}

PANEL_TITLE = {"expansion": "Expansion", "acceptance": "Acceptance"}
# The class names of 04_TreePhenoRAG.tex's placeholder for this figure, word for word.
POSITIVE_LABEL = {"expansion": "term or descendant annotated", "acceptance": "term annotated"}
NEGATIVE_LABEL = {"expansion": "neither annotated", "acceptance": "term not annotated"}

# The classes are the palette's two outcomes (rule 2): annotated green, not annotated vermillion --
# hues, so neither curve can be mistaken for the axes, the grid or the dotted threshold.
POSITIVE_STYLE = dict(color=P.POSITIVE, ls="solid", lw=1.3)
NEGATIVE_STYLE = dict(color=P.NEGATIVE, ls=(0, (4.0, 1.6)), lw=1.1)
THRESHOLD_STYLE = dict(color=C.NEUTRAL, lw=0.9, ls=(0, (1.4, 1.4)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p) - math.log1p(-p)


def _load():
    summary = {r["decision"]: r for r in C.table("score_distribution_summary")}
    for d in DECISIONS:
        if d not in summary:
            raise SystemExit(f"score_distribution_summary.csv has no {d} row")
    return summary


def _n(x) -> str:
    return f"{int(x):,}".replace(",", " ")


def _threshold_logit(summary) -> float:
    t = C.f(summary, "threshold_logit")
    return t if t == t else _logit(C.f(summary, "threshold"))


def _two_panels(height=2.5):
    C.style.apply()
    return plt.subplots(1, 2, figsize=(C.WIDTH_FULL, height))


def _title(ax, decision, summary):
    pooling = summary.get("pooling", "")
    ax.set_title(f"{PANEL_TITLE[decision]} ({POOLING_TXT.get(pooling, pooling)})",
                 fontsize=C.style.TICK_FONT_SIZE + 0.5, pad=4)


def _panel_legend(axes):
    """Each panel carries its own legend: the class labels differ per decision.

    The annotated (green) class is listed first although it is plotted second -- drawn last so it
    sits on top of the far larger negative class, read first because it is what the threshold is
    for. Entries that are not a class (the PR view's curve, prevalence, threshold) keep their order.
    """
    positive = tuple(POSITIVE_LABEL.values())
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith(positive))
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                  fontsize=C.style.TICK_FONT_SIZE - 0.5, frameon=False)


def _thresholds_sentence(summary) -> str:
    e, a = summary["expansion"], summary["acceptance"]
    return (r"Vertical lines: $\tau_{\mathrm{prune}} = "
            + _sci(C.f(e, "threshold")) + r"$ ($" + f"{_threshold_logit(e):.1f}" + r"$) "
            r"and $\tau_{\mathrm{accept}} = " + f"{C.f(a, 'threshold'):g}" + r"$ ($"
            + f"{_threshold_logit(a):.1f}" + r"$)")


def _sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _population_sentence(summary) -> str:
    e, a = summary["expansion"], summary["acceptance"]
    # "the most frequent configuration": the thresholds drawn are ONE configuration's, applied to
    # every report in-sample, which is not Table 1's pooled out-of-fold estimator (there each fold
    # has its own tau_prune). Said in the note because the figure needs one threshold to draw.
    return (r"over all " + f"{int(C.f(e, 'n')):,}".replace(",", r"\,") + r" (report, term) "
            r"pairs scored on HCY under the most frequent configuration. Left: expansion score ("
            + POOLING_TEX.get(e["pooling"], e["pooling"])
            + r"), positive if an annotated term lies at or below the term ("
            + f"{C.f(e, 'prevalence') * 100:.1f}" + r"\,\% of pairs). Right: acceptance score ("
            + POOLING_TEX.get(a["pooling"], a["pooling"]) + r"), positive if the term is "
            r"annotated (" + f"{C.f(a, 'prevalence') * 100:.2f}" + r"\,\%)")


def _main_result(summary) -> str:
    e, a = summary["expansion"], summary["acceptance"]
    return (r"\textbf{Both pooled scores separate their classes (ROC AUC "
            + f"{C.f(e, 'auc'):.2f}" + " and " + f"{C.f(a, 'auc'):.2f}"
            + r"), but the acceptance threshold keeps only "
            + f"{C.f(a, 'positive_above_threshold') * 100:.0f}" + r"\,\% of the positives.}")


# -- view 1: class-normalised histograms on the logit scale -------------------

def histograms(summary) -> None:
    """Class-normalised histograms of the logit of both pooled scores."""
    fig, axes = _two_panels()
    for ax, decision in zip(axes, DECISIONS):
        rows = C.table(f"score_logit_{decision}")
        s = summary[decision]
        edges = [C.f(rows[0], "lo")] + [C.f(r, "hi") for r in rows]
        width = edges[1] - edges[0]
        for key, st, label in (("n_negative", NEGATIVE_STYLE, NEGATIVE_LABEL[decision]),
                               ("n_positive", POSITIVE_STYLE, POSITIVE_LABEL[decision])):
            total = sum(C.f(r, key) for r in rows)
            dens = [C.f(r, key) / total / width for r in rows]
            n_class = int(C.f(s, "n_negative" if key == "n_negative" else "n_positive"))
            ax.step(edges, dens + [dens[-1]], where="post", label=f"{label} (n = {_n(n_class)})",
                    zorder=3, **st)
        ax.axvline(_threshold_logit(s), zorder=4, **THRESHOLD_STYLE)
        ax.set_xlabel("pooled score (logit)")
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.28)   # headroom for the legend
        _title(ax, decision, s)
        C.style.despine(ax)
    axes[0].set_ylabel("density within class")
    _panel_legend(axes)
    C.save(fig, "fig_ch4_score_distributions")
    C.write_figure_tex(
        "fig_ch4_score_distributions", label="ch4-score-distributions", script=SCRIPT,
        caption=(r"Distribution of the pooled expansion score (left) and acceptance score (right) "
                 r"per class on HCY."),
        note=(
            r"Class-normalised histograms of the pooled scores on the logit scale, "
            r"$\log(p/(1-p))$, " + _population_sentence(summary) + r". Each class is normalised "
            r"to unit area over 60 shared bins. " + _thresholds_sentence(summary) + r"."))


# -- view 2: survival curves --------------------------------------------------

def survival(summary) -> None:
    """Share of positives and negatives above each threshold, for both pooled scores."""
    fig, axes = _two_panels()
    for ax, decision in zip(axes, DECISIONS):
        rows = C.table(f"score_survival_{decision}")
        s = summary[decision]
        t = [C.f(r, "t_logit") for r in rows]
        ax.plot(t, [C.f(r, "negative_above") for r in rows], zorder=3,
                label=f"{NEGATIVE_LABEL[decision]} (n = {_n(C.f(s, 'n_negative'))})",
                **NEGATIVE_STYLE)
        ax.plot(t, [C.f(r, "positive_above") for r in rows], zorder=3,
                label=f"{POSITIVE_LABEL[decision]} (n = {_n(C.f(s, 'n_positive'))})",
                **POSITIVE_STYLE)
        x = _threshold_logit(s)
        ax.axvline(x, zorder=2, **THRESHOLD_STYLE)
        for key, st in (("positive_above_threshold", POSITIVE_STYLE),
                        ("negative_above_threshold", NEGATIVE_STYLE)):
            y = C.f(s, key)
            ax.plot([x], [y], marker="o", ms=4.2, color=st["color"], mfc="white", mew=1.1,
                    zorder=5)
            ax.annotate(f"{y:.0%}" if y >= 0.01 else f"{y:.1%}", (x, y),
                        textcoords="offset points", xytext=(5, 3), fontsize=C.style.TICK_FONT_SIZE,
                        color=st["color"], zorder=5)
        ax.set_xlabel("threshold t on the pooled score (logit)")
        ax.set_ylim(-0.02, 1.25)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        _title(ax, decision, s)
        C.style.despine(ax)
    axes[0].set_ylabel("share of class with score ≥ t")
    _panel_legend(axes)
    C.save(fig, "fig_ch4_score_distributions_2")
    C.write_figure_tex(
        "fig_ch4_score_distributions_2", label="ch4-score-distributions-2", script=SCRIPT,
        caption=(
            _main_result(summary)),
        note=(
            r"Share of each class whose pooled score is at least $t$, against $t$ on the logit "
            r"scale, " + _population_sentence(summary) + r". " + _thresholds_sentence(summary)
            + r". The marked crossings are the shares of each class the threshold keeps. No "
            r"intervals: the curves count every scored node."))


# -- view 3: precision-recall -------------------------------------------------

def precision_recall(summary) -> None:
    """Precision-recall curves of both pooled scores, with the selected thresholds marked."""
    fig, axes = _two_panels()
    for ax, decision in zip(axes, DECISIONS):
        rows = C.table(f"score_pr_{decision}")
        s = summary[decision]
        ax.plot([C.f(r, "recall") for r in rows], [C.f(r, "precision") for r in rows],
                color=P.TREE, lw=1.3, zorder=3,
                label=f"AP = {C.f(s, 'average_precision'):.2f}")
        prevalence = C.f(s, "prevalence")
        ax.axhline(prevalence, color=C.NEUTRAL, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2,
                   label=f"prevalence = {prevalence:.3f}")
        ax.plot([C.f(s, "recall_at_threshold")], [C.f(s, "precision_at_threshold")],
                marker="*", ms=9, color=P.INK, mfc=P.WHITE, mew=1.0, ls="none", zorder=5,
                label=(f"selected threshold (P = {C.f(s, 'precision_at_threshold'):.2f}, "
                       f"R = {C.f(s, 'recall_at_threshold'):.2f})"))
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("recall")
        _title(ax, decision, s)
        C.style.despine(ax)
    axes[0].set_ylabel("precision")
    _panel_legend(axes)
    C.save(fig, "fig_ch4_score_distributions_3")
    e, a = summary["expansion"], summary["acceptance"]
    C.write_figure_tex(
        "fig_ch4_score_distributions_3", label="ch4-score-distributions-3", script=SCRIPT,
        caption=(
            r"\textbf{At the selected thresholds, expansion reaches precision "
            + f"{C.f(e, 'precision_at_threshold'):.2f}" + r" at recall "
            + f"{C.f(e, 'recall_at_threshold'):.2f}" + r", and acceptance precision "
            + f"{C.f(a, 'precision_at_threshold'):.2f}" + r" at recall "
            + f"{C.f(a, 'recall_at_threshold'):.2f}" + r".}"),
        note=(
            r"Precision against recall of each pooled score as its threshold varies, "
            + _population_sentence(summary) + r". Diamonds mark the selected thresholds. The "
            r"dashed line is the positive-class prevalence, the precision of a random ranking. AP "
            r"is average precision. No intervals: the curves count every scored node."))


def main() -> None:
    """Build this script's thesis artifacts from the stored tables. Exits 3 when a source table is missing."""
    summary = _load()
    missing = False
    for view in (histograms, survival, precision_recall):
        try:
            view(summary)
        except SystemExit as exc:
            # One view's CSV missing must not take the other two with it.
            print(f"  skipped {view.__name__}: {exc}")
            missing = True
    if missing:
        raise SystemExit(C.MISSING_SOURCE_EXIT)


if __name__ == "__main__":
    main()
