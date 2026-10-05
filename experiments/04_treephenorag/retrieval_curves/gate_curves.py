"""
The retrieval-curve analysis ``report`` stage, the gate curves, their bootstrap intervals, Figure A and Figure B.

Reads only ``qs_scores.csv.gz`` (written by :mod:`query_sets`), so this whole stage re-runs
locally on a laptop with no model, no context directory and no GPU.

What it computes
----------------
For every (figure, facet, condition, gate, parameter) it reports **one configuration**: a mean bag
size and a pair-level hit rate, each with a 95 % report-clustered bootstrap interval.

* **bag size**, sentences surviving the gate for one (patient, query) pair. ``top-k`` gives
  ``min(k, n_segments)``, which is why the mean at k=5 is ~4.6 and not 5; ``global tau`` gives
  ``#{sim >= tau}``.
* **hit**, 1 if at least one ground truth sentence (true-path rule) survives, else 0. R5's is the
  closed-form *expectation*, in [0, 1].

Both axes are ratios of two sums over the same pairs, so both move when a patient is resampled. The table carries an interval for each and the figures band the hit rate.

Why the tau grid is chosen the way it is
----------------------------------------
``tau`` is a **global** cutoff: one number applied to every query of every patient. Its natural
scale differs between R1, R1u and R3 (three different score distributions), so a shared numeric
grid would put the three curves in unrelated places on the x axis. Instead each condition's grid is
placed *through the cost axis*: since the pooled mean bag size at ``tau`` is
``#{rows with sim >= tau} / n_pairs``, the tau that produces a mean bag of ``b`` is the
``round(b * n_pairs)``-th largest score in the condition. The grid is therefore evenly spaced in the
quantity being plotted, for every condition, without any per-condition hand tuning.

The grid is built once over **all** pairs and then reused inside every facet of Figure B. That is
deliberate and it is what a global cutoff means: inside a depth band the curve's x position is
that band's own mean bag at the cohort-wide tau, not a tau refitted to the band.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hpo_extraction.evaluation.stats.bootstrap import ReportResampler, percentile_ci  # noqa: E402

from query_sets import (  # noqa: E402
    CONDITION_NOTE,
    CONDITIONS,
    CLOSURE_OF,
    DELTAS_FILE,
    DEPTH_BANDS,
    FACET_ALL,
    FIG_A,
    FIG_B,
    FIG_C,
    K_GRID,
    N_CONTINUOUS,
    OPERATING_FILE,
    PAIRS_FILE,
    SCORED_CONDITIONS,
    SCORES_FILE,
    SHIPPED_K,
    SUMMARY_FILE,
    X_MAX,
    r5_bernoulli_hit,
    r5_topk_hit,
)

GATES = ("topk", "tau")

#: Colour carries the Condition, dash pattern carries the GATE, the same split ``scripts/figures``
#: uses, so a reader who has seen fig6 reads these without relearning the encoding. Okabe-Ito,
#: and colour is never the only cue: every condition also owns a marker.
#: Two representation families, each with a plain and a closure condition. The families are separated by
#: hue (synthetic-sentence = blue/green, ontology = pink/orange) so the R1->R1u and R3->R3u steps read as the
#: same intervention applied twice.
CONDITION_STYLE = {
    "R1":  dict(color="#0072B2", marker="o"),
    "R1u": dict(color="#009E73", marker="s"),
    "R3":  dict(color="#CC79A7", marker="D"),
    "R3u": dict(color="#E69F00", marker="v"),
    "R5":  dict(color="#8A8A8A", marker="^"),
}
GATE_LS = {"topk": "solid", "tau": (0, (5.0, 1.6))}

#: Ink used for the two gate swatches in the legend. darker than R5's grey, which a
#: reader would otherwise take for a fifth condition.
GATE_LEGEND_INK = "#222222"

#: The configuration the pipeline actually ships (``top_n=5``, R1), marked in every panel in a
#: shape used for nothing else. White edge so it reads on top of a curve rather than erasing it.
SHIPPED_STYLE = dict(color="black", marker="*", markersize=8.5,
                     markeredgecolor="white", markeredgewidth=0.6)

CONDITION_LEGEND = {
    "R1": "R1  own exemplars",
    "R1u": "R1u  exemplars + closure",
    "R3": "R3  label+def+syn (1 vec)",
    "R3u": "R3u  label+def+syn + closure",
    "R5": "R5  random (control)",
}


# ── Per-condition reduction ─────────────────────────────────────────────────────────

class ConditionPairs:
    """One condition's scores reduced to the per-(patient, query) arrays every gate needs.

    Both gates depend on each pair only through its *best* ground truth sentence, the earliest rank for
    ``top-k``, the highest cosine for ``tau``, because the hit is "at least one survives". The
    full score rows are still needed for the cost axis, which counts every surviving sentence.
    """

    def __init__(self, sub: pd.DataFrame):
        key = sub["patient_id"].astype(str) + "|" + sub["query_hpo"].astype(str)
        codes, uniques = pd.factorize(key, sort=True)
        n = len(uniques)
        self.n_pairs = n
        self.codes = codes

        self.patient = np.empty(n, dtype=object)
        self.query = np.empty(n, dtype=object)
        self.patient[codes] = sub["patient_id"].to_numpy()
        self.query[codes] = sub["query_hpo"].to_numpy()

        self.n_segments = np.zeros(n, dtype=np.int64)
        self.hop = np.zeros(n, dtype=np.int64)
        self.depth = np.zeros(n, dtype=np.int64)
        self.n_segments[codes] = sub["n_segments"].to_numpy()
        self.hop[codes] = sub["hop"].to_numpy()
        self.depth[codes] = sub["depth"].to_numpy()
        self.band = np.empty(n, dtype=object)
        self.band[codes] = sub["depth_band"].astype(str).to_numpy()

        rel = sub["is_gt_relevant"].to_numpy(dtype=bool)
        self.n_relevant = np.bincount(codes[rel], minlength=n).astype(np.int64)
        self.best_gold_rank = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
        np.minimum.at(self.best_gold_rank, codes[rel], sub["rank"].to_numpy()[rel])
        self.best_gold_cos = np.full(n, -np.inf, dtype=np.float64)
        np.maximum.at(self.best_gold_cos, codes[rel], sub["cosine_sim"].to_numpy()[rel])

        self.cos = sub["cosine_sim"].to_numpy(dtype=np.float64)

    def frame(self, arm: str) -> pd.DataFrame:
        """The pairs as a data frame, labelled with the condition name ``arm``."""
        return pd.DataFrame({
            "arm": arm,
            "patient_id": self.patient,
            "query_hpo": self.query,
            "hop": self.hop,
            "depth": self.depth,
            "depth_band": self.band,
            "n_segments": self.n_segments,
            "n_relevant": self.n_relevant,
            "best_gold_rank": self.best_gold_rank,
            "best_gold_cos": self.best_gold_cos,
        })


def tau_grid(cos: np.ndarray, n_pairs: int, x_max: float, n_points: int) -> np.ndarray:
    """Global cutoffs placed so the pooled mean bag size walks an even grid up to *x_max*.

    ``mean_bag(tau) = #{sim >= tau} / n_pairs``, so the tau giving mean bag ``b`` is the
    ``round(b * n_pairs)``-th largest score. Descending tau, i.e. ascending bag size.
    """
    order = np.sort(cos)[::-1]
    targets = np.linspace(x_max / n_points, x_max, n_points)
    m = np.clip(np.round(targets * n_pairs).astype(np.int64), 1, len(order))
    return order[m - 1]


def survivor_counts(codes: np.ndarray, cos: np.ndarray, taus_desc: np.ndarray,
                    n_pairs: int) -> np.ndarray:
    """``surv[pair, j]`` = sentences of that pair with ``sim >= taus_desc[j]``.

    One ``searchsorted`` + reverse cumulative sum over the whole score table, not one
    boolean mask per tau: the grid has 120 points and the table has hundreds of thousands of rows.
    """
    taus_asc = np.ascontiguousarray(taus_desc[::-1])
    bucket = np.searchsorted(taus_asc, cos, side="right")   # #taus <= cos, in [0, J]
    hist = np.zeros((n_pairs, len(taus_asc) + 1), dtype=np.int64)
    np.add.at(hist, (codes, bucket), 1)
    # cos >= taus_asc[j]  <=>  j < bucket, so surv_asc[:, j] = sum over bucket > j.
    tail = np.cumsum(hist[:, ::-1], axis=1)[:, ::-1]
    return tail[:, 1:][:, ::-1]


# ── Bootstrap ─────────────────────────────────────────────────────────────────

class PatientBootstrap:
    """Report-clustered resampling, shared across every condition, gate, facet and parameter.

    The sampling unit is the **patient**. One patient contributes many (patient, query) pairs and
    they stand or fall together, so both the numerator and the denominator of every rate are
    resampled with the patient, not independently.

    ``ReportResampler`` fixes the ``(n_resamples x n_patients)`` index matrix. This class turns it
    once into a **count** matrix, which makes every curve a matrix product instead of a Python
    loop over draws. Reusing one instance is what makes the intervals on two conditions comparable:
    they come from the identical draws.
    """

    def __init__(self, patients, n_resamples: int = 10_000, seed: int = 0, ci: float = 0.95):
        self.patients = list(patients)
        self.row_of = {p: i for i, p in enumerate(self.patients)}
        self.ci = ci
        self.resampler = ReportResampler(len(self.patients), n_resamples=n_resamples, seed=seed)
        idx = self.resampler.indices
        n_pat = len(self.patients)
        offsets = np.arange(idx.shape[0], dtype=np.int64)[:, None] * n_pat
        flat = np.bincount((idx + offsets).ravel(), minlength=idx.shape[0] * n_pat)
        self.counts = flat.reshape(idx.shape[0], n_pat).astype(np.float64)

    def rows(self, patient_ids) -> np.ndarray:
        """Row indices of *patient_ids* in the bootstrap matrix."""
        return np.fromiter((self.row_of[p] for p in patient_ids), dtype=np.int64,
                           count=len(patient_ids))

    def curve(self, pat_rows: np.ndarray, hit: np.ndarray, bag: np.ndarray) -> dict:
        """Point estimates and percentile intervals for a whole parameter grid at once.

        *hit* and *bag* are ``(n_pairs, n_params)``; *pat_rows* maps each pair to its patient.
        """
        n_pat = len(self.patients)
        n_par = hit.shape[1]
        num_hit = np.zeros((n_pat, n_par), dtype=np.float64)
        num_bag = np.zeros((n_pat, n_par), dtype=np.float64)
        np.add.at(num_hit, pat_rows, hit)
        np.add.at(num_bag, pat_rows, bag)
        den = np.bincount(pat_rows, minlength=n_pat).astype(np.float64)

        total = den.sum()
        point_hit = num_hit.sum(axis=0) / total
        point_bag = num_bag.sum(axis=0) / total

        draw_den = self.counts @ den
        with np.errstate(invalid="ignore", divide="ignore"):
            draw_hit = (self.counts @ num_hit) / draw_den[:, None]
            draw_bag = (self.counts @ num_bag) / draw_den[:, None]
        draw_hit[draw_den == 0] = np.nan
        draw_bag[draw_den == 0] = np.nan

        hit_lo, hit_hi, bag_lo, bag_hi = [], [], [], []
        for j in range(n_par):
            lo, hi = percentile_ci(draw_hit[:, j], self.ci)
            hit_lo.append(lo)
            hit_hi.append(hi)
            lo, hi = percentile_ci(draw_bag[:, j], self.ci)
            bag_lo.append(lo)
            bag_hi.append(hi)
        return {
            "hit_rate": point_hit, "hit_lo": np.array(hit_lo), "hit_hi": np.array(hit_hi),
            "mean_bag": point_bag, "bag_lo": np.array(bag_lo), "bag_hi": np.array(bag_hi),
            "n_pairs": int(len(pat_rows)), "n_patients": int((den > 0).sum()),
            # Kept so two conditions can be DIFFERENCED on the same draws. Two marginal intervals
            # cannot answer "is R3 above R1": they overlap freely for paired measurements whose
            # difference is nowhere near zero, because the shared cohort variance sits in both.
            "draws_hit": draw_hit,
        }

    def paired_delta(self, draws_a: np.ndarray, draws_b: np.ndarray,
                     point_a: np.ndarray, point_b: np.ndarray) -> dict:
        """``a - b`` with a percentile interval, from the draws both were computed on.

        Valid only because ``self.counts`` is the same matrix for every condition: draw *d* of condition A and
        draw *d* of condition B are the same resampled cohort, so their difference removes the cohort
        variance the two marginal intervals share.
        """
        diff = draws_a - draws_b
        lo, hi = zip(*(percentile_ci(diff[:, j], self.ci) for j in range(diff.shape[1])))
        return {
            "delta": point_a - point_b,
            "delta_lo": np.array(lo),
            "delta_hi": np.array(hi),
            # Fraction of draws on the other side of zero, a two-sided bootstrap p, floored at
            # 1/n_resamples because 0 draws out of 10 000 is not evidence of p = 0.
            "p_two_sided": np.array([
                max(2 * min((diff[:, j] > 0).mean(), (diff[:, j] < 0).mean()),
                    1.0 / diff.shape[0])
                for j in range(diff.shape[1])
            ]),
        }


# ── Configurations ──────────────────────────────────────────────────────────

def _gate_matrices(arm: str, pairs: ConditionPairs, sel: np.ndarray, surv: np.ndarray,
                   taus: np.ndarray, p_grid: np.ndarray):
    """``{gate: (params, hit, bag)}`` for one condition restricted to the pairs in *sel*."""
    n = pairs.n_segments[sel].astype(np.float64)
    g = pairs.n_relevant[sel].astype(np.float64)
    ks = np.asarray(K_GRID, dtype=np.float64)

    if arm == "R5":
        hit_k = np.column_stack([r5_topk_hit(n, g, int(k)) for k in K_GRID])
        bag_k = np.minimum(ks[None, :], n[:, None])
        hit_c = np.column_stack([r5_bernoulli_hit(g, float(p)) for p in p_grid])
        bag_c = p_grid[None, :] * n[:, None]
        return {"topk": (np.asarray(K_GRID, float), hit_k, bag_k),
                "tau": (p_grid, hit_c, bag_c)}

    rank = pairs.best_gold_rank[sel].astype(np.float64)
    hit_k = (rank[:, None] <= ks[None, :]).astype(np.float64)
    bag_k = np.minimum(ks[None, :], n[:, None])

    cos = pairs.best_gold_cos[sel]
    hit_t = (cos[:, None] >= taus[None, :]).astype(np.float64)
    bag_t = surv[sel].astype(np.float64)
    return {"topk": (np.asarray(K_GRID, float), hit_k, bag_k),
            "tau": (taus, hit_t, bag_t)}


#: Every condition is differenced against this one, the shipped system is the thing a change has to
#: beat, so the sign of a delta reads as "better than what we run today".
BASELINE_CONDITION = "R1"


def operating_points(pairs_by_condition: dict, surv_by_condition: dict, taus_by_condition: dict,
                     p_grid: np.ndarray, boot: PatientBootstrap
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(operating points, paired deltas)``.

    The first is one row per (figure, facet, condition, gate, parameter), what both figures are drawn
    from. The second differences every condition against :data:`BASELINE_ARM` **on the same bootstrap
    draws**, which is the only way to read the gap between two curves: the marginal intervals of
    two conditions measured on one cohort share that cohort's variance and overlap even when the paired
    difference is far from zero.

    Deltas are computed for the ``top-k`` gate only. It is the one gate whose parameter means the
    same thing in every condition. The tau grids are per-condition by design and R5's continuous
    parameter is a keep-probability, so differencing them at equal index would compare two
    different settings.
    """
    rows: list[dict] = []
    delta_rows: list[dict] = []
    draws: dict[tuple, tuple] = {}

    facets = [("A", "gold terms", lambda p: p.hop == 0)]
    facets += [("B", f"depth {b}", (lambda b: (lambda p: p.band == b))(b)) for b in DEPTH_BANDS]
    facets += [("C", FACET_ALL, lambda p: np.ones(p.n_pairs, dtype=bool))]

    for arm in CONDITIONS:
        src = pairs_by_condition[BASELINE_CONDITION if arm == "R5" else arm]
        for figure, facet, mask_fn in facets:
            sel = np.flatnonzero(mask_fn(src))
            if len(sel) == 0:
                continue
            pat_rows = boot.rows(src.patient[sel])
            mats = _gate_matrices(arm, src, sel,
                                  surv_by_condition.get(arm), taus_by_condition.get(arm), p_grid)
            for gate, (params, hit, bag) in mats.items():
                out = boot.curve(pat_rows, hit, bag)
                if gate == "topk":
                    draws[(figure, facet, arm)] = (out["draws_hit"], out["hit_rate"], params)
                for j, param in enumerate(params):
                    rows.append({
                        "figure": figure, "facet": facet, "arm": arm, "gate": gate,
                        "param": float(param),
                        "mean_bag": out["mean_bag"][j],
                        "bag_lo": out["bag_lo"][j], "bag_hi": out["bag_hi"][j],
                        "hit_rate": out["hit_rate"][j],
                        "hit_lo": out["hit_lo"][j], "hit_hi": out["hit_hi"][j],
                        "n_pairs": out["n_pairs"], "n_patients": out["n_patients"],
                    })

    # Two families of contrast, both paired on the same draws:
    #   * every condition against the shipped one, "is this worth switching to";
    #   * every closure condition against ITS OWN base, the closure effect, held apart from the
    #     representation. R1u-R1 and R3u-R3 are the same intervention on two indices, and that
    #     comparison is the whole reason the 2x2 exists. Subtracting two deltas-vs-R1 would give
    #     The point estimate but no interval, since the two are not independent.
    contrasts = [(arm, BASELINE_CONDITION) for arm in CONDITIONS if arm != BASELINE_CONDITION]
    contrasts += [(closed, base) for closed, base in CLOSURE_OF.items()
                  if base != BASELINE_CONDITION]

    for (figure, facet, arm), (d_condition, p_condition, params) in draws.items():
        for contrast_condition, reference in contrasts:
            if contrast_condition != arm:
                continue
            base = draws.get((figure, facet, reference))
            if base is None:
                continue
            d_base, p_base, _ = base
            delta = boot.paired_delta(d_condition, d_base, p_condition, p_base)
            for j, param in enumerate(params):
                delta_rows.append({
                    "figure": figure, "facet": facet, "arm": arm, "vs": reference,
                    "gate": "topk", "param": float(param),
                    "delta": delta["delta"][j],
                    "delta_lo": delta["delta_lo"][j], "delta_hi": delta["delta_hi"][j],
                    "p_two_sided": delta["p_two_sided"][j],
                })
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


# ── Figures ───────────────────────────────────────────────────────────────────

_RC = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.4,
    "legend.frameon": False, "figure.constrained_layout.use": True,
}


def _draw_panel(ax, table: pd.DataFrame, x_max: float, band_alpha: float = 0.13) -> None:
    for arm in CONDITIONS:
        for gate in GATES:
            sub = table[(table["arm"] == arm) & (table["gate"] == gate)]
            sub = sub[sub["mean_bag"] <= x_max].sort_values("mean_bag")
            if sub.empty:
                continue
            st = CONDITION_STYLE[arm]
            x = sub["mean_bag"].to_numpy()
            ax.fill_between(x, sub["hit_lo"], sub["hit_hi"],
                            color=st["color"], alpha=band_alpha, linewidth=0)
            ax.plot(x, sub["hit_rate"].to_numpy(), color=st["color"], linestyle=GATE_LS[gate],
                    marker=st["marker"] if gate == "topk" else None,
                    markerfacecolor="white" if gate == "topk" else None,
                    markeredgecolor=st["color"], zorder=3)
    shipped = table[(table["arm"] == "R1") & (table["gate"] == "topk")
                    & (np.isclose(table["param"], SHIPPED_K))]
    if not shipped.empty:
        ax.plot(shipped["mean_bag"], shipped["hit_rate"], **SHIPPED_STYLE, linestyle="none",
                zorder=5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.03)
    ax.grid(True, color="#B8B8B8", linewidth=0.35, alpha=0.55)
    ax.set_axisbelow(True)


def _legend_handles():
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CONDITION_STYLE[a]["color"], marker=CONDITION_STYLE[a]["marker"],
                      markerfacecolor="white", label=CONDITION_LEGEND[a]) for a in CONDITIONS]
    handles += [
        Line2D([], [], color=GATE_LEGEND_INK, linestyle=GATE_LS["topk"],
               label=r"indexed by $k$ (top-$k$)"),
        Line2D([], [], color=GATE_LEGEND_INK, linestyle=GATE_LS["tau"],
               label=r"indexed by $\tau$ (global cutoff)"),
        Line2D([], [], linestyle="none", label=r"shipped gate: $k=5$, R1", **SHIPPED_STYLE),
    ]
    return handles


def plot_figure_a(points: pd.DataFrame, path: Path, x_max: float = X_MAX) -> None:
    """Plot retrieval of the evidence segment against segments forwarded, all annotated terms, and save it."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        _draw_panel(ax, points[(points["figure"] == "A")], x_max)
        ax.set_xlabel("mean bag size — sentences forwarded per (patient, term) pair")
        ax.set_ylabel("terminal hit rate")
        ax.legend(handles=_legend_handles(), loc="lower right", ncol=1)
        fig.savefig(path.with_suffix(".png"), dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    print(f"      -> {path.with_suffix('.png').name} / {path.with_suffix('.pdf').name}")


def plot_figure_c(points: pd.DataFrame, path: Path, x_max: float = X_MAX) -> None:
    """Figure A's axes over the **whole** query set, every annotated term and every ancestor.

    This is the operating curve of the retrieval stage as the pipeline actually issues it: the
    pipeline queries `target_symptoms_all.csv`, i.e. The annotated terms *plus* their ancestors, so a
    number taken here is the one that bounds an end-to-end run. Figure A's is the ceiling on the
    terminal terms alone and reads higher. The y-axis label says which set this is.
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        _draw_panel(ax, points[points["figure"] == "C"], x_max)
        sub = points[points["figure"] == "C"]
        n_pairs = int(sub["n_pairs"].max()) if not sub.empty else 0
        ax.set_xlabel("mean bag size — sentences forwarded per (patient, term) pair")
        ax.set_ylabel(f"hit rate — all {n_pairs} queries")
        ax.legend(handles=_legend_handles(), loc="lower right", ncol=1)
        fig.savefig(path.with_suffix(".png"), dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    print(f"      -> {path.with_suffix('.png').name} / {path.with_suffix('.pdf').name}")


def plot_figure_b(points: pd.DataFrame, path: Path, x_max: float = X_MAX) -> None:
    """Plot the same curves per depth band of the annotated term, and save it."""
    facets = [f"depth {b}" for b in DEPTH_BANDS]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8), sharex=True, sharey=True)
        flat = axes.ravel()
        n_cols = axes.shape[1]
        for ax, facet in zip(flat, facets):
            sub = points[(points["figure"] == "B") & (points["facet"] == facet)]
            _draw_panel(ax, sub, x_max)
            n_pairs = int(sub["n_pairs"].max()) if not sub.empty else 0
            ax.set_title(f"{facet}  ({n_pairs} pairs)")
        # `sharex` hides the tick labels of every panel that has one below it. The grid is not
        # full, so the last panel in a short column has nothing below it and would be left with a
        # bare, unlabelled axis.
        for col in range(n_cols):
            in_col = [i for i in range(len(facets)) if i % n_cols == col]
            if in_col:
                flat[max(in_col)].tick_params(labelbottom=True)
        for ax in flat[len(facets):]:
            ax.axis("off")
            ax.legend(handles=_legend_handles(), loc="center", ncol=1)
        fig.supxlabel("mean bag size — sentences forwarded per (patient, term) pair", fontsize=9)
        fig.supylabel("hit rate (true-path rule)", fontsize=9)
        fig.savefig(path.with_suffix(".png"), dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    print(f"      -> {path.with_suffix('.png').name} / {path.with_suffix('.pdf').name}")


# ── Summary ───────────────────────────────────────────────────────────────────

def _at_shipped(points: pd.DataFrame, figure: str, facet: str) -> pd.DataFrame:
    sub = points[(points["figure"] == figure) & (points["facet"] == facet)
                 & (points["gate"] == "topk") & (np.isclose(points["param"], SHIPPED_K))]
    return sub.set_index("arm")


def _cost_to_match(points: pd.DataFrame, figure: str, facet: str, arm: str,
                   target: float) -> float:
    """Mean bag size at which *condition*'s top-k curve first reaches *target*, linearly interpolated.

    The cost axis exists so a difference in hit rate can be quoted as a price. NaN when the condition
    never reaches the target inside the drawn range, which is itself the answer.
    """
    sub = points[(points["figure"] == figure) & (points["facet"] == facet)
                 & (points["arm"] == arm) & (points["gate"] == "topk")].sort_values("mean_bag")
    x = sub["mean_bag"].to_numpy()
    y = sub["hit_rate"].to_numpy()
    hit = np.flatnonzero(y >= target)
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    if i == 0 or y[i] == y[i - 1]:
        return float(x[i])
    return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def write_summary(points: pd.DataFrame, deltas: pd.DataFrame, counts: dict, gold: dict,
                  r3: dict, out_dir: Path, n_resamples: int, seed: int,
                  ungated_bag: float, x_max: float) -> None:
    """Write a Markdown summary of the curves, the paired differences and the inputs to *out_dir*."""
    lines: list[str] = []
    add = lines.append
    add("# exp00_07 — five query-set variants on one cost axis\n")
    add("Retrieval scores a sentence against a *representation of the query term*. Five of them — "
        "a 2x2 of representation (LLM exemplar corpus vs ontology text) by pooling (own vs "
        "descendant closure), plus chance — with one max-pooled-cosine scoring rule, one gold "
        "standard and one cost axis.\n")

    add("## Ground truth\n")
    add("Curated HCY gold, segment-level, true-path rule. Negatives assume the annotation was "
        "exhaustive for these patients, which is what the curation pass makes safe.\n")
    add("| quantity | value |")
    add("|---|---|")
    for key in ("gold_source", "n_candidates", "n_excluded_by_policy", "n_unanchored",
                "n_folded_duplicates", "n_excluded_by_anchor_how", "n_patients",
                "n_annotations", "n_annotated_hpos", "anchor_how"):
        if key in gold:
            add(f"| `{key}` | {gold[key]} |")
    for key in ("n_query_hpos", "n_pairs", "n_gold_pairs", "n_leaf_queries", "n_ctx_hpos",
                "n_ctx_sentences"):
        if key in counts:
            add(f"| `{key}` | {counts[key]} |")
    add("")

    add("## The arms\n")
    add("| arm | representation R(v) | note |")
    add("|---|---|---|")
    add("| R1 | `U_v` | " + CONDITION_NOTE["R1"] + " |")
    add("| R1u | `U_v` union `U_y` for every y in desc_closure(v) | " + CONDITION_NOTE["R1u"] + " |")
    add("| R3 | `embed(label + \" \" + definition + \" \" + synonyms)` | " + CONDITION_NOTE["R3"] + " |")
    add("| R3u | R3(v) union R3(y) for every y in desc_closure(v) | " + CONDITION_NOTE["R3u"] + " |")
    add("| R5 | none | " + CONDITION_NOTE["R5"] + " |")
    add("")
    add("R1u and R3u pool over the **identical** node set — `build_union_members` at `kinf`, "
        "restricted to terms the exemplar corpus covers — so the two closure arms differ only in "
        "the representation being pooled, not in the pool.\n")
    if r3:
        add(f"R3/R3u template: `{r3.get('template')}`, embedded over the whole closure universe "
            f"({r3.get('universe')} terms, of which {r3.get('n_query_terms')} are ever queried) — "
            f"{r3.get('n_terms')} terms, "
            f"{r3.get('n_without_definition')} with no `def:`, {r3.get('n_label_only')} "
            f"label-only (median {r3.get('median_chars')} characters).\n")
    else:
        add("R3 provenance not recorded — this report was drawn from a scores file written "
            "without the `compute` stage.\n")

    add("## Figure A — gold terms, at the shipped gate (k = 5)\n")
    add("| arm | mean bag | terminal hit rate | 95 % CI |")
    add("|---|---|---|---|")
    tab = _at_shipped(points, "A", "gold terms")
    for arm in CONDITIONS:
        if arm in tab.index:
            r = tab.loc[arm]
            add(f"| {arm} | {r['mean_bag']:.2f} | {r['hit_rate']:.3f} | "
                f"[{r['hit_lo']:.3f}, {r['hit_hi']:.3f}] |")
    add("")

    add("## Figure B — ancestor closure, at the shipped gate (k = 5)\n")
    add("Facets are the query term's **absolute depth** below `HP:0000118` (BFS, "
        "`tree.depth_dict`): band 1 is an organ-system node, band 6+ a specific finding.\n")
    header = "| depth band | pairs | " + " | ".join(CONDITIONS) + " |"
    add(header)
    add("|---" * (2 + len(CONDITIONS)) + "|")
    for band in DEPTH_BANDS:
        tab = _at_shipped(points, "B", f"depth {band}")
        if tab.empty:
            continue
        n_pairs = int(tab["n_pairs"].max())
        cells = []
        for arm in CONDITIONS:
            cells.append(f"{tab.loc[arm, 'hit_rate']:.3f}" if arm in tab.index else "—")
        add(f"| {band} | {n_pairs} | " + " | ".join(cells) + " |")
    add("")

    tab = _at_shipped(points, "C", FACET_ALL)
    if not tab.empty:
        n_all = int(tab["n_pairs"].max())
        n_gold = counts.get("n_gold_pairs", 0)
        add("## Figure C — the whole query set, at the shipped gate (k = 5)\n")
        add(f"All **{n_all}** (patient, term) pairs the pipeline would issue — the {n_gold} gold "
            f"terms *and* every ancestor of them — pooled rather than split by depth. Figure A is "
            f"the {n_gold} terminal terms on their own and Figure B is this same set divided into "
            f"depth bands, so the three overlap by construction: **this** is the number that "
            f"bounds an end-to-end run, because `target_symptoms_all.csv` is what the pipeline "
            f"actually queries. It is lower than Figure A's, and quoting A's in its place would "
            f"overstate the retrieval ceiling.\n")
        add("| arm | mean bag | hit rate | 95 % CI | vs Figure A |")
        add("|---|---|---|---|---|")
        fig_a = _at_shipped(points, "A", "gold terms")
        for arm in CONDITIONS:
            if arm not in tab.index:
                continue
            r = tab.loc[arm]
            gap = (f"{r['hit_rate'] - fig_a.loc[arm, 'hit_rate']:+.3f}"
                   if arm in fig_a.index else "—")
            add(f"| {arm} | {r['mean_bag']:.2f} | {r['hit_rate']:.3f} | "
                f"[{r['hit_lo']:.3f}, {r['hit_hi']:.3f}] | {gap} |")
        add("")

    add("## Paired differences vs R1, at the shipped gate (k = 5)\n")
    add("Differenced on the **same** bootstrap draws. Two marginal intervals measured on one "
        "cohort share that cohort's variance and overlap freely even when the paired difference "
        "is nowhere near zero, so this table — not the two intervals above — is what says whether "
        "an arm beats the shipped one. `top-k` only: it is the one gate whose parameter means the "
        "same thing in every arm, since the tau grids are per-arm by construction and R5's "
        "continuous parameter is a keep-probability.\n")
    at_k5 = deltas[np.isclose(deltas["param"], SHIPPED_K)] if len(deltas) else deltas
    add("| figure | facet | arm | delta vs R1 | 95 % CI | bootstrap p |")
    add("|---|---|---|---|---|---|")
    for _, r in at_k5[at_k5["vs"] == BASELINE_CONDITION].iterrows():
        add(f"| {r['figure']} | {r['facet']} | {r['arm']} | {r['delta']:+.3f} | "
            f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}] | {r['p_two_sided']:.4f} |")
    add("")

    closure = at_k5[at_k5["vs"] != BASELINE_CONDITION]
    add("### The closure effect, held apart from the representation\n")
    add("The same intervention — pool the term's vectors with its whole descendant closure — "
        "applied to each index over the **identical** node set. Read the two rows of a facet "
        "against each other: they say whether the closure's value belongs to the pooling or to "
        "the exemplar corpus it was discovered on. `R1u - R1` is the R1u row of the table above; "
        "the rest are differenced against their own base rather than against R1, because "
        "subtracting two deltas-vs-R1 gives a point estimate with no interval.\n")
    add("| figure | facet | contrast | delta | 95 % CI | bootstrap p |")
    add("|---|---|---|---|---|---|")
    for closed, base in CLOSURE_OF.items():
        src = at_k5[(at_k5["arm"] == closed) & (at_k5["vs"] == base)]
        for _, r in src.iterrows():
            add(f"| {r['figure']} | {r['facet']} | {closed} - {base} | {r['delta']:+.3f} | "
                f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}] | {r['p_two_sided']:.4f} |")
    if closure.empty:
        add("| — | — | (no non-baseline closure pair) | — | — | — |")
    add("")

    add("## What the cost axis prices\n")
    tab = _at_shipped(points, "A", "gold terms")
    if "R1" in tab.index:
        target = float(tab.loc["R1", "hit_rate"])
        base = _cost_to_match(points, "A", "gold terms", "R1", target)
        add(f"Mean bag size at which each arm's top-k curve first reaches **{target:.3f}** — the "
            f"hit rate R1 buys at k = 5. This is the whole point of putting the arms on a cost "
            f"axis: a difference in hit rate becomes a difference in sentences forwarded.\n")
        add("| arm | mean bag to match R1 | relative cost |")
        add("|---|---|---|")
        for arm in CONDITIONS:
            cost = _cost_to_match(points, "A", "gold terms", arm, target)
            if not np.isfinite(cost):
                add(f"| {arm} | never, inside the drawn range | — |")
                continue
            rel = "—" if arm == "R1" or not np.isfinite(base) else f"{cost / base:.2f}x"
            add(f"| {arm} | {cost:.2f} | {rel} |")
        add("")

    add("## Method\n")
    add(f"- **Gates.** `top-k` keeps the k best-ranked sentences (k integer, so only the marked "
        f"points exist). `global tau` keeps `sim >= tau`, one cutoff shared by every query.\n")
    add(f"- **Cost axis.** Mean sentences surviving the gate per (patient, term) pair. At k=5 "
        f"this is below 5 because short reports run out of sentences.\n")
    add(f"- **Hit.** Pair-level: at least one gold sentence survives. `AnyYesAggregator` calls a "
        f"term positive on a single Yes, so a second gold sentence buys nothing downstream.\n")
    add(f"- **Uncertainty.** {n_resamples} report-clustered bootstrap draws, seed {seed}, "
        f"`evaluation.stats.bootstrap.ReportResampler`. The sampling unit is the **patient**: "
        f"one patient's terms stand or fall together, and numerator and denominator are "
        f"resampled with it. One resampler is shared by every arm, gate and facet, so the "
        f"intervals are comparable rather than merely simultaneous.\n")
    add(f"- **R5 is exact.** Its top-k hit rate is `1 - C(n-g, k')/C(n, k')` and its continuous "
        f"arm is `1 - (1-p)^g`; no random draw is taken, so the only spread in its band is the "
        f"cohort's.\n")
    add(f"- **The x axis stops at {x_max:g}.** Forwarding the whole report — no gate at all — "
        f"would average {ungated_bag:.1f} sentences per pair, and beyond ~{x_max:g} the arms are "
        f"indistinguishable, so the axis stops here and the ungated cost is stated instead.\n")

    (out_dir / SUMMARY_FILE).write_text("\n".join(lines), encoding="utf-8")
    print(f"      -> {SUMMARY_FILE}")


# ── Stage ─────────────────────────────────────────────────────────────────────

def run_report(output_dir: str | Path, n_resamples: int = 10_000, seed: int = 0,
               ci: float = 0.95, x_max: float = X_MAX) -> dict:
    """Gate curves + bootstrap intervals + Figure A + Figure B, from ``qs_scores.csv.gz``."""
    out_dir = Path(output_dir)
    print("[1/5] Loading scores...")
    scores = pd.read_csv(out_dir / SCORES_FILE, compression="gzip")
    scores["depth_band"] = scores["depth_band"].astype(str)
    present = [a for a in SCORED_CONDITIONS if a in set(scores["arm"])]
    if set(present) != set(SCORED_CONDITIONS):
        gone = sorted(set(SCORED_CONDITIONS) - set(present))
        raise ValueError(f"{SCORES_FILE} is missing arms: {gone}")

    print("[2/5] Reducing to (patient, query) pairs...")
    pairs_by_condition = {arm: ConditionPairs(scores[scores["arm"] == arm].reset_index(drop=True))
                    for arm in SCORED_CONDITIONS}
    reference = pairs_by_condition["R1"]
    for arm, p in pairs_by_condition.items():
        if p.n_pairs != reference.n_pairs or not np.array_equal(p.n_relevant,
                                                               reference.n_relevant):
            raise AssertionError(
                f"arm {arm} does not score the same pairs as R1 — the arms would not be paired")
    patients = sorted(set(reference.patient))
    pd.concat([p.frame(a) for a, p in pairs_by_condition.items()], ignore_index=True).to_csv(
        out_dir / PAIRS_FILE, index=False)

    print("[3/5] Placing the global tau grids...")
    taus_by_condition, surv_by_condition = {}, {}
    for arm, p in pairs_by_condition.items():
        taus = tau_grid(p.cos, p.n_pairs, x_max, N_CONTINUOUS)
        taus_by_condition[arm] = taus
        surv_by_condition[arm] = survivor_counts(p.codes, p.cos, taus, p.n_pairs)
    mean_n = float(reference.n_segments.mean())
    p_max = min(1.0, x_max / mean_n)
    p_grid = np.linspace(p_max / N_CONTINUOUS, p_max, N_CONTINUOUS)
    print(f"      {mean_n:.1f} sentences per report on average; R5 sweeps p up to {p_max:.3f}")

    print(f"[4/5] Bootstrapping ({n_resamples} draws over {len(patients)} patients)...")
    boot = PatientBootstrap(patients, n_resamples=n_resamples, seed=seed, ci=ci)
    points, deltas = operating_points(pairs_by_condition, surv_by_condition, taus_by_condition, p_grid, boot)
    points.to_csv(out_dir / OPERATING_FILE, index=False)
    deltas.to_csv(out_dir / DELTAS_FILE, index=False)
    print(f"      -> {OPERATING_FILE} ({len(points)} operating points)")
    print(f"      -> {DELTAS_FILE} ({len(deltas)} paired deltas vs {BASELINE_CONDITION})")

    print("[5/5] Drawing Figure A, Figure B and Figure C...")
    plot_figure_a(points, out_dir / FIG_A, x_max)
    plot_figure_b(points, out_dir / FIG_B, x_max)
    plot_figure_c(points, out_dir / FIG_C, x_max)

    gold = json.loads((out_dir / "gold_provenance.json").read_text()) \
        if (out_dir / "gold_provenance.json").is_file() else {}
    r3 = json.loads((out_dir / "r3_provenance.json").read_text()) \
        if (out_dir / "r3_provenance.json").is_file() else {}
    counts = {
        "n_query_hpos": int(len(set(reference.query))),
        "n_pairs": int(reference.n_pairs),
        "n_gold_pairs": int((reference.hop == 0).sum()),
        "n_patients": len(patients),
    }
    write_summary(points, deltas, counts, gold, r3, out_dir, n_resamples, seed,
                  ungated_bag=mean_n, x_max=x_max)

    metrics: dict[str, float] = {}
    tab = _at_shipped(points, "A", "gold terms")
    for arm in CONDITIONS:
        if arm in tab.index:
            metrics[f"figA_hit_at_k5_{arm}"] = float(tab.loc[arm, "hit_rate"])
            metrics[f"figA_bag_at_k5_{arm}"] = float(tab.loc[arm, "mean_bag"])
    for band in DEPTH_BANDS:
        sub = _at_shipped(points, "B", f"depth {band}")
        for arm in CONDITIONS:
            if arm in sub.index:
                key = f"figB_hit_at_k5_d{band.replace('-', '_').replace('+', 'p')}_{arm}"
                metrics[key] = float(sub.loc[arm, "hit_rate"])
    tab = _at_shipped(points, "C", FACET_ALL)
    for arm in CONDITIONS:
        if arm in tab.index:
            metrics[f"figC_hit_at_k5_{arm}"] = float(tab.loc[arm, "hit_rate"])
    return metrics
