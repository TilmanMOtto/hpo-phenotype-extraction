"""Per-model driver report for the Free Listing generation run SLM ensemble, headless, prints Markdown.

    python apps/phenojury_ui/drivers.py                       # both cohorts, default cluster paths
    python apps/phenojury_ui/drivers.py --k 1 --k 2 --k 4     # extra configurations

Answers two questions the shipped tables cannot: **where the ensemble's recall ceiling comes
from** (k=1 is the union of all voters, so its recall is the ceiling of every k-of-N rule over the
same detections), and **which models carry it**, solo precision/recall, annotated terms only one model
found, leave-one-out and exact Shapley, plus the greedy subset that reproduces the union.

Every number is re-run from the artifacts the driver already wrote, through the same
``registry``/``votes`` code the Dash app uses, so it agrees with the UI's Models view by
construction. CPU-only. No GPU, no model reload.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from apps.phenojury_ui import votes  # noqa: E402
from apps.phenojury_ui.detections import popcount  # noqa: E402
from apps.phenojury_ui.registry import Registry  # noqa: E402

from hpo_extraction.paths import lookup as _lookup  # noqa: E402

DEFAULT_OUTPUT_BASE = _lookup("results_dir")

logger = logging.getLogger(__name__)


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return precision, recall, f1


def _score_mask(masks, gold, report_ids, subset_mask: int, k: int) -> tuple[int, int, int]:
    tp = n_pred = n_gold = 0
    for report_id in report_ids:
        gold_set = set(gold.get(report_id, ()))
        pred = {h for h, mask in masks.get(report_id, {}).items()
                if popcount(mask & subset_mask) >= k}
        tp += len(gold_set & pred)
        n_pred += len(pred)
        n_gold += len(gold_set)
    return tp, n_pred, n_gold


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def external_gold(cohort: str, hcy_gt: str | None, gsc_dir: str | None) -> dict | None:
    """The cohort's ground truth from the ground-truth file, or ``None`` if no path was given.

    The ground truth baked onto the predictions' summary lines is the one the *run* used, and for HCY that
    is stale, ``result_tables`` scored 98 HCY reports against a different (full-ontology) file and said
    so in its log. Passing ``--hcy-gt`` makes these tables comparable to the thesis tables.
    """
    if cohort == "hcy" and hcy_gt:
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset
        raw = HCYDataset(hcy_gt, "").load_ground_truth()
    elif cohort == "gsc" and gsc_dir:
        from hpo_extraction.evaluation.datasets.gsc import load_gsc_ground_truth
        raw = load_gsc_ground_truth(gsc_dir)
    else:
        return None
    return {str(k): {c.strip() for c in v if c and str(c).strip()} for k, v in raw.items()}


def report(bundle: dict, ks: list[int], gold: dict | None = None) -> str:
    """Markdown report of one run: per-juror recall ceiling and contribution at each k in *ks*."""
    models = bundle["models"]
    gold = bundle["gold"] if gold is None else gold
    min_count = bundle["min_count"]
    masks = bundle["masks"](min_count)
    report_ids = bundle["report_ids"]
    full = bundle["mask_of"](models)
    n_gold = sum(len(set(gold.get(r, ()))) for r in report_ids)

    lines = [f"## {bundle['cohort']}, {len(report_ids)} reports, {n_gold} annotated pairs, "
             f"{len(models)} voting models (min_detection_count={min_count})", ""]

    # ── the k sweep: k=1 is the union, hence the ceiling ─────────────────────
    rows = []
    for k in range(1, len(models) + 1):
        tp, n_pred, _ = _score_mask(masks, gold, report_ids, full, k)
        precision, recall, f1 = _prf(tp, n_pred, n_gold)
        rows.append([f"k={k}", tp, n_pred - tp, n_gold - tp,
                     f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])
    lines += ["### Vote sweep (k=1 is the union, its recall is the ceiling)", "",
              _table(["rule", "TP", "FP", "FN", "precision", "recall", "F1"], rows), ""]

    # ── each model alone ─────────────────────────────────────────────────────
    solo = []
    for i, model in enumerate(models):
        tp, n_pred, _ = _score_mask(masks, gold, report_ids, 1 << i, 1)
        precision, recall, _ = _prf(tp, n_pred, n_gold)
        solo.append({"model": model, "tp": tp, "n_pred": n_pred,
                     "precision": precision, "recall": recall})
    uniq = {r["model"]: r for r in votes.unique_recall(bundle, gold, models, min_count)}
    rows = [[s["model"], s["n_pred"], s["tp"], f"{s['precision']:.3f}", f"{s['recall']:.3f}",
             uniq[s["model"]]["n_unique"],
             f"{uniq[s['model']]['n_unique'] / n_gold:.3f}" if n_gold else "-"]
            for s in sorted(solo, key=lambda s: -s["recall"])]
    lines += ["### Each model alone, and the recall only it supplies", "",
              _table(["model", "terms proposed", "TP", "precision alone", "recall alone",
                      "ground truth only it found", "unique recall"], rows), ""]

    # ── greedy union coverage: how few models reproduce the ceiling ──────────
    covered: set[tuple[str, str]] = set()
    all_gold_hits = {m: set() for m in models}
    for i, model in enumerate(models):
        bit = 1 << i
        for report_id in report_ids:
            for hpo_id in set(gold.get(report_id, ())):
                if masks.get(report_id, {}).get(hpo_id, 0) & bit:
                    all_gold_hits[model].add((report_id, hpo_id))
    rows, remaining = [], dict(all_gold_hits)
    while remaining:
        best = max(remaining, key=lambda m: len(remaining[m] - covered))
        gain = len(remaining[best] - covered)
        if gain == 0:
            break
        covered |= remaining.pop(best)
        rows.append([len(rows) + 1, best, gain, len(covered),
                     f"{len(covered) / n_gold:.3f}" if n_gold else "-"])
    lines += ["### Greedy union coverage, which models build the ceiling", "",
              _table(["#", "model added", "new annotated terms", "cumulative", "cumulative recall"],
                     rows), ""]

    # ── contribution at each requested k ─────────────────────────────────────
    for k in ks:
        k = max(1, min(k, len(models)))
        shap = {r["model"]: r["shapley"] for r in votes.shapley(bundle, gold, models, k, min_count)}
        loo = {r["model"]: r for r in votes.leave_one_out(bundle, gold, models, k, min_count)}
        loo_recall = {}
        for i, model in enumerate(models):
            tp, n_pred, _ = _score_mask(masks, gold, report_ids, full & ~(1 << i),
                                        min(k, max(1, len(models) - 1)))
            loo_recall[model] = _prf(tp, n_pred, n_gold)[1]
        base_recall = _prf(*_score_mask(masks, gold, report_ids, full, k)[:2], n_gold)[1]
        rows = [[m, f"{shap.get(m, 0.0):+.4f}", f"{loo[m]['f1_without']:.3f}",
                 f"{loo[m]['delta']:+.4f}", f"{base_recall - loo_recall[m]:+.4f}"]
                for m in sorted(models, key=lambda m: -shap.get(m, 0.0))]
        lines += [f"### Contribution at k={k}", "",
                  _table(["model", "Shapley (µF1)", "F1 without it", "leave-one-out ΔF1",
                          "leave-one-out Δrecall"], rows), ""]

    return "\n".join(lines)


def main() -> None:
    """Print the per-juror report of each cohort."""
    ap = argparse.ArgumentParser(description="Per-juror driver report of the Free Listing generation run")
    ap.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--cohort", action="append", default=None,
                    help="Restrict to these run ids (default: all found)")
    ap.add_argument("--k", type=int, action="append", default=None,
                    help="Configurations for the contribution tables (default 1 and 2)")
    ap.add_argument("--hcy-gt", default=None,
                    help="hcy_ground_truth_raw.csv, score HCY against the file the result tables used "
                         "instead of the ground truth baked into the run's predictions")
    ap.add_argument("--gsc-dir", default=None, help="GSC+ folder, same purpose")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)-7s | %(name)s | %(message)s")
    registry = Registry(output_base=args.output_base, cache_dir=args.cache_dir)
    problems = registry.validate()
    for problem in problems:
        print(f"! {problem}", file=sys.stderr)
    run_ids = args.cohort or registry.run_ids()
    if not run_ids:
        sys.exit(1)

    print("# Free Listing generation run: recall ceiling and per-juror drivers\n")
    for run_id in run_ids:
        bundle = registry.get_bundle(run_id)
        gold = external_gold(bundle["cohort"], args.hcy_gt, args.gsc_dir)
        if gold is not None:
            gold = {r: gold.get(r, set()) for r in bundle["report_ids"]}
            print(f"*{bundle['cohort']}: scored against the ground-truth file, not the run's "
                  f"recorded gold.*\n")
        print(report(bundle, args.k or [1, 2], gold))


if __name__ == "__main__":
    main()
