"""RAG-HPO's published GSC+ (114) numbers, approached one change at a time -> t6_raghpo_replication.csv.

    python experiments/06_comparison/comparison/replication.py \
        --results-dir <results_dir> \
        --output-dir  <results_dir>

Chapter 6 reports RAG-HPO 70B (the RAG-HPO 70B baseline: upstream's current code, Llama-3.3-70B) at the published
precision but 0.12 short of the published recall. Three earlier runs walk from that row towards the
published configuration, each changing one thing:

    the RAG-HPO 70B baseline  current upstream code (2025-07-30), Llama-3.3-70B-Instruct, greedy
    the RAG-HPO reproduction with the published code  published code (upstream 25c1ea7) and phrase database, Groq's Llama-3-70B tool-use
    the RAG-HPO reproduction with Llama-3 70B  + Meta-Llama-3-70B-Instruct, the model the paper's result columns name
    the RAG-HPO reproduction with the published prompt  + the extraction prompt printed in the paper (Additional file 2, Fig. S1)

This scores all four on the published frame with the comparison's own ground truth loader, alignment and
metrics, imported rather than copied, so the RAG-HPO 70B baseline's row here is t1_overall.csv's row by
construction (tables_ch6.py checks it). The published row itself stays in t1_literature.csv.

GSC+ ONLY. The four artifacts are the `gsc` run directories. No HCY path is read, and a results
directory is refused if a condition resolves to one.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hpo_extraction.evaluation.result_tables.loaders as loaders  # noqa: E402, the result-table library's ground truth and prediction readers
from hpo_extraction.evaluation.stats import ReportResampler, bootstrap_reports  # noqa: E402
from hpo_extraction.evaluation.metrics import flat_report  # noqa: E402

# The comparison's run.py by path: the result-table library also has a run.py, and it is earlier on sys.path.
_spec = importlib.util.spec_from_file_location("exp13_25_run", _HERE / "run.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

COHORT = "gsc_raghpo_ann"

#: One row per run, in the order of the walk. `change` is what differs from the row above.
CONDITIONS = [
    dict(arm="current", experiment="baseline_raghpo_70b",
         code="current (2025-07-30)", backend="Llama-3.3-70B-Instruct", prompt="repository",
         decoding="greedy", change="the thesis's RAG-HPO 70B row"),
    dict(arm="published_code", experiment="raghpo_reproduction_published_code",
         code="published (25c1ea7)", backend="Llama-3-Groq-70B-Tool-Use", prompt="repository",
         decoding="sampled, T = 0.2",
         change="published code, phrase database and retrieval; the backend upstream's client names"),
    dict(arm="published_backend", experiment="raghpo_reproduction_llama3_70b",
         code="published (25c1ea7)", backend="Meta-Llama-3-70B-Instruct", prompt="repository",
         decoding="sampled, T = 0.2",
         change="the checkpoint the paper's result columns name"),
    dict(arm="published_prompt", experiment="raghpo_reproduction_published_prompt",
         code="published (25c1ea7)", backend="Meta-Llama-3-70B-Instruct", prompt="paper (Fig. S1)",
         decoding="sampled, T = 0.2",
         change="the extraction prompt printed in the paper"),
]

FIELDS = ["arm", "experiment", "code", "backend", "prompt", "decoding", "change", "cohort",
          "n_reports", "n_gold_scored", "tp", "fp", "fn", "macro_precision", "macro_recall",
          "macro_f1_of_means", "micro_precision", "micro_recall", "micro_f1", "micro_f1_lo",
          "micro_f1_hi", "path"]


def main() -> None:
    """Score the RAG-HPO reproduction runs on GSC+ (114) and write ``t6_raghpo_replication.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hpo-json", default=str(_REPO / "resources/util/hpo.json"))
    ap.add_argument("--gsc-dir", default=str(_REPO / "resources/data/GSC_2024"))
    ap.add_argument("--raghpo-dir", default=str(_REPO / "resources/data/GSC_RAGHPO"))
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    args = ap.parse_args()

    paths = {a["arm"]: Path(args.results_dir) / a["experiment"] / "gsc" / "rag_hpo_predictions.jsonl"
             for a in CONDITIONS}
    if any("hcy" in str(p).lower() for p in paths.values()):
        raise SystemExit("refusing: an arm resolves to an HCY path -- this table is GSC+ only")
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("missing artifact(s):\n  " + "\n  ".join(missing))

    view = E._load_view(args.hpo_json)
    gold = loaders.load_gold(COHORT, gsc_dir=args.gsc_dir, raghpo_dir=args.raghpo_dir)
    sampler = ReportResampler(len(gold), n_resamples=args.n_bootstrap, seed=0)

    rows = []
    for arm in CONDITIONS:
        predicted, _ = loaders.load_predictions(str(paths[arm["arm"]]))
        report_ids, gold_sets, pred_sets, book, _raw = E.align(predicted, gold, view)
        if book["n_reports_absent"]:
            raise SystemExit(f"{arm['experiment']}: {book['n_reports_absent']} report(s) of "
                             f"{COHORT} absent from the artifact")
        flat = flat_report(gold_sets, pred_sets)
        ci = bootstrap_reports(E.flat_metric, E.flat_units(gold_sets, pred_sets),
                               resampler=sampler)["micro_f1"]
        rows.append({**arm, "cohort": COHORT, "n_reports": len(report_ids),
                     "n_gold_scored": book["n_gold_scored"],
                     **{k: flat[k] for k in ("tp", "fp", "fn", "macro_precision", "macro_recall",
                                             "macro_f1_of_means", "micro_precision",
                                             "micro_recall", "micro_f1")},
                     "micro_f1_lo": ci["lo"], "micro_f1_hi": ci["hi"],
                     "path": str(paths[arm["arm"]])})
        print(f"{arm['experiment']:32s} P_M={flat['macro_precision']:.3f} "
              f"R_M={flat['macro_recall']:.3f} F1_mu={flat['micro_f1']:.3f}")

    out = Path(args.output_dir) / "comparison" / "tables" / "t6_raghpo_replication.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
