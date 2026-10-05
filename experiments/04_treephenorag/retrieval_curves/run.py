"""
The retrieval-curve analysis, four query-set variants for the retrieval stage, compared on one cost axis.


Predecessor is an earlier exploratory run, which swept how deep the synthetic-sentence-sentence pool reaches down the
ontology and found the descendant closure the best retrieval setting in the project. That
sweep moves *within one family* of representations. It never asked what the family is worth:
how much of the synthetic-sentence corpus's performance an ontology lookup would deliver for free
(``label + definition + synonyms``, one vector, no generator model), and how much of it is
above chance at all.

So this experiment holds the encoder, the segments, the curated ground truth, the true-path rule and
the max-pooled-cosine scoring rule fixed, and changes only the set of vectors a query term is
represented by:

    R1   U_v                                          the shipped system  (= an earlier exploratory run k0)
    R1u  U_v union every descendant's U_y             (= an earlier exploratory run kinf)
    R3   embed(label + " " + definition + " " + synonyms)      one vector, no generator
    R5   nothing, the report's sentences in random order      chance, in closed form

R1 and R1u are therefore built-in controls that must reproduce an earlier exploratory run, and ``compute``
asserts it: R1 equals ``SymptomScoreCalculator`` to 1e-5, R1u is never below R1, and the two
coincide on every leaf query.

The result is not a ranking table. It is an **operating curve**: mean sentences forwarded per
(patient, term) pair against the pair-level hit rate, with both gates (top-k and a global
cosine cutoff) on the same axis, because their natural parameters are neither comparable with
each other nor across conditions whose score distributions differ.

  Figure A   annotated terms, terminal hit rate
  Figure B   ancestor closure, faceted by the query term's absolute ontology depth
  Figure C   the whole query set, annotated terms AND ancestors, pooled. The one that bounds an
             end-to-end run, since the pipeline queries target_symptoms_all.csv, not the ground truth
             terms alone

Two stages, both written to ``output_dir/treephenorag_retrieval_curves/``:
  1. ``compute``   score every (condition, patient, query HPO, segment)  -> ``qs_scores.csv.gz``
  2. ``report``    gate curves, bootstrap intervals, Figure A, Figure B

``report`` reads only ``qs_scores.csv.gz``, so ``stages=[report]`` re-draws locally without the
model or the context directory. See ``experiment.md``.

Usage:
    python experiments/04_treephenorag/retrieval_curves/run.py \\
        sent_transformer_dir=/path/to/all-mpnet-base-v2 \\
        context_dir=resources/context_data/context_HCY_llama_api/llama-3.3-70b-instruct \\
        segmented_reports_path=/path/to/segmented_reports.csv \\
        annotations_path=/path/to/hcy/curated_ground_truth_<date>/hcy_curated_annotations.csv \\
        output_dir=/path/to/output

The ground truth is the **curated** HCY one, as in an earlier exploratory run and an earlier exploratory run and for the same reason: every
number here is a statement about where a sentence ranks, and an annotation whose trigger word
occurs nowhere in the report has no sentence to rank.
``hpo_extraction.evaluation.retrieval_analysis.load_inputs`` applies the curated table's own policy and
evidence rule.
"""

import logging
import sys
import time
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

# The REPO ROOT as well as src/, as the ground-truth build does. `pip install -e .` only exposes `src`
# (pyproject: packages.find where=["src"]), but `hpo_extraction.evaluation.retrieval_analysis.load_inputs` reads
# The segmented reports through `hpo_extraction.data.segmented_reports`, so this experiment
# and the curation UI agree on the segmentation, and `app` lives at the repo root.
from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402
for _p in (str(_REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gate_curves  # noqa: E402
import query_sets  # noqa: E402

logger = logging.getLogger(__name__)

EXP_ID = "treephenorag_retrieval_curves"


def _setup_logging(run_output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(run_output_dir / "run.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


@hydra.main(version_base=None, config_path="../../../configs/experiments/04_treephenorag", config_name="retrieval_curves")
def run_analysis(cfg: DictConfig) -> None:
    """Run the configured stages (compute the curves, write the report) into ``<output_dir>/treephenorag_retrieval_curves``."""
    run_output_dir = Path(cfg.output_dir) / EXP_ID
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_output_dir)

    stages = list(cfg.stages)
    logger.info("Starting %s | stages=%s | output=%s", EXP_ID, stages, run_output_dir)

    mlflow.set_experiment(EXP_ID)
    with mlflow.start_run():
        mlflow.log_params({
            "context_dir": cfg.context_dir,
            "sent_transformer_dir": cfg.sent_transformer_dir,
            "annotations_path": cfg.annotations_path,
            # A hit rate is only comparable with another one taken against the same annotation
            # set, so log which ground truth this run scored against.
            "gold_source": cfg.gold_source,
            "require_location": cfg.require_location,
            "exclude_location_how": ",".join(cfg.exclude_location_how or []),
            "arms": ",".join(query_sets.CONDITIONS),
            "sent_scoring": "max",       # SymptomScoreCalculator default, as every run.py uses
            "ancestor_scope": "phenotypic_abnormality_subtree",
            "r3_template": query_sets.R3_TEMPLATE,
            "n_bootstrap": cfg.n_bootstrap,
            "bootstrap_unit": "patient",
            "bootstrap_seed": cfg.bootstrap_seed,
            "x_max": cfg.x_max,
            "stages": ",".join(stages),
        })

        t0 = time.time()

        if "compute" in stages:
            logger.info("── compute ──")
            counts = query_sets.run_compute(
                segmented_reports=cfg.segmented_reports_path,
                annotations=cfg.annotations_path,
                context_dir=cfg.context_dir,
                sent_transformer_dir=cfg.sent_transformer_dir,
                output_dir=run_output_dir,
                hpo_json=cfg.hpo_json_path,
                gold_source=cfg.gold_source,
                require_location=cfg.require_location,
                exclude_location_how=tuple(cfg.exclude_location_how or ()),
            )
            mlflow.log_metrics(counts)
            logger.info("compute: %s", counts)
            for name in (query_sets.SCORES_FILE, query_sets.GOLD_FILE, query_sets.R3_FILE,
                         query_sets.SEGMENTS_JSONL):
                mlflow.log_artifact(str(run_output_dir / name))

        if "report" in stages:
            logger.info("── report ──")
            metrics = gate_curves.run_report(
                output_dir=run_output_dir,
                n_resamples=cfg.n_bootstrap,
                seed=cfg.bootstrap_seed,
                ci=cfg.ci,
                x_max=cfg.x_max,
            )
            mlflow.log_metrics(metrics)
            for arm in query_sets.CONDITIONS:
                key = f"figA_hit_at_k5_{arm}"
                if key in metrics:
                    logger.info("report: Figure A, %s at k=5 — hit rate %.3f (bag %.2f)",
                                arm, metrics[key], metrics[f"figA_bag_at_k5_{arm}"])
            for name in (query_sets.OPERATING_FILE, query_sets.PAIRS_FILE,
                         query_sets.SUMMARY_FILE,
                         f"{query_sets.FIG_A}.pdf", f"{query_sets.FIG_A}.png",
                         f"{query_sets.FIG_B}.pdf", f"{query_sets.FIG_B}.png",
                         f"{query_sets.FIG_C}.pdf", f"{query_sets.FIG_C}.png"):
                mlflow.log_artifact(str(run_output_dir / name))

        # No mean_time_per_report_s: this experiment runs no per-report LLM inference. The cost
        # axis of both figures, sentences forwarded per pair, is the quantity it measures
        # instead.
        mlflow.log_metric("total_runtime_s", time.time() - t0)

        cfg_path = run_output_dir / "config_resolved.yaml"
        cfg_path.write_text(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(str(cfg_path))

    logger.info("Done in %.1fs | %s", time.time() - t0, run_output_dir)


if __name__ == "__main__":
    run_analysis()
