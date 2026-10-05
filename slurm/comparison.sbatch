#!/bin/bash
#SBATCH --job-name=comparison
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# The cross-system comparison (thesis chapter 6 and appendix D), then the RAG-HPO reproduction table
# (appendix A.5). CPU only; reads every method's stored predictions.
source slurm/common.sh
python experiments/06_comparison/comparison/run.py \
    'cohorts=[hcy,gsc_raghpo_ann,gsc_2024_eval_206]' \
    "hcy_annotations_path=$(p hcy.curated_annotations)" \
    "tree_attribution_path=$RESULTS/treephenorag_protocol/tables/recall_attribution.csv" "$@"
python experiments/06_comparison/comparison/replication.py \
    --results-dir "$RESULTS" --output-dir "$OUTPUT"
