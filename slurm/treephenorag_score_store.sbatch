#!/bin/bash
#SBATCH --job-name=treephenorag_scores
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=40:00:00
#SBATCH --array=0-39
# Stored verifier scores for every report and all 18 354 terms at S = 10 (thesis appendix B.2).
# One array task per shard of the reports. INDEX selects the retrieval index:
#   INDEX=term_information sbatch slurm/treephenorag_score_store.sbatch     (selected in the thesis)
#   INDEX=synthetic        sbatch slurm/treephenorag_score_store.sbatch
# DATASET=gsc runs GSC+ (38 shards: --array=0-37, N_SHARDS=38).
source slurm/common.sh
INDEX=${INDEX:-term_information}
DATASET=${DATASET:-hcy}
N_SHARDS=${N_SHARDS:-40}
case "$INDEX" in
  term_information) SCRIPT=experiments/04_treephenorag/score_store/run_term_information.py ;;
  synthetic)        SCRIPT=experiments/04_treephenorag/score_store/run_synthetic.py ;;
  *) echo "INDEX must be term_information or synthetic" >&2; exit 2 ;;
esac
# HCY runs score only the reports of the curated ground truth (118 of the files in the input folder).
COHORT_ARGS=()
[[ "$DATASET" == hcy ]] && COHORT_ARGS=("report_ids_path=$(p hcy.curated_ground_truth)")
python "$SCRIPT" \
    dataset="$DATASET" \
    target_symptoms_path=resources/data/hpo_code_lists/target_symptoms_all.csv \
    "${COHORT_ARGS[@]}" \
    report_shard="${SLURM_ARRAY_TASK_ID:-0}" \
    n_report_shards="$N_SHARDS" "$@"
