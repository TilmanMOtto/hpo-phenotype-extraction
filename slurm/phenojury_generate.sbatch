#!/bin/bash
#SBATCH --job-name=phenojury_generate
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
# Juror generations over every segment (thesis section 5.3). One array task per juror for Free
# Listing; one task per (prompt, juror) for the three other prompts (24 tasks):
#   sbatch slurm/phenojury_generate.sbatch                                  # Free Listing
#   PROMPTS=other sbatch --array=0-23 slurm/phenojury_generate.sbatch       # the other three
# DATASET=gsc for GSC+.
source slurm/common.sh
DATASET=${DATASET:-hcy}
JURORS=(apertus deepseek intelligent_internet openbiollm llama medpsy medgemma phi4)
TS=resources/data/hpo_code_lists/target_symptoms_all.csv
if [[ "${PROMPTS:-free_listing}" == free_listing ]]; then
  KEY=${JURORS[${SLURM_ARRAY_TASK_ID:-0}]}
  python experiments/05_phenojury/generate/free_listing.py \
      dataset="$DATASET" stage=extract model_key="$KEY" target_symptoms_path="$TS" "$@"
else
  python experiments/05_phenojury/generate/other_prompts.py \
      dataset="$DATASET" stage=extract array_index="${SLURM_ARRAY_TASK_ID:-0}" \
      'prompt_keys=[q2_sentence_last,q4_span_json,q7_recall]' target_symptoms_path="$TS" "$@"
fi
