#!/bin/bash
#SBATCH --job-name=phenojury_normalise
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# Maps every juror's candidates to HPO terms for 4 prompts x 4 normalisers (thesis section 5.3 and
# appendix C.2). One array task per (prompt, juror, normaliser) cell; the script lists the cells:
#   python experiments/05_phenojury/normalise/run.py stage=plan
# DATASET=gsc for GSC+.
source slurm/common.sh
python experiments/05_phenojury/normalise/run.py \
    dataset="${DATASET:-hcy}" stage=link array_index="${SLURM_ARRAY_TASK_ID:-0}" \
    'prompt_keys=[p0_baseline,q2_sentence_last,q4_span_json,q7_recall]' "$@"
