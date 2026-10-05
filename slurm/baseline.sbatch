#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# One baseline run (thesis section 3.5 and appendix A.5). METHOD selects it, DATASET the cohort:
#   METHOD=phenobert   sbatch --partition=compute --gres=none --cpus-per-task=8 --mem=32G slurm/baseline.sbatch
#   METHOD=raghpo_8b   sbatch slurm/baseline.sbatch
#   METHOD=raghpo_70b  sbatch --gres=gpu:rtx4090:4 slurm/baseline.sbatch
#   METHOD=autopcr_8b  sbatch slurm/baseline.sbatch       (after slurm/autopcr_parse.sbatch)
#   METHOD=autopcr_70b sbatch --gres=gpu:rtx4090:4 slurm/baseline.sbatch
#   METHOD=raghpo_published_code | raghpo_llama3_70b | raghpo_published_prompt  (4 GPUs, GSC+ only)
source slurm/common.sh
DATASET=${DATASET:-hcy}
TS=resources/data/hpo_code_lists/target_symptoms_all.csv
RAGHPO_DB=$(p raghpo.vector_db_dir)
# The reproduction runs embed with BGE-small, read from the first model folder under `snapshots/` of its cache folder.
bge_dir() { ls -d "$(p models.bge_small_cache)"/snapshots/*/ 2>/dev/null | head -1; }
case "${METHOD:?set METHOD}" in
  phenobert)   python experiments/06_comparison/baselines/phenobert.py dataset="$DATASET" \
                   target_symptoms_path="$TS" "phenobert_dir=$(p phenobert.dir)" phenobert_threads=8 "$@" ;;
  raghpo_8b)   python experiments/06_comparison/baselines/raghpo_8b.py dataset="$DATASET" \
                   target_symptoms_path="$TS" "sbert_model=$(p models.sapbert_mnli)" \
                   "rag_hpo_dir=$RAGHPO_DB" "$@" ;;
  raghpo_70b)  python experiments/06_comparison/baselines/raghpo_70b.py dataset="$DATASET" \
                   target_symptoms_path="$TS" "sbert_model=$(p models.sapbert_mnli)" \
                   "rag_hpo_dir=$RAGHPO_DB" "$@" ;;
  autopcr_8b)  python experiments/06_comparison/baselines/autopcr_8b.py dataset="$DATASET" \
                   target_symptoms_path="$TS" "ccr_model=$(p models.sapbert)" "$@" ;;
  autopcr_70b) python experiments/06_comparison/baselines/autopcr_70b.py dataset="$DATASET" \
                   target_symptoms_path="$TS" "ccr_model=$(p models.sapbert)" "$@" ;;
  raghpo_published_code)   python experiments/06_comparison/raghpo_reproduction/published_code.py dataset=gsc \
                               "rag_hpo_dir=$RAGHPO_DB" "embed_model=$(bge_dir)" "$@" ;;
  raghpo_llama3_70b)       python experiments/06_comparison/raghpo_reproduction/llama3_70b.py dataset=gsc \
                               "rag_hpo_dir=$RAGHPO_DB" "embed_model=$(bge_dir)" "$@" ;;
  raghpo_published_prompt) python experiments/06_comparison/raghpo_reproduction/published_prompt.py dataset=gsc \
                               "rag_hpo_dir=$RAGHPO_DB" "embed_model=$(bge_dir)" "$@" ;;
  *) echo "unknown METHOD $METHOD" >&2; exit 2 ;;
esac
