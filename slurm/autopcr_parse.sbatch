#!/bin/bash
#SBATCH --job-name=autopcr_parse
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# AutoPCR's phrase extraction with the constituency parser (thesis appendix A.5). Stages the corpus
# with the main environment, then parses once in the parser's own environment and copies the
# phrases beside both linkers' corpora. DATASET=gsc for GSC+.
source slurm/common.sh
DATASET=${DATASET:-hcy}
python experiments/06_comparison/baselines/autopcr_8b.py dataset="$DATASET" stage_only=true \
    target_symptoms_path=resources/data/hpo_code_lists/target_symptoms_all.csv
NLTK_DATA="$(p autopcr.parser_assets)/nltk_data" "$(p autopcr.parser_venv)/bin/python" \
    third_party/AutoPCR/extract_phrases.py \
    --corpus "$OUTPUT/baseline_autopcr_8b/$DATASET/autopcr_corpus/corpus_test.tsv" \
    --copy_to "$OUTPUT/baseline_autopcr_70b/$DATASET/autopcr_corpus"
