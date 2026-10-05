#!/bin/bash
#SBATCH --job-name=segment_reports
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Segments a corpus with the Stanza clinical tokenizer (thesis section 3.3) and writes
# segmented_reports.csv, the segment numbering every later stage refers to:
#   sbatch slurm/segment_reports.sbatch                 # HCY, to the path file's hcy.segmented_reports
#   DATASET=gsc sbatch slurm/segment_reports.sbatch     # GSC+, to <output>/gsc_segmentation/
# Stanza loads a model, so this runs as a job and not on the login node. It must be the Stanza copy
# under stanza_dir: stanza_dir_phenobert is a different Stanza version.
source slurm/common.sh
DATASET=${DATASET:-hcy}
if [ "$DATASET" = "gsc" ]; then
  INPUT=resources/data/GSC_2024
  OUT=$OUTPUT/gsc_segmentation/segmented_reports.csv
else
  INPUT=$(p hcy.input_dir)
  OUT=$(p hcy.segmented_reports)
fi
python experiments/03_setup/segment_reports.py \
    --corpus "$DATASET" \
    --input_dir "$INPUT" \
    --stanza_dir "$(p stanza_dir)" \
    --output_path "$OUT"
