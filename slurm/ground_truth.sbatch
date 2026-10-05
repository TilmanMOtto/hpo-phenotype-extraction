#!/bin/bash
#SBATCH --job-name=ground_truth
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Builds the curated HCY ground truth from the curation log and writes the dated dataset directory
# (thesis section 3.2 and appendix A.3). Then writes the fold file of the nested cross-validation.
#   sbatch slurm/ground_truth.sbatch [Hydra overrides]
source slurm/common.sh
python experiments/03_setup/ground_truth/run.py "hcy_gt_path=$(p hcy.raw_ground_truth)" "$@"
python experiments/03_setup/make_folds.py --gold "$(p hcy.curated_ground_truth)" --out "$(p hcy.folds)"
