#!/bin/bash
#SBATCH --job-name=retrieval_curves
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Retrieval of the evidence segment by term representation (thesis Figure 4.5). Encoder on CPU.
source slurm/common.sh
python experiments/04_treephenorag/retrieval_curves/run.py "$@"
