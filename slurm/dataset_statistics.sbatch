#!/bin/bash
#SBATCH --job-name=dataset_statistics_hcy
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# The HCY column of the dataset table (thesis Table 3.1). The GSC+ columns run anywhere:
#   python experiments/03_setup/dataset_statistics.py --cohorts gsc_raghpo_ann gsc_2024_eval_206
source slurm/common.sh
python experiments/03_setup/dataset_statistics.py \
    --cohorts hcy \
    --hcy_curated_gt_path "$(p hcy.curated_ground_truth)" \
    --hcy_input_dir "$(p hcy.input_dir)" \
    --hpo_json_path resources/util/hpo.json \
    --output_dir "$OUTPUT/dataset_statistics"
