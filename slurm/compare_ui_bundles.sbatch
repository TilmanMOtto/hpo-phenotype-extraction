#!/bin/bash
#SBATCH --job-name=compare_ui_bundles
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Builds the Compare UI bundles of the 20+20 documents and exports the side-by-side examples of the
# appendix (thesis Tables D.1 and D.2). The HCY bundles contain report text and stay on LeoMed.
source slurm/common.sh
BUNDLES=$OUTPUT/compare_ui_bundles
for cohort in hcy gsc; do
  python apps/compare_ui/build.py --cohort "$cohort" --output-base "$RESULTS" \
      --out "$BUNDLES/$cohort" "$@"
done
python apps/compare_ui/export_examples.py --spec "$(p hcy.qualitative_examples)" \
    --bundle-root "$BUNDLES/examples" --out "$BUNDLES/qualitative"
