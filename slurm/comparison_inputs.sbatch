#!/bin/bash
#SBATCH --job-name=comparison_inputs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# PhenoJury's two rows of the cross-system comparison (full pool and core-3 jury) on the three
# frames the comparison reports, with the normaliser fixed to PhenoBERT and the prompt selected
# inside the folds (thesis chapter 6). Writes under <output>/comparison_inputs/. Run before
# slurm/comparison.sbatch.  COHORTS="hcy gsc gsc206" selects frames.
source slurm/common.sh
OUT=$OUTPUT/comparison_inputs
COHORTS=${COHORTS:-"hcy gsc gsc206"}
CORE3=${CORE3:-medgemma,medpsy,phi4}
want () { [[ " $COHORTS " == *" $1 "* ]]; }
# Extra arguments of the job go to every run. Saved here: inside a function, "$@" is the function's.
EXTRA=("$@")

phenojury_rows () {   # $1 cohort dir, $2 gold, $3 folds, $4 detections cohort (default $1)
  python experiments/05_phenojury/protocol/run.py \
      "cohort=$1" "source_cohort=${4:-$1}" 'stages=[audit,grid,headline,sensitivity]' \
      "grid_dir=$RESULTS/phenojury_normalisation" "gold_path=$2" "folds_path=$3" \
      'prompts=[p0_baseline,q2_sentence_last,q4_span_json,q7_recall]' \
      'normalisers=[phenobert_candidates]' "+fixed_juries.core3=[$CORE3]" \
      "output_dir=$OUT" "${EXTRA[@]}"
}

want hcy && phenojury_rows hcy "$(p hcy.curated_ground_truth)" "$(p hcy.folds)"
for frame in gsc gsc206; do
  want "$frame" || continue
  WORK=$OUT/phenojury_protocol/$frame; mkdir -p "$WORK"
  if [[ "$frame" == gsc ]]; then NAME=gsc_raghpo_ann; else NAME=gsc_2024_eval_206; fi
  GOLD=$WORK/${NAME}_gold.csv; FOLDS=$WORK/folds_${NAME}.csv
  [[ -s "$GOLD" ]] || python experiments/03_setup/write_gsc_ground_truth.py --frame "$NAME" --out "$GOLD"
  [[ -s "$FOLDS" ]] || python experiments/03_setup/make_folds.py --gold "$GOLD" --out "$FOLDS"
  phenojury_rows "$frame" "$GOLD" "$FOLDS" gsc
done
