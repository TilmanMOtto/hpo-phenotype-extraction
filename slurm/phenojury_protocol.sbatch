#!/bin/bash
#SBATCH --job-name=phenojury_protocol
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# The PhenoJury chapter (thesis chapter 5 and appendix C): nested cross-validation over prompt,
# normaliser, jury, rule, unit and k, with the core-3 jury and its one-member swap as fixed juries.
#   sbatch slurm/phenojury_protocol.sbatch                    # HCY
#   COHORT=gsc206 sbatch slurm/phenojury_protocol.sbatch      # GSC+ (206)
source slurm/common.sh
COHORT=${COHORT:-hcy}
CORE3=${CORE3:-medgemma,medpsy,phi4}
CORE3_SWAP=${CORE3_SWAP-intelligent_internet,medpsy,phi4}
SWAP_ARGS=()
[[ -n "$CORE3_SWAP" ]] && SWAP_ARGS=("+fixed_juries.core3swap=[$CORE3_SWAP]")

if [[ "$COHORT" == hcy ]]; then
  GOLD=$(p hcy.curated_ground_truth); FOLDS=$(p hcy.folds)
  J0=$RESULTS/baseline_phenobert/hcy/predictions.csv
  J0_SOURCE=$RESULTS/baseline_phenobert/hcy/phenobert_predictions.jsonl
  COHORT_ARGS=(cohort=hcy "annotations_path=$(p hcy.curated_annotations)")
else
  WORK=$OUTPUT/phenojury_protocol/gsc206; mkdir -p "$WORK"
  GOLD=$WORK/gsc_2024_eval_206_gold.csv; FOLDS=$WORK/folds_gsc_2024_eval_206.csv
  J0=$WORK/j0_phenobert_predictions.csv
  J0_SOURCE=$RESULTS/baseline_phenobert/gsc/phenobert_predictions.jsonl
  [[ -s "$GOLD" ]] || python experiments/03_setup/write_gsc_ground_truth.py \
      --frame gsc_2024_eval_206 --out "$GOLD"
  COHORT_ARGS=(cohort=gsc206 source_cohort=gsc)
fi
[[ -s "$FOLDS" ]] || python experiments/03_setup/make_folds.py --gold "$GOLD" --out "$FOLDS"
if [[ ! -s "$J0" && -s "$J0_SOURCE" ]]; then
  python experiments/06_comparison/predictions_to_sets.py "$J0_SOURCE" "$J0" --restrict-to "$GOLD"
fi

python experiments/05_phenojury/protocol/run.py \
    "${COHORT_ARGS[@]}" \
    "gold_path=$GOLD" "folds_path=$FOLDS" "j0_predictions_path=$J0" \
    'prompts=[p0_baseline,q2_sentence_last,q4_span_json,q7_recall]' \
    "+fixed_juries.core3=[$CORE3]" "${SWAP_ARGS[@]}" \
    n_workers="${SLURM_CPUS_PER_TASK:-1}" "$@"
