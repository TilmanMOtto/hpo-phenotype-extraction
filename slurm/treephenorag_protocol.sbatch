#!/bin/bash
#SBATCH --job-name=treephenorag_protocol
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# The TreePhenoRAG chapter (thesis chapter 4 and appendix B): nested cross-validation on the stored
# scores, the ablations, calibration, diagnostics and the GSC+ (114) transfer. CPU only.
# Concatenates the score-store shards first, then runs every stage.
source slurm/common.sh
C_SYN="$RESULTS/treephenorag_scores_synthetic/hcy"
C_TERM="$RESULTS/treephenorag_scores_terminfo/hcy"
G_TERM="$RESULTS/treephenorag_scores_terminfo/gsc"
GSC_GOLD="$OUTPUT/treephenorag_protocol/gsc_gold"

concat_shards () {   # $1 directory, $2 variant, $3 number of shards
  local out="$1/$2_calls.jsonl"
  [[ -s "$out" ]] && { echo "  $(basename "$out") exists, kept"; return 0; }
  local shards=() i
  for ((i = 0; i < $3; i++)); do
    [[ -s "$1/$2_s${i}of$3_calls.jsonl" ]] && shards+=("$1/$2_s${i}of$3_calls.jsonl")
  done
  [[ ${#shards[@]} -gt 0 ]] || { echo "  no shards under $1" >&2; return 1; }
  cat "${shards[@]}" > "$out.tmp" && mv "$out.tmp" "$out"
}
concat_shards "$C_SYN" tree_score_cache 40
concat_shards "$C_TERM" tree_score_cache_r3u 40

GSC_ARGS=()
if [[ -d "$G_TERM" ]] && concat_shards "$G_TERM" tree_score_cache_r3u 38; then
  mkdir -p "$GSC_GOLD"
  for frame in gsc_raghpo_ann gsc_raghpo; do
    [[ -s "$GSC_GOLD/$frame.csv" ]] || python experiments/03_setup/write_gsc_ground_truth.py \
        --frame "$frame" --out "$GSC_GOLD/$frame.csv"
  done
  GSC_ARGS=("gsc.gold_path=$GSC_GOLD/gsc_raghpo_ann.csv"
            "gsc.gold_path_corpus=$GSC_GOLD/gsc_raghpo.csv"
            "gsc.caches.ontology_r3.calls_path=$G_TERM/tree_score_cache_r3u_calls.jsonl")
fi

python experiments/04_treephenorag/protocol/run.py \
    tau_cache=0.00015 \
    "evidence_path=$(p hcy.curated_annotations)" \
    'stages=[ingest,containment,select,retrieval,poolings,ablations,calibration,distributions,diagnostics,instruments]' \
    "${GSC_ARGS[@]}" "$@"
