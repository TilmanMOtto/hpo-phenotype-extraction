#!/bin/bash
#SBATCH --job-name=autopcr_index
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# AutoPCR's term dictionary and SapBERT index (thesis appendix A.5), built once from
# resources/util/hpo.json into third_party/AutoPCR/dict/HPO, the ontology_dict of both AutoPCR
# baseline configs. Ends with a self-check of the top-1 retrieval of a few probe phrases.
# AutoPCR's UMLS synonym table is optional: without third_party/AutoPCR/data/HPO_UMLS-synonyms.json
# the linker prompt has three fields instead of four.
source slurm/common.sh
DICT=third_party/AutoPCR/dict/HPO
SAPBERT=$(p models.sapbert)
UMLS=third_party/AutoPCR/data/HPO_UMLS-synonyms.json
python third_party/AutoPCR/build_dict_from_hpo_json.py -o "$DICT"
if [ -f "$UMLS" ]; then
  python third_party/AutoPCR/build_index.py --ontology_dict "$DICT" --ccr "$SAPBERT" --umls_synonyms "$UMLS"
else
  echo "NOTE: $UMLS absent, building without the UMLS synonym field" >&2
  python third_party/AutoPCR/build_index.py --ontology_dict "$DICT" --ccr "$SAPBERT"
fi
python -m hpo_extraction.baselines.autopcr_runner --selfcheck --ontology_dict "$DICT" --ccr "$SAPBERT"
