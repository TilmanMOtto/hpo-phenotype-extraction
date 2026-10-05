# AutoPCR (copied from upstream)

Upstream: <https://github.com/yctao7/AutoPCR>, "AutoPCR: Automated Phenotype Concept Recognition by
Prompting" (Tao et al.), commit `c2b46a9c088c696533a43c5cb8e50c7500f055fe` (2026-03-25). Licence:
upstream's, in `LICENSE`.

Used by the two AutoPCR baselines (`experiments/06_comparison/baselines/autopcr_{8b,70b}.py`)
through `hpo_extraction.baselines.autopcr_runner` and `hpo_extraction.baselines.autopcr_experiment`.
Every change to upstream's files is listed in `PATCHES.md`. Setup steps are in
`third_party/README.md`.

## What was copied

| File | Role |
|---|---|
| `nn_model2.py` | the retriever and linker (`bioTag_SapBERT`), the method itself |
| `HPO_evaluation.py` | `run_gsc_test_ner`, the per-document pipeline |
| `build_dict.py` | term tokenisation and lemmatisation, and the mapping-file writers |
| `build_index.py` | the SapBERT and FAISS index builder |
| `evaluate.py` | upstream's own metric, kept as an independent check of the scoring here |
| `ee/{__init__,pbert,utils}.py` | entity extraction: Stanza i2b2 NER and n-gram enumeration |
| `ee/benepar.py` | the `neural+` and `neural++` extraction: benepar constituents and coordination splitting |
| `utils/{__init__,llm,prompts}.py` | the linking prompt and the client interface it calls |

Added here: `build_dict_from_hpo_json.py` (builds upstream's dictionary from
`resources/util/hpo.json`), `extract_phrases.py` (runs the benepar phrase extraction in the parser
environment) and `requirements_ee.txt` (that environment, with upstream's own versions).

Not copied: the rule-based extraction path (`--ee rule`), the optional fine-tuned linker,
post-processing tools, and upstream's corpora, ontology release and results. The cohorts come from
this repository's loaders and the ontology from `resources/util/hpo.json`.

## The published setting runs in two environments

Upstream's default extraction, which its GSC+ command runs, is `neural++`: the Stanza phrases plus
the constituents benepar finds (`phrases_benepar.json`) and their coordination-split variants
(`phrases_conjunct.json`). The parser stack needs other versions of transformers and protobuf than
the main environment, so the parse runs once in its own environment
(`slurm/autopcr_parse.sbatch`) and writes the two JSON files beside the staged corpus. The linker
runs in the main environment, reads them, and refuses to start without them
(`autopcr_runner.require_parser_cache`). One parse serves both linkers, so the difference between the
8B and the 70B run comes from the linker alone.

## The ontology is built from this repository's file

Upstream parses `hp_20240208.obo`. Here every method is scored against `resources/util/hpo.json`
(fingerprint `8330ca317e71`), so `build_dict_from_hpo_json.py` builds the dictionary from that file
and then calls upstream's own `word_hpo_map`, `hpo_word_map`, `alt_hpo` and `build_index.py`, which
keeps the file formats upstream's. It checks its vocabulary against
`hpo_extraction.retrieval.surface_index.build_surface_index` and stops on a mismatch.

## Build products (not tracked)

```
dict/HPO/   the ontology dictionary and the SapBERT/FAISS index (slurm/autopcr_index.sbatch)
data/       HPO_UMLS-synonyms.json from upstream's data/ (8.5 MB, optional)
```

The index must be built with the SapBERT model the linker uses
(`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`): AutoPCR's thresholds (0.95 and 0.85) were set
for that encoder.
