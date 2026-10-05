# third_party

Code from other projects that the baselines of the thesis run (thesis section 3.5, appendix A.5).
Each has its own licence. AutoPCR, RAG-HPO and PhenoBERT are copied into this folder with their
licence files. Files written by a build step and model weights (indexes, vector databases,
PhenoBERT's weights) are not tracked (`.gitignore`).

## AutoPCR (`AutoPCR/`)

Upstream <https://github.com/yctao7/AutoPCR>, commit `c2b46a9c088c696533a43c5cb8e50c7500f055fe`.
Only the files of the published setting (`ee="neural++"`) are copied. Every change to upstream is
listed in `AutoPCR/PATCHES.md` and marked with `# PhenoRAG patch:` in the source. The linker runs
through `hpo_extraction.baselines.autopcr_runner`, with a local language model.

Three steps, once:

1. **Dictionary and index.** `sbatch slurm/autopcr_index.sbatch` builds `AutoPCR/dict/HPO` from
   `resources/util/hpo.json` and the SapBERT index (`models.sapbert` in the path file), then runs a
   self-check. AutoPCR's UMLS synonym table, `AutoPCR/data/HPO_UMLS-synonyms.json` (8.5 MB, from
   upstream's `data/`), is optional: without it the linker prompt has three fields instead of four.
2. **Parser environment.** The constituency parser (benepar 0.2.0, spaCy 3.7.5, `en_core_web_trf`
   3.7.3, `benepar_en3_large`) needs `transformers==4.49.0` and `protobuf==6.31.1`, which conflict
   with the main environment. It gets its own virtual environment (`autopcr.parser_venv` in the
   path file):

   ```bash
   # on a machine with internet: download the two models that are not on PyPI
   bash experiments/06_comparison/baselines/stage_autopcr_parser_assets.sh
   # copy the staged files to <autopcr.parser_assets>/staged/ on LeoMed, then on the login node:
   python3.10 -m venv <autopcr.parser_venv>
   <autopcr.parser_venv>/bin/pip install -r third_party/AutoPCR/requirements_ee.txt
   <autopcr.parser_venv>/bin/pip install --no-deps <autopcr.parser_assets>/staged/en_core_web_trf-3.7.3-py3-none-any.whl
   mkdir -p <autopcr.parser_assets>/nltk_data/models
   python -m zipfile -e <autopcr.parser_assets>/staged/benepar_en3_large.zip <autopcr.parser_assets>/nltk_data/models
   ```

   The MD5 of `benepar_en3_large.zip` must be `395cf4036073b377346f59254430c2d1`.
3. **NLTK corpora.** `punkt_tab`, `averaged_perceptron_tagger_eng` and `wordnet`, in `~/nltk_data`
   (`python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng wordnet` on a machine with
   internet, then copy the folder). With NLTK 3.9.1 the older names `punkt` and
   `averaged_perceptron_tagger` do not work.

Then, per cohort: `DATASET=<hcy|gsc> sbatch slurm/autopcr_parse.sbatch` parses the corpus once for
both linkers, and `METHOD=autopcr_8b` or `METHOD=autopcr_70b` with `slurm/baseline.sbatch` runs the
linker.

On LeoMed the environment, assets and index of the thesis runs already exist at the locations the
path file names.

## RAG-HPO (`RAG-HPO/`)

Upstream <https://github.com/PoseyPod/RAG-HPO>, MIT licence. Two versions are kept:

| File | Upstream state | Used by |
|---|---|---|
| `rag_hpo_lib.py`, `system_prompts.json`, `build_vector_db.py` | commit `d2dd604` (2025-07-30) | the RAG-HPO baselines (8B and 70B) |
| `hp.obo` | HPO release 2026-06-23, downloaded when the baselines' database was built | the RAG-HPO baselines (8B and 70B) |
| `rag_hpo_lib_paper.py`, `system_prompts_paper.json`, `build_vector_db_paper.py` | commit `25c1ea7`, the code behind the published tables | the three reproduction runs |
| `system_prompts_figs1.json` | the prompt printed in the paper's Figure S1 | the reproduction with the published prompt |

`README.md` and `LICENSE` are upstream's. The notebook code was exported to the two `rag_hpo_lib`
modules with retrieval, parsing and mapping unchanged (module docstrings list what was removed).

The runs read their vector databases (`hpo_meta*.json`, `hpo_embedded*.npz`) and prompt files from
`raghpo.vector_db_dir` in the path file. On LeoMed that is the folder the thesis runs used. Its prompt files are identical to the ones
here (`tests/prompt_hashes.json`).
To build the databases again, into this folder:

```bash
python third_party/RAG-HPO/build_vector_db.py         # SapBERT-mnli encoder
python third_party/RAG-HPO/build_vector_db_paper.py \
    --hpo-json <hp.json of release 2024-08-13> --model <BGE-small snapshot folder>
```

`build_vector_db_paper.py` refuses any `hp.json` other than release 2024-08-13. `build_vector_db.py`
downloads the current `hp.obo` when the local file is older than its refresh period, so a rebuild
would use a newer HPO release than the stored database: reproduce the thesis with the stored one.
Both builds load their encoder from the Hugging Face cache.

## PhenoBERT (`PhenoBERT/`)

Upstream <https://github.com/EclipseCN/PhenoBERT>, commit `195d5df2ad9c6265e1de523f139694c7646eee29`,
MIT licence (`PhenoBERT/LICENSE`). PhenoBERT is a baseline and the default normaliser of PhenoJury.
It runs as a subprocess in its own environment (`hpo_extraction.phenojury.phenobert`), from
`PhenoBERT/phenobert/utils/` (`phenobert.dir` in the path file), and reads its data and weights by
relative paths (`../data/`, `../models/`, `../embeddings/`), so upstream's layout is kept.

Copied: the code, the four data files annotation reads, and upstream's `LICENSE`, `README.md` and
`requirements.txt`. The changes to upstream (loading model files under PyTorch 2.6 and later) are
listed in `PhenoBERT/PATCHES.md`, together with what was not copied.

**Weights.** The model weights and embeddings (about 2.3 GB, published by the PhenoBERT authors,
see upstream's README) are not tracked. `phenobert_weights.py` links them into
`PhenoBERT/phenobert/models/` and `PhenoBERT/phenobert/embeddings/` from `phenobert.weights_dir`,
a folder in PhenoBERT's layout. On LeoMed that is the checkout the thesis runs used:

```bash
python third_party/phenobert_weights.py            # check the source folder, list the links
python third_party/phenobert_weights.py --apply    # create them
```

**Environment.** PhenoBERT needs Stanza 1.4.1 and runs in its own environment
(`phenobert.python`). On LeoMed it exists. Elsewhere:

```bash
python -m venv --system-site-packages <phenobert env>      # from the main environment
<phenobert env>/bin/pip install stanza==1.4.1 prettytable
<phenobert env>/bin/python third_party/phenobert_stanza_offline.py
```

`phenobert_stanza_offline.py` patches Stanza 1.4.1 so that it never downloads, which compute nodes
need. PhenoBERT reads Stanza models from `stanza_dir_phenobert` (Stanza 1.4.1 format, with the
`mimic` package), not from `stanza_dir`, whose format differs. Under PyTorch 2.6 or later, Stanza
1.4.1 also needs `weights_only=False` in its model loaders (`stanza/models/*/trainer.py`,
`models/common/pretrain.py`, `models/charlm.py`, `models/langid/model.py`). The thesis runs used
loaders edited that way by hand. The edit is not scripted here.
