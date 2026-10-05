# Working on LeoMed

LeoMed is the ETH Zurich cluster for sensitive biomedical data. The HCY reports, every model and the
stored results of the thesis are there, and every GPU run of the thesis ran there with SLURM.

All work on LeoMed follows these rules:

1. **Compute nodes have no outbound internet.** Every model loads from a local folder: the
   templates set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and the configs take model
   folders from the path file, never hub names alone. A model, corpus or package that is not on
   LeoMed yet is downloaded on a machine with internet and copied in. The login node reaches the
   Python package index but not Hugging Face or GitHub downloads.
2. **HCY data stays on LeoMed.** Reports, segmentations, annotations, the curation log, the curated
   ground truth and every per-report output on HCY (predictions, generations, scores, Compare UI
   bundles) are never copied off the cluster, never committed, and never sent to a hosted model.
   Aggregate tables (the CSV files the figure scripts read) contain no report text and may be copied.
   GSC+ outputs are public data.

## Access

You need a LeoMed account and membership in the group that owns the project storage. Ask the
supervisor of the project for both. Logins go through two jump hosts. An `~/.ssh/config` for that:

```
Host jumphost.inf.ethz.ch
    User <ETH user name>
Host jump-biomed.leomed.ethz.ch
    User <ETH user name>
    ProxyJump jumphost.inf.ethz.ch
Host leomed
    HostName login-biomed.leomed.ethz.ch
    User <ETH user name>
    ProxyJump jump-biomed.leomed.ethz.ch
```

Then `ssh leomed`. The apps are reached through a tunnel, `ssh -L 8057:localhost:8057 leomed`, and
listen on `127.0.0.1` only (`apps/README.md`).

## Where things are

`configs/cluster_leomed.yaml` holds every location, and it is the only file with absolute paths.

| Key | Contents |
|---|---|
| `group_dir` | the project storage |
| `hcy.*` | HCY reports (`hcy.input_dir`), segmentation, raw and curated ground truth, folds, curation log |
| `models_dir`, `models.*` | language models, encoders, SapBERT, BGE-small, the eight jurors |
| `stanza_dir`, `stanza_dir_phenobert` | Stanza models (two versions) |
| `phenobert.*`, `autopcr.*`, `raghpo.*` | PhenoBERT code folder, weights and interpreter, AutoPCR parser environment, RAG-HPO vector databases |
| `results_dir` | the stored results of the thesis runs (old folder names) |
| `output_dir`, `logs_dir` | where new runs write |
| `thesis_dir` | the thesis LaTeX source, which the figure scripts write into |
| `conda.*` | the conda installation and the environment of the thesis runs |

`python -m hpo_extraction.path_lookup <key>` prints one value. If your runs must write somewhere
else (your own folder), copy the file, change `repo_dir`, `output_dir` and `logs_dir`, and set
`export HPO_PATHS=<your copy>`. Keep `results_dir` pointing at the stored results.

### Stored results

The thesis runs wrote their results under older folder names than the scripts here use. Links with
the current names sit beside them in `results_dir` and `hcy.dir` (created on 2026-10-04). Should a
stored folder be added later, the same tool adds its link (a dry run lists what it would do):

```bash
python tools/link_stored_results.py --root "$(python -m hpo_extraction.path_lookup results_dir)"          # dry run
python tools/link_stored_results.py --root "$(python -m hpo_extraction.path_lookup results_dir)" --apply
python tools/link_stored_results.py --root "$(python -m hpo_extraction.path_lookup hcy.dir)" --apply
```

Nothing is moved or deleted. The links let a new stage find the outputs of a stored one.

| Stored folder | Current name |
|---|---|
| `exp13_21_tree_score_cache` | `treephenorag_scores_synthetic` |
| `exp13_24_tree_score_cache_r3u` | `treephenorag_scores_terminfo` |
| `exp13_22_tree_protocol` | `treephenorag_protocol` |
| `exp00_07_query_set_variants` | `treephenorag_retrieval_curves` |
| `exp13_06_slm_ensemble` | `phenojury_generation_free_listing` |
| `exp14_01_prompt_full` | `phenojury_generation_other_prompts` |
| `exp14_03_normaliser_grid` | `phenojury_normalisation` |
| `exp14_04_prompt_jury_protocol` | `phenojury_protocol` |
| `exp13_08_phenobert` | `baseline_phenobert` |
| `exp13_03_raghpo_llama8b` | `baseline_raghpo_8b` |
| `exp13_04_raghpo_llama70b` | `baseline_raghpo_70b` |
| `exp13_19_autopcr_llama8b` | `baseline_autopcr_8b` |
| `exp13_20_autopcr_llama70b` | `baseline_autopcr_70b` |
| `exp13_11_raghpo_paper70b` | `raghpo_reproduction_published_code` |
| `exp13_15_raghpo_paper70b_meta` | `raghpo_reproduction_llama3_70b` |
| `exp13_17_raghpo_figs1_prompt` | `raghpo_reproduction_published_prompt` |
| `exp13_25_final_comparison` | `comparison` |
| `exp13_25_inputs` | `comparison_inputs` |
| `exp13_18_curated_gold_eval` | `hcy_ground_truth` |
| `index_statistics` | `synthetic_sentence_statistics` |
| `compare_bundles` | `compare_ui_bundles` |
| `curated_gold_<date>` (in `hcy.dir`) | `curated_ground_truth_<date>` |

PhenoBERT's weights are linked the same way, once per checkout
(`python third_party/phenobert_weights.py --apply`, `third_party/README.md`).

## Environment

The thesis runs used the conda environment named by `conda.env` in the path file. Use it to
reproduce thesis numbers:

```bash
source <conda.base>/bin/activate
conda activate <conda.env>
export PYTHONPATH=$PWD/src      # for interactive use; the SLURM templates set it themselves
```

`environment.yml` holds exact versions of the local development environment, which differ from
the cluster environment in at least PyTorch and transformers. To record the cluster versions:

```bash
conda env export -n <conda.env> --no-builds > environment_leomed.yml
```

PhenoBERT and AutoPCR's parser run in their own environments (`third_party/README.md`).

## Submitting jobs

```bash
cd <repo_dir>
conda activate <conda.env>
mkdir -p logs
sbatch slurm/treephenorag_protocol.sbatch                 # one job
INDEX=term_information sbatch slurm/treephenorag_score_store.sbatch   # an array job (40 shards)
squeue -u $USER                                           # queue
sacct -j <job id> --format=JobID,State,Elapsed,MaxRSS     # finished jobs
```

The job takes over the environment of the shell that submits it. `slurm/README.md` lists every
template with its GPU needs, and each config header in `configs/experiments/` names the template
that runs it. Logs go to `logs/<job name>_<job id>.out` and `.err`.

GPU jobs request the `gpu` partition. The 70B models (RAG-HPO 70B, AutoPCR 70B, the RAG-HPO
reproduction) need four GPUs (`--gres=gpu:rtx4090:4`), as the header of `slurm/baseline.sbatch`
shows. CPU-only analyses (the two protocols, the comparison) run on `compute`.
