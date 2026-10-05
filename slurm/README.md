# slurm

Job templates for LeoMed. Submit from the repository root with the conda environment active (the
job takes over the environment of the shell that submits it), after `mkdir -p logs`:

```bash
sbatch slurm/<template>.sbatch
```

`common.sh` is sourced by every template. It sets `HPO_PATHS` (default
`configs/cluster_leomed.yaml`), switches Hugging Face to offline mode, puts `src/` on the Python path,
and defines `p <key>`, which prints a value of the path file. Variables such as `DATASET`, `METHOD`
or `INDEX` select what a template runs, and extra arguments are passed on to the script as Hydra
overrides. The header of each template lists its options.

| Template | Thesis | GPU |
|---|---|---|
| `segment_reports.sbatch` | segmentation of HCY or GSC+ | no |
| `ground_truth.sbatch` | curated HCY ground truth | no |
| `dataset_statistics.sbatch` | HCY column of Table 3.1 | no |
| `synthetic_sentences.sh` | synthetic sentences (needs a hosted model and internet, not a compute node) | no |
| `treephenorag_score_store.sbatch` | stored verifier scores, array of 40 shards | yes |
| `treephenorag_protocol.sbatch` | chapter 4 tables and figures | no |
| `retrieval_curves.sbatch` | Figure 4.5 | no |
| `phenojury_generate.sbatch` | juror generations, one array task per juror (and prompt) | yes |
| `phenojury_normalise.sbatch` | normalisation grid, array (SapBERT on CPU) | no |
| `phenojury_protocol.sbatch` | chapter 5 tables and figures | no |
| `baseline.sbatch` | PhenoBERT, RAG-HPO, AutoPCR and the RAG-HPO reproduction | yes, except PhenoBERT |
| `autopcr_index.sbatch` | AutoPCR's dictionary and SapBERT index | yes |
| `autopcr_parse.sbatch` | AutoPCR's constituency parse, once per cohort | yes |
| `comparison_inputs.sbatch` | PhenoJury rows of the comparison | no |
| `comparison.sbatch` | chapter 6 tables, RAG-HPO reproduction table | no |
| `compare_ui_bundles.sbatch` | Compare UI bundles, qualitative examples of appendix D.1 | no |

Logs go to `logs/<job name>_<job id>.out`. The templates send no e-mail and no notification.
