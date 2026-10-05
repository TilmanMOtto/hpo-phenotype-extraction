# configs

| File | Purpose |
|---|---|
| `cluster_leomed.yaml` | Every location on LeoMed: data, models, results, environments. The only file in the repository with absolute paths. |
| `local_example.yaml` | The same keys for another machine. Copy it, fill in your folders, and set `HPO_PATHS=<your copy>`. |
| `experiments/<chapter>/<name>.yaml` | One Hydra config per thesis run. The header gives the thesis section, the command, the models with their quantisation, the ontology file, and the output folder. |
| `apps/treephenorag.yaml`, `apps/phenojury.yaml` | Settings of the two applications. The defaults are the configurations selected in the thesis. |

The path file is chosen by the environment variable `HPO_PATHS`, and `configs/cluster_leomed.yaml`
is the default. Configs refer to its keys as `${paths:<key>}`, for example
`${paths:models.llama_3_1_8b_instruct}`. `python -m hpo_extraction.path_lookup <key>` prints one value.

Values in the experiment configs are those of the thesis runs. Keys whose value is empty on
purpose (for example `prompt_keys`, `evidence_path`, `hcy_gt_path`) are filled in by the SLURM
template or on the command line, as the comment above each key says.

The ontology of every run is `resources/util/hpo.json` (SHA-256 fingerprint `8330ca317e71`). The
thesis names HPO release 2024-08-13. The file itself has no version field
(`resources/util/PROVENANCE.md`).
