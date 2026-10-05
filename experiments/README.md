# experiments

The scripts behind every table, figure and number of the thesis, grouped by thesis chapter.
`docs/thesis_map.md` maps each thesis artifact to its script, config, inputs and outputs.

| Folder | Thesis | Scripts |
|---|---|---|
| `03_setup/` | chapter 3, appendix A | segmentation of the reports, the curated HCY ground truth (`ground_truth/`), the GSC+ subsets, fold files, dataset statistics |
| `04_treephenorag/` | chapter 4, appendix B | synthetic sentences, stored verifier scores (`score_store/`), the TreePhenoRAG protocol (`protocol/`), retrieval curves, index statistics |
| `05_phenojury/` | chapter 5, appendix C | juror generation (`generate/`), normalisation (`normalise/`), the PhenoJury protocol (`protocol/`), the count of labels and synonyms |
| `06_comparison/` | chapter 6, appendices A.5 and D | the baselines, the RAG-HPO reproduction, the cross-system comparison, conversion of predictions, GSC+ timings |
| `figures/` | all chapters | the generators of every figure, table and inline number (`make_all.py`) |

Prompts, thresholds, seeds, defaults, data order, sampling and metrics are those of the runs that
produced the thesis numbers (`docs/reproduction.md`), and `tests/unit/test_prompts_unchanged.py`
guards every prompt.

Most scripts are Hydra applications configured by `configs/experiments/<chapter>/<name>.yaml`. Each
config header names its command, model and the SLURM template that runs it on LeoMed (`slurm/`).
Paths in the configs are `${paths:<key>}` references into the path file.

A run writes into `<output_dir>/<result folder>/` (`output_dir` from the path file). The result
folders of the thesis runs on LeoMed have older names, with links under the current names beside
them (`docs/cluster.md` lists both).
