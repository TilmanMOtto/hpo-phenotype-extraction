# Thesis map

Which script produces each table, figure and number of the thesis, from which inputs.
Abbreviations: HCY (the clinical cohort), GSC+ (the public corpus), CV (cross-validation).

Every generated table, figure and inline number is written by `experiments/figures/`. The
generators compute nothing themselves: they read the CSV files that six analysis runs write
(section 1), and those runs read the outputs of the model runs (section 2).

```bash
python experiments/figures/make_all.py              # every chapter, then an audit of the output folders
python experiments/figures/make_all.py --chapter 4  # one chapter
```

They read from `results_dir` and write into `thesis_dir/thesis_figures/` (PDF) and
`thesis_dir/thesis_figures_latex/` (LaTeX), both from the path file. A generator whose source table
is missing prints "skipped, waiting on a source" and exits with status 3, and `make_all.py` lists it.

## 1. Analysis runs

All run on CPU. "LeoMed" means the run reads HCY data.

| Run | Command | Writes (under `results_dir`) |
|---|---|---|
| dataset statistics | HCY: `sbatch slurm/dataset_statistics.sbatch` (LeoMed). GSC+: `python experiments/03_setup/dataset_statistics.py --cohorts gsc_raghpo_ann gsc_2024_eval_206` | `dataset_statistics/dataset_statistics.csv` |
| synthetic-sentence statistics | `python experiments/04_treephenorag/index_statistics.py` | `synthetic_sentence_statistics/index_statistics.csv` |
| TreePhenoRAG protocol | `sbatch slurm/treephenorag_protocol.sbatch` (LeoMed) | `treephenorag_protocol/{tables/*.csv, selected_configuration.json, predictions/}` |
| retrieval curves | `sbatch slurm/retrieval_curves.sbatch` (LeoMed) | `treephenorag_retrieval_curves/{operating_points,paired_deltas}.csv` |
| PhenoJury protocol | `sbatch slurm/phenojury_protocol.sbatch`, and `COHORT=gsc206` for GSC+ (206) (LeoMed) | `phenojury_protocol/{hcy,gsc206}/{tables/*.csv, manifest.json}` |
| comparison | `sbatch slurm/comparison_inputs.sbatch`, then `sbatch slurm/comparison.sbatch` (LeoMed) | `comparison/{tables/*.csv, manifest.json}`, including `t6_raghpo_replication.csv` |
| qualitative examples | `sbatch slurm/compare_ui_bundles.sbatch` (LeoMed) | `compare_ui_bundles/qualitative/examples_{hcy,gsc}.csv` |

## 2. Model runs and data preparation

| Run | Command | Inputs | Output used downstream | GPU |
|---|---|---|---|---|
| synthetic sentences (appendix B.8) | `bash slurm/synthetic_sentences.sh` (needs a hosted Llama-3.3-70B-Instruct) | `resources/util/hpo.json` | `resources/synthetic_sentences/` (in the repository) | hosted |
| segmentation | `sbatch slurm/segment_reports.sbatch`, `DATASET=gsc` for GSC+ | reports | `segmented_reports.csv` per cohort | no |
| curated ground truth (3.2, A.3) | `sbatch slurm/ground_truth.sbatch` | curation log | `hcy.curated_dir` | no |
| fold files (3.4) | `python experiments/03_setup/make_folds.py` (seed 0) | ground truth | `folds_<cohort>.csv` | no |
| GSC+ (206) and (114) subsets | `experiments/03_setup/extract_gsc_eval_split.py`, `extract_raghpo_subset.py`, `write_gsc_ground_truth.py` | AutoPCR and RAG-HPO releases | `resources/data/GSC_2024/`, `resources/data/GSC_RAGHPO/` | no |
| stored verifier scores (B.2) | `INDEX=term_information sbatch slurm/treephenorag_score_store.sbatch` and `INDEX=synthetic`, 40 shards each, `DATASET=gsc` for GSC+ | segments, `hpo.json`, Llama-3.1-8B-Instruct (8-bit), all-mpnet-base-v2 | `treephenorag_scores_terminfo/`, `treephenorag_scores_synthetic/` | yes |
| juror generations (5.3) | `sbatch slurm/phenojury_generate.sbatch` (Free Listing), `PROMPTS=other sbatch --array=0-23 slurm/phenojury_generate.sbatch` | segments, eight jurors | `phenojury_generation_free_listing/`, `phenojury_generation_other_prompts/` | yes |
| normalisation (5.3, C.2) | `sbatch --array=0-<n-1> slurm/phenojury_normalise.sbatch` | generations, PhenoBERT, SapBERT | `phenojury_normalisation/` | no |
| PhenoBERT baseline | `METHOD=phenobert sbatch ... slurm/baseline.sbatch` | reports | `baseline_phenobert/` | no |
| RAG-HPO 8B and 70B | `METHOD=raghpo_8b` / `raghpo_70b` with `slurm/baseline.sbatch` | reports, RAG-HPO vector database | `baseline_raghpo_8b/`, `baseline_raghpo_70b/` | yes |
| AutoPCR 8B and 70B | `slurm/autopcr_index.sbatch` once, `slurm/autopcr_parse.sbatch` per cohort, then `METHOD=autopcr_8b` / `autopcr_70b` | reports, SapBERT index | `baseline_autopcr_8b/`, `baseline_autopcr_70b/` | yes |
| RAG-HPO reproduction (A.5) | `METHOD=raghpo_published_code`, `raghpo_llama3_70b`, `raghpo_published_prompt` | GSC+ (114) | `raghpo_reproduction_*/` | yes |
| GSC+ timings per subset | `python experiments/06_comparison/recover_gsc_subset_timing.py` | job logs | `gsc_subset_timing/gsc/subset_seconds.csv` | no |

The configs in `configs/experiments/` hold every setting of these runs, and their headers name the
template and the models.

## 3. Generated tables and figures

| Thesis float (label) | Section | Generator | Reads | Run |
|---|---|---|---|---|
| `tab_ch3_datasets` (tab:datasets) | 3.2 | `tables_ch3.py` | `dataset_statistics.csv` | dataset statistics |
| `fig_ch3_protocol` (fig:ch3-protocol) | 3.4 | `fig_ch3_protocol.py` | none (schematic) | |
| `fig_ch3_hpo_fragment` | A.1 | `fig_ch3_hpo_fragment.py` | `resources/util/hpo.json` | |
| `fig_ch4_pipeline` | 4.2 | `fig_ch4_pipeline.py` | none (schematic) | |
| `fig_ch4_ablation_ladder` | 4.5 | `fig_ch4_ablation_ladder.py` | `LADDER` of the TreePhenoRAG protocol | |
| `tab_ch4_main_comparison` (tab:ch4-main-comparison) | 4.6 | `tables_ch4.py` | `comparison/tables/t1_overall.csv` | comparison |
| `fig_ch4_recall_decomposition` | 4.6 | `fig_ch4_recall_decomposition.py` | `recall_decomposition.csv` | TreePhenoRAG protocol |
| `fig_ch4_score_distributions` | 4.6 | `fig_ch4_score_distributions.py` | `score_distribution_summary.csv`, `score_{logit,survival,pr}_{expansion,acceptance}.csv` | TreePhenoRAG protocol |
| `fig_ch4_error_analysis` | 4.6 | `fig_ch4_error_analysis.py` | `error_taxonomy.csv` | TreePhenoRAG protocol |
| `tab_ch4_ablation_ladder` (tab:tpr-ladder) | 4.6 | `tables_ch4.py` | `core_quality.csv`, `ablation_tests.csv` | TreePhenoRAG protocol |
| `fig_ch4_retrieval` | 4.6 | `fig_ch4_retrieval.py` | `operating_points.csv`, `paired_deltas.csv`, `selected_configuration.json` | retrieval curves, TreePhenoRAG protocol |
| `fig_ch4_cost_coverage` | 4.6 | `fig_ch4_cost_coverage.py` | `alpha_sensitivity.csv` | TreePhenoRAG protocol |
| `tab_ch4_selection_frequency` (tab:tpr-stability) | B.1 | `tables_ch4.py` | `selection_choices.csv`, `selection_stability.csv` | TreePhenoRAG protocol |
| `tab_ch4_pooling_operators` (tab:tpr-pooling) | B.3 | `tables_ch4.py` | `pooling_operators.csv`, `acceptance_ranking.csv` | TreePhenoRAG protocol |
| `tab_ch4_segment_ablation` (tab:tpr-segments) | B.3 | `tables_ch4.py` | `s_ablation.csv` | TreePhenoRAG protocol |
| `tab_ch4_retrieval_sufficiency` (tab:app-retrieval) | B.5 | `tables_ch4.py` | `retrieval_sufficiency.csv` | TreePhenoRAG protocol |
| `tab_ch4_alpha_sensitivity` (tab:app-alpha) | B.6 | `tables_ch4.py` | `alpha_sensitivity.csv` | TreePhenoRAG protocol |
| `fig_ch4_reliability` | B.7 | `fig_ch4_reliability.py` | `score_logit_acceptance.csv`, `calibration_crossfit.csv`, `score_distribution_summary.csv` | TreePhenoRAG protocol |
| `fig_ch5_pipeline` | 5.2 | `fig_ch5_pipeline.py` | none (schematic) | |
| `tab_ch5_main_comparison` (tab:ch5-main-comparison) | 5.5 | `tables_ch5.py` | `s7_headline.csv`, `t1_overall.csv` | PhenoJury protocol, comparison |
| `tab_ch5_jury_ablation` (tab:pj-main) | 5.5 | `tables_ch5.py` | `s7_headline.csv`, `s7_significance.csv` | PhenoJury protocol |
| `fig_ch6_subgroups` (fig:ch6-subgroups) | 5.5 | `fig_ch6_subgroups.py` | `t1_overall.csv`, `t3_subgroups.csv` | comparison |
| `fig_ch5_recall_decomposition` | 5.5 | `fig_ch5_recall_decomposition.py` | `s9_recall_decomposition.csv`, `gsc206/manifest.json` | PhenoJury protocol |
| `fig_ch5_error_analysis` | 5.5 | `fig_ch5_error_analysis.py` | `s9_error_taxonomy.csv` | PhenoJury protocol |
| `fig_ch5_aggregation_curves` | 5.5 | `fig_ch5_aggregation_curves.py` | `s3_curves.csv` | PhenoJury protocol |
| `tab_ch5_prompt_sensitivity` (tab:pj-prompts) | 5.5 | `tables_ch5.py` | `s8_prompt_sensitivity.csv`, `s4_interaction.csv` | PhenoJury protocol |
| `tab_ch5_normaliser_exchange` (tab:pj-normaliser) | C.2 | `tables_ch5.py` | `s2_prompt_normaliser_grid.csv` | PhenoJury protocol |
| `tab_ch5_model_prompt_interaction` (tab:app-interaction) | C.3 | `tables_ch5.py` | `s4_interaction.csv`, `s4_decomposition.csv` | PhenoJury protocol |
| `tab_ch5_selection_frequency` (tab:app-pj-stability) | C.4 | `tables_ch5.py` | `s5_selection_frequency.csv`, `s5_selection.csv` | PhenoJury protocol |
| `tab_ch6_overall_comparison` (tab:ch6-overall-comparison) | 6.1 | `tables_ch6.py` | `t1_overall.csv` | comparison |
| `tab_ch6_literature` (tab:ch6-literature) | 6.2 | `tables_ch6.py` | `t1_literature.csv` (published values, typed into `configs/experiments/06_comparison/comparison.yaml`, key `literature`), `t1_overall.csv` | comparison |
| `fig_ch6_cost_quality` | 6.4 | `fig_ch6_cost_quality.py` | `t1_overall.csv`, `t4_cost.csv` | comparison |
| `tab_ch6_raghpo_replication` (tab:app-raghpo-replication) | A.5 | `tables_ch6.py` | `t6_raghpo_replication.csv`, `t1_literature.csv`, `t1_overall.csv` | comparison |
| `tab_ch6_cost_detail` (tab:ch6-cost-detail) | A.6 | `tables_ch6.py` | `t4_cost.csv`, `t1_overall.csv` | comparison |
| `tab_ch6_hierarchy` (tab:ch6-hierarchy) | D | `tables_ch6.py` | `t2_hierarchy.csv` | comparison |
| `tab_ch6_decisive_subset` (tab:ch6-decisive-subset) | D | `tables_ch6.py` | `t1_overall.csv`, `t3_label_audit.csv`, `t3_subgroups.csv` | comparison |
| `tab_ch6_qualitative_hcy` (tab:app-qualitative-hcy) | D.1 | `tables_ch6.py` | `examples_hcy.csv` (HCY text, LeoMed only) | qualitative examples |
| `tab_ch6_qualitative_gsc` (tab:app-qualitative-gsc) | D.1 | `tables_ch6.py` | `examples_gsc.csv` | qualitative examples |
| `num_ch3_inline` (`\chThree*`) | 3, A | `num_ch3_inline.py` | `dataset_statistics.csv` | dataset statistics |
| `num_ch4_inline` (`\chFour*`) | 4, B | `num_ch4_inline.py` | protocol tables (`calls_per_report`, `captured_mass`, `replay_fidelity`, `containment`, `calibration_crossfit`, `delta_m_sensitivity`, `recall_decomposition`, `blocking_causes`, `blocking_depth`, `near_miss`), `index_statistics.csv` | TreePhenoRAG protocol, synthetic-sentence statistics |
| `num_ch5_inline` (`\chFive*`) | 5, C | `num_ch5_inline.py` | `manifest.json`, `s0_cell_inventory`, `s5_selection_frequency`, `s7_headline`, `s7_significance`, `s7_exploratory_fixed_jury`, `s2_gold_restricted`, `t4_cost.csv` | PhenoJury protocol, comparison |

Built but not included in the thesis: `fig_ch4_score_distributions_2` and `_3`, `fig_ch6_headline`,
`fig_ch6_pr_plane`, `fig_ch6_hierarchy`, `tab_ch5_union_recall_gate`, `tab_ch5_leave_one_out`,
`tab_ch6_complementarity`, `tab_ch6_overall_macro`, `tab_ch6_scoring`.

Seven inline-number macros have no source and expand to a bold `??`: `\chFourPeakGpuGb`,
`\chFourDeployHoursOnt`, `\chFourDeployHoursEx`, `\chFivePhenobertProposedShareGsc`,
`\chFivePeakMemSequential`, `\chFivePeakMemParallel`, `\chFiveSapbertIndexBuild`. None of them is used
in the thesis text. `num_ch4_inline.md` and `num_ch5_inline.md` say what would produce each.

## 4. Tables typed by hand

| Table | Section | Source of the values |
|---|---|---|
| tab:tpr-operators | 4.3 | definitions of the pooling operators, `hpo_extraction.treephenorag.pooling` and the `POOLINGS` map of `treephenorag/pipeline.py` |
| tab:pj-jurors | 5.3 | the juror list, `models.jurors.*` in the path file and `MODEL_KEYS` of `phenojury/generation.py` |
| tab:tpr-hparams | B.1 | grids in `configs/experiments/04_treephenorag/protocol.yaml` (`poolings_prune`, `poolings_accept`, `tau_accept_grid`, `s_ablation`, `alpha_sweep`, `delta_m_sweep`) |
| tab:pj-hparams | C.1 | `configs/experiments/05_phenojury/protocol.yaml`, `normalise.yaml` (`sapbert_tau: 0.8`), `generate_free_listing.yaml` (`max_new_tokens: 512`, 1536 for three models) |

## 5. Numbers typed in the text

Most numbers in the text are read from the CSV files of section 1, by section:

| Section | Source |
|---|---|
| Abstract, 1, 7 | `comparison/tables/t1_overall.csv`, `t3_subgroups.csv` |
| 3.1, 3.2, A.2, A.3 | `dataset_statistics.csv`, `t3_label_audit.csv`, `t3_subgroups.csv`, `resources/data/GSC_2024/` id lists, `hpo.json` (`experiments/05_phenojury/count_surface_strings.py` counts its terms and strings) |
| 3.4 | fold files (`make_folds.py`, seed 0), `n_bootstrap` and `n_permutations` in the configs |
| 4.6, 4.7, appendix B | `treephenorag_protocol/tables/` (`selection_*`, `recall_decomposition`, `core_quality`, `score_distribution_summary`, `calibration_crossfit`, `captured_mass`, `error_taxonomy`, `ablation_tests`, `pooling_operators`, `acceptance_ranking`, `s_ablation`, `retrieval_sufficiency`, `alpha_sensitivity`, `cache_calls`, `calls_per_report`), `selected_configuration.json`, `t1_overall.csv`, `t4_cost.csv`, `configs/experiments/04_treephenorag/synthetic_sentences.yaml` |
| 5.5, appendix C | `phenojury_protocol/{hcy,gsc206}/tables/` (`s7_headline`, `s7_significance`, `s9_recall_decomposition`, `s9_error_taxonomy`, `s2_prompt_normaliser_grid`, `s5_selection_frequency`, `s5_selection`, `s4_interaction`, `s4_decomposition`, `s8_prompt_sensitivity`), `t3_subgroups.csv` |
| 6.1 to 6.4, appendix D | `comparison/tables/` (`t1_overall`, `t1_literature`, `t2_hierarchy`, `t3_subgroups`, `t4_cost`, `t5_complementarity`, `t6_raghpo_replication`) |
| 4.3, A.2 | arithmetic on the cohort size (1/(n+1) for n = 94) and on 1,146 / (118 x 18,354) |
| 5.3, A.5 | configs: `max_new_tokens`, `sapbert_tau`, AutoPCR's `tau_1`, `tau_2`, `k` |
| examples in 4.6, 5.5, D.2 | per-sentence generations in the Compare UI bundles (HCY on LeoMed) |

### Numbers with no script source

| Number | Section | Origin |
|---|---|---|
| 283 h 25 min of generation for 18,281 terms, 73 terms from an earlier run | 4.5, 6.4, B.8 | printed by the synthetic-sentence generation run at its end. The log is not in this repository, and its location on LeoMed is not recorded. |
| 23 records in the second annotation pass | 3.2, A.3 | the curation record on LeoMed |
| 118 patients, cohort composition, 30 years, 18 records without phenotype, micro F1 0.90 of PhenoRAG | 3.2, 4.1, A.2 | cited from Berndt et al. 2025 |
| MedGemma selected in all 50 HCY evaluations in an earlier run | C.4 | an earlier PhenoJury protocol run with prompt and normaliser fixed, whose tables were replaced when the protocol was rerun |
| Phi-4: about a quarter of the layers in CPU memory | C.1 | the device map of the model loader, not written to a table |

## 6. Stored results

The CSV files above exist on LeoMed under `results_dir`, in folders with older names and links
with the current ones beside them (`docs/cluster.md`), so `make_all.py` reads them directly. A copy of the aggregate tables (no report text) can be made with
`rsync` to another machine and read through `HPO_PATHS`.
