# Compare UI

The five methods of the comparison (TreePhenoRAG, PhenoJury, PhenoBERT, RAG-HPO, AutoPCR) on one
document at a time, next to the ground truth: which terms each method found, missed or added, and
the evidence it gave. It covers the 20 HCY reports and 20 GSC+ abstracts drawn for the qualitative
appendix (thesis appendix D.1).

The app reads small per-document JSON bundles and opens no model. The bundles are built once on
LeoMed, where the predictions and the HCY text are.

## Build the bundles and the appendix examples

```bash
DATASET=gsc sbatch slurm/segment_reports.sbatch   # once: the GSC+ segmentation the bundles index
sbatch slurm/compare_ui_bundles.sbatch            # bundles for both cohorts, then the examples
```

The template runs `apps/compare_ui/build.py` for each cohort and then `apps/compare_ui/export_examples.py`,
which writes `examples_{hcy,gsc}.csv` for `tab_ch6_qualitative_{hcy,gsc}` of the thesis. The HCY
bundles and examples contain report text and stay on LeoMed.

The documents were drawn by `select_hcy_documents.py` and `select_gsc_documents.py` (seeds 20260909
and 20260919). Those two scripts are kept so that the draw can be checked, not to be rerun.

## Run

```bash
python apps/compare_ui/app.py --bundles <output>/compare_ui_bundles --port 8057
python apps/compare_ui/app.py --check       # list what the bundles contain, then exit
python apps/compare_ui/app.py --selftest    # synthetic bundles, no data needed
```

The scores the app shows per cohort are read from the comparison's `t1_overall.csv`, not recomputed.
