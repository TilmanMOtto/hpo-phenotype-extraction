# TreePhenoRAG UI

Inspection of TreePhenoRAG runs that wrote one result folder per threshold: for one report or a
whole cohort, which nodes of the ontology were visited, where pruning cut off an annotated term,
what kind of error each false positive is, and how well the acceptance score is calibrated.

The app discovers run folders under `--output-base` by their stored names. These are the runs of the
earlier TreePhenoRAG variants (`exp13_00_tree_gate_lr`, `exp13_01_tree_noisyor`,
`exp13_02_tree_no_prune`, `exp13_09_tree_lse_beta1`, and the flat baseline `exp13_05_topm_retrieval_slm`),
which are kept on LeoMed. The thesis results of chapter 4 come from the stored verifier scores and
the TreePhenoRAG protocol, not from these runs. The app is kept as a diagnostic tool.

## Run

```bash
python apps/treephenorag_ui/app.py --port 8056          # on LeoMed, through an SSH tunnel
python apps/treephenorag_ui/app.py --selfcheck          # validate every discovered run, then exit
```

Options: `--output-base` (folder with `<run>/<cohort>/`), `--hcy-gt`, `--hcy-segments`, `--gsc-dir`,
`--deepdive-frame` (the 20-report HCY draw of `apps/compare_ui/select_hcy_documents.py`),
`--thesis-tables` (result tables used for the self-check's cross-check), `--cache-dir`.

The unit tests (`tests/unit/test_treephenorag_ui.py`) build synthetic run folders with
`tests/fixtures/exp13_output.py`. The app starts on those folders and lists their runs.
