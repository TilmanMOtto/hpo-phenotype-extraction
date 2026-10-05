# src/hpo_extraction

The Python package. Install it with `pip install -e .` from the repository root. The two methods,
the thesis scripts in `experiments/` and the apps all import it, so every function exists once.

| Subpackage | Contents |
|---|---|
| `treephenorag/` | TreePhenoRAG: traversal of the ontology (`traversal`), pooling operators P0 to P4 and LSE (`pooling`), the verifier prompt (`verifier_prompt`), the stored-score builder (`score_store`) and its offline evaluation (`stored_scores`), threshold selection (`selection`), and the application (`pipeline`, `cli`). |
| `phenojury/` | PhenoJury: the four prompts (`prompts`), juror generation (`generation`, `generation_prompts`), normalisers (`normalisers`, `phenobert`), voting (`vote`), and the application (`pipeline`, `cli`). |
| `retrieval/` | Sentence encoding, retrieval of segments for a term, the term-information index, the label and synonym index of the dictionary normaliser (`surface_index`, `ontology_index`), descendant closure. |
| `ontology/` | The HPO graph (`hpo_tree`) read from `resources/util/hpo.json`, phrase items, NLTK data path. |
| `data/` | Report loading and segmentation (Stanza, or a rule-based splitter). |
| `models/` | Loading of the verifier and the jurors from local folders, GPU memory helpers. |
| `evaluation/` | Metrics of the thesis (`metrics/`: flat and hierarchical scores, calibration, error categories, recall decomposition, cost), statistics (`stats/`: bootstrap, permutation tests, Holm correction, conformal risk control, folds), datasets, and the result-table library the comparison uses (`result_tables/`). |
| `baselines/` | Runners for PhenoBERT, RAG-HPO (current and published code) and AutoPCR. |
| `curation/` | Reading and writing the curation event log, shared by the Curation UI and the ground-truth build. |
| `utils/` | Resuming interrupted runs, MLflow guard. |

Top-level modules:

- `paths.py` reads the path file (`configs/cluster_leomed.yaml`, or the file `HPO_PATHS` names) and
  registers the `${paths:<key>}` resolver the experiment configs use. `python -m
  hpo_extraction.path_lookup <key>` prints one value (the SLURM templates use it).
- `results.py` defines the output of both applications (`ReportResult`, `TermScore`, `Evidence`)
  and reads and writes it as JSON Lines (`docs/data_formats.md`).
