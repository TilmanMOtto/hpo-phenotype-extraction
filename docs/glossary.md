# Glossary

Terms of the thesis and the names the code uses for them. Comments, docstrings and documentation
use the thesis term. Code names that stored files depend on keep their original form
(`docs/stored_formats.md`), and some of those contain words the documentation avoids.

Abbreviations: HPO (Human Phenotype Ontology), HCY (the clinical cohort of the thesis), GSC+ (the
public corpus of 228 annotated abstracts), CV (cross-validation), CRC (conformal risk control), LSE
(log-sum-exp).

## Data

| Thesis term | Code |
|---|---|
| report | `report_id`, in older modules `patient_id` or `doc_id` |
| segment | `segment_idx`, `sent_index`, `sentence_number` (one number with three names) |
| ground truth | `gold`, `gold_path`, `ground_truth_path` |
| annotated term, annotated pair | `gold_terms`, `n_gold` (stored column names) |
| curated ground truth | `hcy_ground_truth_curated.csv`, built by `experiments/03_setup/ground_truth/` |
| evidence location (segment and trigger word of an annotation) | `location`, `evidence_location`, stored columns `anchor`, `anchor_how` |
| HCY, GSC+ (114), GSC+ (206) | cohort keys `hcy`, `gsc_raghpo_ann`, `gsc_2024_eval_206` (`gsc206` in the PhenoJury protocol) |
| development split | repetition 0, outer fold 0 training split, `dev_reports` |
| laboratory-value and implicit-description subgroups | `lab_value`, `implicit`, the `q_*` qualifier columns |

## TreePhenoRAG

| Thesis term | Code |
|---|---|
| verifier | `LlamaLogitsLLM`, `verifier` (Llama-3.1-8B-Instruct, 8-bit) |
| segment score, margin | `sigma` (probability of *Yes*), `margin` (logit of *Yes* minus logit of *No*) |
| expansion score, expansion threshold | `prune_score`, `pool_pr`, `tau_prune`, `expansion_threshold` |
| acceptance score, acceptance threshold | `accept_score`, `pool_acc`, `tau_accept`, `acceptance_threshold` |
| pooling operators P0 to P4, LSE | `P0`, `P1`, `P2`, `P3_1`, `P3_2`, `P3_S`, `P4`, `lse_beta1` |
| synthetic sentences, synthetic-sentence index | `resources/synthetic_sentences/`, retrieval index key `exemplar` |
| term information, term-information index | label, definition and synonyms, `term_information_index`, retrieval index key `ontology_r3` |
| descendant closure | `descendant_closure`, config value `ctx_type: union` |
| stored scores, offline evaluation | `score_store`, `stored_scores`, `evaluate_offline_cohort`, `OfflineResult` |
| coverage | `attained_coverage` |
| risk tolerance alpha, conformal risk control | `alpha`, `crc` |
| variants V0 to V3, upper bound | `LADDER`, `oracle_expansion` |
| transferred configuration (GSC+ (114)) | `transfer`, predictions `gsc_raghpo_ann_transferred.csv` |

## PhenoJury

| Thesis term | Code |
|---|---|
| juror, jury | `model_key`, `jurors`, `slm` in older modules |
| full pool, core-3 jury | `full_pool`, `fixed:core3` |
| Free Listing, HPO-Guided, Evidence Spans, Recall-First | prompt keys `p0_baseline`, `q2_sentence_last`, `q4_span_json`, `q7_recall` |
| normaliser: PhenoBERT, Dictionary, SapBERT | `phenobert_candidates`, `dictionary`, `sapbert` (`phenobert_raw` is PhenoBERT run on the report itself) |
| exact rule, closure rule | `rule: exact`, `rule: closure` (`closure_reduced` in the vote module) |
| vote scope: report, window, segment | `unit` |
| vote threshold k | `k` |
| PhenoBERT alone (J0) | `phenobert` row, `j0_predictions_path` |
| jury variant | `condition` (`ConditionResult`, `evaluate_conditions`), stored column `arm` |

## Evaluation

| Thesis term | Code |
|---|---|
| micro precision, recall, F1 | `micro_precision`, `micro_recall`, `micro_f1` |
| hierarchical precision, recall and F, CoPHE | `micro_hp`, `micro_hr`, `micro_hf`, `micro_cophe_f1` (and the `macro_` columns) |
| false-positive categories | error buckets `ancestor`, `descendant`, `sibling`, `same_branch`, `unrelated`, `no_gold`, `invalid` |
| recall decomposition | buckets `pruning`, `retrieval`, `judgement`, `pooling`, `residual` |
| configuration (point on a precision-recall curve) | stored column `operating_point` |
| nested cross-validation | 5 outer folds, 10 repetitions, `folds_<cohort>.csv` |
