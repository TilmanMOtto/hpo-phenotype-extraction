# resources

Data files the code reads. None of them contains patient data.

| Path | Contents | Notes |
|---|---|---|
| `util/hpo.json` | The HPO graph every method and metric uses (18,354 terms below *Phenotypic abnormality*) | `util/PROVENANCE.md` |
| `util/minimal_gate.json` | Weights of the learned retrieval gate | |
| `util/stopwords.txt`, `util/NUM.txt` | Word lists of the phrase handling in `hpo_extraction.ontology.hpo_items` | |
| `synthetic_sentences/llama-3.3-70b-instruct/HP_*.txt` | 40 synthetic sentences per term, generated from the ontology alone by Llama-3.3-70B-Instruct (thesis appendix B.8) | `experiments/04_treephenorag/synthetic_sentences/` |
| `data/GSC_2024/` | The GSC+ corpus (228 abstracts) and the AutoPCR evaluation and development splits (206 and 22) | `data/GSC_2024/PROVENANCE.md` |
| `data/GSC_RAGHPO/` | The 114 GSC+ documents and the re-annotation of the RAG-HPO paper | `data/GSC_RAGHPO/PROVENANCE.md` |
| `data/hpo_code_lists/` | Lists of HPO codes the HCY annotation targeted (codes only) | |
| `nltk_data/` | NLTK's stopwords corpus | `nltk_data/PROVENANCE.md` |
| `reference/abbreviations.csv` | Abbreviations and spelling variants of the dictionary normaliser | `reference/PROVENANCE.md` |

HCY reports, segmentations, annotations and the curated ground truth are not here. They stay on
LeoMed (`hcy.*` keys of `configs/cluster_leomed.yaml`).
