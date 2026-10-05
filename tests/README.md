# tests

```bash
python -m pytest tests -q
```

Unit tests of the package, the thesis scripts and the apps, on synthetic data only (`fixtures/`
builds toy ontologies, synthetic run folders and synthetic reports with identifiers `SYN001` and up).
No test needs a GPU, a model file, the network or HCY data. Three AutoPCR tests skip themselves when
the NLTK corpora `punkt_tab`, `averaged_perceptron_tagger_eng` and `wordnet` are not installed.

Tests that guard the thesis results:

| Test | Checks |
|---|---|
| `test_prompts_unchanged.py` | every prompt template hashes to the value recorded from the thesis runs (`tests/prompt_hashes.json`) |
| `test_applications.py` | both applications end to end on synthetic text with stand-in models, and that the pooling of the application equals the pooling the thesis numbers came from |
| `test_stored_label_aliases.py` | curation logs written with the old source names load under the new ones |
