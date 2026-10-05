# hpo-phenotype-extraction

Extraction of Human Phenotype Ontology (HPO) terms from clinical text, with the two methods of the
master's thesis this repository accompanies:

- **TreePhenoRAG** walks the HPO graph from the top. At each term it retrieves the segments of the
  report that best match the term, asks a language model (the verifier) whether each segment
  states the term, and pools the answers into two scores: one decides whether to visit the term's
  children, the other whether to output the term.
- **PhenoJury** asks several small language models (the jurors) to list the phenotypes in each
  segment, maps what they write to HPO terms with a normaliser (PhenoBERT by default), and keeps the
  terms that at least *k* jurors name.

The repository contains both methods as a Python package with command-line tools, the scripts that
produced every table, figure and number of the thesis, the browser apps used for curation and
inspection, and job templates for the LeoMed cluster.

Abbreviations: HCY (the clinical cohort of the thesis, on LeoMed only), GSC+ (a public corpus of 228
annotated abstracts).

## Installation

```bash
conda env create -f environment.yml
conda activate hpo-extraction
pip install -e .
python -m pytest tests -q          # about two minutes, CPU only
```

Python 3.10. The methods need a GPU and local copies of the models (`configs/local_example.yaml`
lists them). PhenoBERT, used by PhenoJury, needs its own environment and its published weights
(`third_party/README.md`).

## Five-minute example

```bash
python examples/stand_in_demo.py --out output/stand_in_demo
```

This runs both methods on the two synthetic reports in `examples/synthetic_reports/`, with stand-in
models in place of the language models, so it needs no GPU and no model files. It writes one JSON
Lines file per method and prints the terms found:

```
treephenorag: wrote output/stand_in_demo/treephenorag_stand_in.jsonl
  syn_report_1: HP:0001250 Seizure (1.00)
  syn_report_2: no terms
phenojury: wrote output/stand_in_demo/phenojury_stand_in.jsonl
  syn_report_1: HP:0001250 Seizure (1.00)
  syn_report_2: no terms
```

The stand-ins are simple rules, so the terms say nothing about the methods' quality. Everything
between the models is the real code. The output format is described in `docs/data_formats.md`.

## Run on new text

Put each report in a `.txt` file. Point the path file at your model folders (copy
`configs/local_example.yaml`, fill it in, `export HPO_PATHS=<your copy>`), then:

```bash
hpo-treephenorag --config configs/apps/treephenorag.yaml --input reports/ --output treephenorag.jsonl
hpo-phenojury    --config configs/apps/phenojury.yaml    --input reports/ --output phenojury.jsonl
```

From Python:

```python
from hpo_extraction.treephenorag import TreePhenoRAG

method = TreePhenoRAG.from_config("configs/apps/treephenorag.yaml")
result = method.extract("He has had recurrent seizures since the age of two.", "report_1")
for term in result.terms:
    print(term.hpo_id, term.label, term.score, term.evidence[0].text)
```

`hpo_extraction.phenojury.PhenoJury` works the same way. The defaults in `configs/apps/` are the
configurations selected in the thesis:

| Method | Default |
|---|---|
| TreePhenoRAG | term-information index, 10 segments per term, expansion pooling P4 with threshold 5.58e-5, acceptance pooling LSE with threshold 0.99, verifier Llama-3.1-8B-Instruct in 8-bit |
| PhenoJury | eight jurors, HPO-Guided prompt, PhenoBERT normaliser, exact rule, vote per segment, k = 3 |

TreePhenoRAG makes many verifier calls per report (about 31,000 on HCY with these settings), so
`subtree_root` and `segments_per_term` in the config are the settings that bound its cost.

Clinical text with patient data must stay on the machine it is allowed on. On LeoMed that is the
cluster itself (`docs/cluster.md`).

## Reproduce the thesis

`docs/thesis_map.md` lists, for every table, figure and number, the script and command that produce
it. The short version:

```bash
python experiments/figures/make_all.py     # every generated table, figure and inline number
```

rebuilds them from the stored result tables. The result tables themselves come from the analysis
runs (`slurm/*protocol*.sbatch`, `slurm/comparison.sbatch`), which read the model outputs of the GPU
runs. All of these run on LeoMed (`docs/cluster.md`, `slurm/README.md`). `docs/reproduction.md` says
how far the thesis results were reproduced with this code.

## Layout

```
src/hpo_extraction/   the package: treephenorag/, phenojury/, retrieval, ontology, models, evaluation, baselines
experiments/          the thesis scripts, by chapter (03_setup, 04_treephenorag, 05_phenojury, 06_comparison, figures)
apps/                 Curation UI, Compare UI, TreePhenoRAG UI, PhenoJury UI
configs/              path files (cluster_leomed.yaml, local_example.yaml), one config per thesis run, app configs
slurm/                job templates for LeoMed
resources/            ontology, synthetic sentences, GSC+ corpus, word lists
third_party/          AutoPCR, RAG-HPO and PhenoBERT (copied, with their changes listed)
examples/             synthetic reports and the stand-in example
tests/                unit tests on synthetic data
docs/                 thesis map, reproduction, cluster guide, formats, glossary
tools/                links for stored results on LeoMed, privacy scans
```

Each folder has a short README.
