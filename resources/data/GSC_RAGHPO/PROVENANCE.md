# GSC_RAGHPO, the corpus the RAG-HPO paper actually evaluated on

`resources/data/GSC_2024` is the full 228-document GSC+ corpus. The RAG-HPO paper reports its GSC
numbers on **neither the same documents nor the same ground truth**, so its published table is not
comparable to ours without this resource.

**Paper**: Garcia BT, Westerfield L, Yelemali P, Gogate N, Rivera-Munoz EA, Du H, Dawood M, Jolly A,
Lupski JR, Posey JE. *Improving automated deep phenotyping through large language models using
retrieval-augmented generation.* Genome Medicine 2025;17:91. doi:10.1186/s13073-025-01521-w

## What they did

> "We also evaluated RAG-HPO and other HPO analysis software using the gold standard corpora (GSC),
> which contains 229 entries describing genetic conditions. After applying the same filtering
> criteria as for our case study cohort (i.e., removing entries with fewer than four distinct
> phenotypes and manually assigning HPO terms), we curated a final dataset of 114 entries
> comprising 1013 HPO terms, of which 415 were unique, averaging 8.9 terms per case."

Two changes, not one: a **document filter** and a **re-annotation**. Their annotation policy is one
most-specific term per phenotype, and they explicitly strip the broader ancestor terms the original
GSC+ annotations carry alongside the specific ones (their example: entry 1347096 annotates both
HP:0002671 *basal cell carcinoma* and the more general HP:0030731 *carcinoma*. They keep only the
former). That is why their term count is *lower* than the original annotations for the very same
114 documents.

Note the document filter cannot be reproduced from a rule: applied to the original GSC+
annotations, "at least four distinct phenotypes" keeps 181 of 228 documents, not 114. Only one of
their 114 documents falls below four terms under the original annotations. The subset is a property
of their annotation, so it has to be taken from their data, which is what this resource does.

## Source

| | |
|---|---|
| Repository | https://github.com/PoseyPod/RAG-HPO |
| File | `RAG-HPO Tests and Data Analysis copy.xlsx` (repo root) = the paper's Additional file 3 |
| Downloaded | 2026-08-05 |
| Sheets used | `GSC Input` (col `ID`) → `document_ids.txt`; `GSC Manual Annotations ` (cols `ID`, `hpo_term`, `hpo_description`) → `annotations.csv`, the trailing space in the second sheet name is upstream's |

Regenerate with:

```bash
python experiments/03_setup/extract_raghpo_subset.py
```

The script cross-checks every id against `resources/data/GSC_2024` and fails if one is absent. The
workbook itself is not vendored.

## Contents

* `document_ids.txt`, 114 GSC+ document ids, one per line, sorted the way
  `hpo_extraction.evaluation.datasets.gsc._gsc_doc_ids` sorts the corpus (lexicographic on the string, so
  `998578` sorts last). **All 114 are present in `resources/data/GSC_2024`.**
* `annotations.csv`, `doc_id,hpo_id,hpo_description`, 1011 rows, deduplicated on
  `(doc_id, hpo_id)` and sorted. Rows whose `hpo_term` is not a well-formed `HP:#######` are
  dropped. Upstream's `Category` column is constant (`Abnormal`) and is not carried.

## Reproduction of the paper's Table 1

| | paper | this resource + `GSC_2024` |
|---|---|---|
| their subset, **their** annotation | 114 docs, 1013 terms, 415 unique, 8.9/doc | 114, **1011**, 415, 8.87 |
| their subset, **original** GSC+ annotation | 114 docs, 1323 terms, 447 unique, 11.6/doc | 114, **1322**, **448**, 11.60 |

The two-term and one-term gaps are unexplained and small. Most likely they are duplicate rows that
our `(doc_id, hpo_id)` dedup collapses and their count did not, or a row dropped by our
`HP:#######` well-formedness filter. They are recorded here rather than reconciled, because any
reconciliation would be a guess dressed up as agreement.

For orientation: of the 1011 pairs in their annotation, 899 also appear in the original GSC+
annotation of the same document. So roughly 11% of their ground truth is terms the corpus does not
annotate, and the original annotation carries 423 pairs they dropped.

## Recall ceilings

Not every ground truth code exists as a scorable node in `resources/util/hpo.json`, so each ground truth
imposes a different hard ceiling on recall for any method that predicts from the ontology graph.
This is the number to quote before comparing anything:

| cohort | documents | doc-term pairs | unique terms | recall ceiling |
|---|---|---|---|---|
| `gsc` (full corpus) | 228 | 1823 | 507 | 0.851 |
| `gsc_raghpo` (their docs, original ground truth) | 114 | 1322 | 448 | 0.887 |
| `gsc_raghpo_ann` (their docs, their ground truth) | 114 | 1011 | 415 | 0.948 |

Their annotation is the easiest of the three to score well on, before any method is involved: it is
newer, so fewer of its codes are obsolete or outside the phenotypic-abnormality subtree.

## How the result-table library uses this

Two derived cohorts, both scored from the **existing** full-GSC+ prediction artifacts under
`<results_dir>/<exp_id>/gsc/`, no inference is re-run, because every driver writes one summary
line per report and subsetting report ids at scoring time is exact.

* `gsc_raghpo`, their 114 documents, original GSC+ ground truth. Isolates the document-selection effect.
* `gsc_raghpo_ann`, their 114 documents, their ground truth. The actual replication. This is the row to
  compare against their Table 5.

One note when reading their Table 5: their precision, recall and F1 are each the **mean of the per-case value** (their Methods: "calculate precision, recall, and F1 scores per case and we reported the average of these scores"). Their released workbook reproduces LLaMA-3 70B's F1 of 0.71 as the mean per-case F1 (0.708). The harmonic mean of the two means would be 0.726. The TP/FP/FN they print are pooled, so micro P/R/F1 follow from them (LLaMA-3 70B: 0.681 / 0.758 / 0.718), and the thesis tables compare micro against micro on that basis. (Corrected 2026-09-28: this paragraph used to say their F1 was the harmonic mean of the two averages, i.e. our `macro_f1_of_means`.)
