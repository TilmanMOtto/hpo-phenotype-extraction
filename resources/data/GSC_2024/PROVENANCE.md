# GSC_2024, the corpus, and the two document subsets vendored beside it

`Text/` and `Annotations/` are the full 228-document GSC-2024 corpus (FastHPOCR's refinement of
GSC+), one file per PubMed id in each. `Annotations/<id>` lines are
`start:end <TAB> HP:####### <TAB> mention`; `hpo_extraction.evaluation.datasets.gsc.load_gsc_ground_truth` reads
them with no ancestor closure, which is the ground truth every earlier method is scored against.

**Verified 2026-09-28** (thesis review item N03):

- `Annotations/` is byte-identical to Groza's release
  (`github.com/tudorgroza/code-for-papers/gsc-2024/gsc_2024.tar.gz`), 1 823 doc–term pairs.
- It is **not** the original GSC+ (Lobo et al. 2017, `lasigeBioTM/IHP/GSC+.rar`, 1 757 pairs). Only
  1 403 pairs are shared (420 only here, 354 only there), so the thesis must say "GSC-2024" wherever
  it names the ground truth, and cite FastHPOCR (`groza2024`), not `lobo2017`.
- AutoPCR's released `GSC-2024_{dev,test}_gold.tsv` is a strict subset of this ground truth: all 1 552 of
  its pairs are here, and the 271 missing ones are outside phenotypic abnormality (inheritance mode,
  onset, …). On the 206 evaluation documents they carry 1 416 pairs against our 1 415 below $v_0$,
  so their document-level F1 and ours are scored on the same ground truth.

Alongside the corpus this directory vendors **two id lists** and nothing else. Both narrow the
document set. Neither touches the ground truth.

| file | n | what it is |
|---|---|---|
| `eval_206_ids.txt` | 206 | the documents AutoPCR (Tao et al.) evaluate on |
| `dev_22_ids.txt` | 22 | the development documents they hold out |

## Why the split is here at all

AutoPCR's published figures are over **206 of the 228** abstracts, the corpus is split 22
development / 206 evaluation, following PhenoTagger's split. A number computed over all 228 is
therefore not their number, and `comparison`'s literature table could only print their row as "not
comparable". With this split the cohort `gsc_2024_eval_206` exists and their **document-level**
$F_1$ becomes readable against ours on the same documents and the same ground truth.

Their main result is still not comparable and this does not change that: it is mention-level (a hit
needs a matching code **and** overlapping offsets), which no method in `t1_overall.csv` emits.

**Paper**: Tao Y, et al. *AutoPCR: automated phenotype concept recognition by prompting.*
Bioinformatics 2026;42(Supplement 1):btag304.

## Source

| | |
|---|---|
| Repository | https://github.com/yctao7/AutoPCR |
| File | `data/corpus.zip` → `corpus/GSC-2024/GSC-2024_{dev,test}_gold.tsv` |
| Downloaded | 2026-09-17 |
| Format | PubTator-style: blank-line-separated blocks, first line the bare PubMed id |

Regenerate with:

```bash
python experiments/03_setup/extract_gsc_eval_split.py
```

**Only the ids are taken.** Those files also carry the abstract text and Tao et al.'s own
annotations. Pulling those in would change the ground truth as well as the document set, and then
a difference between our row and theirs would have two causes instead of one. The ground truth stays
`Annotations/`.

The script asserts the split is 22/206 and that every id exists in this corpus, and fails rather
than writing a shrunken cohort that still looks like a complete replication. All 228 ids match
and the union of the two lists is the corpus, with no leftovers on either side.

## Recall ceilings

Not every ground truth code exists as a scorable node in `resources/util/hpo.json`, so each cohort imposes
a different hard ceiling on recall for any method predicting from the ontology graph. Regenerate
with a script that is not part of this repository:

| cohort | documents | doc-term pairs | unique terms | recall ceiling |
|---|---|---|---|---|
| `gsc` (full corpus) | 228 | 1823 | 507 | 0.851 |
| `gsc_2024_eval_206` (AutoPCR's frame) | 206 | 1659 | 472 | 0.853 |
| `gsc_raghpo` (RAG-HPO's docs, original ground truth) | 114 | 1322 | 448 | 0.887 |
| `gsc_raghpo_ann` (RAG-HPO's docs, their ground truth) | 114 | 1011 | 415 | 0.948 |

The evaluation split is not an easier corpus than the whole of it: dropping the 22 development
documents moves the ceiling by 0.002. Whatever separates our AutoPCR rows from theirs, it is not
the document filter.

## How the derived cohort is scored

`gsc_2024_eval_206` is a **derived** cohort, like the two `gsc_raghpo*` ones: no driver is ever run
against it. It is the `gsc` run restricted to these 206 ids at scoring time
(`result_tables/discovery.py:DERIVED_COHORTS`, mirrored in `comparison/roster.py:ARTIFACT_COHORT`). Every
driver writes one summary line per report, so restricting afterwards is bit-identical to re-running
inference on the subset.

**An exception is PhenoJury**, and it is the same exception the RAG-HPO cohort has: its
configuration is chosen by nested cross-validation, so a fold's configuration selected with the
other 22 documents in its training split is not an out-of-fold estimate for this cohort. Its ground truth
and folds are rebuilt on the 206 and the protocol is re-run, see
`slurm/comparison_inputs.sbatch`.
