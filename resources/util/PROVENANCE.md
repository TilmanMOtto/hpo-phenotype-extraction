# `hpo.json`, the ontology graph this project traverses

Every HPO code this project predicts, prunes, scores or reports is resolved against **this file**.
It had no provenance record until now, and that absence was essential in a way worth stating
plainly: the recall ceilings quoted in the thesis and below are properties of this file,
not of "HPO", and they cannot be reproduced without it.

## Identity

| | |
|---|---|
| Path | `resources/util/hpo.json` (tracked in git) |
| SHA-256 | `8330ca317e713b63617ea31faa5877be5862d4ca374e03de9202114f99597ef0` |
| **Fingerprint** (first 12 hex) | **`8330ca317e71`** |
| Size | 15 509 708 bytes |
| File mtime | 2026-03-23, when the file was first imported |

`hpo_extraction.ontology.hpo_tree.ontology_fingerprint()` returns the fingerprint. Every record the earlier runs emit
carries it as `ontology_version`.

## What is in it

Computed with `HPOTree` + `hpo_extraction.evaluation.metrics.ontology.OntologyView`, not transcribed:

| | |
|---|---|
| Terms in the file | 19 434 |
| Root (`HPOTree.root`) | `HP:0000118` *Phenotypic abnormality* |
| Phenotypic-abnormality subtree, incl. root | 18 355 |
| **Scorable** (subtree minus `HP:0000001`, `HP:0000118`) | 18 354 |
| `alt_id` → primary entries | 3 826 |
| Layer-1 organ-system branches | 23 |

Schema is PhenoBERT's, not obographs: each key is an HP id mapping to
`{Id, Name, Alt_id, Def, Comment, Synonym, Xref, Is_a, Son, Father, Child}`, where `Father`/`Child`
are **transitive** and `Son`/`Is_a` are direct.

## The release is unknown, and that is the honest statement

The file carries **no `data-version` key, no version IRI, and no metadata of any kind**. It was not
downloaded by any script in this repository. Nothing recoverable from its contents identifies the
upstream HPO release, and term count alone does not fix one.

This is a genuine limitation, and it has a measured consequence: an earlier exploratory run found that **272 of
GSC+'s 1 823 annotated pairs (14.9 %) name terms absent from this graph**, onset modifiers, inheritance
codes and clinical modifiers that sit outside the phenotypic-abnormality subtree. Some of that is
structural (they are not phenotypes), but some is release drift, and `alt_id` remapping against a
known release is the open question that would separate the two.

Two other HPO copies are used, both only by RAG-HPO and neither loaded by `HPOTree`. `hp.obo` is
vendored. `hp.json` is gitignored and placed by hand (third_party/README.md):

| file | release | fixed how |
|---|---|---|
| `third_party/RAG-HPO/hp.obo` | `hp/releases/2026-06-23` | whatever was latest when `build_vector_db.py` ran |
| `third_party/RAG-HPO/hp.json` | `2024-08-13` | **enforced**, `build_vector_db_paper.py:detect_release()` refuses to build on any other release |

## Recall ceilings, quote these before comparing anything

Not every ground truth code exists as a scorable node here, so each ground truth imposes a different hard
ceiling on recall for any method predicting from the graph. Recomputed 2026-08-25 against this file;
all three reproduce the values already published in `resources/data/GSC_RAGHPO/PROVENANCE.md`.

| cohort | documents | doc-term pairs | unique terms | recall ceiling |
|---|---|---|---|---|
| `gsc` (full corpus) | 228 | 1 823 | 507 | **0.851** |
| `gsc_raghpo` (their docs, original ground truth) | 114 | 1 322 | 448 | **0.887** |
| `gsc_raghpo_ann` (their docs, their ground truth) | 114 | 1 011 | 415 | **0.948** |

Computed with a script that is not part of this repository.

## Do not swap this file

Replacing it with a current HPO release would move all three ceilings, and with them every table in the
thesis, silently, because nothing would fail. The
graph is therefore **fixed by content**, and re-deriving against a known release is a deliberate,
versioned migration with a re-scoring pass attached, not a dependency bump.

If it is ever replaced, the fingerprint changes, `ontology_version` changes with it, and records
produced under the two graphs stay distinguishable. That is the whole point of stamping it.
