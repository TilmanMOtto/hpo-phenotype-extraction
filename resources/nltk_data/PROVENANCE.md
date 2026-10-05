# `resources/nltk_data/`

NLTK's `stopwords` corpus, vendored **unmodified**, so that a checkout of this repository runs
without a per-user `~/nltk_data`.

| | |
|---|---|
| Corpus | `stopwords` (all languages, as NLTK ships it) |
| Source | `nltk.download("stopwords")` |
| Size | 180 KB |
| Consumed by | `src/hpo_extraction/ontology/hpo_items.py` → `hpo_extraction.ontology.nltk_data.ensure_nltk_path()` |

## Why it is in the repository at all

`hpo_extraction.ontology.hpo_items` evaluates `stopwords.words("english")` in a **class body**
(`PhraseItem.StopWords`). That runs when the module is *imported*, not when a phrase is first
parsed, so a machine that cannot reach the corpus does not degrade, it dies, with a `LookupError`
that names NLTK and says nothing about this project.

NLTK searches `~/nltk_data` first, and `nltk.download` writes there. That is per user, and
the HCY curation UI is run by clinical curators who have access to the group's work directory and
none to anybody's home directory. Before this was vendored, they could not start the app at all.

## Why only `stopwords`

It is the only corpus the curation UI touches. That is measured, not assumed: its selftest passes
against a search path containing this directory and nothing else
(`tests/unit/test_curation_ui.py::TestNltkData`).

`wordnet`, `punkt` and `averaged_perceptron_tagger` come to ~48 MB, belong to the segmentation and
AutoPCR baseline rather than to any of the apps, and are installed with `nltk.download`
(third_party/README.md).

## Do not edit the files

Vendoring a *modified* corpus would make this directory something other than what its name claims,
and the claim above, "this is NLTK's stopwords corpus", is the whole of its provenance. If a
different corpus is needed, add it whole, alongside, and extend `nltk_data.BUNDLED_CORPORA`.

The repo copy is **prepended** to `nltk.data.path`, never substituted for it: a machine that has run
`nltk.download` keeps everything it has, and `NLTK_DATA` still works. It is simply no longer
required.
