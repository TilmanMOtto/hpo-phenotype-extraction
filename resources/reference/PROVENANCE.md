# resources/reference

`abbreviations.csv` is the expansion table of `hpo_extraction.retrieval.ontology_index.OntologyIndex`.
The dictionary normaliser of PhenoJury (`hpo_extraction.phenojury.pipeline.dictionary_normaliser`
and the normalisation run, `experiments/05_phenojury/normalise/run.py`) loads this index, so the
table is part of the thesis runs that use the dictionary normaliser (thesis section 5.3, appendix C.2).

The table was written by hand, from a specification and from abbreviations observed in the reports.
It holds abbreviations and spellings only, no report text.

## Format

Lines starting with `#` are comments. `read_reference_csv` skips them, and plain `csv.DictReader`
would not: it would take the first comment line as the header and load an empty index without an
error. Columns: `kind,surface,expansion,note`.

* `abbreviation`: the ontology has no form of the string (`C3`, `5MTHF`, `tHcy`, `MCV`).
* `variant`: a spelling, orthography or register variant (`homocystine`/`homocysteine`,
  `fits`/`seizures`, British/American spellings, `plasmatic`/`plasma`).

34 rows. Most expand to an analyte name that HPO does not carry as a label (there is no *cobalamin*
term, there is *Decreased circulating cobalamin concentration*), so the expansion is matched through
the whole index (exact, synonym, fuzzy, definition), not looked up once. A row whose expansion
matches no term is loaded, counted, and has no effect.

## The fuzzy match has a polarity guard

Measured on this ontology, the edit-ratio similarity of `macrocytosis` and `microcytosis` (opposites)
is 0.917, higher than that of the intended spelling repair `homocvsteinaemia`/`homocystinemia`
(0.867). No threshold separates the two cases. `ontology_index.ANTONYM_MORPHEMES` therefore rejects a
fuzzy match that crosses a polarity pair (`macro`/`micro`, `hyper`/`hypo`, `increased`/`decreased`)
at any ratio.

## Changing the table

Every row needs a `note` a reader can check. No test covers this table. After a change, check that
the index loads every row (the command prints the row count, 34 today):

```bash
python -c "from hpo_extraction.retrieval.ontology_index import load_index; print(load_index().n_abbreviation_rows)"
```
