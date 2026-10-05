# Input and output of the two applications

Both applications (`hpo-treephenorag`, `hpo-phenojury`, and the Python classes
`hpo_extraction.treephenorag.TreePhenoRAG` and `hpo_extraction.phenojury.PhenoJury`) read plain text
and write the same JSON Lines format. Abbreviations: HPO (Human Phenotype Ontology), LSE
(log-sum-exp pooling).

## Input

- One report per `.txt` file. `--input` names a file or a folder (every `*.txt` in it, sorted by name).
- The report identifier is the file name without `.txt`.
- `--encoding` sets the text encoding (default `utf-8`, the HCY reports are `latin1`).
- The text is split into segments (sentences) with the Stanza clinical tokenizer when `stanza_dir`
  is set in the app config, and with a rule-based sentence splitter otherwise. The thesis results
  used Stanza, and the two splitters can give different segment numbers.

From Python, `extract(text, report_id)` takes one report as a string, and `extract_many({report_id:
text})` takes several.

## Output

One JSON object per line, one line per report (shortened):

```json
{"report_id": "syn_report_1",
 "terms": [
   {"hpo_id": "HP:0001250", "label": "Seizure", "score": 0.9975,
    "evidence": [{"segment_index": 1, "text": "He has had recurrent seizures since the age of two.", "score": 0.9975}]}
 ],
 "n_segments": 4,
 "n_model_calls": 1041,
 "method": "treephenorag",
 "settings": {"segments_per_term": 3, "expansion_pooling": "P4", "...": "..."}}
```

| Field | Meaning |
|---|---|
| `report_id` | identifier of the report |
| `terms` | the predicted HPO terms, highest `score` first. Empty when nothing is predicted. |
| `terms[].hpo_id`, `terms[].label` | identifier and label in `resources/util/hpo.json` |
| `terms[].score` | between 0 and 1, see below |
| `terms[].evidence` | the segments that support the term, most supportive first |
| `evidence[].segment_index` | position of the segment in the report, from 0 |
| `evidence[].text` | the segment |
| `evidence[].score` | support of this segment, see below |
| `n_segments` | number of segments of the report |
| `n_model_calls` | language-model calls spent on the report: verifier calls (TreePhenoRAG) or juror generations (PhenoJury) |
| `method` | `treephenorag` or `phenojury` |
| `settings` | the settings the prediction was made with |

### Scores of TreePhenoRAG

- `terms[].score` is the acceptance score of the term: the LSE pooling (`lse_beta1`) of the
  verifier's segment scores, between 0 and 1. A term is predicted when its acceptance score is at
  least the acceptance threshold (default 0.99) and the traversal reached it.
- `evidence[].score` is the verifier's probability of *Yes* for that segment and term,
  between 0 and 1. It is the logistic function of the margin between the *Yes* and *No* logits. The
  evidence lists the `segments_per_term` segments retrieved for the term (default 10).
- A node is expanded (its children are visited) when its expansion score, the pooling P4 of its
  segment scores, is at least the expansion threshold (default 5.58e-5).

### Scores of PhenoJury

- `terms[].score` is the share of jurors that support the term: the number of supporting jurors
  divided by the jury size, between 0 and 1. A term is predicted when at least `k` jurors support it
  within one vote scope (`unit`: the report, a window of segments, or one segment).
- `evidence[].score` is the number of jurors that named the term in that segment, between 1 and the
  jury size.

## Example

`examples/stand_in_demo.py` writes `examples/treephenorag_stand_in.jsonl` and
`examples/phenojury_stand_in.jsonl` from the two reports in `examples/synthetic_reports/`, with
stand-in models.
