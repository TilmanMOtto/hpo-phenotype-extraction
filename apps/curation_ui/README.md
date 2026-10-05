# Curation UI

Review of the HCY annotations, one report at a time. A curator confirms or removes each existing
annotation, adds terms that are in the report but were never annotated, and records where in the
report each term comes from (segment and trigger word). Every click is appended to an event log
under `<hcy.dir>/curation/`. The curated ground truth of the thesis is built from that log by
`experiments/03_setup/ground_truth/` (thesis section 3.2, appendix A.3).

## Run

```bash
python apps/curation_ui/app.py --check        # paths readable, curation log writable, then exit
python apps/curation_ui/app.py --port 8055    # on LeoMed, then open it through an SSH tunnel
python apps/curation_ui/app.py --selftest     # synthetic reports, no data needed
```

Options: `--hcy-dir` (segments, annotation files and `curation/`), `--phenobert-dir` (the PhenoBERT
baseline run, shown as reference, blank to leave it out), `--curation-dir`, `--author` (recorded on
every event, default the login name). Several curators can work at once on different ports.

## Screens

- **Curate**, with two modes.
  - *Approve*: every annotation the files carry, one card per term, with **Keep**, **Remove** and
    **Needs work**. Pending suggestions get **Approve**, **Reject** and **Needs work**.
  - *Edit*: select a segment or highlight words in the report, then propose a term. Each
    annotation can carry qualifiers (negated, family, laboratory value, implicit, unsure).
- **Locate evidence**: the annotations whose trigger word could not be found in the report. A term
  enters the curated ground truth only once it has a segment and a trigger word.
- **Overview**: progress over the cohort, and the button that writes the curated ground truth.

A term is kept when the report says that this patient has this phenotype. It is removed when the
finding belongs to a relative, is negated, is only queried or ruled out, is a procedure or a drug, or
when the code does not match the text.

## Annotation sources

| Source | Meaning | Verdict |
|---|---|---|
| `daphne` | confirmed pass over the annotations (`annotations_confirmed.csv`) | yes |
| `prior_annotation` | the earlier annotation round | yes, for terms `daphne` does not carry |
| `prior_annotation_2` | a second annotator's code list, without evidence | reference only |
| PhenoBERT | the baseline's detections | reference only |

The stored log keeps the source names it was written with. `hpo_extraction.curation.labels` maps
them to the names above when the log is read.
