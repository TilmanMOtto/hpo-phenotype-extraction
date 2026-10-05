# PhenoJury UI

Inspection of the Free Listing generation run (`phenojury_generation_free_listing/<cohort>/`): what
each of the eight jurors wrote, how a k-of-N vote turns that into terms, which jurors made each
decision, where each missed annotated term was lost (no juror wrote anything, PhenoBERT did not
link it, it was negated, or too few jurors named it),
and how the jury compares with PhenoBERT run on the raw report. The *Annotate* tab records, by hand,
the phenotypes in each juror reply, to separate generation errors from normalisation errors.

Every number is recomputed from the run's files, and the scorecard first checks the recomputation
against the files the run wrote.

## Run

```bash
python apps/phenojury_ui/app.py --port 8058      # on LeoMed, through an SSH tunnel
python apps/phenojury_ui/app.py --selftest       # verification gates and every view, then exit
python apps/phenojury_ui/drivers.py              # Markdown report: recall ceiling and drivers per juror
```

Options: `--output-base` (folder holding `phenojury_generation_free_listing/`), `--pb-base` (the
PhenoBERT baseline), `--frame` (the 20-report HCY draw), `--annotations-dir` (where manual
annotations are written, default beside each run), `--cache-dir`.

The unit tests (`tests/unit/test_phenojury_ui.py`) build a synthetic run with
`tests/fixtures/exp13_output.py`. The app starts on that run and lists its cohorts.
