"""The curated HCY ground truth as a **dataset**, written back into the cohort directory.

:mod:`curated_gold` answers "what is in the ground truth" for one experiment. This module writes that
answer down as something a second person can pick up a year later: a dated directory beside the
files it was built from, holding the annotation table, the two-column ground truth every scorer already
reads, the per-report manifest, and a ``README.md`` that explains all three.

Three rules shape what it writes, and each is a decision somebody could get wrong the other way:

**Nothing is deleted, only flagged.** The annotation table carries *every* candidate, including
the ones the default policy drops for a ``family`` or ``unsure`` qualifier, and the ones it drops
for having no findable trigger word. ``in_gold`` and ``exclude_reason`` say what happened to each,
and one boolean column per qualifier means a reader who disagrees with the policy can rebuild the
ground truth with a ``groupby`` instead of asking for the curation log. A dataset that has already applied
its own opinions is one nobody else can re-use.

**The two-column file is the applied policy.** ``hcy_ground_truth_curated.csv`` is the filtered
set, ``patient_id,hpo_codes``, because that is the shape ``HCYDataset.load_ground_truth`` wants
and the point of shipping it is that no downstream job has to reimplement the filter. Both files
ship, and the README says which is which.

**It is patient data.** It is written into ``hcy_dir`` with the curation log's own modes
(``store.FILE_MODE`` / ``store.DIR_MODE``) so the group that already has the cohort can read it and
nobody else can, and the README says on its first screen that it must not leave the cluster
(``docs/cluster.md``).

The directory is **dated and never overwritten in place**, ``curated_ground_truth_2026-09-08/``, because
the curation log keeps growing and the number in a thesis has to name the version it came from.
Re-running on the same day rewrites that day's directory, which is the behaviour you want while
iterating and the one the manifest's ``generated_at`` disambiguates afterwards.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path

from hpo_extraction.curation import labels as vocab
from hpo_extraction.curation import store

logger = logging.getLogger(__name__)

#: The dated directory this module writes, under ``hcy_dir``.
DIR_PREFIX = "curated_ground_truth_"

ANNOTATIONS_FILE = "hcy_curated_annotations.csv"
GOLD_FILE = "hcy_ground_truth_curated.csv"
REPORTS_FILE = "hcy_curated_reports.csv"
MANIFEST_FILE = "manifest.json"
README_FILE = "README.md"

#: The annotation qualifiers, in vocabulary order. Read from :mod:`hpo_extraction.curation.labels`
#: rather than listed here, so a column cannot drift from the checkbox that wrote it.
QUALIFIERS: tuple[str, ...] = tuple(vocab.label_ids("annotation"))

#: The report-level labels, same rule. The scope is spelled ``patient`` in
#: :data:`hpo_extraction.curation.labels.SCOPES`, one report, one patient, and asking for ``report``
#: silently returns nothing, which is a column set that vanishes, not a failure.
REPORT_LABELS: tuple[str, ...] = tuple(vocab.label_ids("patient"))

#: The report-level difficulty grades, for the README's own description of the column.
DIFFICULTIES: tuple[str, ...] = tuple(v for v, _, _ in vocab.difficulty_for("patient"))

ANNOTATION_FIELDS: list[str] = [
    "patient_id", "hpo_code", "hpo_name",
    "in_gold", "exclude_reason",
    "source", "status",
    "segment_idx", "trigger_word", "segment_text",
    "anchored", "anchor_how",
    "qualifiers", *[f"q_{name}" for name in QUALIFIERS],
    "note", "edited", "original_hpo_code", "key",
]

REPORT_FIELDS: list[str] = [
    "patient_id", "in_cohort", "cohort_reason",
    "n_segments", "n_annotations", "n_gold_terms", "n_dropped", "n_unanchored",
    "n_prior_annotation", "n_daphne", "n_suggestions", "n_adjudicated",
    "is_confirmed", "difficulty", "report_labels", *[f"r_{name}" for name in REPORT_LABELS],
    "n_comments",
]


def dataset_dir(hcy_dir: str, date: str = "") -> Path:
    """``<hcy_dir>/curated_ground_truth_<YYYY-MM-DD>``, the directory :func:`write` fills."""
    return Path(hcy_dir) / f"{DIR_PREFIX}{date or today()}"


def today() -> str:
    """Today's date as ``YYYY-MM-DD``, the suffix of the dated dataset folder."""
    return _dt.date.today().isoformat()


def _flag(values, name: str) -> int:
    return 1 if name in set(values or ()) else 0


def annotation_rows(result) -> list:
    """One row per candidate annotation, the ones in the ground truth and the ones excluded alike."""
    rows = []
    for term in sorted(result.terms, key=lambda t: (t.patient_id, t.hpo_code, t.source, t.key)):
        labels = tuple(term.labels or ())
        row = {
            "patient_id": term.patient_id,
            "hpo_code": term.hpo_code,
            "hpo_name": term.hpo_name,
            "in_gold": int(bool(term.in_gold)),
            "exclude_reason": "" if term.in_gold else term.reason,
            "source": term.source,
            "status": term.status or "unruled",
            "segment_idx": "" if term.segment_idx is None else term.segment_idx,
            "trigger_word": term.trigger_word,
            "segment_text": term.segment_text,
            "anchored": int(bool(term.anchored)),
            "anchor_how": term.how,
            "qualifiers": ";".join(labels),
            "note": term.note,
            "edited": int(bool(term.edited)),
            "original_hpo_code": term.original_hpo_code if term.edited else "",
            "key": term.key,
        }
        for name in QUALIFIERS:
            row[f"q_{name}"] = _flag(labels, name)
        rows.append(row)
    return rows


def report_rows(result) -> list:
    """One row per report the inputs mention, in or out of the cohort."""
    n_terms: dict = {}
    for term in result.terms:
        n_terms[term.patient_id] = n_terms.get(term.patient_id, 0) + 1
    rows = []
    for report in sorted(result.reports, key=lambda r: r.patient_id):
        labels = tuple(report.report_labels or ())
        row = {
            "patient_id": report.patient_id,
            "in_cohort": int(bool(report.in_cohort)),
            "cohort_reason": report.reason,
            "n_segments": report.n_segments,
            "n_annotations": n_terms.get(report.patient_id, 0),
            "n_gold_terms": report.n_gold_terms,
            "n_dropped": report.n_dropped,
            "n_unanchored": report.n_unanchored,
            "n_prior_annotation": report.n_prior_annotation,
            "n_daphne": report.n_daphne,
            "n_suggestions": report.n_suggestions,
            "n_adjudicated": report.n_adjudicated,
            "is_confirmed": int(bool(report.is_confirmed)),
            "difficulty": report.difficulty,
            "report_labels": ";".join(labels),
            "n_comments": report.n_comments,
        }
        for name in REPORT_LABELS:
            row[f"r_{name}"] = _flag(labels, name)
        rows.append(row)
    return rows


def _digest(path: str) -> str:
    """SHA-256 of an input file, so a number in a thesis can name the log that produced it."""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:16]
    except OSError:
        return ""


def counts(result) -> dict:
    """Everything the README states, computed once so the prose and the manifest cannot disagree."""
    terms = result.terms
    by_reason: dict = {}
    for term in terms:
        if not term.in_gold:
            by_reason[term.reason] = by_reason.get(term.reason, 0) + 1
    by_how: dict = {}
    for term in terms:
        if term.anchored:
            by_how[term.how] = by_how.get(term.how, 0) + 1
    by_source: dict = {}
    for term in terms:
        if term.in_gold:
            by_source[term.source] = by_source.get(term.source, 0) + 1
    by_status: dict = {}
    for term in terms:
        by_status[term.status or "unruled"] = by_status.get(term.status or "unruled", 0) + 1
    qualifier_counts = {name: sum(1 for t in terms if name in set(t.labels or ()))
                        for name in QUALIFIERS}
    return {
        **result.summary(),
        "n_reports_total": len(result.reports),
        "n_annotations": len(terms),
        "n_gold_by_source": dict(sorted(by_source.items())),
        "n_dropped_by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "n_anchored_by_how": dict(sorted(by_how.items(), key=lambda kv: -kv[1])),
        "n_by_status": dict(sorted(by_status.items(), key=lambda kv: -kv[1])),
        "n_by_qualifier": qualifier_counts,
    }


def manifest(result, paths: dict, stats: dict, date: str) -> dict:
    """Provenance: what was read, when, by whom, and under which policy."""
    log_path = os.path.join(paths.get("curation_dir", ""), store.EVENTS_FILE)
    policy = result.policy
    return {
        "dataset": f"{DIR_PREFIX}{date}",
        "date": date,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "generated_by": _safe_user(),
        "built_by": "experiments/03_setup/ground_truth",
        "inputs": {
            "segments": paths.get("segments", ""),
            "prior_annotation": paths.get("prior_annotation", ""),
            "confirmed": paths.get("confirmed", ""),
            "curation_log": log_path,
            "curation_log_sha256_16": _digest(log_path),
            "phenobert": paths.get("phenobert", ""),
        },
        "policy": {
            "criteria": list(result.criteria),
            "prior_annotation_fallback": policy.prior_annotation_fallback,
            "require_evidence": policy.require_evidence,
            "include_undecided": policy.include_undecided,
            "include_suggested": policy.include_suggested,
            "exclude_labels": list(policy.exclude_labels),
        },
        "counts": stats,
        "files": {
            "annotations": ANNOTATIONS_FILE,
            "gold": GOLD_FILE,
            "reports": REPORTS_FILE,
            "readme": README_FILE,
        },
    }


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover, a container with no passwd entry
        return "unknown"


def write(hcy_dir: str, result, paths: dict, date: str = "", dest: str = "") -> dict:
    """Write the dated dataset directory. Returns the manifest.

    *dest* overrides the location entirely, used by the tests, and by anyone who wants the dataset
    somewhere other than beside the cohort.
    """
    date = date or today()
    out = Path(dest) if dest else dataset_dir(hcy_dir, date)
    out.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out, store.DIR_MODE)
    except OSError as exc:  # a directory somebody else owns is not this job's problem
        logger.debug("could not chmod %s to %o: %s", out, store.DIR_MODE, exc)

    stats = counts(result)
    store.write_csv_atomic(str(out / ANNOTATIONS_FILE), ANNOTATION_FIELDS,
                           annotation_rows(result))
    store.write_csv_atomic(str(out / REPORTS_FILE), REPORT_FIELDS, report_rows(result))
    store.write_csv_atomic(
        str(out / GOLD_FILE), ["patient_id", "hpo_codes"],
        [{"patient_id": pid, "hpo_codes": ";".join(sorted(result.gold[pid]))}
         for pid in sorted(result.gold)])

    payload = manifest(result, paths, stats, date)
    (out / MANIFEST_FILE).write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                     encoding="utf-8")
    (out / README_FILE).write_text(readme(payload, result), encoding="utf-8")
    for name in (ANNOTATIONS_FILE, REPORTS_FILE, GOLD_FILE, MANIFEST_FILE, README_FILE):
        store.share(str(out / name))

    logger.info("curated dataset → %s (%d annotation row(s), %d gold pair(s) over %d report(s))",
                out, stats["n_annotations"], stats["n_gold_pairs"], stats["n_reports_in_cohort"])
    payload["path"] = str(out)
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# The README
# ──────────────────────────────────────────────────────────────────────────────
def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _counts_table(mapping: dict, first: str, second: str = "n") -> str:
    if not mapping:
        return "_(none)_"
    return _table([first, second], [[k, v] for k, v in mapping.items()])


def readme(payload: dict, result) -> str:
    """The curator-facing description of the dataset. Every number comes from *payload*."""
    stats = payload["counts"]
    policy = payload["policy"]
    inputs = payload["inputs"]

    qualifier_rows = [[f"`{name}`", vocab.display(name),
                       stats["n_by_qualifier"].get(name, 0),
                       "**excluded**" if name in policy["exclude_labels"] else "kept",
                       vocab.help_for(name)]
                      for name in QUALIFIERS]

    report_label_rows = [[f"`{name}`", vocab.display(name), vocab.help_for(name)]
                         for name in REPORT_LABELS]

    return f"""# HCY curated gold — {payload['date']}

> **Patient data.** This directory holds clinical report sentences (`segment_text`) and the
> annotations on them. It stays on the cluster. Do not copy it to a laptop, do not put it in the
> repository, and do not send any of it to a hosted model (see `docs/cluster.md` of the code
> repository).

Built by `experiments/03_setup/ground_truth` on {payload['generated_at']} by
`{payload['generated_by']}`, from the curation pass in `app/hcy_curation_ui`.

## What this is

A ground-truth set for HPO phenotype extraction over the HCY cohort, produced by reading every
report one at a time against three existing annotation files and repairing what was wrong. It
replaces the code-only ground truth: **every term here names the words it came from**, so a miss
can be told apart from a term the report never states.

**{stats['n_gold_pairs']} (report, term) pairs over {stats['n_reports_in_cohort']} reports**, out of
{stats['n_annotations']} candidate annotations considered on
{stats['n_reports_total']} reports the inputs mention.

## The rule

A phenotype is in the gold when **all** of these hold.

1. It comes from `prior_annotation` (`hcy_holistic_ground_truth.csv`), from `daphne`
   (`annotations_confirmed.csv`), or from a suggestion a curator added in the app.
2. It is **anchored**: it sits on a segment of the report, on a trigger word that actually occurs
   in that segment. Checked per phenotype across every row carrying it, so one file recording an
   annotation badly does not discard the file that recorded it well.
3. It is not under an open **deletion proposal**, and no curator has ruled against it
   (`removed`, `rejected`, `needs_work`).
4. It carries none of the excluded qualifiers: {', '.join(f'`{v}`' for v in policy['exclude_labels']) or '_(none)_'}.

An annotation **nobody has ruled on is in**. The curation pass read every report; the keep/remove
verdicts are an adjudication backlog, not evidence about the annotation. Waiting for one would
shrink the gold to whatever went through Approve mode and measure the backlog instead of the data.

`prior_annotation_2` (`hcy_ground_truth_marc2.csv`) is deliberately **not** a source. It is a second annotator's
code list with no trigger words, so it could never satisfy rule 2, and nothing in it was ever put
on a screen for anybody to rule on. It remains useful as a cross-check.

## The files

| File | One row per | Use it for |
|---|---|---|
| `{payload['files']['gold']}` | report | **scoring.** `patient_id,hpo_codes` — semicolon-joined, sorted. The rule above already applied. Drop-in for `hcy_gt_path=` and `ground_truth_path=`. |
| `{payload['files']['annotations']}` | candidate annotation | **everything else.** Every term considered, in the gold or not, with its evidence and its qualifiers. This is the file to re-filter if you disagree with the rule. |
| `{payload['files']['reports']}` | report | cohort membership, per-report counts, and the report-level labels. |
| `{MANIFEST_FILE}` | — | provenance: input paths, the curation log's hash, the policy, the counts. |

`{payload['files']['gold']}` is derivable from `{payload['files']['annotations']}` — it is the
`in_gold = 1` rows, grouped by `patient_id`. Both ship so that no downstream job has to reimplement
the filter, and so that anyone who wants a different one can build it:

```python
import pandas as pd
ann = pd.read_csv("{payload['files']['annotations']}")

# the shipped gold, rebuilt
ann[ann.in_gold == 1].groupby("patient_id").hpo_code.apply(lambda s: ";".join(sorted(set(s))))

# a gold that keeps family-history findings after all
keep = (ann.anchored == 1) & (ann.q_unsure_report == 0) & (ann.q_unsure_annotation == 0) \\
     & (~ann.status.isin(["removed", "rejected", "needs_work", "delete_suggested"]))
ann[keep].groupby("patient_id").hpo_code.apply(lambda s: ";".join(sorted(set(s))))
```

### Reports that are **not** in the cohort

A report with no annotation from any source is absent from `{payload['files']['gold']}` — not
present with an empty cell. An empty cell means *this report has no phenotypes*, which is a claim
the curation pass can make; a report nobody annotated is a different thing and would hand every
method free precision. `{payload['files']['reports']}` lists them with `in_cohort = 0` and the
reason. Cohort criteria in force: {', '.join(f'`{c}`' for c in policy['criteria'])}.

## Columns — `{payload['files']['annotations']}`

| Column | Meaning |
|---|---|
| `patient_id` | the report |
| `hpo_code`, `hpo_name` | the phenotype, on the pinned HPO release (`resources/util/hpo.json`) |
| `in_gold` | `1` if it passed the rule above |
| `exclude_reason` | why not, when `in_gold = 0`. `no_evidence` = unanchored; `label:…` = an excluded qualifier; `status:…` = a curator's verdict |
| `source` | `prior_annotation`, `daphne`, or `new` (added during curation) |
| `status` | the curator's verdict: `unruled`, `approved`, `kept`, `suggested`, `removed`, `rejected`, `needs_work`, `delete_suggested` |
| `segment_idx` | index into `segmented_reports.csv` for this patient — the sentence the term came from |
| `trigger_word` | the words in that sentence that earned the term, **verbatim** (original whitespace and capitalisation) |
| `segment_text` | that sentence, as the curator read it |
| `anchored` | `1` when `segment_idx` and `trigger_word` are both set and the trigger really occurs there |
| `anchor_how` | how it was placed — see the table below |
| `qualifiers` | semicolon-joined qualifier ids |
| `q_*` | one `0`/`1` column per qualifier, for joining and grouping |
| `note` | free text a curator left on this annotation |
| `edited` | `1` if the curation log rewrote the code the file originally carried |
| `original_hpo_code` | what the file said, when `edited = 1` |
| `key` | the annotation's stable address in the curation log (`source|patient|code[|slot]`), for tracing a row back to `curation_events.jsonl` |

### `anchor_how` — how much the placement is worth

| Value | Meaning |
|---|---|
| `curated` | a **person** placed it, in the Anchor tab or a row editor. The strongest. |
| `segment` | the file named the segment, and the trigger really is in it |
| `offset` | the file's character offset, mapped through an alignment of the segments against the file's own copy of the report |
| `context` | the file's sentence context matched to a segment, then the trigger found inside it |
| `lexical` | nothing but the trigger word, scanned across the whole report. **This is the tool's guess, not the annotator's claim** — a term whose trigger occurs twice may be sitting on the wrong occurrence. Discount accordingly. |

{_counts_table(stats['n_anchored_by_how'], 'anchor_how')}

## The qualifiers

Ticked by a curator on the annotation itself, while deciding the term. They are a property of the
annotation, not of the report, and they are the reason this dataset can answer *why* a model missed
something rather than only *how often*.

{_table(['id', 'name', 'n', 'in this gold', 'when to use it'], qualifier_rows)}

`family` is the interesting exclusion, and the one worth re-examining before using this gold: the
finding is real and the text does support it, it just belongs to a relative. A system that extracts
it is doing something defensible, so counting it as a false positive measures the annotation
convention rather than the system. The two `unsure` labels are a different claim — that a model
disagreeing there is not demonstrably wrong — which is the only defensible reason to take a term out
of a denominator rather than count it as a miss.

**Nothing is deleted.** Every excluded annotation is in `{payload['files']['annotations']}` with
`in_gold = 0`, so any of these decisions can be reversed downstream without going back to the log.

## Report-level labels — `{payload['files']['reports']}`

One row per report, a `difficulty` grade ({' / '.join(f'`{d}`' for d in DIFFICULTIES) or '—'}) and a
`0`/`1` column per label, prefixed `r_`. Join on `patient_id` to slice recall by what kind of report
it is — "does this method miss the reports that are heavy on family history?" is a `groupby`, not a
new experiment.

{_table(['id', 'name', 'when to use it'], report_label_rows)}

## Where the numbers came from

**Gold terms by source**

{_counts_table(stats['n_gold_by_source'], 'source')}

**Every candidate, by status**

{_counts_table(stats['n_by_status'], 'status')}

**Excluded candidates, by reason**

{_counts_table(stats['n_dropped_by_reason'], 'reason')}

{stats['n_terms_unanchored']} of {stats['n_annotations']} candidate annotations could not be
anchored: their trigger word occurs nowhere in the report, or the report has no segmentation. Those
are the rows the Anchor tab could not close, and they are the honest cost of requiring evidence.

## Inputs

| | |
|---|---|
| segmentation | `{inputs['segments']}` |
| `prior_annotation` | `{inputs['prior_annotation']}` |
| `daphne` | `{inputs['confirmed']}` |
| curation log | `{inputs['curation_log']}` (sha256 `{inputs['curation_log_sha256_16']}…`) |

The curation log is append-only and keeps growing, which is why this directory is dated: a number
in a thesis has to name the version of the gold it came from. Rebuild with

```bash
python experiments/03_setup/ground_truth/run.py \\
  stages=[dataset] hcy_dir=<hcy_dir> output_dir=<output_dir>
```

## Caveats

* **Unruled terms are in.** {stats['n_by_status'].get('unruled', 0)} of {stats['n_annotations']}
  candidates carry no keep/remove verdict. They were read during the pass, but nobody re-asserted
  them one by one. `variant=approved_only` in exp13_18 prices what restricting to verdicts costs.
* **`lexical` anchors are guesses.** See the table above.
* **The HPO release is pinned by content hash** (`resources/util/PROVENANCE.md`). Codes here
  resolve against that file. Do not re-map them onto a current release without re-reading
  `PROVENANCE.md` first — every published recall ceiling is a property of the pinned ontology.
* **A term can appear twice for one report** in `{payload['files']['annotations']}` when two
  sources annotated it, or one source annotated it on two different triggers. That is one phenotype
  with two pieces of evidence; `{payload['files']['gold']}` de-duplicates.
"""
