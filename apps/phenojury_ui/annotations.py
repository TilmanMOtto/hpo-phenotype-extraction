"""Manual HPO annotation of what each SLM wrote, and the comparison against PhenoBERT.

The recall autopsy can say an annotated term was ``not_linked``, but that fate covers two entirely
different failures and cannot separate them from anything on disk:

* the model never wrote the finding, so there was nothing to ground. Or
* the model wrote it plainly and **PhenoBERT failed to link it**.

Only a person reading the generation can tell those apart, and the difference decides what to fix, a better prompt, or a better grounder. This module is the record of that reading.

The unit is **one ``(model, report, sentence)`` reply**, which is the text
``phenobert_runner.build_phenobert_input`` handed PhenoBERT. Annotating at that granularity holds
the SLM fixed and varies only the linker, so the comparison is a measurement of PhenoBERT rather
than of the ensemble. Annotating whole reports instead would let a term found in the wrong sentence
score as agreement, which is the thing most worth catching.

Storage is an **append-only event log plus derived CSVs**, and the primitives are imported from
``app/hcy_curation_ui/store``, not copied, the same move ``app/prompt_lab_ui/store`` makes.
That app arrived at this shape the hard way and its reasons all apply here: an append is crash-safe,
the log is the authority so a schema change is a re-fold, not a migration, and the file modes
matter because this lands in a shared cluster directory where the second person to click would
otherwise hit a ``PermissionError`` after forming a judgement.

Two rules carried over:

``annotate`` **replaces** the whole term set for its reply, never merges. The UI sends what the
picker now holds, and merging would make removing a wrong term impossible.

``skip`` is a **first-class state**, distinct from never-visited. "This reply asserts no phenotype"
is a finding, not an absence of one, it is the denominator of PhenoBERT's precision, and a store
that could not express it would report a false-positive rate over an unknown sample.

The ``run_id`` is part of every key, so one directory holds the Free Listing generation run and each earlier prompt beside
each other: the same reader, the same reports, two prompts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid

from hpo_extraction.curation.store import (  # noqa: F401 - re-exported for this app's callers
    DIR_MODE,
    FILE_MODE,
    _flock,
    flatten,
    now_iso,
    read_events,
    share,
    write_csv_atomic,
    writability_problem,
)

logger = logging.getLogger(__name__)

#: Where a run's annotations live when no directory was configured, beside the generations they
#: were made against, so an annotation pass cannot drift away from the artifacts it describes.
SUBDIR = "slm_annotations"

EVENTS_FILE = "slm_annotation_events.jsonl"
TERMS_FILE = "slm_annotations.csv"
AGREEMENT_FILE = "slm_annotation_agreement.csv"

#: Actions the fold understands. Anything else is counted and skipped, not crashing a
#: session: a log written by a newer version of this app must still open in an older one.
ACTIONS = frozenset({"annotate", "skip", "unskip", "note"})

#: One row per (reply, term) a person recorded.
TERMS_FIELDS = ["run_id", "cohort", "prompt_key", "model", "report_id", "sentence_number",
                "hpo_code", "author", "updated_at", "note"]

#: One row per annotated reply, carrying both sides of the comparison and its verdict counts.
AGREEMENT_FIELDS = ["run_id", "cohort", "prompt_key", "model", "report_id", "sentence_number",
                    "status", "n_manual", "n_phenobert", "n_agreed", "n_pb_missed",
                    "n_pb_spurious", "manual", "phenobert", "pb_missed", "pb_spurious",
                    "pb_negated", "author", "updated_at", "note"]


def key_of(run_id: str, model: str, report_id: str, sentence_number) -> str:
    """The stable address of one reply.

    ``run_id`` leads because the same report and sentence exist in every prompt's run, and an
    annotation of what *this* prompt made the model write says nothing about what another did.
    """
    return f"{run_id}|{model}|{report_id}|{sentence_number}"


def default_dir(run_dir: str, configured: str = "") -> str:
    """Where to record annotations for a run: the configured directory, or beside the run."""
    if configured:
        return configured
    return os.path.join(run_dir, SUBDIR)


# ──────────────────────────────────────────────────────────────────────────────
# The log
# ──────────────────────────────────────────────────────────────────────────────
class AnnotationLog:
    """The append-only log for one annotations directory, and the state folded from it.

    One instance per process, but several readers may share a directory over their own tunnels, so
    the in-memory list is a cache of a file others also append to and is re-read whenever the file
    has changed. Without that, each process folds only its own clicks and the derived CSVs it then
    rewrites would silently revert somebody else's work.
    """

    def __init__(self, directory: str, author: str = "unknown"):
        self.dir = directory
        self.author = author or "unknown"
        # Neither step may raise: this runs while the app is being constructed, so there is nowhere
        # to report a failure in words yet. ``writability_problem`` says it at startup instead.
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as exc:
            logger.warning("could not create %s: %s", self.dir, exc)
        else:
            share(self.dir, DIR_MODE)
            for name in (EVENTS_FILE, TERMS_FILE, AGREEMENT_FILE):
                path = os.path.join(self.dir, name)
                if os.path.exists(path):
                    share(path)
        self._lock = threading.Lock()
        self._stat: tuple[int, int] | None = None
        self.events: list[dict] = []
        self._reload()

    @property
    def path(self) -> str:
        """Path of the annotation event log."""
        return os.path.join(self.dir, EVENTS_FILE)

    @property
    def terms_path(self) -> str:
        """Path of the derived per-term annotation table."""
        return os.path.join(self.dir, TERMS_FILE)

    @property
    def agreement_path(self) -> str:
        """Path of the derived agreement table against PhenoBERT."""
        return os.path.join(self.dir, AGREEMENT_FILE)

    # ── keeping up with other readers ────────────────────────────────────────
    def _stat_log(self) -> tuple[int, int] | None:
        try:
            info = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (info.st_size, info.st_mtime_ns)

    def _reload(self) -> None:
        before = self._stat_log()
        self.events = read_events(self.path)
        self._stat = before

    def refresh(self) -> bool:
        """Pick up events another process appended. ``True`` when the log had grown."""
        with self._lock:
            current = self._stat_log()
            if current == self._stat:
                return False
            self._reload()
            return True

    # ── writing ──────────────────────────────────────────────────────────────
    def append(self, action: str, **fields) -> dict:
        """Write one event and return it. The only mutating entry point in this module."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        event = {
            "event_id": uuid.uuid4().hex,
            "ts": now_iso(),
            "author": self.author,
            "action": action,
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            # Take in whatever anyone else wrote first, so ``events`` stays a true prefix of the
            # file and the commit that follows sees their work as well as ours.
            current = self._stat_log()
            if current != self._stat:
                self._reload()
            fresh = not os.path.exists(self.path)
            with open(self.path, "a", encoding="utf-8") as handle:
                _flock(handle)
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if fresh:
                share(self.path)
            self.events.append(event)
            self._stat = self._stat_log()
        return event

    def fold(self) -> dict:
        """The current state, see :func:`fold_events`. Picks up other readers first."""
        self.refresh()
        return fold_events(self.events)

    def commit(self, rows: list[dict], agreement_rows: list[dict]) -> None:
        """Rewrite both derived CSVs from the folded state. Atomic, after every mutation.

        Full rewrites, not appends: both files are small, and neither is ever a stale view
        of the log, a property worth more here than the writes it would save.
        """
        write_csv_atomic(self.terms_path, TERMS_FIELDS, rows)
        write_csv_atomic(self.agreement_path, AGREEMENT_FIELDS, agreement_rows)


def fold_events(events) -> dict:
    """Fold the log into ``{"replies": {key: row}, "n_skipped": int}``.

    Last event addressing a key wins, so re-running a prefix yields the state as of that moment, which is what makes the log auditable, not merely appended to.
    """
    replies: dict[str, dict] = {}
    n_skipped = 0

    for event in events:
        action = event.get("action")
        if action not in ACTIONS:
            n_skipped += 1
            continue
        key = event.get("key")
        if not key:
            n_skipped += 1
            continue

        row = replies.get(key)
        if row is None:
            row = {
                "key": key,
                "run_id": event.get("run_id", ""),
                "cohort": event.get("cohort", ""),
                "prompt_key": event.get("prompt_key", ""),
                "model": event.get("model", ""),
                "report_id": event.get("report_id", ""),
                "sentence_number": event.get("sentence_number"),
                "hpo_codes": [],
                "status": "",
                "note": "",
                "author": "",
                "updated_at": "",
            }
            replies[key] = row

        if action == "annotate":
            # Replace, never merge, the picker sends its whole contents and a merge would make
            # removing a term impossible.
            row["hpo_codes"] = [str(c) for c in (event.get("hpo_codes") or ()) if c]
            row["status"] = "annotated"
        elif action == "skip":
            # A reading, not an absence of one: "nothing here" is the denominator of PhenoBERT's
            # precision. The terms are cleared because a skipped reply asserts no phenotype.
            row["hpo_codes"] = []
            row["status"] = "skipped"
        elif action == "unskip":
            row["status"] = "annotated" if row["hpo_codes"] else ""
        if event.get("note") is not None and action in ("note", "annotate", "skip"):
            row["note"] = event.get("note", "")
        row["author"] = event.get("author", row["author"])
        row["updated_at"] = event.get("ts", row["updated_at"])

    return {"replies": replies, "n_skipped": n_skipped}


def visited(folded: dict) -> dict:
    """``{key: row}`` for replies somebody has actually ruled on, annotated or skipped."""
    return {key: row for key, row in folded["replies"].items()
            if row["status"] in ("annotated", "skipped")}


# ──────────────────────────────────────────────────────────────────────────────
# The comparison
# ──────────────────────────────────────────────────────────────────────────────
def phenobert_terms(bundle: dict, model: str, report_id: str, sentence_number) -> dict:
    """What PhenoBERT extracted from this reply: ``{"positive": set, "negated": set}``.

    The negated detections are kept apart, not folded into either side. A term PhenoBERT
    found and marked absent is a *decision* it made, not a miss, counting it as a disagreement
    would blame the linker for reading the sentence correctly.
    """
    rows = bundle["pb_index"]["by_sentence"].get((model, str(report_id), int(sentence_number)), ())
    positive = {r["hpo_id"] for r in rows if not r["negated"]}
    negated = {r["hpo_id"] for r in rows if r["negated"]}
    return {"positive": positive, "negated": negated}


def compare_reply(bundle: dict, row: dict) -> dict:
    """One annotated reply, both sides and the three verdict sets.

    ``pb_missed`` is what the reader saw and PhenoBERT did not; ``pb_spurious`` the reverse. Each
    disagreeing term is graded with :func:`relations.classify` against the *other* side, so
    "PhenoBERT returned the parent of the right term" is separated from "PhenoBERT returned
    something unrelated", a distinction any single rate hides, and the reason for doing this by
    hand at all.
    """
    from . import relations

    manual = {str(c) for c in row.get("hpo_codes") or ()}
    pb = phenobert_terms(bundle, row["model"], row["report_id"], row["sentence_number"])
    tree = bundle.get("tree")

    missed = manual - pb["positive"]
    spurious = pb["positive"] - manual
    return {
        **{field: row.get(field, "") for field in
           ("key", "run_id", "cohort", "prompt_key", "model", "report_id", "sentence_number",
            "status", "author", "updated_at", "note")},
        "manual": sorted(manual),
        "phenobert": sorted(pb["positive"]),
        "pb_negated": sorted(pb["negated"]),
        "pb_missed": sorted(missed),
        "pb_spurious": sorted(spurious),
        "n_manual": len(manual),
        "n_phenobert": len(pb["positive"]),
        "n_agreed": len(manual & pb["positive"]),
        "n_pb_missed": len(missed),
        "n_pb_spurious": len(spurious),
        # How wrong each disagreement is, against the opposite side as the reference set.
        "missed_relations": ({code: relations.classify(tree, code, pb["positive"])
                              for code in sorted(missed)} if tree is not None else {}),
        "spurious_relations": ({code: relations.classify(tree, code, manual)
                                for code in sorted(spurious)} if tree is not None else {}),
    }


def compare(bundle: dict, folded: dict, report_ids=None) -> dict:
    """PhenoBERT-as-linker, measured against the replies somebody has read.

    Scoped to this bundle's run: a directory may hold several runs' annotations, and scoring one
    run's generations against another's reading would be meaningless. Restricted further to
    *report_ids* when a subset filter is active, so the panel agrees with the banner above it.

    Precision and recall are **PhenoBERT's**, with the manual reading as the reference. They are
    reported beside the number of replies they were computed over, because that count is the whole
    note: this is a measurement over what has been annotated so far, not over the cohort.
    """
    wanted = None if report_ids is None else {str(r) for r in report_ids}
    rows = []
    for row in visited(folded).values():
        if row["run_id"] != bundle["run_id"]:
            continue
        if wanted is not None and str(row["report_id"]) not in wanted:
            continue
        rows.append(compare_reply(bundle, row))

    per_model: dict[str, dict] = {}
    per_report: dict[str, dict] = {}
    totals = _empty_tally()
    relation_tally: dict[str, dict[str, int]] = {"pb_missed": {}, "pb_spurious": {}}

    for row in rows:
        _add(totals, row)
        _add(per_model.setdefault(row["model"], _empty_tally()), row)
        _add(per_report.setdefault(row["report_id"], _empty_tally()), row)
        for side, field in (("pb_missed", "missed_relations"),
                            ("pb_spurious", "spurious_relations")):
            for relation in row.get(field, {}).values():
                relation_tally[side][relation] = relation_tally[side].get(relation, 0) + 1

    return {
        "rows": sorted(rows, key=lambda r: (r["report_id"], r["model"],
                                            int(r["sentence_number"] or 0))),
        "totals": _score(totals),
        "per_model": {model: _score(t) for model, t in per_model.items()},
        "per_report": {report: _score(t) for report, t in per_report.items()},
        "relations": relation_tally,
        "n_replies": len(rows),
        "n_skipped": sum(1 for r in rows if r["status"] == "skipped"),
    }


def _empty_tally() -> dict:
    return {"n_replies": 0, "n_skipped": 0, "n_manual": 0, "n_phenobert": 0,
            "n_agreed": 0, "n_pb_missed": 0, "n_pb_spurious": 0}


def _add(tally: dict, row: dict) -> None:
    tally["n_replies"] += 1
    tally["n_skipped"] += row["status"] == "skipped"
    for field in ("n_manual", "n_phenobert", "n_agreed", "n_pb_missed", "n_pb_spurious"):
        tally[field] += row[field]


def _score(tally: dict) -> dict:
    """PhenoBERT's precision/recall against the manual reading, or ``None`` where undefined.

    ``None``, not 0.0 for an empty denominator: a set of replies in which nobody read any
    phenotype gives PhenoBERT nothing to recall, which is not the same as it recalling none.
    """
    precision = (tally["n_agreed"] / tally["n_phenobert"]) if tally["n_phenobert"] else None
    recall = (tally["n_agreed"] / tally["n_manual"]) if tally["n_manual"] else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None if precision is None or recall is None else 0.0
    return {**tally, "precision": precision, "recall": recall, "f1": f1}


def term_rows(folded: dict) -> list[dict]:
    """The flat ``slm_annotations.csv`` rows, one per (reply, term), every run in the directory."""
    rows = []
    for row in sorted(visited(folded).values(),
                      key=lambda r: (r["run_id"], r["report_id"], r["model"],
                                     int(r["sentence_number"] or 0))):
        for code in row["hpo_codes"] or [""]:
            rows.append({
                "run_id": row["run_id"], "cohort": row["cohort"],
                "prompt_key": row["prompt_key"], "model": row["model"],
                "report_id": row["report_id"], "sentence_number": row["sentence_number"],
                "hpo_code": code, "author": row["author"],
                "updated_at": row["updated_at"], "note": row["note"],
            })
    return rows


def agreement_rows(comparison: dict) -> list[dict]:
    """The ``slm_annotation_agreement.csv`` rows, straight off :func:`compare`."""
    return [{**row, "manual": row["manual"], "phenobert": row["phenobert"],
             "pb_missed": row["pb_missed"], "pb_spurious": row["pb_spurious"],
             "pb_negated": row["pb_negated"]}
            for row in comparison["rows"]]
