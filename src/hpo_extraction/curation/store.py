"""Persistence: an append-only event log, and the two files derived from it.

Everything a curator does is one appended JSON line in ``curation_events.jsonl``. Nothing is ever
edited in place and nothing is ever deleted, which buys three things at once:

* **crash safety**, an append is a single short write. The most a crash can cost is the action in
  flight. A UI that rewrites a growing state file on every click can lose the whole session;
* **provenance**, "who removed HP:0001250 from SYN004, when, and did anyone put it back" is a
  ``grep``, not an archaeology project. That question will be asked about this ground-truth set;
* **no lost work on a schema change**, the log is the authority and the derived files are
  disposable, so a new column is a re-fold, not a migration.

The current state is the log **folded in order**: the last event addressing a key wins. Two kinds
of key exist, and the distinction is the whole data model:

``patient_id``  a ``label`` event with no ``target_key`` addresses the *report*, the labels and
                the difficulty grade that describe the whole document. A ``label`` event *with* a
                ``target_key`` is retired: annotation qualifiers now ride on ``suggest`` and
                ``edit``. The fold still honours one, so a log written before the change re-runs.
``target_key``  addresses something that already existed, ``prior_annotation|SYN004|HP:0001250``,
                ``daphne|…``, ``prior_annotation_2|…``, ``phenobert|SYN004|HP:0001250|312``. The trailing slot
                is there because a source can carry one code twice for one report, and a verdict on
                one must not settle the other. Verdicts, evidence locations and edits land here.
``event_id``    addresses a suggestion made in this app. The ``suggest`` event *is* the row. Later
                ``approve``/``reject``/``withdraw`` events name it in ``target_key``.

Derived, rewritten atomically (``tmp`` + ``os.replace``) after every mutation:

``curation_current.csv``            the flat state, one row per decided or proposed annotation
``curation_comments.csv``           free-text commentary on a report, one row per comment
``curation_labels.csv``             the **annotation**-level qualifiers, long form
``curation_report_labels.csv``      the **report**-level difficulty labels, one row per report, wide
``hcy_ground_truth_curated.csv``    on request only: the two-column ``patient_id,hpo_codes`` shape,
                                    so the curated ground truth drops into any cluster script's
                                    ``ground_truth_path=`` with no adapter

The two label files are separate and shaped differently because they are joined differently. An
annotation qualifier keys on ``(patient_id, hpo_code)`` and there are many per report, so long form
and a ``groupby`` is the natural read. A report label keys on ``patient_id`` alone, there is one
row per report, and the vocabulary is small and fixed, so one wide row with a column per label
joins to anything in a single ``merge`` and reads in a spreadsheet without pivoting.

Both are written single-line-per-record with newlines flattened out of every field, the convention
``app/annotation_ui`` arrived at the hard way: a CSV that survives being opened and re-saved by a
spreadsheet.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone

from hpo_extraction.curation.labels import translate_stored_labels

logger = logging.getLogger(__name__)

#: Mode for everything this module creates. Group read/write, and **nothing for others**.
#:
#: Several curators share one ``curation_dir``, each under their own account, and until this was
#: set they could not: the log was created 0644 by whoever ran first, so the second curator's very
#: first click died with a ``PermissionError``, after they had read a report and formed a verdict,
#: which is the worst possible moment. The four derived CSVs were worse still at 0600
#: (``NamedTemporaryFile``'s default, preserved through ``os.replace``): unreadable even to the
#: person who was going to open them in a spreadsheet.
#:
#: The last digit is 0 on purpose and is not an oversight to tidy up later. This is a directory of
#: patient data on a shared cluster filesystem. It is opened up to the *group* that already has the
#: cohort, and to nobody else.
FILE_MODE = 0o660

#: Mode for ``curation/`` itself. Group rwx, plus **setgid** (the ``2``), so a file created in it
#: takes over the directory's group rather than the creator's primary group, without which the
#: second curator's files land in a group the first cannot read, and ``FILE_MODE`` grants access to
#: nobody in particular.
DIR_MODE = 0o2770

EVENTS_FILE = "curation_events.jsonl"
CURRENT_FILE = "curation_current.csv"
COMMENTS_FILE = "curation_comments.csv"
LABELS_FILE = "curation_labels.csv"
REPORT_LABELS_FILE = "curation_report_labels.csv"
EXPORT_FILE = "hcy_ground_truth_curated.csv"

#: Actions the fold understands. Anything else is logged and skipped, not crashing a
#: session, a log written by a newer version of this app must still open in an older one.
ACTIONS = frozenset({
    "suggest", "withdraw", "approve", "reject",
    "suggest_delete", "withdraw_delete",
    "verdict", "anchor", "edit", "label", "comment", "delete_comment",
    "confirm_patient", "unconfirm_patient",
})

#: The fields an ``edit`` event may rewrite. not ``status``: repairing an annotation's
#: evidence and deciding whether it belongs are two judgements, and letting one event carry both
#: would make "who kept this term" unanswerable from the log.
#:
#: ``labels`` is here because the qualifiers, negated, family, lab value, are a property of the
#: annotation, ticked on the form that creates it. Correcting one is a repair, like
#: correcting a trigger word, and must no more imply a verdict than that does.
EDITABLE = ("segment_idx", "segment_text", "trigger_word", "hpo_code", "hpo_name", "note",
            "labels")

#: The statuses a row can carry. ``suggested`` and ``delete_suggested`` are the two a row can hold
#: while still waiting on somebody.
IN_GOLD = frozenset({"approved", "kept"})

#: An annotation somebody has proposed removing. It stays *out* of the ground truth while proposed, a term
#: under active dispute is not something to ship, and the approver settles it with the same
#: keep/remove verdict buttons every other row uses: Remove accepts the proposal, Keep rejects it.
DELETE_SUGGESTED = "delete_suggested"

CURRENT_FIELDS = [
    "patient_id", "segment_idx", "segment_text", "hpo_code", "hpo_name",
    "trigger_word", "source", "status", "difficulty", "labels",
    "delete_reason", "delete_by", "author", "updated_at", "note", "key",
]

#: One row per comment, newest last. Comments are about a *report*, so they key on ``patient_id``
#: alone, the same join as ``curation_report_labels.csv``.
COMMENTS_FIELDS = ["patient_id", "comment_id", "ts", "author", "text"]

#: Long-form, one row per (annotation, label). Wide would need a column per label and a schema
#: change every time the vocabulary grows. Long is what a ``groupby`` wants anyway, which is the
#: only thing this file is for, slicing recall by why an annotation was hard.
LABELS_FIELDS = ["patient_id", "hpo_code", "source", "key", "kind", "label",
                 "label_display", "label_group"]

#: The fixed part of the report-label row. The per-label ``0``/``1`` columns follow, named by label
#: id in vocabulary order, so the header is stable between runs and a diff shows real changes.
#:
#: ``n_comments``, not a note column: free text about a report lives in
#: ``curation_comments.csv``, where it can be appended to and attributed, and duplicating the last
#: one here would make it ambiguous which was the record.
REPORT_LABELS_HEAD = ["patient_id", "difficulty", "labels", "n_labels", "n_comments"]
REPORT_LABELS_TAIL = ["author", "updated_at"]


def report_labels_fields() -> list[str]:
    """The report-label header: the fixed columns, one per report-level label, then the metadata."""
    from . import labels as vocab

    return REPORT_LABELS_HEAD + vocab.label_ids("patient") + REPORT_LABELS_TAIL

_WS_RUN = re.compile(r"\s*[\r\n]+\s*")


def share(path: str, mode: int = FILE_MODE) -> bool:
    """``chmod`` *path*, best effort. ``True`` if it now has *mode*.

    Best effort because the common case for failing is the benign one: the file belongs to the
    curator who started first, ``chmod`` on somebody else's file is refused no matter how open the
    permissions already are, and it does not need changing anyway. Raising there would turn "the
    other curator got here first" into a crash.

    What must not happen is failing *silently* when it counts, so the caller gets a boolean and
    the startup check (:func:`writability_problem`) tests the thing that actually counts, whether
    this user can write, not trusting that this call did anything.
    """
    try:
        os.chmod(path, mode)
        return True
    except OSError as exc:
        logger.debug("could not chmod %s to %o: %s", path, mode, exc)
        return False


def writability_problem(curation_dir: str) -> str:
    """Why this user cannot record anything here, as a sentence, or ``""`` when they can.

    Checked **at startup**, because the alternative is discovering it on the first click. A curator
    who has read a report, formed a verdict and pressed *Keep* has done the expensive part of the
    work. A ``PermissionError`` at that point reads as "the tool is broken", and the judgement is
    gone. This is also why the message names the fix, not the fault: the person who hits it
    is not the person who can run ``chmod``.
    """
    if not os.path.isdir(curation_dir):
        return f"The curation directory does not exist and could not be created: {curation_dir}"
    fix = (f"Ask for: chmod {DIR_MODE & 0o7777:o} {curation_dir} && "
           f"chmod {FILE_MODE:o} {curation_dir}/curation_*")
    if not os.access(curation_dir, os.W_OK | os.X_OK):
        return (f"No write access to {curation_dir} — every action would fail on save. {fix}")
    log_path = os.path.join(curation_dir, EVENTS_FILE)
    if os.path.exists(log_path) and not os.access(log_path, os.W_OK):
        return (f"The curation log exists but is not writable by you: {log_path}. It belongs to "
                f"whoever curated here first. {fix}")
    return ""


def flatten(value) -> str:
    """One CSV cell: newline runs collapsed to a space, and a list joined with ``;``.

    The newline rule is ``annotation_ui.app._flatten``'s, a record that spans lines is one a
    viewer, an Excel round-trip or a naive ``wc -l`` can corrupt or miscount.

    The list rule exists because the label set is a list and ``str(["a", "b"])`` writes
    ``['a', 'b']``, a cell that has to be parsed back with ``ast.literal_eval`` to be read at all.
    ``;`` is the separator the ground truth files already use for this.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = ";".join(str(v) for v in value)
    return _WS_RUN.sub(" ", str(value)).strip()


def now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``, the timestamp format of the event log."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def target_key(source: str, patient_id: str, hpo_code: str, slot=None) -> str:
    """The stable address of an annotation that already existed.

    *slot* is what separates one code a source carries **twice** for one patient: PhenoBERT's
    character offset, or the occurrence index :func:`sources.assign_slots` hands a repeated
    annotation. Two triggers for one phenotype is a normal annotation, and a curator keeping one
    must not implicitly keep the other.

    The first occurrence carries no slot, so its key stays the short one and appending a second
    occurrence to a source file never re-keys the annotation somebody already adjudicated.
    """
    key = f"{source}|{patient_id}|{hpo_code}"
    return f"{key}|{slot}" if slot is not None else key


# ──────────────────────────────────────────────────────────────────────────────
# The log
# ──────────────────────────────────────────────────────────────────────────────
class EventLog:
    """The append-only log and the state folded from it. One instance per curation directory.

    One instance per *process*, though, and several curators share one ``curation_dir``, each
    through their own tunnel and their own process. So the in-memory ``events`` list is a cache of
    a file somebody else is also appending to, and it is re-read whenever the file has changed
    (:meth:`refresh`). Without that, each process folds only its own clicks, and ``commit`` then
    rewrites the four derived CSVs from that partial view, one curator's save silently reverting
    another's. The JSONL itself never loses anything (append-only under ``flock``), so the damage
    is confined to the derived files and heals on a re-read. Re-reading is simply cheaper than
    explaining that to the person whose afternoon disappeared.
    """

    def __init__(self, curation_dir: str, author: str = "unknown"):
        self.dir = curation_dir
        self.author = author or "unknown"
        # ``makedirs`` applies the umask to its ``mode``, so the setgid bit and the group write bit
        # would both be filtered out of it. Create, then set the mode we actually mean.
        #
        # Neither step is allowed to raise. A curator whose group has no write access to the parent
        # would otherwise get a ``PermissionError`` traceback out of a constructor, before the app
        # exists, so with nowhere to say it in words. Construction stays total and
        # :func:`writability_problem` reports it at startup as a sentence naming the fix.
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as exc:
            logger.warning("could not create %s: %s", self.dir, exc)
        else:
            share(self.dir, DIR_MODE)
            self._share_existing()
        self._lock = threading.Lock()
        self._stat: tuple[int, int] | None = None
        self.events: list[dict] = []
        self._reload()

    def _share_existing(self) -> None:
        """Re-open the files an earlier run left private. Best effort, every startup.

        Without this the mode fix only reaches files created *after* it shipped: a directory that
        has already been curated in holds a ``0644`` log and four ``0600`` CSVs, and the second
        curator still cannot write, the bug persists where there is already work to lose.

        Every startup, not once, because "once" would need a marker file, and the marker
        would be the thing that gets out of step. Five ``chmod`` calls on files we usually own
        cost nothing, and on files we do not they fail and are ignored, which is correct: they
        belong to a curator who has already run this same code.
        """
        for name in (EVENTS_FILE, CURRENT_FILE, COMMENTS_FILE, LABELS_FILE, REPORT_LABELS_FILE,
                     EXPORT_FILE):
            path = os.path.join(self.dir, name)
            if os.path.exists(path):
                share(path)

    # ── keeping up with the other curators ───────────────────────────────────
    def _stat_log(self) -> tuple[int, int] | None:
        """``(size, mtime_ns)`` of the log, or ``None`` when it does not exist yet."""
        try:
            info = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (info.st_size, info.st_mtime_ns)

    def _reload(self) -> None:
        """Re-read the whole log. Caller holds the lock (or is ``__init__``).

        The stat is taken *before* the read and kept even if the file grew during it. Erring that
        way costs one redundant re-read. Erring the other way would record a stat newer than the
        events actually held and skip the next one.
        """
        before = self._stat_log()
        self.events = read_events(self.path)
        self._stat = before

    def _refresh_locked(self) -> bool:
        """Re-read if the file changed underneath us. ``True`` when it did. Lock held."""
        current = self._stat_log()
        if current == self._stat:
            return False
        self._reload()
        return True

    def refresh(self) -> bool:
        """Pick up events other processes appended. ``True`` when the log had grown."""
        with self._lock:
            return self._refresh_locked()

    @property
    def path(self) -> str:
        """Path of the append-only event log."""
        return os.path.join(self.dir, EVENTS_FILE)

    @property
    def current_path(self) -> str:
        """Path of the file that holds the current state folded from the log."""
        return os.path.join(self.dir, CURRENT_FILE)

    @property
    def export_path(self) -> str:
        """Path of the curated ground-truth export."""
        return os.path.join(self.dir, EXPORT_FILE)

    # ── writing ──────────────────────────────────────────────────────────────
    def append(self, action: str, patient_id: str, **fields) -> dict:
        """Write one event and return it. The only mutating entry point in this app.

        Serialised on a lock and flushed to the OS before returning, so the caller can tell the
        curator "saved" and mean it. The whole-file lock is deliberate over a per-line one: a
        session is one person clicking, and the contention that counts is two browser tabs, which
        a lock resolves and a race does not.
        """
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        event = {
            "event_id": fields.pop("event_id", None) or uuid.uuid4().hex,
            "ts": now_iso(),
            "author": self.author,
            "action": action,
            "patient_id": str(patient_id),
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            # Take in anything another curator wrote first, so ``events`` stays a true prefix of
            # The file and the commit that follows this call sees their work as well as ours.
            self._refresh_locked()
            fresh = not os.path.exists(self.path)
            with open(self.path, "a", encoding="utf-8") as handle:
                _flock(handle)
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if fresh:
                # Only on the line that created it: re-chmodding on every append would be a syscall
                # per click to no purpose, and would fight a deliberate change somebody had made.
                share(self.path)
            self.events.append(event)
            self._stat = self._stat_log()
        return event

    def commit(self, rows_by_patient, patient_state=None, patient_ids=None,
               comments=None) -> None:
        """Rewrite the four derived CSVs from the folded state.

        Called after every mutation. All four are small for this cohort, a few thousand rows
        between them, so atomic full rewrites cost single-digit milliseconds and none is ever a
        stale view of the log, a property worth far more here than the writes they save.

        *patient_ids* is the cohort, so the report-label file can carry a row for every report,
        not only the labelled ones. Without it the file falls back to the patients that
        have annotations, which understates the cohort.
        """
        rows: list[dict] = []
        for patient_id in sorted(rows_by_patient):
            rows.extend(rows_by_patient[patient_id])
        write_csv_atomic(self.current_path, CURRENT_FIELDS, rows)
        write_comments(self.comments_path, comments or {})
        write_labels(self.labels_path, rows_by_patient)
        write_report_labels(self.report_labels_path, patient_state or {},
                            patient_ids if patient_ids is not None else rows_by_patient,
                            comments or {})

    # ── reading ──────────────────────────────────────────────────────────────
    def fold(self) -> dict:
        """The current state, see :func:`fold_events`. Picks up other curators first."""
        self.refresh()
        return fold_events(self.events)

    @property
    def labels_path(self) -> str:
        """Path of the per-annotation difficulty labels export."""
        return os.path.join(self.dir, LABELS_FILE)

    @property
    def report_labels_path(self) -> str:
        """Path of the per-report difficulty labels export."""
        return os.path.join(self.dir, REPORT_LABELS_FILE)

    @property
    def comments_path(self) -> str:
        """Path of the report comments export."""
        return os.path.join(self.dir, COMMENTS_FILE)


def read_events(path: str) -> list[dict]:
    """Every parseable line, in order. A torn final line is skipped, never fatal.

    A truncated last line is what a crash mid-append leaves behind, and refusing to open the
    log because of it would turn a lost click into a lost session.
    """
    events: list[dict] = []
    if not os.path.isfile(path):
        return events
    n_bad = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if isinstance(record, dict) and record.get("action"):
                # The stored log predates the renaming of two source labels. See labels.py.
                events.append(translate_stored_labels(record))
            else:
                n_bad += 1
    if n_bad:
        logger.warning("%s: skipped %d unparseable line(s)", os.path.basename(path), n_bad)
    return events


def fold_events(events) -> dict:
    """Fold the log into ``{"rows": {key: row}, "confirmed": set, "n_skipped": int}``.

    Order is the log's order and the last event addressing a key wins, so re-running a prefix of the
    log yields the state as of that moment, which is what makes the log auditable, not
    merely appended to.
    """
    rows: dict[str, dict] = {}
    confirmed: set[str] = set()
    patients: dict[str, dict] = {}
    comments: dict[str, list[dict]] = {}
    n_skipped = 0

    for event in events:
        action = event.get("action")
        patient_id = str(event.get("patient_id", ""))
        if action not in ACTIONS:
            n_skipped += 1
            continue

        if action == "confirm_patient":
            confirmed.add(patient_id)
            continue
        if action == "unconfirm_patient":
            confirmed.discard(patient_id)
            continue

        if action == "comment":
            # Append, never replace: a comment is a thing somebody said at a time, and the next
            # one is a second thing, not a correction of the first.
            comments.setdefault(patient_id, []).append({
                "comment_id": event["event_id"],
                "ts": event.get("ts", ""),
                "author": event.get("author", ""),
                "text": event.get("text", ""),
            })
            continue
        if action == "delete_comment":
            target = event.get("target_key")
            kept = [c for c in comments.get(patient_id, []) if c["comment_id"] != target]
            if kept:
                comments[patient_id] = kept
            else:
                comments.pop(patient_id, None)
            continue

        if action == "label" and not event.get("target_key"):
            # A report-level label event replaces the whole set, not adding to it: the UI
            # sends what the checkboxes now say, and a merge would make unticking impossible.
            patients[patient_id] = {
                "labels": list(event.get("labels") or ()),
                "difficulty": event.get("difficulty", ""),
                "author": event.get("author", ""),
                "updated_at": event.get("ts", ""),
                "note": event.get("note", patients.get(patient_id, {}).get("note", "")),
            }
            continue

        if action == "suggest":
            # A suggestion is addressed by its own event id, it had no prior existence to name.
            rows[event["event_id"]] = {
                "key": event["event_id"],
                "patient_id": patient_id,
                "segment_idx": event.get("segment_idx"),
                "segment_text": event.get("segment_text", ""),
                "hpo_code": event.get("hpo_code", ""),
                "hpo_name": event.get("hpo_name", ""),
                "trigger_word": event.get("trigger_word", ""),
                "source": "new",
                "status": "suggested",
                "difficulty": event.get("difficulty", ""),
                "labels": list(event.get("labels") or ()),
                "delete_reason": "",
                "delete_by": "",
                "author": event.get("author", ""),
                "updated_at": event.get("ts", ""),
                "note": event.get("note", ""),
            }
            continue

        key = event.get("target_key")
        if not key:
            n_skipped += 1
            continue

        if action == "withdraw":
            # Only a suggestion this app made can be withdrawn out of existence. An event naming a
            # ground truth row here would otherwise delete the app's whole record of that annotation,
            # including the verdict somebody already passed on it.
            if rows.get(key, {}).get("source") == "new":
                rows.pop(key, None)
            else:
                n_skipped += 1
            continue

        row = rows.get(key)
        if row is None:
            # A verdict on an existing annotation is the first time that annotation appears in the
            # log: the ground truth files are the input, so the event carries the identifying fields.
            source, _, rest = key.partition("|")
            row = {
                "key": key,
                "patient_id": patient_id,
                "segment_idx": event.get("segment_idx"),
                "segment_text": event.get("segment_text", ""),
                "hpo_code": event.get("hpo_code") or (rest.split("|") + ["", ""])[1],
                "hpo_name": event.get("hpo_name", ""),
                "trigger_word": event.get("trigger_word", ""),
                "source": source,
                "status": "",
                "difficulty": "",
                "labels": [],
                "delete_reason": "",
                "delete_by": "",
                "author": event.get("author", ""),
                "updated_at": event.get("ts", ""),
                "note": event.get("note", ""),
            }
            rows[key] = row

        if action == "suggest_delete":
            # A proposal, not a decision. The row leaves the ground truth while it is open, a term under
            # active dispute is not something to ship, and the approver settles it with the same
            # verdict buttons every other row uses.
            row["status"] = DELETE_SUGGESTED
            row["delete_reason"] = event.get("text", "")
            row["delete_by"] = event.get("author", "")
        elif action == "withdraw_delete":
            # Back to undecided, not to "kept": withdrawing a proposal is the proposer changing
            # their mind, which says nothing about whether the annotation is correct.
            if row.get("status") == DELETE_SUGGESTED:
                row["status"] = ""
            row["delete_reason"] = ""
            row["delete_by"] = ""
        elif action == "edit":
            # Repairing an annotation in place: its trigger word, its segment, its code, its note.
            # Only the fields the event actually names are touched, so an edit that fixes a trigger
            # cannot silently blank a segment somebody else located. The status is never among
            # them, see ``EDITABLE``.
            for field in EDITABLE:
                if field in event:
                    # A list is copied, not aliased: the event dict outlives this fold and a
                    # shared list would let a later mutation of the row rewrite history.
                    row[field] = (list(event[field] or ())
                                  if field == "labels" else event[field])
        elif action == "anchor":
            # Locating records evidence, not a decision: a term's status is untouched by learning
            # which sentence it came from.
            row["segment_idx"] = event.get("segment_idx", row.get("segment_idx"))
            row["segment_text"] = event.get("segment_text", row.get("segment_text", ""))
            row["trigger_word"] = event.get("trigger_word", row.get("trigger_word", ""))
        elif action == "approve":
            row["status"] = "approved"
        elif action == "reject":
            row["status"] = "rejected"
        elif action == "verdict":
            row["status"] = event.get("status", "")
        elif action == "label":
            # Same replace-not-merge rule as the report-level case, for the same reason.
            row["labels"] = list(event.get("labels") or ())
            row["difficulty"] = event.get("difficulty", "")
        if action != "edit":
            # An ``edit`` has already written the fields it named. Falling through here
            # would make clearing a note impossible, the generic rule only writes truthy values,
            # so the old text would survive its own deletion.
            if event.get("note"):
                row["note"] = event["note"]
            if event.get("hpo_name"):
                row["hpo_name"] = event["hpo_name"]
        row["author"] = event.get("author", row.get("author", ""))
        row["updated_at"] = event.get("ts", row.get("updated_at", ""))

    return {"rows": rows, "confirmed": confirmed, "patients": patients,
            "comments": comments, "n_skipped": n_skipped}


# ──────────────────────────────────────────────────────────────────────────────
# derived files
# ──────────────────────────────────────────────────────────────────────────────
def write_csv_atomic(path: str, fieldnames: list[str], rows) -> None:
    """Write *rows* to *path* via ``tmp`` + ``os.replace``.

    A reader, the curator's own ``pandas.read_csv``, or a scoring job, never sees a half-written
    file: ``os.replace`` is atomic within a filesystem, so the path either holds the whole old file
    or the whole new one. The temp file is created in the same directory for that reason.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", delete=False, dir=directory, prefix=".tmp-", suffix=".csv",
        newline="", encoding="utf-8",
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: flatten(row.get(name, "")) for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        # ``NamedTemporaryFile`` creates 0600 and ``os.replace`` preserves the *new* file's mode,
        # so without this every derived CSV ends up private to whoever last wrote it, including
        # from the curator who is going to open it in a spreadsheet.
        share(handle.name)
        os.replace(handle.name, path)
    except BaseException:
        # Leave no ``.tmp-*`` behind on a failed write. The next run would not know it was garbage.
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def write_labels(path: str, rows_by_patient) -> dict:
    """The **annotation** qualifiers, long-form: one row per (annotation, label).

    Joined to a predictions dump on ``(patient_id, hpo_code)`` this answers the question the
    aggregate cannot, whether the misses are concentrated in terms belonging to a relative, in
    terms the report negates, in terms carried only by a lab value, or in terms the report never
    names outright.

    ``kind`` is ``"qualifier"``. It used to also carry ``"difficulty"`` rows for the annotation-level
    grade. That grade is gone, the report level keeps its own, but the column stays, because a
    file written before the change is still read with the same reader, and a ``groupby(["kind",
    "label"])`` over both is still one call.
    """
    from . import labels as vocab

    rows: list[dict] = []

    def emit(patient_id, row, kind, value):
        entry = vocab.INDEX.get(value, {})
        rows.append({
            "patient_id": patient_id, "hpo_code": row.get("hpo_code", ""),
            "source": row.get("source", ""), "key": row.get("key", ""),
            "kind": kind, "label": value,
            "label_display": entry.get("display", value),
            "label_group": entry.get("group", "unknown"),
        })

    for patient_id in sorted(rows_by_patient):
        for row in rows_by_patient[patient_id]:
            for value in row.get("labels") or ():
                emit(patient_id, row, "qualifier", value)

    write_csv_atomic(path, LABELS_FIELDS, rows)
    return {"path": path, "n_rows": len(rows)}


def write_comments(path: str, comments) -> dict:
    """Every comment, one row each, oldest first within a report.

    Long form and append-only, matching the log: a comment is a thing somebody said at a time, so
    it keeps its author and timestamp. Joined on ``patient_id`` alone, comments are about the
    report, not about a term.
    """
    rows = []
    for patient_id in sorted(comments):
        for entry in comments[patient_id]:
            rows.append({
                "patient_id": patient_id,
                "comment_id": entry.get("comment_id", ""),
                "ts": entry.get("ts", ""),
                "author": entry.get("author", ""),
                "text": entry.get("text", ""),
            })
    write_csv_atomic(path, COMMENTS_FIELDS, rows)
    return {"path": path, "n_rows": len(rows)}


def write_report_labels(path: str, patient_state, patient_ids, comments=None) -> dict:
    """The **report** difficulty labels: one row per report, wide.

    Every patient gets a row, labelled or not, the same rule the ground truth export follows, and for the
    same reason. A file that silently omitted the unlabelled reports would make "how much of the
    cohort has been characterised" unanswerable from the file itself, which is the first thing
    anyone will ask of it.

    Each label is its own ``0``/``1`` column so a stratified table is a ``merge`` and a comparison,
    with no parsing. The joined ``labels`` cell is kept beside them for reading.
    """
    from . import labels as vocab

    ids = vocab.label_ids("patient")
    comments = comments or {}
    rows = []
    n_labelled = 0
    for patient_id in sorted(patient_ids):
        entry = patient_state.get(patient_id) or {}
        chosen = vocab.clean(entry.get("labels"), "patient")
        grade = entry.get("difficulty", "")
        if chosen or grade:
            n_labelled += 1
        rows.append({
            "patient_id": patient_id,
            "difficulty": grade,
            "labels": ";".join(chosen),
            "n_labels": len(chosen),
            "n_comments": len(comments.get(patient_id) or ()),
            **{value: int(value in chosen) for value in ids},
            "author": entry.get("author", ""),
            "updated_at": entry.get("updated_at", ""),
        })

    write_csv_atomic(path, report_labels_fields(), rows)
    return {"path": path, "n_rows": len(rows), "n_labelled": n_labelled}


def export_gold(path: str, rows_by_patient, patient_ids) -> dict:
    """Write the two-column ``patient_id,hpo_codes`` ground truth from rows whose status is in the ground truth.

    Every patient in *patient_ids* gets a line, including those with no terms, an empty cell is a
    report with no phenotypes, and dropping it would silently shrink the cohort the way a filtered
    ground truth file quietly changes a denominator.
    """
    written = 0
    n_terms = 0
    rows = []
    for patient_id in sorted(patient_ids):
        codes: list[str] = []
        for row in rows_by_patient.get(patient_id, []):
            code = str(row.get("hpo_code", "")).strip()
            if row.get("status") in IN_GOLD and code and code not in codes:
                codes.append(code)
        rows.append({"patient_id": patient_id, "hpo_codes": ";".join(codes)})
        written += 1
        n_terms += len(codes)
    write_csv_atomic(path, ["patient_id", "hpo_codes"], rows)
    return {"path": path, "n_patients": written, "n_pairs": n_terms}


def _flock(handle) -> None:
    """Advisory exclusive lock, where the platform has one.

    Two browser tabs on one log is the realistic concurrency here, and interleaved appends would
    corrupt a line. Windows has no ``fcntl``. The lock is then skipped, not the write.
    """
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass
