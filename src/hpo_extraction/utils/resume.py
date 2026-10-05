"""Resume primitives for the long-running cluster jobs.

Both earlier drivers process a cohort report by report over many hours, and both used to lose
everything to a SLURM wall-clock kill: output files were opened with ``"w"``, written through
Python's buffer and only closed at the very end, so a SIGKILL discarded whatever had not reached
8 KB (an earlier exploratory run's ``tau_0.5/*.jsonl`` were literally 0 bytes after 48 h of GPU time), and a resubmit
restarted from report 0.

Two small pieces fix that, and are shared so the tree driver and the SLM-ensemble driver behave
identically:

* :class:`Checkpoint`, an append-only record of the units (reports) that are fully written. It
  also decides the *file mode* for the run's outputs, so a fresh run truncates and a resumed run
  appends. Units already in the resume file are skipped by the caller.
* :class:`GracefulStop`, SIGUSR1/SIGTERM (and an optional self-imposed deadline) flip a flag the
  driver polls *between* units, so a job that is about to hit its wall clock stops on a unit
  boundary with everything flushed instead of being killed mid-write.

The unit of progress is coarse (one report): fine enough that a kill costs minutes,
coarse enough that flushing per unit is free next to the SLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Iterable

logger = logging.getLogger(__name__)


def _read_done(path: str) -> set[str]:
    """Unit ids recorded as complete. A truncated final line (hard kill mid-write) is ignored."""
    done: set[str] = set()
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("unit_id")
            if uid is not None:
                done.add(str(uid))
    return done


class Checkpoint:
    """Append-only progress log for one worker.

    Args:
        run_dir: the run's output directory, the resume file lives beside its artifacts.
        key: worker identity, e.g. ``"tree_gate_lr_s3of8"`` or ``"extract_phi4"``. It names the
            file, so the array tasks that share a run directory never touch each other's progress.
        resume: ``False`` starts from scratch and truncates both the resume file and (via
            :attr:`file_mode`) the run's outputs.

    A unit is marked complete only *after* its records are written and flushed, so the invariant is
    "everything in :attr:`done` is on disk in full". The converse is not guaranteed: a hard kill can
    leave a partially written unit that is not marked, which the next run re-does and re-appends, ``evaluation/merge_shards.py`` drops the stale partial block when consolidating.
    """

    def __init__(self, run_dir: str, key: str, resume: bool = True) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, f".checkpoint_{key}.jsonl")
        self.key = key
        self.done: set[str] = _read_done(self.path) if resume else set()
        self.resumed = bool(self.done)
        self._f = open(self.path, "a" if resume else "w", encoding="utf-8")

    @property
    def file_mode(self) -> str:
        """Mode the run's output files must be opened with: append when continuing, else truncate."""
        return "a" if self.resumed else "w"

    @property
    def n_done(self) -> int:
        """Number of units already completed."""
        return len(self.done)

    def is_done(self, unit_id) -> bool:
        """True when *unit_id* was completed in an earlier session."""
        return str(unit_id) in self.done

    def pending(self, unit_ids: Iterable) -> list:
        """The subset of ``unit_ids`` still to do, order preserved."""
        return [u for u in unit_ids if str(u) not in self.done]

    def mark(self, unit_id, **extra) -> None:
        """Record a unit as fully written. Flushed immediately, this is the durability point."""
        self._f.write(json.dumps({"unit_id": str(unit_id), "ts": time.time(), **extra}) + "\n")
        self._f.flush()
        self.done.add(str(unit_id))

    def close(self) -> None:
        """Close the resume file."""
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


#: Signals that request a graceful stop: SLURM's ``--signal=B:USR1@300`` pre-emption warning, and
#: The hard one. Built by lookup rather than written literally because ``SIGUSR1`` does not exist on
#: Windows and a default argument is evaluated at *definition* time, naming it directly makes
#: merely importing this module (and every driver that touches it) an ImportError there, long
#: before any handler would be installed. ``GracefulStop`` already tolerates a signal the platform
#: refuses. This extends the same tolerance to the default. The cluster is Linux and installs both.
_DEFAULT_STOP_SIGNALS = tuple(
    sig for sig in (getattr(signal, "SIGUSR1", None), getattr(signal, "SIGTERM", None))
    if sig is not None
)


class GracefulStop:
    """Stop-requested flag driven by signals and/or an elapsed-time budget.

    SLURM's ``--signal=B:USR1@300`` warns the batch script five minutes before the wall clock. The
    job script forwards that to the driver, which finishes the current report and exits cleanly.
    ``max_runtime_s`` gives the same behaviour without any signal wiring (useful when running
    outside SLURM, or as a belt-and-braces margin below the ``--time`` limit).

    Handler registration is best-effort: :func:`signal.signal` only works on the main thread, and
    the flag is simply never set elsewhere.

    The default signal tuple is :data:`_DEFAULT_STOP_SIGNALS`, built by lookup, see the note
    there for why it is not written literally.
    """

    def __init__(self, max_runtime_s: float | None = None, signals=_DEFAULT_STOP_SIGNALS):
        self.requested = False
        self.reason = ""
        self.max_runtime_s = float(max_runtime_s) if max_runtime_s else None
        self._t0 = time.time()
        for sig in signals:
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError, AttributeError):  # not main thread / unsupported
                logger.debug("Could not install handler for %s", sig)

    def _handle(self, signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self.requested = True
        self.reason = f"{name} received"
        logger.warning("Graceful stop requested (%s) — finishing the current unit …", self.reason)

    @property
    def elapsed_s(self) -> float:
        """Seconds since the job started."""
        return time.time() - self._t0

    def should_stop(self) -> bool:
        """True once a signal arrived or the time budget is spent. Poll between units."""
        if not self.requested and self.max_runtime_s is not None:
            if self.elapsed_s >= self.max_runtime_s:
                self.requested = True
                self.reason = f"max_runtime_s={self.max_runtime_s:g} reached"
                logger.warning("Graceful stop: %s", self.reason)
        return self.requested


def open_jsonl(path: str, mode: str, encoding: str = "utf-8"):
    """Open a JSONL artifact, terminating a partial final line before appending to it.

    A process killed mid-write can leave a line with no trailing newline. Appending straight onto
    it would glue the stump to the first record of the resumed session, corrupting a record that is
    otherwise perfectly good, so close the line off first. The stump itself stays behind as one
    unparsable line, which ``evaluation/merge_shards.py`` drops.
    """
    if mode == "a" and os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
        if needs_newline:
            with open(path, "a", encoding=encoding) as fix:
                fix.write("\n")
    return open(path, mode, encoding=encoding)


def flush_all(handles: Iterable) -> None:
    """Flush every open file handle in ``handles`` (closed/None entries are skipped)."""
    for fh in handles:
        if fh is not None and not getattr(fh, "closed", False):
            fh.flush()
