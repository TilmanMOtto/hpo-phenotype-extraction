"""The bundle: one report, everything five methods did to it, in one JSON small enough to serve.

A bundle is the contract between the two halves of this package. The builder produces it where the
artifacts are. The app consumes it and never looks at anything else. Two properties are what make
that split worth having, and both are enforced here rather than hoped for.

**It is bounded.** :data:`MAX_BUNDLE_BYTES` is a hard cap, not a target. TreePhenoRAG alone writes
~39 MB of verifier calls per report, and the failure mode of a builder that "mostly" keeps bundles
small is one enormous report that hangs a callback months later. :func:`write` raises instead.

**It names its inputs.** Every artifact the builder read contributes a SHA-256 prefix to
``inputs``. The app compares those against what is on disk and says so when they have moved, which
is the same rule ``app/exp13_ui``'s disk cache uses -- a stale panel that still renders is worse
than one that does not, because it shows plausible numbers computed from data that has since moved.

Coordinates
-----------
``segments[i]["text"]`` is the segment **as the report spells it** (``spans.segment_texts``), and
every ``span`` in the bundle is a half-open ``[start, end)`` into that string. Nothing in a bundle
indexes the tokenized sentence, and nothing indexes the whole report: mixing those two coordinate
systems is the off-by-a-few-characters bug ``src/hpo_extraction/curation/spans.py`` exists to prevent, and
a bundle that carried both would reintroduce it at the JSON boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: Bumped whenever a field changes meaning. A bundle from an older schema is refused, not
#: read leniently -- the views would otherwise draw a field that no longer means what they think.
SCHEMA_VERSION = 1

#: 2 MB. Comfortably above the ~400 KB a dense report measures and far below anything that would
#: make a callback feel slow over a tunnel.
MAX_BUNDLE_BYTES = 2 * 1024 * 1024

INDEX_FILE = "index.json"

#: The three verdicts a (method, term) pair can carry. ``fn`` is an annotated term the method did not
#: predict. There is no ``tn`` -- the negative class is the whole ontology.
OUTCOMES = ("tp", "fp", "fn")


def digest(path, length: int = 16) -> str:
    """SHA-256 prefix of a file, or empty when it is not there.

    Absence returns empty, not raising: a method whose artifacts are missing is a column
    the app greys out, not a build that fails.
    """
    path = Path(path)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


def stamp(path, name_relative_to=None) -> dict:
    """``{path, sha256, size, mtime}`` for one input, or an empty dict.

    The digest is what :func:`drifted` checks. ``size`` and ``mtime`` are carried for the human
    reading a provenance panel, who wants to know *when* an artifact moved and not only that it
    did. ``path`` is stored relative to *name_relative_to* when given, so a bundle built on the
    cluster still names its inputs legibly when read from anywhere else.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    stat = path.stat()
    shown = str(path)
    if name_relative_to:
        try:
            shown = str(path.relative_to(Path(name_relative_to)))
        except ValueError:
            pass
    return {"path": shown, "sha256": digest(path), "size": stat.st_size,
            "mtime": int(stat.st_mtime)}


def bundle_path(root, report_id: str) -> Path:
    """Path of the bundle of *report_id* under *root*."""
    return Path(root) / (str(report_id) + ".json")


def index_path(root) -> Path:
    """Path of the bundle index under *root*."""
    return Path(root) / INDEX_FILE


# -- writing -----------------------------------------------------------------

def write(root, bundle: dict) -> Path:
    """Write one bundle atomically. Raises if it is malformed or over the cap.

    Atomic because the app may be running against this directory while a rebuild is in flight, and
    a half-written bundle is indistinguishable from a report whose methods all found nothing.
    """
    validate(bundle)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_BUNDLE_BYTES:
        raise ValueError(
            "bundle for {!r} is {:,} bytes, over the {:,} cap. Something is keeping per-call or "
            "per-node rows it should be filtering -- apps/compare_ui/methods/tree.py is the usual "
            "culprit.".format(bundle.get("report_id"), len(encoded), MAX_BUNDLE_BYTES))
    path = bundle_path(root, str(bundle["report_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)
    return path


def write_index(root, index: dict) -> Path:
    """Write *index* under *root* and return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = index_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


# -- reading -----------------------------------------------------------------

def read(root, report_id: str):
    """One bundle, or ``None`` if it is absent, unreadable or from another schema.

    Unreadable degrades to ``None`` for the same reason every loader in the sibling apps does: a
    corrupt bundle must cost one report, not the whole session.
    """
    path = bundle_path(root, report_id)
    if not path.is_file():
        return None
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA_VERSION:
        return None
    return bundle


def read_index(root):
    """The bundle index under *root*, or None when it does not exist."""
    path = index_path(root)
    if not path.is_file():
        return None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or index.get("schema") != SCHEMA_VERSION:
        return None
    return index


def drifted(bundle: dict, output_base: str = "") -> list:
    """Which of a bundle's recorded inputs no longer match the file on disk.

    Empty means the bundle still describes the artifacts it was built from. A non-empty list is the
    app's cue to show a staleness banner: it is information for the reader, not a reason to refuse
    to draw -- the bundle is still internally consistent, it just no longer matches the cluster.

    Inputs whose path cannot be reached from here -- the common case when the app runs somewhere
    the experiment directories are not mounted -- are **not** reported as drift. Absence of
    evidence is not evidence of change, and treating it as such would light the banner permanently.
    """
    out = []
    for name, record in sorted((bundle.get("inputs") or {}).items()):
        recorded = (record or {}).get("sha256")
        shown = (record or {}).get("path")
        if not recorded or not shown:
            continue
        candidate = Path(shown)
        if not candidate.is_absolute() and output_base:
            candidate = Path(output_base) / candidate
        if not candidate.is_file():
            continue
        if digest(candidate) != recorded:
            out.append(name)
    return out


# -- the schema --------------------------------------------------------------

_REQUIRED = ("schema", "report_id", "cohort", "segments", "terms", "gold", "methods", "metrics")


def validate(bundle: dict) -> None:
    """Raise unless *bundle* is the shape every view assumes. Cheap enough to run on every write.

    This is the only place the shape is written down as code, not prose, so the checks are
    the ones a view would otherwise have to defend against: a span that does not index its segment,
    an outcome outside the vocabulary, a term drawn with no entry in ``terms``. Each of those
    renders as something subtly wrong, not as an error, which is the class of bug a
    render harness cannot catch.
    """
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict, got " + type(bundle).__name__)
    missing = [k for k in _REQUIRED if k not in bundle]
    if missing:
        raise ValueError("bundle is missing {}".format(missing))
    if bundle["schema"] != SCHEMA_VERSION:
        raise ValueError("bundle schema {} != {}".format(bundle["schema"], SCHEMA_VERSION))

    segments = bundle["segments"]
    if not isinstance(segments, list):
        raise TypeError("segments must be a list")
    for i, seg in enumerate(segments):
        if seg.get("idx") != i:
            raise ValueError(
                "segments[{}] carries idx {!r}; it must be its own position, because every span "
                "in the bundle addresses a segment by position".format(i, seg.get("idx")))
        for field in ("text", "gap_before"):
            if not isinstance(seg.get(field), str):
                raise TypeError("segments[{}].{} must be a string".format(i, field))

    terms = bundle["terms"]
    if not isinstance(terms, dict):
        raise TypeError("terms must be a dict keyed by HPO id")

    for row in bundle["gold"]:
        _check_placement(row, segments, where="gold")
        if row["hpo_id"] not in terms:
            raise ValueError("annotated term {} has no entry in terms".format(row["hpo_id"]))

    for key, block in bundle["methods"].items():
        for mark in block.get("marks") or ():
            _check_placement(mark, segments, where="methods[{}]".format(key))
            if mark.get("outcome") not in OUTCOMES:
                raise ValueError(
                    "methods[{}] mark {} has outcome {!r}, not one of {}".format(
                        key, mark.get("hpo_id"), mark.get("outcome"), OUTCOMES))
            if mark["hpo_id"] not in terms:
                raise ValueError(
                    "methods[{}] draws {} with no entry in terms".format(key, mark["hpo_id"]))


def _check_placement(row: dict, segments: list, where: str) -> None:
    if not row.get("hpo_id"):
        raise ValueError("{}: a row carries no hpo_id".format(where))
    idx, span = row.get("segment_idx"), row.get("span")
    if idx is None:
        if span is not None:
            raise ValueError(
                "{}: {} carries a span with no segment".format(where, row["hpo_id"]))
        return
    if not 0 <= idx < len(segments):
        raise ValueError("{}: {} names segment {} of {}".format(
            where, row["hpo_id"], idx, len(segments)))
    if span is None:
        return
    start, end = span
    length = len(segments[idx]["text"])
    if not 0 <= start < end <= length:
        raise ValueError("{}: {} spans {} of a {}-char segment {}".format(
            where, row["hpo_id"], span, length, idx))
