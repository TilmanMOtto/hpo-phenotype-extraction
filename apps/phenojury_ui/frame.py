"""The HCY deep-dive sampling frame, finding it, reading it, and selecting reports with it.

``apps/compare_ui/select_hcy_documents.py`` draws 20 HCY reports into seven cells and writes the draw, the
pools it drew from and the digests of its inputs into one directory. This module is the reader.

Two properties of that file decide the whole design here:

**It is a pre-registration, not a shortlist.** The draw is written before anything is read, from a
recorded seed, so a report that turns out to be interesting cannot be added afterwards without the
diff showing it. Nothing in this app may therefore *edit* the frame, it is read-only input, and the
app's job is to make the recorded draw workable, not to curate it.

**The sample is purposive, so no aggregate over it means anything.** Cells are defined on the very
outcome a reader would want to measure (``pb_best`` / ``pb_worst`` are PhenoBERT's own per-report
score) and on annotation qualifiers. Any rate computed over these 20 reports is a rate over the
selection rule. The frame says so in its own prose; :data:`WARNING` carries that sentence into the
UI, and the subset banner is the reason this module exposes it as a constant rather than leaving it
to whichever view remembered.

Everything returns ``None`` / ``[]``, not raising, the same contract :mod:`loaders` follows:
a missing or unreadable frame must cost the subset control and nothing else.
"""

from __future__ import annotations

import csv
import json
import logging
import os

logger = logging.getLogger(__name__)

#: The directory ``select_hcy_deepdive.py --out`` writes, and the two files read from it.
FRAME_DIRNAME = "hcy_deepdive_frame"
FRAME_FILE = "frame.json"
POOLS_FILE = "pools.csv"

#: How far up from *output_base* to look for the frame directory. Matches
#: ``pbstandalone._MAX_WALK_UP`` and covers every layout ``loaders.find_runs`` accepts, the frame
#: sits beside the experiment directories, so from a cohort or a prompt cell that is three levels.
_MAX_WALK_UP = 4

#: The subset value meaning "no filter". Not ``None``, because it round-trips through a
#: ``dcc.Store`` and a dropdown, and ``None`` there is indistinguishable from "never set".
ALL = "all"

#: The subset value meaning "every drawn report, all cells together".
SAMPLE = "__sample__"

#: The sentence the banner shows whenever a subset is active. It is this module's constant rather
#: than a view's string because it is a claim about the *data*, and the moment two views own two
#: copies is the moment one of them stops matching what the frame actually says.
WARNING = (
    "Purposive sample, cells are defined on PhenoBERT's own per-report score and on annotation "
    "qualifiers, so any rate computed here is a rate over the selection rule, not over the cohort. "
    "No aggregate metric on this screen may be quoted. Mechanisms travel; numbers stay on the full "
    "cohort."
)

#: Columns of ``pools.csv`` worth showing on a report's hover. The four qualifiers last, in the
#: order ``select_hcy_deepdive.LABEL_CELLS`` defines them.
_POOL_NUMERIC = ("n_gold", "n_pred", "tp", "fp", "fn", "precision", "recall", "f1",
                 "family", "lab_value", "implicit", "negated")


# ──────────────────────────────────────────────────────────────────────────────
# discovery
# ──────────────────────────────────────────────────────────────────────────────
def is_frame_dir(path: str) -> bool:
    """True if *path* holds a ``frame.json``. The pools CSV is optional. The draw is not."""
    return bool(path) and os.path.isfile(os.path.join(path, FRAME_FILE))


def find_frame(output_base: str, override: str | None = None) -> str | None:
    """The frame directory for *output_base*, or ``None``.

    With *override* set, **only** the override is considered, the same rule
    ``pbstandalone.find_run`` follows, and for the same reason: a reader who typed a path wants that
    path, and silently falling back to a derived one would filter their screen by somebody else's
    draw. Both ``<override>`` and ``<override>/hcy_deepdive_frame`` are accepted, so pointing at the
    parent and at the frame directory itself both work.

    Without one, walk up from *output_base* looking for a ``hcy_deepdive_frame/`` sibling. The
    script's default ``--out`` puts it beside the experiment directories, which is the cluster
    layout, so on real data it simply appears.
    """
    if override:
        for candidate in _override_candidates(override):
            if is_frame_dir(candidate):
                return candidate
        return None

    for candidate in searched_paths(output_base):
        if is_frame_dir(candidate):
            return candidate
    return None


def _override_candidates(override: str) -> list[str]:
    return [override, os.path.join(override, FRAME_DIRNAME)]


def searched_paths(output_base: str, override: str | None = None) -> list[str]:
    """Where :func:`find_frame` would have looked, shown when it found nothing.

    Naming the paths is the difference between "no frame" and "no frame *here*", and only the
    second is something a reader can act on.
    """
    if override:
        return _override_candidates(override)
    if not output_base:
        return []
    paths = [output_base, os.path.join(output_base, FRAME_DIRNAME)]
    current = os.path.abspath(output_base)
    for _ in range(_MAX_WALK_UP):
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent
        paths.append(os.path.join(current, FRAME_DIRNAME))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# reading
# ──────────────────────────────────────────────────────────────────────────────
def load(frame_dir: str | None) -> dict | None:
    """Read a frame directory, or ``None`` when there is nothing readable there.

    Returns ``{"dir", "generated", "params", "inputs", "cells", "picks", "selected", "cell_of",
    "pools", "scored"}``. ``cell_of`` is the inversion the UI actually wants; ``scored`` is
    ``pools.csv`` keyed by report id, or ``{}`` when that file is absent, the cells still work
    without it, they just lose the numbers on the hover.
    """
    if not is_frame_dir(frame_dir or ""):
        return None
    path = os.path.join(frame_dir, FRAME_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("%s is not a frame object", path)
        return None

    picks = {cell: [str(r) for r in ids or ()]
             for cell, ids in (payload.get("picks") or {}).items()}
    # The frame's own key order is the cell order the pre-registration lists them in. Keep it, so
    # The dropdown reads like the document, not alphabetically.
    cells = list(picks)
    selected = [str(r) for r in (payload.get("selected") or ())]
    if not selected:
        selected = sorted({r for ids in picks.values() for r in ids})

    return {
        "dir": frame_dir,
        "generated": payload.get("generated", ""),
        "params": payload.get("params") or {},
        "inputs": payload.get("inputs") or {},
        "cell_sizes": payload.get("cell_sizes") or {},
        "cells": cells,
        "picks": picks,
        "pools": {cell: [str(r) for r in ids or ()]
                  for cell, ids in (payload.get("pools") or {}).items()},
        "selected": selected,
        "cell_of": {report_id: cell for cell, ids in picks.items() for report_id in ids},
        "scored": _load_pools(frame_dir),
    }


def _load_pools(frame_dir: str) -> dict:
    """``pools.csv`` keyed by report id, numbers coerced. ``{}`` when it is absent or unreadable."""
    path = os.path.join(frame_dir, POOLS_FILE)
    if not os.path.isfile(path):
        return {}
    rows: dict[str, dict] = {}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                report_id = str(row.get("patient_id", "")).strip()
                if not report_id:
                    continue
                rows[report_id] = {key: _number(row.get(key)) for key in _POOL_NUMERIC}
                rows[report_id]["cell"] = row.get("cell", "")
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}
    return rows


def _number(value):
    """``float`` where it parses, ``None`` where the cell is blank, never a silent ``0``.

    ``select_hcy_deepdive.score`` writes an empty precision/recall/F1 for a report with no ground truth,
    because those are undefined, not zero. Reading a blank as 0.0 would put an unannotated
    report at the bottom of a ranking it does not belong in at all.
    """
    text = (value or "").strip() if isinstance(value, str) else value
    if text in (None, ""):
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


# ──────────────────────────────────────────────────────────────────────────────
# selecting
# ──────────────────────────────────────────────────────────────────────────────
def applies_to(frame: dict | None, report_ids) -> bool:
    """True when this frame names at least one report of *report_ids*.

    The frame is drawn on HCY. Opening a GSC+ cohort with it loaded is normal and must not offer a
    filter that would empty every screen, so the control asks this first.
    """
    if not frame:
        return False
    return bool(set(frame["selected"]) & {str(r) for r in report_ids})


def cells_present(frame: dict | None, report_ids) -> list[str]:
    """The cells with at least one report in *report_ids*, in the frame's own order."""
    if not frame:
        return []
    available = {str(r) for r in report_ids}
    return [cell for cell in frame["cells"]
            if any(report_id in available for report_id in frame["picks"][cell])]


def select(frame: dict | None, subset: str | None, report_ids: list[str]) -> list[str]:
    """*report_ids* filtered to *subset*, **in the caller's order**.

    Order is the bundle's, not the frame's: it is the driver's own report order, and every table in
    the app is read against it. Always intersected, not substituted, so a frame naming a
    report this cohort does not have cannot introduce one.
    """
    if not subset or subset == ALL or not frame:
        return list(report_ids)
    if subset == SAMPLE:
        wanted = set(frame["selected"])
    else:
        wanted = set(frame["picks"].get(subset) or ())
    return [report_id for report_id in report_ids if report_id in wanted]


def cell_of(frame: dict | None, report_id: str) -> str:
    """The cell *report_id* was drawn into, or ``""``, it may be in the cohort but not the draw."""
    if not frame:
        return ""
    return frame["cell_of"].get(str(report_id), "")


def describe_report(frame: dict | None, report_id: str) -> str:
    """The hover text for a report's cell badge: the cell, then the frame's own row for it."""
    cell = cell_of(frame, report_id)
    if not cell:
        return ""
    row = (frame.get("scored") or {}).get(str(report_id)) or {}
    lines = [f"Deep-dive cell: {cell}"]
    if row:
        gold, pred = row.get("n_gold"), row.get("n_pred")
        lines.append(f"gold {_show(gold)} · PhenoBERT predicted {_show(pred)}")
        lines.append(f"P {_show(row.get('precision'))} · R {_show(row.get('recall'))} · "
                     f"F1 {_show(row.get('f1'))}")
        qualifiers = [f"{name} {row[name]}" for name in
                      ("family", "lab_value", "implicit", "negated") if row.get(name)]
        if qualifiers:
            lines.append("annotations: " + ", ".join(qualifiers))
    return "\n".join(lines)


def _show(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def summary(frame: dict | None, report_ids) -> str:
    """One line naming the draw, for the sidebar. ``""`` when no frame is loaded."""
    if not frame:
        return ""
    n_here = len(set(frame["selected"]) & {str(r) for r in report_ids})
    seed = frame["params"].get("seed", "?")
    drawn = frame.get("generated") or "?"
    return (f"{len(frame['selected'])} reports over {len(frame['cells'])} cells · "
            f"seed {seed} · drawn {drawn} · {n_here} in this cohort")
