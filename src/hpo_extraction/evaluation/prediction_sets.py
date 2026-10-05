"""Per-report prediction sets, written as the two-column CSV this repo's ground truth files already use.

Most earlier drivers emit ``*_predictions.jsonl`` and the scorers read it through
``result_tables/loaders.load_predictions``. Two methods cannot: ``treephenorag_protocol`` and ``phenojury_protocol`` select
their configuration **inside nested cross-validation**, so what they predict for a report depends
on which fold that report fell in, and the thing worth keeping is the *pooled out-of-fold* set
rather than any single fold's output. Neither driver runs inference at all, both re-run a cache, so there is no per-term record to attach the set to.

Hence this format: ``patient_id,hpo_codes`` with the codes semicolon-joined, identical to
``hcy_ground_truth_curated.csv`` and to ``curated_gold.write_gold_csv``'s output. A pooled
prediction set written here is read back by anything that reads a ground truth file, which is what lets
``comparison`` put a protocol-selected method in the same table as a fixed one.

**A report that was predicted empty gets a line with an empty cell.** Dropping it would hand the
method free precision on every report it said nothing about, the same trap
``loaders.load_predictions`` avoids by trusting the ``summary`` line over the per-term lines.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Mapping

#: Column names, matching the ground truth files byte for byte so one reader serves both.
ID_COLUMN = "patient_id"
CODE_COLUMN = "hpo_codes"

#: Separator inside the code cell. ``HCYDataset.load_ground_truth`` accepts ``;`` or ``,``. We
#: always write ``;`` because a comma would need the cell quoted to survive the CSV dialect.
CODE_SEPARATOR = ";"


def write_prediction_sets(path: str | Path, sets: Mapping[str, Iterable[str]]) -> Path:
    """Write ``{report_id: codes}`` to *path*, atomically. Returns the path.

    Atomic because these files are read by a later stage in the same job on the cluster, and a
    half-written file is indistinguishable from a method that predicted nothing for the tail of
    the cohort.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[ID_COLUMN, CODE_COLUMN])
        writer.writeheader()
        for report_id in sorted(sets):
            writer.writerow({
                ID_COLUMN: report_id,
                CODE_COLUMN: CODE_SEPARATOR.join(sorted(sets[report_id])),
            })
    os.replace(tmp, path)
    return path


def read_prediction_sets(path: str | Path) -> dict[str, set[str]]:
    """``{report_id: set of HPO ids}``, the inverse of :func:`write_prediction_sets`.

    Accepts either separator, and an empty cell yields an empty set, not a set containing
    the empty string.
    """
    out: dict[str, set[str]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return out
        id_column = reader.fieldnames[0]
        code_column = reader.fieldnames[1] if len(reader.fieldnames) > 1 else CODE_COLUMN
        for row in reader:
            cell = (row.get(code_column) or "").strip()
            codes = {c.strip() for part in cell.split(CODE_SEPARATOR) for c in part.split(",")}
            out[str(row[id_column]).strip()] = {c for c in codes if c}
    return out
