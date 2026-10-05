r"""Stratified, repeated, nested cross-validation folds over reports, written once, read by both.

Both architectures select things: TreePhenoRAG picks a pooling rule and two thresholds, PhenoJury
picks a jury, a vote rule and a threshold. Evaluating either on the reports its selection saw is
optimistically biased, and the bias can be as large as the differences being reported. Nested CV
answers that: the **outer** split measures, the **inner** split selects, and no report is ever used
for both at once.

Four design points, each a correctness requirement rather than a preference:

**Reports are the unit.** Cells within a report are correlated, so splitting on cells would leak a
patient across the boundary.

**Stratified by gold-set size.** With 118 patients and ground-truth sets ranging from none to several dozen
terms, an unstratified split can hand one fold most of the heavily annotated patients, and micro
:math:`F_1` is dominated by those. Stratifying on the tertile of gold-set size keeps the
folds comparable. The biochemical presentation would be the other natural axis (the cohort divides
44 isolated / 74 combined), but that label exists only as an aggregate in the dataset
documentation, never per patient, so it cannot be used here, and the thesis says so, not
implying a stratification it did not perform.

**Repetitions, not a single split.** One 5-fold split of 118 patients is itself a random draw, and
the spread across draws is reportable information. Ten repetitions with different shuffles give it.

**The assignment is a file.** :func:`write_folds` and :func:`read_folds` round-trip to CSV, and that
file is committed. "Both architectures used the same folds" is then something a reader can check,
not something the text asserts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def size_tertiles(gold_sizes: Mapping[str, int]) -> dict[str, int]:
    """``{report_id: 0|1|2}`` by tertile of gold-set size, ties broken deterministically.

    Ranked, not cut at the empirical 33rd/67th percentiles: with many reports sharing a size, a dozen patients with five annotated terms is common, percentile cuts produce wildly
    uneven strata, whereas splitting the *rank order* into three contiguous blocks cannot. Reports
    are ordered by ``(size, report_id)``, so the result never depends on dict ordering.
    """
    ordered = sorted(gold_sizes, key=lambda r: (gold_sizes[r], r))
    n = len(ordered)
    if n == 0:
        return {}
    return {rid: min(2, (position * 3) // n) for position, rid in enumerate(ordered)}


def stratified_report_folds(
    report_ids: Sequence[str],
    strata: Mapping[str, int] | None = None,
    k: int = 5,
    seed: int = 0,
) -> list[list[str]]:
    """``k`` disjoint folds covering every report once.

    Dealt round-robin *within* each stratum after shuffling it, which keeps both the fold sizes and
    the stratum shares balanced to within one report. Returns the folds themselves, not
    train/eval pairs, because nested CV needs to re-partition them.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    ids = list(report_ids)
    if len(ids) < k:
        raise ValueError(f"cannot make {k} folds from {len(ids)} reports")

    strata = dict(strata) if strata is not None else {rid: 0 for rid in ids}
    missing = [rid for rid in ids if rid not in strata]
    if missing:
        raise ValueError(f"{len(missing)} report(s) have no stratum, e.g. {missing[:3]}")

    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    # The deal is offset per stratum so fold 0 does not collect the first member of every stratum
    # at once, which would bias it toward one end of each stratum's shuffle.
    offset = 0
    for stratum in sorted({strata[rid] for rid in ids}):
        members = sorted(rid for rid in ids if strata[rid] == stratum)
        for position, index in enumerate(rng.permutation(len(members))):
            folds[(position + offset) % k].append(members[index])
        offset += len(members)
    return [sorted(f) for f in folds]


def nested_folds(
    report_ids: Sequence[str],
    strata: Mapping[str, int] | None = None,
    k_outer: int = 5,
    k_inner: int = 5,
    repetitions: int = 10,
    seed: int = 0,
) -> list[dict]:
    """One row per ``(repetition, outer fold)``, each carrying its own inner folds.

    Each row is ``{repetition, outer_fold, eval_ids, train_ids, inner_folds}``. ``eval_ids`` is
    scored and never selected on; ``inner_folds`` partitions ``train_ids`` for the selection step.
    Every repetition reshuffles from ``seed + repetition``, so the whole structure, all
    ``repetitions x k_outer`` rows and their inner partitions, is reproducible from one integer.
    """
    ids = list(report_ids)
    rows: list[dict] = []
    for rep in range(repetitions):
        outer = stratified_report_folds(ids, strata, k=k_outer, seed=seed + rep)
        for f, eval_ids in enumerate(outer):
            train_ids = sorted(set(ids) - set(eval_ids))
            inner = stratified_report_folds(
                train_ids, strata, k=k_inner, seed=seed + 1000 * (rep + 1) + f
            )
            rows.append({
                "repetition": rep,
                "outer_fold": f,
                "eval_ids": eval_ids,
                "train_ids": train_ids,
                "inner_folds": inner,
            })
    return rows


FOLD_COLUMNS = ("repetition", "outer_fold", "inner_fold", "report_id", "stratum", "role")


def write_folds(path: str | Path, rows: Iterable[Mapping], strata: Mapping[str, int]) -> Path:
    """Flatten :func:`nested_folds` to one row per (repetition, outer fold, report) and write it.

    ``role`` is ``eval`` for an outer-fold evaluation report and ``train`` otherwise; ``inner_fold``
    is the report's inner fold index, or ``-1`` for evaluation reports, which have none.

    Every report appears once per outer fold, as ``eval`` in the one fold that evaluates it and as
    ``train`` in the other ``k_outer - 1``, so the file is ``n_reports x k_outer x repetitions``
    rows: 5 900 for the shipped 118 x 5 x 10. That redundancy is the point. Each line is a complete
    statement of one report's role in one fold, checkable with a single ``groupby`` and without
    reconstructing anything.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FOLD_COLUMNS))
        writer.writeheader()
        for row in rows:
            for rid in row["eval_ids"]:
                writer.writerow({
                    "repetition": row["repetition"], "outer_fold": row["outer_fold"],
                    "inner_fold": -1, "report_id": rid,
                    "stratum": strata.get(rid, 0), "role": "eval",
                })
            for i, fold in enumerate(row["inner_folds"]):
                for rid in fold:
                    writer.writerow({
                        "repetition": row["repetition"], "outer_fold": row["outer_fold"],
                        "inner_fold": i, "report_id": rid,
                        "stratum": strata.get(rid, 0), "role": "train",
                    })
    return path


def read_folds(path: str | Path) -> list[dict]:
    """The inverse of :func:`write_folds`, returning :func:`nested_folds`' own structure."""
    grouped: dict[tuple[int, int], dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            key = (int(rec["repetition"]), int(rec["outer_fold"]))
            row = grouped.setdefault(key, {
                "repetition": key[0], "outer_fold": key[1],
                "eval_ids": [], "train_ids": [], "_inner": {},
            })
            if rec["role"] == "eval":
                row["eval_ids"].append(rec["report_id"])
            else:
                row["train_ids"].append(rec["report_id"])
                row["_inner"].setdefault(int(rec["inner_fold"]), []).append(rec["report_id"])

    out = []
    for key in sorted(grouped):
        row = grouped[key]
        inner = row.pop("_inner")
        row["eval_ids"] = sorted(row["eval_ids"])
        row["train_ids"] = sorted(row["train_ids"])
        row["inner_folds"] = [sorted(inner[i]) for i in sorted(inner)]
        out.append(row)
    return out
