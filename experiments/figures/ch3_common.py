"""Shared loading and labels for the chapter-3 (experimental setup) artifacts.

Chapter 3 reads one file, ``output/dataset_statistics/dataset_statistics.csv``, written by
``experiments/03_setup/dataset_statistics.py``: one row per (cohort, statistic), the ground truth counted as the
scoring harness counts it. Nothing is recomputed here. The only arithmetic is formatting.

The file is filled from two machines -- HCY on the cluster (``slurm/dataset_statistics.sbatch``),
the GSC+ columns anywhere -- so a cohort can be missing while the others are present. That is a
partial table with its missing column stated, not a skipped one.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_REPO = os.path.dirname(os.path.dirname(_HERE))

# Generic LaTeX plumbing, reused rather than re-implemented.
from ch5_common import tex_escape, write_table_tex  # noqa: E402,F401
from hpo_extraction.paths import results_dir, thesis_dir  # noqa: E402

DEFAULT_RESULTS = os.path.join(str(results_dir()), "dataset_statistics")
DEFAULT_TEXDIR = os.path.join(str(thesis_dir()), "thesis_figures_latex")

#: Same as ch5_common / ch6_common, so the drivers treat a skip identically.
MISSING_SOURCE_EXIT = 3
MISSING_TEX = "---"

#: The table's columns, in order. The keys are the comparison's cohort names.
#: GSC+ is reported in two frames, each the one a published baseline can be read against:
#: RAG-HPO's 114 re-annotated abstracts and AutoPCR's 206-abstract evaluation split. The full
#: 228-document corpus is where the runs execute, not a reported frame.
COHORTS = ("hcy", "gsc_raghpo_ann", "gsc_2024_eval_206")
COHORT_HEADER = {
    "hcy": "HCY",
    "gsc_raghpo_ann": "GSC+ (114)",
    "gsc_2024_eval_206": "GSC+ (206)",
}
#: For running text inside a table note, where the header's parentheses would nest.
COHORT_SHORT = {"hcy": "HCY", "gsc_raghpo_ann": "RAG-HPO subset",
                "gsc_2024_eval_206": "AutoPCR split"}

#: What each cohort's command is, for the note a partial table carries.
PRODUCER = {
    "hcy": "sbatch slurm/dataset_statistics.sbatch",
    "gsc_raghpo_ann": "python scripts/dataset_statistics.py --cohorts gsc_raghpo_ann "
                      "gsc_2024_eval_206",
    "gsc_2024_eval_206": "python scripts/dataset_statistics.py --cohorts gsc_raghpo_ann "
                         "gsc_2024_eval_206",
}


def load(results_dir: str) -> dict[str, dict[str, float]]:
    """``{cohort: {statistic: value}}``, or exit 3 when the file does not exist at all."""
    path = os.path.join(results_dir, "dataset_statistics.csv")
    if not os.path.exists(path):
        print(f"  skipped: no {path} -- run scripts/dataset_statistics.py", file=sys.stderr)
        raise SystemExit(MISSING_SOURCE_EXIT)
    df = pd.read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for row in df.itertuples():
        out.setdefault(row.cohort, {})[row.statistic] = float(row.value)
    return out


def count(v) -> str:
    """An integer count with a LaTeX thin thousands separator: ``1{,}551``."""
    if v is None:
        return MISSING_TEX
    return f"{int(round(v)):,}".replace(",", "{,}")


def num(v, nd=1) -> str:
    """A quantile or mean: no decimals when it is whole, else ``nd``."""
    if v is None:
        return MISSING_TEX
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.{nd}f}"
