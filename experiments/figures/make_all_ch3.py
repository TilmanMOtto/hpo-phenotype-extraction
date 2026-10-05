#!/usr/bin/env python
"""Rebuild the chapter-3 (experimental setup) table and inline numbers.

    python experiments/figures/make_all_ch3.py

Reads ``output/dataset_statistics/dataset_statistics.csv`` (``experiments/03_setup/dataset_statistics.py``. The HCY column comes from ``slurm/dataset_statistics.sbatch``) and writes LaTeX into
thesis/thesis_figures_latex/. Each script is runnable alone with the same flags.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch3_common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["tables_ch3.py", "num_ch3_inline.py", "fig_ch3_hpo_fragment.py",
           "fig_ch3_protocol.py"]


def main():
    """Build the chapter 3 artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--texdir", default=None)
    args = ap.parse_args()

    failed, skipped = [], []
    for name in SCRIPTS:
        cmd = [sys.executable, os.path.join(HERE, name)]
        if args.results:
            cmd += ["--results", args.results]
        if args.texdir:
            cmd += ["--texdir", args.texdir]
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.call(cmd)
        # 3: the source CSV does not exist yet -- an ordinary state of the world, not a crash.
        if rc == C.MISSING_SOURCE_EXIT:
            skipped.append(name)
        elif rc != 0:
            failed.append(name)

    print()
    if skipped:
        print("skipped (waiting on a source): " + ", ".join(skipped))
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all {len(SCRIPTS) - len(skipped)} of {len(SCRIPTS)} script(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
