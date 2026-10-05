#!/usr/bin/env python
"""Rebuild every chapter-5 (PhenoJury) figure, table and inline number from the PhenoJury protocol's output.

    python experiments/figures/make_all_ch5.py
    python experiments/figures/make_all_ch5.py --preview     # also write review PNGs

Writes PDFs into thesis/thesis_figures/ and LaTeX into thesis/thesis_figures_latex/. Each script
is runnable alone with the same flags. This is only a driver, so a failure names the script that
failed rather than being swallowed.

Each entry declares which flags its script accepts. That is not bureaucracy: the three kinds of
script here take genuinely different arguments -- a plotted figure takes all four, a table writer
has no figure directory, and the pipeline schematic is drawn from its own source and reads no
results directory -- and passing a flag a script does not define is an argparse error, not a no-op.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch5_common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> the flags that script accepts
SCRIPTS = [
    ("fig_ch5_pipeline.py", {"figdir", "texdir"}),
    ("fig_ch5_aggregation_curves.py", {"results", "figdir", "texdir", "preview"}),
    ("fig_ch5_recall_decomposition.py", {"results", "figdir", "texdir", "preview"}),
    ("fig_ch5_error_analysis.py", {"results", "figdir", "texdir", "preview"}),
    ("tables_ch5.py", {"results", "texdir"}),
    ("num_ch5_inline.py", {"results", "texdir"}),
]


def main():
    """Build the chapter 5 artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--texdir", default=None)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    failed, skipped = [], []
    for name, accepts in SCRIPTS:
        cmd = [sys.executable, os.path.join(HERE, name)]
        for flag in ("results", "figdir", "texdir"):
            value = getattr(args, flag)
            if value and flag in accepts:
                cmd += [f"--{flag}", value]
        if args.preview and "preview" in accepts:
            cmd += ["--preview"]
        print(f"\n=== {name} ===", flush=True)
        code = subprocess.call(cmd)
        # 3 means "the source CSV does not exist yet" (ch5_common.MISSING_SOURCE_EXIT), which is
        # An ordinary state of the world -- several specified artifacts wait on an experiment
        # stage. Anything else non-zero is a real crash and still fails the build.
        if code == C.MISSING_SOURCE_EXIT:
            skipped.append(name)
        elif code != 0:
            failed.append(name)

    print()
    if skipped:
        print("skipped (waiting on a source, see docs/thesis_map.md): " + ", ".join(skipped))
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all {len(SCRIPTS) - len(skipped)} of {len(SCRIPTS)} script(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
