#!/usr/bin/env python
"""Rebuild every final-comparison figure and table from the comparison output folder.

    python experiments/figures/make_all_ch6.py
    python experiments/figures/make_all_ch6.py --preview     # also write review PNGs

Writes PDFs into thesis/thesis_figures/ and LaTeX into thesis/thesis_figures_latex/. Each script
is runnable alone with the same flags. This is only a driver, so a failure names the script that
failed rather than being swallowed.

TreePhenoRAG's GSC+ cell is absent until the TreePhenoRAG protocol's transfer stage runs (queued behind the
the term-information score store GSC+ cache array). Every script here draws it as an explicit em dash and keeps going, so
this driver is the whole of what has to be re-run once that lands.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ch6_common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "fig_ch6_headline.py",
    "fig_ch6_pr_plane.py",
    "fig_ch6_subgroups.py",
    "fig_ch6_hierarchy.py",
    "fig_ch6_cost_quality.py",
    "tables_ch6.py",
]

TABLES_ONLY = {"tables_ch6.py"}


def main():
    """Build the chapter 6 artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None)
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--texdir", default=None)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    failed, skipped = [], []
    for name in SCRIPTS:
        cmd = [sys.executable, os.path.join(HERE, name)]
        if args.results:
            cmd += ["--results", args.results]
        if args.texdir:
            cmd += ["--texdir", args.texdir]
        if name not in TABLES_ONLY:
            if args.figdir:
                cmd += ["--figdir", args.figdir]
            if args.preview:
                cmd += ["--preview"]
        print(f"\n=== {name} ===")
        rc = subprocess.call(cmd)
        # 3 means the source CSV does not exist yet (ch6_common.MISSING_SOURCE_EXIT) -- an
        # ordinary state of the world, not a crash. Anything else non-zero still fails.
        if rc == C.MISSING_SOURCE_EXIT:
            skipped.append(name)
        elif rc != 0:
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
