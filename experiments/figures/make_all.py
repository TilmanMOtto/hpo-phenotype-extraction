#!/usr/bin/env python
r"""Rebuild every thesis figure, table and inline number: the setup chapter and the three results chapters.

    python experiments/figures/make_all.py
    python experiments/figures/make_all.py --chapter 5      # just one chapter
    python experiments/figures/make_all.py --check-only     # audit, build nothing

Four per-chapter drivers do the work and are each runnable alone:

    make_all_ch3.py   Setup         <- output/dataset_statistics/
    make_all_ch4.py   TreePhenoRAG   <- output/treephenorag_protocol/tables/
    make_all_ch5.py   PhenoJury      <- output/phenojury_protocol/{hcy,gsc206}/tables/
    make_all_ch6.py   Comparison     <- output/comparison/tables/

Afterwards this script AUDITS the figure directories, and that audit is the reason it exists rather
than being a three-line shell loop. Renaming artifacts is how a figure directory rots: the old PDF
stays on disk, no script produces it any more, and it keeps compiling into the thesis for months
because LaTeX has no idea it is stale. So every built artifact must have a producing script, and
every producing script must have built something. An orphan is reported by name and sets the exit
code.

A missing SOURCE is not an orphan and does not fail the build: several specified artifacts are
waiting on experiment stages that have not run (see docs/thesis_map.md). Those scripts skip themselves
with a message, and the audit knows the difference between "nothing produced this" and "this
declined to run".
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thesis_scan  # noqa: E402
from hpo_extraction.paths import thesis_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
# The artifacts live inside the thesis subtree, not beside this folder: Overleaf compiles
# thesis/ as its project root and cannot reach out of it.
FIG_DIR = os.path.join(str(thesis_dir()), "thesis_figures")
TEX_DIR = os.path.join(str(thesis_dir()), "thesis_figures_latex")

DRIVERS = {3: "make_all_ch3.py", 4: "make_all_ch4.py", 5: "make_all_ch5.py", 6: "make_all_ch6.py"}

# Artifacts with no producing script in THIS folder, and why that is correct, not an
# oversight. Anything listed here must name what does produce it, or the exemption is just a
# silenced warning.
EXPECTED_ORPHANS: set[str] = set()
# tab_union (a concatenation of every tab_ch*.tex) was deleted on 2026-09-27: nothing \input it,
# it went stale between rebuilds (V0 2 415 calls, V3 31 228), and a second copy of every label is
# one \input away from a duplicate-label build. Do not bring it back.

STEM = re.compile(r"^(?:fig|tab|num)_ch\d_[a-z0-9_]+$")

# A colour written as a literal anywhere but palette.py. The thesis has one colour system, and a
# hex code typed into a figure script is how it stops being one: the value may even be right
# today, but nothing ties it to the rule it was meant to follow, so the next edit to the palette
# leaves it behind. Named greys and "white" are left alone. Every hue goes through palette.py.
HEX_LITERAL = re.compile(r"""["']#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?["']""")
PALETTE_MODULE = "palette.py"


def run(chapters) -> list[str]:
    """Run the drivers of *chapters* and return the names of those that failed."""
    failed = []
    for ch in chapters:
        name = DRIVERS[ch]
        # flush: this process's stdout is block-buffered when piped, while the child writes
        # straight through -- without it every banner lands after the output it introduces.
        print(f"\n{'=' * 62}\n=== chapter {ch}: {name}\n{'=' * 62}", flush=True)
        if subprocess.call([sys.executable, os.path.join(HERE, name)]) != 0:
            failed.append(name)
    return failed


def declared_stems() -> dict[str, str]:
    """Every artifact stem the scripts in this folder claim to write, mapped to its producer.

    Read out of the source, not by importing: importing runs the module-level work of four
    plotting scripts, and a source scan is both faster and immune to a script that cannot run
    because its CSV is missing -- which is when the audit counts most.
    """
    produced: dict[str, str] = {}
    # The scripts name their outputs three different ways -- a bare stem passed to a writer that
    # appends the suffix (chapter 4), a full filename in a module constant (the chapter-5 figures),
    # and a filename built inside an os.path.join (the chapter-5/6 tables) -- so the extension is
    # optional here and stripped afterwards. Matching only the bare form silently reported every
    # chapter-5 and chapter-6 artifact as an orphan.
    pattern = re.compile(
        r"""["']((?:tab|fig|num)_ch\d_[a-z0-9_]+?)(?:\.(?:pdf|png|tex|md))?["']""")
    for entry in sorted(os.listdir(HERE)):
        if not entry.endswith(".py") or entry.startswith("make_all"):
            continue
        with open(os.path.join(HERE, entry), encoding="utf-8") as fh:
            body = fh.read()
        for stem in pattern.findall(body):
            produced.setdefault(stem, entry)
    return produced


def colour_literals() -> list[str]:
    """Every hex colour literal in a figure script outside palette.py, as ``file:line: literal``."""
    found = []
    for entry in sorted(os.listdir(HERE)):
        if not entry.endswith(".py") or entry == PALETTE_MODULE:
            continue
        with open(os.path.join(HERE, entry), encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                for hit in HEX_LITERAL.findall(line):
                    found.append(f"{entry}:{n}: {hit}")
    return found


def audit() -> list[str]:
    """Problems in the output folders: orphans, names off the convention, pasted copies, stray colours."""
    problems = []
    produced = declared_stems()

    # 0. Colours that bypass the palette.
    for hit in colour_literals():
        problems.append(f"colour literal outside palette.py (use a palette name): {hit}")

    on_disk: dict[str, set[str]] = {}
    for directory, suffixes in ((FIG_DIR, (".pdf", ".png")), (TEX_DIR, (".tex", ".md"))):
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            stem, ext = os.path.splitext(entry)
            if ext not in suffixes:
                continue
            on_disk.setdefault(stem, set()).add(os.path.join(directory, entry))

    # 1. Anything on disk that no script claims to write.
    for stem, paths in sorted(on_disk.items()):
        if stem in produced or stem in EXPECTED_ORPHANS:
            continue
        where = ", ".join(os.path.relpath(p, _REPO) for p in sorted(paths))
        problems.append(f"orphan artifact (no script writes it): {where}")

    # 2. Anything named like an artifact but not following the chapter-and-function convention.
    for stem in sorted(on_disk):
        if stem in EXPECTED_ORPHANS:
            continue
        if not STEM.match(stem):
            problems.append(f"off-convention name (want {{tab,fig,num}}_ch<N>_<function>): {stem}")

    # 3. A float the thesis typesets by hand under a generated file's label: a pasted copy. It
    #    compiles, and it stops tracking its CSV the moment it is pasted.
    problems.extend(f"pasted copy of a generated float: {p}" for p in thesis_scan.pasted_copies())

    # 4. Built but not in the PDF. Often deliberate (a table kept for the findings files), so it
    #    is information -- but it is also what a forgotten \input looks like.
    wired = {stem for _, stem in thesis_scan.generated_inputs()}
    unwired = sorted(s for s in on_disk if s.startswith(("tab_", "fig_")) and s not in wired
                     and s in produced)
    if unwired:
        print("\nbuilt but not \\input by the thesis:")
        for stem in unwired:
            print(f"  {stem:<42} {produced[stem]}")

    # 5. A script that claims a stem nothing appeared for. Usually a skipped artifact waiting on a
    #    CSV, so it is reported as information, not as a problem.
    waiting = sorted(set(produced) - set(on_disk))
    if waiting:
        print("\nnot built this run (waiting on a source, see docs/thesis_map.md):")
        for stem in waiting:
            print(f"  {stem:<42} {produced[stem]}")

    print(f"\naudited {len(on_disk)} artifact(s) against {len(produced)} declared stem(s)")
    return problems


def main() -> int:
    """Build every chapter, then audit. Exits 0 when everything is built or skipped for a missing source."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chapter", type=int, choices=sorted(DRIVERS), action="append",
                    help="build only this chapter (repeatable); default is all of them")
    ap.add_argument("--check-only", action="store_true",
                    help="run the orphan audit without rebuilding anything")
    args = ap.parse_args()

    failed = [] if args.check_only else run(args.chapter or sorted(DRIVERS))
    problems = audit()

    print()
    for problem in problems:
        print(f"  {problem}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    if problems:
        print(f"{len(problems)} audit problem(s) -- delete the stale file, add the script that "
              f"writes it, or \\input the generated float in place of the pasted one")
    if not failed and not problems:
        print("all chapters built; no orphans, no off-convention names, no stray colours")
    return 1 if (failed or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
