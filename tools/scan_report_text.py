#!/usr/bin/env python
"""Check that no HCY report text is in the repository. Run on LeoMed, where the reports are.

    python tools/scan_report_text.py [--input_dir <hcy.input_dir>] [--n 8]

Builds the set of every run of *n* consecutive words (default 8) in the HCY reports, then looks for
those runs in every file ``git ls-files`` lists. A hit names the file, the line and how many runs
it shares with the reports, and never prints the matched text, so the output itself can leave the
cluster. Words are lower-cased and stripped of punctuation before matching. Eight words are
specific enough that a hit is report text and not a common phrase. Exit status 1 when anything is
found.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-zA-ZäöüÄÖÜß0-9]+")


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Every run of *n* consecutive words of *text*, lower-cased."""
    words = [w.lower() for w in WORD.findall(text)]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main() -> int:
    """Scan the tracked files. Exit status 1 when a file shares a run of words with a report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input_dir", default=None, help="HCY reports (default: hcy.input_dir of the path file)")
    ap.add_argument("--n", type=int, default=8, help="words per run (default 8)")
    ap.add_argument("--encoding", default="latin1", help="encoding of the reports (default latin1)")
    args = ap.parse_args()
    if args.input_dir is None:
        sys.path.insert(0, str(REPO / "src"))
        from hpo_extraction.paths import lookup
        args.input_dir = lookup("hcy.input_dir")
    reports = sorted(Path(args.input_dir).glob("*.txt"))
    if not reports:
        print(f"no .txt reports under {args.input_dir}", file=sys.stderr)
        return 2
    grams: set[tuple[str, ...]] = set()
    for path in reports:
        grams |= ngrams(path.read_text(encoding=args.encoding, errors="replace"), args.n)
    files = subprocess.run(["git", "ls-files"], cwd=REPO, check=True, capture_output=True,
                           text=True).stdout.splitlines()
    hits = 0
    for rel in files:
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n_line, line in enumerate(text.splitlines(), 1):
            shared = len(ngrams(line, args.n) & grams)
            if shared:
                hits += 1
                print(f"{rel}:{n_line}: shares {shared} run(s) of {args.n} words with a report")
    print(f"{len(reports)} reports, {len(grams):,} runs of {args.n} words, {len(files):,} files scanned, "
          f"{hits} line(s) with a match")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
