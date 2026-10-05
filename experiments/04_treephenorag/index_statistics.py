#!/usr/bin/env python
r"""Size of the synthetic-sentence index: the numbers behind app:tpr-generation.

    python experiments/04_treephenorag/index_statistics.py

Counts the stored sentences per term as the index loader parses them
(``hpo_extraction.data.loading``: latin1, ``str.splitlines``, bullets stripped, empty lines dropped) --
latin1 splits on byte 0x85 inside UTF-8 characters, so a plain line count disagrees. Restricted to
the terms below $v_0$, the ones the traversal scores. Writes
``output/synthetic_sentence_statistics/index_statistics.csv`` (statistic, value, definition).

The files are generated from the ontology alone and tracked in the repo, so this runs anywhere.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402
for _p in (_REPO / "experiments" / "03_setup",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CONTEXT = _REPO / "resources/synthetic_sentences/llama-3.3-70b-instruct"


def main() -> int:
    """Count the synthetic sentences per term and write ``index_statistics.csv``."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context_dir", default=str(DEFAULT_CONTEXT))
    ap.add_argument("--hpo_json_path", default=str(_REPO / "resources/util/hpo.json"))
    ap.add_argument("--output_dir", default=str(_REPO / "output" / "synthetic_sentence_statistics"))
    args = ap.parse_args()

    from hpo_extraction.data.loading import _BULLET
    from dataset_statistics import load_view

    view = load_view(args.hpo_json_path)
    below = {n for n in view.tree.depth_dict if view.depth(n) is not None}

    per_term = {}
    for fname in os.listdir(args.context_dir):
        if not fname.endswith(".txt"):
            continue
        term = fname[:-4].replace("_", ":")
        if view.depth(term) is None:
            continue
        with open(os.path.join(args.context_dir, fname), encoding="latin1") as fh:
            raw = fh.read()
        per_term[term] = sum(1 for line in raw.splitlines() if _BULLET.sub("", line.strip()))

    counts = sorted(per_term.values())
    q1, median, q3 = statistics.quantiles(counts, n=4, method="inclusive")
    stats = {
        "n_terms_with_file": (len(counts), "terms below v0 with a sentence file"),
        "n_sentences": (sum(counts), "sentences over those terms, as the loader parses them"),
        "sentences_median": (median, "sentences per term, median"),
        "sentences_q1": (q1, "25th percentile of the same"),
        "sentences_q3": (q3, "75th percentile of the same"),
        "sentences_min": (counts[0], "fewest sentences for one term"),
        "sentences_max": (counts[-1], "most sentences for one term"),
        "n_terms_below_v0": (len(below), "terms with a depth below v0 in the ontology file"),
        "n_terms_without_file": (len(below - set(per_term)),
                                 "terms below v0 with no sentence file (fall back to the label)"),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index_statistics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["statistic", "value", "definition"])
        for key, (value, definition) in stats.items():
            writer.writerow([key, value, definition])
    print(f"{len(counts)} terms, {sum(counts)} sentences, median {median:g} "
          f"[{q1:g}, {q3:g}] -> {out / 'index_statistics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
