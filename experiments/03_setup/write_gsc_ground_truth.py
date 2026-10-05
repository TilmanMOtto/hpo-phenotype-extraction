#!/usr/bin/env python
"""Write the ground truth of one GSC+ evaluation frame as a two-column CSV.

    python experiments/03_setup/write_gsc_ground_truth.py --frame gsc_raghpo_ann --out <file.csv>

Frames: ``gsc_raghpo_ann`` (GSC+ (114), the RAG-HPO authors' re-annotation), ``gsc_raghpo`` (the
same 114 documents under the corpus annotation) and ``gsc_2024_eval_206`` (GSC+ (206), AutoPCR's
evaluation split under GSC-2024). The ground truth is read with
``hpo_extraction.evaluation.result_tables.loaders.load_gold``, the call the comparison makes, so
this file and the comparison's denominator have one definition. The PhenoJury and TreePhenoRAG
protocols read these files for their GSC+ runs.

The checks and the output format are those of the cluster scripts the thesis runs used.
"""
from __future__ import annotations

import argparse
import csv

from hpo_extraction.evaluation.result_tables import loaders

FRAMES = ("gsc_raghpo_ann", "gsc_raghpo", "gsc_2024_eval_206")


def main() -> int:
    """Write the ground truth of one GSC+ frame. Returns the exit status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", required=True, choices=FRAMES)
    ap.add_argument("--out", required=True, help="CSV to write (patient_id, hpo_codes)")
    ap.add_argument("--gsc_dir", default="resources/data/GSC_2024")
    ap.add_argument("--raghpo_dir", default="resources/data/GSC_RAGHPO")
    args = ap.parse_args()

    if args.frame == "gsc_2024_eval_206":
        gold = loaders.load_gold(args.frame, gsc_dir=args.gsc_dir)
        if len(gold) != 206:
            raise SystemExit(f"expected 206 documents, got {len(gold)}: the vendored split is not "
                             f"AutoPCR's evaluation frame")
    else:
        gold = loaders.load_gold(args.frame, raghpo_dir=args.raghpo_dir, gsc_dir=args.gsc_dir)
    empty = [d for d, v in gold.items() if not v]
    if empty:
        raise SystemExit(f"{len(empty)} of {len(gold)} documents carry no annotated term "
                         f"(e.g. {empty[0]}); scoring those under the empty-set convention would "
                         f"credit every method for finding nothing. Refusing.")
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "hpo_codes"])
        for doc in sorted(gold):
            writer.writerow([doc, ";".join(sorted(gold[doc]))])
    print(f"{len(gold)} document(s), {sum(len(v) for v in gold.values())} annotated pair(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
