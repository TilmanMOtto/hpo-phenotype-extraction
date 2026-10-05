"""Write the nested cross-validation fold assignment both results chapters read.

Selection is part of every method here, TreePhenoRAG picks a pooling rule and two thresholds,
PhenoJury picks a jury, a rule and a threshold, so a score measured on the reports the selection
saw is optimistically biased. Nested CV answers that, and it only means anything if both
architectures use the *same* partition: otherwise a difference between them is partly a difference
between two random splits.

So the assignment is written once, here, and read from a file thereafter.

**The file is not committed, and that is deliberate.** It names patient ids and carries a per-patient
stratum derived from gold-set size, which makes it HCY patient data, and this repository keeps
patient data out of the tree (``docs/cluster.md``).
What *is* committed is this script and its seed, which is the whole reproducibility claim: the file
is a pure function of ``(gold file, seed)``, and :func:`hpo_extraction.evaluation.stats.folds.write_folds` is fixed
by a byte-identity test. Regenerate rather than copy.

Stratification is by tertile of gold-set size. The cohort's other natural axis, biochemical
presentation, 44 isolated against 74 combined, exists only as an aggregate in
``context/HCY_kispi Dataset.md``. There is no per-patient column anywhere in the repository, so it
cannot be used, and the thesis says so, not implying a stratification it did not perform.

Usage::

    python experiments/03_setup/make_folds.py \\
        --ground truth <hcy curated ground-truth dir>/hcy_ground_truth_curated.csv \\
        --out  <hcy curated ground-truth dir>/folds_hcy.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


from hpo_extraction.evaluation.stats import nested_folds, read_folds, size_tertiles, write_folds  # noqa: E402


def load_gold_sizes(path: Path) -> dict[str, int]:
    """``{patient_id: n_gold_terms}`` from the two-column ground truth export."""
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            codes = [c.strip() for c in (rec.get("hpo_codes") or "").split(";") if c.strip()]
            sizes[rec["patient_id"]] = len(codes)
    return sizes


def main() -> int:
    """Write the nested cross-validation fold file for a ground truth (5 outer folds, 10 repetitions, seed 0)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k-outer", type=int, default=5)
    ap.add_argument("--k-inner", type=int, default=5)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sizes = load_gold_sizes(args.gold)
    report_ids = sorted(sizes)
    strata = size_tertiles(sizes)

    print(f"gold      {args.gold}")
    print(f"          sha256:{hashlib.sha256(args.gold.read_bytes()).hexdigest()[:16]}")
    print(f"reports   {len(report_ids)}  ({sum(sizes.values())} gold pairs)")

    by_stratum = Counter(strata.values())
    for s in sorted(by_stratum):
        members = [r for r in report_ids if strata[r] == s]
        lo, hi = min(sizes[r] for r in members), max(sizes[r] for r in members)
        print(f"  stratum {s}: {by_stratum[s]:>3} reports, gold size {lo}-{hi}")

    rows = nested_folds(report_ids, strata, k_outer=args.k_outer, k_inner=args.k_inner,
                        repetitions=args.repetitions, seed=args.seed)
    path = write_folds(args.out, rows, strata)

    # Read it straight back: the file, not the in-memory structure, is what the chapters consume.
    restored = read_folds(path)
    assert len(restored) == len(rows)
    for a, b in zip(rows, restored):
        assert a["eval_ids"] == b["eval_ids"] and a["inner_folds"] == b["inner_folds"]

    eval_sizes = sorted({len(r["eval_ids"]) for r in rows})
    print(f"\nwrote     {path}")
    print(f"          {len(rows)} (repetition, outer fold) rows; eval folds of {eval_sizes} reports")
    print(f"          sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")

    meta = path.with_suffix(".json")
    meta.write_text(json.dumps({
        "generated_by": "scripts/make_folds.py",
        "gold": str(args.gold),
        "gold_sha256_16": hashlib.sha256(args.gold.read_bytes()).hexdigest()[:16],
        "n_reports": len(report_ids),
        "k_outer": args.k_outer, "k_inner": args.k_inner,
        "repetitions": args.repetitions, "seed": args.seed,
        "stratified_by": "tertile of gold-set size",
        "stratum_sizes": {str(k): v for k, v in sorted(by_stratum.items())},
        "folds_sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }, indent=2), encoding="utf-8")
    print(f"          manifest {meta.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
