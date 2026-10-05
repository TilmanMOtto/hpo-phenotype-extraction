"""Convert a ``*_predictions.jsonl`` into the two-column prediction-set CSV.

Most earlier drivers write the predictions contract: per-term lines followed by one authoritative
``summary: true`` line per report. Two consumers want the *set* instead, ``phenojury_protocol``'s ``j0_predictions_path`` (T5.2's PhenoBERT-without-an-LM row) and anything else
reading through ``hpo_extraction.evaluation.prediction_sets``, because they re-run cached generations and cannot
run the method themselves. This is the adapter between the two, and nothing more.

    python experiments/06_comparison/predictions_to_sets.py \
        output/baseline_phenobert/hcy/phenobert_predictions.jsonl \
        output/baseline_phenobert/hcy/predictions.csv

Both halves are the repo's own: the summary line is read by
``result_tables/loaders.load_predictions`` and the file is written by
``hpo_extraction.evaluation.prediction_sets.write_prediction_sets``. Neither is reimplemented here, so a report
that predicted **nothing** survives the round trip as an empty cell rather than vanishing, which
is the only way this conversion can silently hand a method free precision.

``--restrict-to`` takes a ground truth or fold CSV and keeps only its reports, reporting any it could not
find. A cohort mismatch is the expected failure here: the PhenoBERT baseline annotates every report staged for
it, which need not be the cohort the ground truth defines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

from hpo_extraction.evaluation.prediction_sets import (  # noqa: E402
    read_prediction_sets, write_prediction_sets,
)
from hpo_extraction.evaluation.result_tables.loaders import load_predictions  # noqa: E402


def main() -> int:
    """Convert a ``*_predictions.jsonl`` into a prediction-set CSV. Returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a *_predictions.jsonl")
    parser.add_argument("destination", type=Path, help="the prediction-set CSV to write")
    parser.add_argument("--restrict-to", type=Path, default=None,
                        help="a gold/prediction-set CSV whose report ids define the cohort")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"no such file: {args.source}", file=sys.stderr)
        return 1

    predicted, _gold = load_predictions(args.source)
    if not predicted:
        print(f"{args.source} carries no summary lines — nothing to convert", file=sys.stderr)
        return 1

    n_source = len(predicted)
    if args.restrict_to is not None:
        wanted = set(read_prediction_sets(args.restrict_to))
        missing = sorted(wanted - set(predicted))
        extra = sorted(set(predicted) - wanted)
        predicted = {rid: predicted.get(rid, set()) for rid in sorted(wanted)}
        if missing:
            # Not fatal: an absent report scores as an empty prediction, which is the honest
            # reading. It is printed because it is also what a staging bug looks like.
            print(f"WARNING: {len(missing)} report(s) in {args.restrict_to.name} have no "
                  f"prediction and are written empty: {', '.join(missing[:8])}"
                  f"{' ...' if len(missing) > 8 else ''}", file=sys.stderr)
        if extra:
            print(f"dropped {len(extra)} report(s) outside the cohort", file=sys.stderr)

    write_prediction_sets(args.destination, predicted)
    n_empty = sum(1 for codes in predicted.values() if not codes)
    n_codes = sum(len(codes) for codes in predicted.values())
    print(f"{args.source.name}: {n_source} report(s) -> {args.destination} "
          f"({len(predicted)} report(s), {n_codes} code(s), {n_empty} empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
