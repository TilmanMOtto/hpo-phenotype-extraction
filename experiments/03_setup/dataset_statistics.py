#!/usr/bin/env python
r"""Dataset statistics after preprocessing: the numbers behind Table 3.1 (`tab:datasets`).

    # HCY -- on the cluster, where the curated ground truth and the reports live
    python experiments/03_setup/dataset_statistics.py --cohorts hcy \
        --hcy_curated_gt_path $HCY/curated_ground_truth_2026-09-12/hcy_ground_truth_curated.csv \
        --hcy_input_dir $HCY/input

    # GSC+ -- anywhere, the corpus is in the repo
    python experiments/03_setup/dataset_statistics.py --cohorts gsc_raghpo_ann gsc_2024_eval_206

Writes ``output/dataset_statistics/dataset_statistics.csv`` -- one row per (cohort, statistic) -- and
a ``manifest.json`` beside it. Rows for a cohort not named in ``--cohorts`` are **kept**, so the two
invocations above can run on different machines and fill one file between them. The figure script
``experiments/figures/tables_ch3.py`` turns it into the LaTeX table.

**The ground truth is counted as it is scored.** Each cohort is loaded through the result-table library's ``load_gold`` and
passed through ``thesis_metrics.normalise_pair`` -- the same two calls the comparison makes before any
metric -- so a "annotated pair" here is a unit a recall figure divides by: obsolete ids remapped,
out-of-subontology ids dropped (their number is kept in ``n_gold_out_of_subtree``), no specificity
reduction. The report list is the ground truth's, including reports with an empty ground-truth set.

Aggregates only. Nothing written here carries report text or a per-report row, so the output may
leave the cluster. The inputs may not.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402

#: The cohort keys are the comparison's, so the table's columns name the same frames its results do.
COHORTS = ("hcy", "gsc_raghpo_ann", "gsc_2024_eval_206")


def _paths(repo: Path) -> None:
    for p in (repo,):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_view(hpo_json: str | None):
    """An ``OntologyView`` over *hpo_json* (the repository's ontology file when None)."""
    from hpo_extraction.evaluation.metrics import OntologyView
    from hpo_extraction.ontology.hpo_tree import HPOTree

    tree = HPOTree(hpo_json) if hpo_json else HPOTree()
    tree.buildHPOTree()
    return OntologyView(tree)


def load_cohort(cohort: str, args) -> tuple[dict[str, set[str]], dict[str, str]]:
    """``(gold, reports)`` for one cohort, through the loaders the scoring harness uses."""
    import hpo_extraction.evaluation.result_tables.loaders as loaders

    if cohort == "hcy":
        if not (args.hcy_curated_gt_path and args.hcy_input_dir):
            raise SystemExit("cohort 'hcy' needs --hcy_curated_gt_path and --hcy_input_dir")
        gold = loaders.load_gold("hcy_curated", hcy_curated_gt_path=args.hcy_curated_gt_path)
        reports = loaders.load_reports("hcy_curated", input_dir=args.hcy_input_dir)
    else:
        gold = loaders.load_gold(cohort, gsc_dir=args.gsc_dir, raghpo_dir=args.raghpo_dir)
        reports = loaders.load_reports(cohort, gsc_dir=args.gsc_dir, raghpo_dir=args.raghpo_dir)
    missing = sorted(set(gold) - set(reports))
    if missing:
        raise SystemExit(f"{cohort}: {len(missing)} gold report(s) have no text, e.g. "
                         f"{missing[:5]} -- the length statistic would silently skip them")
    return gold, reports


def quartiles(values) -> tuple[float, float, float]:
    """``(median, first quartile, third quartile)`` of *values*."""
    q1, med, q3 = np.percentile(np.asarray(list(values), dtype=float), [25, 50, 75])
    return float(med), float(q1), float(q3)


def cohort_statistics(cohort: str, gold: dict, reports: dict, view) -> list[dict]:
    """Statistics of one cohort as scored: reports, pairs, terms, empty reports, dropped pairs.

    Args:
        cohort: cohort key (``hcy``, ``gsc_raghpo_ann``, ``gsc_2024_eval_206``).
        gold: ground truth, ``{report: set of HPO identifiers}``.
        reports: ``{report: text}``, used for length statistics.
        view: ``OntologyView`` that normalises the identifiers.

    Returns:
        Rows of ``{cohort, statistic, value, definition}``.
    """
    from hpo_extraction.evaluation.metrics import normalise_pair

    report_ids = sorted(gold)
    scored: dict[str, set[str]] = {}
    book = Counter()
    for rid in report_ids:
        g, _pred, counts = normalise_pair(gold[rid], set(), view)
        scored[rid] = g
        book.update({k: v for k, v in counts.items() if k.startswith("n_gold")})

    pairs = [(rid, term) for rid in report_ids for term in sorted(scored[rid])]
    doc_freq = Counter(term for _rid, term in pairs)
    depths = [view.depth(term) for _rid, term in pairs]
    if any(d is None for d in depths):
        raise SystemExit(f"{cohort}: a scored gold term has no depth -- normalisation let an "
                         "unscorable id through")
    words = [len(reports[rid].split()) for rid in report_ids]
    w_med, w_q1, w_q3 = quartiles(words)
    # A term and its direct parent both in one report's ground truth: segments are annotated with the most
    # specific term they support, so this counts findings stated at two levels in two places.
    parent_child = sum(1 for rid in report_ids for term in scored[rid]
                       for parent in view._direct_parents(term) if parent in scored[rid])
    anc_desc = sum(1 for rid in report_ids for term in scored[rid]
                   for other in scored[rid] if other != term and view.is_ancestor(other, of=term))
    d_med, d_q1, d_q3 = quartiles(depths)
    n_terms = len(doc_freq)

    stats = {
        "n_reports": (len(report_ids), "reports in the gold, including empty gold sets"),
        "n_reports_empty_gold": (sum(1 for r in report_ids if not scored[r]),
                                 "reports whose scored gold set is empty"),
        "n_gold_pairs_raw": (sum(len(set(v)) for v in gold.values()),
                             "(report, term) pairs as loaded, before normalisation"),
        "n_gold_out_of_subtree": (book["n_gold_out_of_subtree"],
                                  "gold pairs dropped: valid id outside the subontology"),
        "n_gold_nonexistent": (book["n_gold_nonexistent"],
                               "gold pairs dropped: id not in the ontology snapshot"),
        "n_gold_pairs": (len(pairs), "(report, term) pairs as scored"),
        "n_unique_terms": (n_terms, "distinct terms over the scored pairs"),
        "terms_per_report_mean": (len(pairs) / len(report_ids),
                                  "scored pairs / reports, empty gold sets included"),
        "terms_per_nonempty_report_mean": (
            len(pairs) / max(1, sum(1 for r in report_ids if scored[r])),
            "scored pairs / reports with a non-empty gold set"),
        "depth_median": (d_med, "shortest-path depth below HP:0000118, over scored pairs"),
        "depth_q1": (d_q1, "25th percentile of the same"),
        "depth_q3": (d_q3, "75th percentile of the same"),
        "n_parent_child_pairs": (parent_child, "(report, term, direct parent) with both in the "
                                 "report's scored gold"),
        "n_ancestor_descendant_pairs": (anc_desc, "(report, ancestor, descendant) with both in "
                                        "the report's scored gold"),
        "singleton_term_share": (100.0 * sum(1 for c in doc_freq.values() if c == 1) / n_terms,
                                 "percent of distinct terms gold in exactly one report"),
        "words_median": (w_med, "whitespace-delimited words per report, as the systems read it"),
        "words_q1": (w_q1, "25th percentile of the same"),
        "words_q3": (w_q3, "75th percentile of the same"),
        "words_mean": (statistics.fmean(words), "mean of the same"),
    }
    return [{"cohort": cohort, "statistic": k, "value": v, "definition": d}
            for k, (v, d) in stats.items()]


def reannotation_statistics(gold: dict, args) -> list[dict]:
    """RAG-HPO's re-annotation of its 114 documents against the corpus's own annotation of them.

    Raw pairs, as loaded and before restriction to the label space: the comparison is between two
    annotations of the same documents, not between two scorings. GSC+ only, so it runs anywhere.
    """
    import hpo_extraction.evaluation.result_tables.loaders as loaders

    original = loaders.load_gold("gsc", gsc_dir=args.gsc_dir, raghpo_dir=args.raghpo_dir)
    missing = sorted(set(gold) - set(original))
    if missing:
        raise SystemExit(f"re-annotation: {len(missing)} of its documents are not in the corpus, "
                         f"e.g. {missing[:5]}")
    ours = {(d, t) for d in gold for t in gold[d]}
    theirs = {(d, t) for d in gold for t in original[d]}
    kept, union = ours & theirs, ours | theirs
    stats = {
        "reann_pairs_original": (len(theirs), "corpus annotation of the same documents, raw pairs"),
        "reann_kept": (len(kept), "raw pairs in both annotations"),
        "reann_added": (len(ours - theirs), "raw pairs only in the re-annotation"),
        "reann_removed": (len(theirs - ours), "raw pairs only in the corpus annotation"),
        "reann_union": (len(union), "raw pairs in either annotation"),
        "reann_jaccard": (len(kept) / len(union), "kept / union"),
        "reann_fewer_share": (100.0 * (1 - len(ours) / len(theirs)),
                              "percent fewer raw pairs in the re-annotation"),
    }
    return [{"cohort": "gsc_raghpo_ann", "statistic": k, "value": v, "definition": d}
            for k, (v, d) in stats.items()]


#: The curation manifest's drop reasons -> the admission criterion of the appendix they fail.
#: A reason not listed aborts, so a new policy rule cannot be counted under the wrong criterion.
DROP_CRITERION = {
    "status:delete_suggested": "deletion",
    "no_evidence": "no_evidence",
    "label:unsure_annotation": "uncertain",
    "label:unsure_report": "uncertain",
    "label:unsure_report,unsure_annotation": "uncertain",
    "label:family": "family",
    "status:needs_work": "unfinished",
}


def curation_statistics(gt_path: str) -> list[dict]:
    """The curation counts the ground-truth build recorded in the curated ground truth's own manifest.json.

    Aggregates the policy already wrote down -- candidates, admitted, dropped by criterion -- so the
    appendix quotes the ground truth's build record rather than a recount that could apply the policy
    differently.
    """
    man = json.loads((Path(gt_path).parent / "manifest.json").read_text())
    counts = man["counts"]
    unknown = sorted(set(counts["n_dropped_by_reason"]) - set(DROP_CRITERION))
    if unknown:
        raise SystemExit(f"curation manifest has drop reason(s) {unknown} -- map them in "
                         "DROP_CRITERION")
    dropped = Counter()
    for reason, n in counts["n_dropped_by_reason"].items():
        dropped[DROP_CRITERION[reason]] += int(n)
    n_candidates, n_dropped = int(counts["n_annotations"]), int(counts["n_terms_dropped"])
    if sum(dropped.values()) != n_dropped:
        raise SystemExit(f"curation manifest: drop reasons sum to {sum(dropped.values())}, "
                         f"n_terms_dropped is {n_dropped}")
    stats = {
        "curation_candidates": (n_candidates, "candidate annotations the policy considered"),
        "curation_admitted": (n_candidates - n_dropped, "candidates admitted to the gold"),
        "curation_dropped": (n_dropped, "candidates dropped by the admission criteria"),
        **{f"curation_dropped_{k}": (dropped[k], f"dropped: {k}")
           for k in ("deletion", "no_evidence", "uncertain", "family", "unfinished")},
    }
    return [{"cohort": "hcy", "statistic": k, "value": v, "definition": d}
            for k, (v, d) in stats.items()]


def main() -> int:
    """Compute the statistics of the selected cohorts and merge them into ``dataset_statistics.csv``."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS,
                    default=["gsc_raghpo_ann", "gsc_2024_eval_206"])
    ap.add_argument("--hcy_curated_gt_path", default="")
    ap.add_argument("--hcy_input_dir", default="")
    ap.add_argument("--gsc_dir", default=str(_REPO / "resources" / "data" / "GSC_2024"))
    ap.add_argument("--raghpo_dir", default=str(_REPO / "resources" / "data" / "GSC_RAGHPO"))
    ap.add_argument("--hpo_json_path", default="")
    ap.add_argument("--output_dir", default=str(_REPO / "output" / "dataset_statistics"))
    args = ap.parse_args()

    _paths(_REPO)
    from hpo_extraction.ontology.hpo_tree import ontology_fingerprint

    view = load_view(args.hpo_json_path or None)
    rows = []
    for cohort in args.cohorts:
        gold, reports = load_cohort(cohort, args)
        rows += cohort_statistics(cohort, gold, reports, view)
        if cohort == "hcy":
            rows += curation_statistics(args.hcy_curated_gt_path)
        if cohort == "gsc_raghpo_ann":
            rows += reannotation_statistics(gold, args)
        got = {r["statistic"]: r["value"] for r in rows if r["cohort"] == cohort}
        print(f"{cohort}: {got['n_reports']} reports, {got['n_gold_pairs']} pairs, "
              f"{got['n_unique_terms']} terms, {got['n_reports_empty_gold']} empty")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "dataset_statistics.csv"
    new = pd.DataFrame(rows)
    if csv_path.exists():
        # Keep what another machine wrote for the other cohorts. Replace only what ran here. A
        # cohort no longer in COHORTS is a retired frame (the 228-document `gsc` one) and is
        # dropped, so the file never carries a column the table cannot show.
        old = pd.read_csv(csv_path)
        keep = old["cohort"].isin(COHORTS) & ~old["cohort"].isin(args.cohorts)
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(csv_path, index=False)

    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    manifest = {k: v for k, v in manifest.items() if k in COHORTS}
    stamp = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ontology_version": (ontology_fingerprint(args.hpo_json_path) if args.hpo_json_path
                             else ontology_fingerprint()),
        "inputs": ({"hcy_curated_gt_path": args.hcy_curated_gt_path,
                    "hcy_input_dir": args.hcy_input_dir}
                   if "hcy" in args.cohorts else
                   {"gsc_dir": args.gsc_dir, "raghpo_dir": args.raghpo_dir}),
    }
    for cohort in args.cohorts:
        manifest[cohort] = stamp
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
