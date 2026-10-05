"""T6.3, do the two contributed architectures fail on the same terms, or on different ones?

Every other table in this chapter asks how *well* each method does. This one asks whether running
both would buy anything, which is a different question and not answerable from any pair of scores:
two methods at identical F1 might miss the same 400 terms or disjoint sets of 400, and only the
second case makes an ensemble worth building.

**The split is chapter 4's, not a new one.** ``treephenorag_protocol`` attributes each of TreePhenoRAG's false
negatives to the stage that lost it, and two of those stages mean genuinely different things here:

``pruning`` / ``retrieval``   the verifier **never read the evidence**. A pruning loss was never
                              scored at all. A retrieval loss *was* scored, but on segments that do
                              not contain the curator's evidence -- so "never scored" is true of the
                              first bucket only, and must not label the group. No threshold anywhere
                              in the configuration space could have recovered either on evidence, so
                              a method that finds these is supplying *reach* the architecture lacks.
``judgement`` / ``pooling``   the verifier read the evidence and the traversal decided wrongly. A
                              method that finds these is *disagreeing with a verdict*, not extending
                              coverage.

Collapsing those into one "overlap" number would hide which is true, and they support opposite
conclusions: only the first is an argument for running both pipelines.

Both numbers are set algebra over two prediction files plus that attribution, so this is CPU-only.
What it needs that chapter 4 did not previously write is the attribution **per pair**, the bucket
totals cannot answer a question about which pairs, except by assuming the two methods' misses are
independent, which is what is being measured.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger("exp13_25.complementarity")

#: The two groups of chapter-4 buckets, and what recovering each one would mean. Order is the
#: order T6.3 reports them: structural reach first, because it is the one that would justify
#: running both architectures.
BUCKET_GROUPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("evidence never read (pruning + retrieval)", ("pruning", "retrieval"),
     "terms whose evidence the verifier never read -- never scored (pruning), or scored on "
     "segments without the evidence (retrieval); no threshold in its configuration space "
     "recovers them on evidence"),
    ("evidence read, judged wrongly (judgement + pooling)", ("judgement", "pooling"),
     "terms whose evidence the verifier read and the traversal decided wrongly"),
)


def load_attribution(path: str | Path) -> list[dict]:
    """``recall_attribution.csv`` from the TreePhenoRAG protocol: one row per missed annotated pair.

    Returns ``[]`` when absent, which is an ordinary state of the world -- chapter 4 has to have
    been re-run with its ``diagnostics`` stage since the column was added -- and the caller writes
    a stated-absent note rather than a number.
    """
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("report_id") and r.get("hpo_id")]
    logger.info("attribution: %d missed pair(s) from %s", len(rows), path)
    return rows


def build(attribution: list[dict], recovered_by, gold, view, label: str) -> list[dict]:
    """One row per bucket group: how many of TreePhenoRAG's losses *label* recovers.

    Args:
        attribution: the TreePhenoRAG protocol's per-pair rows.
        recovered_by: ``{report_id: set of predicted HPO ids}`` for the other method.
        ground truth: the cohort ground truth, used only to drop attribution rows for pairs this chapter does not
            score -- the two experiments can disagree about the cohort, and silently counting a
            pair outside this table's denominator would make the share wrong.
        view: for resolving ids, so an alternate id on one side does not read as a miss.
        label: the recovering method's name, for the log and the row.

    ``share_recovered`` is over the pairs in that group, so the two rows are not complements of
    each other and must not be added together.
    """
    if not attribution:
        return []

    scored = {(rid, view.resolve(t) or t) for rid, terms in gold.items() for t in terms}
    resolved = {rid: {view.resolve(t) or t for t in terms}
                for rid, terms in (recovered_by or {}).items()}

    rows = []
    for name, buckets, meaning in BUCKET_GROUPS:
        pairs = set()
        for record in attribution:
            if record.get("bucket") not in buckets:
                continue
            key = (record["report_id"], view.resolve(record["hpo_id"]) or record["hpo_id"])
            if key in scored:
                pairs.add(key)
        found = {(rid, term) for rid, term in pairs if term in resolved.get(rid, ())}
        rows.append({
            "bucket": name,
            "buckets": "+".join(buckets),
            "meaning": meaning,
            "recovered_by": label,
            "n_missed": len(pairs),
            "n_recovered": len(found),
            "share_recovered": round(len(found) / len(pairs), 4) if pairs else "",
        })
        logger.info("complementarity | %-42s %4d missed, %4d recovered by %s (%.1f%%)",
                    name, len(pairs), len(found), label,
                    100 * len(found) / len(pairs) if pairs else float("nan"))
    return rows
