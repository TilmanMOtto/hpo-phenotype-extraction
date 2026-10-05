"""Why each missed annotated term died, the recall autopsy.

Recall is where this method loses (GSC+ µR 0.521 at its best configuration), and "recall is 0.52"
is not actionable. Each annotated term the ensemble failed to predict died at one of five
places in the generate → ground → vote chain, and which one decides what would fix it:

``not_written``      no model wrote anything on any sentence of this report. A generation failure, more tokens, a different prompt, a different model.
``not_linked``       models wrote text, but PhenoBERT never grounded this term anywhere in the
                     report. A grounding failure, the ontology linker, not the SLMs.
``negated``          PhenoBERT found the term and every occurrence was marked negated, so it was
                     dropped before it could vote. Either the report really does negate it (the
                     annotation is wrong) or the negation detector is.
``below_min_count``  found, un-negated, but no model reached ``min_detection_count``. A threshold
                     cost, recoverable from this screen.
``below_votes``      1..k-1 models found it. A *voting* cost, and the only fate that lowering k
                     recovers, the number the k slider trades against precision.

The ladder is ordered, exhaustive and mutually exclusive by design: a term with at least k
votes is predicted and is not an FN at all, so ``below_votes`` catches everything the four earlier
rungs do not. ``no_evidence`` exists only so the partition can never silently fail. It should always
be empty, and :func:`verify.gate_fate_partition` asserts the counts add up.

Two honest limitations, surfaced in the UI rather than hidden:

* ``not_written`` is a per-**report** verdict, not a per-term one. The Free Listing generation run has no segment-level
  annotation, so nothing on disk says which sentence a given annotated term should have come from. The
  strongest defensible statement is "no model wrote anything anywhere in this report".
* ``not_linked`` vs ``negated`` vs ``below_min_count`` needs the PhenoBERT TSVs. On a run whose
  ``phenobert_output_*`` directories were not kept, the three collapse into ``not_linked`` and
  :func:`build` says so via ``degraded``.
"""

from __future__ import annotations

from . import memo
from .detections import popcount

#: Best → worst along the pipeline. The order is the reading order in every chart.
FATE_ORDER = ("below_votes", "below_min_count", "negated", "not_linked", "not_written",
              "no_evidence")

FATE_HELP = {
    "below_votes": "found by 1..k-1 models, recoverable by lowering k",
    "below_min_count": "found, but no model reached min_detection_count",
    "negated": "found and marked negated, so it never voted",
    "not_linked": "models wrote text, but PhenoBERT never grounded this term",
    "not_written": "no model wrote anything at all for this report",
    "no_evidence": "unclassified, should never occur; see verify.gate_fate_partition",
}

#: The fate a lower k would recover, which is the only one the configuration controls.
RECOVERABLE = "below_votes"


def build(bundle: dict, config: dict, gold: dict, predicted: dict,
          report_ids=None) -> dict:
    """``{"rows": [...], "counts": {fate: n}, "degraded": bool}``.

    One row per missed annotated term: ``report_id, hpo_id, fate, n_votes, voters, n_negated_models``.
    ``voters`` is a bitmask over the configured subset, so the models view can attribute a
    ``below_votes`` miss to the models that *did* find it, the ones an ensemble one notch smaller
    would have needed.

    Memoised on the bundle: it is a whole-cohort walk, and three views want it. The deep dive in
    particular needs the cohort-wide result even to describe one report, handing it a one-report
    ground truth dict would classify every other report's annotations as missed.

    The memo key is the *configuration*, so it may only be used when ``predicted`` is what that
    configuration predicts. That is checked by identity against the (also memoised)
    :func:`votes.predicted_sets`, which every caller in the app goes through. Anything else is
    computed fresh.
    """
    from . import votes as votes_mod

    reusable = (memo.own_gold(bundle, gold)
                and predicted is votes_mod.predicted_sets(bundle, config))
    if not reusable:
        return _build(bundle, config, gold, predicted, report_ids)
    return memo.cached(bundle, ("autopsy", memo.config_key(config)),
                       lambda: _build(bundle, config, gold, predicted, report_ids))


def _build(bundle: dict, config: dict, gold: dict, predicted: dict,
           report_ids=None) -> dict:
    models = config["models"]
    subset_mask = bundle["mask_of"](models)
    masks = bundle["masks"](config["min_count"])
    raw_masks = bundle["masks"](1)
    k = max(1, min(int(config["k"]), len(models) or 1))
    is_k_rule = config["rule"] != "plurality"

    pb_by_term = bundle["pb_index"]["by_term"]
    degraded = not bundle["has_phenobert"]
    active = set(models)
    wrote_any = bundle["wrote_any_by_report"]

    rows: list[dict] = []
    counts = {fate: 0 for fate in FATE_ORDER}
    for report_id in (bundle["report_ids"] if report_ids is None else report_ids):
        gold_set = set(gold.get(report_id, ()))
        missed = gold_set - set(predicted.get(report_id, ()))
        for hpo_id in sorted(missed):
            mask = masks.get(report_id, {}).get(hpo_id, 0) & subset_mask
            pb_rows = [r for r in pb_by_term.get((report_id, hpo_id), ()) if r["model"] in active]
            fate = _fate(
                votes=popcount(mask),
                k=k if is_k_rule else 1,
                pb_rows=pb_rows,
                raw_mask=raw_masks.get(report_id, {}).get(hpo_id, 0) & subset_mask,
                report_wrote=wrote_any.get(report_id, False),
                degraded=degraded,
            )
            counts[fate] += 1
            rows.append({
                "report_id": report_id, "hpo_id": hpo_id, "fate": fate,
                "n_votes": popcount(mask), "voters": mask,
                "n_negated_models": len({r["model"] for r in pb_rows if r["negated"]}),
            })
    return {"rows": rows, "counts": counts, "degraded": degraded}


def _fate(*, votes: int, k: int, pb_rows: list[dict], raw_mask: int, report_wrote: bool,
          degraded: bool) -> str:
    """The single rung this miss fell off. Order counts. See the module docstring."""
    if votes >= 1:
        # It cleared min_detection_count and still lost, so the vote threshold is what stopped it.
        return "below_votes" if votes < k else "no_evidence"
    if not report_wrote:
        return "not_written"
    if degraded or not pb_rows:
        return "not_linked"
    if all(r["negated"] for r in pb_rows):
        return "negated"
    if raw_mask:
        # Un-negated detections exist and would have voted at min_detection_count = 1.
        return "below_min_count"
    return "not_linked"


def blame(rows: list[dict], models: list[str]) -> list[dict]:
    """Per model: how many ``below_votes`` misses it *did* find, the near-recoveries it carried.

    A model high here is one whose vote is being outvoted: it is finding annotated terms the rest of
    the ensemble is not, and the k threshold is discarding them.
    """
    out = []
    for i, model in enumerate(models):
        bit = 1 << i
        recoverable = [r for r in rows if r["fate"] == RECOVERABLE]
        out.append({
            "model": model,
            "n_found_but_outvoted": sum(1 for r in recoverable if r["voters"] & bit),
            "n_recoverable_total": len(recoverable),
        })
    return out


def recoverable_by_k(bundle: dict, config: dict, gold: dict, report_ids=None) -> list[dict]:
    """How many currently-missed annotated terms each lower k would recover, and at what FP cost.

    The honest form of "just lower k": every row pairs the recall gained against the false positives
    bought with it, so the trade is visible in the same table, not across two charts.
    """
    if not memo.own_gold(bundle, gold):
        return _recoverable_by_k(bundle, config, gold)
    return memo.cached(bundle, ("recoverable_by_k", memo.config_key(config)),
                       lambda: _recoverable_by_k(bundle, config, gold, report_ids))


def _recoverable_by_k(bundle: dict, config: dict, gold: dict, report_ids=None) -> list[dict]:
    from . import votes as votes_mod

    models = config["models"]
    subset_mask = bundle["mask_of"](models)
    masks = bundle["masks"](config["min_count"])
    current = votes_mod.vote_k_from_masks(masks, subset_mask, config["k"])

    rows = []
    for k in range(1, max(1, len(models)) + 1):
        candidate = votes_mod.vote_k_from_masks(masks, subset_mask, k)
        gained_tp = gained_fp = 0
        for report_id in (bundle["report_ids"] if report_ids is None else report_ids):
            gold_set = set(gold.get(report_id, ()))
            new_terms = candidate.get(report_id, set()) - current.get(report_id, set())
            gained_tp += len(new_terms & gold_set)
            gained_fp += len(new_terms - gold_set)
        rows.append({"k": k, "gained_tp": gained_tp, "gained_fp": gained_fp,
                     "ratio": round(gained_tp / gained_fp, 3) if gained_fp else None})
    return rows
