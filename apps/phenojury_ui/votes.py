"""Aggregation re-run: k-of-N, plurality, arbitrary subsets, and who was decisive.

Every rule the driver swept is a pure function of the vote masks, so all of it re-runs offline at
zero GPU cost. That is the same invariant the driver relies on (``slm_ensemble_experiment`` module
docstring: "score once, re-run the decision logic"), which is what makes the k slider and the model
checklist honest live controls rather than approximations.

**Scoring comes from the driver, when the driver can be imported.** :func:`score` wraps
``slm_ensemble_experiment._safe_micro_macro``, and :func:`shipped_vote_k` /
:func:`shipped_plurality` call ``vote_k_sets`` / ``plurality_sets`` themselves. That buys the
guarantee that "recomputed" in :mod:`verify` means *the driver's own arithmetic on the app's
inputs*, so a gate failure can only be a data problem, never reimplementation drift.

It is not free. ``hpo_extraction.phenojury.generation`` imports ``torch`` and ``mlflow`` at module level
and reaches ``hpo_extraction.treephenorag.score_store``, which imports ``sentence_transformers``, the whole training
stack, none of which this app needs. On a CPU login node that chain can fail, and when it did the
first time, it took the entire app down: the exception escaped bundle construction and every view
fell back to its empty state.

So the import is **optional**. If it fails, the fallbacks below take over, they are the same
arithmetic written against the same inputs, they are compared against the driver's functions in
``test_exp13_06_ui`` (where the imports do work), and :func:`driver_status` reports the degradation
so the verification banner can say the gates ran unpinned. What is lost is only the "cannot be
reimplementation drift" claim. The gates still compare recomputed numbers against shipped
artifacts, which is what they are for.

**Decisiveness.** For a k-of-N rule a model *m* is decisive for term *t* in report *r* when *t* is
predicted with the full voter set and is not predicted without *m*. Since the rule is a threshold
on a count, that reduces to ``votes(t) == k and m voted for t``, every voter of a term sitting
on the threshold is essential, and no voter of a term above it is. :func:`decisive_mask`
returns that, and ``test_exp13_06_ui`` cross-checks it against brute-force leave-one-out
over all 2^n subsets.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from hpo_extraction.phenojury.ensemble_votes import (  # The shared mask arithmetic, see that module
    decisive_mask,
    plurality_from_masks,
    popcount,
    vote_k_from_masks,
)

from . import memo

logger = logging.getLogger(__name__)

#: Metric keys, in the order the scorecard shows them.
METRIC_KEYS = ("micro_precision", "micro_recall", "micro_f1",
               "macro_precision", "macro_recall", "macro_f1")

ZERO_METRICS = {k: 0.0 for k in METRIC_KEYS}


@lru_cache(maxsize=1)
def _driver():
    """``(module, reason)``, the driver, or ``(None, why not)``. Imported at most once.

    Never raises. An unimportable training stack must cost the pinning guarantee, not the app.
    """
    try:
        from hpo_extraction.phenojury import generation as slm_ensemble_experiment
    except Exception as exc:  # noqa: BLE001 - any failure in that import chain degrades the same
        logger.warning(
            "hpo_extraction.phenojury.generation could not be imported (%s: %s), the aggregation re-run "
            "falls back to this module's own implementation, and the verification gates are no "
            "longer pinned to the driver's functions. Everything else works normally.",
            type(exc).__name__, exc)
        return None, f"{type(exc).__name__}: {exc}"
    return slm_ensemble_experiment, ""


def driver_status() -> tuple[bool, str]:
    """``(pinned, reason)`` for the verification banner."""
    module, reason = _driver()
    return module is not None, reason


def prompts() -> tuple[str | None, str | None, str]:
    """``(system, user, reason)``, the extraction prompt, imported, not transcribed.

    The pipeline does not persist its prompts, so the deep dive shows the driver's own constants. A
    transcribed copy would drift the moment anyone edited the experiment. When the driver will not
    import there is nothing honest to substitute, so the view says so instead of guessing.
    """
    module, reason = _driver()
    if module is None:
        return None, None, reason
    return module.SYSTEM_PROMPT, module.USER_TEMPLATE, ""


def score(gold: dict, predicted: dict, report_ids: list[str]) -> dict:
    """Micro/macro P/R/F1 in the earlier runs'open-set frame, via the driver's own guarded scorer."""
    gold_sets = [set(gold.get(r, ())) for r in report_ids]
    pred_sets = [set(predicted.get(r, ())) for r in report_ids]
    module, _ = _driver()
    if module is not None:
        return module._safe_micro_macro(gold_sets, pred_sets)
    return _fallback_micro_macro(gold_sets, pred_sets)


def shipped_vote_k(report_hpos: dict, models: list[str], report_ids: list[str], k: int,
                   min_count: int) -> dict:
    """``vote_k_sets`` as the driver called it, the reference for gate G1."""
    module, _ = _driver()
    if module is not None:
        return module.vote_k_sets(report_hpos, models, report_ids, k, min_count)
    return _fallback_vote_k(report_hpos, models, report_ids, k, min_count)


def shipped_plurality(sent_hpos: dict, models: list[str], report_ids: list[str],
                      min_count: int) -> tuple[dict, list[dict]]:
    """``plurality_sets`` as the driver called it, the reference for gate G1."""
    module, _ = _driver()
    if module is not None:
        return module.plurality_sets(sent_hpos, models, report_ids, min_count)
    return _fallback_plurality(sent_hpos, models, report_ids, min_count)


# ── fallbacks, used only when the driver cannot be imported ──────────────────
# Line-for-line equivalents of slm_ensemble_experiment._safe_micro_macro / vote_k_sets /
# plurality_sets. `test_exp13_06_ui.TestDriverFallback` asserts each one against the function it
# mirrors, so drift fails a test, not quietly changing a cohort's numbers.

def _fallback_micro_macro(gold_sets: list[set], pred_sets: list[set]) -> dict:
    """``_safe_micro_macro``, every denominator guarded, since a high k can predict nothing."""
    from hpo_extraction.evaluation.set_metrics import calc_metric   # stdlib-only module, always importable

    n = len(gold_sets)
    if n == 0:
        return dict(ZERO_METRICS)

    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    n_pred = sum(len(p) for p in pred_sets)
    n_gold = sum(len(g) for g in gold_sets)
    micro_p = tp / n_pred if n_pred else 0.0
    micro_r = tp / n_gold if n_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    per_report = [calc_metric(g, p) for g, p in zip(gold_sets, pred_sets)]
    macro_p = sum(x[0] for x in per_report) / n
    macro_r = sum(x[1] for x in per_report) / n
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0
    return {"micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
            "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1}


def _fallback_vote_k(report_hpos: dict, models: list[str], report_ids: list[str], k: int,
                     min_count: int) -> dict:
    """``vote_k_sets``, keep a term iff at least *k* models detected it at or above *min_count*."""
    from collections import Counter

    out: dict = {}
    for rid in report_ids:
        tally: Counter = Counter()
        for model in models:
            for hpo_id, count in report_hpos.get(model, {}).get(rid, {}).items():
                if count >= min_count:
                    tally[hpo_id] += 1
        out[rid] = {h for h, n in tally.items() if n >= k}
    return out


def _fallback_plurality(sent_hpos: dict, models: list[str], report_ids: list[str],
                        min_count: int) -> tuple[dict, list[dict]]:
    """``plurality_sets``, per-sentence argmax, ties keep every winner, unioned over sentences."""
    from collections import Counter

    selected: dict = {rid: set() for rid in report_ids}
    stats = {m: {"n_votes_cast": 0, "n_votes_won": 0} for m in models}

    for rid in report_ids:
        sent_nums: set = set()
        for model in models:
            sent_nums |= set(sent_hpos.get(model, {}).get(rid, {}).keys())
        for sent_num in sent_nums:
            model_votes: dict = {}
            tally: Counter = Counter()
            for model in models:
                hpos = {h for h, c in
                        sent_hpos.get(model, {}).get(rid, {}).get(sent_num, {}).items()
                        if c >= min_count}
                if hpos:
                    model_votes[model] = hpos
                    for h in hpos:
                        tally[h] += 1
            if not tally:
                continue
            best = max(tally.values())
            winners = {h for h, c in tally.items() if c == best}
            selected[rid] |= winners
            for model, hpos in model_votes.items():
                for h in hpos:
                    stats[model]["n_votes_cast"] += 1
                    if h in winners:
                        stats[model]["n_votes_won"] += 1

    vote_rows = []
    for model in models:
        cast, won = stats[model]["n_votes_cast"], stats[model]["n_votes_won"]
        vote_rows.append({"model": model, "n_votes_cast": cast, "n_votes_won": won,
                          "pct_votes_won": round(100.0 * won / cast, 2) if cast else 0.0})
    return selected, vote_rows


# ── the mask-native re-run (what the live controls use) ──────────────────────





# ── the report subset ────────────────────────────────────────────────────────
#
# Every whole-cohort walk below takes an optional *report_ids*. It defaults to the bundle's own
# list, so drivers.py, the unit tests and any caller predating the deep-dive filter are unchanged;
# passing a subset is what views/common.report_ids does when a sampling frame is filtering the
# screen. The alternative, filtering the bundle itself, would also filter verify's gates, which
# must keep comparing against the shipped full-cohort artifacts to mean anything.

def _ids(bundle: dict, report_ids=None) -> list[str]:
    return list(bundle["report_ids"]) if report_ids is None else list(report_ids)


def _ids_key(bundle: dict, report_ids) -> tuple:
    """The memo-key component for a subset, empty for the whole cohort.

    Empty, not the full list so an unfiltered call keeps hitting the entries it always did. A filtered one gets its own, because a table over twenty reports and a table over all of them
    are different answers to the same question.
    """
    if report_ids is None or list(report_ids) == list(bundle["report_ids"]):
        return ()
    return tuple(report_ids)


def predicted_sets(bundle: dict, config: dict) -> dict:
    """``{report: set(hpo)}`` at the current UI configuration.

    ``config`` carries ``rule`` (``"vote_k"`` or ``"plurality"``), ``k``, ``models`` (the active
    subset) and ``min_count``.

    Memoised on the bundle. A single render of the deep dive asks for this three times, and the
    autopsy, the FP taxonomy and the term explorer each open by asking for it again, the reader
    changed one control, so recomputing it six times is six times the same answer.
    """
    return memo.cached(bundle, ("predicted_sets", memo.config_key(config)),
                       lambda: _predicted_sets(bundle, config))


def _predicted_sets(bundle: dict, config: dict) -> dict:
    subset_mask = bundle["mask_of"](config["models"])
    if config["rule"] == "plurality":
        return plurality_from_masks(bundle["sentence_masks"](config["min_count"]), subset_mask)
    k = max(1, min(int(config["k"]), len(config["models"]) or 1))
    return vote_k_from_masks(bundle["masks"](config["min_count"]), subset_mask, k)


# ── decisiveness ─────────────────────────────────────────────────────────────



def decisive_rows(bundle: dict, config: dict, gold: dict, report_ids=None) -> list[dict]:
    """One row per predicted term: its voters, whether they were decisive, and the outcome.

    Only defined for the k-of-N rule, plurality's winner set is a per-sentence argmax, so
    "decisive" would mean something different there and is reported as no voters being pivotal.
    """
    if not memo.own_gold(bundle, gold):
        return _decisive_rows(bundle, config, gold, report_ids)
    return memo.cached(bundle,
                       ("decisive_rows", memo.config_key(config), _ids_key(bundle, report_ids)),
                       lambda: _decisive_rows(bundle, config, gold, report_ids))


def _decisive_rows(bundle: dict, config: dict, gold: dict, report_ids=None) -> list[dict]:
    subset_mask = bundle["mask_of"](config["models"])
    masks = bundle["masks"](config["min_count"])
    k = max(1, min(int(config["k"]), len(config["models"]) or 1))
    predicted = predicted_sets(bundle, config)
    is_k_rule = config["rule"] != "plurality"

    rows: list[dict] = []
    for report_id in _ids(bundle, report_ids):
        gold_set = set(gold.get(report_id, ()))
        for hpo_id in sorted(predicted.get(report_id, ())):
            mask = masks.get(report_id, {}).get(hpo_id, 0)
            voters = mask & subset_mask
            dec = decisive_mask(mask, subset_mask, k) if is_k_rule else 0
            rows.append({
                "report_id": report_id, "hpo_id": hpo_id,
                "voters": voters, "n_votes": popcount(voters),
                "decisive": dec, "n_decisive": popcount(dec),
                "outcome": "TP" if hpo_id in gold_set else "FP",
            })
    return rows


def voter_profile(rows: list[dict], models: list[str]) -> list[dict]:
    """Per model: how often it voted, how often it was decisive, and for what.

    ``n_decisive_fp`` is the number the models view leads with, a model that is essential only
    for false positives is doing active harm at this k, which no aggregate precision number says.
    """
    out = []
    for i, model in enumerate(models):
        bit = 1 << i
        voted = [r for r in rows if r["voters"] & bit]
        dec = [r for r in voted if r["decisive"] & bit]
        out.append({
            "model": model,
            "n_voted": len(voted),
            "n_voted_tp": sum(1 for r in voted if r["outcome"] == "TP"),
            "n_voted_fp": sum(1 for r in voted if r["outcome"] == "FP"),
            "n_decisive": len(dec),
            "n_decisive_tp": sum(1 for r in dec if r["outcome"] == "TP"),
            "n_decisive_fp": sum(1 for r in dec if r["outcome"] == "FP"),
            "pct_decisive": round(100.0 * len(dec) / len(voted), 2) if voted else 0.0,
        })
    return out


def coalitions(rows: list[dict], models: list[str], top_n: int = 12) -> list[dict]:
    """The commonest voter sets, most frequent first, the UpSet input."""
    counts: dict[int, dict] = {}
    for r in rows:
        c = counts.setdefault(r["voters"], {"voters": r["voters"], "n": 0, "n_tp": 0})
        c["n"] += 1
        c["n_tp"] += r["outcome"] == "TP"
    ordered = sorted(counts.values(), key=lambda c: (-c["n"], c["voters"]))[:top_n]
    for c in ordered:
        c["models"] = [m for i, m in enumerate(models) if c["voters"] >> i & 1]
        c["size"] = popcount(c["voters"])
        c["precision"] = round(c["n_tp"] / c["n"], 4) if c["n"] else 0.0
    return ordered


# ── sweeps and per-model contribution ────────────────────────────────────────

def sweep_k(bundle: dict, gold: dict, models: list[str], min_count: int,
            report_ids=None) -> list[dict]:
    """One row per k from 1 to ``len(models)``: term count and the six metrics.

    Also carries ``n_tp`` / ``n_fp`` / ``n_fn`` so the scorecard can show the trade in counts as
    well as in rates, at high k the rates move little while the term count collapses, and only the
    counts make that visible.
    """
    if not memo.own_gold(bundle, gold):
        return _sweep_k(bundle, gold, models, min_count, report_ids)
    return memo.cached(bundle, ("sweep_k", tuple(sorted(models)), int(min_count),
                                _ids_key(bundle, report_ids)),
                       lambda: _sweep_k(bundle, gold, models, min_count, report_ids))


def _sweep_k(bundle: dict, gold: dict, models: list[str], min_count: int,
             report_ids=None) -> list[dict]:
    masks = bundle["masks"](min_count)
    subset_mask = bundle["mask_of"](models)
    report_ids = _ids(bundle, report_ids)
    rows = []
    for k in range(1, max(1, len(models)) + 1):
        predicted = vote_k_from_masks(masks, subset_mask, k)
        rows.append({"rule": f"vote_k{k}", "k": k, **_counts(gold, predicted, report_ids),
                     **score(gold, predicted, report_ids)})
    predicted = plurality_from_masks(bundle["sentence_masks"](min_count), subset_mask)
    rows.append({"rule": "agg_plurality", "k": None, **_counts(gold, predicted, report_ids),
                 **score(gold, predicted, report_ids)})
    return rows


def _counts(gold: dict, predicted: dict, report_ids: list[str]) -> dict:
    tp = fp = fn = 0
    for report_id in report_ids:
        g = set(gold.get(report_id, ()))
        p = set(predicted.get(report_id, ()))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return {"n_predicted": tp + fp, "n_gold": tp + fn, "n_tp": tp, "n_fp": fp, "n_fn": fn}


def _f1_for_mask(masks: dict, gold: dict, report_ids: list[str], subset_mask: int, k: int) -> float:
    """Micro F1 for one subset, the reference definition of the subset value function."""
    tp = n_pred = n_gold = 0
    for report_id in report_ids:
        g = set(gold.get(report_id, ()))
        p = {h for h, mask in masks.get(report_id, {}).items()
             if popcount(mask & subset_mask) >= k}
        tp += len(g & p)
        n_pred += len(p)
        n_gold += len(g)
    if not n_pred or not n_gold or not tp:
        return 0.0
    precision, recall = tp / n_pred, tp / n_gold
    return 2 * precision * recall / (precision + recall)


def mask_histogram(bundle: dict, gold: dict, min_count: int,
                   report_ids=None) -> tuple[list[tuple], int]:
    """``([(mask, n_terms, n_gold_terms)], total_gold)``, the cohort collapsed by vote mask.

    Every subset evaluation is "which candidate terms clear k on this subset", i.e. a popcount per
    ``(report, term)`` pair. A cohort has tens of thousands of those and at most ``2^n`` distinct
    masks, 256 for eight models. Two pairs with the same mask always answer the same question, so
    collapsing them to a count makes each evaluation a loop over the *distinct masks*, not
    over the cohort. That is what makes an exact 256-subset Shapley a fraction of a second instead
    of a wait, and it is exact, not an approximation.

    ``total_gold`` is counted from the annotations, not from the histogram: an annotated term no
    model ever detected has no mask and must still be in the recall denominator.
    """
    if not memo.own_gold(bundle, gold):
        return _mask_histogram(bundle, gold, min_count, report_ids)
    return memo.cached(bundle, ("mask_histogram", int(min_count), _ids_key(bundle, report_ids)),
                       lambda: _mask_histogram(bundle, gold, min_count, report_ids))


def _mask_histogram(bundle: dict, gold: dict, min_count: int,
                    report_ids=None) -> tuple[list[tuple], int]:
    masks = bundle["masks"](min_count)
    tally: dict[int, list[int]] = {}
    n_gold = 0
    for report_id in _ids(bundle, report_ids):
        gold_set = set(gold.get(report_id, ()))
        n_gold += len(gold_set)
        for hpo_id, mask in masks.get(report_id, {}).items():
            entry = tally.setdefault(mask, [0, 0])
            entry[0] += 1
            entry[1] += hpo_id in gold_set
    return [(mask, n, n_gold_terms) for mask, (n, n_gold_terms) in tally.items()], n_gold


def _f1_for_hist(hist: list[tuple], n_gold: int, subset_mask: int, k: int) -> float:
    """Micro F1 for one subset, off the histogram. Equal to :func:`_f1_for_mask` by design."""
    tp = n_pred = 0
    for mask, n_terms, n_gold_terms in hist:
        if popcount(mask & subset_mask) >= k:
            n_pred += n_terms
            tp += n_gold_terms
    if not n_pred or not n_gold or not tp:
        return 0.0
    precision, recall = tp / n_pred, tp / n_gold
    return 2 * precision * recall / (precision + recall)


def leave_one_out(bundle: dict, gold: dict, models: list[str], k: int, min_count: int,
                  report_ids=None) -> list[dict]:
    """Micro F1 with each model removed, against the full-ensemble baseline.

    The removal keeps *k* fixed unless that would exceed the surviving ensemble, in which case k is
    clamped, dropping a model from a unanimity rule otherwise makes the rule unsatisfiable and the
    "contribution" measured would be the clamping, not the model.
    """
    hist, n_gold = mask_histogram(bundle, gold, min_count, report_ids)
    full = bundle["mask_of"](models)
    base = _f1_for_hist(hist, n_gold, full, min(k, len(models)))
    rows = []
    for i, model in enumerate(models):
        without = full & ~(1 << i)
        k_eff = min(k, max(1, len(models) - 1))
        f1 = _f1_for_hist(hist, n_gold, without, k_eff)
        rows.append({"model": model, "f1_without": f1, "delta": base - f1, "k_effective": k_eff})
    return rows


def unique_recall(bundle: dict, gold: dict, models: list[str], min_count: int,
                  report_ids=None) -> list[dict]:
    """Annotated terms only this model detected, recall that dies with it, independent of k."""
    masks = bundle["masks"](min_count)
    ids = _ids(bundle, report_ids)
    rows = []
    for i, model in enumerate(models):
        bit = 1 << i
        n_unique = n_found = 0
        for report_id in ids:
            gold_set = set(gold.get(report_id, ()))
            for hpo_id in gold_set:
                mask = masks.get(report_id, {}).get(hpo_id, 0)
                if mask & bit:
                    n_found += 1
                    if mask == bit:
                        n_unique += 1
        rows.append({"model": model, "n_gold_found": n_found, "n_unique": n_unique})
    return rows


def shapley(bundle: dict, gold: dict, models: list[str], k: int, min_count: int,
            report_ids=None) -> list[dict]:
    """Exact Shapley value of each model, with micro F1 as the value function.

    Exact, not sampled: 2^n subsets is 256 for eight models, and each is a popcount per
    candidate term, so the whole thing is cheaper than the sampling machinery would be.

    The value function is ``F1(vote_k(S))`` with ``k`` clamped to ``|S|`` and ``v(∅) = 0``. Clamping
    is a real choice and it is the reason the numbers here answer "what does this model contribute
    to an ensemble of this shape", not "…at literally this k", at k=8 an unclamped value function
    is zero for all 255 proper subsets and every model's Shapley value collapses to F1/8.

    The 2^n evaluation is the slowest thing the Models view does, and nothing about it changes
    while the reader is on that screen, so it is memoised on the bundle.
    """
    if not memo.own_gold(bundle, gold):
        return _shapley(bundle, gold, models, k, min_count, report_ids)
    return memo.cached(bundle, ("shapley", tuple(sorted(models)), int(k), int(min_count),
                                _ids_key(bundle, report_ids)),
                       lambda: _shapley(bundle, gold, models, k, min_count, report_ids))


def _shapley(bundle: dict, gold: dict, models: list[str], k: int, min_count: int,
             report_ids=None) -> list[dict]:
    n = len(models)
    if n == 0:
        return []
    if n > 12:  # 4096 subsets is still fine. Beyond that, refuse, not hang the callback
        logger.warning("skipping exact Shapley for %d models", n)
        return [{"model": m, "shapley": float("nan")} for m in models]

    hist, n_gold = mask_histogram(bundle, gold, min_count, report_ids)
    bit_of = [1 << i for i in range(n)]
    order = {m: i for i, m in enumerate(models)}
    subset_of = {m: bit_of[order[m]] for m in models}

    value: dict[int, float] = {0: 0.0}
    for subset in range(1, 1 << n):
        size = popcount(subset)
        value[subset] = _f1_for_hist(hist, n_gold, subset, min(k, size))

    # Shapley weight for a coalition of size s in an n-player game: s!(n-s-1)!/n!
    factorial = [1.0] * (n + 1)
    for i in range(1, n + 1):
        factorial[i] = factorial[i - 1] * i
    weight = [factorial[s] * factorial[n - s - 1] / factorial[n] for s in range(n)]

    out = []
    for model in models:
        bit = subset_of[model]
        total = 0.0
        for subset in range(1 << n):
            if subset & bit:
                continue
            total += weight[popcount(subset)] * (value[subset | bit] - value[subset])
        out.append({"model": model, "shapley": total})
    return out
