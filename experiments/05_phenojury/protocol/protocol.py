"""The selection machinery for the PhenoJury protocol, with the juror generalised to a (model, prompt, normaliser).

``an earlier exploratory run`` established the protocol this reuses, nested CV, backward elimination
with a size tolerance, report-level bootstrap, paired approximate randomisation, over a juror pool
that was eight *models* sharing one prompt and one normaliser. Everything here is that same
procedure with the juror widened to a triple, which is what turns S1 through S8 into one question
asked of different slices of a 96-cell pool rather than eight unrelated analyses.

Three things are genuinely new and each is here, not in ``an earlier exploratory run``:

**The vote unit.** The shipped ``vote_k`` rules are report-level by design, ``ensemble_eval.derive_patient_hpos`` sums detection counts across sentences *before* the tally, so
a juror contributes one vote per ``(report, term)`` however many segments it named it in.
``agg_plurality`` is the only shipped rule counting per segment. ``an earlier exploratory run`` declared a
``vote_unit`` stage and never implemented one. :data:`UNITS` implements all three, and ``window`` is
just ``segment`` with each juror's per-sentence set pre-expanded over its neighbours, so the vote
arithmetic has one definition and the unit is a property of how the input was packed.

**All thresholds from one pass.** ``an earlier exploratory run`` re-votes for every *k*. Since a *k*-of-N rule is a
threshold on a count, :func:`max_counts` computes the count once per ``(rule, unit, subset, fold)``
and every *k* is then a comparison against it. That is an eightfold saving, and it is what pays for
the unit axis: three units at one pass each cost about what one unit at eight passes cost before.

**Reduction across units happens before thresholding,.** Under a segment unit a term is
accepted when *k* jurors agree on it **in some one segment**, so the report-level quantity is the
*maximum* count over segments, not the sum. Summing would let two jurors in two different sentences
manufacture a vote of two that no segment ever cast.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

logger = logging.getLogger(__name__)

from hpo_extraction.phenojury.vote import (  # noqa: F401  (re-exported for run.py and the tests)
    RULES, UNITS, Juror, Packed, max_counts, pack, packings_for, predict, predict_all_k,
    reduce_specific, sets_at_k,
)
from hpo_extraction.phenojury.vote import _closure_cached  # noqa: F401






# ── Loading the grid ─────────────────────────────────────────────────────────

def load_detections_file(path: Path) -> dict[str, dict[int, set[str]]]:
    """``{report: {sentence: {hpo}}}`` from one ``detections_*.jsonl``.

    Accepts both id spellings the cache uses, ``report_id`` in the detections files,
    ``patient_id`` in the extractions, because a loader that handles one of them fails on half the
    artifacts and does so silently.
    """
    out: dict[str, dict[int, set[str]]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("report_id", rec.get("patient_id", "")))
            hpo = rec.get("hpo_id")
            if not rid or not hpo:
                continue
            try:
                sent = int(rec["sentence_number"])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault(rid, {}).setdefault(sent, set()).add(hpo)
    return out


def load_grid(
    grid_dir: Path,
    cohort: str,
    prompts: Sequence[str],
    normalisers: Sequence[str],
) -> tuple[list[Juror], dict[Juror, dict], list[dict]]:
    """``(jurors, per_juror, audit)`` over the normalisation run's re-normalised output.

    ``per_juror[juror]`` is ``{report: {sentence: {hpo}}}``, **unreduced and per segment**, which
    is the S0 pre-condition the whole analysis rests on. ``audit`` is one row per attempted cell
    carrying whether it was found and how much it holds, so a hole in the grid is a row in a table,
    not a condition that quietly scores zero.
    """
    jurors: list[Juror] = []
    per_juror: dict[Juror, dict] = {}
    audit: list[dict] = []
    # A juror admitted twice would occupy two bit positions and cast two votes, which inflates every
    # threshold silently. The caller can legitimately name the same reader twice, `normalisers`
    # plus `reference_normaliser` is the obvious way, so dedupe here, not trusting it.
    normalisers = list(dict.fromkeys(normalisers))
    prompts = list(dict.fromkeys(prompts))
    for prompt in prompts:
        cell_dir = Path(grid_dir) / cohort / prompt
        for normaliser in normalisers:
            paths = sorted(cell_dir.glob(f"detections_*__{normaliser}.jsonl"))
            if not paths:
                audit.append({"prompt": prompt, "normaliser": normaliser, "model": "",
                              "found": False, "n_reports": 0, "n_pairs": 0,
                              "note": f"no detections_*__{normaliser}.jsonl in {cell_dir}"})
                continue
            for path in paths:
                model = path.name[len("detections_"):-len(f"__{normaliser}.jsonl")]
                data = load_detections_file(path)
                n_pairs = sum(len(v) for r in data.values() for v in r.values())
                found = bool(data)
                audit.append({"prompt": prompt, "normaliser": normaliser, "model": model,
                              "found": found, "n_reports": len(data), "n_pairs": n_pairs,
                              "note": "" if found else "file present but empty"})
                if not found:
                    continue
                juror = Juror(model, prompt, normaliser)
                if juror in per_juror:
                    continue
                jurors.append(juror)
                per_juror[juror] = data
    jurors.sort()
    return jurors, per_juror, audit


# ── Packing: report -> unit -> juror -> terms ────────────────────────────────







# ── Voting ───────────────────────────────────────────────────────────────────









# ── Scoring ──────────────────────────────────────────────────────────────────

def score(pred, gold, report_ids, view, normalise_pair):
    """``(gold_sets, pred_sets)`` aligned on *report_ids*, through the fixed normalisation."""
    golds, preds = [], []
    for rid in report_ids:
        g, p, _ = normalise_pair(gold.get(rid, set()), pred.get(rid, set()), view)
        golds.append(g)
        preds.append(p)
    return golds, preds


# ── Selection ────────────────────────────────────────────────────────────────

class Config(NamedTuple):
    """One voting configuration.

    Attributes:
        rule: ``exact`` or ``closure_reduced``.
        unit: ``report``, ``window`` or ``segment``.
        k: minimum number of supporting jurors, from 1 to the jury size.
    """
    rule: str
    unit: str
    k: int


def select_config(
    packings: Mapping[tuple[str, str], Packed],
    gold,
    inner_folds: Sequence[Sequence[str]],
    subset: frozenset[int],
    view,
    normalise_pair,
    micro_prf,
    rules: Sequence[str],
    units: Sequence[str],
) -> tuple[Config, float]:
    """The ``(rule, unit, k)`` with the best mean inner-fold micro F1.

    Ties go to the simpler choice, in the order the caller listed the rules and units and then to
    the lower *k*, the same tie-break ``an earlier exploratory run`` uses, so a run of this with one prompt and one
    normaliser reproduces it, not merely resembling it.
    """
    best: tuple | None = None
    for rule_index, rule in enumerate(rules):
        for unit_index, unit in enumerate(units):
            packed = packings[(rule, unit)]
            per_fold: dict[int, list[float]] = {}
            for fold in inner_folds:
                for k, pred in predict_all_k(packed, fold, subset, rule, view).items():
                    g, p = score(pred, gold, fold, view, normalise_pair)
                    per_fold.setdefault(k, []).append(micro_prf(g, p)[2])
            for k, scores in per_fold.items():
                mean = sum(scores) / len(scores)
                key = (mean, -rule_index, -unit_index, -k)
                if best is None or key > best[0]:
                    best = (key, Config(rule, unit, k), mean)
    if best is None:
        raise ValueError("no configuration evaluated — empty rules or units")
    return best[1], best[2]


def backward_elimination(
    packings, gold, inner_folds, subset, view, normalise_pair, micro_prf, rules, units,
    epsilon: float,
) -> tuple[frozenset[int], Config, list[dict]]:
    """Drop the juror whose removal most helps, repeatedly. Keep the smallest jury within *epsilon*.

    Returns ``(chosen_subset, config, path)``. The path is every jury visited with its inner score,
    which is what makes "the procedure chose this" auditable, not asserted, and on
    ``an earlier exploratory run``'s run it is what showed the selection to be stable (43 of 50 outer folds chose the
    identical jury) and simultaneously worthless (+0.009 over the full pool, p = 0.167).
    """
    current = frozenset(subset)
    path: list[dict] = []
    while True:
        config, mean = select_config(packings, gold, inner_folds, current, view, normalise_pair,
                                     micro_prf, rules, units)
        path.append({"subset": sorted(current), "size": len(current), "rule": config.rule,
                     "unit": config.unit, "k": config.k, "inner_f1": mean})
        if len(current) <= 1:
            break
        best_drop, best_score = None, -1.0
        for juror in sorted(current):
            candidate = current - {juror}
            _config, candidate_mean = select_config(
                packings, gold, inner_folds, candidate, view, normalise_pair, micro_prf,
                rules, units)
            if candidate_mean > best_score:
                best_drop, best_score = juror, candidate_mean
        current = current - {best_drop}

    ceiling = max(step["inner_f1"] for step in path)
    # The tolerance is what stops the procedure chasing a 0.001 inner improvement bought with three
    # more jurors. Without it the "selected" jury is almost always the full pool.
    eligible = [step for step in path if step["inner_f1"] >= ceiling - epsilon]
    chosen = min(eligible, key=lambda step: step["size"])
    return (frozenset(chosen["subset"]),
            Config(chosen["rule"], chosen["unit"], chosen["k"]), path)


# ── The (prompt, normaliser) pair, selected inside the folds ─────────────────





class Arm(NamedTuple):
    """One out-of-fold estimate: which jurors may vote, and what the inner CV may choose for them.

    Every condition selects the (prompt, normaliser) pair inside the folds. The fields narrow the rest.
    ``models`` pins the jury to named models (``None`` is the pair's whole pool), by name, because
    a pool index means a different juror under every pair. ``select_jury`` runs backward
    elimination over those models. ``prompts`` narrows the candidate pairs, which is how S8 forces
    one prompt and still lets the folds choose its normaliser. ``rules``/``units`` narrow the vote
    space: a single juror is scored at exact/report/k=1, so the pair is the only thing chosen for it.
    """

    label: str
    models: tuple[str, ...] | None = None
    select_jury: bool = False
    prompts: tuple[str, ...] | None = None
    rules: tuple[str, ...] | None = None
    units: tuple[str, ...] | None = None


class FoldChoice(NamedTuple):
    """What one condition chose under one pair in one outer fold, and the inner score it chose it on."""

    pair: tuple[str, str]
    inner_f1: float
    models: tuple[str, ...]
    config: Config


class ConditionResult(NamedTuple):
    """One condition's out-of-fold outcome.

    ``pooled`` and ``assignment`` come from repetition 0 only (every report once);
    ``choices`` and ``inner_scores`` cover every outer fold of every repetition, which is what the
    selection frequencies are read from. ``union`` is the k=1 vote of the chosen pair's whole pool
    at the chosen rule and unit, the "named by anybody" set S9 separates vote from naming misses by.
    """

    pooled: dict[str, set[str]]
    choices: list[dict]
    inner_scores: list[dict]
    assignment: dict[str, FoldChoice]
    union: dict[str, set[str]]


def choose_pair(scores: Mapping[tuple[str, str], float],
                order: Sequence[tuple[str, str]]) -> tuple[str, str]:
    """The pair with the best inner score. A tie goes to the pair listed first in *order*.

    *order* is ``prompts x normalisers`` as configured, so the tie-break is fixed before any score
    exists, the same discipline as :func:`select_config`'s tie-break to the simpler rule.
    """
    if not scores:
        raise ValueError("no candidate pair was scored")
    rank = {pair: i for i, pair in enumerate(order)}
    return max(scores, key=lambda pair: (scores[pair], -rank.get(pair, len(rank))))


def evaluate_pair(pair, arms, jurors, per_juror, gold, folds, view, normalise_pair, micro_prf,
                  rules, units, epsilon, report_ids) -> dict[str, list]:
    """Every condition's inner-CV choice under ONE pair, for every outer fold.

    ``{arm label: [(FoldChoice, pred, union) or None per fold]}``. ``pred`` and ``union`` are filled
    only for repetition 0, the one that is pooled. The pair's packings are built here and dropped on
    return, so memory is one pair's worth however many pairs are compared, and the function is a
    unit of work a process pool can run per pair.

    The inner score a pair competes on is the best inner F1 the condition's own procedure reaches under
    it: ``select_config``'s mean for a fixed jury, and the ceiling of the elimination path for a
    selected one. The ceiling, not the chosen step, because the ε-rule is a statement about
    jury *size* within a pair, applying it across pairs would let a pair win for being prunable.
    """
    pool = [j for j in jurors if (j.prompt, j.normaliser) == tuple(pair)]
    live = [a for a in arms if a.prompts is None or pair[0] in a.prompts]
    out: dict[str, list] = {a.label: [None] * len(folds) for a in arms}
    if not pool or not live:
        return out
    for arm in live:
        missing = set(arm.rules or ()) - set(rules) | set(arm.units or ()) - set(units)
        if missing:
            raise ValueError(f"arm {arm.label!r} asks for {sorted(missing)}, which is not packed")
    packings = packings_for(pool, per_juror, report_ids, rules, units, view)
    everyone = frozenset(range(len(pool)))
    for arm in live:
        members = frozenset(i for i, juror in enumerate(pool)
                            if arm.models is None or juror.model in arm.models)
        if not members:
            continue
        condition_rules, condition_units = list(arm.rules or rules), list(arm.units or units)
        for f, row in enumerate(folds):
            inner = row["inner_folds"]
            if arm.select_jury:
                subset, config, path = backward_elimination(
                    packings, gold, inner, members, view, normalise_pair, micro_prf,
                    condition_rules, condition_units, epsilon)
                inner_f1 = max(step["inner_f1"] for step in path)
            else:
                subset = members
                config, inner_f1 = select_config(packings, gold, inner, subset, view,
                                                 normalise_pair, micro_prf, condition_rules, condition_units)
            pred = union = None
            if row["repetition"] == 0:
                pred = predict(packings, config.rule, config.unit, row["eval_ids"], subset,
                               config.k, view)
                union = predict(packings, config.rule, config.unit, row["eval_ids"], everyone, 1,
                                view)
            choice = FoldChoice(tuple(pair), inner_f1,
                                tuple(sorted(pool[i].model for i in subset)), config)
            out[arm.label][f] = (choice, pred, union)
    logger.info("pair (%s, %s): %d arm(s) x %d fold(s) selected", pair[0], pair[1], len(live),
                len(folds))
    return out


#: Read by :func:`_evaluate_pair_job` in a forked worker. Module state, not an argument
#: because the per-juror cache is the large object here and fork shares it copy-on-write, where
#: pickling it into every task would copy it once per pair.
_POOL_CONTEXT: dict = {}


def _evaluate_pair_job(pair):
    return pair, evaluate_pair(pair, **_POOL_CONTEXT)


def evaluate_conditions(arms: Sequence[Arm], pairs: Sequence[tuple[str, str]], jurors, per_juror, gold,
                  folds, view, normalise_pair, micro_prf, rules, units, epsilon, report_ids,
                  n_workers: int = 1) -> dict[str, ConditionResult]:
    """Nested CV with the (prompt, normaliser) pair as one more thing the inner folds choose.

    In each outer fold, every candidate pair runs the condition's own inner procedure on that fold's
    training split, and the pair with the best inner score is the one that predicts the held-out
    reports. Nothing about the pair is fixed on reports an outer fold later scores, which is the
    point: choosing it once on a development split that overlaps the evaluation folds put about
    80 % of the pooled reports behind the choice that scored them.

    The loop runs pair-outermost so each pair's packings are built once and shared by every condition,
    and with ``n_workers > 1`` pairs run in forked processes. The combination is done here, after
    every pair has reported, so the result does not depend on the order the workers finish in.
    """
    pairs = [tuple(p) for p in pairs]
    context = dict(arms=list(arms), jurors=jurors, per_juror=per_juror, gold=gold, folds=folds,
                   view=view, normalise_pair=normalise_pair, micro_prf=micro_prf,
                   rules=list(rules), units=list(units), epsilon=epsilon, report_ids=report_ids)
    by_pair: dict[tuple[str, str], dict[str, list]] = {}
    if n_workers > 1 and len(pairs) > 1:
        import multiprocessing
        _POOL_CONTEXT.update(context)
        try:
            with multiprocessing.get_context("fork").Pool(min(n_workers, len(pairs))) as workers:
                for pair, result in workers.imap_unordered(_evaluate_pair_job, pairs):
                    by_pair[tuple(pair)] = result
        finally:
            _POOL_CONTEXT.clear()
    else:
        for pair in pairs:
            by_pair[pair] = evaluate_pair(pair, **context)

    results: dict[str, ConditionResult] = {}
    for arm in arms:
        pooled: dict[str, set[str]] = {}
        union: dict[str, set[str]] = {}
        assignment: dict[str, FoldChoice] = {}
        choices, inner_scores = [], []
        for f, row in enumerate(folds):
            candidates = {pair: by_pair[pair][arm.label][f] for pair in pairs
                          if by_pair.get(pair, {}).get(arm.label, [None] * len(folds))[f]}
            if not candidates:
                raise ValueError(f"arm {arm.label!r}: no pair holds any of its jurors")
            pair = choose_pair({p: c[0].inner_f1 for p, c in candidates.items()}, pairs)
            choice, pred, condition_union = candidates[pair]
            for p, (c, _pred, _union) in candidates.items():
                inner_scores.append({"repetition": row["repetition"],
                                     "outer_fold": row["outer_fold"], "prompt": p[0],
                                     "normaliser": p[1], "inner_f1": round(c.inner_f1, 6),
                                     "chosen": p == pair})
            if row["repetition"] == 0:
                pooled.update(pred)
                union.update(condition_union)
                assignment.update({rid: choice for rid in row["eval_ids"]})
            choices.append({"repetition": row["repetition"], "outer_fold": row["outer_fold"],
                            "prompt": pair[0], "normaliser": pair[1],
                            "inner_f1": round(choice.inner_f1, 6),
                            "rule": choice.config.rule, "unit": choice.config.unit,
                            "k": choice.config.k, "size": len(choice.models),
                            "subset": list(choice.models)})
        results[arm.label] = ConditionResult(pooled, choices, inner_scores, assignment, union)
    return results


# ── The curated ground truth's segment evidence locations (S1) ──────────────────────────────────

#: Values of the sidecar's ``in_gold`` column that mean "this annotation is in the ground truth".
_IN_GOLD_TRUE = {"1", "true", "yes", "y", "t"}


def load_curated_annotations(path: Path, *, gold_only: bool = True) -> list[dict]:
    """``[{report_id, segment_idx, hpo_id, trigger}]`` from the ground-truth build's annotation sidecar.

    Two filters, and both change the number materially on ``curated_ground_truth_2026-09-12``:

    **``in_gold``.** The sidecar is the curation *record*, not the ground truth: it carries every annotation
    the pass considered, including the ones its policy excluded, each with an
    ``exclude_reason``. On that export 210 of 1649 rows are ``in_gold=0``. Counting them would put
    terms in the located denominator that are not in the ground truth at all, understating the rate by
    inflating what the ensemble was supposed to find. Pass ``gold_only=False`` only to measure the
    excluded set on purpose.

    **A located segment.** An annotation curation never located has no segment to be found in, so
    it cannot enter a *segment-scoped* denominator either way; :func:`anchored_generation_rate`
    reports how many were dropped so the shrinkage is visible, not assumed.

    A file with no ``in_gold`` column is read whole, so the loader still works on the older sidecar
    layout, but it logs that it could not apply the filter, because silently measuring a different
    denominator is the failure this docstring exists to prevent.
    """
    rows: list[dict] = []
    if not Path(path).exists():
        return rows
    n_total = n_excluded = n_unlocated = 0
    saw_in_gold = False
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            n_total += 1
            if "in_gold" in rec:
                saw_in_gold = True
                if gold_only and (rec.get("in_gold") or "").strip().lower()                         not in _IN_GOLD_TRUE:
                    n_excluded += 1
                    continue
            raw = (rec.get("segment_idx") or rec.get("segment") or "").strip()
            hpo = (rec.get("hpo_id") or rec.get("hpo_code") or "").strip()
            rid = (rec.get("patient_id") or rec.get("report_id") or "").strip()
            if not hpo or not rid:
                continue
            if not raw:
                n_unlocated += 1
                continue
            try:
                segment_idx = int(float(raw))
            except ValueError:
                n_unlocated += 1
                continue
            rows.append({"report_id": rid, "segment_idx": segment_idx, "hpo_id": hpo,
                         "trigger": (rec.get("trigger_word") or rec.get("trigger") or "").strip()})
    if gold_only and not saw_in_gold:
        logger.warning(
            "%s has no `in_gold` column — reading every annotation row. If this sidecar mixes "
            "excluded annotations with gold ones, the anchored denominator is too large.", path)
    logger.info(
        "curated annotations: %d row(s) -> %d located gold annotation(s) "
        "(%d excluded by in_gold, %d with no located segment)",
        n_total, len(rows), n_excluded, n_unlocated)
    return rows


def anchored_generation_rate(
    packed_segment: Packed, annotations: Sequence[dict], jurors_mask: frozenset[int],
    view, tolerance: int = 0,
) -> dict:
    """How often the pool named an annotated term **in the segment the curator put it in**.

    This is S1's third ceiling and it is the strictest of the three: the union recall asks whether
    anybody anywhere in the report produced the identifier, which a pool of 96 jurors emitting
    thousands of candidates can satisfy by accident. Restricting to the annotated segment is what
    separates "the method found this finding" from "the method emitted this identifier".

    The unit is the **annotated pair**, not the annotation row: a term located in two segments is one
    finding, located if the pool named it in either. ``n_annotation_rows`` is reported beside
    ``n_gold_pairs`` so the collapse is visible.

    ``coordinate_ok`` is reported and not assumed. The annotation's ``segment_idx`` and the run's
    ``sentence_number`` must index the *same* segmentation or this number is meaningless. The check
    here is the cheap necessary one, the named segment exists in the run's own segmentation for
    that report, and a low value means the two coordinate systems have drifted, not that recall is
    poor.
    """
    # The unit is the Annotated pair, not the annotation row. A term the curator located in two
    # different segments is one thing to find, and counting it twice would weight the pairs that
    # happen to be mentioned repeatedly. On curated_gold_2026-09-12 that is 1439 in-ground truth rows over
    # 1146 pairs, so the distinction moves the denominator by a quarter.
    by_pair: dict[tuple[str, str], set[int]] = {}
    for annotation in annotations:
        by_pair.setdefault(
            (annotation["report_id"], annotation["hpo_id"]), set()
        ).add(int(annotation["segment_idx"]))

    n_total = n_hit = n_in_report = n_coordinate_ok = n_unresolvable = 0
    for (rid, hpo_id), located in sorted(by_pair.items()):
        report = packed_segment.get(rid)
        if report is None:
            continue
        n_total += 1
        resolved = view.resolve(hpo_id)
        if resolved is None:
            # An identifier the fixed release does not know. It cannot be found by anybody, so it
            # is counted in the denominator and never in the numerator, and reported separately,
            # because a ground-truth set full of them would look like a recall failure.
            n_unresolvable += 1
            continue
        segments = {int(x) for x in report}
        # The coordinate check passes if ANY of the pair's annotated segments exists in the run's
        # own segmentation. A pair located only outside it is the drift this column detects.
        if located & segments:
            n_coordinate_ok += 1
        window = {seg + delta for seg in located
                  for delta in range(-tolerance, tolerance + 1)}
        anchored = any(
            any(resolved in terms
                for index, terms in report.get(segment, {}).items() if index in jurors_mask)
            for segment in window
        )
        elsewhere = any(
            any(resolved in terms
                for index, terms in jurors.items() if index in jurors_mask)
            for jurors in report.values()
        )
        n_hit += int(anchored)
        n_in_report += int(elsewhere)
    return {
        "n_gold_pairs": n_total,
        "n_annotation_rows": len(annotations),
        "n_unresolvable": n_unresolvable,
        "anchored_generation_rate": n_hit / max(1, n_total),
        "report_level_generation_rate": n_in_report / max(1, n_total),
        "coordinate_ok_rate": n_coordinate_ok / max(1, n_total),
        "tolerance": tolerance,
    }
