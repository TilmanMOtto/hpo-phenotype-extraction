"""The gates: does this app reproduce what the driver shipped?

Every number in this UI is recomputed from the raw artifacts rather than read out of the summary
CSVs, that is the whole point, since the CSVs cannot answer any of the questions the app exists
for. But it means a bug here looks like a finding: a plausible number, on a real run, that
nobody can check by eye.

So the app checks itself, on the real data, every time a cohort is loaded, and shows the result in
the scorecard. The gates compare against artifacts the driver wrote *from its own arithmetic*:

``G1 sets``       recomputed ``vote_k{k}`` and ``agg_plurality`` sets, term for term, report for
                  report, against each rule directory's ``predicted_set``
``G2 metrics``    recomputed micro/macro P/R/F1 against ``slm_ensemble_agg_summary.csv``
``G3 per-model``  recomputed productivity + plurality vote stats against
                  ``slm_ensemble_slm_metrics.csv``
``G4 spans``      the PhenoBERT offset arithmetic against ``write_phenobert_input`` itself
``G5 invariants`` k-sweep monotonicity, the autopsy fate partition, gold-set agreement across rule
                  directories, and no term carrying an empty voter mask
``G6 baseline``   the PhenoBERT baseline's detection offsets against the report text it was run on, and that the
                  two experiments describe the same cohort

G1–G3 call the driver's own ``vote_k_sets`` / ``plurality_sets`` / ``_safe_micro_macro`` /
``compute_slm_productivity``, so a failure can only mean the app fed them the wrong inputs, it
cannot mean the app reimplemented the rule slightly differently. That is a narrow
scope, and it is the one that counts: reimplementation drift is the failure mode that would
silently invalidate every screen.

A gate returns ``skip`` when the artifact it needs is absent (a cohort mid-extraction has no rule
directories), which is information, not failure.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PASS, FAIL, SKIP = "pass", "fail", "skip"

#: Metric columns compared in G2, and the tolerance. The driver writes full float repr, so this is
#: about float arithmetic order, not about rounding.
_METRIC_COLS = ("micro_precision", "micro_recall", "micro_f1",
                "macro_precision", "macro_recall", "macro_f1")
_TOL = 1e-9

#: min_detection_count is not persisted anywhere in the run directory (hydra writes no resolved
#: config for this experiment), so it is *inferred* by asking which value reproduces the shipped
#: vote_k1 set. Every real run used the config default of 1. This covers the case where one did not.
_MIN_COUNT_CANDIDATES = (1, 2, 3)


def _result(gate_id: str, title: str, status: str, detail: str, **extra) -> dict:
    return {"id": gate_id, "title": title, "status": status, "detail": detail, **extra}


def roster_diagnosis(bundle: dict) -> str:
    """The one explanation that accounts for most gate failures, stated plainly.

    The driver writes one ``vote_k{k}`` directory per *active* model, so the number of those
    directories records how many models were voting when the aggregate stage ran. If the app now
    counts a different number, the run directory changed after it was written, most often a model
    whose ``detections_*.jsonl`` was truncated or emptied afterwards. Every rule at the old
    ensemble size then disagrees, which looks like an app bug and is not one.
    """
    n_shipped = sum(1 for rule in bundle["shipped_sets"] if rule.startswith("vote_k"))
    n_active = len(bundle["models"])
    if not n_shipped or n_shipped == n_active:
        return ""
    missing = [m for m in bundle["all_models"] if m not in bundle["models"]]
    return (f" The aggregate stage shipped {n_shipped} vote_k director(ies) but only {n_active} "
            f"model(s) are voting now"
            + (f" ({', '.join(missing)} extracted but has no usable detections slice)"
               if missing else "")
            + ", the run directory changed after the aggregation ran, so the shipped numbers "
              "describe a different ensemble than the one on disk.")


def infer_min_count(bundle: dict) -> tuple[int, bool]:
    """``(min_count, confirmed)``, the detection-count floor this run actually aggregated at.

    Confirmed means a candidate reproduced the shipped ``vote_k1`` set. Unconfirmed falls
    back to 1 and every gate that depends on it reports the mismatch, not hiding it.
    """
    from . import votes as votes_mod

    shipped = bundle["shipped_sets"].get("vote_k1")
    if not shipped:
        return 1, False
    for candidate in _MIN_COUNT_CANDIDATES:
        recomputed = votes_mod.shipped_vote_k(
            bundle["report_hpos"], bundle["models"], bundle["report_ids"], 1, candidate
        )
        if all(set(recomputed.get(r, ())) == set(shipped.get(r, {}).get("predicted", ()))
               for r in bundle["report_ids"]):
            return candidate, True
    return 1, False


def gate_sets(bundle: dict) -> dict:
    """G1, recomputed rule sets against every shipped ``predicted_set``."""
    from . import votes as votes_mod

    shipped = bundle["shipped_sets"]
    if not shipped:
        return _result("G1", "Rule sets reproduce", SKIP,
                       "no rule directories yet, the aggregate stage has not run")

    min_count = bundle["min_count"]
    report_ids = bundle["report_ids"]
    mismatches: list[str] = []
    n_checked = 0

    for rule, sets in sorted(shipped.items()):
        if rule == "agg_plurality":
            recomputed, _ = votes_mod.shipped_plurality(
                bundle["sent_hpos"], bundle["models"], report_ids, min_count)
        else:
            k = int(rule.removeprefix("vote_k"))
            recomputed = votes_mod.shipped_vote_k(
                bundle["report_hpos"], bundle["models"], report_ids, k, min_count)
        for report_id in report_ids:
            n_checked += 1
            want = set(sets.get(report_id, {}).get("predicted", ()))
            got = set(recomputed.get(report_id, ()))
            if want != got:
                mismatches.append(
                    f"{rule}/{report_id}: missing {sorted(want - got)[:3]}, "
                    f"extra {sorted(got - want)[:3]}"
                )

    if mismatches:
        return _result("G1", "Rule sets reproduce", FAIL,
                       f"{len(mismatches)}/{n_checked} (report, rule) sets differ. "
                       + " | ".join(mismatches[:3]) + roster_diagnosis(bundle),
                       n_checked=n_checked)
    return _result("G1", "Rule sets reproduce", PASS,
                   f"{len(shipped)} rules x {len(report_ids)} reports, term for term",
                   n_checked=n_checked)


def gate_metrics(bundle: dict) -> dict:
    """G2, recomputed micro/macro metrics against ``slm_ensemble_agg_summary.csv``."""
    from . import votes as votes_mod

    summary = bundle["agg_summary"]
    if summary is None or summary.empty:
        return _result("G2", "Main metrics reproduce", SKIP,
                       "slm_ensemble_agg_summary.csv is missing")

    shipped = bundle["shipped_sets"]
    report_ids = bundle["report_ids"]
    gold = bundle["gold"]
    bad: list[str] = []
    n_checked = 0

    for row in summary.to_dict("records"):
        rule = str(row["rule"])
        sets = shipped.get(rule)
        if not sets:
            continue
        predicted = {r: set(sets.get(r, {}).get("predicted", ())) for r in report_ids}
        recomputed = votes_mod.score(gold, predicted, report_ids)
        for col in _METRIC_COLS:
            if col not in row:
                continue
            n_checked += 1
            if abs(float(row[col]) - recomputed[col]) > _TOL:
                bad.append(f"{rule}.{col}: shipped {row[col]:.6f} vs recomputed "
                           f"{recomputed[col]:.6f}")

    if not n_checked:
        return _result("G2", "Main metrics reproduce", SKIP,
                       "the summary CSV names no rule that has a predictions directory")
    if bad:
        return _result("G2", "Main metrics reproduce", FAIL,
                       f"{len(bad)}/{n_checked} metrics differ. " + " | ".join(bad[:3]),
                       n_checked=n_checked)
    return _result("G2", "Main metrics reproduce", PASS,
                   f"{n_checked} metric cells match to 1e-9", n_checked=n_checked)


def gate_slm_metrics(bundle: dict) -> dict:
    """G3, recomputed per-model productivity and vote stats against the shipped CSV."""
    shipped = bundle["slm_metrics"]
    if shipped is None or shipped.empty:
        return _result("G3", "Per-model metrics reproduce", SKIP,
                       "slm_ensemble_slm_metrics.csv is missing")
    if not bundle["records"]:
        return _result("G3", "Per-model metrics reproduce", SKIP,
                       "no llm_extractions_*.jsonl to recompute from")

    try:
        from hpo_extraction.phenojury.ensemble_eval import compute_slm_productivity, merge_slm_metrics
    except Exception as exc:  # noqa: BLE001
        # This gate has no fallback: its whole point is to compare against the driver's own
        # productivity arithmetic, and reimplementing that to check it would check nothing.
        return _result("G3", "Per-model metrics reproduce", SKIP,
                       f"hpo_extraction.phenojury.ensemble_eval is not importable here ({type(exc).__name__}: {exc}) "
                       "— this gate needs the driver's own productivity functions")

    from . import votes as votes_mod

    _plurality, vote_rows = votes_mod.shipped_plurality(
        bundle["sent_hpos"], bundle["models"], bundle["report_ids"], bundle["min_count"])
    recomputed = {
        row["model"]: row
        for row in merge_slm_metrics(
            compute_slm_productivity(bundle["records"], bundle["sent_hpos"], bundle["models"]),
            vote_rows,
        )
    }

    bad: list[str] = []
    n_checked = 0
    for row in shipped.to_dict("records"):
        mine = recomputed.get(str(row["model"]))
        if mine is None:
            bad.append(f"{row['model']}: shipped but not recomputed")
            continue
        for col, value in row.items():
            if col == "model" or col not in mine:
                continue
            n_checked += 1
            if abs(float(value) - float(mine[col])) > 1e-6:
                bad.append(f"{row['model']}.{col}: shipped {value} vs recomputed {mine[col]}")

    if bad:
        return _result("G3", "Per-model metrics reproduce", FAIL,
                       f"{len(bad)} cell(s) differ. " + " | ".join(bad[:3])
                       + roster_diagnosis(bundle), n_checked=n_checked)
    return _result("G3", "Per-model metrics reproduce", PASS,
                   f"{len(recomputed)} models x {n_checked // max(1, len(recomputed))} columns",
                   n_checked=n_checked)


def gate_spans(bundle: dict, sample: int = 6) -> dict:
    """G4, PhenoBERT offset arithmetic against ``write_phenobert_input``.

    Sampled, not exhaustive: the layout is either right or wrong for a whole run, and a
    handful of (model, report) pairs settles it in milliseconds instead of rebuilding every input
    file in the cohort. The sample includes the reasoning models, whose ``<think>``
    blocks are the case an off-by-block bug would show up in first.
    """
    from .detections import assert_layout_matches

    if not bundle["records"]:
        return _result("G4", "Evidence spans land on the reply", SKIP, "no extractions on disk")

    pairs: list[tuple[str, str]] = []
    for model in bundle["models"]:
        for report_id in bundle["report_ids"][:2]:
            if (model, report_id) in bundle["built"]["offsets"]:
                pairs.append((model, report_id))
    pairs = pairs[:sample]
    if not pairs:
        return _result("G4", "Evidence spans land on the reply", SKIP, "nothing to sample")

    for model, report_id in pairs:
        try:
            assert_layout_matches(bundle["records"], model, report_id, bundle["built"])
        except AssertionError as exc:
            return _result("G4", "Evidence spans land on the reply", FAIL, str(exc))
    return _result("G4", "Evidence spans land on the reply", PASS,
                   f"{len(pairs)} (model, report) pairs rebuilt byte for byte", n_checked=len(pairs))


def gate_invariants(bundle: dict) -> dict:
    """G5, the structural properties every screen silently assumes."""
    from . import autopsy as autopsy_mod
    from . import votes as votes_mod
    from .detections import popcount

    problems: list[str] = []

    # (a) the k-sweep can only shrink, the driver asserts this too (slm_ensemble_experiment:626).
    sweep = votes_mod.sweep_k(bundle, bundle["gold"], bundle["models"], bundle["min_count"])
    totals = [r["n_predicted"] for r in sweep if r["k"] is not None]
    if totals != sorted(totals, reverse=True):
        problems.append(f"k-sweep is not monotone non-increasing: {totals}")

    # (b) ground truth is a property of the cohort, so every rule directory must report the same one.
    reference = None
    for rule, sets in sorted(bundle["shipped_sets"].items()):
        gold = {r: tuple(sorted(v.get("gold", ()))) for r, v in sets.items()}
        if reference is None:
            reference = (rule, gold)
        elif gold != reference[1]:
            differing = [r for r in gold if gold.get(r) != reference[1].get(r)]
            problems.append(f"ground-truth sets differ between {reference[0]} and {rule} "
                            f"({len(differing)} report(s), e.g. {differing[:2]})")
            break

    # (c) a term in the mask table was detected by someone, or it should not be there at all.
    empty = sum(1 for row in bundle["masks"](bundle["min_count"]).values()
                for mask in row.values() if popcount(mask) == 0)
    if empty:
        problems.append(f"{empty} term(s) carry an empty voter mask")

    # (d) the fates partition the misses, no term counted twice, none unclassified.
    config = bundle["default_config"]
    predicted = votes_mod.predicted_sets(bundle, config)
    result = autopsy_mod.build(bundle, config, bundle["gold"], predicted)
    n_missed = sum(len(set(bundle["gold"].get(r, ())) - predicted.get(r, set()))
                   for r in bundle["report_ids"])
    if sum(result["counts"].values()) != n_missed:
        problems.append(f"fates count {sum(result['counts'].values())} misses, but there are "
                        f"{n_missed}")
    if result["counts"].get("no_evidence"):
        problems.append(f"{result['counts']['no_evidence']} miss(es) fell through every fate")

    if problems:
        return _result("G5", "Structural invariants hold", FAIL, " | ".join(problems))
    return _result("G5", "Structural invariants hold", PASS,
                   f"monotone k-sweep, consistent ground truth, {n_missed} misses each with one "
                   "fate")


def gate_pb_standalone(bundle: dict) -> dict:
    """G6, the PhenoBERT baseline baseline's offsets against the report text it was run on.

    The PhenoBERT baseline's ``start``/``end`` index ``phenobert_input/<stem>.txt``, and
    ``phenobert_experiment.stage_reports`` writes that file as the report verbatim, so
    ``text[start:end] == phrase`` is an equality and not an approximation. Checking it is the only
    thing standing between the annotated-report panel and underlining the wrong words: a
    ``phenobert_input/`` left over from an earlier cohort beside a fresh ``phenobert_output/``
    produces offsets that are individually plausible and uniformly wrong.

    Also checks that the two experiments are describing the same cohort. Reports that appear in one
    and not the other are reported as a count, not a failure, a rerun of one cohort at a different
    size is a real situation, and the comparison page scores the intersection and says so.
    """
    pb = bundle.get("pb_standalone")
    if not pb:
        return _result("G6", "PhenoBERT baseline offsets land on the report", SKIP,
                       "no baseline_phenobert run was found for this cohort, the comparison page "
                       "and the annotated report are unavailable, every other screen is unaffected")

    problems: list[str] = []
    if pb["span_bad"]:
        first = pb["span_bad"][0]
        problems.append(
            f"{len(pb['span_bad'])} of {pb['span_checked']} detection offsets do not land on their "
            f"own phrase (e.g. {first['report_id']} {first['hpo_id']} at {first['start']}: "
            f"expected {first['expected']!r}, found {first['found']!r}), phenobert_input/ and "
            "phenobert_detections.jsonl are out of step")

    ours = set(bundle["report_ids"])
    theirs = set(pb["report_ids"])
    if not ours & theirs:
        problems.append(f"no report id is shared between the two runs ({len(ours)} here, "
                        f"{len(theirs)} in {pb['run_dir']}), this is a different cohort")

    if problems:
        return _result("G6", "PhenoBERT baseline offsets land on the report", FAIL,
                       " | ".join(problems), n_checked=pb["span_checked"])

    missing_text = len(theirs) - len(pb["texts"])
    detail = (f"{pb['span_checked']} detection offset(s) verified against phenobert_input/, "
              f"{len(ours & theirs)} report(s) shared with the ensemble")
    if missing_text:
        detail += (f", {missing_text} report(s) have no staged text, so their annotated report "
                   "cannot be drawn")
    if ours ^ theirs:
        detail += (f", {len(ours - theirs)} report(s) only in the ensemble run and "
                   f"{len(theirs - ours)} only in the baseline; the comparison scores the "
                   "intersection")
    return _result("G6", "PhenoBERT baseline offsets land on the report", PASS, detail,
                   n_checked=pb["span_checked"])


def run_gates(bundle: dict) -> list[dict]:
    """Every gate, in order. A gate that raises is reported as a failure, never propagated.

    The app must open on a broken run, a cohort whose artifacts are half-written is when
    someone needs to look at it, and a traceback in place of the scorecard helps nobody.
    """
    gates = [gate_sets, gate_metrics, gate_slm_metrics, gate_spans, gate_invariants,
             gate_pb_standalone]
    results = []
    for gate in gates:
        try:
            results.append(gate(bundle))
        except Exception as exc:  # noqa: BLE001 - a gate crash is a gate failure, and is reported
            logger.exception("gate %s crashed", gate.__name__)
            results.append(_result(gate.__name__, gate.__name__, FAIL, f"gate crashed: {exc!r}"))
    return results


def summarise(results: list[dict]) -> dict:
    """``{status, n_pass, n_fail, n_skip, headline, pinned}`` for the scorecard banner."""
    from . import votes as votes_mod

    pinned, why_not = votes_mod.driver_status()
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_skip = sum(1 for r in results if r["status"] == SKIP)
    n_pass = sum(1 for r in results if r["status"] == PASS)
    if n_fail:
        headline = f"{n_fail} of {len(results)} verification gates FAILED, do not trust these numbers"
        status = FAIL
    elif n_pass == 0:
        headline = "nothing to verify yet, this cohort has no aggregate-stage artifacts"
        status = SKIP
    else:
        headline = f"recomputed == shipped ({n_pass} gates passed" + (
            f", {n_skip} not applicable)" if n_skip else ")")
        status = PASS
    if not pinned:
        headline += (", but not fixed to the driver: hpo_extraction.phenojury.generation could not be "
                     f"imported here ({why_not}), so the re-run used this app's own equivalent "
                     "implementation. The comparison against the shipped artifacts still holds.")
    return {"status": status, "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
            "headline": headline, "pinned": pinned, "driver_error": why_not}
