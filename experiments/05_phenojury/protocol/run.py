"""The PhenoJury protocol, PhenoJury when the prompt and the normaliser are also chosen inside the folds.


Analysis only: no model, no GPU. It reads the normalisation run's re-normalised detections, the curated ground truth and
the shared fold assignment.

Stages, named for the sections of the analysis plan so a table and the paragraph reading it carry
the same label:

``audit``        S0  what the cache actually holds, and whether the grid is complete.
``ceilings``     S1  union recall at k=1, the normaliser-union gap, the located generation rate.
``grid``         S2  prompt x normaliser, jointly, inside the training split.
``curves``       S3  the k sweep per prompt, with verbosity and parse-failure beside it.
``interaction``  S4  the model x prompt matrix and its two-way decomposition.
``selection``    S5  backward elimination over jury, rule, unit and k.
``diversity``    S6  the matched model-diversity against prompt-diversity contrast, and why.
``headline``     S7  one number from the nested procedure, plus the pre-specified test family.
``sensitivity``  S8  every prompt's score for the final jury, against the jury's own effect.
``report``       assemble results.md.
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hpo_extraction.evaluation.stats import (  # noqa: E402
    ReportResampler, bootstrap_reports, holm, paired_randomisation, read_folds,
)
from hpo_extraction.evaluation.prediction_sets import (  # noqa: E402
    read_prediction_sets, write_prediction_sets,
)
from hpo_extraction.evaluation.metrics import (OntologyView, error_taxonomy_existential,  # noqa: E402
                                       macro_prf_by_report, micro_prf, normalise_pair)

from protocol import (  # noqa: E402
    Arm, Juror, anchored_generation_rate, evaluate_conditions, load_curated_annotations, load_grid,
    max_counts, pack, predict_all_k, score, sets_at_k,
)

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "phenojury_protocol"
logger = logging.getLogger("exp14_04")


# ── Inputs ───────────────────────────────────────────────────────────────────

def load_gold(path: Path) -> dict[str, set[str]]:
    """The two-column curated ground truth. Same reader as an earlier exploratory run, so the two agree by design."""
    gold: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            codes = [c.strip() for c in (rec.get("hpo_codes") or "").split(";") if c.strip()]
            gold[rec["patient_id"]] = set(codes)
    return gold


def dev_ids(folds, cfg, report_ids) -> list[str]:
    """The reports the descriptive stages may look at.

    One fixed in advance training split rather than the whole cohort, because S2 to S6 are grids and a
    grid read on the reports S7 then evaluates on is a selection procedure wearing a description's
    clothes. Which split is fixed in config and therefore predates the tables.
    """
    if bool(cfg.dev_use_full_cohort):
        logger.warning("dev_use_full_cohort=true — every descriptive table below is IN-SAMPLE")
        return list(report_ids)
    for row in folds:
        if row["repetition"] == int(cfg.dev_repetition) and \
                row["outer_fold"] == int(cfg.dev_outer_fold):
            return sorted(row["train_ids"])
    raise ValueError(
        f"no fold row for repetition={cfg.dev_repetition} outer_fold={cfg.dev_outer_fold}")


def _slug(label: str) -> str:
    """A filename for a config label. ``single:medgemma`` -> ``single_medgemma``."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in label)


def write_table(rows: list[dict], path: Path, fieldnames=None) -> Path:
    """Write *rows* to a CSV at *path* (an empty file when there are no rows) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(fieldnames) if fieldnames else sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── Scoring helpers shared by every stage ────────────────────────────────────

class Scorer:
    """Everything a stage needs to turn a prediction into a number, in one object.

    Exists so the ontology view, the ground truth and the resampler are threaded, not rebuilt: one
    ``ReportResampler`` must be shared across every row of every table or the intervals are not
    paired and differences between them cannot be read.
    """

    def __init__(self, gold, view, report_ids, n_bootstrap: int):
        self.gold = gold
        self.view = view
        self.report_ids = list(report_ids)
        self.sampler = ReportResampler(len(self.report_ids), n_resamples=n_bootstrap, seed=0)

    def units(self, pred, report_ids=None):
        """``[(ground-truth set, predicted set)]`` per report, after normalisation."""
        ids = list(report_ids or self.report_ids)
        g, p = score(pred, self.gold, ids, self.view, normalise_pair)
        return list(zip(g, p))

    @staticmethod
    def prf(units):
        """Micro P/R/F1 only. This is the **inner-fold hot path**, the selection loop asks for it
        thousands of times per fold, so it stays the one cheap reduction and grows nothing."""
        return micro_prf([g for g, _ in units], [p for _, p in units])

    @staticmethod
    def metric_fn(units):
        """Micro and macro P/R/F1, for the reported rows only.

        Macro here is the mean of the per-report triples, matching ``comparison``'s
        ``t1_overall.csv`` and ``treephenorag_protocol``'s ladder, so the three chapters quote one convention.
        Computed inside the bootstrap's statistic, not beside it, which is what makes the
        macro figure provably over the same units as the micro one.
        """
        gold = [g for g, _ in units]
        pred = [p for _, p in units]
        p, r, f = micro_prf(gold, pred)
        ma_p, ma_r, ma_f, _ = macro_prf_by_report(gold, pred)
        return {"micro_precision": p, "micro_recall": r, "micro_f1": f,
                "macro_precision": ma_p, "macro_recall": ma_r, "macro_f1": ma_f}

    def interval(self, units):
        """Bootstrap CI, only valid on the full cohort, where the resampler's width matches.

        Off the full cohort the point estimates still stand and are returned without an interval,
        through the same ``metric_fn``: two hand-built dicts here is how one of them ends up
        missing a key the caller reads.
        """
        point = self.metric_fn(units)
        if len(units) != len(self.report_ids):
            out = {name: {"point": value} for name, value in point.items()}
            out["micro_f1"].update(lo=float("nan"), hi=float("nan"))
            return out
        return bootstrap_reports(self.metric_fn, units, resampler=self.sampler)

    def paired_interval(self, units_a, units_b):
        """``(delta, lo, hi)`` of micro F1, A minus B, both conditions resampled by the SAME draws.

        A non-significant test says nothing about how large a difference the data still allows. This interval does, which is what a claim of "similar" needs (check Q21). Both unit lists
        are over ``report_ids`` in order, so a resampled row picks the same reports for both conditions.
        """
        if not (len(units_a) == len(units_b) == len(self.report_ids)):
            return float("nan"), float("nan"), float("nan")

        def delta(pairs):
            return {"delta": self.prf([a for a, _ in pairs])[2] - self.prf([b for _, b in pairs])[2]}

        ci = bootstrap_reports(delta, list(zip(units_a, units_b)), resampler=self.sampler)["delta"]
        return ci["point"], ci["lo"], ci["hi"]


# ── S0 ───────────────────────────────────────────────────────────────────────

def stage_audit(cfg, jurors, audit_rows, per_juror, tables) -> list[dict]:
    """What the cache holds, and a hard stop if the S0 pre-condition is not met.

    Two things are checked and neither is a formality:

    * **Per-segment structure.** Everything below re-runs over ``{report: {sentence: {hpo}}}``. A
      cell reduced to report level cannot support S5's unit axis or S1's located rate, and would
      silently answer the segment questions with report-level numbers.
    * **Cell completeness.** an earlier exploratory run D1 records ``q7_recall x apertus`` failing grounding three
      times while its generation succeeded, which made that condition a seven-model ensemble. S4's matrix
      would have a hole and S6's "matched generation calls" would not be matched.
    """
    by_cell: dict[tuple[str, str], list[Juror]] = defaultdict(list)
    for juror in jurors:
        by_cell[(juror.prompt, juror.normaliser)].append(juror)

    rows = []
    sizes = {len(v) for v in by_cell.values()}
    for (prompt, normaliser), members in sorted(by_cell.items()):
        n_sent = sum(len(s) for j in members for s in per_juror[j].values())
        rows.append({
            "prompt": prompt, "normaliser": normaliser, "n_jurors": len(members),
            "models": ";".join(sorted(j.model for j in members)),
            "n_report_sentence_cells": n_sent,
            "per_segment": n_sent > 0,
        })
    write_table(rows, tables / "s0_cache_audit.csv")
    write_table(audit_rows, tables / "s0_cell_inventory.csv")

    flat = [r for r in rows if not r["per_segment"]]
    if flat:
        raise SystemExit(
            "S0 FAILED — these cells carry no (report, sentence) structure, so the unit axis and "
            f"the anchored generation rate are not computable from them: {flat}"
        )
    if len(sizes) > 1:
        logger.warning(
            "S0 — the grid is RAGGED: cells hold %s jurors. S4's matrix has a hole and S6's "
            "matched-call contrast is not matched; both stages report the imbalance rather than "
            "hiding it.", sorted(sizes))
    else:
        logger.info("S0 — grid complete: %d cell(s), %d juror(s) each", len(by_cell), sizes.pop())
    return rows


# ── S1 ───────────────────────────────────────────────────────────────────────

def stage_ceilings(cfg, jurors, per_juror, scorer, ids, view, tables) -> list[dict]:
    """The three numbers that bound everything downstream and involve no choice.

    Union recall at k=1 is the ceiling nothing below it exceeds. The normaliser-union gap is a
    **lower bound on normaliser loss**: an annotated term some reader recovers and the chosen one does not
    is lost to the reader, not to the model. The located rate is the strictest of the three, restricted to the segment the curator put the finding in, so a pool emitting thousands of
    candidates cannot satisfy it by accident.
    """
    rows = []

    def union_row(label, pool, normaliser_note=""):
        if not pool:
            return
        packed = pack(pool, per_juror, ids, "report", "exact", view)
        pred = sets_at_k(max_counts(packed, ids, frozenset(range(len(pool)))), 1)
        units = scorer.units(pred, ids)
        p, r, f = scorer.prf(units)
        rows.append({
            "ceiling": label, "n_jurors": len(pool), "note": normaliser_note,
            "micro_precision": round(p, 4), "micro_recall": round(r, 4), "micro_f1": round(f, 4),
            "preds_per_report": round(sum(len(x) for _, x in units) / max(1, len(units)), 1),
        })

    readers = list(dict.fromkeys(list(cfg.normalisers) + [str(cfg.reference_normaliser)]))
    for normaliser in readers:
        pool = [j for j in jurors if j.normaliser == normaliser]
        union_row(f"union@k=1, all prompts x models, {normaliser}", pool)

    all_grid = [j for j in jurors if j.normaliser in set(cfg.normalisers)]
    union_row("normaliser-union@k=1 (every reader)", all_grid,
              "the gap to the best single reader is a LOWER BOUND on normaliser loss")

    for prompt in list(cfg.prompts):
        pool = [j for j in jurors if j.prompt == prompt and j.normaliser in set(cfg.normalisers)]
        union_row(f"union@k=1, prompt {prompt}, every reader", pool)

    write_table(rows, tables / "s1_ceilings.csv")

    anchored = []
    if cfg.annotations_path:
        annotations = load_curated_annotations(Path(cfg.annotations_path))
        if not annotations:
            logger.warning("S1 — %s carried no locatable annotation; anchored rate skipped",
                           cfg.annotations_path)
        else:
            for normaliser in list(cfg.normalisers):
                pool = [j for j in jurors if j.normaliser == normaliser]
                if not pool:
                    continue
                packed = pack(pool, per_juror, ids, "segment", "exact", view)
                mask = frozenset(range(len(pool)))
                for tolerance in (0, 1):
                    stats = anchored_generation_rate(packed, annotations, mask, view, tolerance)
                    anchored.append({"normaliser": normaliser, "n_jurors": len(pool),
                                     **{k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in stats.items()}})
            write_table(anchored, tables / "s1_anchored_generation.csv")
    else:
        logger.info("S1 — no annotations_path; the anchored generation rate is skipped, not faked")
    return rows + anchored


# ── S2 ───────────────────────────────────────────────────────────────────────

def stage_grid(cfg, jurors, per_juror, scorer, dev, view, tables) -> list[dict]:
    """The 4 x 3 prompt x normaliser grid, at the full pool and a fixed reference configuration.

    The point of evaluating the pair, not picking a prompt and then a reader: the prompts
    differ in *output form*, canonical names, verbatim spans, full clauses, and the readers differ
    in which form they handle. A sequential choice locks in a confound. If the best prompt is the
    same under all three readers, say so and collapse to sequential selection. If not, the pair is
    the unit.
    """
    rows = []
    # The rule and unit are held at an earlier exploratory run's own selected point so the grid varies only the pair;
    # k is swept and the best is reported per cell. That best-k IS an oracle over k, but it is an
    # oracle *inside the descriptive split*, and S5/S7 select k in folds, so the grid stays a
    # description of the pair and the main stays honest. The `k` column records which won.
    reference_rule, reference_unit = "exact", "report"
    # The grid is scored on the development split, so `scorer.interval` (sized to the full cohort)
    # would return no interval. One resampler sized to `dev`, shared by every cell, keeps the
    # cells' intervals mutually comparable -- the same discipline the main's resampler keeps.
    dev_sampler = ReportResampler(len(dev), n_resamples=int(cfg.n_bootstrap), seed=0)
    for prompt in list(cfg.prompts):
        for normaliser in dict.fromkeys(list(cfg.normalisers)
                                        + [str(cfg.reference_normaliser)]):
            pool = [j for j in jurors if j.prompt == prompt and j.normaliser == normaliser]
            if not pool:
                rows.append({"prompt": prompt, "normaliser": normaliser, "n_jurors": 0,
                             "note": "cell absent"})
                continue
            packed = pack(pool, per_juror, dev, reference_unit, reference_rule, view)
            best = None
            for k, pred in predict_all_k(packed, dev, frozenset(range(len(pool))),
                                         reference_rule, view).items():
                units = scorer.units(pred, dev)
                p, r, f = scorer.prf(units)
                if best is None or f > best["micro_f1"]:
                    best = {"prompt": prompt, "normaliser": normaliser, "n_jurors": len(pool),
                            "k": k, "rule": reference_rule, "unit": reference_unit,
                            "micro_precision": round(p, 4), "micro_recall": round(r, 4),
                            "micro_f1": round(f, 4),
                            "is_reference_reader": normaliser == str(cfg.reference_normaliser),
                            "note": "", "_units": units}
            ci = bootstrap_reports(scorer.metric_fn, best.pop("_units"), resampler=dev_sampler)
            best["micro_f1_lo"] = round(ci["micro_f1"].get("lo", float("nan")), 4)
            best["micro_f1_hi"] = round(ci["micro_f1"].get("hi", float("nan")), 4)
            rows.append(best)
    write_table(rows, tables / "s2_prompt_normaliser_grid.csv")
    return _grid_report(cfg, rows)


def stage_grid_restricted(cfg, jurors, per_juror, gold, view, dev, tables) -> list[dict]:
    """T5.6's second block: the same grid on ground truth the PhenoBERT reader could not have influenced.

    See :func:`gold_provenance` for what the restriction actually is and why it is not literally
    "PhenoBERT-proposed pairs removed" -- PhenoBERT is not a ground truth source, so that set is empty,
    and the recorded provenance the threat maps onto is the curator's own suggestions.

    Scored with its own :class:`Scorer`, because the denominator changed: reusing the full-ground truth
    scorer would divide by the wrong number of pairs and quietly inflate every recall in the
    block.
    """
    if not cfg.get("annotations_path"):
        logger.info("S2 restricted — no annotations_path, so gold provenance is unavailable and "
                    "T5.6's second block is skipped (see docs/thesis_map.md)")
        return []
    exclude = {s.strip().lower() for s in (cfg.get("gold_exclude_sources") or [])}
    restricted, stats = gold_provenance(cfg.annotations_path, gold, view, exclude)
    if not restricted:
        logger.warning("S2 restricted — %s, so the second block is skipped",
                       stats.get("restriction"))
        return []
    logger.info("S2 restricted — dropped %d of %d gold pair(s) sourced only from %s; "
                "sidecar sources: %s", stats["n_removed"], stats["n_gold_pairs"],
                stats["restriction"], stats["sources"])
    if not stats["n_removed"]:
        logger.warning("S2 restricted — the restriction removed NOTHING. That is the finding: no "
                       "gold pair is attributable to %s alone, so the pre-annotation bias threat "
                       "has no recorded instance on this gold.", stats["restriction"])

    report_ids = sorted(restricted)
    scorer = Scorer(restricted, view, report_ids, int(cfg.n_bootstrap))
    ids = [r for r in dev if r in restricted]
    rows = []
    for prompt in list(cfg.prompts):
        for normaliser in dict.fromkeys(list(cfg.normalisers)
                                        + [str(cfg.reference_normaliser)]):
            pool = [j for j in jurors if j.prompt == prompt and j.normaliser == normaliser]
            if not pool:
                continue
            packed = pack(pool, per_juror, ids, "report", "exact", view)
            best = None
            for k, pred in predict_all_k(packed, ids, frozenset(range(len(pool))),
                                         "exact", view).items():
                units = scorer.units(pred, ids)
                p, r, f = scorer.prf(units)
                if best is None or f > best["micro_f1"]:
                    best = {"prompt": prompt, "normaliser": normaliser, "n_jurors": len(pool),
                            "k": k, "rule": "exact", "unit": "report",
                            "micro_precision": round(p, 4), "micro_recall": round(r, 4),
                            "micro_f1": round(f, 4),
                            "is_reference_reader": normaliser == str(cfg.reference_normaliser),
                            **{k2: v for k2, v in stats.items() if not isinstance(v, dict)}}
            if best:
                rows.append(best)
    write_table(rows, tables / "s2_gold_restricted.csv")
    return rows


def _grid_report(cfg, rows) -> list[dict]:
    """Log which prompt won under which reader, and hand the rows back unchanged.

    Split out of :func:`stage_grid` only so the restricted condition can reuse the grid body without
    re-emitting this, since the restricted ground truth answers a different question.
    """
    # The question the grid exists to answer, answered in the log, not left to the reader.
    winners = {}
    for normaliser in list(cfg.normalisers):
        cells = [r for r in rows if r.get("normaliser") == normaliser and r.get("micro_f1")]
        if cells:
            winners[normaliser] = max(cells, key=lambda r: r["micro_f1"])["prompt"]
    if winners:
        if len(set(winners.values())) == 1:
            logger.info("S2 — the best prompt is %s under ALL readers; selection may collapse to "
                        "sequential, and the chapter should say that was checked rather than "
                        "assumed", next(iter(winners.values())))
        else:
            logger.info("S2 — the best prompt DIFFERS by reader (%s); the (prompt, normaliser) "
                        "pair is the unit of selection", winners)
    return rows


# ── S3 ───────────────────────────────────────────────────────────────────────

def stage_curves(cfg, jurors, per_juror, scorer, dev, view, tables, normaliser) -> list[dict]:
    """Whole PR curves per prompt, plus the verbosity that moves the configuration along them.

    Prompts differ in how much they write, and findings-per-segment sets the configuration
    directly: a verbose prompt produces more candidates, which raises recall and lowers precision at
    fixed k. That is a shift along the curve, not a quality difference, so single-point F1
    comparisons between prompts are not interpretable and the curve is the comparison.
    """
    rows, diagnostics = [], []
    # Every (rule, scope) the selection may choose, not just the one an earlier exploratory run happened to fix.
    # The scope is what "two jurors agreed" MEANS: at report scope they may have read the term out
    # of different sentences, which is a weaker claim than segment-scope agreement and buys recall
    # at k > 1 that a stricter scope would refuse. If the curves rank differently under closure
    # than under exact then rule and scope are not separable, which is the finding F5.1 carries --
    # and it is not visible at a single fixed point.
    combinations = [(rule, unit) for rule in list(cfg.rules) for unit in list(cfg.units)]
    for prompt in list(cfg.prompts):
        pool = [j for j in jurors if j.prompt == prompt and j.normaliser == normaliser]
        if not pool:
            continue
        for rule, unit in combinations:
            packed = pack(pool, per_juror, dev, unit, rule, view)
            for k, pred in predict_all_k(packed, dev, frozenset(range(len(pool))),
                                         rule, view).items():
                units = scorer.units(pred, dev)
                p, r, f = scorer.prf(units)
                rows.append({"prompt": prompt, "normaliser": normaliser, "rule": rule,
                             "unit": unit, "n_jurors": len(pool), "k": k,
                             "micro_precision": round(p, 4), "micro_recall": round(r, 4),
                             "micro_f1": round(f, 4),
                             "n_predicted": sum(len(x) for _, x in units),
                             "preds_per_report": round(
                                 sum(len(x) for _, x in units) / max(1, len(units)), 1)})

        # Verbosity and silence, per juror. A prompt that wins on F1 while producing nothing for
        # two of its jurors is not winning, it is excluding them, which is why these columns sit
        # beside the curve, not in a separate note.
        for juror in pool:
            data = per_juror[juror]
            covered = [rid for rid in dev if data.get(rid)]
            n_segments = sum(len(data.get(rid, {})) for rid in dev)
            n_terms = sum(len(v) for rid in dev for v in data.get(rid, {}).values())
            diagnostics.append({
                "prompt": prompt, "normaliser": normaliser, "model": juror.model,
                "n_reports_with_any": len(covered), "n_reports": len(dev),
                "empty_report_rate": round(1 - len(covered) / max(1, len(dev)), 4),
                "n_segments_with_any": n_segments,
                "findings_per_segment": round(n_terms / max(1, n_segments), 3),
                "n_terms": n_terms,
            })
    write_table(rows, tables / "s3_curves.csv")
    write_table(diagnostics, tables / "s3_prompt_diagnostics.csv")
    return rows


# ── S4 ───────────────────────────────────────────────────────────────────────

def two_way_decomposition(matrix: dict[tuple[str, str], float]) -> dict:
    """Model, prompt and interaction sums of squares over the per-juror score matrix.

    A two-way decomposition without replication, so the residual **is** the interaction: there is
    one score per (model, prompt) cell and nothing else for the residual to absorb. Reported as
    shares of the total, because the absolute sums of squares mean nothing on their own and the
    branch this stage exists to decide is which effect dominates.
    """
    models = sorted({m for m, _ in matrix})
    prompts = sorted({p for _, p in matrix})
    cells = [matrix[(m, p)] for m in models for p in prompts if (m, p) in matrix]
    if not cells or len(cells) < len(models) * len(prompts):
        return {"complete": False, "n_cells": len(cells),
                "n_expected": len(models) * len(prompts),
                "note": "a missing cell is not a zero, so no decomposition is reported"}
    if len(models) < 2 or len(prompts) < 2:
        # With one row or one column the residual has no degrees of freedom, so the decomposition
        # would attribute all variance to whichever axis survives, a fact about the run's shape,
        # not about the ensemble. Refuse, not print 1.000.
        return {"complete": False, "n_models": len(models), "n_prompts": len(prompts),
                "n_cells": len(cells), "n_expected": len(cells),
                "note": "degenerate matrix: a two-way decomposition needs at least 2 models "
                        "and 2 prompts"}
    grand = sum(cells) / len(cells)
    row_mean = {m: sum(matrix[(m, p)] for p in prompts) / len(prompts) for m in models}
    col_mean = {p: sum(matrix[(m, p)] for m in models) / len(models) for p in prompts}
    ss_model = len(prompts) * sum((row_mean[m] - grand) ** 2 for m in models)
    ss_prompt = len(models) * sum((col_mean[p] - grand) ** 2 for p in prompts)
    ss_total = sum((v - grand) ** 2 for v in cells)
    ss_inter = ss_total - ss_model - ss_prompt
    return {
        "complete": True, "n_models": len(models), "n_prompts": len(prompts),
        "grand_mean": round(grand, 4),
        "ss_model": round(ss_model, 6), "ss_prompt": round(ss_prompt, 6),
        "ss_interaction": round(ss_inter, 6), "ss_total": round(ss_total, 6),
        "share_model": round(ss_model / ss_total, 4) if ss_total else 0.0,
        "share_prompt": round(ss_prompt / ss_total, 4) if ss_total else 0.0,
        "share_interaction": round(ss_inter / ss_total, 4) if ss_total else 0.0,
    }


def stage_interaction(cfg, jurors, per_juror, scorer, dev, view, tables, normaliser) -> list[dict]:
    """The model x prompt matrix of single-juror scores, and what drives its variance.

    Weak interaction justifies a global prompt. Strong interaction means a global prompt is the
    wrong design and per-juror assignment is right in principle, but that selects over 4^8, so it
    enters S5 as a restricted variant on the elimination path and inside the training split only,
    never as the main.

    ``findings_per_segment`` sits in the same table on purpose: an interaction driven by a juror
    that cannot follow one prompt's output contract is a formatting artefact, not a substantive one,
    and an earlier exploratory run measured openbiollm at 0.008 format compliance on q3 and 0.401 on q4.
    """
    rows = []
    matrix: dict[tuple[str, str], float] = {}
    for juror in jurors:
        if juror.normaliser != normaliser:
            continue
        packed = pack([juror], per_juror, dev, "report", "exact", view)
        pred = sets_at_k(max_counts(packed, dev, frozenset({0})), 1)
        units = scorer.units(pred, dev)
        p, r, f = scorer.prf(units)
        data = per_juror[juror]
        n_segments = sum(len(data.get(rid, {})) for rid in dev)
        n_terms = sum(len(v) for rid in dev for v in data.get(rid, {}).values())
        matrix[(juror.model, juror.prompt)] = f
        rows.append({"model": juror.model, "prompt": juror.prompt, "normaliser": normaliser,
                     "micro_precision": round(p, 4), "micro_recall": round(r, 4),
                     "micro_f1": round(f, 4),
                     "findings_per_segment": round(n_terms / max(1, n_segments), 3),
                     "n_segments_with_any": n_segments})
    write_table(rows, tables / "s4_interaction.csv")
    decomposition = two_way_decomposition(matrix)
    write_table([decomposition], tables / "s4_decomposition.csv")
    if decomposition.get("complete"):
        logger.info("S4 — variance shares: model %.3f, prompt %.3f, interaction %.3f",
                    decomposition["share_model"], decomposition["share_prompt"],
                    decomposition["share_interaction"])
    else:
        logger.warning("S4 — no decomposition (%d of %d cells): %s",
                       decomposition.get("n_cells"), decomposition.get("n_expected"),
                       decomposition.get("note", ""))
    return rows


# ── S5 / S7 ──────────────────────────────────────────────────────────────────

def grid_pairs(cfg) -> list[tuple[str, str]]:
    """The candidate (prompt, normaliser) pairs, in the order ties are broken in.

    The reference reader is not a candidate: it reads the whole generation where the others read
    parsed candidates, so it would compare a parser against a linker (see config).
    """
    return [(p, n) for p in cfg.prompts for n in cfg.normalisers]


def plan_conditions(cfg, stages, models) -> list[Arm]:
    """Every out-of-fold estimate the requested stages will read, so one pass can compute them all.

    One pass, not one per stage because the pair loop is the expensive part: each pair's
    packings are built once and every condition is selected against them.
    """
    arms: list[Arm] = []
    singles = ("headline" in stages) or ("lomo" in stages)
    if "selection" in stages:
        arms.append(Arm("phenojury", select_jury=True))
    if "headline" in stages:
        arms.append(Arm("full_pool"))
        for label, wanted in (OmegaConf.to_container(cfg.get("fixed_juries")) or {}).items():
            missing = sorted(set(wanted) - set(models))
            if missing:
                logger.warning("S7 — fixed jury %r: %s not in the pool; scoring the rest",
                               label, ", ".join(missing))
            present = tuple(sorted(set(wanted) & set(models)))
            if present:
                arms.append(Arm(f"fixed:{label}", models=present))
            else:
                logger.warning("S7 — fixed jury %r: no member is in the pool, skipping", label)
    if singles:
        # A jury of one admits only k=1, and the vote unit cannot matter to one voter, so the
        # rule and unit are fixed: the pair is the only thing the folds choose for a single juror.
        arms.extend(Arm(f"single:{m}", models=(m,), rules=("exact",), units=("report",))
                    for m in models)
    if "lomo" in stages:
        arms.extend(Arm(f"lomo:{m}", models=tuple(x for x in models if x != m)) for m in models)
    if "sensitivity" in stages:
        arms.extend(Arm(f"prompt:{p}", prompts=(p,)) for p in cfg.prompts)
    return arms


def pair_frequency(choices: list[dict]) -> list[dict]:
    """How often each prompt, normaliser and pair was chosen across every outer fold."""
    rows = []
    for axis, key in (("prompt", lambda c: c["prompt"]), ("normaliser", lambda c: c["normaliser"]),
                      ("pair", lambda c: f"{c['prompt']}|{c['normaliser']}")):
        for value, count in Counter(key(c) for c in choices).most_common():
            rows.append({"axis": axis, "value": value, "n_folds": count,
                         "share": round(count / max(1, len(choices)), 4)})
    return rows


def modal_pair(choices: list[dict]) -> tuple[str, str] | None:
    """The most frequent (prompt, normaliser) pair among fold choices, or None."""
    counts = Counter((c["prompt"], c["normaliser"]) for c in choices)
    return counts.most_common(1)[0][0] if counts else None


# ── S6 ───────────────────────────────────────────────────────────────────────

def stage_lomo(results, models, jurors, per_juror, scorer, report_ids, tables,
               baseline_units) -> list[dict]:
    """T5.7, what the jury loses without each member, beside that member's own quality.

    The pairing is the whole point. A juror can be individually weak and still carry the jury, if
    what it gets wrong is not what the others get wrong. And a juror can be individually strong and
    contribute nothing, because two other members already cover everything it finds. Neither case
    is visible in a main number, and the leave-one-out column is the only place either shows
    up. This is the mechanism behind H5.2.

    Each condition is re-selected out of fold: the jury is fixed to ``pool \\ {juror}`` and the pair,
    rule, scope and k are still chosen inside each training split, as the full-pool condition
    does. Removing a juror and reusing the *full* jury's choices would confound "this juror
    mattered" with "the remaining seven wanted a different threshold", which is a different claim.
    """
    base_f1 = Scorer.prf(baseline_units)[2] if baseline_units else float("nan")
    by_key = {(j.model, j.prompt, j.normaliser): j for j in jurors}
    rows = []
    for model in models:
        without, alone = results.get(f"lomo:{model}"), results.get(f"single:{model}")
        if without is None or alone is None:
            continue
        units = scorer.units(without.pooled, report_ids)
        p, r, f = Scorer.prf(units)

        # That model alone, out of fold, at the pair each training split chose for it.
        sp, sr, sf = Scorer.prf(scorer.units(alone.pooled, report_ids))

        # Verbosity belongs to one (model, prompt, normaliser) cell, so it is read at the pair the
        # folds chose most often for this model alone, and the row names that pair.
        prompt, normaliser = modal_pair(alone.choices)
        juror = by_key[(model, prompt, normaliser)]
        data = per_juror[juror]
        n_segments = sum(len(data.get(rid, {})) for rid in report_ids)
        n_terms = sum(len(v) for rid in report_ids for v in data.get(rid, {}).values())
        covered = [rid for rid in report_ids if data.get(rid)]
        rows.append({
            "model": model,
            "solo_modal_prompt": prompt,
            "solo_modal_normaliser": normaliser,
            "jury_micro_f1_without": round(f, 4),
            "delta_micro_f1": round(f - base_f1, 4) if base_f1 == base_f1 else "",
            "jury_micro_precision_without": round(p, 4),
            "jury_micro_recall_without": round(r, 4),
            "micro_precision": round(sp, 4),
            "micro_recall": round(sr, 4),
            "micro_f1": round(sf, 4),
            "candidates_per_segment": round(n_terms / max(1, n_segments), 3),
            "empty_answer_rate": round(1 - len(covered) / max(1, len(report_ids)), 4),
            # T5.7 asks for a parse-failure rate and it is NOT derivable from these inputs.
            # The normalisation run's detections are one row per successful detection
            # (report_id, model, sentence_number, hpo_id, count), so a generation that failed to
            # parse leaves no row -- and is therefore indistinguishable from a model that read the
            # sentence and correctly found nothing. Emitting the empty-answer rate under a
            # parse-failure heading would conflate a bug with a judgement, so the column is left
            # blank and the reason is logged once below. Closing it needs the normalisation run to record
            # failures, not a change here.
            "parse_failure_rate": "",
            "n_segments_with_any": n_segments,
            "modal_size": Counter(c["size"] for c in without.choices).most_common(1)[0][0],
            "modal_pair_without": "|".join(modal_pair(without.choices)),
        })
        logger.info("S6 LOMO without %-22s jury F1=%.4f (delta %+.4f) | own F1=%.4f",
                    model, f, f - base_f1 if base_f1 == base_f1 else float("nan"), sf)
    logger.info("S6 LOMO — parse_failure_rate is left EMPTY: exp14_03's detections record only "
                "successful detections, so a failed parse leaves no row and cannot be told apart "
                "from a model that correctly found nothing. See docs/thesis_map.md.")
    write_table(rows, tables / "s6_lomo.csv")
    return rows


#: The three places an annotated term can be lost by a vote over candidate sets, in the order T5.8
#: reports them. They are exhaustive over false negatives by design: a missed term either
#: had a descendant or ancestor named (specificity), or was named by too few jurors to reach k
#: (vote), or was named by nobody at all (naming).
JURY_BUCKETS = ("specificity", "vote", "naming")


def stage_jury_decomposition(gold, view, report_ids, tables, predicted, union,
                             tables_name="s9_recall_decomposition.csv"):
    """T5.8, where the jury's misses are, split by the stage that lost them.

    The split counts because it says what to do next, and the main recall does not
    distinguish the three: a **vote** miss is addressable by the aggregation rule, a **naming**
    miss by the normaliser or the prompt, and a **specificity** miss by neither -- it is a
    disagreement about how precise an answer the report actually supports.

    ``vote`` and ``naming`` are separable here without a manual sample, and that is worth stating
    because the analysis plan allowed for them staying merged: the per-juror candidate sets are on
    disk, so "some juror named it and fewer than k did" and "no juror named it" are decidable by
    counting, not by reading. ``merged`` is therefore written False. It would flip to True
    only if the per-juror sets were unavailable, in which case the table says so in one sentence
    instead of inventing a boundary.

    *union* is every term any juror of the pool proposed -- the k=1 vote -- taken per report at the
    pair, rule and scope that report's own outer fold chose. A term outside it was named by nobody.
    """
    counts = Counter()
    for rid in report_ids:
        gold_set, pred_set = normalise_pair(gold.get(rid, set()), predicted.get(rid, set()),
                                            view)[:2]
        proposed = union.get(rid, set())
        for term in gold_set - pred_set:
            resolved = view.resolve(term) or term
            # Specificity first: if the jury answered somewhere on this term's own path, it found
            # The finding and disagreed about its level. Filing that as a naming miss would blame
            # The normaliser for a judgement the jury actually made.
            # `descendants_or_self`, not `descendants` -- the latter does not exist on
            # OntologyView, and this stage had never run, so the name went unnoticed. Both
            # accessors are reflexive, which is why `resolved` is subtracted back out below.
            relatives = set(view.ancestors(resolved)) | set(view.descendants_or_self(resolved))
            if (relatives - {resolved}) & pred_set:
                counts["specificity"] += 1
            elif resolved in {view.resolve(p) or p for p in proposed}:
                counts["vote"] += 1
            else:
                counts["naming"] += 1

    total = sum(counts.values())
    rows = [{"bucket": b, "count": counts.get(b, 0),
             "share": round(counts.get(b, 0) / total, 4) if total else 0.0,
             "merged": False}
            for b in JURY_BUCKETS]
    write_table(rows, tables / tables_name,
                fieldnames=["bucket", "count", "share", "merged"])
    logger.info("S9 — jury recall decomposition over %d false negative(s): %s", total,
                {b: counts.get(b, 0) for b in JURY_BUCKETS})
    return rows


def stage_jury_error_taxonomy(gold, view, report_ids, tables, predicted,
                              tables_name="s9_error_taxonomy.csv"):
    """S9's other half, what the selected jury's false positives are.

    The same function the TreePhenoRAG protocol writes chapter 4's ``error_taxonomy.csv`` with, so the two methods'
    bars are filed by one rule. Each report goes through ``normalise_pair`` first, the policy every
    score in this experiment is taken under: an out-of-subtree prediction is not scored and so is
    not a false positive, while an identifier the ontology does not know is kept and lands in
    ``invalid``, not disappearing. The false-positive total therefore equals the one behind
    s7_headline's precision.
    """
    gold_sets, pred_sets = [], []
    for rid in report_ids:
        g, p = normalise_pair(gold.get(rid, set()), predicted.get(rid, set()), view)[:2]
        gold_sets.append(g)
        pred_sets.append(p)
    taxonomy = error_taxonomy_existential(gold_sets, pred_sets, view)
    rows = [{"bucket": b, "count": taxonomy["counts"][b], "fraction": taxonomy["fractions"][b]}
            for b in taxonomy["counts"]]
    write_table(rows, tables / tables_name, fieldnames=["bucket", "count", "fraction"])
    logger.info("S9 — jury false-positive taxonomy over %d false positive(s): %s",
                taxonomy["n_fp"], taxonomy["counts"])
    return rows


def stage_call_budget(cfg, jurors, per_juror, pool, report_ids, tables) -> list[dict]:
    """The generation-call budget, for the numbers the prose quotes inline.

    NOT the number of generations run. ``per_juror`` is the NORMALISED payload, which has an entry
    for a segment only where a juror's output linked to at least one term, so a sentence every juror
    answered with nothing is not counted although it cost J calls. On GSC+ that is 3.3 of the 6.9
    sentences of an abstract (check X25). The generations actually run are the comparison's
    ``t4_cost.csv`` ``llm_calls_per_report``. Quote that, not this.
    """
    if not pool:
        return []
    # Segments per report, from the juror with the most complete coverage: the per-juror payload
    # is keyed by segment, so the largest set is the segmentation the run actually used.
    per_report = {}
    for rid in report_ids:
        per_report[rid] = max((len(per_juror[j].get(rid, {})) for j in pool), default=0)
    n_segments = sum(per_report.values())
    rows = [{
        "n_jurors": len(pool),
        "n_reports": len(report_ids),
        "mean_segments_per_report": round(n_segments / max(1, len(report_ids)), 3),
        "mean_calls_per_report": round(len(pool) * n_segments / max(1, len(report_ids)), 1),
        "total_calls": len(pool) * n_segments,
        "median_segments_per_report": float(
            sorted(per_report.values())[len(per_report) // 2]) if per_report else 0.0,
        "max_segments_per_report": max(per_report.values(), default=0),
    }]
    write_table(rows, tables / "s0_call_budget.csv")
    logger.info("call budget: %d juror(s) x %.1f segment(s) = %.0f generation(s) per report",
                len(pool), rows[0]["mean_segments_per_report"], rows[0]["mean_calls_per_report"])
    return rows


def gold_provenance(annotations_path, gold, view, exclude_sources) -> tuple[dict, dict]:
    """``(restricted_gold, stats)``, the ground truth with pairs from ``exclude_sources`` removed.

    **This is not quite the restriction T5.6 asks for, and the difference is worth stating.** The
    pre-annotation bias threat is that part of the ground truth was built by curating PhenoBERT's own
    suggestions, so a PhenoBERT-based normaliser is graded partly against its own proposals. But
    PhenoBERT is *not* a ground truth source: the ground-truth build's policy admits ``prior_annotation``, ``daphne`` and a
    curator's own ``suggestion``, and the curation UI draws PhenoBERT as reference only, never
    giving it a verdict. So no annotated pair carries a recorded PhenoBERT provenance and the literal
    restriction has nothing to remove.

    What *is* recorded, and what this restricts on, is the ``source`` column. The sharpest
    available version of the threat is the curator's own suggestions: those were written with
    PhenoBERT's output on screen, so they are the rows whose existence PhenoBERT could plausibly
    have caused. ``prior_annotation`` and ``daphne`` are independent annotation files that predate the
    curation pass. Dropping the suggestions therefore answers a real question -- does the
    PhenoBERT reader's advantage survive on ground truth it could not have influenced? -- and the table
    says which restriction it applied, not implying the other one.
    """
    stats = {"n_gold_pairs": sum(len(v) for v in gold.values()), "n_removed": 0,
             "sources": {}, "restriction": ",".join(sorted(exclude_sources))}
    if not annotations_path or not Path(annotations_path).exists():
        return {}, stats

    drop: set[tuple[str, str]] = set()
    seen = Counter()
    with Path(annotations_path).open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if "source" not in rec:
                return {}, {**stats, "restriction": "no source column in the sidecar"}
            rid = (rec.get("patient_id") or rec.get("report_id") or "").strip()
            hpo = (rec.get("hpo_code") or rec.get("hpo_id") or "").strip()
            source = (rec.get("source") or "").strip().lower()
            if not (rid and hpo):
                continue
            seen[source] += 1
            if source in exclude_sources:
                drop.add((rid, view.resolve(hpo) or hpo))
    stats["sources"] = dict(seen)

    # A pair is kept unless EVERY row recording it came from an excluded source: two files
    # recording one annotation, one of them independent, is an independently-attested pair.
    keep_anyway = set()
    with Path(annotations_path).open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            source = (rec.get("source") or "").strip().lower()
            if source in exclude_sources:
                continue
            rid = (rec.get("patient_id") or rec.get("report_id") or "").strip()
            hpo = (rec.get("hpo_code") or rec.get("hpo_id") or "").strip()
            if rid and hpo:
                keep_anyway.add((rid, view.resolve(hpo) or hpo))
    drop -= keep_anyway

    restricted = {}
    for rid, terms in gold.items():
        kept = {t for t in terms if (rid, view.resolve(t) or t) not in drop}
        restricted[rid] = kept
    stats["n_removed"] = stats["n_gold_pairs"] - sum(len(v) for v in restricted.values())
    stats["share_gold_from_phenobert"] = round(
        stats["n_removed"] / max(1, stats["n_gold_pairs"]), 4)
    return restricted, stats


def error_diversity(a_units, b_units) -> dict:
    """Jaccard of the two jurors' false-positive sets, and the correlation of their decisions.

    Voting can only help where errors decorrelate, so this is the mechanism behind any ranking in
    S6 and the part worth publishing. ``phi`` is the correlation of the two binary decision vectors
    over the union of every ``(report, term)`` either juror touched; ``fp_jaccard`` is the sharper
    number, because agreeing on a *mistake* is what a vote cannot fix.
    """
    fp_a, fp_b, all_a, all_b = set(), set(), set(), set()
    for index, ((gold_a, pred_a), (_gold_b, pred_b)) in enumerate(zip(a_units, b_units)):
        for term in pred_a:
            all_a.add((index, term))
            if term not in gold_a:
                fp_a.add((index, term))
        for term in pred_b:
            all_b.add((index, term))
            if term not in gold_a:
                fp_b.add((index, term))
    union_fp = fp_a | fp_b
    universe = all_a | all_b
    n11 = len(all_a & all_b)
    n10 = len(all_a - all_b)
    n01 = len(all_b - all_a)
    n = len(universe)
    # Phi over the union universe: n00 is 0 by design there, so this is the Jaccard-like
    # agreement on what was predicted, not a full 2x2 correlation, and is labelled as such.
    return {
        "fp_jaccard": round(len(fp_a & fp_b) / len(union_fp), 4) if union_fp else 0.0,
        "pred_jaccard": round(n11 / n, 4) if n else 0.0,
        "n_fp_a": len(fp_a), "n_fp_b": len(fp_b), "n_fp_shared": len(fp_a & fp_b),
        "n_pred_only_a": n10, "n_pred_only_b": n01, "n_pred_both": n11,
    }


def stage_diversity(cfg, jurors, per_juror, scorer, dev, view, tables, normaliser,
                    best_model: str, best_prompt: str) -> list[dict]:
    """Cells A-D at matched jury size, and the pairwise error diversity that explains them.

    Cell A is one juror and is the **true floor**, because greedy decoding makes four copies of one
    juror identical, self-consistency is unavailable here, so the single juror is the correct
    control. That is why this contrast is interesting, not a rediscovery of self-consistency,
    and the write-up has to say it.

    C and D use the single best model so no cell is carried by having a stronger member.
    """
    size = int(cfg.diversity_jury_size)
    prompts = list(cfg.prompts)
    by_prompt_model = {(j.prompt, j.model): j for j in jurors if j.normaliser == normaliser}

    def cell(label, members, composition, expected):
        """One cell, with an explicit verdict on whether it is actually matched.

        A cell built from two jurors where four were asked for is not a weaker version of the
        contrast, it is a different one, the comparison is *at matched generation calls*, and a
        short cell breaks that. So the shortfall is a column, not a silence.
        """
        members = [m for m in members if m is not None]
        # Duplicates arise legitimately: cell C asks for one model under four prompts, and if only
        # one prompt was run they are the same juror. Collapsing them is right (greedy decoding
        # makes copies identical) and the count must then show 1, not 4.
        unique = list(dict.fromkeys(members))
        if not unique:
            return None
        packed = pack(unique, per_juror, dev, "report", "exact", view)
        best = None
        for k, pred in predict_all_k(packed, dev, frozenset(range(len(unique))),
                                     "exact", view).items():
            units = scorer.units(pred, dev)
            p, r, f = scorer.prf(units)
            if best is None or f > best["micro_f1"]:
                best = {"cell": label, "composition": composition, "n_jurors": len(unique),
                        "n_expected": expected, "matched": len(unique) == expected,
                        "note": "" if len(unique) == expected else
                                f"only {len(unique)} distinct juror(s) available where {expected} "
                                f"were asked for — this cell is NOT matched and must not be "
                                f"compared with the others on generation cost",
                        "k": k, "micro_precision": round(p, 4), "micro_recall": round(r, 4),
                        "micro_f1": round(f, 4),
                        "members": ";".join(sorted(m.key for m in unique))}
        return best

    # Model diversity: the `size` best models on the selected prompt, ranked on the dev split.
    def juror_f1(juror):
        packed = pack([juror], per_juror, dev, "report", "exact", view)
        pred = sets_at_k(max_counts(packed, dev, frozenset({0})), 1)
        return scorer.prf(scorer.units(pred, dev))[2]

    same_prompt = [j for j in jurors if j.normaliser == normaliser and j.prompt == best_prompt]
    ranked_models = sorted(same_prompt, key=juror_f1, reverse=True)

    second_model = next((j.model for j in ranked_models if j.model != best_model), best_model)
    rows = [
        cell("A", [by_prompt_model.get((best_prompt, best_model))],
             "1 model, 1 prompt — the floor (greedy decoding makes copies identical)", 1),
        cell("B", ranked_models[:size], f"{size} models, selected prompt — model diversity", size),
        cell("C", [by_prompt_model.get((p, best_model)) for p in prompts[:size]],
             f"1 model ({best_model}), {size} prompts — prompt diversity", size),
        cell("D", [by_prompt_model.get((p, m))
                   for p in prompts[:2] for m in (best_model, second_model)],
             f"2 models x 2 prompts ({best_model}, {second_model}) — crossed", 4),
    ]
    rows = [r for r in rows if r]
    write_table(rows, tables / "s6_cells.csv")
    short = [r for r in rows if not r["matched"]]
    if short:
        logger.warning("S6 — %s cell(s) could not be filled at the requested size; the contrast "
                       "is NOT matched on generation calls and the table says so per row: %s",
                       len(short), [r["cell"] for r in short])

    # The mechanism, over every pair in the pool, not only the four cells: a ranking without
    # it is an outcome, and the decorrelation is the explanation.
    pairs = []
    pool = [j for j in jurors if j.normaliser == normaliser]
    unit_cache = {}
    for juror in pool:
        packed = pack([juror], per_juror, dev, "report", "exact", view)
        pred = sets_at_k(max_counts(packed, dev, frozenset({0})), 1)
        unit_cache[juror] = scorer.units(pred, dev)
    for a, b in itertools.combinations(pool, 2):
        kind = ("model_diverse" if a.prompt == b.prompt else
                "prompt_diverse" if a.model == b.model else "crossed")
        pairs.append({"a": a.key, "b": b.key, "kind": kind,
                      **error_diversity(unit_cache[a], unit_cache[b])})
    write_table(pairs, tables / "s6_error_diversity.csv")

    for kind in ("model_diverse", "prompt_diverse", "crossed"):
        subset = [p["fp_jaccard"] for p in pairs if p["kind"] == kind]
        if subset:
            logger.info("S6 — mean FP Jaccard, %-14s %.4f over %d pair(s)",
                        kind, sum(subset) / len(subset), len(subset))
    return rows


# ── The report ───────────────────────────────────────────────────────────────

def write_report(cfg, out_dir: Path, tables: Path, prompt: str, normaliser: str,
                 n_dev: int, n_reports: int) -> Path:
    """Assemble ``results.md`` from whatever tables this run produced.

    a renderer and not a second analysis: it reads the CSVs and states which stages
    ran, so a partial run produces an honest partial document, not a complete-looking one.
    An earlier exploratory run declared this stage and never wrote it, and the finding that quotes its numbers had to
    be assembled by hand.
    """
    def table(name: str, limit: int | None = None) -> str:
        path = tables / f"{name}.csv"
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return f"_`{name}.csv` was not produced by this run._\n"
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return f"_`{name}.csv` is empty._\n"
        fields = list(rows[0])
        shown = rows[:limit] if limit else rows
        out = ["| " + " | ".join(fields) + " |",
               "|" + "|".join(["---"] * len(fields)) + "|"]
        for row in shown:
            out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
        if limit and len(rows) > limit:
            out.append(f"\n_{len(rows) - limit} further row(s) in `{name}.csv`._")
        return "\n".join(out) + "\n"

    frame = ("the whole cohort (IN-SAMPLE)" if bool(cfg.dev_use_full_cohort)
             else f"the training half of repetition {cfg.dev_repetition}, outer fold "
                  f"{cfg.dev_outer_fold} ({n_dev} of {n_reports} reports)")
    doc = f"""# exp14_04 — {cfg.cohort}: PhenoJury with the prompt and the normaliser selected in-fold

Pair the descriptive stages hold fixed (best on the descriptive split): **({prompt}, {normaliser})**.
Stages run: `{', '.join(stages_run(cfg))}`.

**How to read this.** S2, S3, S4 and S6 are **descriptive**, computed on {frame}, and carry no
p-values — differences there are cluster-bootstrapped, not tested. S7 is the only inferential
result and its family was fixed in config before the run. **No out-of-fold row is scored at the
descriptive pair**: every S5, S7, S8 and LOMO row chooses its (prompt, normaliser) inside each outer
training split, and S5 below says which pair each fold chose.

## S0 — what the cache holds

{table("s0_cache_audit")}

## S1 — the ceilings

Union recall at k=1 bounds everything below it. The normaliser-union gap is a **lower bound on
normaliser loss**: a gold term some reader recovers and the selected one does not was lost to the
reader, not to the model.

{table("s1_ceilings")}

### Anchored generation

Restricted to the segment the curator put the finding in — the strictest of the three ceilings, and
the one that separates "the method found this finding" from "the method emitted this identifier".
Read `coordinate_ok_rate` first: if it is low, the gold's `segment_idx` and the run's
`sentence_number` index different segmentations and the rate beside it means nothing.

{table("s1_anchored_generation")}

## S2 — prompt and normaliser, jointly

{table("s2_prompt_normaliser_grid")}

## S3 — the k curves, and the verbosity that moves the operating point

{table("s3_curves", limit=40)}

{table("s3_prompt_diagnostics", limit=40)}

## S4 — model x prompt

{table("s4_interaction", limit=40)}

{table("s4_decomposition")}

## S5 — what the selection chose, and how often

{table("s5_selection_frequency", limit=40)}

The full pool's pair choices, for comparison:

{table("s7_full_pool_pair_frequency")}

## S6 — the matched diversity contrast

Cell A is **one** juror, not four copies of one: greedy decoding makes those identical, so
self-consistency is unavailable and the single juror is the correct floor. That is why this
contrast is interesting rather than a rediscovery of self-consistency.

{table("s6_cells")}

## S7 — the headline and the pre-specified family

{table("s7_headline")}

{table("s7_significance")}

## S8 — prompt sensitivity

If the spread across prompts exceeds the difference between the selected jury and the best single
juror, prompt choice matters more than the jury does, and that is what the chapter must say.

{table("s8_prompt_sensitivity")}
"""
    path = out_dir / "results.md"
    path.write_text(doc, encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def stages_run(cfg) -> list[str]:
    """The stage names the config asks for, in order."""
    return [s for s in cfg.stages]


# ── Driver ───────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../../configs/experiments/05_phenojury", config_name="protocol")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    out_dir = Path(cfg.output_dir) / EXP_ID / str(cfg.cohort)
    tables = out_dir / "tables"
    predictions = out_dir / "predictions"
    tables.mkdir(parents=True, exist_ok=True)
    stages = list(cfg.stages)
    logger.info("stages=%s cohort=%s", stages, cfg.cohort)

    # `phenojury` is produced by `selection` and consumed by `headline`, so running the second
    # without the first silently drops every comparison the test family names. Say it once here
    #, not let it surface as two "one side was not produced" warnings much later.
    if "headline" in stages and "selection" not in stages:
        logger.warning(
            "headline without selection: `phenojury` will not exist, so every comparison naming "
            "it is skipped and s7_significance.csv will be short or absent. Add `selection`.")

    from hpo_extraction.ontology.hpo_tree import HPOTree
    view = OntologyView(HPOTree())

    # Order-preserving dedupe: the reference reader is often also listed in `normalisers`, and a
    # juror loaded twice would cast two votes.
    all_normalisers = list(dict.fromkeys(list(cfg.normalisers) + [str(cfg.reference_normaliser)]))
    # Which detections to read, which may not be the cohort being scored: a derived cohort is a
    # narrower view of a corpus whose generations were produced once (see `source_cohort`).
    source_cohort = str(cfg.get("source_cohort") or cfg.cohort)
    if source_cohort != str(cfg.cohort):
        logger.info("cohort=%s scored from the %s detections (derived cohort)",
                    cfg.cohort, source_cohort)
    jurors, per_juror, inventory = load_grid(
        Path(cfg.grid_dir), source_cohort, list(cfg.prompts), all_normalisers)
    gold = load_gold(Path(cfg.gold_path))
    report_ids = sorted(gold)
    logger.info("%d juror(s) over %d prompt(s) x %d reader(s); %d report(s), %d gold pair(s)",
                len(jurors), len({j.prompt for j in jurors}), len({j.normaliser for j in jurors}),
                len(report_ids), sum(len(v) for v in gold.values()))

    scorer = Scorer(gold, view, report_ids, int(cfg.n_bootstrap))
    folds = read_folds(Path(cfg.folds_path)) if cfg.folds_path else []
    dev = dev_ids(folds, cfg, report_ids) if folds else list(report_ids)
    logger.info("descriptive split: %d of %d report(s)%s", len(dev), len(report_ids),
                "  [IN-SAMPLE]" if bool(cfg.dev_use_full_cohort) else "")

    if "audit" in stages:
        stage_audit(cfg, jurors, inventory, per_juror, tables)
    if "ceilings" in stages:
        stage_ceilings(cfg, jurors, per_juror, scorer, report_ids, view, tables)

    # The pair the DESCRIPTIVE stages hold fixed (S3, S4, S6's cells, the call budget), chosen on
    # The dev split by the grid, never by hand. It is not the pair any out-of-fold number is scored
    # at: every estimate below selects its own pair inside each outer training split, because the
    # dev split holds the held-out reports of outer folds 1 to 4 of the very repetition that is
    # pooled, and a pair fixed on it put ~80 % of the pooled reports behind the choice that
    # scored them.
    selected_normaliser = list(cfg.normalisers)[0]
    selected_prompt = list(cfg.prompts)[0]
    if "grid" in stages:
        grid_rows = stage_grid(cfg, jurors, per_juror, scorer, dev, view, tables)
        scored = [r for r in grid_rows
                  if r.get("micro_f1") and not r.get("is_reference_reader")]
        if scored:
            best = max(scored, key=lambda r: r["micro_f1"])
            selected_prompt, selected_normaliser = best["prompt"], best["normaliser"]
            logger.info("S2 — selected pair on the dev split: (%s, %s) at micro-F1 %.4f",
                        selected_prompt, selected_normaliser, best["micro_f1"])

    # T5.6's second block. It needs the selected pair only for logging, so it runs here, right
    # after the grid it restricts.
    if "restricted" in stages:
        stage_grid_restricted(cfg, jurors, per_juror, gold, view, dev, tables)

    if "curves" in stages:
        stage_curves(cfg, jurors, per_juror, scorer, dev, view, tables, selected_normaliser)
    if "interaction" in stages:
        stage_interaction(cfg, jurors, per_juror, scorer, dev, view, tables, selected_normaliser)

    # The eight models at the descriptive pair: the call budget and S6's cells read it.
    pool = [j for j in jurors
            if j.prompt == selected_prompt and j.normaliser == selected_normaliser]
    models = sorted({j.model for j in jurors if j.normaliser in set(cfg.normalisers)})
    units_by_label: dict[str, list] = {}
    quality_rows: list[dict] = []

    def record(label, pred, choices=None, note=""):
        units = scorer.units(pred, report_ids)
        ci = scorer.interval(units)
        row = {"config": label, "note": note,
               "micro_precision": round(ci["micro_precision"]["point"], 4),
               "micro_recall": round(ci["micro_recall"]["point"], 4),
               "micro_f1": round(ci["micro_f1"]["point"], 4),
               "micro_f1_lo": round(ci["micro_f1"].get("lo", float("nan")), 4),
               "micro_f1_hi": round(ci["micro_f1"].get("hi", float("nan")), 4),
               "macro_precision": round(ci["macro_precision"]["point"], 4),
               "macro_recall": round(ci["macro_recall"]["point"], 4),
               "macro_f1": round(ci["macro_f1"]["point"], 4),
               "preds_per_report": round(
                   sum(len(p) for _, p in units) / max(1, len(units)), 1)}
        if choices:
            row["selection_modes"] = json.dumps(
                Counter((c["rule"], c["unit"], c["k"]) for c in choices).most_common(3),
                default=str)
            row["pair_modes"] = json.dumps(
                Counter(f"{c['prompt']}|{c['normaliser']}" for c in choices).most_common(3))
        quality_rows.append(row)
        units_by_label[label] = units
        # The pooled out-of-fold set itself, not just its score. The comparison puts this method in a
        # table beside the fixed-operating-point ones, and hierarchy-aware and subgroup metrics
        # cannot be recovered from a (P, R, F1) triple, they need the sets.
        # Written over `report_ids`, so a report the configuration predicted nothing for gets a
        # line with an empty cell, not no line, the set that was scored.
        write_prediction_sets(predictions / f"{_slug(label)}.csv",
                              {r: set(pred.get(r, ())) for r in report_ids})
        logger.info("%-46s F1=%.4f [%.4f, %.4f]  P=%.4f R=%.4f  macro F1=%.4f", label,
                    row["micro_f1"], row["micro_f1_lo"], row["micro_f1_hi"],
                    row["micro_precision"], row["micro_recall"], row["macro_f1"])
        return units

    # The J x m call budget. Cheap, and it needs the pool but nothing the folds produce.
    if "budget" in stages and pool:
        stage_call_budget(cfg, jurors, per_juror, pool, report_ids, tables)

    # Every out-of-fold estimate, in one pass over the candidate pairs. Each condition selects its
    # (prompt, normaliser) pair inside every outer training split alongside whatever else it
    # selects there. See `protocol.evaluate_arms`.
    results = {}
    pairs = grid_pairs(cfg)
    arms = plan_conditions(cfg, stages, models) if folds else []
    if arms:
        logger.info("out-of-fold: %d arm(s) x %d pair(s) x %d outer fold(s), %d worker(s)",
                    len(arms), len(pairs), len(folds), int(cfg.n_workers))
        results = evaluate_conditions(arms, pairs, jurors, per_juror, gold, folds, view, normalise_pair,
                                micro_prf, list(cfg.rules), list(cfg.units), float(cfg.epsilon),
                                report_ids, n_workers=int(cfg.n_workers))
    in_fold_pairs = {}

    if "selection" in stages and "phenojury" in results:
        chosen = results["phenojury"]
        choices = chosen.choices
        record("phenojury", chosen.pooled, choices,
               "nested selection over pair/jury/rule/unit/k")
        write_table([{**c, "subset": ";".join(c["subset"])} for c in choices],
                    tables / "s5_selection.csv")
        frequency = pair_frequency(choices)
        for axis in ("rule", "unit", "k", "size"):
            for value, count in Counter(c[axis] for c in choices).most_common():
                frequency.append({"axis": axis, "value": value, "n_folds": count,
                                  "share": round(count / max(1, len(choices)), 4)})
        for model in models:
            n = sum(1 for c in choices if model in c["subset"])
            frequency.append({"axis": "juror", "value": model, "n_folds": n,
                              "share": round(n / max(1, len(choices)), 4)})
        write_table(frequency, tables / "s5_selection_frequency.csv",
                    fieldnames=["axis", "value", "n_folds", "share"])
        # Every pair's inner score in every outer fold, so the margin a pair won by is auditable.
        write_table(chosen.inner_scores, tables / "s5_pair_inner_scores.csv",
                    fieldnames=["repetition", "outer_fold", "prompt", "normaliser", "inner_f1",
                                "chosen"])
        in_fold_pairs["phenojury"] = modal_pair(choices)
        logger.info("S5 — pairs chosen across %d outer fold(s): %s", len(choices),
                    dict(Counter(f"{c['prompt']}|{c['normaliser']}" for c in choices)))
        logger.info("S5 — jury sizes chosen across %d outer fold(s): %s",
                    len(choices), dict(Counter(c["size"] for c in choices)))

    if "headline" in stages and "full_pool" in results:
        record("full_pool", results["full_pool"].pooled, results["full_pool"].choices,
               "every juror; pair/rule/unit/k by inner CV")
        in_fold_pairs["full_pool"] = modal_pair(results["full_pool"].choices)
        write_table(pair_frequency(results["full_pool"].choices),
                    tables / "s7_full_pool_pair_frequency.csv",
                    fieldnames=["axis", "value", "n_folds", "share"])

        # J0, PhenoBERT off the report text, no language model anywhere in it. Read from a
        # prediction-set CSV because the PhenoJury protocol re-runs cached generations and cannot run the
        # tagger itself. Scored and given an interval; NOT tested, since the family
        # was fixed before the run (see config).
        if cfg.get("j0_predictions_path"):
            path = Path(cfg.j0_predictions_path)
            if path.exists():
                j0 = read_prediction_sets(path)
                missing = [r for r in report_ids if r not in j0]
                if missing:
                    logger.warning("J0 — %d of %d report(s) absent from %s; they are scored as "
                                   "empty predictions, which is what a method that did not "
                                   "answer them earns", len(missing), len(report_ids), path.name)
                record("phenobert", {r: j0.get(r, set()) for r in report_ids},
                       note=f"J0: PhenoBERT on the report text, no LM ({path.name}); "
                            "descriptive — not in the pre-registered family")
            else:
                logger.warning("J0 — j0_predictions_path %s does not exist, so T5.2's $J_0$ row "
                               "is absent, see docs/thesis_map.md", path)

        # A jury named by hand, the models S5 selects in nearly every fold of BOTH cohorts. The
        # jury is fixed, but pair/rule/unit/k are still chosen inside each training split, so the
        # estimate stays out-of-fold. These rows are NOT added to `test_family`:
        # naming a comparator after reading S5 invalidates the Holm adjustment for the whole
        # family, so they are scored, given intervals, and tested only in the exploratory table.
        # `plan_arms` has already warned about any member missing from the pool.
        for arm in arms:
            if arm.label.startswith("fixed:"):
                record(arm.label, results[arm.label].pooled, results[arm.label].choices,
                       "fixed jury (" + "+".join(arm.models) + "); pair/rule/unit/k by inner CV")
        # One model alone. A jury of one admits only k=1, so the pair is the only thing its
        # training split chooses -- but it IS chosen there, as it is for every jury it is compared
        # against, or the comparison would give the jury a selection step the single juror lacks.
        for model in models:
            single = results.get(f"single:{model}")
            if single is not None:
                record(f"single:{model}", single.pooled, single.choices,
                       note="single juror; pair by inner CV, exact/report/k=1")
        singles = {k: v for k, v in units_by_label.items() if k.startswith("single:")}
        if singles:
            best_single = max(singles, key=lambda k: Scorer.prf(singles[k])[2])
            units_by_label["best_single_juror"] = singles[best_single]
            logger.info("S7 — best single juror: %s", best_single)
        tests, intervals, family = {}, {}, []
        for pair in cfg.test_family:
            a, b = list(pair)
            if a not in units_by_label or b not in units_by_label or not units_by_label[b]:
                logger.warning("S7 — skipping %s vs %s: one side was not produced", a, b)
                continue
            family.append(f"{a} vs {b}")
            result = paired_randomisation(
                [u[0] for u in units_by_label[a]],
                [u[1] for u in units_by_label[a]], [u[1] for u in units_by_label[b]],
                lambda g, p: dict(zip(("micro_precision", "micro_recall", "micro_f1"),
                                      micro_prf(g, p))),
                n_permutations=int(cfg.n_permutations), seed=0)
            tests[f"{a} vs {b}"] = result["p_value"]
            intervals[f"{a} vs {b}"] = scorer.paired_interval(units_by_label[a],
                                                              units_by_label[b])
            logger.info("%-46s delta=%+.4f [%+.4f, %+.4f] p=%.4f", f"{a} vs {b}",
                        result["delta"], *intervals[f"{a} vs {b}"][1:], result["p_value"])
        # EXPLORATORY, not the fixed in advance family, and carrying no multiplicity correction.
        # The comparator was named after reading S5's selection frequencies, so a Holm-adjusted
        # p-value here would be arithmetic dressed as inference. Quote the interval, not the p.
        # Fixed juries are also compared with EACH OTHER: a swap of one member (check X24) is a
        # question about two fixed juries, which neither's comparison with the pool answers.
        exploratory = []
        fixed = [lbl for lbl in units_by_label if lbl.startswith("fixed:")]
        for i, label in enumerate(fixed):
            for other in fixed[i + 1:] + ["full_pool", "phenojury", "best_single_juror"]:
                if other not in units_by_label or not units_by_label[other]:
                    continue
                res = paired_randomisation(
                    [u[0] for u in units_by_label[label]],
                    [u[1] for u in units_by_label[label]], [u[1] for u in units_by_label[other]],
                    lambda g, p: dict(zip(("micro_precision", "micro_recall", "micro_f1"),
                                          micro_prf(g, p))),
                    n_permutations=int(cfg.n_permutations), seed=0)
                _, lo, hi = scorer.paired_interval(units_by_label[label], units_by_label[other])
                exploratory.append({
                    "comparison": f"{label} vs {other}", "delta_micro_f1": round(res["delta"], 4),
                    "delta_lo": round(lo, 4), "delta_hi": round(hi, 4),
                    "p_raw_uncorrected": res["p_value"], "pre_registered": False,
                    "note": "exploratory; comparator named after reading S5 — no Holm correction"})
                logger.info("%-46s delta=%+.4f [%+.4f, %+.4f] p=%.4f (exploratory)",
                            f"{label} vs {other}", res["delta"], lo, hi, res["p_value"])
        if exploratory:
            write_table(exploratory, tables / "s7_exploratory_fixed_jury.csv",
                        fieldnames=["comparison", "delta_micro_f1", "delta_lo", "delta_hi",
                                    "p_raw_uncorrected", "pre_registered", "note"])

        if tests:
            adjusted = holm(tests, alpha=float(cfg.alpha_holm))
            write_table(
                [{"comparison": name, "family_size": len(family),
                  "family": " | ".join(family),
                  **{k: values[k] for k in ("p_raw", "p_holm", "reject")},
                  **dict(zip(("delta_micro_f1", "delta_lo", "delta_hi"),
                             (round(v, 4) for v in intervals[name])))}
                 for name, values in adjusted.items()],
                tables / "s7_significance.csv")

    # T5.7 and T5.8 both describe the SELECTED jury, so both run after `headline` has resolved it.
    # The baseline they are measured against is `phenojury` where selection ran and `full_pool`
    # otherwise, stated, not silently substituted, because a delta against the wrong
    # baseline is the kind of error that reads as a finding.
    if "lomo" in stages and results:
        baseline = units_by_label.get("phenojury") or units_by_label.get("full_pool")
        if baseline is None:
            logger.warning("S6 LOMO — neither phenojury nor full_pool was produced, so there is "
                           "no jury to leave a model out OF; add `selection` or `headline`")
        else:
            which = "phenojury" if "phenojury" in units_by_label else "full_pool"
            logger.info("S6 LOMO — deltas are against %s", which)
            stage_lomo(results, models, jurors, per_juror, scorer, report_ids, tables, baseline)

    if "decomposition" in stages and results:
        label = "phenojury" if "phenojury" in results else "full_pool"
        if label not in results:
            logger.warning("S9 — no selected jury to decompose; add `selection` or `headline`")
        else:
            # Each report at the pair, rule and scope its own outer fold chose: the decomposition
            # is taken at the configuration the jury is actually reported at, report by report.
            stage_jury_decomposition(gold, view, report_ids, tables, results[label].pooled,
                                     results[label].union)
            stage_jury_error_taxonomy(gold, view, report_ids, tables, results[label].pooled)

    if "diversity" in stages and pool:
        best_model = pool[0].model
        if "best_single_juror" in units_by_label:
            singles = {k: v for k, v in units_by_label.items() if k.startswith("single:")}
            if singles:
                best_model = max(singles, key=lambda k: Scorer.prf(singles[k])[2]).split(":", 1)[1]
        stage_diversity(cfg, jurors, per_juror, scorer, dev, view, tables, selected_normaliser,
                        best_model, selected_prompt)

    if "sensitivity" in stages and results:
        # S8: the final jury's score under EVERY prompt, against the jury's own effect. If the
        # spread across prompts exceeds the J4-minus-J0 difference, prompt choice counts more than
        # The jury does and the chapter has to say that plainly, not claim an ensemble
        # contribution. That reading is pre-committed in experiment.md.
        sensitivity = []
        for prompt in list(cfg.prompts):
            forced = results.get(f"prompt:{prompt}")
            if forced is None:
                continue
            pooled = forced.pooled
            units = scorer.units(pooled, report_ids)
            ci = scorer.interval(units)
            # The full pool at a FORCED prompt, out-of-fold, with the normaliser still chosen in
            # each training split. Identical to `full_pool.csv` only if every fold of the full
            # pool chose this prompt. Otherwise it is the only way to quote both cohorts at one
            # pre-declared prompt, which is what the comparison does.
            write_prediction_sets(predictions / f"prompt_{_slug(prompt)}.csv",
                                  {r: set(pooled.get(r, ())) for r in report_ids})
            normalisers = Counter(c["normaliser"] for c in forced.choices)
            modal_normaliser = normalisers.most_common(1)[0][0]
            sensitivity.append({
                "prompt": prompt, "normaliser": modal_normaliser,
                "normaliser_modes": json.dumps(normalisers.most_common()),
                "n_jurors": sum(1 for j in jurors
                                if j.prompt == prompt and j.normaliser == modal_normaliser),
                "micro_precision": round(ci["micro_precision"]["point"], 4),
                "micro_recall": round(ci["micro_recall"]["point"], 4),
                "micro_f1": round(ci["micro_f1"]["point"], 4),
                "micro_f1_lo": round(ci["micro_f1"].get("lo", float("nan")), 4),
                "micro_f1_hi": round(ci["micro_f1"].get("hi", float("nan")), 4),
                # Document-macro beside micro, as in the main table (s7_headline.csv).
                "macro_precision": round(ci["macro_precision"]["point"], 4),
                "macro_recall": round(ci["macro_recall"]["point"], 4),
                "macro_f1": round(ci["macro_f1"]["point"], 4)})
        write_table(sensitivity, tables / "s8_prompt_sensitivity.csv")
        if sensitivity:
            spread = (max(r["micro_f1"] for r in sensitivity)
                      - min(r["micro_f1"] for r in sensitivity))
            jury_effect = None
            if "phenojury" in units_by_label and "best_single_juror" in units_by_label:
                jury_effect = (Scorer.prf(units_by_label["phenojury"])[2]
                               - Scorer.prf(units_by_label["best_single_juror"])[2])
            logger.info("S8 — prompt spread %.4f%s", spread,
                        f" against a jury effect of {jury_effect:+.4f}"
                        if jury_effect is not None else "")
            if jury_effect is not None and spread > jury_effect:
                logger.warning("S8 — the prompt spread EXCEEDS the jury effect. The chapter must "
                               "report prompt choice as the larger lever.")

    if quality_rows:
        write_table(quality_rows, tables / "s7_headline.csv")

    if "report" in stages:
        write_report(cfg, out_dir, tables, selected_prompt, selected_normaliser, len(dev),
                     len(report_ids))

    manifest = {
        "cohort": str(cfg.cohort), "source_cohort": source_cohort,
        "n_jurors": len(jurors), "n_reports": len(report_ids),
        # `selected_*` is the DEV-split pair the descriptive stages hold fixed (S3's curves are
        # drawn at it). No out-of-fold row is scored at it: those choose their pair per fold, and
        # `in_fold_modal_pair` is the pair each chose most often across all outer folds.
        "pair_selection": "in_fold",
        "selected_prompt": selected_prompt, "selected_normaliser": selected_normaliser,
        "in_fold_modal_pair": {label: list(pair) for label, pair in in_fold_pairs.items()
                               if pair},
        "dev_reports": len(dev), "dev_in_sample": bool(cfg.dev_use_full_cohort),
        "rules": list(cfg.rules), "units": list(cfg.units),
        "test_family": [list(p) for p in cfg.test_family],
        "n_bootstrap": int(cfg.n_bootstrap), "n_permutations": int(cfg.n_permutations),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_dir / "manifest.json")

    try:
        import mlflow
        mlflow.set_experiment(cfg.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"{EXP_ID}_{cfg.cohort}"):
            mlflow.log_params({k: v for k, v in manifest.items() if not isinstance(v, list)})
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
            for path in sorted(tables.glob("*.csv")):
                mlflow.log_artifact(str(path))
    except Exception as exc:                                   # pragma: no cover - logging only
        logger.warning("MLflow logging skipped: %s", exc)


if __name__ == "__main__":
    main()
