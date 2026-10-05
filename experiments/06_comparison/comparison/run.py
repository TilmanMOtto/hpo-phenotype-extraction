"""The comparison, the final comparison table for the discussion: five systems, three tables, two cohorts.


Every table in `experiments/findings/` so far answers "what did this experiment measure". This one
answers the question the discussion asks: **on the same reports, against the same ground truth, how do the
published methods and ours compare**, and it is allowed to contain only the rows that comparison
needs.

Three tables, one CSV each:

``t1_overall.csv``     flat quality, micro P/R/F1 and three macro averages, with report-level
                       bootstrap intervals.
``t2_hierarchy.csv``   hierarchy-aware quality, ancestor-closure hP/hR/hF and CoPHE, which is what
                       says whether a method's errors are near-misses or category errors.
``t3_subgroups.csv``   recall on the curated HCY qualifiers, lab values and implicit descriptions, plus the family attribution counts. HCY only: GSC+ has no such labels.
``t4_cost.csv``        seconds per report, model calls per report, peak GPU and the setup, read
                       from the experiment that *incurred* the cost, which is never the one that
                       scored it for the two protocol methods. See :mod:`cost`.

## The three things that make it a comparison rather than a collage

**One ground truth and one report list per cohort.** HCY is the curated ground truth at a fixed dated version
(118 reports, 1146 pairs, the PhenoRAG paper's own cohort). GSC+ is `gsc_raghpo_ann`: RAG-HPO's 114
documents under RAG-HPO's own re-annotation, which is the only GSC+ view the published table may be
read against. A method missing a report is reported, not silently dropped: `t0_coverage.csv` carries
every method's report count beside the cohort's.

**Three estimators, named in every row.** RAG-HPO, AutoPCR and PhenoBERT have nothing to tune, so
their number is a plain cohort estimate. PhenoJury on both cohorts, and TreePhenoRAG on HCY, choose
their configuration inside nested cross-validation, so theirs is the pooled out-of-fold set.
TreePhenoRAG on GSC+ is a **transfer**: the configuration selected on all of HCY, applied to the
abstracts unchanged, with nothing fitted on GSC+ at all, the strictest of the three, and the one
most likely to be misread as the weakest. Quoting a same-data-tuned number for the tunable methods
against untuned baselines is the bias `treephenorag_protocol` and `an earlier exploratory run` exist to remove. The cost of
removing it is that one column holds three estimators, and the ``estimator`` column says which is
which on every line.

**Published numbers live in their own file with a comparability verdict.** `t1_literature.csv` holds
Garcia et al.'s Table 5 and Tao et al.'s AutoPCR tables. Garcia's rows share this table's documents
and ground truth and are marked comparable, against `micro_f1`, recomputed from the TP/FP/FN they print,
because their P, R and F1 are each a mean of per-case values that no column here is defined as. Tao et al.'s are **not**: mention-level on 206 documents of the original GSC+ annotation.
That row carries its three differences in ``why_not``, not sitting in the main table
looking like a competitor.

Analysis only: no model, no GPU. It reads prediction artifacts and runs on CPU in minutes.

Usage:
    python experiments/06_comparison/comparison/run.py \\
        results_dir=/path/to/output output_dir=/path/to/output \\
        hcy_curated_gt_path=/path/to/hcy/curated_ground_truth_2026-09-12/hcy_ground_truth_curated.csv \\
        hcy_annotations_path=/path/to/hcy/curated_ground_truth_2026-09-12/hcy_curated_annotations.csv
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402
for _p in (str(_REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hpo_extraction.evaluation.result_tables.loaders as loaders  # noqa: E402, the result-table library's translation layer, reused verbatim
import hpo_extraction.evaluation.result_tables.report as report_mod  # noqa: E402, the result-table library's CSV + LaTeX + markdown writer
import complementarity as complementarity_mod  # noqa: E402
import cost as cost_mod  # noqa: E402
import roster as roster_mod  # noqa: E402
import subgroups as subgroups_mod  # noqa: E402
from hpo_extraction.evaluation.prediction_sets import read_prediction_sets  # noqa: E402
from hpo_extraction.evaluation.stats import ReportResampler, bootstrap_reports  # noqa: E402
from hpo_extraction.evaluation.metrics import (  # noqa: E402
    OntologyView, flat_report, h_counts, hierarchy_report, normalise_pair,
)

EXP_ID = "comparison"
logger = logging.getLogger(EXP_ID)

#: The cohort whose qualifier labels exist. GSC+ has no curation pass, so table 3 is HCY-only and
#: says so, not printing an empty GSC+ half.
SUBGROUP_COHORT = "hcy"


def _setup_logging(run_output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(run_output_dir / "run.log"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _load_view(hpo_json_path: str) -> OntologyView:
    """The ontology wrapper the hierarchy metrics go through. Built once, ~15 s."""
    from hpo_extraction.ontology.hpo_tree import HPOTree

    tree = HPOTree(hpo_json_path) if hpo_json_path else HPOTree()
    tree.buildHPOTree()
    return OntologyView(tree)


# ── loading ──────────────────────────────────────────────────────────────────

def load_cohort_gold(cfg: DictConfig, cohort: str) -> dict:
    """The ground truth for *cohort*, through the result-table library's loader so there is one definition of each."""
    if cohort == "hcy":
        path = cfg.get("hcy_curated_gt_path")
        if not path:
            raise ValueError(
                "cohort 'hcy' needs hcy_curated_gt_path — the curated gold "
                "(<hcy_dir>/curated_ground_truth_<date>/hcy_ground_truth_curated.csv). The raw gold is "
                "NOT a substitute: exp13_18 measured it at 87 % omission.")
        return loaders.load_gold("hcy_curated", hcy_curated_gt_path=str(path))
    return loaders.load_gold(cohort, gsc_dir=str(cfg.get("gsc_dir") or ""),
                             raghpo_dir=str(cfg.get("raghpo_dir") or ""))


def load_method(row, results_dir: str, cohort: str, override: str = "",
                declared: bool = True) -> tuple[dict | None, str, str]:
    """``(predicted_by_report, path, status)`` for one roster row on one cohort.

    ``status`` is ``ok`` or a reason string printed verbatim in the tables. A method with no
    artifact is a **named absence**, the path it looked for is in the reason, not a row
    that quietly does not appear. ``declared=False`` means config says this cell does not exist at
    all, which is a different statement from "the file is missing" and is reported as one.
    """
    if not declared:
        return None, "", "not applicable: no run declared for this method on this cohort"
    path = roster_mod.predictions_path(row, results_dir, cohort, override)
    if not Path(path).exists():
        return None, path, "missing: no artifact at this path"
    if row.provenance == "protocol":
        return read_prediction_sets(path), path, "ok"
    predicted, _gold = loaders.load_predictions(path)
    return predicted, path, "ok"


def align(predicted: dict, gold: dict,
          view: OntologyView) -> tuple[list, list, list, dict, list]:
    """``(report_ids, gold_sets, pred_sets, bookkeeping)`` over the cohort, normalised.

    The report list is the **ground truth's**, not the intersection: a report the method emitted nothing
    for still costs it every annotated term there. Only a report genuinely absent from the artifact is
    counted separately, in ``n_reports_absent``, that is a coverage failure and must not be
    disguised as an empty prediction.

    This is also where a derived cohort is derived: ``gsc_raghpo_ann``'s ground truth names 114 of the 228
    documents the ``gsc`` artifact covers, so iterating the ground truth restricts the scoring to them.
    """
    report_ids = sorted(gold)
    absent = [r for r in report_ids if r not in predicted]
    gold_sets, pred_sets, raw = [], [], []
    book = {"n_gold_out_of_subtree": 0, "n_gold_nonexistent": 0,
            "n_pred_out_of_subtree": 0, "n_pred_nonexistent": 0}
    for report_id in report_ids:
        g, p, counts = normalise_pair(gold[report_id], predicted.get(report_id, set()), view)
        gold_sets.append(g)
        pred_sets.append(p)
        raw.append((set(gold[report_id]), set(predicted.get(report_id, set()))))
        for k, v in counts.items():
            book[k] += v
    book["n_reports_absent"] = len(absent)
    book["reports_absent"] = ";".join(absent[:10])
    book["n_gold_pairs"] = sum(len(v) for v in gold.values())
    book["n_gold_scored"] = sum(len(g) for g in gold_sets)
    return report_ids, gold_sets, pred_sets, book, raw


# ── the three tables ─────────────────────────────────────────────────────────

# Both bootstraps resample **per-report count triples**, not per-report sets.
#
# `bootstrap_reports` recomputes the metric on all 10 000 draws, so a metric that re-derives its
# counts each time pays for it 10 000 times over. For the hierarchy that is fatal, not slow:
# `h_prf` takes an ancestor closure of both sides, which would be ~1.2 M closure computations per
# (method, cohort) cell. The counts are a sufficient statistic for every **micro** average, so they
# are computed once per report and the resample is arithmetic.
#
# This is the shortcut `hpo_extraction.evaluation.stats.bootstrap`'s docstring warns against for the
# *label-macro* averages, which depend on the set of terms present in each draw and cannot be
# recovered from pooled counts. That is why only the micro figures carry intervals here. The macro
# columns are point estimates, and the tables do not pretend otherwise.

def flat_units(gold_sets, pred_sets) -> list:
    """Per-report ``(tp, fp, fn)``."""
    return [(len(g & p), len(p - g), len(g - p)) for g, p in zip(gold_sets, pred_sets)]


def flat_metric(units) -> dict:
    """Micro precision, recall and F1 (each between 0 and 1) from per-report ``(tp, fp, fn)`` units."""
    tp = sum(u[0] for u in units)
    fp = sum(u[1] for u in units)
    fn = sum(u[2] for u in units)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"micro_precision": p, "micro_recall": r,
            "micro_f1": 2 * p * r / (p + r) if (p + r) else 0.0}


def hierarchy_units(gold_sets, pred_sets, view: OntologyView) -> list:
    """Per-report ``(|An(pred) & An(gold)|, |An(pred)|, |An(gold)|)``, eq. (3)'s three counts."""
    return [h_counts(g, p, view) for g, p in zip(gold_sets, pred_sets)]


def hierarchy_metric(units) -> dict:
    """Hierarchical precision, recall and F (each between 0 and 1) from per-report closure-overlap units."""
    overlap = sum(u[0] for u in units)
    n_pred = sum(u[1] for u in units)
    n_gold = sum(u[2] for u in units)
    hp = overlap / n_pred if n_pred else 0.0
    hr = overlap / n_gold if n_gold else 0.0
    return {"micro_hp": hp, "micro_hr": hr,
            "micro_hf": 2 * hp * hr / (hp + hr) if (hp + hr) else 0.0}


def t1_row(row, cohort: str, gold_sets, pred_sets, report_ids, book, sampler,
           point: str = "") -> dict:
    """One line of the flat-quality table, with a report-level interval on micro-F1."""
    flat = flat_report(gold_sets, pred_sets)
    ci = bootstrap_reports(flat_metric, flat_units(gold_sets, pred_sets), resampler=sampler)
    return {
        "method": row.key, "label": row.label, "family": row.family, "cohort": cohort,
        "estimator": row.estimator_for(cohort), "operating_point": point, "status": "ok",
        "n_reports": len(report_ids),
        "tp": flat["tp"], "fp": flat["fp"], "fn": flat["fn"],
        "micro_precision": flat["micro_precision"],
        "micro_recall": flat["micro_recall"],
        "micro_f1": flat["micro_f1"],
        "micro_f1_lo": ci["micro_f1"]["lo"], "micro_f1_hi": ci["micro_f1"]["hi"],
        "micro_precision_lo": ci["micro_precision"]["lo"],
        "micro_precision_hi": ci["micro_precision"]["hi"],
        "micro_recall_lo": ci["micro_recall"]["lo"],
        "micro_recall_hi": ci["micro_recall"]["hi"],
        "macro_precision": flat["macro_precision"],
        "macro_recall": flat["macro_recall"],
        "macro_f1": flat["macro_f1"],
        # The legacy scorer's convention: the harmonic mean of the two averages. NOT Garcia et
        # al.'s: their Table 5 F1 is the mean of per-case F1s (`macro_f1`) -- see the config.
        "macro_f1_of_means": flat["macro_f1_of_means"],
        "macro_term_precision": flat["macro_term_precision"],
        "macro_term_recall": flat["macro_term_recall"],
        "macro_term_f1": flat["macro_term_f1"],
        "macro_term_n_units": flat["macro_term_n_units"],
        "preds_per_report": sum(len(p) for p in pred_sets) / max(1, len(pred_sets)),
        **book,
        "note": row.note,
    }


def policy_row(row, cohort: str, gold_sets, pred_sets, raw, book) -> dict:
    """The same cell under both scoring policies, so a cross-check against the ground-truth build is one lookup.

    **This table's rows are normalised. The result-table library's and the ground-truth build's are not**, and the difference is
    worth about 0.002-0.006 micro-F1 on the methods that emit codes outside the phenotypic
    abnormality subtree. Neither convention is wrong, but they cannot be mixed, and this experiment
    has no choice: `phenojury_protocol` and `treephenorag_protocol` score through `normalise_pair`, so PhenoJury's 0.6389
    and TreePhenoRAG's number are normalised figures. Scoring the baselines raw would put two
    policies in one column, a far worse error than disagreeing with an earlier table by 0.003.

    The two knobs, both from `thesis_metrics.normalise.split_unscorable`:

    * a **ground truth** term outside the phenotypic-abnormality subtree leaves the denominator. On GSC+
      this is 53 of 1011 pairs (inheritance, clinical-course and similar codes RAG-HPO's
      re-annotation carries), which no method in this roster searches for. Charging every one of
      them as a false negative would measure the annotation's scope, not the methods.
    * a **predicted** term outside it stops being a false positive. This is the half that flatters,
      and it is why the column is printed, not described: RAG-HPO-70B emits 26 such codes on
      HCY and 54 on GSC+, PhenoBERT 8 and 3, AutoPCR and PhenoJury none at all.
    """
    raw_report = flat_report([g for g, _ in raw], [p for _, p in raw])
    return {
        "method": row.key, "label": row.label, "cohort": cohort,
        "micro_f1_normalised": flat_report(gold_sets, pred_sets)["micro_f1"],
        "micro_f1_raw_strings": raw_report["micro_f1"],
        "delta": (flat_report(gold_sets, pred_sets)["micro_f1"] - raw_report["micro_f1"]),
        "n_gold_pairs": book["n_gold_pairs"],
        "n_gold_scored": book["n_gold_scored"],
        "n_gold_out_of_subtree": book["n_gold_out_of_subtree"],
        "n_pred_out_of_subtree": book["n_pred_out_of_subtree"],
        "n_pred_nonexistent": book["n_pred_nonexistent"],
        "note": "raw_strings is the exp13_07 / exp13_18 convention; this table reports normalised",
    }


def t2_row(row, cohort: str, gold_sets, pred_sets, report_ids, view, sampler,
           point: str = "") -> dict:
    """One line of the hierarchy-aware table: closure hP/hR/hF and CoPHE, micro and macro."""
    h = hierarchy_report(gold_sets, pred_sets, view)
    ci = bootstrap_reports(hierarchy_metric,
                           hierarchy_units(gold_sets, pred_sets, view), resampler=sampler)
    out = {
        "method": row.key, "label": row.label, "family": row.family, "cohort": cohort,
        "estimator": row.estimator_for(cohort), "operating_point": point, "status": "ok",
        "n_reports": len(report_ids),
    }
    for key, value in h.items():
        if key in ("n_reports", "gold_ancestor_counts"):
            continue
        out[key] = value
    out["micro_hf_lo"] = ci["micro_hf"]["lo"]
    out["micro_hf_hi"] = ci["micro_hf"]["hi"]
    out["note"] = row.note
    return out


def t3_rows(row, predicted: dict, groups: dict, samplers: dict) -> list[dict]:
    """The subgroup lines for one method: two recall slices plus the attribution count.

    *samplers* is built once per subgroup and shared across methods, so every method's interval
    comes from the identical resamples, which is what makes a difference of intervals inside one
    subgroup mean anything (``hpo_extraction.evaluation.stats.bootstrap``, "Draw the indices once").
    """
    out = []
    for label in subgroups_mod.RECALL_LABELS:
        group = groups[label]
        units = subgroups_mod.recall_units(group, predicted)
        if not units:
            out.append({"method": row.key, "label": row.label, "subgroup": label,
                        "status": "empty: no gold term carries this label"})
            continue
        ci = bootstrap_reports(subgroups_mod.recall_metric, units, resampler=samplers[label])
        found = sum(len(p) for _, p in units)
        total = sum(len(g) for g, _ in units)
        out.append({
            "method": row.key, "label": row.label, "family": row.family,
            "subgroup": label, "kind": "recall", "status": "ok",
            "n_reports": len(units), "n_gold": total, "n_found": found,
            "recall": found / total if total else 0.0,
            "recall_lo": ci["recall"]["lo"], "recall_hi": ci["recall"]["hi"],
            "note": row.note,
        })
    group = groups[subgroups_mod.ATTRIBUTION_LABEL]
    counts = subgroups_mod.attribution_counts(group, predicted)
    out.append({
        "method": row.key, "label": row.label, "family": row.family,
        "subgroup": subgroups_mod.ATTRIBUTION_LABEL, "kind": "attribution", "status": "ok",
        "n_reports": counts["n_reports"], "n_gold": 0, "n_found": counts["n_emitted"],
        "n_emitted": counts["n_emitted"], "n_attribution_pairs": counts["n_pairs"],
        "emitted_pairs": counts["emitted_pairs"],
        "note": "count only: these terms belong to a relative and are NOT in the gold, so every "
                "one emitted is already a false positive in t1. No rate is quoted — "
                f"{counts['n_pairs']} pairs in {counts['n_reports']} reports supports none.",
    })
    return out


# ── literature ───────────────────────────────────────────────────────────────

def literature_rows(cfg: DictConfig) -> list[dict]:
    """The published numbers, from config, with every declared field present in every row."""
    rows = []
    for item in (OmegaConf.to_container(cfg.get("literature")) or []):
        rows.append({field: item.get(field, "") for field in roster_mod.LITERATURE_FIELDS})
    return rows


def _preamble(cfg: DictConfig, gold_by_cohort: dict, n_missing: int) -> str:
    sizes = ", ".join(
        f"`{c}` {len(g)} reports / {sum(len(v) for v in g.values())} gold pairs"
        for c, g in gold_by_cohort.items())
    return f"""Generated by `experiments/{EXP_ID}/run.py`.

Cohorts: {sizes}.

**Two estimators share one column.** RAG-HPO, AutoPCR and PhenoBERT have nothing to tune, so their
rows are plain cohort estimates at a single operating point. PhenoJury and TreePhenoRAG select their
operating point inside nested cross-validation, so their rows are pooled out-of-fold sets over the
same reports. The `estimator` column carries the distinction; do not quote the two as if they were
produced the same way.

**PhenoJury is quoted with its prompt selected inside each outer training split**, as chapter 5's
protocol does, with the reader pinned to `{roster_mod.PHENOJURY_NORMALISER}`, so these rows are
chapter 5's system. `t1_sensitivity.csv` says which prompts the folds chose on each cohort and what
each prompt would score if it were forced instead.

**Published figures are in `t1_literature.csv`, never in the headline table.** Each carries the
column of `t1_overall.csv` it is defined the same way as, and a comparability verdict with its
reason. {n_missing} roster cell(s) had no artifact and are reported with the path they looked for.
"""


def _read_csv_rows(path: Path) -> list[dict]:
    import csv
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _phenojury_tables_dir(overrides: dict, results_dir: str, cohort: str) -> Path | None:
    """``<run>/tables`` beside the PhenoJury prediction set the config names for *cohort*.

    The tables sit beside the prediction set (``<run>/predictions/full_pool.csv`` -> ``<run>/tables/``),
    so the cohort-to-directory mapping is written down once, in the config.
    """
    template = str((overrides.get("phenojury") or {}).get(cohort) or "")
    if not template:
        return None
    return Path(template.format(results_dir=results_dir, cohort=cohort)).parent.parent / "tables"


def _prompt_frequency(tables_dir: Path) -> list[dict]:
    return [r for r in _read_csv_rows(tables_dir / "s7_full_pool_pair_frequency.csv")
            if r.get("axis") == "prompt"]


def phenojury_modal_pairs(overrides: dict, results_dir: str, cohorts: list[str]) -> dict:
    """``{cohort: [prompt, normaliser]}`` -- the prompt the full pool's folds chose most often.

    For captions only (``tables_ch6.phenojury_prompt_phrase``, which says "most often"): the row
    itself is the in-fold selection, not this pair.
    """
    out = {}
    for cohort in cohorts:
        tables_dir = _phenojury_tables_dir(overrides, results_dir, cohort)
        freq = _prompt_frequency(tables_dir) if tables_dir else []
        if freq:
            top = max(freq, key=lambda r: int(r["n_folds"]))
            out[cohort] = [top["value"], roster_mod.PHENOJURY_NORMALISER]
    return out


def prompt_sensitivity_rows(overrides: dict, results_dir: str, cohorts: list[str]) -> list[dict]:
    """One ``t1_sensitivity`` row per cohort, read from the PhenoJury preparation run.

    Nothing here is typed in: this note used to quote hand-copied F1 values, which went stale the
    moment the prompt stopped being a constant.
    """
    rows = []
    for cohort in cohorts:
        tables_dir = _phenojury_tables_dir(overrides, results_dir, cohort)
        if tables_dir is None:
            continue
        freq = _prompt_frequency(tables_dir)
        forced = _read_csv_rows(tables_dir / "s8_prompt_sensitivity.csv")
        total = sum(int(r["n_folds"]) for r in freq)
        decision = ("selected in-fold: " + ", ".join(f"{r['value']} {r['n_folds']}/{total}"
                                                     for r in freq)
                    if freq else f"selected in-fold (no pair frequencies under {tables_dir})")
        alternative = ("forced: " + "; ".join(
            f"{r['prompt']} {float(r['micro_f1']):.4f} "
            f"[{float(r['micro_f1_lo']):.4f}, {float(r['micro_f1_hi']):.4f}]" for r in forced)
            if forced else f"no s8_prompt_sensitivity.csv under {tables_dir}")
        rows.append({"axis": "phenojury prompt", "cohort": cohort, "decision": decision,
                     "alternative": alternative, "source": str(tables_dir)})
    return rows


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="comparison")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    t0 = time.time()
    if not cfg.get("output_dir"):
        raise ValueError("output_dir is required")
    if not cfg.get("results_dir"):
        raise ValueError("results_dir is required — the directory holding the exp13/exp14 runs")
    run_output_dir = Path(cfg.output_dir) / EXP_ID
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_output_dir)
    (run_output_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    cohorts = list(cfg.cohorts)
    unknown = [c for c in cohorts if c not in roster_mod.COHORTS]
    if unknown:
        raise ValueError(f"unknown cohort(s) {unknown}; this experiment reports on "
                         f"{list(roster_mod.COHORTS)}")
    tables = report_mod.Tables(run_output_dir)
    results_dir = str(cfg.results_dir)
    # {method_key: {cohort: path}}, see roster.predictions_path for why this cannot be derived.
    overrides = OmegaConf.to_container(cfg.get("protocol_predictions")) or {}

    logger.info("── ontology ──")
    view = _load_view(str(cfg.get("hpo_json_path") or ""))

    gold_by_cohort = {c: load_cohort_gold(cfg, c) for c in cohorts}
    for cohort, gold in gold_by_cohort.items():
        logger.info("cohort %-16s %d reports, %d gold pairs", cohort, len(gold),
                    sum(len(v) for v in gold.values()))

    coverage, t1, t2, t3, policy = [], [], [], [], []
    loaded: dict = {}

    for cohort in cohorts:
        gold = gold_by_cohort[cohort]
        sampler = ReportResampler(len(gold), n_resamples=int(cfg.n_bootstrap), seed=0)
        for row in roster_mod.ROSTER:
            cell = overrides.get(row.key, {})
            declared = (row.key not in overrides) or bool(cell.get(cohort, ""))
            predicted, path, status = load_method(
                row, results_dir, cohort, str(cell.get(cohort, "") or ""), declared)
            if predicted is None:
                logger.warning("%-14s %-16s %s — %s", row.key, cohort, status, path)
                coverage.append({"method": row.key, "label": row.label, "cohort": cohort,
                                 "status": status, "path": path, "n_reports_predicted": 0})
                blank = {"method": row.key, "label": row.label, "family": row.family,
                         "cohort": cohort, "estimator": row.estimator_for(cohort), "status": status,
                         "note": row.note + (" | " if row.note else "") + f"looked for {path}"}
                t1.append(blank)
                t2.append(dict(blank))
                continue

            report_ids, gold_sets, pred_sets, book, raw = align(predicted, gold, view)
            loaded[(row.key, cohort)] = predicted
            coverage.append({
                "method": row.key, "label": row.label, "cohort": cohort, "status": "ok",
                "path": path, "n_reports_predicted": len(predicted),
                "n_reports_scored": len(report_ids),
                "n_reports_absent": book["n_reports_absent"],
                "estimator": row.estimator_for(cohort),
            })
            if book["n_reports_absent"]:
                logger.warning("%-14s %-16s %d cohort report(s) absent from the artifact (%s) — "
                               "scored as empty predictions, which costs the method every gold "
                               "term there.", row.key, cohort, book["n_reports_absent"],
                               book["reports_absent"])
            point = roster_mod.operating_point(row, results_dir, cohort)
            t1.append(t1_row(row, cohort, gold_sets, pred_sets, report_ids, book, sampler, point))
            policy.append(policy_row(row, cohort, gold_sets, pred_sets, raw, book))
            t2.append(t2_row(row, cohort, gold_sets, pred_sets, report_ids, view, sampler, point))
            last = t1[-1]
            logger.info("%-14s %-16s F1=%.4f [%.4f, %.4f]  P=%.4f R=%.4f", row.key, cohort,
                        last["micro_f1"], last["micro_f1_lo"], last["micro_f1_hi"],
                        last["micro_precision"], last["micro_recall"])

    n_missing = sum(1 for c in coverage if c["status"] != "ok")

    tables.add("t0_coverage", coverage,
               "Table 0 — what was found on disk, per (method, cohort)",
               columns=["label", "cohort", "status", "n_reports_scored", "n_reports_absent"],
               in_report=True)
    tables.add("t1_overall", t1,
               "Table 1 — overall extraction quality (micro and macro)",
               columns=["label", "cohort", "estimator", "operating_point", "tp", "fp", "fn",
                        "micro_precision", "micro_recall", "micro_f1", "micro_f1_lo",
                        "micro_f1_hi", "macro_precision", "macro_recall", "macro_f1",
                        "macro_f1_of_means", "macro_term_f1"],
               note="**A number the protocol chose must be quoted with its operating point.** "
                    "Unlike a swept number there is no configuration file a reader can look it up "
                    "in: the retrieval index, poolings, thresholds and k were selected inside the "
                    "folds, so `operating_point` reports the modal selection and how uniform it "
                    "was. Where an axis is not unanimous the label is a summary of several runs — "
                    "TreePhenoRAG's HCY index splits 4 `ontology_r3` to 1 `exemplar`, so one "
                    "fold's reports came from the other retrieval cache entirely.")
    tables.add("t2_hierarchy", t2,
               "Table 2 — hierarchy-aware quality: ancestor closure and CoPHE",
               columns=["label", "cohort", "estimator", "micro_hp", "micro_hr", "micro_hf",
                        "micro_hf_lo", "micro_hf_hi", "macro_hf",
                        "micro_cophe_precision", "micro_cophe_recall", "micro_cophe_f1"])

    # ── Table 3: HCY qualifier subgroups ────────────────────────────────────
    annotations_path = str(cfg.get("hcy_annotations_path") or "")
    if SUBGROUP_COHORT not in cohorts:
        tables.add_note("Table 3 — curated subgroups",
                        f"Skipped: `{SUBGROUP_COHORT}` is not in `cohorts`, and the qualifier "
                        f"labels exist only for it.")
    elif not annotations_path:
        tables.add_note("Table 3 — curated subgroups",
                        "Skipped: `hcy_annotations_path` is unset. It is the annotation sidecar "
                        "`hcy_curated_annotations.csv` beside the curated gold, and the qualifier "
                        "columns exist nowhere else — so the slice is skipped with a note rather "
                        "than approximated from the gold file.")
    else:
        rows = subgroups_mod.load_annotations(annotations_path)
        gold = gold_by_cohort[SUBGROUP_COHORT]
        groups = subgroups_mod.build(rows, gold)
        # One resampler per subgroup, shared by every method scored on it.
        samplers = {label: ReportResampler(max(1, len(groups[label].report_ids)),
                                           n_resamples=int(cfg.n_bootstrap), seed=0)
                    for label in subgroups_mod.RECALL_LABELS}
        audit = subgroups_mod.audit(rows, gold)
        for line in audit:
            logger.info("qualifier %-20s %-18s %4d labelled, %4d in gold, %2d reports",
                        line["label"], line["role"], line["n_pairs_labelled"],
                        line["n_pairs_in_gold"], line["n_reports_in_gold"])
        tables.add("t3_label_audit", audit,
                   "Table 3a — every curation qualifier, and which ones are sliced on",
                   columns=["label", "role", "n_pairs_labelled", "n_pairs_in_gold",
                            "n_pairs_not_in_gold", "n_reports_in_gold"])
        for row in roster_mod.ROSTER:
            predicted = loaded.get((row.key, SUBGROUP_COHORT))
            if predicted is None:
                t3.append({"method": row.key, "label": row.label, "subgroup": "-",
                           "status": "missing: no artifact on this cohort"})
                continue
            t3 += t3_rows(row, predicted, groups, samplers)
        tables.add("t3_subgroups", t3,
                   "Table 3 — recall on curated subgroups, and family attribution counts",
                   columns=["label", "subgroup", "kind", "n_reports", "n_gold", "n_found",
                            "recall", "recall_lo", "recall_hi", "n_attribution_pairs",
                            "n_emitted"],
                   note="`lab_value` and `implicit` are **recall only**: a prediction outside the "
                        "slice is neither a hit nor a miss, so there is no precision to report. "
                        "`family` is a **count**, not a rate — those terms belong to a relative "
                        "and are deliberately not in the gold, so each one emitted is already a "
                        "false positive in Table 1.")

    # ── T6.3: complementarity ───────────────────────────────────────────────
    # Two numbers, and they are not each other's complement. See
    # complementarity.py for why the split of TreePhenoRAG's losses counts more than the total.
    attribution_path = str(cfg.get("tree_attribution_path") or "")
    if not attribution_path:
        tables.add_note("Table 5 — complementarity",
                        "Skipped: `tree_attribution_path` is unset. It is "
                        "`recall_attribution.csv` from exp13_22's `diagnostics` stage, one row "
                        "per missed gold pair. The bucket totals in "
                        "`recall_decomposition.csv` cannot substitute: attributing recovery needs "
                        "to know WHICH pairs were lost, and inferring it from totals would assume "
                        "the two methods' misses are independent — the thing being measured.")
    else:
        attribution = complementarity_mod.load_attribution(attribution_path)
        if not attribution:
            tables.add_note("Table 5 — complementarity",
                            f"Skipped: no attribution rows at `{attribution_path}`. Re-run "
                            "exp13_22 with `stages=[...,diagnostics]` to write it.")
        else:
            comp = []
            # The roster row whose predictions do the recovering. Written into every row, so the
            # thesis caption names the configuration it counts (the full pool) instead of saying
            # "PhenoJury" about a system with three configurations.
            recovered_by = "phenojury"
            for cohort in cohorts:
                other = loaded.get((recovered_by, cohort))
                if other is None:
                    logger.warning("complementarity — PhenoJury has no artifact on %s, so there "
                                   "is nothing to recover WITH; skipped", cohort)
                    continue
                for row in complementarity_mod.build(attribution, other,
                                                     gold_by_cohort[cohort], view, "PhenoJury"):
                    comp.append({"cohort": cohort, "recovered_by_method": recovered_by, **row})
            if comp:
                tables.add("t5_complementarity", comp,
                           "Table 5 — how much of what one architecture misses the other recovers",
                           columns=["cohort", "bucket", "recovered_by", "n_missed",
                                    "n_recovered", "share_recovered", "meaning"],
                           note="The two rows are **not complements**: each asks what share of "
                                "one group of TreePhenoRAG's losses PhenoJury recovers, so their "
                                "shares have different denominators and must not be added. The "
                                "split is chapter 4's own — losses to pruning and retrieval are "
                                "terms the traversal never scored, where no threshold could have "
                                "helped, while losses at scored nodes are verdicts it got wrong. "
                                "Only a high figure in the first row argues for running both "
                                "pipelines; a high figure in the second says one disagrees with "
                                "the other's judgement, which is an argument for fixing the "
                                "judgement.")
            else:
                tables.add_note("Table 5 — complementarity",
                                "Skipped: PhenoJury has no artifact on any configured cohort.")

    tables.add("t1_scoring_policy", policy,
               "The same cells under both scoring conventions",
               columns=["label", "cohort", "micro_f1_normalised", "micro_f1_raw_strings", "delta",
                        "n_gold_pairs", "n_gold_scored", "n_pred_out_of_subtree"],
               note="Every number in Tables 1-3 is **normalised** through "
                    "`thesis_metrics.normalise_pair`, because `exp14_04` and `exp13_22` score that "
                    "way and PhenoJury's and TreePhenoRAG's published figures are therefore "
                    "normalised ones. `exp13_07` and `exp13_18` compare raw strings, so their "
                    "tables differ from this one by `delta`. The two knobs are gold terms outside "
                    "the phenotypic-abnormality subtree leaving the denominator (53 of GSC+'s 1011 "
                    "pairs) and predicted ones outside it no longer counting as false positives.")

    # ── Cost: runtime, model calls, memory and setup ────────────────────────
    cost_specs = OmegaConf.to_container(cfg.get("cost")) or {}
    subset_path = Path(str(cfg.get("subset_timing_csv") or "").format(results_dir=results_dir))
    subset_timing = cost_mod.load_subset_timing(subset_path) if cfg.get("subset_timing_csv") else {}
    logger.info("GSC+ subset timing: %d row(s) from %s", len(subset_timing), subset_path)
    cost_rows = []
    for cohort in cohorts:
        for row in roster_mod.ROSTER:
            spec = cost_specs.get(row.key)
            if not spec:
                cost_rows.append({"method": row.key, "label": row.label, "cohort": cohort,
                                  "source": "not declared in `cost`"})
                continue
            got = cost_mod.collect(spec, results_dir, cohort,
                                   roster_mod.artifact_cohort(cohort))
            got.method, got.label = row.key, row.label
            cost_mod.apply_subset_timing(got, row.key, subset_timing, subset_path)
            cost_rows.append(cost_mod.asdict(got))
            logger.info("%-14s %-16s %s s/report, %s calls/report, %s GB peak  [%s]",
                        row.key, cohort,
                        f"{got.seconds_per_report:.2f}" if got.seconds_per_report else "n/a",
                        f"{got.llm_calls_per_report:.1f}" if got.llm_calls_per_report else "n/a",
                        f"{got.peak_gpu_gb:.1f}" if got.peak_gpu_gb else "n/a", got.source)
    tables.add("t4_cost", cost_rows,
               "Table 4 — cost: runtime, model calls, peak memory and setup",
               columns=["label", "cohort", "cost_experiment", "seconds_per_report",
                        "llm_calls_per_report", "peak_gpu_gb", "model_load_s", "accelerator",
                        "checkpoint"],
               note="**The cost experiment is not the scoring experiment.** PhenoJury is scored "
                    "from exp14_04 and TreePhenoRAG from exp13_22, both CPU replays of minutes; "
                    "what they actually cost is the generation and cache-building runs named in "
                    "`cost_experiment`. Reading a runtime off the scoring experiment would report "
                    "the cost of the analysis and make the two most expensive systems look like "
                    "the two cheapest. Every cell carries its `source`, and `scope` says what "
                    "the number covers. Read PhenoBERT's scope before quoting its runtime: that "
                    "run reused a completed CNN output, so the figure is the linking pass alone "
                    "and is not its end-to-end time. `n/a` means no artifact records the "
                    "quantity; nothing here is estimated.")

    lit = literature_rows(cfg)
    tables.add("t1_literature", lit,
               "Published figures, with the column each is defined the same way as",
               columns=["label", "source", "cohort", "unit", "averaging", "compare_against",
                        "comparable", "precision", "recall", "f1"],
               note="A row with `comparable: false` shares neither this table's documents nor its "
                    "gold; `why_not` says which. It is printed for context and must not be put in "
                    "the same column as a measured row. "
                    "**Even the comparable rows have a denominator caveat.** Garcia et al. score "
                    "over all 1011 pairs of their re-annotation; this table scores over the 958 "
                    "that lie inside the phenotypic-abnormality subtree, because the other 53 are "
                    "codes no method in the roster searches for. Their recall is therefore over a "
                    "denominator 5.2 % larger than ours, which moves a comparison in their "
                    "disfavour. `t1_scoring_policy.csv` carries both counts.")

    sensitivity = prompt_sensitivity_rows(overrides, results_dir, cohorts)
    tables.add("t1_sensitivity", sensitivity,
               "PhenoJury's prompt: what the folds chose, and what forcing one would score",
               columns=["axis", "cohort", "decision", "alternative"],
               note="The decision is the in-fold choice of the full pool, counted over every outer "
                    "fold of every repetition; only repetition 0 is pooled into the headline row. "
                    "Each alternative is the full pool with that prompt forced on every fold, "
                    "out-of-fold, with rule, unit and k still selected inside the folds.")

    preamble = _preamble(cfg, gold_by_cohort, n_missing)
    report_mod.write_results_md(run_output_dir / "results.md", tables, preamble,
                                title="exp13_25 — the final comparison")
    manifest = {
        "exp_id": EXP_ID,
        "cohorts": cohorts,
        "roster": [r.key for r in roster_mod.ROSTER],
        "gold_paths": {"hcy": str(cfg.get("hcy_curated_gt_path") or ""),
                       "gsc": str(cfg.get("raghpo_dir") or "")},
        "hcy_annotations_path": annotations_path,
        "protocol_predictions": overrides,
        # The modal in-fold pair per cohort, for captions; `pair_selection` says it is modal.
        "phenojury_prompt": phenojury_modal_pairs(overrides, results_dir, cohorts),
        "pair_selection": "in_fold",
        "phenojury_normaliser": roster_mod.PHENOJURY_NORMALISER,
        "n_bootstrap": int(cfg.n_bootstrap),
        "cost_experiments": {k: v.get("exp_id", "") for k, v in cost_specs.items()},
        "n_missing_cells": n_missing,
        "cohort_sizes": {c: {"n_reports": len(g), "n_pairs": sum(len(v) for v in g.values())}
                         for c, g in gold_by_cohort.items()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (run_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("done in %.1fs — %s", time.time() - t0, run_output_dir / "tables")
    if n_missing:
        logger.warning("%d roster cell(s) had no artifact; see tables/t0_coverage.csv", n_missing)


if __name__ == "__main__":
    main()
