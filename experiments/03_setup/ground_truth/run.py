"""The ground-truth build, every earlier method re-scored on the curated HCY ground truth, and that ground truth shipped.


The curation of HCY (``app/hcy_curation_ui``) has been walked end to end. What is not finished is
the *adjudication*, most rows carry no keep/remove verdict, and the mistake available in either
direction is to confuse the two. Scoring only against approved terms would measure the backlog. Scoring against every code the files carry would measure the pre-curation ground truth. So the ground truth here is
defined by **evidence**:

    every phenotype from ``prior_annotation``, ``daphne`` or a curator's own suggestion that sits on a
    segment and a trigger word, minus the disputed ones and minus three qualifiers.

Four things make the comparison a measurement rather than a mixture:

1. **The cohort is every annotated report**, and the ones no source annotated are absent from both
   ground-truth sets, not scored with an empty one, see ``curated_gold.cohort``.
2. **Both ground-truth sets are scored over that same cohort.** The `original` rows are the untouched
   ``hcy_ground_truth`` restricted to those reports, so `curated` minus `original` is a gold-side
   difference and nothing else. Comparing curated-subset against original-full-cohort would mix a
   change of ground truth with a change of denominator, which is the one error this table cannot survive.
3. **The evidence rule is the app's own.** ``curated_gold.anchor_index`` runs the same functions
   ``reader.needs_anchor`` puts behind the Evidence location tab, so a term this experiment drops is a term
   the curation screen showed as still needing work.
4. **The ground truth policy is priced, not assumed.** Six one-change variants of the inclusion rule are
   scored side by side (``gold_variants``), so "how much of this is the family exclusion", or the
   evidence requirement, is a row, not an argument.

Analysis only: no model, no GPU, no inference. It reads the artifacts an earlier exploratory run..the RAG-HPO reproduction with the published prompt wrote,
plus the curation log, and runs on CPU in seconds.

The `dataset` stage writes the ground truth back into the cohort directory as a **dated, self-describing
dataset**, the annotation table with every qualifier flagged, not filtered, the two-column
scoring ground truth, the per-report manifest and a `README.md` explaining all of it. That directory, not
this run directory, is what a second person picks up later. See :mod:`dataset`.

The ground truth is also written into the run directory as a plain two-column CSV, so the full thesis metric
set over it is one further command:

    python experiments/result_tables/run.py \\
      cohorts=[hcy,hcy_curated] \\
      hcy_curated_gt_path=<output_dir>/hcy_ground_truth/hcy_ground_truth_curated_eval.csv

Usage:
    # the ground truth and the shipped dataset alone, seconds, no results_dir needed
    python experiments/03_setup/ground_truth/run.py \\
        stages=[variants,dataset] hcy_dir=/path/to/hcy output_dir=/path/to/output

    # the full pass
    python experiments/03_setup/ground_truth/run.py \\
        results_dir=/path/to/output \\
        hcy_dir=/path/to/hcy \\
        hcy_gt_path=/path/to/hcy/hcy_ground_truth_raw.csv \\
        output_dir=/path/to/output
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from hpo_extraction.paths import REPO_ROOT as _REPO  # noqa: E402
for _p in (str(_REPO),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hpo_extraction.evaluation.result_tables.discovery as discovery  # noqa: E402, the result-table library's filesystem scan, reused verbatim
import hpo_extraction.evaluation.result_tables.loaders as loaders  # noqa: E402, the result-table library's translation layer, reused verbatim
import hpo_extraction.evaluation.result_tables.report as report_mod  # noqa: E402
import hpo_extraction.evaluation.result_tables.sections as sections  # noqa: E402

import curated_ground_truth as curated_gold  # noqa: E402
import dataset as dataset_mod  # noqa: E402, writes the shipped, dated dataset

EXP_ID = "hcy_ground_truth"
logger = logging.getLogger(EXP_ID)

#: The two cohort labels every table splits on. They are ground-truth sets over one identical report list,
#: which is the only way the delta column means anything.
CURATED = "curated"
ORIGINAL = "original"


def _setup_logging(run_output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(run_output_dir / "run.log"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _load_view(hpo_json_path: str):
    """The ontology wrapper the hierarchy metrics go through. Built once, ~15 s."""
    from hpo_extraction.evaluation.metrics import OntologyView
    from hpo_extraction.ontology.hpo_tree import HPOTree

    tree = HPOTree(hpo_json_path) if hpo_json_path else HPOTree()
    tree.buildHPOTree()
    return OntologyView(tree)


# ──────────────────────────────────────────────────────────────────────────────
# scoring
# ──────────────────────────────────────────────────────────────────────────────
def _runs_for(avail, gold_by_cohort: dict, report_ids: list) -> list:
    """Every (method, ground truth, configuration) with predictions on disk, aligned to *report_ids*.

    The restriction is applied to the **predictions** before alignment, not afterwards, so
    both ground-truth sets meet the same report list even where one of them happens to carry a
    report the other does not. ``loaders.aligned_sets`` intersects, and an intersection taken
    against two different left-hand sides is two different denominators.
    """
    keep = set(report_ids)
    runs = []
    for spec in discovery.METHODS:
        cell = avail.get(spec.key, "hcy")
        if not cell.usable:
            continue
        for point in cell.operating_points:
            predicted, _ = loaders.load_predictions(cell.files[point]["predictions"])
            restricted = {r: p for r, p in predicted.items() if r in keep}
            if not restricted:
                logger.warning("%s/%s: no predicted report is in the curated cohort — skipped.",
                               spec.key, point or "-")
                continue
            missing = keep - set(restricted)
            if missing:
                logger.warning("%s/%s: %d curated report(s) absent from the predictions "
                               "(e.g. %s) — excluded from this method's row.",
                               spec.key, point or "-", len(missing), sorted(missing)[0])
            for cohort_label, gold in gold_by_cohort.items():
                rids, gold_sets, pred_sets = loaders.aligned_sets(restricted, gold)
                if not rids:
                    continue
                runs.append({
                    "method": spec.key, "cohort": cohort_label, "point": point,
                    "spec": spec, "cell": cell,
                    "report_ids": rids, "gold_sets": gold_sets, "pred_sets": pred_sets,
                    "predicted": restricted,
                })
    return runs


def _delta_rows(core_rows: list) -> list:
    """One row per method: its main numbers under both ground-truth sets, and the difference.

    Joined on the **main** row of each (method, ground truth) pair, which is that method's best
    configuration *under that ground truth*. Two ground truths can prefer different configurations, and forcing
    them onto one would report a method at a threshold it would not have chosen, so the operating
    point each side used is a column, and a row where they differ says so.
    """
    headline = {(r["method"], r["cohort"]): r
                for r in core_rows if r.get("is_headline") and r.get("status") == "ok"}
    rows = []
    for spec in discovery.METHODS:
        cur = headline.get((spec.key, CURATED))
        org = headline.get((spec.key, ORIGINAL))
        if cur is None or org is None:
            continue
        row = {
            "method": spec.key, "label": spec.label, "experiment": spec.exp_id,
            "n_reports": cur["n_reports"],
            "operating_point_curated": cur["operating_point"],
            "operating_point_original": org["operating_point"],
            "same_operating_point": cur["operating_point"] == org["operating_point"],
        }
        for metric in ("micro_precision", "micro_recall", "micro_f1", "macro_f1"):
            row[f"{metric}_original"] = org[metric]
            row[f"{metric}_curated"] = cur[metric]
            row[f"delta_{metric}"] = cur[metric] - org[metric]
        for count in ("tp", "fp", "fn"):
            row[f"{count}_original"] = org[count]
            row[f"{count}_curated"] = cur[count]
            row[f"delta_{count}"] = cur[count] - org[count]
        rows.append(row)
    return sorted(rows, key=lambda r: -r["delta_micro_f1"])


def _gold_diff_rows(curated: dict, original: dict) -> list:
    """Per (report, term): which ground truth carries it. The whole gold-side change, listed not counted."""
    rows = []
    for rid in sorted(curated):
        cur, org = curated[rid], original.get(rid, set())
        for code in sorted(cur | org):
            in_cur, in_org = code in cur, code in org
            rows.append({
                "patient_id": rid, "hpo_code": code,
                "in_curated": in_cur, "in_original": in_org,
                "change": "added" if in_cur and not in_org
                          else "removed" if in_org and not in_cur else "unchanged",
            })
    return rows


def _variant_rows(built: dict, original: dict) -> list:
    """One row per ground truth variant: how big it is, and how far it is from the original ground truth."""
    rows = []
    for name, result in built.items():
        gold = result.gold
        added = removed = 0
        for rid, codes in gold.items():
            org = original.get(rid, set())
            added += len(codes - org)
            removed += len(org - codes)
        policy = result.policy
        rows.append({
            "variant": name,
            "is_default": name == curated_gold.DEFAULT_VARIANT,
            "prior_annotation_fallback": policy.prior_annotation_fallback,
            "include_undecided": policy.include_undecided,
            "include_suggested": policy.include_suggested,
            "exclude_labels": ";".join(policy.exclude_labels) or "-",
            "n_reports": len(gold),
            "n_gold_pairs": sum(len(v) for v in gold.values()),
            "n_terms_dropped": sum(1 for t in result.terms if not t.in_gold),
            "added_vs_original": added,
            "removed_vs_original": removed,
        })
    return rows


def _drop_reason_rows(terms: list) -> list:
    """Why each excluded candidate was excluded, aggregated. The denominator's audit trail."""
    counts: dict = {}
    for term in terms:
        if term.in_gold:
            continue
        key = (term.reason, term.source)
        counts[key] = counts.get(key, 0) + 1
    return [{"reason": reason, "source": source, "n_terms": n}
            for (reason, source), n in sorted(counts.items(), key=lambda kv: -kv[1])]


def _location_rows(terms: list) -> list:
    """How the located candidates were placed, by strategy and source.

    `curated` and `lexical` are the two ends of it, a person pointing at the words, against this
    experiment scanning the report for them, and a ground truth whose terms are mostly the latter is one
    to look at before quoting. Unanchored candidates get their own line, not being absent,
    so the column sums to the candidate count.
    """
    counts: dict = {}
    for term in terms:
        key = (term.how or "(unanchored)", term.source)
        counts[key] = counts.get(key, 0) + 1
    return [{"how": how, "source": source, "n_terms": n,
             "in_gold": sum(1 for t in terms
                            if (t.how or "(unanchored)") == how and t.source == source
                            and t.in_gold)}
            for (how, source), n in sorted(counts.items(), key=lambda kv: -kv[1])]


class _EmptyAvail:
    """Stand-in for the preamble when the score stage was skipped, no cells were looked at."""

    def counts(self) -> dict:
        return {"present": 0, "missing": 0, "partial": 0}


def _preamble(cfg: DictConfig, result, avail, n_runs: int, original: dict) -> str:
    counts = avail.counts()
    summary = result.summary()
    in_cohort = len(result.gold)
    seen = len(result.reports)
    return f"""Generated by `{EXP_ID}`.

**Cohort.** {in_cohort} of {seen} HCY reports qualify, on any of `{', '.join(result.criteria)}`.
The other {seen - in_cohort} carry no annotation from any source and are absent from both gold sets
rather than scored with an empty one — see `tables/cohort_manifest.csv`.

**Gold.** {result.n_pairs} (report, term) pairs under the `{cfg.gold_variant}` policy, against
{sum(len(original.get(r, set())) for r in result.gold)} in the original gold over the same reports.
The rule is **evidence, not approval**: an annotation is in when it sits on a segment and a trigger
word that occurs in it, is not disputed, and carries none of
`{', '.join(result.policy.exclude_labels) or '—'}`. {summary['n_terms_dropped']} of
{summary['n_terms_considered']} candidate annotations were excluded, {summary['n_terms_unanchored']}
of them for want of a findable trigger word; `tables/drop_reasons.csv` says why,
`tables/anchor_how.csv` says how the survivors were placed, and `tables/gold_terms.csv` carries the
per-term decision.

**Comparison.** Every table below carries each method twice — once against `{CURATED}`, once
against `{ORIGINAL}` — over one identical report list, so a difference between the two is
gold-side by construction. {n_runs} (method, gold, operating point) runs scored;
{counts['present']} of the {counts['present'] + counts['missing'] + counts['partial']} exp13 cells
were usable (`missing`/`partial` cells are listed in `availability.md`).
"""


# ──────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../../../configs/experiments/03_setup", config_name="ground_truth")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    t0 = time.time()
    if not cfg.get("output_dir"):
        raise ValueError("output_dir is required")
    run_output_dir = Path(cfg.output_dir) / EXP_ID
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_output_dir)
    (run_output_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    stages = list(cfg.stages)
    tables = report_mod.Tables(run_output_dir)
    metrics_json: dict = {}

    # ── ground truth ────────────────────────────────────────────────────────────────
    logger.info("── the curated gold ──")
    paths = curated_gold.resolve_paths(cfg.get("hcy_dir") or "", cfg.get("prior_annotation_path") or "",
                                       cfg.get("confirmed_path") or "",
                                       cfg.get("curation_dir") or "",
                                       cfg.get("segments_path") or "",
                                       cfg.get("phenobert_dir") or "")
    logger.info("segments: %s", paths["segments"])
    logger.info("prior_annotation: %s", paths["prior_annotation"])
    logger.info("confirmed (daphne): %s", paths["confirmed"])
    logger.info("curation log dir: %s", paths["curation_dir"])

    annotations = curated_gold.load_annotations(paths)
    if not annotations.get("daphne") and not annotations.get("prior_annotation"):
        raise FileNotFoundError(
            f"neither annotation file could be read under {paths['hcy_dir']!r} — set hcy_dir, or "
            f"prior_annotation_path/confirmed_path, to the directory holding "
            f"{curated_gold.sources.HOLISTIC_FILE} and {curated_gold.sources.CONFIRMED_FILE}")
    # The reading surface. Required by the evidence rule, which cannot check that a trigger word
    # occurs in its segment without the segments.
    corpus = curated_gold.load_corpus(paths)
    fold = curated_gold.load_curation(paths["curation_dir"])

    criteria = tuple(cfg.criteria)
    policy = curated_gold.GoldPolicy(
        prior_annotation_fallback=str(cfg.prior_annotation_fallback),
        require_evidence=bool(cfg.require_evidence),
        include_undecided=bool(cfg.include_undecided),
        include_suggested=bool(cfg.include_suggested),
        exclude_labels=tuple(cfg.exclude_labels or ()),
    )
    variant_name = str(cfg.gold_variant)
    if variant_name != "config":
        if variant_name not in curated_gold.VARIANTS:
            raise ValueError(f"gold_variant must be 'config' or one of "
                             f"{list(curated_gold.VARIANTS)}, got {variant_name!r}")
        policy = curated_gold.VARIANTS[variant_name]
    logger.info("policy: %s", policy)

    # Loaded before the cohort, not after, because `original_gold` is a cohort criterion: the
    # reports the original ground truth enumerates, including the ones it lists with an empty code list,
    # can be what admits a report. Everything below still reads `original_full` for the comparison.
    original_full = loaders.load_gold("hcy", hcy_gt_path=cfg.get("hcy_gt_path")) \
        if cfg.get("hcy_gt_path") else {}
    if "original_gold" in criteria and not original_full:
        # Failing open here would silently drop the reports the criterion exists to admit, and the
        # only visible trace would be a cohort five reports smaller than the one the config asks
        # for, which is the kind of difference this experiment is supposed to make legible.
        raise ValueError(
            "criteria includes 'original_gold', which admits reports by their presence in the "
            "ORIGINAL gold, but hcy_gt_path is unset or empty. Set hcy_gt_path (e.g. "
            "<hcy_dir>/hcy_ground_truth_raw.csv), or drop 'original_gold' from criteria.")

    reports = curated_gold.cohort(annotations, fold, criteria, corpus,
                                  prior_gold_ids=set(original_full))
    result = curated_gold.build(annotations, fold, policy, criteria, reports=reports,
                                corpus=corpus)
    if not result.gold:
        raise RuntimeError(
            "no report qualifies for the curated cohort. Either both annotation files are empty "
            "and the curation log carries no suggestion, or `criteria` excludes every signal. "
            "Check tables/cohort_manifest.csv.")
    if not result.n_pairs:
        raise RuntimeError(
            "the cohort is non-empty but the gold is: every candidate annotation was dropped. The "
            "usual cause is a segmentation that does not match the reports the annotations were "
            "written against, which makes every trigger word unanchorable — check "
            "tables/drop_reasons.csv for a `no_evidence` count equal to the candidate count, and "
            f"check that {paths['segments']} is the segmentation the curation app was run on.")

    # A report that qualified on its annotations and then lost all of them to the evidence rule is
    # scored against an empty ground truth, so every prediction on it is a false positive. That is the
    # right arithmetic if the report really has no locatable phenotype and the wrong arithmetic if
    # its segmentation is simply missing, and the two are indistinguishable from a P/R table, so
    # they are named here instead.
    starved = sorted(r.patient_id for r in result.reports
                     if r.in_cohort and not r.n_gold_terms and r.n_unanchored)
    if starved:
        logger.warning("%d report(s) are in the cohort with an EMPTY gold because every candidate "
                       "annotation was unanchorable (%s%s). Every prediction on them scores as a "
                       "false positive. Check that %s covers them.",
                       len(starved), ", ".join(starved[:5]),
                       "" if len(starved) <= 5 else ", …", paths["segments"])

    gold_path = run_output_dir / cfg.gold_filename
    curated_gold.write_gold_csv(gold_path, result.gold)
    logger.info("curated gold written → %s", gold_path)

    original = {rid: set(original_full.get(rid, set())) for rid in result.gold}
    if not original_full:
        logger.warning("hcy_gt_path is unset — the `%s` comparison rows will be empty.", ORIGINAL)
    else:
        unknown = [rid for rid in result.gold if rid not in original_full]
        if unknown:
            logger.warning("%d curated report(s) have no row in the original gold file "
                           "(e.g. %s) — scored against an empty original gold.",
                           len(unknown), unknown[0])
        # A report admitted ONLY because the original ground truth lists it, whose original row is not
        # empty, is the one case where this criterion is doing something indefensible: the original
        # ground truth claims terms there and the curated ground truth would score every one of them as a false
        # positive. It does not arise on today's data, all five such reports are empty rows, but
        # it must never pass in silence if the inputs change.
        by_id = {r.patient_id: r for r in result.reports}
        contested = sorted(rid for rid in result.gold
                           if by_id[rid].reason == "original_gold" and original_full.get(rid))
        if contested:
            logger.warning(
                "%d report(s) entered the cohort on the `original_gold` signal ALONE while the "
                "original gold lists terms for them (%s%s): their curated gold is empty, so every "
                "term the original gold claims scores as a false positive. Check that the curation "
                "pass actually reached them before trusting these rows.",
                len(contested), ", ".join(contested[:5]),
                "" if len(contested) <= 5 else ", …")

    tables.add("cohort_manifest", [asdict(r) for r in result.reports],
               "Which HCY reports the curated cohort admits, and on which signal",
               ["patient_id", "in_cohort", "reason", "n_segments", "n_prior_annotation", "n_daphne",
                "n_suggestions", "n_adjudicated", "is_confirmed", "n_gold_terms", "n_dropped",
                "n_unanchored", "difficulty"],
               note="`in_cohort=False` reports appear in neither gold set — they are the ones no "
                    "annotation source carries anything for, and an empty gold cell would claim "
                    "instead that they have no phenotypes.")
    # ``segment_text`` is dropped here and nowhere else. ``Tables.add`` writes every field to the
    # CSV regardless of the printed column list, and a segment is a **sentence of a patient
    # report**, this run directory carries codes, ids and trigger words and has never carried
    # prose. The sentence does ship, in the dataset the `dataset` stage writes into `hcy_dir`,
    # where it sits under the curation log's own group-only modes.
    tables.add("gold_terms", [{k: v for k, v in asdict(t).items() if k != "segment_text"}
                              for t in result.terms],
               "Every candidate gold term, with the reason it is in or out",
               ["patient_id", "hpo_code", "hpo_name", "source", "status", "labels",
                "anchored", "how", "segment_idx", "trigger_word", "in_gold", "reason"],
               note="One row per candidate, not per gold pair: a term dropped for an `unsure` "
                    "label or for want of a findable trigger word is here with that reason, so "
                    "the denominator can be reconstructed.",
               in_report=False)
    tables.add("drop_reasons", _drop_reason_rows(result.terms),
               "Why candidate annotations did not enter the curated gold",
               note="`no_evidence` is the evidence rule — no segment, or a trigger word that "
                    "occurs nowhere in the report. `label:*` are the qualifier exclusions, "
                    "`status:*` the curator verdicts. A large `unruled:excluded` count means the "
                    "policy in force is one of the restrictive variants.")
    tables.add("anchor_how", _location_rows(result.terms),
               "How each surviving annotation was placed on its words",
               note="`curated` is a person's own placement and `lexical` is this experiment's "
                    "guess — the trigger word scanned across the whole report. The two are not "
                    "the same evidence, and a gold whose terms are mostly `lexical` is one to "
                    "look at before quoting.")
    tables.add("gold_diff", _gold_diff_rows(result.gold, original),
               "Curated gold against the original, per (report, term)",
               ["patient_id", "hpo_code", "in_original", "in_curated", "change"],
               note="Restricted to the curated cohort, so every row is a real gold-side change "
                    "rather than a report the original file simply also covers.",
               in_report=False)

    changes = _gold_diff_rows(result.gold, original)
    metrics_json["gold"] = {
        **result.summary(),
        "gold_variant": variant_name,
        "n_original_pairs_in_cohort": sum(len(v) for v in original.values()),
        "n_terms_added": sum(1 for r in changes if r["change"] == "added"),
        "n_terms_removed": sum(1 for r in changes if r["change"] == "removed"),
        "gold_csv": str(gold_path),
    }
    logger.info("gold: %d pair(s) over %d report(s); %+d vs the original gold on the same reports",
                result.n_pairs, len(result.gold),
                result.n_pairs - metrics_json["gold"]["n_original_pairs_in_cohort"])

    if "variants" in stages:
        logger.info("── gold-policy sensitivity ──")
        built = curated_gold.variants(annotations, fold, criteria, corpus=corpus,
                                      prior_gold_ids=set(original_full))
        variant_rows = _variant_rows(built, original)
        tables.add("gold_variants", variant_rows,
                   "Gold-policy sensitivity: one change each from the default",
                   note="One shared cohort, six gold definitions. Each row differs from "
                        "`default` in exactly one field, so its `n_gold_pairs` is the price of "
                        "that single decision — `keep_unanchored` is what the evidence rule "
                        "costs, `keep_family` what the one editorial exclusion costs.")
        metrics_json["gold_variants"] = variant_rows

    # ── the shipped dataset ─────────────────────────────────────────────────
    if "dataset" in stages:
        logger.info("── the shipped dataset ──")
        if not cfg.get("hcy_dir") and not cfg.get("dataset_dir"):
            raise ValueError("stages includes 'dataset', which needs hcy_dir (or dataset_dir) to "
                             "know where to write")
        payload = dataset_mod.write(cfg.get("hcy_dir") or "", result, paths,
                                    date=str(cfg.get("dataset_date") or ""),
                                    dest=str(cfg.get("dataset_dir") or ""))
        logger.info("dataset → %s", payload["path"])
        logger.info("  %s  — the scoring gold (patient_id,hpo_codes)",
                    payload["files"]["gold"])
        logger.info("  %s  — every candidate, qualifiers flagged not filtered",
                    payload["files"]["annotations"])
        logger.info("  %s  — how to use it", payload["files"]["readme"])
        metrics_json["dataset"] = {k: v for k, v in payload.items() if k != "counts"}
        (run_output_dir / "dataset_manifest.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ── score ───────────────────────────────────────────────────────────────
    core_rows: list = []
    avail = _EmptyAvail()
    if "score" in stages:
        logger.info("── re-scoring every exp13 method ──")
        if not cfg.get("results_dir"):
            raise ValueError("stages includes 'score', which requires results_dir")
        avail = discovery.discover(cfg.results_dir)
        report_mod.write_availability_md(
            run_output_dir / "availability.md",
            discovery.availability_rows(avail), discovery.metric_support_rows(),
            {(s.key, c): discovery.rerun_command(s, c)
             for s in discovery.METHODS for c in discovery.BASE_COHORTS},
            avail.counts())

        gold_by_cohort = {CURATED: result.gold}
        if original_full:
            gold_by_cohort[ORIGINAL] = original
        runs = _runs_for(avail, gold_by_cohort, result.report_ids)
        logger.info("%d (method, gold, operating point) run(s) to score", len(runs))

        core_rows = sections.core_quality(runs)
        gaps = [sections.placeholder_row(spec.key, CURATED,
                                         avail.get(spec.key, "hcy").detail or "cell is unusable")
                for spec in discovery.METHODS if not avail.get(spec.key, "hcy").usable]
        tables.add("core_quality", core_rows + gaps,
                   "Flat P/R/F1 (eq. 1) at every operating point, under both gold sets",
                   ["label", "cohort", "operating_point", "n_reports", "micro_precision",
                    "micro_recall", "micro_f1", "macro_f1", "status", "reason"],
                   note="`cohort` here is the **gold set**, not the corpus: both are the same "
                        "curated report list. `macro_f1` is the mean of per-report F1, the "
                        "convention exp13_07 uses.")
        headline = [r for r in core_rows if r.get("is_headline")]
        tables.add("core_quality_headline", headline + gaps,
                   "Main comparison: each method at its best operating point, both gold sets",
                   ["label", "cohort", "operating_point", "n_reports", "tp", "fp", "fn",
                    "micro_precision", "micro_recall", "micro_f1", "macro_f1", "status", "reason"])
        metrics_json["core_quality"] = core_rows + gaps

        delta = _delta_rows(core_rows)
        tables.add("curated_vs_original", delta,
                   "What curating the gold is worth, per method",
                   ["label", "n_reports", "micro_f1_original", "micro_f1_curated",
                    "delta_micro_f1", "delta_micro_precision", "delta_micro_recall",
                    "delta_tp", "delta_fp", "delta_fn", "same_operating_point"],
                   note="Each side is that method's best operating point **under its own gold**; "
                        "`same_operating_point=False` marks the methods where the two golds "
                        "disagree about which threshold is best, which is itself a result.")
        metrics_json["curated_vs_original"] = delta

        if "hierarchy" in stages:
            logger.info("── hierarchy-aware quality (eqs. 3-4) ──")
            view = _load_view(cfg.get("hpo_json_path") or "")
            hier_rows = sections.hierarchy_quality(runs, view)
            tables.add("hierarchy", hier_rows,
                       "Hierarchy-aware quality under both gold sets",
                       ["label", "cohort", "operating_point", "micro_f1_flat",
                        "micro_cophe_f1", "micro_hp", "micro_hr", "micro_hf", "hf_minus_f1"],
                       note="A curated gold that adds specific terms moves hF less than F1, "
                            "because an ancestor of a newly added term was already credited.")
            metrics_json["hierarchy"] = hier_rows

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        n_runs = len([r for r in core_rows if r.get("status") == "ok"])
        report_mod.write_results_md(
            run_output_dir / "results.md", tables,
            _preamble(cfg, result, avail, n_runs, original),
            title="exp13_18 — exp13 on the curated HCY gold")

    duration = time.time() - t0
    metrics_json["duration_s"] = round(duration, 2)
    (run_output_dir / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, default=str), encoding="utf-8")

    logger.info("── headline ──")
    for row in metrics_json.get("curated_vs_original", [])[:5]:
        logger.info("  %-42s micro-F1 %.4f → %.4f (%+.4f)", row["label"],
                    row["micro_f1_original"], row["micro_f1_curated"], row["delta_micro_f1"])

    try:
        import mlflow

        mlflow.set_experiment(cfg.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"{EXP_ID}_hcy"):
            mlflow.log_params({"gold_variant": variant_name, "criteria": ",".join(criteria),
                               "prior_annotation_fallback": policy.prior_annotation_fallback,
                               "require_evidence": policy.require_evidence,
                               "exclude_labels": ",".join(policy.exclude_labels)})
            mlflow.log_metrics({
                "n_reports_in_cohort": len(result.gold),
                "n_gold_pairs": result.n_pairs,
                "n_terms_unanchored": result.summary()["n_terms_unanchored"],
                "n_terms_added": metrics_json["gold"]["n_terms_added"],
                "n_terms_removed": metrics_json["gold"]["n_terms_removed"],
                **{f"delta_micro_f1_{r['method']}": r["delta_micro_f1"]
                   for r in metrics_json.get("curated_vs_original", [])},
            })
            mlflow.log_artifacts(str(run_output_dir))
    except Exception as exc:  # pragma: no cover, logging must never fail the analysis
        logger.warning("MLflow logging skipped: %s", exc)

    logger.info("Done in %.1f s → %s", duration, run_output_dir)


if __name__ == "__main__":
    main()
