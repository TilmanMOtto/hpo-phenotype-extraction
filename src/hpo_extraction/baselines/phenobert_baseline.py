"""Driver for the PhenoBERT baseline, PhenoBERT run directly on the raw report.

PhenoBERT (Feng et al., 2023) already lives in this repo, but only as the *grounding* step inside
the SLM ensemble (the Free Listing generation run), where it turns SLM-generated phrases into HPO codes. What it does on
its own, handed the patient report, has never been measured on either cohort, so the claim that
the ensemble improves on the published tool currently rests on that tool's own paper rather than on
these reports. This driver measures it, emitting the artifacts every other earlier method
emits so ``result_tables`` scores it through the same code path.

The method needs no GPU and no LLM: one ``annotate.py`` subprocess over a directory of reports,
inside the dedicated PhenoBERT env (``stanza==1.4.1``, incompatible with this one, see
:mod:`hpo_extraction.phenojury.phenobert`).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path


from hpo_extraction.phenojury.phenobert import (
    diagnose_phenobert_output,
    iter_phenobert_rows,
    run_phenobert,
)
from hpo_extraction.ontology.hpo_tree import DEFAULT_HPO_JSON, HPOTree, hpo_label, resolve_hpo
from hpo_extraction.utils.mlflow_guard import mlflow_run


def _setup_logging(output_dir: str) -> logging.Logger:
    log_path = os.path.join(output_dir, "run.log")
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("exp13_phenobert")


def safe_name(report_id: str) -> str:
    """Filename stem for a report id, HCY ids carry a ``:`` (``load_txt`` puts it there)."""
    return str(report_id).replace(":", "_")


def stage_reports(reports: dict[str, str], report_ids: list[str], input_dir: str) -> dict[str, str]:
    """Write one ``<stem>.txt`` per report and return ``{stem: report_id}``.

    The text goes in **verbatim**: the TSV's columns 1-2 are character offsets into this file, and
    they are the only thing tying a detection back to the phrase that produced it. Any
    normalisation here would silently invalidate every offset in the detections artifact.

    The id map is returned, not re-derived by inverting :func:`safe_name`, because that
    inversion is not a function: ``a_b`` could have come from ``a_b`` or ``a:b``.
    """
    os.makedirs(input_dir, exist_ok=True)
    stem_to_id: dict[str, str] = {}
    for report_id in report_ids:
        stem = safe_name(report_id)
        if stem in stem_to_id:
            raise ValueError(
                f"report ids {stem_to_id[stem]!r} and {report_id!r} both stage as {stem}.txt — "
                "PhenoBERT would annotate one and report it as the other"
            )
        stem_to_id[stem] = report_id
        with open(os.path.join(input_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(reports[report_id])
    return stem_to_id


def _output_is_complete(output_dir: str, stems: list[str]) -> bool:
    """True when *output_dir* holds an output file for every staged report.

    ``phenobert_output_is_usable`` only asks whether *some* detections are there, which a job
    killed part-way through the directory also satisfies. Reusing that cache would silently score a
    prefix of the cohort, so completeness is checked against the staged inputs instead.
    """
    if not os.path.isdir(output_dir):
        return False
    present = {os.path.splitext(f)[0] for f in os.listdir(output_dir)
               if os.path.isfile(os.path.join(output_dir, f))}
    return set(stems).issubset(present)


def collect_detections(
    output_dir: str, stem_to_id: dict[str, str], hpo_tree: HPOTree
) -> tuple[list[dict], dict[str, int]]:
    """Every PhenoBERT detection as a record, plus counters.

    Records carry ``negated``: a negated mention is still evidence the tool *saw* the phenotype and
    ruled it out, which is the case the side-by-side error analysis wants to separate
    from "never found it". Only non-negated detections become predictions.

    Codes are mapped onto the current ontology release (:func:`hpo_extraction.ontology.hpo_tree.resolve_hpo`). Ones that resolve to nothing are kept with ``resolved: false`` and counted, never dropped
    quietly.
    """
    records: list[dict] = []
    counts = {"n_rows": 0, "n_negated": 0, "n_unmapped": 0, "n_alt_id": 0, "n_unknown_file": 0}

    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem = os.path.splitext(fname)[0]
        report_id = stem_to_id.get(stem)
        if report_id is None:
            counts["n_unknown_file"] += 1
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()

        for start, end, phrase, code, after in iter_phenobert_rows(raw):
            counts["n_rows"] += 1
            negated = bool(after) and after[-1].strip().lower() == "neg"
            counts["n_negated"] += int(negated)
            resolved = resolve_hpo(hpo_tree, code)
            if resolved is None:
                counts["n_unmapped"] += 1
            elif resolved != code:
                counts["n_alt_id"] += 1
            score = None
            if after:
                try:
                    score = float(after[0])
                except ValueError:
                    score = None
            records.append({
                "report_id": report_id,
                "hpo_id": resolved or code,
                "hpo_label": hpo_label(hpo_tree, resolved or code),
                "raw_hpo_id": code,
                "resolved": resolved is not None,
                "phrase": phrase,
                "start": start,
                "end": end,
                "score": score,
                "negated": negated,
            })
    return records, counts


def predicted_sets(records: list[dict], report_ids: list[str]) -> dict[str, list[str]]:
    """``{report_id: sorted predicted codes}``, positive, ontology-resolved detections only."""
    out: dict[str, set[str]] = {rid: set() for rid in report_ids}
    for rec in records:
        if rec["negated"] or not rec["resolved"]:
            continue
        out.setdefault(rec["report_id"], set()).add(rec["hpo_id"])
    return {rid: sorted(codes) for rid, codes in out.items()}


def write_predictions(
    path: str, report_ids: list[str], predicted: dict[str, list[str]],
    gold: dict[str, list[str]], hpo_tree: HPOTree,
) -> int:
    """The earlier predictions artifact: per-term lines, then one authoritative summary line.

    ``result_tables/loaders.load_predictions`` reads the summary line, it is the only place a report
    that predicted *nothing* can appear at all, so it is written for every report, including those.
    """
    n_pred = 0
    with open(path, "w", encoding="utf-8") as f:
        for rid in report_ids:
            gold_set = set(gold.get(rid, []))
            codes = predicted.get(rid, [])
            n_pred += len(codes)
            for code in codes:
                f.write(json.dumps({
                    "report_id": rid, "hpo_id": code, "hpo_label": hpo_label(hpo_tree, code),
                    "prediction": 1, "ground_truth": int(code in gold_set),
                }) + "\n")
            f.write(json.dumps({
                "report_id": rid, "summary": True,
                "predicted_set": codes, "gold_set": sorted(gold_set),
            }) + "\n")
    return n_pred


def execute(cfg, exp_id: str) -> None:
    """Run the PhenoBERT baseline on one cohort and write its predictions.

    Args:
        cfg: Hydra config of the PhenoBERT baseline (configs/experiments/06_comparison/baseline_phenobert.yaml).
        exp_id: name of the result folder, created as ``<output_dir>/<exp_id>/<dataset>/``.
    """
    # MLflow and the cohort loaders are imported here, not at module scope so the parsing
    # and writing functions above, everything between the TSV and the predictions contract, and
    # The only part with logic worth testing, stay importable without the inference stack.
    import mlflow

    from hpo_extraction.data.loading import load_cohort

    run_output_dir = os.path.join(cfg.output_dir, exp_id, cfg.dataset)
    os.makedirs(run_output_dir, exist_ok=True)
    logger = _setup_logging(run_output_dir)
    logger.info("Starting %s | dataset=%s phenobert_dir=%s p1=%s p2=%s p3=%s | out=%s",
                exp_id, cfg.dataset, cfg.phenobert_dir, cfg.phenobert_p1, cfg.phenobert_p2,
                cfg.phenobert_p3, run_output_dir)

    reports, gold = load_cohort(cfg, logger)
    report_ids = sorted(reports)
    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("PILOT: restricted to %d report(s)", len(report_ids))

    pb_input_dir = os.path.join(run_output_dir, "phenobert_input")
    pb_output_dir = os.path.join(run_output_dir, "phenobert_output")
    stem_to_id = stage_reports(reports, report_ids, pb_input_dir)
    stems = sorted(stem_to_id)

    t0 = time.time()
    if cfg.get("reuse_phenobert", True) and _output_is_complete(pb_output_dir, stems):
        logger.info("Reusing complete PhenoBERT output: %s", pb_output_dir)
        annotate_s = 0.0
    else:
        run_phenobert(
            phenobert_dir=cfg.phenobert_dir,
            input_dir=pb_input_dir,
            output_dir=pb_output_dir,
            phenobert_python=cfg.get("phenobert_python") or None,
            stanza_dir=cfg.get("phenobert_stanza_dir") or None,
            p1=cfg.phenobert_p1,
            p2=cfg.phenobert_p2,
            p3=cfg.phenobert_p3,
            n_threads=cfg.phenobert_threads,
        )
        annotate_s = time.time() - t0
        if not _output_is_complete(pb_output_dir, stems):
            produced = {os.path.splitext(f)[0] for f in os.listdir(pb_output_dir)}
            missing = [s for s in stems if s not in produced]
            logger.warning("%d of %d reports have no output file (e.g. %s)",
                           len(missing), len(stems), missing[:3])

    # A directory full of files that parse to nothing looks like a cohort with no
    # phenotypes. The Free Listing generation run shipped a whole grid of green jobs that had grounded zero HPOs before
    # anyone noticed, so this is a hard failure, not an empty result file.
    diag = diagnose_phenobert_output(pb_output_dir)
    logger.info("PhenoBERT output | %d files, %d rows, %d HP: rows, %d with a sentence index",
                diag["n_files"], diag["n_rows"], diag["n_hp_rows"], diag["n_indexed_rows"])
    if diag["n_hp_rows"] == 0:
        logger.error(
            "NO DETECTIONS | %d files / %d rows in %s contain no HP: code. Check the staged input "
            "(%s) is non-empty and that phenobert_python points at the PhenoBERT env.",
            diag["n_files"], diag["n_rows"], pb_output_dir, pb_input_dir,
        )
        raise SystemExit(3)

    hpo_tree = HPOTree()
    records, counts = collect_detections(pb_output_dir, stem_to_id, hpo_tree)
    predicted = predicted_sets(records, report_ids)
    duration = time.time() - t0
    logger.info(
        "Detections | %d rows, %d negated, %d alt-id remapped, %d not in this HPO release, "
        "%d output files with no staged report",
        counts["n_rows"], counts["n_negated"], counts["n_alt_id"], counts["n_unmapped"],
        counts["n_unknown_file"],
    )
    if counts["n_unmapped"]:
        logger.warning(
            "%d detection(s) carry codes absent from %s — PhenoBERT ships an older HPO release. "
            "They are in phenobert_detections.jsonl with resolved=false and excluded from the "
            "predicted set.", counts["n_unmapped"], DEFAULT_HPO_JSON,
        )

    with mlflow_run(f"{exp_id}_{cfg.dataset}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "method": "phenobert", "dataset": cfg.dataset,
            "phenobert_dir": cfg.phenobert_dir, "phenobert_p1": cfg.phenobert_p1,
            "phenobert_p2": cfg.phenobert_p2, "phenobert_p3": cfg.phenobert_p3,
            "phenobert_threads": cfg.phenobert_threads, "n_reports": len(report_ids),
        })

        preds_path = os.path.join(run_output_dir, "phenobert_predictions.jsonl")
        n_pred = write_predictions(preds_path, report_ids, predicted, gold, hpo_tree)

        det_path = os.path.join(run_output_dir, "phenobert_detections.jsonl")
        with open(det_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        timing_path = os.path.join(run_output_dir, "phenobert_timing.jsonl")
        with open(timing_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "stage": "annotate", "n_reports": len(report_ids),
                "n_detections": len(records), "n_predictions": n_pred,
                "annotate_s": round(annotate_s, 3), "duration_s": round(duration, 3),
                "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
            }) + "\n")

        for path in (preds_path, det_path, timing_path,
                     os.path.join(run_output_dir, "run.log")):
            mlflow.log_artifact(path)

        mlflow.log_metrics({
            "n_reports": len(report_ids), "n_predictions": n_pred,
            "n_detections": len(records), "n_negated_detections": counts["n_negated"],
            "n_unmapped_codes": counts["n_unmapped"], "n_alt_id_codes": counts["n_alt_id"],
            "duration_seconds": duration,
            "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
            # No GPU stage at all, logged so the §Scalability table reads 0, not blank,
            # which is the whole point of the comparison against the LLM baselines.
            "peak_gpu_mem_bytes": 0,
        })

    logger.info("Done | %d reports | %d detections → %d predicted terms | %.1fs (%.2fs/report)",
                len(report_ids), len(records), n_pred, duration,
                duration / max(len(report_ids), 1))
