"""Driver for the AutoPCR 8B baseline / the AutoPCR 70B baseline, AutoPCR run on the patient reports, on HCY and GSC+.

AutoPCR (Tao et al., github.com/yctao7/AutoPCR, "Automated Phenotype Concept Recognition by
Prompting") is the closest published competitor to this thesis' own design: extract candidate
phrases, retrieve ontology concepts for each with SapBERT/FAISS, and, for the phrases retrieval is
*unsure* about, let a language model pick the concept. RAG-HPO and PhenoBERT, the two external
baselines already in the earlier runs, do not use an LLM as a linker, so without this row the comparison has no
member of the method family it is arguing against.

Two things are not upstream's:

* **The linker is a local model.** Upstream calls OpenAI / Groq / Together / vLLM. HCY is
  KiSpi patient data and may not be read by a hosted model (docs/cluster.md),
  so ``autopcr_runner.LocalPrompter`` supplies the same interface over a local LLaMA and refuses to
  run against an API provider at all.
* **The ontology is this repo's fixed release.** Upstream builds its dictionary from
  ``hp_20240208.obo``. Here it is built from ``resources/util/hpo.json`` by
  ``third_party/AutoPCR/build_dict_from_hpo_json.py``, so AutoPCR answers from the same 18 354 terms every
  other method is measured against rather than from a different HPO version.

Everything between the staged corpus and the output TSV is upstream's ``run_gsc_test_ner``,
unmodified. The artifacts emitted are the ones every other earlier method emits, so
``result_tables`` scores this through the same code path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path


from hpo_extraction.baselines.autopcr_runner import (
    build_model,
    check_dictionary,
    diagnose_autopcr_output,
    invalidate_stale_phrase_cache,
    link_source,
    load_local_linker,
    normalise_code,
    parse_autopcr_tsv,
    phrase_counts,
    require_parser_cache,
    run_autopcr,
    stage_corpus,
)
from hpo_extraction.models.gpu_memory import peak_gpu_bytes_all_devices, reset_peak_all_devices
from hpo_extraction.utils.mlflow_guard import mlflow_run
from hpo_extraction.ontology.hpo_tree import DEFAULT_HPO_JSON, HPOTree, hpo_label, resolve_hpo

VARIANT = "autopcr"


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
    return logging.getLogger("exp13_autopcr")


def collect_detections(
    out_path: str, pmid_to_id: dict[str, str], hpo_tree: HPOTree
) -> tuple[list[dict], dict[str, int]]:
    """Every accepted mention as a record, plus counters.

    ``linked_by`` is the column this experiment exists to produce: it separates the mentions
    SapBERT's index resolved on its own from the ones the language model decided, which is the
    only place AutoPCR's contribution over a pure retrieval baseline is visible.

    Codes are mapped onto the fixed ontology release (``hpo_extraction.ontology.hpo_tree.resolve_hpo``). Ones
    that resolve to nothing are kept with ``resolved: false`` and counted, never dropped quietly, a linker free-texting an id that does not exist is a finding, not noise.
    """
    with open(out_path, encoding="utf-8") as f:
        raw = f.read()

    records: list[dict] = []
    counts = {"n_rows": 0, "n_unmapped": 0, "n_alt_id": 0, "n_unknown_pmid": 0, "n_llm_linked": 0}

    for pmid, start, end, phrase, code, score in parse_autopcr_tsv(raw):
        counts["n_rows"] += 1
        report_id = pmid_to_id.get(pmid)
        if report_id is None:
            counts["n_unknown_pmid"] += 1
            continue
        resolved = resolve_hpo(hpo_tree, code)
        if resolved is None:
            counts["n_unmapped"] += 1
        elif resolved != code:
            counts["n_alt_id"] += 1
        source = link_source(score)
        counts["n_llm_linked"] += int(source == "llm")
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
            "linked_by": source,
        })
    return records, counts


def predicted_sets(records: list[dict], report_ids: list[str]) -> dict[str, list[str]]:
    """``{report_id: sorted predicted codes}``, ontology-resolved detections only.

    Unlike PhenoBERT there is no negation filter to apply: AutoPCR's neural extraction path carries
    a ``no_flag`` on its phrase items but never propagates it to the output, so assertion status is
    simply not modelled. A negated mention is a false positive by design, and that is a
    property of the method, not of this driver.
    """
    out: dict[str, set[str]] = {rid: set() for rid in report_ids}
    for rec in records:
        if not rec["resolved"]:
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


def write_segments(
    path: str, report_ids: list[str], calls: list[tuple[list[str], list]],
    tau_1: float, hpo_tree: HPOTree,
) -> int:
    """Ranked SapBERT candidates per **accepted** phrase.

    ``calls`` is one entry per document, in the order ``run_gsc_test_ner`` walks the staged corpus,
    captured by ``autopcr_runner.RecordingSapBERT``. The pairing with ``report_ids`` is positional
    and checked by the caller.

    Only phrases whose top candidate clears ``tau_1``, upstream's acceptance rule, are written.
    AutoPCR enumerates every n-gram up to length 10 of every clinical sub-sentence, so the full
    candidate list runs to millions of rows per cohort while saying nothing about any term the
    method actually predicted. The restriction is therefore deliberate, and it is why
    ``SEGMENT_RETRIEVAL`` is declared unsupported for this method in ``discovery.METRIC_SUPPORT``,
    not computed from this file.
    """
    n_rows = 0
    with open(path, "w", encoding="utf-8") as f:
        for rid, (phrases, anns) in zip(report_ids, calls):
            for phrase, cands in zip(phrases, anns):
                if not cands:
                    continue
                top_score = cands[0][1]
                if top_score is None or abs(float(top_score)) < tau_1:
                    continue
                chosen = normalise_code(str(cands[0][0]))
                for rank, (code, score) in enumerate(cands, start=1):
                    code = normalise_code(str(code))
                    f.write(json.dumps({
                        "report_id": rid,
                        "hpo_id": code,
                        "hpo_label": hpo_label(hpo_tree, code),
                        "rank": rank,
                        "text": phrase,
                        "cosine_sim": None if score is None else float(score),
                        # The concept AutoPCR settled on for this phrase. For an LLM-linked phrase
                        # it is the model's answer, which is why the score beside it is the -1.0
                        # sentinel, not a similarity.
                        "slm_verdict": "Yes" if rank == 1 and code == chosen else "No",
                    }) + "\n")
                    n_rows += 1
    return n_rows


def execute(cfg, exp_id: str) -> None:
    """Run the AutoPCR baseline on one cohort and write its predictions.

    Args:
        cfg: Hydra config of the AutoPCR baseline (configs/experiments/06_comparison/baseline_autopcr_*.yaml).
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
    logger.info(
        "Starting %s | dataset=%s ee=%s tau_1=%s tau_2=%s k=%s linker=%s ccr=%s | out=%s",
        exp_id, cfg.dataset, cfg.ee, cfg.tau_1, cfg.tau_2, cfg.k, cfg.llama_dir, cfg.ccr_model,
        run_output_dir,
    )

    reports, gold = load_cohort(cfg, logger)
    report_ids = sorted(reports)
    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("PILOT: restricted to %d report(s)", len(report_ids))

    corpus_dir = os.path.join(run_output_dir, "autopcr_corpus")
    corpus_path = os.path.join(corpus_dir, "corpus_test.tsv")
    out_path = os.path.join(run_output_dir, "autopcr_output.tsv")
    pmid_to_id = stage_corpus(reports, report_ids, corpus_path)
    dropped = invalidate_stale_phrase_cache(corpus_dir, set(pmid_to_id))
    if dropped:
        logger.info("Dropped phrase cache(s) that did not cover this corpus: %s", ", ".join(dropped))

    # First half of the neural++ chain (cluster/run_<ds>_exp13_19_extract.sh): stage the corpus the
    # parser in autopcr_ee_venv will read, and stop, no dictionary, no linker, no GPU.
    if cfg.get("stage_only", False):
        logger.info("STAGE ONLY | %d report(s) staged to %s", len(report_ids), corpus_path)
        return

    # Before the linker is placed: a missing parser cache found afterwards costs the allocation.
    require_parser_cache(corpus_dir, str(cfg.ee), dataset=str(cfg.dataset))

    # Six stat calls, before tens of minutes of placing a 70B on the GPU. A missing index
    # discovered afterwards costs the whole allocation.
    check_dictionary(cfg.ontology_dict)

    t0 = time.time()
    prompter = load_local_linker(
        cfg.llama_dir,
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
        max_new_tokens=int(cfg.get("max_new_tokens", 32)),
    )
    model = build_model(
        ontology_dict=cfg.ontology_dict,
        ccr_model=cfg.ccr_model,
        tau_1=float(cfg.tau_1),
        tau_2=float(cfg.tau_2),
        k=int(cfg.k),
        prompt_fn=prompter,
        el=str(cfg.get("el", "local")),
        seed=int(cfg.get("seed", 0)),
        use_cache=bool(cfg.get("use_cache", True)),
        device=cfg.get("ccr_device") or None,
        use_gpu_index=bool(cfg.get("use_gpu_index", False)),
    )
    load_s = time.time() - t0

    # Reset after load so the peak is inference's -- the weights stay allocated, so they still
    # count. Every card: the 70B linker is sharded across four, and device 0 alone holds a quarter.
    reset_peak_all_devices()
    t1 = time.time()
    run_autopcr(
        model, corpus_path, out_path, cfg.ontology_dict,
        tau_1=float(cfg.tau_1), ee=str(cfg.ee),
        only_longest=bool(cfg.get("only_longest", False)),
        abbr_recog=bool(cfg.get("abbr_recog", False)),
        stanza_dir=cfg.get("stanza_dir") or None,
    )
    annotate_s = time.time() - t1
    # Phrases per extraction source (stanza / benepar / conjunct), what each ee step contributed.
    n_phrases = phrase_counts(corpus_dir)
    logger.info("Phrases per source: %s", n_phrases)

    # A file full of rows that parse to nothing looks like a cohort with no phenotypes.
    # The Free Listing generation run shipped a whole grid of green jobs that had grounded zero HPO terms before anyone
    # noticed, so this is a hard failure, not an empty result file.
    diag = diagnose_autopcr_output(out_path)
    logger.info("AutoPCR output | %d blocks, %d rows, %d with an HP: code, %d unparsed",
                diag["n_blocks"], diag["n_rows"], diag["n_hp_rows"], diag["n_unparsed"])
    if diag["n_hp_rows"] == 0:
        logger.error(
            "NO DETECTIONS | %d blocks / %d rows in %s contain no HP: code. Check the staged "
            "corpus (%s) is non-empty and that the dictionary at %s was built for this ontology.",
            diag["n_blocks"], diag["n_rows"], out_path, corpus_path, cfg.ontology_dict,
        )
        raise SystemExit(3)

    hpo_tree = HPOTree()
    records, counts = collect_detections(out_path, pmid_to_id, hpo_tree)
    predicted = predicted_sets(records, report_ids)
    duration = time.time() - t1
    peak_mem = peak_gpu_bytes_all_devices()

    logger.info(
        "Detections | %d rows, %d LLM-linked, %d alt-id remapped, %d not in this HPO release, "
        "%d rows for an unknown pmid | %d linker calls",
        counts["n_rows"], counts["n_llm_linked"], counts["n_alt_id"], counts["n_unmapped"],
        counts["n_unknown_pmid"], prompter.n_calls,
    )
    if counts["n_unmapped"]:
        logger.warning(
            "%d detection(s) carry codes absent from %s. They are in autopcr_detections.jsonl "
            "with resolved=false and excluded from the predicted set. For an LLM-linked row this "
            "means the linker answered with an id that is not in the menu it was shown.",
            counts["n_unmapped"], DEFAULT_HPO_JSON,
        )
    if counts["n_unknown_pmid"]:
        logger.warning("%d output row(s) name a pmid that was never staged — check the corpus",
                       counts["n_unknown_pmid"])

    with mlflow_run(f"{exp_id}_{cfg.dataset}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "method": "autopcr", "dataset": cfg.dataset,
            "ee": cfg.ee, "tau_1": cfg.tau_1, "tau_2": cfg.tau_2, "k": cfg.k,
            "ccr_model": cfg.ccr_model, "llama_dir": cfg.llama_dir,
            "only_longest": cfg.get("only_longest", False),
            "abbr_recog": cfg.get("abbr_recog", False),
            "use_cache": cfg.get("use_cache", True),
            "ontology_dict": cfg.ontology_dict, "n_reports": len(report_ids),
        })

        preds_path = os.path.join(run_output_dir, f"{VARIANT}_predictions.jsonl")
        n_pred = write_predictions(preds_path, report_ids, predicted, gold, hpo_tree)

        det_path = os.path.join(run_output_dir, f"{VARIANT}_detections.jsonl")
        with open(det_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        seg_path = os.path.join(run_output_dir, f"{VARIANT}_retrieved_segments.jsonl")
        if len(model.calls) == len(report_ids):
            n_seg = write_segments(seg_path, report_ids, model.calls, float(cfg.tau_1), hpo_tree)
        else:
            # The pairing is positional: one predict_llm call per document, in staged order. If
            # that ever stops holding, the candidates would be attributed to the wrong reports,
            # so the file is not written at all, not written wrong.
            n_seg = 0
            logger.error(
                "candidate capture recorded %d call(s) for %d staged report(s) — refusing to "
                "write %s, the positional pairing no longer holds",
                len(model.calls), len(report_ids), seg_path,
            )

        timing_path = os.path.join(run_output_dir, f"{VARIANT}_timing.jsonl")
        with open(timing_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "stage": "annotate", "n_reports": len(report_ids),
                "n_detections": len(records), "n_predictions": n_pred,
                "n_llm_linked": counts["n_llm_linked"], "n_llm_calls": prompter.n_calls,
                "ee": str(cfg.ee), "n_phrases": n_phrases,
                "model_load_s": round(load_s, 3), "annotate_s": round(annotate_s, 3),
                "duration_s": round(duration, 3),
                "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
                "peak_gpu_gb": round(peak_mem / 1024 ** 3, 2),
            }) + "\n")

        artifacts = [preds_path, det_path, timing_path, os.path.join(run_output_dir, "run.log")]
        if n_seg:
            artifacts.insert(2, seg_path)
        for path in artifacts:
            mlflow.log_artifact(path)

        metrics = {
            "n_reports": len(report_ids), "n_predictions": n_pred,
            "n_detections": len(records), "n_llm_linked": counts["n_llm_linked"],
            "n_llm_calls": prompter.n_calls,
            "n_unmapped_codes": counts["n_unmapped"], "n_alt_id_codes": counts["n_alt_id"],
            "duration_seconds": duration,
            "mean_time_per_report_s": duration / len(report_ids) if report_ids else 0.0,
            "slm_calls_per_report": prompter.n_calls / len(report_ids) if report_ids else 0.0,
            "peak_gpu_mem_bytes": float(peak_mem),
            **{f"n_{source}": n for source, n in n_phrases.items()},
        }
        mlflow.log_metrics(metrics)

    logger.info(
        "Done | %d reports | %d detections (%d LLM-linked) -> %d predicted terms | %.1fs (%.2fs/report)"
        " | peak %.2f GB",
        len(report_ids), len(records), counts["n_llm_linked"], n_pred, duration,
        duration / max(len(report_ids), 1), peak_mem / 1024 ** 3,
    )
