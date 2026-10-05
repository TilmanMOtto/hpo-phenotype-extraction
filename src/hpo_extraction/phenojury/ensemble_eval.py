"""
Shared evaluation/aggregation for the SLM-ensemble experiments (exp06_XX).

Pipeline recap: N small LLMs each emit free-text symptom mentions per sentence,
PhenoBERT maps each model's output to HPO IDs (with per-sentence attribution via
`sentence_number:` markers, see `hpo_extraction.phenojury.phenobert`), and this module turns the
per-model / per-sentence HPO detections into:

  * per-SLM productivity metrics, how often a model wrote something *and* PhenoBERT
    then found an HPO in it;
  * two patient-level aggregations evaluated against ground truth:
      - union     : a patient's HPOs = union of every HPO any model produced;
      - majority  : per-sentence plurality, the HPO(s) with the most model votes win
                    (models that detected nothing don't vote. No >50% threshold. Ties keep all max-voted HPOs), unioned over sentences;
  * per-SLM "voted for the winner" statistics for the majority method.

GT comparison reuses `HCYDataset.evaluate` (children-aware matching, identical to
an earlier exploratory run): a predicted HPO counts as right if the HPO itself or any child is in the
patient's ground truth.
"""

import json
import logging
import os
import re
import tempfile
import time
from collections import Counter, defaultdict

import mlflow
import pandas as pd

from hpo_extraction.phenojury.phenobert import (
    build_phenobert_input,
    diagnose_phenobert_output,
    parse_phenobert_sentences,
    phenobert_output_is_usable,
    phenobert_output_matches_input,
    run_phenobert,
    write_phenobert_input,
)
from hpo_extraction.models.verdict import strip_think
from hpo_extraction.ontology.hpo_tree import HPOTree
from hpo_extraction.utils.mlflow_guard import mlflow_run

logger = logging.getLogger(__name__)

# Heuristic "the model declined to extract anything" sentinels (after normalisation).
#
# ``findings`` is the empty-JSON case: normalisation strips punctuation, so an earlier exploratory run's
# ``{"findings": []}`` reduces to that one bare word. Without it a q4 cell that correctly declined
# on every sentence would be counted as having written something on every sentence, and its echo
# rate and label exactness, which are means over the records that wrote, would be computed over
# a set of replies containing no terms at all.
_NOTHING_EXACT = {
    "", "no", "none", "na", "n a", "nil", "negative", "no phenotype",
    "no phenotypes", "no symptoms", "no symptom", "no findings", "findings",
}
_NOTHING_SUBSTR = ("no phenotype", "no symptom", "no clinical sign", "no abnormal")


def wrote_something(text: str | None) -> bool:
    """True if an SLM reply is a real extraction (not empty / a 'no phenotype' reply)."""
    if not text:
        return False
    text = strip_think(text)  # drop reasoning CoT so it can't mask a 'no phenotype' reply
    cleaned = re.sub(r"[^a-z0-9/ ]", "", text.strip().lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in _NOTHING_EXACT:
        return False
    return not any(pat in cleaned for pat in _NOTHING_SUBSTR)


# ──────────────────────────────────────────────────────────────────────────────
# PhenoBERT per model (reuse-aware)
# ──────────────────────────────────────────────────────────────────────────────
def run_phenobert_per_model(
    model_name, records, patient_ids, run_output_dir, cfg, reuse, timing_records, logger
) -> dict[str, dict[int, dict[str, int]]]:
    """
    Annotate one model's extractions with PhenoBERT and return per-sentence HPOs::

        {patient_id: {sentence_number: {hpo_id: count}}}

    Reuses an existing `phenobert_output_{model}/` directory when `reuse` is True and it holds
    detections. Otherwise re-runs. Attribution comes from the `sentence_count` column when the
    install writes one, and otherwise from the detection offsets resolved against the very input
    this function generated, so a stock PhenoBERT works too.
    """
    pb_output_dir = os.path.join(run_output_dir, f"phenobert_output_{model_name}")

    # Strip reasoning CoT (<think>/<unused94>/tagless </think>) so PhenoBERT never sees reasoning
    # prose, otherwise symptom words mentioned mid-reasoning become spurious HPO detections (see
    # experiments/findings/exp08_findings.md). The saved raw llm_extractions_{model}.jsonl is
    # untouched. This is a boundary transform only. Built before the reuse branch because the
    # offset attribution needs the same text whether PhenoBERT runs now or ran earlier.
    stripped_records = [
        {**r, "llm_output": strip_think(r.get("llm_output", "") or "")}
        for r in records
    ]
    pb_inputs = {str(pid): build_phenobert_input(stripped_records, pid) for pid in patient_ids}

    # The cache must have been produced from *this* input, not just hold detections: the offsets are
    # what attributes a row to a sentence, and a cache built from a different layout verifies against
    # nothing and parses to an empty result that looks like "PhenoBERT found nothing".
    if reuse and phenobert_output_is_usable(pb_output_dir) and \
            phenobert_output_matches_input(pb_output_dir, pb_inputs):
        logger.info("Reusing cached PhenoBERT output for %s: %s", model_name, pb_output_dir)
        timing_records.append(
            {"stage": f"phenobert_{model_name}", "duration_s": 0.0, "status": "cached"}
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_input_dir = os.path.join(tmpdir, f"pb_input_{model_name}")
            # Stripping can empty a record outright (an unterminated <think> block is dropped
            # whole), and if it empties *every* record PhenoBERT is handed blank files and
            # cheerfully reports zero detections. Count it, and refuse to continue on a total wipe.
            n_raw_nonempty = sum(1 for r in records if (r.get("llm_output") or "").strip())
            n_kept = sum(1 for r in stripped_records if r["llm_output"].strip())
            logger.info(
                "PhenoBERT input %s | %d records, %d non-empty raw, %d survive strip_think",
                model_name, len(records), n_raw_nonempty, n_kept,
            )
            if records and n_kept == 0:
                raise RuntimeError(
                    f"[{model_name}] every extraction is empty after strip_think "
                    f"({n_raw_nonempty}/{len(records)} were non-empty before it) — PhenoBERT would "
                    "score a blank input. Check max_new_tokens for this model: an unterminated "
                    "reasoning block is dropped in full."
                )
            write_phenobert_input(stripped_records, patient_ids, model_input_dir)
            n_input_files = sum(
                1 for fn in os.listdir(model_input_dir)
                if os.path.getsize(os.path.join(model_input_dir, fn)) > 0
            )
            if n_input_files == 0:
                raise RuntimeError(
                    f"[{model_name}] PhenoBERT input directory is all empty files — the "
                    f"{len(records)} extraction records match none of the {len(patient_ids)} "
                    "report ids passed in (id type/format mismatch?)."
                )
            t_pb = time.time()
            run_phenobert(
                phenobert_dir=cfg.phenobert_dir,
                input_dir=model_input_dir,
                output_dir=pb_output_dir,
                phenobert_python=cfg.phenobert_python or None,
                stanza_dir=cfg.phenobert_stanza_dir or None,
                p1=cfg.phenobert_p1,
                p2=cfg.phenobert_p2,
                p3=cfg.phenobert_p3,
                n_threads=cfg.phenobert_threads,
                # Retry-only. A verbose prompt (an earlier exploratory run D1: q7_recall x apertus) makes PhenoBERT
                # exceed the node's RAM on a cohort this driver has no other way to shrink. The
                # runner then re-runs the same command over slices of the same input. Every cell
                # that fits in one pass is untouched.
                chunk_size=cfg.get("phenobert_chunk_size"),
            )
            pb_dur = time.time() - t_pb
            timing_records.append(
                {"stage": f"phenobert_{model_name}", "duration_s": round(pb_dur, 3), "status": "ok"}
            )
            logger.info("PhenoBERT %s: done in %.1fs", model_name, pb_dur)

    diag = diagnose_phenobert_output(pb_output_dir)
    logger.info(
        "PhenoBERT output %s | %d file(s), %d row(s), %d with an HP: id, %d with a sentence index",
        model_name, diag["n_files"], diag["n_rows"], diag["n_hp_rows"], diag["n_indexed_rows"],
    )
    sent_hpos = parse_phenobert_sentences(pb_output_dir, inputs=pb_inputs)
    n_det = sum(len(h) for p in sent_hpos.values() for h in p.values())
    logger.info("PhenoBERT %s: HPOs for %d patients | %d detection(s)",
                model_name, len(sent_hpos), n_det)

    # Detections on disk that none of the attribution routes could place are a broken pipeline, not
    # An empty result, refuse rather than hand back {} (the Free Listing generation run "0 detections" failure).
    if diag["n_hp_rows"] > 0 and n_det == 0:
        raise RuntimeError(
            f"[{model_name}] PhenoBERT wrote {diag['n_hp_rows']} HPO rows but none could be "
            f"attributed to a sentence ({pb_output_dir}). Neither route worked: "
            f"{diag['n_indexed_rows']} rows carry a sentence_count column, and the detection "
            "offsets do not match the reconstructed input — so this output was produced from "
            "different extractions. Re-run with reuse_extractions=false, or inspect with: "
            "python scripts/validate_phenobert_offsets.py <run_dir>"
        )
    return sent_hpos


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation building blocks
# ──────────────────────────────────────────────────────────────────────────────
def build_target_dict(target_symptoms: list[str], hpo_tree: HPOTree):
    """Return (target_dict, filtered_target_symptoms) where target_dict[T] = {T}∪children."""
    target_dict = {
        tc: {tc} | set(hpo_tree.data[tc]["Child"].keys())
        for tc in target_symptoms
        if tc in hpo_tree.data
    }
    filtered = [tc for tc in target_symptoms if tc in target_dict]
    return target_dict, filtered


def derive_patient_hpos(sent_hpos_by_model):
    """Collapse per-sentence detections to per-patient HPO counts (summed over sentences)."""
    out: dict = {}
    for model, patients in sent_hpos_by_model.items():
        out[model] = {}
        for pid, sents in patients.items():
            agg: dict[str, int] = {}
            for hpos in sents.values():
                for h, c in hpos.items():
                    agg[h] = agg.get(h, 0) + c
            out[model][pid] = agg
    return out


def _build_response_dict(patient_ids, target_symptoms, target_dict, selected_by_patient):
    """Map selected raw HPOs → the canonical response_dict consumed by HCYDataset."""
    response_dict: dict = {}
    for pid in patient_ids:
        selected = selected_by_patient.get(pid, set())
        response_dict[pid] = {}
        for tc in target_symptoms:
            positive = bool(target_dict[tc] & selected)
            response_dict[pid][tc] = {0: {"response": "Yes" if positive else "No", "prompt": ""}}
    return response_dict


def aggregate_union(
    per_model_patient_hpos, active_model_names, patient_ids,
    target_symptoms, target_dict, min_detection_count,
):
    """Union: a patient's selected HPOs = every HPO any model detected (>= min count)."""
    selected_by_patient: dict = {}
    for pid in patient_ids:
        selected: set = set()
        for model in active_model_names:
            for h, c in per_model_patient_hpos.get(model, {}).get(pid, {}).items():
                if c >= min_detection_count:
                    selected.add(h)
        selected_by_patient[pid] = selected
    return _build_response_dict(patient_ids, target_symptoms, target_dict, selected_by_patient)


def aggregate_plurality(
    all_records, sent_hpos_by_model, active_model_names, patient_ids,
    target_symptoms, target_dict, min_detection_count,
):
    """
    Per-sentence plurality vote.

    For each (patient, sentence): every model votes once for each HPO it detected
    (>= min count). Models that detected nothing don't vote. The HPO(s) with the most
    votes win (ties → all winners kept). A patient's selected HPOs = union of winners
    over its sentences.

    Returns (response_dict, vote_rows) where vote_rows[i] = per-model
    {n_votes_cast, n_votes_won, pct_votes_won}.
    """
    selected_by_patient: dict = defaultdict(set)
    vote_stats = {m: {"n_votes_cast": 0, "n_votes_won": 0} for m in active_model_names}

    # All (patient, sentence) pairs that any model processed.
    sent_keys: dict = defaultdict(set)
    for model in active_model_names:
        for r in all_records.get(model, []):
            sent_keys[r["patient_id"]].add(r["sentence_number"])

    for pid, sentences in sent_keys.items():
        for sent_num in sentences:
            model_votes: dict = {}  # model -> set(hpo)
            tally: Counter = Counter()
            for model in active_model_names:
                hpos = {
                    h
                    for h, c in sent_hpos_by_model.get(model, {})
                    .get(pid, {})
                    .get(sent_num, {})
                    .items()
                    if c >= min_detection_count
                }
                if hpos:
                    model_votes[model] = hpos
                    for h in hpos:
                        tally[h] += 1
            if not tally:
                continue
            max_votes = max(tally.values())
            winners = {h for h, c in tally.items() if c == max_votes}
            selected_by_patient[pid] |= winners
            for model, hpos in model_votes.items():
                for h in hpos:
                    vote_stats[model]["n_votes_cast"] += 1
                    if h in winners:
                        vote_stats[model]["n_votes_won"] += 1

    response_dict = _build_response_dict(patient_ids, target_symptoms, target_dict, selected_by_patient)

    vote_rows = []
    for model in active_model_names:
        cast = vote_stats[model]["n_votes_cast"]
        won = vote_stats[model]["n_votes_won"]
        vote_rows.append({
            "model": model,
            "n_votes_cast": cast,
            "n_votes_won": won,
            "pct_votes_won": round(100.0 * won / cast, 2) if cast else 0.0,
        })
    return response_dict, vote_rows


# ──────────────────────────────────────────────────────────────────────────────
# SLM-level metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_slm_productivity(all_records, sent_hpos_by_model, active_model_names):
    """
    Per model: how often it wrote something, and of those how often PhenoBERT then
    found an HPO in that sentence. Reports both percentage denominators.
    """
    rows = []
    for model in active_model_names:
        recs = all_records.get(model, [])
        sm = sent_hpos_by_model.get(model, {})
        n_sentences = len(recs)
        n_wrote = 0
        n_wrote_and_hpo = 0
        for r in recs:
            if not wrote_something(r["llm_output"]):
                continue
            n_wrote += 1
            if sm.get(r["patient_id"], {}).get(r["sentence_number"]):
                n_wrote_and_hpo += 1
        rows.append({
            "model": model,
            "n_sentences": n_sentences,
            "n_wrote_something": n_wrote,
            "n_wrote_and_hpo": n_wrote_and_hpo,
            "pct_of_wrote": round(100.0 * n_wrote_and_hpo / n_wrote, 2) if n_wrote else 0.0,
            "pct_of_all": round(100.0 * n_wrote_and_hpo / n_sentences, 2) if n_sentences else 0.0,
        })
    return rows


def merge_slm_metrics(productivity_rows, vote_rows):
    """Join productivity rows with plurality vote-stat rows on model name."""
    vmap = {v["model"]: v for v in vote_rows}
    merged = []
    for p in productivity_rows:
        v = vmap.get(p["model"], {})
        merged.append({
            **p,
            "n_votes_cast": v.get("n_votes_cast", 0),
            "n_votes_won": v.get("n_votes_won", 0),
            "pct_votes_won": v.get("pct_votes_won", 0.0),
        })
    return merged


def write_slm_metrics(rows, path):
    """Write the per-model metric rows to a CSV file."""
    pd.DataFrame(rows).to_csv(path, index=False)


def format_slm_metrics_table(rows) -> str:
    """Render the SLM metrics as a fixed-width table for the terminal."""
    cols = [
        ("model", "model", 22),
        ("n_sentences", "sents", 7),
        ("n_wrote_something", "wrote", 7),
        ("n_wrote_and_hpo", "→HPO", 7),
        ("pct_of_wrote", "%wrote", 8),
        ("pct_of_all", "%all", 8),
        ("n_votes_cast", "votes", 7),
        ("n_votes_won", "won", 7),
        ("pct_votes_won", "%won", 8),
    ]
    header = "  ".join(f"{title:>{w}}" if key != "model" else f"{title:<{w}}" for key, title, w in cols)
    lines = ["", "SLM-level metrics (productivity → PhenoBERT; per-sentence plurality votes)", header, "-" * len(header)]
    for r in rows:
        cells = []
        for key, _, w in cols:
            val = r.get(key, "")
            cells.append(f"{val:<{w}}" if key == "model" else f"{val:>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def write_results_csv(metrics_union, metrics_majority, path):
    """Write the classical base_LLM.csv with two rows (aggregation = union / majority)."""
    rows = [
        {"aggregation": "union", **metrics_union},
        {"aggregation": "majority", **metrics_majority},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator, full post-extraction pipeline (Stage 2 + 3 + eval + logging)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_ensemble(
    *, cfg, run_output_dir, file_prefix, all_records, active_models,
    configured_models, failed_models, patient_ids, timing_records, t0, logger,
):
    """
    Run PhenoBERT (reuse-aware) for each active model, compute SLM-level metrics, build
    the union + per-sentence-plurality aggregations, evaluate both against ground truth,
    and write all artifacts (base_LLM.csv with two rows, slm_metrics.csv + terminal print,
    {prefix}_union/majority_predictions.jsonl, retrieved segments, timing) under an MLflow run.
    """
    active_model_names = [name for name, _ in active_models]
    reuse = bool(getattr(cfg, "reuse_extractions", True))

    # ── Stage 2: PhenoBERT per model → per-sentence HPOs ──────────────────────
    sent_hpos_by_model: dict = {}
    for model_name, _ in active_models:
        sent_hpos_by_model[model_name] = run_phenobert_per_model(
            model_name, all_records[model_name], patient_ids,
            run_output_dir, cfg, reuse, timing_records, logger,
        )
    per_model_patient_hpos = derive_patient_hpos(sent_hpos_by_model)

    # ── Target symptoms + children expansion ──────────────────────────────────
    if cfg.target_symptoms_path:
        target_symptoms = pd.read_csv(cfg.target_symptoms_path)["target_codes"].tolist()
    else:
        all_hpos: set = set()
        for patients in sent_hpos_by_model.values():
            for sents in patients.values():
                for hpos in sents.values():
                    all_hpos.update(hpos.keys())
        target_symptoms = sorted(all_hpos)

    hpo_tree = HPOTree()
    target_dict, target_symptoms = build_target_dict(target_symptoms, hpo_tree)
    logger.info(
        "Aggregating %d active models over %d target symptoms",
        len(active_model_names), len(target_symptoms),
    )

    # ── Stage 3: two aggregations ─────────────────────────────────────────────
    union_response = aggregate_union(
        per_model_patient_hpos, active_model_names, patient_ids,
        target_symptoms, target_dict, cfg.min_detection_count,
    )
    majority_response, vote_rows = aggregate_plurality(
        all_records, sent_hpos_by_model, active_model_names, patient_ids,
        target_symptoms, target_dict, cfg.min_detection_count,
    )

    # ── SLM-level metrics (reuse if present) ──────────────────────────────────
    slm_metrics_path = os.path.join(run_output_dir, "slm_metrics.csv")
    if reuse and os.path.isfile(slm_metrics_path):
        logger.info("Reusing cached SLM metrics: %s", slm_metrics_path)
        slm_rows = pd.read_csv(slm_metrics_path).to_dict("records")
    else:
        productivity_rows = compute_slm_productivity(all_records, sent_hpos_by_model, active_model_names)
        slm_rows = merge_slm_metrics(productivity_rows, vote_rows)
        write_slm_metrics(slm_rows, slm_metrics_path)
    table = format_slm_metrics_table(slm_rows)
    print(table)
    logger.info("SLM metrics written → %s", slm_metrics_path)

    total_duration = time.time() - t0
    n_pos_union = sum(1 for p in union_response.values() for s in p.values() if s[0]["response"] == "Yes")
    n_pos_majority = sum(1 for p in majority_response.values() for s in p.values() if s[0]["response"] == "Yes")

    # ── Logging + GT evaluation ───────────────────────────────────────────────
    with mlflow_run(logger=logger):
        params = {
            "active_models": ",".join(active_model_names),
            "active_model_count": len(active_model_names),
            "failed_models": ",".join(failed_models) if failed_models else "none",
            "phenobert_dir": cfg.phenobert_dir,
            "phenobert_threads": cfg.phenobert_threads,
            "min_detection_count": cfg.min_detection_count,
            "phenobert_p1": cfg.phenobert_p1,
            "phenobert_p2": cfg.phenobert_p2,
            "phenobert_p3": cfg.phenobert_p3,
            "max_new_tokens": cfg.max_new_tokens,
        }
        for model_name, model_path in configured_models:
            params[f"llm_dir_{model_name}"] = model_path
        mlflow.log_params(params)
        mlflow.log_metrics({
            "n_patients": len(patient_ids),
            "n_target_symptoms": len(target_symptoms),
            "union_positives": n_pos_union,
            "majority_positives": n_pos_majority,
            "duration_seconds": total_duration,
            "mean_time_per_report_s": total_duration / len(patient_ids) if patient_ids else 0.0,
        })

        timing_path = os.path.join(run_output_dir, "timing_records.json")
        with open(timing_path, "w") as f:
            json.dump(timing_records, f, indent=2)
        mlflow.log_artifact(timing_path)
        mlflow.log_artifact(slm_metrics_path)

        for model_name in active_model_names:
            p = os.path.join(run_output_dir, f"llm_extractions_{model_name}.jsonl")
            if os.path.isfile(p):
                mlflow.log_artifact(p)

        # Dashboard artifact: retrieved segments (first active model's outputs + union verdict)
        first_model = active_model_names[0]
        segments_path = os.path.join(run_output_dir, f"{file_prefix}_retrieved_segments.jsonl")
        with open(segments_path, "w") as f:
            for patient_id in patient_ids:
                patient_recs = [r for r in all_records[first_model] if r["patient_id"] == patient_id][:5]
                for target_code in target_symptoms:
                    verdict = union_response[patient_id][target_code][0]["response"]
                    for rank, rec in enumerate(patient_recs, start=1):
                        f.write(json.dumps({
                            "patient_id": patient_id,
                            "hpo_id": target_code,
                            "hpo_label": target_code,
                            "rank": rank,
                            "text": rec["llm_output"][:300],
                            "cosine_sim": 0.0,
                            "slm_verdict": verdict,
                        }) + "\n")
        mlflow.log_artifact(segments_path)

        if cfg.target_symptoms_path and cfg.ground_truth_path:
            from hpo_extraction.evaluation.datasets.hcy import HCYDataset
            dataset = HCYDataset(cfg.ground_truth_path, cfg.target_symptoms_path)

            logger.info("=== Union aggregation ===")
            metrics_union = dataset.evaluate(union_response, target_symptoms, output_dir=None)
            logger.info("=== Majority (per-sentence plurality) aggregation ===")
            metrics_majority = dataset.evaluate(majority_response, target_symptoms, output_dir=None)

            results_path = os.path.join(run_output_dir, "base_LLM.csv")
            write_results_csv(metrics_union, metrics_majority, results_path)
            mlflow.log_artifact(results_path)
            mlflow.log_metrics({f"union_{k}": v for k, v in metrics_union.items()})
            mlflow.log_metrics({f"majority_{k}": v for k, v in metrics_majority.items()})
            logger.info(
                "Union   | micro_f1=%.4f macro_f1=%.4f", metrics_union["micro_f1"], metrics_union["macro_f1"]
            )
            logger.info(
                "Majority| micro_f1=%.4f macro_f1=%.4f", metrics_majority["micro_f1"], metrics_majority["macro_f1"]
            )

            gt_dict = dataset.load_ground_truth()
            for agg_name, response in (("union", union_response), ("majority", majority_response)):
                predictions_path = os.path.join(run_output_dir, f"{file_prefix}_{agg_name}_predictions.jsonl")
                with open(predictions_path, "w") as f:
                    for patient_id in response:
                        for hpo_code in target_symptoms:
                            predicted = int(response[patient_id][hpo_code][0]["response"] == "Yes")
                            gt_label = int(bool(target_dict[hpo_code] & set(gt_dict.get(patient_id, []))))
                            f.write(json.dumps({
                                "patient_id": patient_id,
                                "hpo_id": hpo_code,
                                "hpo_label": hpo_code,
                                "prediction": predicted,
                                "ground_truth": gt_label,
                            }) + "\n")
                mlflow.log_artifact(predictions_path)
        else:
            logger.warning(
                "target_symptoms_path/ground_truth_path not set — skipping GT metrics (base_LLM.csv)."
            )

        logger.info(
            "Run complete | patients=%d target_symptoms=%d union_pos=%d majority_pos=%d "
            "active=%d failed=%d duration=%.1fs",
            len(patient_ids), len(target_symptoms), n_pos_union, n_pos_majority,
            len(active_model_names), len(failed_models), total_duration,
        )
