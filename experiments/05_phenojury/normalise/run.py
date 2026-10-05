"""The normalisation run, the same cached generations, read by four normalisers instead of one.


No model is run here and no generation is repeated. Every reply this reads was written by the Free Listing generation run
(``p0_baseline``) or the generation run of the other prompts (``q2_sentence_last``, ``q4_span_json``, ``q7_recall``). The only thing
that varies is what turns a generated string into an HPO identifier. The PhenoJury protocol re-runs the whole
S0-S8 analysis over what this writes.

Stages
------
``plan``            print the array decoding and every cell's status. Runs nothing.
``surface_matrix``  embed the ontology's label strings once for the SapBERT reader (the only GPU).
``link``            one ``(prompt, model, normaliser)`` cell, selected by ``array_index``.
``all``             every cell serially, fine for dictionary, slow for ``phenobert_candidates``.

Why the cells are independent: each reader is a pure function of the cached strings, so the array
can be resubmitted piecewise and a failed cell costs only itself. That is the same invariant
the Free Listing generation run relies on for its vote sweep, one layer earlier in the pipeline.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


from hpo_extraction.phenojury.normalisers import run_phenobert_candidates  # noqa: E402
from hpo_extraction.phenojury.normalisers import (  # noqa: E402
    SAPBERT_ENCODER,
    build_surface_matrix,
    cell_extractions,
    cell_models,
    detections_name,
    grid_cell_dir,
    linkage_name,
    load_surface_matrix,
    normalise_cell,
    source_cell_dir,
    write_detections,
    write_linkage,
)
from hpo_extraction.phenojury.prompt_diagnostics import _shape_of  # noqa: E402

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "phenojury_normalisation"
logger = logging.getLogger("exp14_03")


# ── The array ────────────────────────────────────────────────────────────────

def enumerate_cells(cfg, cohort: str) -> list[dict]:
    """Every ``(prompt, model, normaliser)`` cell, prompt-major then model then normaliser.

    The model list is read **per prompt from disk** rather than from a fixed roster, because the
    roster is not a fact about the cohort: an earlier exploratory run D1 has ``q7_recall x apertus`` failing grounding
    while its generation succeeded, so that condition has seven members and the array must not reserve an
    index for a cell that cannot exist. ``stage=plan`` prints the resulting width, which is what
    ``--array`` should be sized from.
    """
    cells = []
    for prompt_key in list(cfg.prompt_keys):
        cell_dir = source_cell_dir(cfg.results_dir, cohort, prompt_key)
        models = cell_models(cell_dir)
        if not models:
            raise FileNotFoundError(f"{cell_dir} holds no non-empty detections_*.jsonl")
        for model_key in models:
            for normaliser in list(cfg.normalisers):
                cells.append({
                    "prompt_key": prompt_key,
                    "model_key": model_key,
                    "normaliser": normaliser,
                    "source_dir": str(cell_dir),
                })
    return cells


def select_cell(cfg, cells: list[dict]) -> dict:
    """The cell this invocation runs, by explicit keys, or by array index."""
    if cfg.prompt_key and cfg.model_key and cfg.normaliser:
        for cell in cells:
            if (cell["prompt_key"] == cfg.prompt_key
                    and cell["model_key"] == cfg.model_key
                    and cell["normaliser"] == cfg.normaliser):
                return cell
        raise ValueError(
            f"no cell for ({cfg.prompt_key}, {cfg.model_key}, {cfg.normaliser}); "
            f"run stage=plan to see the {len(cells)} that exist"
        )
    if cfg.array_index is None:
        raise ValueError("pass array_index, or all three of prompt_key/model_key/normaliser")
    index = int(cfg.array_index)
    if not 0 <= index < len(cells):
        raise IndexError(
            f"array_index {index} is outside 0..{len(cells) - 1}; size --array from stage=plan"
        )
    return cells[index]


# ── The SapBERT encoder, injected, not imported ───────────────────────

def sapbert_encoder(sapbert_dir: str):
    """``encode_batch(list[str]) -> ndarray`` over a local SapBERT model.

    Defined here, not in ``hpo_extraction.phenojury.normalisers``, so that module imports on a login node with no
    torch, the same separation ``ontology_index.build_definition_index`` keeps and for the same
    reason. Mean-pooled over the attention mask, which is what ``bioTag_SapBERT`` does.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sapbert_dir)
    model = AutoModel.from_pretrained(sapbert_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    logger.info("SapBERT on %s from %s", device, sapbert_dir)

    def encode_batch(texts):
        batch = tokenizer(list(texts), padding=True, truncation=True, max_length=64,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.cpu().numpy()

    return encode_batch


def ensure_surface_matrix(cfg, tree, fingerprint: str):
    """``(ids, texts, vectors)`` from cache, building it if the cache is absent or stale."""
    cache_path = (Path(cfg.surface_matrix_path) if cfg.surface_matrix_path
                  else Path(cfg.output_dir) / EXP_ID / "surface_matrix_sapbert.npz")
    cached = load_surface_matrix(cache_path, fingerprint=fingerprint,
                                 encoder_name=SAPBERT_ENCODER)
    if cached is not None:
        logger.info("surface matrix from cache: %s", cache_path)
        return cached
    if not cfg.sapbert_dir:
        raise ValueError(
            f"no usable surface matrix at {cache_path} and no sapbert_dir to build one — "
            "run stage=surface_matrix first, on a node that has the checkpoint"
        )
    return build_surface_matrix(
        tree, sapbert_encoder(cfg.sapbert_dir), encoder_name=SAPBERT_ENCODER,
        cache_path=cache_path, fingerprint=fingerprint, batch_size=int(cfg.sapbert_batch_size),
    )


# ── Running one cell ─────────────────────────────────────────────────────────



def run_cell(cfg, cohort: str, cell: dict, shared: dict) -> dict:
    """Normalise one cell and write its two artifacts. Returns a status row."""
    started = time.time()
    prompt_key, model_key, normaliser = cell["prompt_key"], cell["model_key"], cell["normaliser"]
    out_dir = grid_cell_dir(cfg.output_dir, EXP_ID, cohort, prompt_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / detections_name(model_key, normaliser)
    link_path = out_dir / linkage_name(model_key, normaliser)

    if cfg.reuse and det_path.exists() and det_path.stat().st_size > 0:
        logger.info("cached, skipping: %s", det_path)
        return {**cell, "status": "cached", "duration_s": 0.0}

    source_dir = Path(cell["source_dir"])
    records = cell_extractions(source_dir, model_key)
    report_ids = sorted({str(r.get("patient_id", r.get("report_id", ""))) for r in records} - {""})
    shape = _shape_of(prompt_key)
    logger.info("%s / %s / %s | %d record(s), %d report(s), shape=%s",
                prompt_key, model_key, normaliser, len(records), len(report_ids), shape)

    kwargs = {"model_key": model_key, "shape": shape}
    work_dir = None
    if normaliser == "phenobert_raw":
        kwargs["cached_detections_path"] = source_dir / f"detections_{model_key}.jsonl"
        pb_dir = source_dir / f"phenobert_output_{model_key}"
        if pb_dir.is_dir():
            # Rebuild the input as the original run did, so the offsets verify. Without it
            # `_sentence_from_offset` refuses every row, which looks identical to "found nothing".
            from hpo_extraction.phenojury.phenobert import build_phenobert_input
            from hpo_extraction.models.verdict import strip_think
            stripped = [{**r, "llm_output": strip_think(r.get("llm_output", "") or "")}
                        for r in records]
            kwargs["phenobert_output_dir"] = str(pb_dir)
            kwargs["phenobert_inputs"] = {
                str(rid): build_phenobert_input(stripped, rid) for rid in report_ids
            }
    elif normaliser == "phenobert_candidates":
        work_dir = out_dir / f".work_{model_key}"
        pb_dir, inputs, line_spans = run_phenobert_candidates(
            cfg, cell, records, report_ids, shape, work_dir)
        kwargs.update(phenobert_output_dir=pb_dir, phenobert_inputs=inputs,
                      phenobert_line_spans=line_spans)
    elif normaliser == "dictionary":
        kwargs.update(surface2hpo=shared["surface2hpo"], ontology_index=shared.get("index"),
                      dictionary_mode=str(cfg.dictionary_mode),
                      dictionary_accept_sources=tuple(cfg.dictionary_accept_sources))
    elif normaliser == "sapbert":
        ids, _texts, vectors = shared["surface_matrix"]
        kwargs.update(surface_ids=ids, surface_vectors=vectors,
                      encode_batch=shared["encode_batch"], tau=float(cfg.sapbert_tau))

    detections, linkage = normalise_cell(
        records, report_ids, normaliser, prompt_key, **kwargs)

    write_detections(detections, model_key, det_path)
    write_linkage(linkage, link_path)
    n_pairs = sum(len(v) for r in detections.values() for v in r.values())
    duration = time.time() - started
    logger.info("wrote %s | %d report(s), %d (sentence, term) pair(s), %d linkage row(s), %.1fs",
                det_path.name, len(detections), n_pairs, len(linkage), duration)
    return {**cell, "status": "ok", "n_reports": len(detections), "n_pairs": n_pairs,
            "n_linkage": len(linkage), "duration_s": round(duration, 1)}


@hydra.main(version_base=None, config_path="../../../configs/experiments/05_phenojury", config_name="normalise")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    stage = str(cfg.stage)
    cohort = str(cfg.dataset)
    out_root = Path(cfg.output_dir) / EXP_ID
    out_root.mkdir(parents=True, exist_ok=True)

    from hpo_extraction.ontology.hpo_tree import HPOTree, ontology_fingerprint
    tree = HPOTree(cfg.hpo_json_path) if cfg.hpo_json_path else HPOTree()
    fingerprint = ontology_fingerprint(cfg.hpo_json_path) if cfg.hpo_json_path else ontology_fingerprint()
    logger.info("ontology fingerprint %s", fingerprint)

    if stage == "surface_matrix":
        ensure_surface_matrix(cfg, tree, fingerprint)
        return

    cells = enumerate_cells(cfg, cohort)
    if stage == "plan":
        logger.info("%d cell(s) on %s — size --array as 0-%d", len(cells), cohort, len(cells) - 1)
        plan_path = out_root / cohort / "cell_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(cells, indent=2), encoding="utf-8")
        for i, cell in enumerate(cells):
            logger.info("  [%3d] %-18s %-22s %s", i, cell["prompt_key"], cell["model_key"],
                        cell["normaliser"])
        # The per-prompt roster is the thing most worth seeing before submitting: a condition with seven
        # members, not eight changes what the PhenoJury protocol's S4 and S6 are allowed to claim.
        for prompt_key in list(cfg.prompt_keys):
            models = sorted({c["model_key"] for c in cells if c["prompt_key"] == prompt_key})
            logger.info("%-18s %d juror(s): %s", prompt_key, len(models), ", ".join(models))
        logger.info("wrote %s", plan_path)
        return

    todo = cells if stage == "all" else [select_cell(cfg, cells)]
    wanted = {c["normaliser"] for c in todo}

    # Build the shared, expensive things once, and only the ones this invocation actually needs, so
    # a dictionary-only run never touches torch and a phenobert-only run never builds an index.
    shared: dict = {}
    if "dictionary" in wanted:
        from hpo_extraction.retrieval.surface_index import build_surface_index
        shared["surface2hpo"], _ = build_surface_index(tree)
        if str(cfg.dictionary_mode) != "exact":
            # Tier 1 is `surface_index` alone. The lemma and search tiers are the only ones that
            # touch OntologyIndex, and building it costs ~3 s that an exact-only run need not pay.
            from hpo_extraction.retrieval.ontology_index import load_index
            shared["index"] = load_index(cfg.hpo_json_path)
    if "sapbert" in wanted:
        shared["surface_matrix"] = ensure_surface_matrix(cfg, tree, fingerprint)
        shared["encode_batch"] = sapbert_encoder(cfg.sapbert_dir)

    rows = [run_cell(cfg, cohort, cell, shared) for cell in todo]

    status_path = out_root / cohort / (
        "cell_status.jsonl" if stage == "all"
        else f"cell_status_{rows[0]['prompt_key']}_{rows[0]['model_key']}_"
             f"{rows[0]['normaliser']}.jsonl")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    logger.info("wrote %s", status_path)

    try:
        import mlflow
        mlflow.set_experiment(cfg.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"{EXP_ID}_{cohort}_{stage}"):
            mlflow.log_params({"cohort": cohort, "stage": stage, "n_cells": len(rows),
                               "ontology_fingerprint": fingerprint,
                               "sapbert_tau": float(cfg.sapbert_tau)})
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
            mlflow.log_artifact(str(status_path))
    except Exception as exc:                                   # pragma: no cover - logging only
        logger.warning("MLflow logging skipped: %s", exc)


if __name__ == "__main__":
    main()
