"""Shared driver for the earlier runs TreePhenoRAG traversal experiments (an earlier exploratory run / 01 / 02).

The three tree experiments differ only in the *prune score* they gate traversal on (calibrated LR
gate / noisyOR / a constant-0.5 single score). Everything else, loading either dataset, building
the whole-ontology contextual retriever, running the live BFS with SLM logit scoring, sweeping
``tau_prune``, and writing the fully-instrumented outputs Step-2 metrics consume, is identical and
lives here. Each experiment's ``run.py`` is a thin ``@hydra.main`` that supplies a ``build_scores``
callback and calls :func:`execute`.

Key implementation choice, **the tau sweep memoises node evaluation.** A node's retrieval + SLM
result is independent of ``tau_prune``. Only the expand/accept *decisions* depend on it, and the
visited set is monotone (a smaller tau expands a superset). So every unique node is scored by the
SLM once per report and the sweep re-runs the frontier logic over the cache. The faithful
per-tau inference cost is therefore the **SLM calls over that tau's visited set**
(``TraversalResult.n_slm_calls``), which the thesis names as the primary, hardware-independent cost
unit. Wall-clock is recorded for the whole report (all unique nodes) and is not per-tau under
memoisation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hpo_extraction.utils.resume import Checkpoint, GracefulStop, flush_all, open_jsonl
from hpo_extraction.retrieval.descendant_closure import UnionScorer, top_sentences
from hpo_extraction.data.loading import LazyContextDict, load_txt
from hpo_extraction.retrieval.encoding import encode_dict
from hpo_extraction.treephenorag.node_metadata import node_metadata
from hpo_extraction.treephenorag.verifier_prompt import BaselinePrompter
from hpo_extraction.retrieval.term_information_index import build_r3_index
from hpo_extraction.data.segmentation import load_stanza, segment_dict, split_sents
from hpo_extraction.treephenorag.traversal import NodeEval, build_children_map, traverse
from hpo_extraction.evaluation.retrieval_analysis import descendants_with_hops
from hpo_extraction.models.llama import LlamaLogitsLLM, load_llama
from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree
from hpo_extraction.utils.mlflow_guard import mlflow_run

SYSTEM_PROMPT = (
    "You are a medical expert deciding whether a patient has a certain symptom. "
    "Answer only with 'Yes' or 'No'."
)


def _setup_logging(output_dir: str, suffix: str = "") -> logging.Logger:
    """File + stream logging for one worker. ``suffix`` keeps the shards of a SLURM array from
    interleaving their lines into a single ``run.log`` (which made it impossible to tell which
    task a message came from, and which task had stopped)."""
    log_path = os.path.join(output_dir, f"run{suffix}.log")
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
    return logging.getLogger("exp13")


def _ontology_context_sentences(hpo_id: str, hpo_tree: HPOTree) -> list[str]:
    """Fallback context for an HPO with no ``.txt`` synthetic-sentence: its own name/synonyms/definition.

    Matches an earlier exploratory run / an earlier exploratory run so a node missing from the curated set still has a column in the
    ``own_max`` matrix rather than silently scoring 0 on its own presence.
    """
    node = HPO_class(hpo_tree.data[hpo_id])
    parts = list(node.name) + list(node.synonym) + list(node.definition)
    return [p for p in parts if p and p.strip()] or [hpo_id]


def _load_report_allowlist(path: str) -> list[str]:
    """Read a report-id allowlist - the first column of a CSV, or one id per line.

    A leading ``id`` / ``patient_id`` / ``report_id`` header cell is skipped, so the curated
    ground truth's own two-column ``hcy_ground_truth_curated.csv`` doubles as the cohort definition.
    That is the point: ``input_dir`` holds 135 HCY ``.txt`` files while the scored cohort is 118,
    and at ~8 GPU-hours per report the 17 extra ones are work no table can ever quote. Reading the
    cohort from the ground truth itself means the run and the scorer cannot disagree about what it is.
    """
    ids: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            cell = row[0].strip()
            if not cell:
                continue
            if not ids and cell.lower() in {"id", "patient_id", "report_id"}:
                continue
            ids.append(cell)
    return ids


def _load_dataset(cfg, logger):
    """Return ``(reports, gt_dict)`` for ``cfg.dataset``, reports ``{id: text}``, gt ``{id: [hpo]}``.

    Ground truth is the *annotated* terms only (no ancestor closure), as both loaders return it.
    """
    if cfg.dataset == "hcy":
        from hpo_extraction.evaluation.datasets.hcy import HCYDataset

        reports = load_txt(cfg.input_dir)
        gt_dict = HCYDataset(cfg.ground_truth_path, cfg.target_symptoms_path).load_ground_truth()
    elif cfg.dataset == "gsc":
        from hpo_extraction.evaluation.datasets.gsc import GSCDataset, load_gsc_reports

        reports = load_gsc_reports(cfg.gsc_dir)
        gt_dict = GSCDataset(cfg.gsc_dir).load_ground_truth()
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r} (expected 'hcy' or 'gsc')")
    logger.info("Dataset %s | %d reports | %d with ground truth",
                cfg.dataset, len(reports), sum(1 for v in gt_dict.values() if v))
    return reports, {k: list(v) for k, v in gt_dict.items()}


def _build_universe(cfg, hpo_tree, sent_model, logger):
    """Build the whole-ontology contextual retriever and the ``(children, roots)`` map.

    Universe = phenotypic-abnormality nodes (optionally restricted to one subtree for pilots and
    for the ontology-size scaling measurement). Returns
    ``(scorer, ctx_dict, children_map, roots, subtree)`` where ``subtree`` is the set of nodes the
    traversal may reach.

    ``cfg.retrieval_index`` chooses what a node's column of the ``own_max`` matrix is built from.
    It is the retrieval-curve analysis's representation axis, and the union reduction on top of it is unchanged, so the
    two settings are that experiment's ``R1u`` and ``R3u``, the same intervention measured on two
    indices:

    ``exemplar`` (default, every shipped run)
        the term's curated ``.txt`` synthetic-sentence sentences where they exist, ``_ontology_context_
        sentences`` otherwise. ~36 vectors per term, ~652k in total.
    ``ontology_r3``
        one vector per term from ``hpo_extraction.retrieval.term_information_index.r3_text``, ``label + definition + synonyms`` as a
        **single** string. No generator model anywhere in the retrieval path, and the context
        matrix shrinks from ~652k rows to one per node.

    The distinction against the ``exemplar`` fallback is real and easy to miss: that fallback reads
    the same ontology fields but embeds them as a *list* of strings, so a term gets several vectors
    and the max pools over them. R3 is one string, one vector, by definition.
    """
    children_map, roots = build_children_map(hpo_tree)
    retrieval_index = str(cfg.get("retrieval_index") or "exemplar")
    if retrieval_index not in ("exemplar", "ontology_r3"):
        raise ValueError(
            f"retrieval_index={retrieval_index!r} (expected 'exemplar' or 'ontology_r3')"
        )
    ctx_dict = (LazyContextDict(cfg.context_dir, sent_model)
                if retrieval_index == "exemplar" else {})

    if cfg.get("restrict_to_subtree"):
        sub_root = cfg.restrict_to_subtree
        subtree = set(descendants_with_hops(hpo_tree, sub_root))
        roots = [sub_root]
        logger.info("Restricted to subtree %s | %d nodes", sub_root, len(subtree))
    else:
        subtree = set(hpo_tree.phenotypic_abnormality) - {hpo_tree.root}

    universe = sorted(h for h in subtree if h in hpo_tree.data)
    if retrieval_index == "ontology_r3":
        # One GPU pass over ~18k short strings, seconds, and nothing is read from disk: the
        # synthetic-sentence corpus does not enter this run at any point.
        vectors, prov = build_r3_index(
            hpo_tree, universe, lambda texts: sent_model.encode(texts, show_progress_bar=False)
        )
        ctx_dict = {h: vectors[i][None, :] for i, h in enumerate(universe)}
        logger.info("Context universe | R3: %d nodes, one vector each | %d without a definition, "
                    "%d label-only, median %d chars | template %r",
                    prov["n_terms"], prov["n_without_definition"], prov["n_label_only"],
                    prov["median_chars"], prov["template"])
    else:
        n_txt = 0
        for h in tqdm(universe, desc="context embeddings", unit="hpo"):
            if ctx_dict.is_file_backed(h):
                _ = ctx_dict[h]
                n_txt += 1
            elif h not in ctx_dict:
                ctx_dict.set_encoded(h, sent_model.encode(
                    _ontology_context_sentences(h, hpo_tree)))
        logger.info("Context universe | %d nodes (%d curated, %d ontology-derived)",
                    len(universe), n_txt, len(universe) - n_txt)

    scorer = UnionScorer(ctx_dict, {h: [h] for h in universe})
    scorer.prepare()
    logger.info("Context matrix ready | %d context sentences, %.2f GB",
                sum(scorer.n_context_sentences.values()),
                sum(scorer.n_context_sentences.values()) * 768 * 4 / 1024 ** 3)
    return scorer, ctx_dict, children_map, roots, subtree


def execute(cfg, exp_id: str, build_scores) -> None:
    """Run one tree-traversal experiment.

    Args:
        cfg: Hydra config (see any exp13_0x/config.yaml for the keys).
        exp_id: the experiment folder name → output subfolder.
        build_scores: ``cfg -> (prune_score_fn, accept_score_fn, variant)`` where
            ``prune_score_fn(margins, node_meta) -> float`` and
            ``accept_score_fn(margins) -> float``.
    """
    prune_score_fn, accept_score_fn, variant = build_scores(cfg)

    run_output_dir = os.path.join(cfg.output_dir, exp_id, cfg.dataset)
    os.makedirs(run_output_dir, exist_ok=True)

    # Shard identity first: it names the log file, the outputs and the model, so every worker
    # of a SLURM array is self-contained inside the shared run directory.
    n_report_shards = int(cfg.get("n_report_shards") or 1)
    report_shard = int(cfg.get("report_shard") or 0)
    sfx = f"_s{report_shard}of{n_report_shards}" if n_report_shards > 1 else ""

    logger = _setup_logging(run_output_dir, suffix=sfx)
    logger.info("Starting %s | dataset=%s variant=%s retrieval_ctx=%s retrieval_index=%s | out=%s",
                exp_id, cfg.dataset, variant, cfg.retrieval_ctx,
                cfg.get("retrieval_index") or "exemplar", run_output_dir)

    S = int(cfg.top_n)
    tau_prune_sweep = [float(t) for t in cfg.tau_prune_sweep]
    tau_accept_sweep = _accept_sweep(cfg)
    batch_calls = bool(cfg.get("batch_calls", False))
    max_nodes = int(cfg.max_nodes) if cfg.get("max_nodes") else None

    hpo_tree = HPOTree()
    hpo_tree.buildHPOTree()
    reports, gt_dict = _load_dataset(cfg, logger)

    sent_model = SentenceTransformer(cfg.sent_transformer_dir)
    scorer, ctx_dict, children_map, roots, subtree = _build_universe(
        cfg, hpo_tree, sent_model, logger
    )

    # Reports → sentences → embeddings (same pipeline as the earlier runs/exp12).
    nlp = load_stanza(stanza_dir=cfg.stanza_dir, mode="TOKENIZER")
    sents_by_report = split_sents(segment_dict(reports, nlp))
    enc_by_report = encode_dict(sents_by_report, sent_model)
    report_ids = sorted(enc_by_report)

    # Cohort allowlist (optional): restrict to the reports that are actually scored downstream.
    # Applied before max_patients and before sharding, so the pilot switch and the round-robin
    # shard slices are taken over the cohort, not over whatever happens to sit in input_dir.
    if cfg.get("report_ids_path"):
        allow = _load_report_allowlist(cfg.report_ids_path)
        keep = set(allow)
        report_ids = [r for r in report_ids if r in keep]
        missing = sorted(keep - set(enc_by_report))
        logger.info("Cohort allowlist %s | %d id(s) listed -> %d report(s)",
                    cfg.report_ids_path, len(allow), len(report_ids))
        if missing:
            logger.warning("Allowlisted id(s) absent from input_dir (%d): %s",
                           len(missing), ", ".join(missing[:10]))
        if not report_ids:
            raise ValueError(
                f"report_ids_path {cfg.report_ids_path!r} matched no report in input_dir"
            )

    if cfg.get("max_patients"):
        report_ids = report_ids[: int(cfg.max_patients)]
        logger.warning("PILOT: restricted to %d report(s)", len(report_ids))

    # Optional report sharding for a SLURM array: each shard processes a round-robin slice of the
    # reports and writes shard-suffixed outputs, so N shards run in parallel and are unioned later
    # by evaluation/merge_shards.py. (The suffix itself was computed at the top, it names the log.)
    if n_report_shards > 1:
        report_ids = report_ids[report_shard::n_report_shards]
        logger.info("Report shard %d/%d → %d reports", report_shard, n_report_shards, len(report_ids))

    # ── Resume: skip reports a previous session already wrote in full ──
    ckpt = Checkpoint(run_output_dir, f"{variant}{sfx}", resume=bool(cfg.get("resume", True)))
    stop = GracefulStop(max_runtime_s=cfg.get("max_runtime_s"))
    cohort_size = len(report_ids)
    todo = ckpt.pending(report_ids)
    n_skipped = cohort_size - len(todo)
    if ckpt.resumed:
        logger.warning("RESUMING | %d/%d reports already done → %d to go",
                       n_skipped, cohort_size, len(todo))
    report_ids = todo
    if not report_ids:
        # Nothing left for this shard, don't pay for the LLM load just to write a header.
        logger.info("Done | nothing to do: all %d report(s) of this shard are already checkpointed",
                    cohort_size)
        ckpt.close()
        return

    tokenizer, model = load_llama(cfg.llama_dir_base, load_in_4bit=bool(cfg.get("load_in_4bit")))
    model.eval()
    llm = LlamaLogitsLLM(model, tokenizer)
    prompter = BaselinePrompter()
    logger.info("Yes id=%d No id=%d", llm._yes_id, llm._no_id)

    # Static per-node metadata (gate features) and descendant columns, cached across reports.
    meta_cache: dict[str, dict] = {}
    desc_cols_cache: dict[str, np.ndarray] = {}

    # Per-call instrumentation. Off by default so every shipped run's artifacts keep their
    # schema. The synthetic-sentence score store turns it on, and it is the only run that has ever recorded a clock.
    record_timing = bool(cfg.get("record_timing", False))

    def node_meta(h: str) -> dict:
        if h not in meta_cache:
            meta_cache[h] = node_metadata(h, hpo_tree)
        return meta_cache[h]

    def desc_cols(h: str) -> np.ndarray:
        if h not in desc_cols_cache:
            desc_cols_cache[h] = scorer.columns_for(descendants_with_hops(hpo_tree, h))
        return desc_cols_cache[h]

    def run_slm(prompts: list[str]) -> list[dict]:
        """Score a node's prompts, optionally recording what each call cost.

        ``record_timing`` attaches a per-call wall clock and prompt-token count. The clock is
        divided evenly across a batch, because a batched forward pass has no per-item time to
        report, the honest per-call quantity is the batch's amortised cost, and reporting it that
        way is what makes the median/mean/p95 triple mean something. Prompt tokens are counted per
        prompt and are exact.
        """
        if not prompts:
            return []
        t0 = time.time() if record_timing else None
        if batch_calls:
            out = llm.generate_batch(prompts, SYSTEM_PROMPT)
        else:
            out = []
            for p in prompts:
                v = llm.generate(p, SYSTEM_PROMPT)
                out.append({**llm.last_detail, "margin": llm.last_margin, "verdict": v})
        if record_timing:
            per_call = (time.time() - t0) / max(1, len(out))
            for rec, prompt in zip(out, prompts):
                rec["wall_clock_s"] = per_call
                rec["n_prompt_tokens"] = len(tokenizer.encode(prompt))
        return out

    # ── Output files: calls + node metadata are tau-independent. The rest is per-operating-point.
    # A point is a (tau_prune, tau_accept) pair, one BFS per tau_prune serves the whole accept
    # grid, so |tau_prune| x |tau_accept| output sets cost |tau_prune| traversals.
    sweeping_accept = len(tau_accept_sweep) > 1
    tau_points = [(p, a) for p in tau_prune_sweep for a in tau_accept_sweep]
    tau_dirs = {}
    for point in tau_points:
        d = os.path.join(run_output_dir, _tau_dir_name(*point, sweeping_accept))
        os.makedirs(d, exist_ok=True)
        tau_dirs[point] = d

    with mlflow_run(f"{exp_id}_{cfg.dataset}{sfx}",
                    experiment=cfg.mlflow_experiment_name, logger=logger):
        mlflow.log_params({
            "experiment": exp_id, "dataset": cfg.dataset, "variant": variant,
            "retrieval_ctx": cfg.retrieval_ctx, "top_n": S,
            "retrieval_index": cfg.get("retrieval_index") or "exemplar",
            "tau_prune_sweep": ",".join(f"{t:g}" for t in tau_prune_sweep),
            "tau_accept_sweep": ",".join(f"{t:g}" for t in tau_accept_sweep),
            "llama_dir_base": cfg.llama_dir_base, "context_dir": cfg.context_dir,
            "restrict_to_subtree": cfg.get("restrict_to_subtree") or "none",
            "batch_calls": batch_calls, "load_in_4bit": bool(cfg.get("load_in_4bit")),
            "n_reports": len(report_ids), "universe_size": len(subtree),
            "report_shard": report_shard, "n_report_shards": n_report_shards,
            "resumed": ckpt.resumed, "n_reports_skipped": n_skipped,
            "cohort_size": cohort_size,
        })

        # Append when resuming, truncate on a fresh run, the model decides, so the outputs
        # and the progress log can never disagree about what is already on disk.
        mode = ckpt.file_mode
        calls_f = open_jsonl(os.path.join(run_output_dir, f"{variant}{sfx}_calls.jsonl"), mode)
        meta_f = open_jsonl(os.path.join(run_output_dir, f"{variant}{sfx}_node_metadata.jsonl"), mode)
        tau_files = {
            point: {
                "nodes": open_jsonl(os.path.join(d, f"{variant}{sfx}_nodes.jsonl"), mode),
                "pred": open_jsonl(os.path.join(d, f"{variant}{sfx}_predictions.jsonl"), mode),
                "trav": open_jsonl(os.path.join(d, f"{variant}{sfx}_traversal.jsonl"), mode),
                "time": open_jsonl(os.path.join(d, f"{variant}{sfx}_timing.jsonl"), mode),
            }
            for point, d in tau_dirs.items()
        }
        all_handles = [calls_f, meta_f] + [fh for fs in tau_files.values() for fh in fs.values()]

        # The node-metadata sidecar is written incrementally (it used to be dumped once at the end,
        # so a wall-clock kill lost it). On resume, seed the seen-set from what is already there.
        evaluated_nodes: set[str] = _existing_metadata_ids(
            os.path.join(run_output_dir, f"{variant}{sfx}_node_metadata.jsonl")
        ) if ckpt.resumed else set()
        if evaluated_nodes:
            logger.info("Node-metadata sidecar already holds %d node(s)", len(evaluated_nodes))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_run = time.time()
        n_done_session = 0
        pbar = tqdm(report_ids, desc=f"{variant}/{cfg.dataset}", unit="report")
        for rid in pbar:
            t_report = time.time()
            sents = sents_by_report[rid]
            scorer.score_patient(np.asarray(enc_by_report[rid]))
            gold = set(gt_dict.get(rid, []))

            node_cache: dict[str, NodeEval] = {}
            node_calls: dict[str, list[dict]] = {}

            def _real_eval(h: str) -> NodeEval:
                meta = node_meta(h)
                if cfg.retrieval_ctx == "union_only":
                    sims = scorer.union_over_columns(desc_cols(h))
                    info = top_sentences(sims, sents, top_n=S)
                    det = run_slm(prompter.build_prompts(info["top_sents"], h))
                    margins = [d["margin"] for d in det]
                    calls = _call_records(rid, h, "union", info, det)
                    return NodeEval(prune_score_fn(margins, meta), accept_score_fn(margins),
                                    len(det), calls)
                # union_prune_own_accept: separate retrieval for each score
                sims_u = scorer.union_over_columns(desc_cols(h))
                info_u = top_sentences(sims_u, sents, top_n=S)
                det_u = run_slm(prompter.build_prompts(info_u["top_sents"], h))
                sims_o = scorer.own_sims(h)
                info_o = top_sentences(sims_o, sents, top_n=S)
                det_o = run_slm(prompter.build_prompts(info_o["top_sents"], h))
                calls = (_call_records(rid, h, "union", info_u, det_u)
                         + _call_records(rid, h, "own", info_o, det_o))
                return NodeEval(
                    prune_score_fn([d["margin"] for d in det_u], meta),
                    accept_score_fn([d["margin"] for d in det_o]),
                    len(det_u) + len(det_o), calls,
                )

            def evaluate_node(h: str) -> NodeEval:
                if h not in node_cache:
                    full = _real_eval(h)
                    node_calls[h] = full.calls
                    node_cache[h] = NodeEval(full.prune_score, full.accept_score,
                                             full.n_slm_calls, [])
                return node_cache[h]

            for t in tau_prune_sweep:
                # One BFS per tau_prune. The accept grid is then exact post-processing over the
                # same visits, no extra SLM calls, so a 6x5 grid costs 6 traversals, not 30.
                res = traverse(children_map, roots, evaluate_node, t, tau_accept_sweep[0],
                               max_nodes=max_nodes)
                for a in tau_accept_sweep:
                    _write_tau(tau_files[(t, a)], variant, rid, res.at_accept(a), gold,
                               wall_clock_s=time.time() - t_report if record_timing else None)

            # calls + evaluated-node bookkeeping: written once from the shared cache
            for h, recs in node_calls.items():
                for rec in recs:
                    calls_f.write(json.dumps(rec) + "\n")
            for h in sorted(set(node_cache) - evaluated_nodes):
                meta_f.write(json.dumps(node_meta(h)) + "\n")
            evaluated_nodes.update(node_cache)

            # Durability point: everything for this report is on disk before it is marked done.
            flush_all(all_handles)
            ckpt.mark(rid, n_nodes=len(node_cache))
            n_done_session += 1
            pbar.set_postfix(nodes=len(evaluated_nodes))

            if stop.should_stop():
                logger.warning(
                    "STOPPED_EARLY | %s | %d/%d reports this session (%d/%d in the cohort) | "
                    "resume by resubmitting the same command",
                    stop.reason, n_done_session, len(report_ids), ckpt.n_done, cohort_size,
                )
                break
        pbar.close()

        calls_f.close()
        meta_f.close()
        for fs in tau_files.values():
            for fh in fs.values():
                fh.close()
        ckpt.close()

        duration = time.time() - t_run
        peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        # Timings are per *session*: a resumed run must not divide its wall clock by reports a
        # previous session paid for.
        mlflow.log_metrics({
            "n_reports": n_done_session,
            "n_reports_cohort_done": ckpt.n_done,
            "duration_seconds": duration,
            "mean_time_per_report_s": duration / n_done_session if n_done_session else 0.0,
            "peak_gpu_mem_bytes": peak_mem,
            "n_unique_nodes_evaluated": len(evaluated_nodes),
        })
        for p in [os.path.join(run_output_dir, f"{variant}{sfx}_calls.jsonl"),
                  os.path.join(run_output_dir, f"{variant}{sfx}_node_metadata.jsonl")]:
            mlflow.log_artifact(p)
        for d in tau_dirs.values():
            mlflow.log_artifacts(d, artifact_path=os.path.basename(d))

        # "Done |" means the whole shard is finished, exp13_status.sh keys its DONE verdict on it,
        # so a session that stopped early must not claim it.
        if ckpt.n_done >= cohort_size:
            logger.info("Done | %d reports in %.1fs | %d unique nodes | peak GPU %.2f GB",
                        n_done_session, duration, len(evaluated_nodes), peak_mem / 1024 ** 3)
        else:
            logger.warning(
                "INCOMPLETE | %d reports in %.1fs this session | %d/%d of the shard done | "
                "%d unique nodes | peak GPU %.2f GB | resubmit to resume",
                n_done_session, duration, ckpt.n_done, cohort_size,
                len(evaluated_nodes), peak_mem / 1024 ** 3,
            )


def _accept_sweep(cfg) -> list[float]:
    """The accept-threshold grid, from ``tau_accept_sweep`` or the scalar ``tau_accept``.

    earlier fixed ``tau_accept`` at 0.5 and swept only ``tau_prune``. Its offline re-run then found
    0.9 optimal everywhere and worth **+0.137 micro-F1** on GSC+, so the main understated the
    traversal ~2.5×. The sweep is therefore first-class here, and because accept decisions do not
    change which nodes are visited (see :meth:`TraversalResult.at_accept`), it is free.

    The scalar is kept working so experiments that were not re-run (an earlier exploratory run) keep their exact
    previous behaviour and output layout.
    """
    sweep = cfg.get("tau_accept_sweep")
    if sweep:
        return sorted({float(t) for t in sweep})
    return [float(cfg.get("tau_accept", 0.5))]


def _tau_dir_name(tau_prune: float, tau_accept: float, sweeping_accept: bool) -> str:
    """Output subdirectory for one configuration.

    Stays ``tau_{prune}`` when the accept grid has a single value, so dumps written before the
    accept sweep existed, and experiments still configured with a scalar ``tau_accept``, keep the
    exact directory names ``result_tables``'s discovery already globs. Only a real 2-D sweep introduces
    the ``_acc_`` suffix.
    """
    if not sweeping_accept:
        return f"tau_{tau_prune:g}"
    return f"tau_{tau_prune:g}_acc_{tau_accept:g}"


def _existing_metadata_ids(path: str) -> set[str]:
    """HPO ids already in the node-metadata sidecar, so a resumed session appends only new nodes."""
    seen: set[str] = set()
    if not os.path.isfile(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail from a hard kill
            hid = rec.get("hpo_id") or rec.get("id")
            if hid:
                seen.add(str(hid))
    return seen


def _captured_mass(logit_yes: float, logit_no: float, logsumexp_all: float) -> float:
    """The share of next-token probability the Yes/No pair holds, thesis \u00a74.1's ``mu``.

    ``mu = (exp(logit_yes) + exp(logit_no)) / exp(logsumexp_all)``, computed in log space so a
    large logit cannot overflow. A low value is the format-failure signal: the verifier would
    rather have emitted some third token entirely, and its margin is then a comparison between two
    tokens it did not want.

    **This is over the singleton Yes/No ids the verifier actually reads**, not over the
    case-and-space token-variant sets the thesis defines ``mu`` on. The two coincide whenever the
    model puts negligible mass on the variants and diverge otherwise, so anything reporting this
    number has to say which one it is.
    """
    return float(np.exp(np.logaddexp(logit_yes, logit_no) - logsumexp_all))


def _call_records(rid, hpo_id, ctx_type, info, details) -> list[dict]:
    """One SLM record per retrieved sentence, the calibration/aggregation/retrieval substrate."""
    recs = []
    for rank, (det, idx, cos) in enumerate(
        zip(details, info["top_indices"], info["top_scores"]), start=1
    ):
        logit_yes = float(det["logit_yes"])
        logit_no = float(det["logit_no"])
        logsumexp_all = float(det["logsumexp_all"])
        recs.append({
            "report_id": rid, "hpo_id": hpo_id, "ctx_type": ctx_type, "rank": rank,
            "sent_index": int(idx), "cosine_sim": round(float(cos), 6),
            "logit_yes": round(logit_yes, 6),
            "logit_no": round(logit_no, 6),
            "logsumexp_all": round(logsumexp_all, 6),
            "top1_token_id": int(det["top1_token_id"]),
            # The winning token's own logit, beside the Yes/No pair. Without it a low captured mass
            # says only "something else won" and cannot say by how much.
            **({"top1_logit": round(float(det["top1_logit"]), 6)}
               if "top1_logit" in det else {}),
            "captured_mass": round(_captured_mass(logit_yes, logit_no, logsumexp_all), 6),
            "margin": round(float(det["margin"]), 6), "verdict": det["verdict"],
        })
        # Present only when the run was instrumented (the synthetic-sentence score store). Absent everywhere else, so a
        # reader must treat them as optional, not assume a schema.
        if "wall_clock_s" in det:
            recs[-1]["wall_clock_s"] = round(float(det["wall_clock_s"]), 6)
        if "n_prompt_tokens" in det:
            recs[-1]["n_prompt_tokens"] = int(det["n_prompt_tokens"])
    return recs


def _write_tau(files, variant, rid, res, gold, wall_clock_s: float | None = None) -> None:
    """Write the per-tau node / prediction / traversal / timing records for one report."""
    for h, v in res.visits.items():
        files["nodes"].write(json.dumps({
            "report_id": rid, "hpo_id": h, "depth": v.depth,
            "prune_score": round(v.prune_score, 6), "accept_score": round(v.accept_score, 6),
            "expanded": v.expanded, "accepted": v.accepted, "is_gold": int(h in gold),
        }) + "\n")

    predicted = sorted(res.accepted)
    for h in predicted:
        files["pred"].write(json.dumps({
            "report_id": rid, "hpo_id": h, "prediction": 1, "ground_truth": int(h in gold),
        }) + "\n")
    files["pred"].write(json.dumps({
        "report_id": rid, "summary": True,
        "predicted_set": predicted, "gold_set": sorted(gold),
    }) + "\n")

    files["trav"].write(json.dumps({
        "report_id": rid,
        "n_unique_nodes_visited": res.n_unique_nodes_visited,
        "total_frontier_insertions": res.total_frontier_insertions,
        "n_slm_calls": res.n_slm_calls,
        "n_accepted": len(res.accepted),
        "depth_table": res.depth_table(),
    }) + "\n")
    # wall_clock_s is the per-report elapsed time, measured one report at a time with the model
    # already loaded. Before the synthetic-sentence score store no run recorded it at all, and the cost tables reconstructed
    # it from model timestamp differences, which include queueing, I/O and the next report's
    # setup. This is the measured quantity the median/mean/p95 triple is defined on.
    files["time"].write(json.dumps({
        "report_id": rid, "n_slm_calls": res.n_slm_calls,
        "n_nodes_visited": res.n_unique_nodes_visited,
        **({} if wall_clock_s is None else {"wall_clock_s": round(float(wall_clock_s), 4)}),
    }) + "\n")
