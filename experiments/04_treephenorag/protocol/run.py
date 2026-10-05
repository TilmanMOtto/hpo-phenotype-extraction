"""The TreePhenoRAG protocol, TreePhenoRAG's configuration chosen by a rule, not by looking at the answer.


Analysis only: no model, no GPU. Everything re-runs over the cached score dumps of the synthetic-sentence score store
(synthetic-sentence retrieval) and the term-information score store (ontology-R3 retrieval) through :mod:`hpo_extraction.treephenorag.stored_scores`, whose
fidelity against the run it re-runs is fixed by ``scripts/verify_replay_fidelity.py`` and whose
bottleneck sweep is fixed against the reference frontier walk in ``tests/unit/test_tree_replay.py``.

Stages
------
``ingest``       one-time ``_calls.jsonl`` -> columnar ``.npz``. Skip once it has run.
``containment``  the hard gate: does the cache host the grid at all.
``select``       E4.3, CRC for tau_prune, inner folds for the poolings, index and tau_accept.
``ablations``    E4.4, the V0-V3 ladder, every rung tuned by that same protocol.
``calibration``  E4.6, fit h, reliability, ECE/SmoothECE, and the GSC+ transfer.
``diagnostics``  E4.7, recall decomposition, FP taxonomy, blocking depth.

Every stage writes its own ``tables/*.csv``; ``README.md`` is the guide to what each column means
and to what was and was not tested.

The expensive axis is expansion, and even that is paid once: for a fixed ``(index, pooling, S)`` the
whole ``tau_prune`` axis collapses to one widest-path value per node, so a threshold is a
comparison rather than a traversal. Acceptance was always free, ``traverse`` expands on the prune
score alone.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stages  # noqa: E402
from hpo_extraction.treephenorag.stored_scores import ingest_score_cache, load_cache_npz, read_cache_provenance  # noqa: E402
from hpo_extraction.treephenorag.selection import OfflineSpace  # noqa: E402
from hpo_extraction.treephenorag.traversal import build_children_map  # noqa: E402
from hpo_extraction.evaluation.prediction_sets import write_prediction_sets  # noqa: E402
from hpo_extraction.evaluation.stats import ReportResampler, read_folds  # noqa: E402
from hpo_extraction.evaluation.metrics import OntologyView, normalise_pair  # noqa: E402

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "treephenorag_protocol"
logger = logging.getLogger("exp13_22")


# ── IO ───────────────────────────────────────────────────────────────────────

def load_gold(path: Path) -> dict[str, set[str]]:
    """``{report: set of HPO identifiers}`` from a two-column ground-truth CSV (``patient_id``, ``hpo_codes``)."""
    gold: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            codes = [c.strip() for c in (rec.get("hpo_codes") or "").split(";") if c.strip()]
            gold[rec["patient_id"]] = set(codes)
    return gold


def load_evidence(path: Path | None) -> dict | None:
    """``{report: {hpo: [segment indices]}}`` from the curated ground truth's annotation table.

    Without this the recall decomposition cannot separate a retrieval miss from a judgement miss,
    and it reports ``evidence_available=False``, not guessing.
    """
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        logger.warning("evidence_path %s does not exist — the retrieval bucket will not be "
                       "measurable and the run will say so", path)
        return None
    out: dict[str, dict[str, list[int]]] = {}
    with path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            rid = rec.get("patient_id") or rec.get("report_id")
            hpo = rec.get("hpo_id") or rec.get("hpo_code")
            seg = rec.get("segment_idx") or rec.get("sent_index")
            if not (rid and hpo) or seg in (None, ""):
                continue
            out.setdefault(rid, {}).setdefault(hpo, []).append(int(seg))
    logger.info("evidence: %d report(s), %d annotated term(s)",
                len(out), sum(len(v) for v in out.values()))
    return out


#: Filename namespace for the GSC+ transfer condition's ingested caches. HCY's keep the bare name so an
#: existing 585 MB `cache_ontology_r3.npz` is not orphaned by this change and re-ingested from 6 GB
#: of JSON. Only the cohort that previously had no name of its own gets one.
GSC_NPZ_PREFIX = "gsc_"


def default_npz(out_dir: Path, index: str, prefix: str = "") -> Path:
    """Where an index's ingested cache lives.

    ``prefix`` namespaces it **by cohort**, and it is not optional decoration. The GSC+ transfer
    condition configures the same retrieval index (``ontology_r3``) as HCY, so without a prefix both
    resolve to one ``cache_ontology_r3.npz``: ``ingest_all`` sees the HCY file already there and
    skips the GSC+ ingest, ``load_spaces`` then filters the HCY cache to 114 GSC+ document ids
    that are not in it, and the transfer is scored over an empty cohort. That produces
    ``transferred = 0.0000`` -- a number, in a results table, describing nothing.
    """
    return out_dir / f"cache_{prefix}{index}.npz"


def ingest_all(cfg, out_dir: Path, ctx_type: str, key: str = "caches",
               prefix: str = "") -> None:
    """Convert every configured ``_calls.jsonl`` once. Idempotent: an existing ``.npz`` is kept.

    ``ctx_type`` is a **parameter**, not read off ``cfg``. The GSC+ condition passes ``cfg.gsc``, which is
    a sub-config with no ``ctx_type`` of its own, so reading it here raised
    ``ConfigAttributeError`` the first time a GSC+ ingest actually had work to do. It never had
    before, because the un-namespaced ``.npz`` always looked like it already existed.
    """
    block = cfg.get(key) or {}
    for index, entry in block.items():
        calls = (entry or {}).get("calls_path")
        if not calls:
            continue
        target = Path((entry or {}).get("npz") or default_npz(out_dir, index, prefix))
        if target.is_file():
            logger.info("%s: %s already exists (%s) — skipping ingest", index, target.name,
                        read_cache_provenance(target).get("n_reports"))
            continue
        logger.info("%s: ingesting %s -> %s", index, calls, target)
        provenance = ingest_score_cache(calls, target, ctx_type=ctx_type)
        logger.info("%s: %d reports, %d rows, S_max=%d, clock=%s tokens=%s", index,
                    provenance["n_reports"], provenance["n_rows"], provenance["s_max"],
                    provenance["has_wall_clock"], provenance["has_prompt_tokens"])


def load_spaces(cfg, out_dir: Path, children_map, roots, report_ids, key="caches", prefix=""):
    """Build a :class:`ReplaySpace` over every index whose ``.npz`` exists.

    Aborts if a cache covers **none** of ``report_ids``. That is always a configuration error --
    the wrong cohort's cache, or an ingest that never ran -- and every downstream metric over an
    empty cohort is 0.0, not an error, which is the one failure mode this file exists to
    prevent. A *partial* overlap is not caught here: it is legitimate (a cohort whose cache is
    still being built) and the row-count log line below is what surfaces it.
    """
    block = cfg.get(key) or {}
    caches = {}
    for index, entry in block.items():
        target = Path((entry or {}).get("npz") or default_npz(out_dir, index, prefix))
        if not target.is_file():
            if (entry or {}).get("calls_path"):
                logger.warning("%s: %s missing — run stages=[ingest] first", index, target)
            continue
        got = load_cache_npz(target, report_ids=report_ids)
        if not got:
            raise SystemExit(
                f"{index}: {target} holds none of the {len(list(report_ids))} requested "
                f"report(s). It carries "
                f"{read_cache_provenance(target).get('n_reports')} report(s) from another cohort, "
                f"so this is a cache/cohort mismatch and every metric below it would be 0.0. "
                f"Check `npz:` for this index, or delete {target.name} and re-run stages=[ingest].")
        logger.info("%s: %d report(s), %d nodes/report, S_max=%d", index, len(got),
                    int(np.mean([c.n_nodes for c in got.values()])) if got else 0,
                    max((c.s_max for c in got.values()), default=0))
        caches[index] = got
    if not caches:
        return None
    return OfflineSpace(children_map, roots, caches)


def write_csv(path: Path, rows) -> None:
    """Write *rows* (dicts) to *path*. The columns are the union of the keys, in order of first appearance."""
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d rows)", path.name, len(rows))


@hydra.main(version_base=None, config_path="../../../configs/experiments/04_treephenorag", config_name="protocol")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
    out_dir = Path(cfg.output_dir) / EXP_ID
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stage_list = list(cfg.stages)
    logger.info("stages=%s", stage_list)

    if "ingest" in stage_list:
        ingest_all(cfg, out_dir, cfg.ctx_type)
        if cfg.get("gsc") and (cfg.gsc.get("caches") or {}):
            # GSC_NPZ_PREFIX, not the bare index name: the transfer condition configures the same
            # retrieval index as HCY, and one shared filename silently reuses HCY's cache.
            # ctx_type comes from the TOP-LEVEL config -- cfg.gsc has none of its own.
            ingest_all(cfg.gsc, out_dir, cfg.ctx_type, key="caches", prefix=GSC_NPZ_PREFIX)
        if stage_list == ["ingest"]:
            return

    from hpo_extraction.ontology.hpo_tree import HPOTree
    tree = HPOTree()
    view = OntologyView(tree)
    children_map, roots = build_children_map(tree)

    # The producer can restrict the universe to one layer-1 subtree (the synthetic-sentence score store's
    # `restrict_to_subtree`, used for pilots and the ontology-size scaling measurement). The re-run
    # has to mirror that, or the traversal reaches roots the cache never scored and the containment
    # gate fires on what is really a configuration mismatch. Full-ontology caches leave this null.
    if cfg.get("restrict_to_subtree"):
        from hpo_extraction.evaluation.retrieval_analysis import descendants_with_hops
        keep = set(descendants_with_hops(tree, cfg.restrict_to_subtree))
        children_map = {h: [c for c in kids if c in keep]
                        for h, kids in children_map.items() if h in keep}
        roots = [cfg.restrict_to_subtree]
        logger.info("restricted to the subtree below %s: %d node(s)",
                    cfg.restrict_to_subtree, len(keep))

    def normalise(g, p):
        gold_set, pred_set, _ = normalise_pair(g, p, view)
        return gold_set, pred_set

    gold = load_gold(Path(cfg.gold_path))
    folds = read_folds(Path(cfg.folds_path))
    report_ids = sorted(gold)
    logger.info("cohort: %d reports, %d gold pairs, %d fold rows",
                len(report_ids), sum(len(v) for v in gold.values()), len(folds))

    space = load_spaces(cfg, out_dir, children_map, roots, report_ids)
    if space is None:
        raise SystemExit("no cache available — run stages=[ingest] with caches.*.calls_path set")
    logger.info("retrieval indices under selection: %s", ", ".join(space.indices))

    grid = {
        "poolings_prune": list(cfg.poolings_prune),
        "poolings_accept": list(cfg.poolings_accept),
        "tau_accept_grid": [float(t) for t in cfg.tau_accept_grid],
        "s_main": int(cfg.s_main),
        "alpha": float(cfg.alpha),
    }
    covered = sorted(set(report_ids) & set(space.report_ids(space.indices[0])))
    if len(covered) != len(report_ids):
        logger.warning("cache covers %d of %d cohort reports — every table below is over the %d",
                       len(covered), len(report_ids), len(covered))
    sampler = ReportResampler(len(covered), n_resamples=int(cfg.n_bootstrap), seed=0)
    quality_rows: list[dict] = []
    # Two things the later stages read, and they are NOT interchangeable:
    #   assignments  each held-out report of repetition 0 with the configuration ITS fold selected.
    #                Every number the chapter quotes "at the configuration" is measured on it, so
    #                it matches Table 1's pooled out-of-fold row by design.
    #   selected     the modal configuration of repetition 0, applied to every report. Only for the
    #                analyses that need ONE threshold drawn on one axis -- the score distributions,
    #                The calibration maps and the GSC+ transfer -- and labelled as such there.
    selected = None
    assignments = None

    # ── Containment: a gate, not a warning ──────────────────────────────────
    if "containment" in stage_list:
        rows = stages.containment_gate(space, gold, float(cfg.tau_cache),
                                       grid["poolings_prune"], [grid["s_main"]])
        write_csv(tables / "containment.csv", rows)
        violations = [r for r in rows if r["n_containment_misses"]]
        if violations:
            logger.error("CONTAINMENT FAILED on %d cell(s)", len(violations))
            raise SystemExit(
                "containment violated — the cache does not host this grid. Widen the cache or "
                "narrow the grid rather than reporting these numbers.")
        exhaustive = {r["n_cached_nodes_min"] for r in rows}
        logger.info("containment OK: %d cells, zero misses, %s nodes cached per report "
                    "(exhaustive => containment holds by construction, not by the g(tau) bound)",
                    len(rows), sorted(exhaustive))

    # ── E4.3: selection under nested CV ─────────────────────────────────────
    if "select" in stage_list:
        got = stages.run_selection(space, folds, gold, normalise, grid, sampler)
        quality_rows.append(got["row"])
        assignments = got["assignments"]
        # The pooled out-of-fold SET, not only its score. `hpo_extraction.evaluation.prediction_sets` exists for
        # The two drivers that select inside nested CV, this one and the PhenoJury protocol, because
        # what a report is predicted depends on the fold it fell in, so there is no per-term record
        # to hang the set on. The comparison reads this file to put TreePhenoRAG in the same table as the
        # fixed methods. Without it chapter 6's row reports `missing` however good the score was.
        write_prediction_sets(out_dir / "predictions" / "nested_cv_pooled.csv", got["pooled"])
        write_csv(tables / "selection_choices.csv", got["choices"])
        write_csv(tables / "crc.csv", got["crc"])
        write_csv(tables / "selection_stability.csv",
                  [{"field": k, **{kk: vv for kk, vv in v.items() if kk != "counts"},
                    "counts": json.dumps(v["counts"], default=str)}
                   for k, v in got["stability"].items()])
        write_csv(tables / "alpha_sensitivity.csv",
                  stages.alpha_sensitivity(space, folds, gold, normalise, grid,
                                           [float(a) for a in cfg.alpha_sweep]))
        # The modal choice over the first repetition: the ONE configuration the single-threshold
        # analyses (distributions, calibration, GSC+ transfer) are drawn at. Everything quoted at
        # The configuration uses `assignments` instead -- see the note where both are declared.
        first = [c for c in got["choices"] if c["repetition"] == 0]
        from collections import Counter
        from hpo_extraction.treephenorag.selection import Configuration
        key = Counter((c["retrieval_index"], c["pool_pr"], c["tau_prune"], c["pool_acc"],
                       c["tau_accept"], c["S"]) for c in first).most_common(1)[0][0]
        selected = Configuration(*key)
        logger.info("modal selected configuration: %s", selected.as_row())
        (out_dir / "selected_configuration.json").write_text(
            json.dumps(selected.as_row(), indent=2), encoding="utf-8")

    # ── T4.1: the retrieval gate ────────────────────────────────────────────
    # Reported before the ablations because it BOUNDS them: no pruning rule, pooling operator or
    # acceptance threshold below can recover an annotated term that never appeared in a retrieved
    # segment. HCY only -- it needs the curated ground truth's per-term evidence segments, which GSC+
    # does not have.
    if "retrieval" in stage_list:
        rows = stages.retrieval_sufficiency(
            space, gold, load_evidence(cfg.get("evidence_path")),
            [int(s) for s in cfg.s_retrieval_gate], view=view)
        if rows:
            write_csv(tables / "retrieval_sufficiency.csv", rows)
        else:
            logger.warning("retrieval gate not written: it needs evidence_path, and without it "
                           "Hit@S is not measurable, see docs/thesis_map.md")

    # ── T4.3: both pooling axes at the selected configuration ───────────────
    if "poolings" in stage_list and assignments is not None:
        write_csv(tables / "pooling_operators.csv", stages.pooling_operators(
            space, gold, normalise, assignments, grid,
            grid["poolings_prune"], grid["poolings_accept"]))

    # ── E4.4: the ablation ladder ───────────────────────────────────────────
    if "ablations" in stage_list:
        rows, units_by_label = stages.run_ladder(space, folds, gold, normalise, grid, sampler,
                                                 selected)
        quality_rows.extend(rows)
        adjusted = stages.paired_tests(units_by_label, int(cfg.n_permutations),
                                       float(cfg.alpha_holm))
        write_csv(tables / "ablation_tests.csv",
                  [{"comparison": k, **v} for k, v in adjusted.items()])
        if assignments is not None:
            write_csv(tables / "s_ablation.csv",
                      stages.s_ablation(space, folds, gold, normalise, grid,
                                        [int(s) for s in cfg.s_ablation]))
            units = stages.oracle_expansion(space, gold, normalise, assignments, view)
            quality_rows.append(stages.score_units(
                "oracle expansion (not an operating point)", units, sampler))

    # ── E4.6: calibration ───────────────────────────────────────────────────
    if "calibration" in stage_list and selected is not None:
        got = stages.run_calibration(space, selected, gold, covered,
                                     list(cfg.calibration_methods), int(cfg.n_calibration_bins))
        write_csv(tables / "calibration.csv", got["rows"])
        # The chapter's reliability figure: raw against CROSS-FITTED Platt, so no bin is drawn
        # under a map that was fitted on it. The in-sample maps above stay as diagnostics.
        crossfit = stages.reliability_crossfit(space, selected, gold, covered, folds,
                                               int(cfg.n_calibration_bins),
                                               int(cfg.n_smooth_ece_bootstrap))
        write_csv(tables / "reliability_crossfit.csv", crossfit["rows"])
        write_csv(tables / "calibration_crossfit.csv", crossfit["summary"])
        for method, h in got["maps"].items():
            write_csv(tables / f"reliability_{method}.csv",
                      stages.reliability_rows(got["scores"], got["labels"], h,
                                              int(cfg.n_calibration_bins)))

        gsc_cfg = cfg.get("gsc") or {}
        if gsc_cfg.get("gold_path"):
            gsc_gold = load_gold(Path(gsc_cfg.gold_path))
            gsc_space = load_spaces(gsc_cfg, out_dir, children_map, roots, sorted(gsc_gold),
                                    prefix=GSC_NPZ_PREFIX)
            if gsc_space is not None and selected.index in gsc_space.indices:
                transfer = stages.transfer_to_gsc(
                    space, gsc_space, selected, gold, gsc_gold, normalise,
                    grid["tau_accept_grid"])
                write_csv(tables / "transfer.csv",
                          [{"arm": k, **{kk: vv for kk, vv in v.items()
                                         if not isinstance(vv, dict)}}
                           for k, v in transfer.items()])
                # The selected configuration applied to GSC+ UNCHANGED: no folds, no risk control,
                # nothing fitted here. That is what `roster.py`'s `estimators` override on the
                # TreePhenoRAG row already promises the reader, and it is a different claim from
                # `transfer.csv` above, that table asks whether a *threshold* survives the genre
                # change under `h`, this row is the configuration's own score on the abstracts.
                # `h` is NOT applied: it reparametrises the acceptance axis, and a row
                # reported under a remapped threshold is not the configuration HCY selected.
                quality_rows.append(stages.score_transferred(
                    gsc_space, selected, gsc_gold, normalise,
                    "GSC+ transfer (HCY-selected configuration)"))
                # Named after the ground truth it was scored against, so the 114-document view and the
                # corpus ground truth cannot be confused for one another by a reader or by the comparison's
                # `protocol_predictions` path.
                write_prediction_sets(
                    out_dir / "predictions" / f"{Path(gsc_cfg.gold_path).stem}_transferred.csv",
                    gsc_space.predictions(selected))
            else:
                logger.warning("GSC+ cache for index %s not available — transfer skipped",
                               selected.index)
        else:
            logger.info("no GSC+ gold configured; the calibration transfer is the Tier-1 evidence "
                        "for h, so this run reports calibration as a diagnostic only")

    # ── E4.6b: do the two scores separate the cases they are thresholding? ──
    # its own stage, not part of `calibration`: calibration asks whether a
    # score means what it says, this asks whether it discriminates at all, and only the second is
    # a statement about the two thresholds the chapter reports.
    if "distributions" in stage_list and selected is not None:
        got = stages.run_distributions(space, selected, gold, covered, view,
                                       int(cfg.n_distribution_bins))
        for decision, block in got.items():
            write_csv(tables / f"score_distribution_{decision}.csv", block["rows"])
            write_csv(tables / f"score_logit_{decision}.csv", block["logit_rows"])
            write_csv(tables / f"score_survival_{decision}.csv", block["survival_rows"])
            write_csv(tables / f"score_pr_{decision}.csv", block["pr_rows"])
        write_csv(tables / "score_distribution_summary.csv",
                  [got[d]["summary"] for d in ("expansion", "acceptance")])

    # ── E4.6c: every acceptance operator ranked on the same pairs (AUC, AP) ──
    # Its own stage so it can run without `select`: it needs only the modal configuration, which
    # `select` wrote to selected_configuration.json, and it rewrites nothing else.
    if "ranking" in stage_list:
        if selected is None:
            from hpo_extraction.treephenorag.selection import Configuration
            saved = json.loads((out_dir / "selected_configuration.json").read_text("utf-8"))
            selected = Configuration(saved["retrieval_index"], saved["pool_pr"],
                                     float(saved["tau_prune"]), saved["pool_acc"],
                                     float(saved["tau_accept"]), int(saved["S"]))
            logger.info("ranking: modal configuration read from selected_configuration.json: %s",
                        selected.as_row())
        write_csv(tables / "acceptance_ranking.csv", stages.acceptance_ranking(
            space, selected, gold, covered, grid["poolings_accept"]))

    # ── E4.7: error analysis ────────────────────────────────────────────────
    if "diagnostics" in stage_list and assignments is not None:
        evidence = load_evidence(cfg.get("evidence_path"))
        # Figures 3 and 5 describe "the pooled out-of-fold predictions", so that is what they are
        # computed on -- per fold, under that fold's configuration (see stages.nested_diagnostics).
        got = stages.nested_diagnostics(space, assignments, gold, view, float(cfg.delta_m),
                                        evidence)
        decomposition = got["decomposition"]
        write_csv(tables / "recall_decomposition.csv", [{
            "bucket": b, "count": decomposition["counts"][b],
            "fraction": decomposition["fractions"][b],
            "evidence_available": decomposition["evidence_available"],
        } for b in decomposition["counts"]])
        write_csv(tables / "blocking_causes.csv",
                  [{"cause": k, "count": v} for k, v in decomposition["blocking_causes"].items()])
        write_csv(tables / "blocking_depth.csv",
                  [{"depth": k, "n_missed_gold": v}
                   for k, v in decomposition["blocking_depth_histogram"].items()])
        write_csv(tables / "error_taxonomy.csv",
                  [{"bucket": b, "count": got["taxonomy"]["counts"][b],
                    "fraction": got["taxonomy"]["fractions"][b]}
                   for b in got["taxonomy"]["counts"]])
        write_csv(tables / "near_miss.csv",
                  [{"distance": k, "n_false_positive": v}
                   for k, v in got["near_miss"]["histogram"].items()])
        # One row per missed annotated pair, not just the bucket totals. The comparison's complementarity
        # table joins on this: asking which of TreePhenoRAG's losses PhenoJury recovers needs to
        # know which pairs were lost and to what, and the totals cannot answer it without
        # assuming the two methods' misses are independent.
        write_csv(tables / "recall_attribution.csv", got["attribution"])
        if not decomposition["evidence_available"]:
            logger.warning("no evidence segments: the retrieval bucket is NOT measurable and must "
                           "not be quoted — set evidence_path to the curated gold's annotations")
        logger.info("recall decomposition: %s", decomposition["counts"])
        logger.info("coverage loss %.4f | loss at scored terms %.4f",
                    decomposition["coverage_loss"], decomposition["loss_at_scored_terms"])

        # The delta_m sweep is a sensitivity check of the decomposition's own margin, run at the
        # modal configuration on every report: it needs one threshold, and it is not quoted as a
        # result at the configuration.
        from hpo_extraction.evaluation.metrics import delta_m_sensitivity
        predicted = space.predictions(selected)
        axis = space.prune_axis(selected.index, selected.pool_pr, selected.s)
        accept = space.accept_scores(selected.index, selected.pool_acc, selected.s)
        ids = [r for r in covered if r in axis.r]
        write_csv(tables / "delta_m_sensitivity.csv", delta_m_sensitivity(
            [float(d) for d in cfg.delta_m_sweep],
            gold_by_report={r: gold.get(r, set()) for r in ids},
            predicted_by_report=predicted,
            scored_by_report={r: set(space.node_names[axis.visited(r, selected.tau_prune)])
                              for r in ids},
            expanded_by_report={r: set(space.node_names[axis.expanded(r, selected.tau_prune)])
                                for r in ids},
            caches=space.caches_by_index[selected.index], view=view,
            pooled_by_report={r: {space.graph.node_ids[i]: float(accept[r][i])
                                  for i in np.flatnonzero(axis.visited(r, selected.tau_prune))}
                              for r in ids},
            threshold=selected.tau_accept, evidence=evidence,
        ))

    # ── The inline numbers the prose quotes ─────────────────────────────────
    # Cheap, and last because nothing depends on them. Each is a number a paragraph asserts, and
    # a number in a paragraph goes stale as easily as one in a table and far less visibly.
    if "instruments" in stage_list:
        cache_rows = stages.cache_calls(space, "hcy")
        gsc_cfg = cfg.get("gsc") or {}
        if gsc_cfg.get("gold_path"):
            gsc_space = load_spaces(gsc_cfg, out_dir, children_map, roots,
                                    sorted(load_gold(Path(gsc_cfg.gold_path))),
                                    prefix=GSC_NPZ_PREFIX)
            if gsc_space is not None:
                cache_rows += stages.cache_calls(gsc_space, "gsc_raghpo_ann")
        write_csv(tables / "cache_calls.csv", cache_rows)
    if "instruments" in stage_list and selected is not None:
        rows = stages.calls_per_report(space, assignments) if assignments is not None else []
        if rows:
            write_csv(tables / "calls_per_report.csv", rows)
        rows = stages.captured_mass(space, selected)
        if rows:
            write_csv(tables / "captured_mass.csv", rows)
        write_csv(tables / "replay_fidelity.csv", stages.replay_fidelity(
            space, selected, [selected.pool_pr], [int(cfg.s_main)]))

    if quality_rows:
        write_csv(tables / "core_quality.csv", quality_rows)

    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    logger.info("wrote %s", tables)


if __name__ == "__main__":
    main()
