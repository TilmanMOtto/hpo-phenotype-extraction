"""A miniature earlier output tree, byte-compatible with what the real drivers write.

The earlier runs'artifacts live on the cluster, so the only way to establish that ``result_tables`` reads them
correctly here is to reproduce the schemas and hand-compute what every metric should say
about them. Each writer below mirrors one dict literal from ``src/core/*_experiment.py``, the
file:line is named in its docstring so a schema change upstream shows up as a failing test rather
than as a silently empty column.

The scenario is built on the toy ontology (``tests/fixtures/toy_ontology``) with two reports:

===========  ==============  ============================================================
report       ground truth            what the tree method does at tau_0.1
===========  ==============  ============================================================
``r1``       ``{C, F}``      visits A,B,C,E,F,G. Accepts C and D  -> 1 TP, 1 FP, 1 FN
``r2``       ``{D}``         visits A,B,D,G. Accepts D        -> 1 TP, 0 FP, 0 FN
===========  ==============  ============================================================

Pooled that is TP=2, FP=1, FN=1, so micro P = R = F1 = 2/3, which is the number
``test_exp13_07_sections`` asserts. At ``tau_0.5`` the traversal prunes ``E``, so ``F`` becomes
unreachable and ``r1`` loses nothing further (it never accepted ``F`` anyway), which is what makes
the two taus differ in reachability recall but not in flat F1, and both facts are fixed.

``an earlier exploratory run`` is written as the one **2-D** sweep (``tau_{p}_acc_{a}/``), because that layout is
otherwise unrepresented and the two directory namings have to be readable side by side. Its
τ_accept = 0.5 slice reproduces the numbers above. Its 0.9 slice keeps ``C`` alone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fixtures.toy_ontology import (
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    HALLUCINATION,
    M,
    NAMES,
    OBSOLETE_C,
)

#: ``{report_id: gold set}``, the scenario's ground truth.
GOLD: dict[str, list[str]] = {"r1": [C, F], "r2": [D]}

#: ``{report_id: accepted set}``, what the tree method reports at every tau in this fixture.
ACCEPTED: dict[str, list[str]] = {"r1": [C, D], "r2": [D]}

#: Nodes visited per report, per tau. tau_0.5 prunes E, so F is never reached.
VISITED: dict[str, dict[str, list[str]]] = {
    "tau_0.1": {"r1": [A, B, C, D, E, F, G], "r2": [A, B, D, G]},
    "tau_0.5": {"r1": [A, B, C, D, E, G], "r2": [A, B, D, G]},
}

#: Which visited nodes had their children revealed, per tau.
EXPANDED: dict[str, dict[str, list[str]]] = {
    "tau_0.1": {"r1": [A, B, E, G], "r2": [A, B, G]},
    "tau_0.5": {"r1": [A, B, G], "r2": [A, B, G]},
}

#: Per-node prune / accept scores. Constant across taus, as the driver memoises node evaluation.
PRUNE_SCORE: dict[str, float] = {A: 0.9, B: 0.8, C: 0.7, D: 0.6, E: 0.3, F: 0.55, G: 0.2}
ACCEPT_SCORE: dict[str, float] = {A: 0.2, B: 0.3, C: 0.9, D: 0.8, E: 0.1, F: 0.4, G: 0.05}

#: The accept thresholds an earlier exploratory run's grid is written at, against ``ACCEPT_SCORE`` above. 0.5
#: reproduces :data:`ACCEPTED` (so the 2-D fixture agrees with the 1-D one where they
#: describe the same run); 0.9 keeps only ``C``, which empties ``r2``'s prediction set, the case
#: where a report exists solely as a summary line.
ACCEPT_GRID: tuple[float, ...] = (0.5, 0.9)

#: Report text, and its segmentation as the pipeline would produce it. Segment 0 evidences C,
#: segment 2 evidences F. Segment 1 evidences nothing.
REPORT_TEXT: dict[str, str] = {
    "r1": "The patient had seizures.\nNo fever was noted.\nMRI showed cerebral atrophy.",
    "r2": "Marked anxiety on examination.\nOtherwise unremarkable.",
}
SEGMENTS: dict[str, list[str]] = {
    "r1": ["The patient had seizures.", "No fever was noted.", "MRI showed cerebral atrophy."],
    "r2": ["Marked anxiety on examination.", "Otherwise unremarkable."],
}
SEGMENT_ANNOTATIONS: dict[str, dict[int, list[str]]] = {
    "r1": {0: [C], 2: [F]},
    "r2": {0: [D]},
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_predictions(path: Path, predicted: dict[str, list[str]],
                      gold: dict[str, list[str]], hpo_label: bool = False,
                      candidates: dict[str, list[str]] | None = None) -> None:
    """The predictions contract shared by every driver.

    Per-term lines then a ``summary: true`` line, ``tree_experiment.py:451-459``. RAG-HPO adds an
    ``hpo_label`` field (``rag_hpo_experiment.py:143``); ``flat_topm`` adds ``candidates`` to the
    summary (``flat_topm_experiment.py:151-159``).
    """
    records: list[dict] = []
    for rid in sorted(set(predicted) | set(gold)):
        gold_set = set(gold.get(rid, []))
        for hpo_id in sorted(predicted.get(rid, [])):
            rec = {"report_id": rid, "hpo_id": hpo_id, "prediction": 1,
                   "ground_truth": int(hpo_id in gold_set)}
            if hpo_label:
                rec["hpo_label"] = f"label of {hpo_id}"
            records.append(rec)
        summary = {"report_id": rid, "summary": True,
                   "predicted_set": sorted(predicted.get(rid, [])),
                   "gold_set": sorted(gold_set)}
        if candidates is not None:
            summary["candidates"] = list(candidates.get(rid, []))
        records.append(summary)
    _write_jsonl(path, records)


def accepted_at(tau: str, tau_accept: float) -> dict[str, list[str]]:
    """Which of *tau*'s visited nodes an accept threshold keeps, an earlier exploratory run's second axis.

    Mirrors ``TraversalResult.at_accept`` (``core/tree_experiment.py:347``): one BFS per τ_prune
    serves the whole accept grid, so the visited set is a function of τ_prune alone and only the
    accept decision moves.
    """
    return {rid: [h for h in nodes if ACCEPT_SCORE[h] >= tau_accept]
            for rid, nodes in VISITED[tau].items()}


def write_nodes(path: Path, tau: str, gold: dict[str, list[str]],
                accepted_by_report: dict[str, list[str]] | None = None) -> None:
    """One line per visited node, ``tree_experiment.py:444-449``.

    Carries both scores and the ``is_gold`` label, which is why the tree's calibration sample
    needs no join. *accepted_by_report* overrides :data:`ACCEPTED` for a 2-D sweep, where the
    accepted set is what the configuration's τ_accept keeps, not a constant.
    """
    records = []
    for rid, nodes in VISITED[tau].items():
        gold_set = set(gold.get(rid, []))
        expanded = set(EXPANDED[tau][rid])
        accepted = set((accepted_by_report or ACCEPTED).get(rid, []))
        for hpo_id in nodes:
            records.append({
                "report_id": rid, "hpo_id": hpo_id, "depth": _depth(hpo_id),
                "prune_score": PRUNE_SCORE[hpo_id], "accept_score": ACCEPT_SCORE[hpo_id],
                "expanded": hpo_id in expanded, "accepted": hpo_id in accepted,
                "is_gold": int(hpo_id in gold_set),
            })
    _write_jsonl(path, records)


def write_traversal(path: Path, tau: str,
                    accepted_by_report: dict[str, list[str]] | None = None) -> None:
    """One line per report with the nested ``depth_table``, ``tree_experiment.py:461-468``."""
    records = []
    for rid, nodes in VISITED[tau].items():
        expanded = set(EXPANDED[tau][rid])
        accepted = set((accepted_by_report or ACCEPTED).get(rid, []))
        by_depth: dict[int, list[str]] = {}
        for hpo_id in nodes:
            by_depth.setdefault(_depth(hpo_id), []).append(hpo_id)

        depth_table, cumulative = [], 0
        for depth in sorted(by_depth):
            group = by_depth[depth]
            cumulative += len(group)
            depth_table.append({
                "depth": depth,
                "n_visited": len(group),
                "n_expanded": sum(1 for h in group if h in expanded),
                "n_accepted": sum(1 for h in group if h in accepted),
                "n_slm_calls": 5 * len(group),
                "cumulative_visited": cumulative,
            })
        records.append({
            "report_id": rid,
            "n_unique_nodes_visited": len(nodes),
            # Two paths reach M in the real DAG. Here the surplus stands for that effect.
            "total_frontier_insertions": len(nodes) + 1,
            "n_slm_calls": 5 * len(nodes),
            "n_accepted": len(accepted),
            "depth_table": depth_table,
        })
    _write_jsonl(path, records)


def write_calls(path: Path, variant: str, ctx_types: tuple[str, ...] = ("union",)) -> None:
    """One line per (node, context, retrieved sentence), ``tree_experiment.py:424-439``.

    ``sent_index`` indexes the segmentation, and the ranks are ordered so the top-ranked segment
    for ``C`` in ``r1`` is the one actually annotated with ``C``, i.e. this fixture describes a
    retriever that works, so a P@S of 0 in a test means the code is wrong, not the fixture.
    """
    ranked = {
        ("r1", C): [0, 1, 2], ("r1", F): [2, 0, 1], ("r1", D): [1, 0, 2],
        ("r2", D): [0, 1], ("r2", B): [1, 0],
    }
    records = []
    for (rid, hpo_id), order in sorted(ranked.items()):
        for ctx in ctx_types:
            for rank, sent_index in enumerate(order, start=1):
                # Rank 1 gets a positive margin (a "Yes"), the rest negative, so the AnyYes
                # accept score reconstructed from these margins is sigmoid(2.0) for every pair.
                margin = 2.0 if rank == 1 else -2.0
                rec = {
                    "report_id": rid, "hpo_id": hpo_id, "rank": rank,
                    "sent_index": sent_index, "cosine_sim": round(0.9 - 0.1 * rank, 6),
                    "logit_yes": margin, "logit_no": -margin, "logsumexp_all": 3.0,
                    "top1_token_id": 9642, "margin": margin,
                    "verdict": "Yes" if margin > 0 else "No",
                }
                if variant != "flat_topm":
                    rec["ctx_type"] = ctx
                records.append(rec)
    _write_jsonl(path, records)


def write_timing(path: Path, candidates: dict[str, list[str]]) -> None:
    """``flat_topm_experiment.py:160-162``, counts only, no wall-clock."""
    _write_jsonl(path, [
        {"report_id": rid, "n_slm_calls": 5 * len(c), "n_candidates": len(c)}
        for rid, c in sorted(candidates.items())
    ])


def write_detections(path: Path, models: tuple[str, ...] = ("m1", "m2", "m3", "m4")) -> None:
    """``slm_ensemble_experiment.py:306-312``, one line per (report, sentence, HPO, model).

    Vote counts are staged so the confidence has several distinct atoms: ``C`` in ``r1`` is found
    by all four models (1.0), ``D`` by two (0.5), ``E`` by one (0.25).
    """
    votes = {("r1", C): 4, ("r1", D): 2, ("r1", E): 1, ("r2", D): 3, ("r2", B): 1}
    records = []
    for (rid, hpo_id), n in sorted(votes.items()):
        for model in models[:n]:
            records.append({"report_id": rid, "model": model, "sentence_number": 0,
                            "hpo_id": hpo_id, "count": 1})
    _write_jsonl(path, records)


def write_resume_state(path: Path, n_reports: int = 2, step: float = 12.5) -> None:
    """``checkpoint.py:97``, the per-report ``ts`` trace ``costs.py`` differences."""
    _write_jsonl(path, [
        {"unit_id": f"r{i + 1}", "ts": 1_700_000_000.0 + i * step, "n_nodes": 7}
        for i in range(n_reports)
    ])


def _depth(hpo_id: str) -> int:
    return {A: 1, G: 1, B: 2, E: 2, C: 3, D: 3, F: 3}[hpo_id]


def build_exp13_tree(root: Path, *, cohorts: tuple[str, ...] = ("hcy", "gsc"),
                     methods: tuple[str, ...] | None = None,
                     taus: tuple[str, ...] = ("tau_0.1", "tau_0.5")) -> Path:
    """Write a complete miniature earlier output tree under *root*, and return it.

    *methods* selects which experiment directories exist. Omitting one is how the placeholder
    path gets tested. The layout matches the drivers:
    ``<root>/<exp_id>/<cohort>/[<operating_point>/]<variant>_<kind>.jsonl``.
    """
    selected = methods if methods is not None else (
        "exp13_00_tree_gate_lr", "exp13_01_tree_noisyor", "exp13_02_tree_no_prune",
        "baseline_raghpo_8b", "exp13_05_topm_retrieval_slm", "phenojury_generation_free_listing",
        "baseline_phenobert", "exp13_09_tree_lse_beta1",
    )
    candidates = {"r1": [C, D, F, E], "r2": [D, B, A]}

    for cohort in cohorts:
        for exp_id in selected:
            run_dir = root / exp_id / cohort
            run_dir.mkdir(parents=True, exist_ok=True)

            if exp_id.startswith(("exp13_00", "exp13_01", "exp13_02")):
                variant = exp_id.removeprefix("exp13_00_").removeprefix(
                    "exp13_01_").removeprefix("exp13_02_")
                sweep = ("tau_0.5",) if exp_id.startswith("exp13_02") else taus
                write_calls(run_dir / f"{variant}_calls.jsonl", variant)
                write_resume_state(run_dir / f".checkpoint_{variant}.jsonl")
                (run_dir / "run.log").write_text(
                    "2026-01-01 | INFO | Done | 2 reports | peak GPU 14.25 GB\n")
                for tau in sweep:
                    write_predictions(run_dir / tau / f"{variant}_predictions.jsonl",
                                      ACCEPTED, GOLD)
                    write_nodes(run_dir / tau / f"{variant}_nodes.jsonl", tau, GOLD)
                    write_traversal(run_dir / tau / f"{variant}_traversal.jsonl", tau)

            elif exp_id.startswith("exp13_09"):
                # The one 2-D sweep: |tau_prune| x |tau_accept| directories named
                # ``tau_{p:g}_acc_{a:g}`` (``tree_experiment._tau_dir_name``). Everything else is
                # The ordinary tree layout, which is the point, one BFS per tau_prune, and the
                # accept grid rides on it.
                variant = "tree_lse_beta1"
                write_calls(run_dir / f"{variant}_calls.jsonl", variant)
                write_resume_state(run_dir / f".checkpoint_{variant}.jsonl")
                for tau in taus:
                    for accept in ACCEPT_GRID:
                        point = f"{tau}_acc_{accept:g}"
                        accepted = accepted_at(tau, accept)
                        write_predictions(run_dir / point / f"{variant}_predictions.jsonl",
                                          accepted, GOLD)
                        write_nodes(run_dir / point / f"{variant}_nodes.jsonl", tau, GOLD,
                                    accepted_by_report=accepted)
                        write_traversal(run_dir / point / f"{variant}_traversal.jsonl", tau,
                                        accepted_by_report=accepted)

            elif exp_id.startswith(("baseline_raghpo_8b", "baseline_raghpo_70b")):
                write_predictions(run_dir / "rag_hpo_predictions.jsonl", ACCEPTED, GOLD,
                                  hpo_label=True)
                _write_jsonl(run_dir / "rag_hpo_retrieved_segments.jsonl", [
                    {"report_id": "r1", "hpo_id": C, "hpo_label": "Seizure", "rank": 1,
                     "text": SEGMENTS["r1"][0], "cosine_sim": 0.81, "slm_verdict": "Yes"},
                ])

            elif exp_id.startswith("exp13_05"):
                write_predictions(run_dir / "flat_topm_predictions.jsonl", ACCEPTED, GOLD,
                                  candidates=candidates)
                write_calls(run_dir / "flat_topm_calls.jsonl", "flat_topm")
                write_timing(run_dir / "flat_topm_timing.jsonl", candidates)

            elif exp_id.startswith("phenojury_generation_free_listing"):
                write_detections(run_dir / "slm_ensemble_detections.jsonl")
                for rule in ("vote_k1", "vote_k2", "agg_plurality"):
                    write_predictions(run_dir / rule / "slm_ensemble_predictions.jsonl",
                                      ACCEPTED, GOLD)
                (run_dir / "slm_ensemble_agg_summary.csv").write_text(
                    "rule,k,n_predicted,micro_precision,micro_recall,micro_f1\n"
                    "vote_k1,1,3,0.6666666666666666,0.6666666666666666,0.6666666666666666\n")

            elif exp_id.startswith("baseline_phenobert"):
                write_predictions(run_dir / "phenobert_predictions.jsonl", ACCEPTED, GOLD,
                                  hpo_label=True)
                write_phenobert_detections(run_dir)
                _write_jsonl(run_dir / "phenobert_timing.jsonl", [
                    {"stage": "annotate", "n_reports": 2, "n_detections": 4, "n_predictions": 3,
                     "annotate_s": 4.0, "duration_s": 4.5, "mean_time_per_report_s": 2.25},
                ])
    return root


#: ``{report_id: [(hpo_id, raw_hpo_id, phrase, score, negated)]}`` for the small the PhenoBERT baseline cell
#: inside :func:`build_exp13_tree`. The staged text is :data:`REPORT_TEXT`, the cohort's own
#: report, verbatim, because that is what ``stage_reports`` writes, and because the deep-dive UI
#: places these offsets onto the *same* report's sentences. A fixture that staged some other text
#: would exercise only the alignment failure and never the annotation itself.
#:
#: The phrases cover the three outcomes a match can have. ``fever`` is the unresolved one for the
#: honest reason that this toy release has no fever term, and the negated one is a mention this
#: report does not actually negate, PhenoBERT's negation detector produces those, and a panel
#: that shows negation is a panel that has to render them.
_PB_CELL: dict[str, list[tuple[str, str, str, float, bool]]] = {
    "r1": [(C, C, "seizures", 0.97, False),
           (HALLUCINATION, HALLUCINATION, "fever", 0.71, False),
           (F, "HP:0000002", "cerebral atrophy", 0.88, False)],
    "r2": [(D, D, "anxiety", 0.93, True)],
}


def write_phenobert_detections(run_dir: Path) -> None:
    """The PhenoBERT baseline's per-detection dump and the staged reports it indexes.

    ``collect_detections`` (``phenobert_experiment.py``) writes character offsets into
    ``phenobert_input/<stem>.txt``, which ``stage_reports`` guarantees is the report verbatim. Both
    halves are written here because the offsets are only meaningful against that text, and because
    the deep-dive UI underlines the report with them, so a fixture that wrote offsets into no text
    would let a broken highlight through.

    Carries the three cases the predicted set is derived by *excluding*: a negated mention, a code
    this ontology release does not have (``resolved: false``), and an alt id that was remapped onto
    its primary, so a regression that stops filtering any of them shows up here.
    """
    run_dir = Path(run_dir)
    (run_dir / "phenobert_input").mkdir(parents=True, exist_ok=True)
    rows = []
    for report_id, detections in sorted(_PB_CELL.items()):
        text = REPORT_TEXT[report_id]
        (run_dir / "phenobert_input" / f"{report_id}.txt").write_text(text, encoding="utf-8")
        for hpo_id, raw_hpo_id, phrase, score, negated in detections:
            start = text.find(phrase)
            assert start >= 0, f"{report_id} does not contain {phrase!r}"
            rows.append({
                "report_id": report_id, "hpo_id": hpo_id,
                "hpo_label": NAMES.get(hpo_id, hpo_id), "raw_hpo_id": raw_hpo_id,
                "resolved": hpo_id != HALLUCINATION, "phrase": phrase,
                "start": start, "end": start + len(phrase),
                "score": score, "negated": negated,
            })
    _write_jsonl(run_dir / "phenobert_detections.jsonl", rows)


# ══════════════════════════════════════════════════════════════════════════════
# The Free Listing generation run, the complete SLM-ensemble run (extract stage + aggregate stage)
# ══════════════════════════════════════════════════════════════════════════════
# ``build_exp13_tree`` writes only as much of the Free Listing generation run as the result-table library reads: the merged detections
# dump and the per-rule predictions. The deep-dive UI reads the whole thing, the raw generations,
# The PhenoBERT TSVs with their character offsets, the per-model detection slices, so the builder
# below produces the full artifact set.
#
# Two rules it follows, both essential:
#
# 1. **The extract-stage artifacts are written, then read back through the real parser.** The TSVs
#    carry genuine character offsets into the file ``write_phenobert_input`` would have written, and
#    ``detections_{model}.jsonl`` is produced by running ``parse_phenobert_sentences`` over them.
#    A fixture that hand-wrote the detections could not catch an attribution bug.
# 2. **The shipped aggregate artifacts come from the driver's own functions**, ``vote_k_sets``,
#    ``plurality_sets``, ``_write_rule``, ``compute_slm_productivity``. So when the UI's
#    verification gates compare "recomputed" against "shipped", they are comparing against real
#    driver output, not against numbers typed into this file.
#
# The scenario is built so that every branch the UI has to explain occurs once:
#
# ==========  ==================  ==================================================
# report      ground truth                what it exercises
# ==========  ==================  ==================================================
# ``r1``      ``{C, F, D}``       C found by all 4 (TP everywhere); F never grounded
#                                 (FN ``not_linked``); D found by one model only
#                                 (FN ``below_votes`` at k>=2, TP at k=1); B is an
#                                 ancestor FP, M a sibling FP
# ``r2``      ``{D, M}``          D unanimous; M detected only as *negated*, so it is
#                                 dropped before it ever votes (FN ``negated``)
# ``r3``      ``{}``              empty ground-truth set, a hallucinated G scores as ``no_gt``
# ``r4``      ``{H}``             every model replies "no phenotype" (FN ``not_written``)
# ==========  ==================  ==================================================
#
# Model quirks covered: ``deepseek`` wraps every reply in a ``<think>`` block (so an offset computed
# on the raw reply instead of the stripped one lands in the wrong place) and its ``r1`` sentence 2
# block is unterminated (``strip_think`` drops the reply whole). ``apertus``/``deepseek`` write the
# 6-column *patched* TSV, ``llama``/``phi4`` the stock 5-column one, so both sentence-attribution
# routes in ``parse_phenobert_sentences`` are exercised.

#: The four ensemble members used here, real ``MODEL_KEYS`` entries, in ``MODEL_KEYS`` order.
SLM_MODELS: tuple[str, ...] = ("apertus", "deepseek", "llama", "phi4")

#: ``{report_id: [sentence]}``, the Stanza segmentation the extract stage would have produced.
SLM_SENTENCES: dict[str, list[str]] = {
    "r1": ["The patient had seizures.",
           "No fever was noted.",
           "MRI showed cerebral atrophy."],
    "r2": ["Marked anxiety on examination.",
           "No spinal cord abnormality was seen."],
    "r3": ["Routine follow-up, nothing new."],
    "r4": ["Discharged in good condition."],
}

#: ``{report_id: gold set}``, annotated terms only, the earlier runs'open-set frame (no ancestor closure).
SLM_GOLD: dict[str, list[str]] = {"r1": [C, D, F], "r2": [D, M], "r3": [], "r4": [H]}

_NOTHING = "no phenotype"

#: ``{model: {(report, sentence): raw llm_output}}``. Anything not listed replies ``no phenotype``.
SLM_REPLIES: dict[str, dict[tuple[str, int], str]] = {
    "apertus": {
        ("r1", 0): "Seizures.",
        ("r2", 0): "Anxiety.",
        ("r2", 1): "Spinal cord abnormality.",
    },
    "deepseek": {
        # The reasoning block names cerebral atrophy, an offset computed before strip_think
        # would land inside it. It is dropped, so nothing here grounds to F.
        ("r1", 0): "<think>Could be fever, or cerebral atrophy. Seizure fits best.</think>Seizures.",
        # Unterminated: strip_think drops the whole reply, leaving an empty PhenoBERT input block.
        ("r1", 2): "<think>The MRI finding might be an atrophy of some kind",
        ("r2", 0): "<think>Anxiety is the obvious one.</think>Anxiety.",
    },
    "llama": {
        ("r1", 0): "Seizures. Abnormal nervous system physiology.",
        ("r2", 0): "Anxiety.",
        ("r2", 1): "Spinal cord abnormality.",
    },
    "phi4": {
        ("r1", 0): "Seizures. Abnormal nervous system physiology. Abnormality of the spinal cord.",
        ("r1", 1): "Anxiety.",
        ("r2", 0): "Anxiety.",
        ("r3", 0): "Skeletal abnormality.",
    },
}

#: ``{model: {(report, sentence): [(phrase, hpo_id, confidence, negated)]}}``.
#: Every phrase must occur verbatim in that model's *stripped* reply, the offsets are found by
#: searching for it, so a typo here surfaces as a missing detection, not a wrong one.
SLM_DETECTIONS: dict[str, dict[tuple[str, int], list[tuple[str, str, float, bool]]]] = {
    "apertus": {
        ("r1", 0): [("Seizures", C, 0.93, False)],
        ("r2", 0): [("Anxiety", D, 0.88, False)],
        ("r2", 1): [("Spinal cord abnormality", M, 0.81, True)],   # negated → never votes
    },
    "deepseek": {
        ("r1", 0): [("Seizures", C, 0.91, False)],
        ("r2", 0): [("Anxiety", D, 0.86, False)],
    },
    "llama": {
        ("r1", 0): [("Seizures", C, 0.90, False),
                    ("Abnormal nervous system physiology", B, 0.72, False)],
        ("r2", 0): [("Anxiety", D, 0.84, False)],
        ("r2", 1): [("Spinal cord abnormality", M, 0.79, True)],   # negated → never votes
    },
    "phi4": {
        ("r1", 0): [("Seizures", C, 0.94, False),
                    ("Abnormal nervous system physiology", B, 0.70, False),
                    ("Abnormality of the spinal cord", M, 0.68, False)],
        ("r1", 1): [("Anxiety", D, 0.83, False)],                  # The lone voter for D in r1
        ("r2", 0): [("Anxiety", D, 0.87, False)],
        ("r3", 0): [("Skeletal abnormality", G, 0.66, False)],     # hallucination, r3 has no ground truth
    },
}

#: Which models write the 6-column *patched* TSV (sentence index in the file) vs the stock 5-column
#: one (attribution recovered from character offsets). Both routes must work.
SLM_PATCHED_TSV: frozenset[str] = frozenset({"apertus", "deepseek"})


def slm_extraction_records(model: str, report_ids: list[str] | None = None) -> list[dict]:
    """This model's ``llm_extractions_{model}.jsonl`` records, ``slm_ensemble_experiment.py:265``."""
    reports = report_ids if report_ids is not None else sorted(SLM_SENTENCES)
    replies = SLM_REPLIES.get(model, {})
    return [
        {
            "model": model,
            "patient_id": rid,
            "sentence_number": i,
            "sentence_text": sent,
            "llm_output": replies.get((rid, i), _NOTHING),
        }
        for rid in reports
        for i, sent in enumerate(SLM_SENTENCES[rid])
    ]


def _write_phenobert_tsvs(pb_dir: Path, model: str, records: list[dict]) -> None:
    """Write one model's ``phenobert_output_{model}/`` with real offsets into the real input file.

    Offsets index the concatenated per-patient file ``write_phenobert_input`` builds from the
    *stripped* replies, which is the contract ``parse_phenobert_sentences`` verifies against. The
    phrase is located inside its own sentence block, so a phrase that also appears in a dropped
    ``<think>`` block cannot be matched by accident.
    """
    from hpo_extraction.phenojury.phenobert import build_phenobert_input
    from hpo_extraction.models.verdict import strip_think

    stripped = [{**r, "llm_output": strip_think(r.get("llm_output", "") or "")} for r in records]
    planned = SLM_DETECTIONS.get(model, {})
    pb_dir.mkdir(parents=True, exist_ok=True)

    for rid in sorted({r["patient_id"] for r in records}):
        text, spans = build_phenobert_input(stripped, rid)
        rows: list[str] = []
        for block_start, block_end, sent_num in spans:
            for phrase, hpo_id, conf, negated in planned.get((rid, sent_num), []):
                start = text.find(phrase, block_start, block_end)
                if start < 0:
                    raise AssertionError(
                        f"fixture is inconsistent: {model}/{rid}/sentence {sent_num} has no "
                        f"occurrence of {phrase!r} in its stripped reply"
                    )
                fields = [str(start), str(start + len(phrase)), phrase, hpo_id, f"{conf:.4f}"]
                if model in SLM_PATCHED_TSV:
                    fields += [str(sent_num), "input"]
                if negated:
                    fields.append("Neg")
                rows.append("\t".join(fields))
        (pb_dir / f"{rid}.txt").write_text("\n".join(rows) + ("\n" if rows else ""),
                                           encoding="utf-8")


def build_slm_ensemble_run(run_dir: Path, models: tuple[str, ...] = SLM_MODELS,
                           min_detection_count: int = 1) -> Path:
    """Write a complete the Free Listing generation run directory under *run_dir*, and return it.

    Both stages, in the order the cluster runs them: one ``extract`` pass per model (generations →
    PhenoBERT TSVs → detections slice → timing), then the ``aggregate`` pass (merged dump, the
    ``vote_k{1..n}`` and ``agg_plurality`` rule directories, the summary CSV, the per-SLM CSV).

    Everything downstream of the TSVs is produced by the real driver code, so the resulting tree is
    what ``slm_ensemble_experiment.execute`` would have written for this scenario.
    """
    from hpo_extraction.phenojury.ensemble_eval import (
        compute_slm_productivity,
        derive_patient_hpos,
        merge_slm_metrics,
        write_slm_metrics,
    )
    from hpo_extraction.phenojury.phenobert import build_phenobert_input, parse_phenobert_sentences
    from hpo_extraction.phenojury.generation import (
        _safe_micro_macro,
        _write_detections,
        _write_rule,
        plurality_sets,
        vote_k_sets,
    )
    from hpo_extraction.models.verdict import strip_think

    import pandas as pd

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_ids = sorted(SLM_SENTENCES)
    logger = logging.getLogger("fixtures.exp13_output")

    # ── stage 1: extract, once per model ─────────────────────────────────────
    all_records: dict[str, list[dict]] = {}
    sent_hpos_by_model: dict[str, dict] = {}
    for model in models:
        records = slm_extraction_records(model, report_ids)
        _write_jsonl(run_dir / f"llm_extractions_{model}.jsonl", records)

        pb_dir = run_dir / f"phenobert_output_{model}"
        _write_phenobert_tsvs(pb_dir, model, records)

        # Read the TSVs back through the production parser, the same call the driver makes.
        stripped = [{**r, "llm_output": strip_think(r.get("llm_output", "") or "")}
                    for r in records]
        inputs = {rid: build_phenobert_input(stripped, rid) for rid in report_ids}
        sent_hpos = parse_phenobert_sentences(str(pb_dir), inputs=inputs)
        n_det = _write_detections(model, sent_hpos, str(run_dir))

        all_records[model] = records
        sent_hpos_by_model[model] = sent_hpos
        _write_jsonl(run_dir / f"slm_ensemble_timing_{model}.jsonl", [
            {"stage": "extract", "model": model, "n_reports": len(report_ids),
             "n_sentences": len(records), "n_detections": n_det,
             "extraction_s": 4.5, "duration_s": 9.0},
            {"stage": f"phenobert_{model}", "duration_s": 4.5, "status": "ok"},
        ])

    # ── stage 2: aggregate ───────────────────────────────────────────────────
    merged = run_dir / "slm_ensemble_detections.jsonl"
    with open(merged, "w", encoding="utf-8") as out:
        for model in models:
            with open(run_dir / f"detections_{model}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)

    gold = {rid: set(SLM_GOLD.get(rid, [])) for rid in report_ids}
    per_model_report_hpos = derive_patient_hpos(sent_hpos_by_model)
    active = list(models)

    summary_rows: list[dict] = []
    for k in range(1, len(active) + 1):
        predicted = vote_k_sets(per_model_report_hpos, active, report_ids, k, min_detection_count)
        metrics, _ = _write_rule(str(run_dir), f"vote_k{k}", predicted, gold, report_ids, logger)
        summary_rows.append({"rule": f"vote_k{k}", "k": k,
                             "n_predicted": sum(len(v) for v in predicted.values()), **metrics})

    plurality, vote_rows = plurality_sets(
        sent_hpos_by_model, active, report_ids, min_detection_count
    )
    metrics_plur, _ = _write_rule(str(run_dir), "agg_plurality", plurality, gold, report_ids, logger)
    summary_rows.append({"rule": "agg_plurality", "k": "",
                         "n_predicted": sum(len(v) for v in plurality.values()), **metrics_plur})

    pd.DataFrame(summary_rows).to_csv(run_dir / "slm_ensemble_agg_summary.csv", index=False)
    write_slm_metrics(
        merge_slm_metrics(compute_slm_productivity(all_records, sent_hpos_by_model, active),
                          vote_rows),
        str(run_dir / "slm_ensemble_slm_metrics.csv"),
    )
    _write_jsonl(run_dir / "slm_ensemble_timing.jsonl", [
        {"stage": "aggregate", "n_reports": len(report_ids), "n_active_models": len(active),
         "n_detection_rows": sum(1 for _ in open(merged, encoding="utf-8")), "duration_s": 0.4},
    ])
    (run_dir / "run.log").write_text(
        "2026-01-01 12:00:00 | INFO | Aggregate done | 4 reports x %d models\n" % len(active))

    # Guard the invariant the driver itself asserts, so a fixture edit that breaks monotonicity
    # fails here, not inside a UI test that has nothing to do with it.
    totals = [r["n_predicted"] for r in summary_rows if r["rule"].startswith("vote_k")]
    assert totals == sorted(totals, reverse=True), f"k-sweep is not monotone: {totals}"
    assert _safe_micro_macro([gold[r] for r in report_ids],
                             [set() for _ in report_ids])["micro_f1"] == 0.0
    return run_dir


def build_slm_ensemble_tree(root: Path, cohorts: tuple[str, ...] = ("hcy", "gsc"),
                            models: tuple[str, ...] = SLM_MODELS) -> Path:
    """``<root>/phenojury_generation_free_listing/<cohort>/`` for each cohort, the UI's discovery layout."""
    for cohort in cohorts:
        build_slm_ensemble_run(Path(root) / "phenojury_generation_free_listing" / cohort, models)
    return Path(root)


def build_prompt_screen_tree(root: Path, exp_id: str = "phenojury_generation_other_prompts",
                             cohorts: tuple[str, ...] = ("hcy",),
                             prompts: tuple[str, ...] = ("q4_span_json", "q7_recall"),
                             models: tuple[str, ...] = SLM_MODELS) -> Path:
    """``<root>/<exp_id>/<cohort>/<prompt_key>/``, the earlier layout, one cell per prompt.

    The cells hold the Free Listing generation run's artifacts because that is literally what ``hpo_extraction.phenojury.generation_prompts`` writes
    into them: it calls the same ``_write_rule`` and ``write_slm_metrics``, one directory deeper.
    Reusing :func:`build_slm_ensemble_run` is therefore not a shortcut, a fixture that wrote its
    own approximation of the layout would stop testing the thing that makes the import work.

    Three real deltas are applied afterwards, and each is one the UI has to cope with:

    * the per-model metrics are named ``slm_metrics.csv``;
    * there is no timing JSONL, earlier logs timing to MLflow only;
    * ``prompt_diagnostics.csv`` is present, and ``prompt_screen_ranking.csv`` sits one level up.
    """
    root = Path(root)
    for cohort in cohorts:
        cohort_dir = root / exp_id / cohort
        for prompt_key in prompts:
            cell = build_slm_ensemble_run(cohort_dir / prompt_key, models)
            (cell / "slm_ensemble_slm_metrics.csv").rename(cell / "slm_metrics.csv")
            (cell / "slm_ensemble_timing.jsonl").unlink()
            with open(cell / "prompt_diagnostics.csv", "w", encoding="utf-8", newline="") as out:
                out.write("prompt_key,model,n_records,n_wrote,echo_rate,label_exactness,"
                          "negation_trap_rate,format_compliance\n")
                for model in models:
                    # label_exactness is empty for a sentence-shaped prompt: the
                    # metric does not apply, and writing 0.0 would report a failure that did not
                    # happen. The UI has to render it as ", ".
                    exactness = "0.392" if prompt_key == "q4_span_json" else ""
                    out.write(f"{prompt_key},{model},4,4,0.12,{exactness},0.0,0.98\n")
        with open(cohort_dir / "prompt_screen_ranking.csv", "w", encoding="utf-8",
                  newline="") as out:
            out.write("prompt_key,prompt_change,n_active_models,best_rule,best_micro_f1\n")
            for prompt_key in prompts:
                out.write(f"{prompt_key},a change,{len(models)},vote_k1,0.5\n")
    return root


# ══════════════════════════════════════════════════════════════════════════════
# The PhenoBERT baseline, the PhenoBERT-standalone baseline, over the *same* reports
# ══════════════════════════════════════════════════════════════════════════════
# ``build_exp13_tree``'s the PhenoBERT baseline branch above writes the artifacts ``result_tables`` reads. The deep-dive
# UI reads more: it puts the report text on screen with these offsets underlined in it, so the text
# has to exist and the offsets have to be real. This builder writes the whole thing for the reports
# :data:`SLM_SENTENCES` defines, which is what makes the two experiments comparable report by report
#, the only way to check the comparison page's arithmetic by hand.
#
# The scenario puts one term on each side of the four-way agreement split, because a
# fixture where the two methods agree everywhere cannot fail the split's arithmetic:
#
# =========  ======  ================  =============================================================
# report     term    who recovers it   why
# =========  ======  ================  =============================================================
# ``r1``     ``C``   both              "seizures" is in the report and every model writes it
# ``r1``     ``D``   ensemble only     phi4 reads anxiety into sentence 1. The word is not in the text
# ``r1``     ``F``   PhenoBERT only    "cerebral atrophy" is in the report, but deepseek's only
#                                      mention of it is inside a ``<think>`` block that is stripped
# ``r2``     ``D``   both              "anxiety" is in the report and three models write it
# ``r2``     ``M``   neither           both find the phrase and both see it negated
# ``r4``     ``H``   neither           nothing mentions the vertebral column at all
# =========  ======  ================  =============================================================

#: ``{report_id: report text}``, the sentences joined, so ``align_sentences`` has an exact answer.
PB_REPORTS: dict[str, str] = {rid: " ".join(sents) for rid, sents in SLM_SENTENCES.items()}

#: ``{report_id: [(phrase, hpo_id, raw_hpo_id, score, negated)]}``. Offsets are found by searching
#: for the phrase in :data:`PB_REPORTS`, so a phrase that is not literally there fails loudly here
#:, not producing a detection that points at the wrong words.
PB_DETECTIONS: dict[str, list[tuple[str, str, str, float, bool]]] = {
    "r1": [
        ("seizures", C, OBSOLETE_C, 0.97, False),      # alt id, remapped onto C
        ("fever", G, G, 0.74, True),                   # negated → never a prediction
        ("cerebral atrophy", F, F, 0.92, False),        # The annotated term the ensemble cannot reach
    ],
    "r2": [
        ("anxiety", D, D, 0.93, False),
        ("spinal cord abnormality", M, M, 0.81, True),  # negated → M is missed by both methods
    ],
    "r3": [],                                           # predicts nothing, still gets a summary line
    "r4": [
        ("good condition", B, B, 0.66, False),          # a false positive only this method makes
        ("Discharged", HALLUCINATION, HALLUCINATION, 0.55, False),   # not in this HPO release
    ],
}


def phenobert_predicted(report_id: str) -> list[str]:
    """What ``phenobert_experiment.predicted_sets`` would keep: positive and resolved only."""
    return sorted({hpo for _p, hpo, _raw, _s, neg in PB_DETECTIONS.get(report_id, [])
                   if not neg and hpo != HALLUCINATION})


def build_phenobert_standalone_run(run_dir: Path) -> Path:
    """Write a complete the PhenoBERT baseline run directory under *run_dir*, and return it.

    Mirrors ``src/hpo_extraction/baselines/phenobert_baseline.py``: ``stage_reports`` writes the report verbatim into
    ``phenobert_input/<stem>.txt``, ``collect_detections`` dumps one row per detection with offsets
    into that file, and ``write_predictions`` writes the per-term rows plus one authoritative
    ``summary`` line per report, including for reports that predicted nothing.
    """
    run_dir = Path(run_dir)
    (run_dir / "phenobert_input").mkdir(parents=True, exist_ok=True)
    (run_dir / "phenobert_output").mkdir(parents=True, exist_ok=True)
    report_ids = sorted(PB_REPORTS)

    detections: list[dict] = []
    for rid in report_ids:
        text = PB_REPORTS[rid]
        stem = rid.replace(":", "_")
        (run_dir / "phenobert_input" / f"{stem}.txt").write_text(text, encoding="utf-8")

        tsv_rows: list[str] = []
        for phrase, hpo_id, raw_hpo_id, score, negated in PB_DETECTIONS.get(rid, []):
            start = text.find(phrase)
            if start < 0:
                raise AssertionError(
                    f"fixture is inconsistent: report {rid} does not contain {phrase!r}, so the "
                    "offsets in PB_DETECTIONS would point at the wrong text")
            end = start + len(phrase)
            detections.append({
                "report_id": rid,
                "hpo_id": hpo_id,
                "hpo_label": NAMES.get(hpo_id, hpo_id),
                "raw_hpo_id": raw_hpo_id,
                "resolved": hpo_id != HALLUCINATION,
                "phrase": phrase,
                "start": start,
                "end": end,
                "score": score,
                "negated": negated,
            })
            fields = [str(start), str(end), phrase, raw_hpo_id, f"{score:.4f}"]
            if negated:
                fields.append("Neg")
            tsv_rows.append("\t".join(fields))
        (run_dir / "phenobert_output" / f"{stem}.txt").write_text(
            "\n".join(tsv_rows) + ("\n" if tsv_rows else ""), encoding="utf-8")

    _write_jsonl(run_dir / "phenobert_detections.jsonl", detections)

    rows: list[dict] = []
    for rid in report_ids:
        gold = sorted(SLM_GOLD.get(rid, []))
        codes = phenobert_predicted(rid)
        rows += [{"report_id": rid, "hpo_id": code, "hpo_label": NAMES.get(code, code),
                  "prediction": 1, "ground_truth": int(code in gold)} for code in codes]
        rows.append({"report_id": rid, "summary": True,
                     "predicted_set": codes, "gold_set": gold})
    _write_jsonl(run_dir / "phenobert_predictions.jsonl", rows)

    _write_jsonl(run_dir / "phenobert_timing.jsonl", [
        {"stage": "annotate", "n_reports": len(report_ids), "n_detections": len(detections),
         "n_predictions": sum(len(phenobert_predicted(r)) for r in report_ids),
         "annotate_s": 3.0, "duration_s": 3.4,
         "mean_time_per_report_s": 3.4 / len(report_ids)},
    ])
    (run_dir / "run.log").write_text(
        "2026-01-01 12:00:00 | INFO | annotate done | %d reports\n" % len(report_ids))
    return run_dir


def build_comparison_tree(root: Path, cohorts: tuple[str, ...] = ("hcy",),
                          models: tuple[str, ...] = SLM_MODELS) -> Path:
    """Both experiments side by side, in the layout the deep-dive UI derives the baseline from.

    ``<root>/phenojury_generation_free_listing/<cohort>/`` and ``<root>/baseline_phenobert/<cohort>/``, which
    is the cluster's own layout, and therefore the one ``pbstandalone.find_run`` has to work on.
    """
    root = Path(root)
    for cohort in cohorts:
        build_slm_ensemble_run(root / "phenojury_generation_free_listing" / cohort, models)
        build_phenobert_standalone_run(root / "baseline_phenobert" / cohort)
    return root


# ──────────────────────────────────────────────────────────────────────────────
# Tree members over the SLM cohort, an earlier exploratory run's pool
#
# ``build_exp13_tree``'s tree fixtures live on the two-report Ground truth scenario. An earlier exploratory run pools them
# beside the ensemble and PhenoBERT, which are written over the four-report :data:`SLM_GOLD`
# scenario, so the tree members need writing on *that* report set: a member covering half the
# cohort is what G6 exists to reject, and a fixture that trips it cannot also test the
# happy path.
# ──────────────────────────────────────────────────────────────────────────────

#: ``{variant: {operating point: {report: predicted terms}}}``, judged against :data:`SLM_GOLD`
#: (``r1``: C, D, F · ``r2``: D, M · ``r3``: none · ``r4``: H, six annotated pairs).
#:
#: Each variant's **first** point is its argmax, and the runner-up is a looser point that trades
#: precision away faster than it buys recall. That ordering is what makes the G7 scan testable:
#: pointing the config at the second entry must fail, not merely score worse.
TREE_POOL_POINTS: dict[str, dict[str, dict[str, list[str]]]] = {
    "tree_gate_lr": {
        # P 1.000 R 0.667 F1 0.800
        "tau_0.016_acc_0.95": {"r1": [C, D], "r2": [D], "r3": [], "r4": [H]},
        # P 0.500 R 1.000 F1 0.667
        "tau_0.0045_acc_0.95": {"r1": [C, D, F, A, B], "r2": [D, M, A], "r3": [A], "r4": [H, A]},
    },
    "tree_lse_beta1": {
        # P 1.000 R 0.667 F1 0.800
        "tau_0.00015_acc_0.95": {"r1": [C, F], "r2": [M], "r3": [], "r4": [H]},
        # P 0.400 R 0.667 F1 0.500
        "tau_0.00015_acc_0.5": {"r1": [C, F, A, B, E], "r2": [M, A], "r3": [A], "r4": [H, B]},
    },
    "tree_noisyor": {
        # P 1.000 R 0.333 F1 0.500
        "tau_0.3": {"r1": [C], "r2": [D], "r3": [], "r4": []},
        # P 0.400 R 0.333 F1 0.364
        "tau_0.7": {"r1": [C, A], "r2": [D, A], "r3": [], "r4": [A]},
    },
    # A single-point sweep, like the real an earlier exploratory run: the argmax is the only choice.
    "tree_no_prune": {
        "tau_0.5": {"r1": [D, A], "r2": [M, B], "r3": [A], "r4": []},
    },
}

#: ``{exp_id: variant}`` for the four tree members, in an earlier exploratory run's ``tree_pool`` order.
TREE_POOL_EXPERIMENTS: dict[str, str] = {
    "exp13_00_tree_gate_lr": "tree_gate_lr",
    "exp13_01_tree_noisyor": "tree_noisyor",
    "exp13_02_tree_no_prune": "tree_no_prune",
    "exp13_09_tree_lse_beta1": "tree_lse_beta1",
}

#: Per-node accept probabilities for the pooled tree fixtures. Distinct per term so that
#: rank-normalising them within a report produces a strict order, not ties.
TREE_POOL_ACCEPT: dict[str, float] = {
    A: 0.12, B: 0.24, C: 0.97, D: 0.81, E: 0.05, F: 0.63, G: 0.02, H: 0.55, M: 0.44,
}

#: Ontology depth per term. ``_depth`` covers only the seven nodes of the two-report scenario;
#: :data:`SLM_GOLD` also reaches ``H`` (under G) and ``M`` (under both B and H).
TREE_POOL_DEPTH: dict[str, int] = {A: 1, G: 1, B: 2, E: 2, H: 2, C: 3, D: 3, F: 3, M: 3}


def write_tree_pool_run(run_dir: Path, variant: str,
                        points: dict[str, dict[str, list[str]]] | None = None) -> Path:
    """One tree variant's sweep over the SLM cohort: predictions and nodes per configuration.

    The nodes file carries **more** rows than the predictions, every predicted term
    plus one rejected node per report. An earlier exploratory run reads ``accept_score`` from it and must keep only
    the rows that survived into the predicted set, so a fixture with no rejected nodes could not
    tell a correct filter from a missing one.
    """
    run_dir = Path(run_dir)
    for point, predicted in (points or TREE_POOL_POINTS[variant]).items():
        write_predictions(run_dir / point / f"{variant}_predictions.jsonl", predicted, SLM_GOLD)
        records = []
        for rid in sorted(SLM_GOLD):
            accepted = list(predicted.get(rid, []))
            # G is never predicted at any point, so it is the rejected node in every report.
            for hpo_id in accepted + [G]:
                records.append({
                    "report_id": rid, "hpo_id": hpo_id, "depth": TREE_POOL_DEPTH.get(hpo_id, 3),
                    "prune_score": PRUNE_SCORE.get(hpo_id, 0.5),
                    "accept_score": TREE_POOL_ACCEPT.get(hpo_id, 0.5),
                    "expanded": True, "accepted": hpo_id in accepted,
                    "is_gold": int(hpo_id in set(SLM_GOLD.get(rid, []))),
                })
        _write_jsonl(run_dir / point / f"{variant}_nodes.jsonl", records)
    return run_dir


def build_tree_pool_members(root: Path, cohorts: tuple[str, ...] = ("hcy", "gsc")) -> Path:
    """All four tree variants, per cohort, in the cluster's ``<exp_id>/<cohort>/<tau>/`` layout."""
    root = Path(root)
    for cohort in cohorts:
        for exp_id, variant in TREE_POOL_EXPERIMENTS.items():
            write_tree_pool_run(root / exp_id / cohort, variant)
    return root
