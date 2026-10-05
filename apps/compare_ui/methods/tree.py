"""TreePhenoRAG: two scores against two thresholds, and -- for a miss -- the ancestor that blocked it.

TreePhenoRAG walks the ontology from the layer-1 roots, expanding a node when its pooled verifier
score clears ``tau_prune`` and accepting it when a second pooling clears ``tau_accept``. So its
answer about one term is four numbers and a path: what the verifier said about this node, what the
traversal needed to reach it at all, and which of the two gates it failed.

**The whole reasoning comes from the ``.npz`` cache, and no ``*_calls.jsonl`` is opened.** That is
not an optimisation, it is the difference between a build that runs and one that does not: the
calls files are 4.3 GB across 40 shards, about 39 MB per report. ``ingest_score_cache`` already
folded them into a columnar archive that keeps the four columns ``load_score_cache`` discards --
``cosine``, ``sent_index``, ``captured_mass`` and the timings -- and ``sent_index`` is
what turns a margin back into "the verifier was shown *this* sentence and said No". Everything this
panel shows is a slice of ``ReportCache``.

**The configuration is the fold's, not the modal one.** ``selected_configuration.json`` reports what
the protocol chose most often. It is a summary, not the thing that produced any particular
prediction. ``nested_evaluation`` pools out-of-fold predictions from **repetition 0 only**, so the
configuration behind report *i* is the one chosen at ``(repetition=0, outer_fold=f)`` where *i* was
evaluated -- and on HCY those genuinely differ: fold 0 selected the ``exemplar`` index and the rest
selected ``ontology_r3``. Explaining fold 0's prediction with fold 1's retrieval would be explaining
a different run.

The acceptance predicate is ``ReplaySpace.predictions``' own, character for character::

    visited_at(r, tau_prune) & (accept >= tau_accept) & in_cache

including the ``in_cache`` term. A node the cache lacks is not accepted however it scores, and
dropping that conjunct here would make this panel disagree with the prediction set it is explaining.
"""

from __future__ import annotations

import csv
import gc
import logging
import os

import numpy as np

from apps.compare_ui.methods import base

logger = logging.getLogger(__name__)

#: Only repetition 0 contributes to the pooled prediction set. See the module docstring.
POOLED_REPETITION = 0

#: How many of a node's retrieved segments to carry into a bundle. S is 10 at the selected
#: configuration, so this keeps all of them. The cap exists so a future S of 50 cannot quietly
#: multiply every bundle by five.
MAX_CALLS = 12

#: ``ingest_score_cache`` names the archive after the retrieval index it was built from.
CACHE_FILE = "cache_{index}.npz"


class TreeAdapter(base.Adapter):
    """Reads TreePhenoRAG predictions and the stored verifier scores, and places each term on its best-scoring segment."""
    key = "treephenorag"

    def load(self, report_ids) -> None:
        """Read the TreePhenoRAG predictions and stored scores of *report_ids*. Sets ``status`` to ``missing`` when absent."""
        paths = self.ctx.paths
        cache_dir = paths.exp(self.method.extra["cache"])

        predictions = self.method.predictions_for(self.ctx.cohort)
        if not predictions:
            self.missing("no prediction set is declared for cohort " + self.ctx.cohort.key)
            return
        predictions = os.path.join(paths.output_base, predictions)
        self._predictions = base.load_prediction_sets(predictions)
        if not self._predictions:
            self.missing(predictions)
            return
        self.ctx.note_input("treephenorag_predictions", predictions)

        self._folds = _fold_of_report(paths, predictions, report_ids)
        choices_path = os.path.join(cache_dir, "tables", "selection_choices.csv")
        self._choices = _selection_choices(choices_path)
        self.ctx.note_input("treephenorag_selection", choices_path)
        if not self._folds:
            self.note = (
                "this cohort was not folded -- the TreePhenoRAG protocol applies the configuration selected on all "
                "of HCY unchanged, so there is no per-fold threshold to read a score against and "
                "no cache to re-run here. The verdict is the transferred prediction set.")
        elif not self._choices:
            self.note = ("no selection_choices.csv, so the fold thresholds are unknown and the "
                         "scores below cannot be read against them")

        self._graph = _graph(self.ctx.view)
        self._per_report = {}
        if self._graph is None:
            self.note = "the ontology graph is unavailable, so the re-run cannot be reconstructed"
            return

        for index, wanted in sorted(_reports_by_index(self._folds, self._choices,
                                                      report_ids).items()):
            path = os.path.join(cache_dir, CACHE_FILE.format(index=index))
            if not os.path.isfile(path):
                logger.warning("no %s; %d report(s) lose their tree reasoning",
                               path, len(wanted))
                continue
            self.ctx.note_input("treephenorag_cache_" + index, path)
            self._offline_index(index, path, wanted)

    def _offline_index(self, index: str, path: str, wanted) -> None:
        """Re-run one retrieval index for the reports that were scored under it.

        One index at a time, and the caches dropped as soon as their reports are reduced: the
        archive's ``margin`` array is every node of every report and materialises in full inside
        ``np.load``. Holding two indices at once doubles a peak that is already the largest thing
        this builder allocates.
        """
        from hpo_extraction.treephenorag import stored_scores as stored_scores

        caches = stored_scores.load_cache_npz(path, report_ids=sorted(wanted))
        for rid in sorted(wanted):
            cache = caches.get(rid)
            if cache is None:
                logger.warning("%s is not in the %s cache", rid, index)
                continue
            config = self._config_for(rid)
            try:
                self._per_report[rid] = _offline(self._graph, cache, config)
            except Exception as exc:                       # pragma: no cover - defensive
                logger.warning("%s: tree re-run failed (%s)", rid, exc)
        del caches
        gc.collect()

    def evidence(self, rv, outcomes, gold_rows):
        """``(marks, reasons)`` of one report, with the segment scores behind each term."""
        state = self._per_report.get(rv.report_id)
        config = self._config_for(rv.report_id)

        placed = {}
        reasons = {}
        if state is None:
            why = (self.note or "no re-run for this report -- the score cache for the fold's "
                                "retrieval index was not readable")
            for code, outcome in outcomes.items():
                reasons[code] = {"kind": "tree", "config": config, "available": False,
                                 "why": why}
            return self.marks_from(outcomes, placed, gold_rows), reasons

        sentences = {}
        for code, outcome in outcomes.items():
            evidence_at = (gold_rows.get(code) or {}).get("segment_idx")
            reason = _reason(self.ctx.view, self._graph, state, config, code, outcome, rv,
                             evidence_at)
            reasons[code] = reason
            sentences[code] = reason.get("segment_idx")
        placed = base.place_by_sentence(rv, sentences)
        return self.marks_from(outcomes, placed, gold_rows), reasons

    def _config_for(self, report_id: str) -> dict:
        fold = self._folds.get(report_id)
        chosen = self._choices.get(fold) if fold is not None else None
        if chosen is None:
            return {"fold": fold, "known": False}
        return dict(chosen, fold=fold, known=True)


# -- the re-run --------------------------------------------------------------

def _graph(view):
    """The traversal DAG below the layer-1 roots. Ontology-only, so built once per process."""
    from apps.compare_ui import sources
    from hpo_extraction.treephenorag import stored_scores as stored_scores

    try:
        children_map, roots, _depths = sources.get_graph(view)
    except Exception as exc:                               # pragma: no cover - env-dependent
        logger.warning("cannot build the traversal graph (%s)", exc)
        return None
    return stored_scores.build_traversal_graph(children_map, roots)


def _offline(graph, cache, config: dict) -> dict:
    """One report's expansion and acceptance state at one configuration.

    Everything here is the reference implementation's own call sequence: pool, align, bottleneck,
    threshold. Nothing is approximated, so the ``accepted`` set this produces is the prediction set
    ``treephenorag_protocol`` wrote -- which is what makes the panel an explanation rather than a reconstruction
    that happens to look similar.
    """
    from hpo_extraction.treephenorag import stored_scores as stored_scores

    s = config.get("S")
    pool_pr = config.get("pool_pr") or "P1"
    pool_acc = config.get("pool_acc") or "lse_beta1"

    prune_scores = stored_scores.align_to_graph(
        graph, cache.node_ids, stored_scores.POOLINGS[pool_pr](cache, s))
    accept_scores = stored_scores.align_to_graph(
        graph, cache.node_ids, stored_scores.POOLINGS[pool_acc](cache, s))
    r = stored_scores.bottleneck_scores(graph, prune_scores)
    in_cache = stored_scores.in_cache_mask(graph, cache.node_ids)

    tau_prune = config.get("tau_prune")
    tau_accept = config.get("tau_accept")
    if tau_prune is None or tau_accept is None:
        visited = expanded = accepted = np.zeros(graph.n_nodes, dtype=bool)
    else:
        visited = stored_scores.visited_at(r, float(tau_prune))
        expanded = stored_scores.expanded_at(r, prune_scores, float(tau_prune))
        accepted = visited & (accept_scores >= float(tau_accept)) & in_cache

    return {
        "prune_scores": prune_scores, "accept_scores": accept_scores, "r": r,
        "in_cache": in_cache, "visited": visited, "expanded": expanded, "accepted": accepted,
        "cache": cache, "s": s,
    }


def _reason(view, graph, state, config, code, outcome, rv, evidence_at=None) -> dict:
    """What TreePhenoRAG can say about one term, and what it can say about losing one."""
    resolved = (view.resolve(code) or code) if view is not None else code
    slot = graph.index.get(resolved)

    reason = {
        "kind": "tree",
        "source": "cache_{}.npz (ingested from *_calls.jsonl)".format(config.get("index") or "?"),
        "config": config,
        "available": True,
        "hpo_id": resolved,
        "in_graph": slot is not None,
    }
    if slot is None:
        reason["why"] = ("outside the traversal graph -- this term is not under a layer-1 "
                         "phenotypic-abnormality root, so no walk can reach it")
        reason["segment_idx"] = None
        return reason

    reason.update({
        "depth": int(graph.depth[slot]),
        "prune_score": _f(state["prune_scores"][slot]),
        "accept_score": _f(state["accept_scores"][slot]),
        "bottleneck_r": _f(state["r"][slot]),
        "visited": bool(state["visited"][slot]),
        "expanded": bool(state["expanded"][slot]),
        "accepted": bool(state["accepted"][slot]),
        "in_cache": bool(state["in_cache"][slot]),
    })

    calls = _calls(state["cache"], resolved, rv, state["s"])
    reason["calls"] = calls
    reason["segment_idx"] = calls[0]["sent_index"] if calls else None

    if outcome == "fn":
        reason.update(_attribution(view, state, graph, resolved, config, evidence_at))
        reason["why"] = _why_missed(reason)
    else:
        reason["why"] = _why_found(reason, config)
    return reason


def _calls(cache, hpo_id: str, rv, s) -> list:
    """The verifier calls behind one node, best margin first, with the sentence resolved.

    Sorted by margin, not by retrieval rank: the reader's question is "what was the best
    case for this term", and rank order buries it under nine Nos when the retrieval was poor.
    ``rank`` is carried so the retrieval order is still readable.
    """
    position = cache.index.get(hpo_id)
    if position is None:
        return []
    width = cache.s_max if not s else min(int(s), cache.s_max)
    valid = np.asarray(cache.mask[position, :width], dtype=bool)
    if not valid.any():
        return []

    margins = np.asarray(cache.margins[position, :width], dtype=float)
    cosine = _row(cache.cosine, position, width)
    sents = _row(cache.sent_index, position, width)
    mass = _row(cache.captured_mass, position, width)

    rows = []
    for rank in np.flatnonzero(valid):
        sent = None if sents is None else int(sents[rank])
        rows.append({
            "rank": int(rank) + 1,
            "sent_index": sent,
            "sentence": rv.display[sent] if sent is not None and 0 <= sent < rv.n_segments else "",
            "margin": _f(margins[rank]),
            "verdict": "Yes" if margins[rank] > 0 else "No",
            "cosine_sim": None if cosine is None else _f(cosine[rank]),
            "captured_mass": None if mass is None else _f(mass[rank]),
        })
    rows.sort(key=lambda row: (-(row["margin"] if row["margin"] is not None else -1e9),
                               row["rank"]))
    return rows[:MAX_CALLS]


def _row(array, position: int, width: int):
    if array is None:
        return None
    return np.asarray(array[position, :width])


def _attribution(view, state, graph, hpo_id: str, config, evidence_at=None) -> dict:
    """Which stage lost this annotated term: the bucket, the blocking ancestor, and the cause there.

    ``attribute_false_negative`` names *where* the traversal stopped; ``_cause_at`` names *why* it
    stopped there -- retrieval never put a relevant sentence in front of the verifier, the verifier
    said No to all of them, the pooling failed to carry them over the threshold, or none of those
    (residual). Both come from ``hpo_extraction.evaluation.metrics.recall_decomposition``, including the
    private one: the three-way test is the definition the recall-decomposition table is built on,
    and a second copy of it here would let this panel and that table disagree about the same miss.

    Two arguments have to be the right ones or the bucket is quietly meaningless, and the first
    real run got both wrong:

    ``pooled`` / ``threshold`` are the **acceptance** score and ``tau_accept``, matching
    ``recall_decomposition``'s own signature (*"pooled_by_report: {report: {hpo: accept score}}"*).
    Passing the pruning score and ``tau_prune`` instead makes every reached-but-rejected term
    ``residual`` -- by design, since a visited node's prune score always clears ``tau_prune``
    -- which is the pooling bucket emptied into a word that means "we do not know".

    ``evidence`` is the segment the curator read the term at, and it is the only thing that makes
    *retrieval* decidable: without it a term whose ten retrieved sentences never included the one
    carrying the finding is indistinguishable from one the verifier simply rejected. It applies
    only when the attribution point is the term itself. For a pruned term the point is an ancestor,
    and the curated ground truth says nothing about where an ancestor should have been found.
    """
    # Imported by name, not as a module: ``thesis_metrics`` re-exports a *function* called
    # ``recall_decomposition``, so ``import ...recall_decomposition as rd`` binds the function and
    # not the module it lives in.
    from hpo_extraction.evaluation.metrics.recall_decomposition import (
        _cause_at, attribute_false_negative)

    names = np.asarray(graph.node_ids)
    scored = set(names[state["visited"]].tolist())
    expanded = set(names[state["expanded"]].tolist())
    out = dict(attribute_false_negative(hpo_id, scored, expanded, view))

    cache = state["cache"]
    at = out.get("lost_at")
    position = cache.index.get(at) if at else None
    width = cache.s_max if not state["s"] else min(int(state["s"]), cache.s_max)
    slot = graph.index.get(at) if at else None

    at_the_term = bool(at) and at == hpo_id
    evidence = (frozenset({int(evidence_at)})
                if at_the_term and evidence_at is not None else None)
    out["evidence_available"] = evidence is not None

    out["cause"] = _cause_at(
        position=position,
        margins_row=None if position is None else cache.margins[position, :width],
        mask_row=None if position is None else cache.mask[position, :width],
        sent_row=None if position is None or cache.sent_index is None
        else cache.sent_index[position, :width],
        evidence=evidence,
        pooled=None if slot is None else float(state["accept_scores"][slot]),
        threshold=None if config.get("tau_accept") is None else float(config["tau_accept"]),
        delta_m=0.0,
    )
    out["bucket"] = "pruning" if out.get("pruned") else out["cause"]
    if at:
        out["lost_at_label"] = _label(view, at)
        out["lost_at_score"] = None if slot is None else _f(state["prune_scores"][slot])
    # The blocking ancestor carries no evidence of its own when it was never scored, so the panel
    # asks for its calls only where there are some to show.
    out["lost_at_in_cache"] = bool(slot is not None and state["in_cache"][slot])
    return out


def _why_missed(reason: dict) -> str:
    config = reason.get("config") or {}
    tau_prune, tau_accept = config.get("tau_prune"), config.get("tau_accept")
    if not reason.get("visited"):
        at = reason.get("lost_at_label") or reason.get("lost_at")
        if not at:
            return "never reached -- no path from a root stayed above tau_prune"
        return ("pruned before it was ever scored: the walk stopped at {} (depth {}), whose own "
                "score {} did not clear tau_prune {}".format(
                    at, reason.get("blocking_depth"), _fmt(reason.get("lost_at_score")),
                    _fmt(tau_prune)))
    if not reason.get("accepted"):
        return ("scored but not accepted: acceptance {} against tau_accept {}".format(
            _fmt(reason.get("accept_score")), _fmt(tau_accept)))
    return "accepted by the re-run but absent from the pooled prediction set"


def _why_found(reason: dict, config: dict) -> str:
    if not reason.get("accepted"):
        return ("predicted but not accepted by this re-run -- the pooled set and the fold "
                "configuration disagree, which is worth investigating")
    return "accepted: {} >= tau_accept {}, reached at bottleneck {}".format(
        _fmt(reason.get("accept_score")), _fmt(config.get("tau_accept")),
        _fmt(reason.get("bottleneck_r")))


# -- configuration -----------------------------------------------------------

def _fold_of_report(paths, predictions: str, report_ids) -> dict:
    """``{report_id: outer_fold}``, or ``{}`` where the cohort was never folded.

    A **transfer** cohort has no folds by design: the TreePhenoRAG protocol applies the configuration chosen
    on all of HCY to the abstracts unchanged, so there is no outer fold a document was held out of
    and no per-fold threshold to read its score against. Returning an empty map, not
    inventing fold 0 is what makes that visible downstream instead of attributing this cohort's
    predictions to a selection that never ran on it.
    """
    from apps.compare_ui.methods import phenojury

    if paths.cohort_obj.key != "hcy":
        return {}
    return phenojury._fold_of_report(paths, "", predictions, report_ids)


def _selection_choices(path: str) -> dict:
    """``{outer_fold: configuration}`` at repetition 0, from the TreePhenoRAG protocol's own table."""
    out: dict = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _int(row.get("repetition")) != POOLED_REPETITION:
                continue
            fold = _int(row.get("outer_fold"))
            if fold is None:
                continue
            out[fold] = {
                "index": str(row.get("retrieval_index") or ""),
                "pool_pr": str(row.get("pool_pr") or "P1"),
                "tau_prune": _f(row.get("tau_prune")),
                "pool_acc": str(row.get("pool_acc") or "lse_beta1"),
                "tau_accept": _f(row.get("tau_accept")),
                "S": _int(row.get("S")),
                "inner_f1": _f(row.get("inner_f1")),
            }
    return out


def _reports_by_index(folds: dict, choices: dict, report_ids) -> dict:
    """``{retrieval_index: {report_id, ...}}`` -- which cache each report has to be re-run from."""
    out: dict = {}
    for rid in report_ids:
        fold = folds.get(rid)
        config = choices.get(fold) if fold is not None else None
        index = (config or {}).get("index")
        if index:
            out.setdefault(index, set()).add(rid)
    return out


# -- small helpers -----------------------------------------------------------

def _label(view, hpo_id: str) -> str:
    from apps.treephenorag_ui import terms

    return terms.label(view, hpo_id) if view is not None else hpo_id


def _f(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    if out in (float("inf"), float("-inf")):
        return None          # +/-inf is a root or an unreachable node; JSON has no spelling for it
    return out


def _fmt(value) -> str:
    if value is None:
        return "?"
    if abs(value) >= 0.01:
        return "{:.4g}".format(value)
    return "{:.3e}".format(value)


def _int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
