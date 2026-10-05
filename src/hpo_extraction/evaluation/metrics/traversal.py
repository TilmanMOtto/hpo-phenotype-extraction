"""§Reachability recall (eq. 6), §Blocking depth distribution, §Traversal cost by depth.

"Traversal-based classifiers suffer error propagation: a pruning decision at an internal node
renders its entire subtree unreachable, irrespective of the identification stage's competence at the
leaf. This upper-bounds recall independently of :math:`\\tau_{\\text{accept}}`."

These three diagnostics decompose the system's recall failures into the part attributable to
*traversal* and the part attributable to *identification*, and localise the traversal part to a
depth.

**Inputs are plain dicts, not files.** Everything here takes ``children_map``, ``roots`` and a
per-report ``expanded`` set. That is the content of the ``*_nodes.jsonl`` artifacts written
by ``hpo_extraction.treephenorag.score_store._write_tau`` (fields ``hpo_id``, ``depth``, ``expanded``, ``accepted``,
``is_gold``), but no module in this package reads a file.

**Reachability in a DAG is set membership.** Eq. (6) is written as a product of indicators over
:math:`\\mathrm{Path}(y)`, with the note that "in a DAG a term is reachable if *any* root-to-term
path survives, so the product is taken over the surviving path of least resistance". Taking the max
over paths of that product is the same as asking whether the breadth-first search reaches the term
at all, so :func:`reachable_set` computes the search once and :func:`reachability_recall` is a set
intersection. That equivalence is the reason no path enumeration appears in this module.

**Blocking depth uses the deepest severing node.** An annotated term usually has several root-to-term
paths, *all* of which must be severed for it to be missed. This package records the depth of the
first non-expanded node on the path the traversal got furthest along, the "path of least
resistance" that eq. (6) already privileges. It is computed by a dynamic program over the DAG
(:math:`O(V + E)`). Enumerating paths is not viable, since a mid-ontology HPO term can have
thousands of distinct root-to-term paths.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Iterable, Mapping, Sequence

_INF = float("inf")


# ── Graph preparation ────────────────────────────────────────────────────────

def parents_from_children(children_map: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Reverse a ``{node: [children]}`` map into ``{node: [parents]}``."""
    parents: dict[str, list[str]] = {n: [] for n in children_map}
    for node, children in children_map.items():
        for child in children:
            parents.setdefault(child, []).append(node)
    return parents


def bfs_depths(children_map: Mapping[str, Sequence[str]], roots: Iterable[str]) -> dict[str, int]:
    """Breadth-first depth from the roots, with ``roots`` at depth 1.

    Matches ``hpo_extraction.treephenorag.traversal.traverse``, which seeds ``depth_of[r] = 1`` for the layer-1 organ
    systems, so a depth computed here lines up with the ``depth`` field in the run artifacts.
    """
    depths: dict[str, int] = {}
    queue: deque[str] = deque()
    for r in roots:
        if r not in depths:
            depths[r] = 1
            queue.append(r)
    while queue:
        node = queue.popleft()
        for child in children_map.get(node, ()):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def topological_order(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Kahn topological order over the nodes reachable from ``roots`` ignoring pruning.

    Returns ``(order, parents_within)`` where ``parents_within`` is restricted to the reachable
    induced subgraph, so a node's in-degree counts only parents that can themselves be reached.

    Public because :mod:`hpo_extraction.treephenorag.stored_scores` builds its bottleneck-reachability sweep on this
    order: a node's parents all precede it, which is what lets one pass compute the whole
    ``tau_prune`` axis. ``apps/treephenorag_ui/pruning.py`` keeps a copy only because this was private.
    """
    reachable: set[str] = set()
    queue: deque[str] = deque()
    for r in roots:
        if r not in reachable:
            reachable.add(r)
            queue.append(r)
    while queue:
        node = queue.popleft()
        for child in children_map.get(node, ()):
            if child not in reachable:
                reachable.add(child)
                queue.append(child)

    parents_within: dict[str, list[str]] = {n: [] for n in reachable}
    in_degree: dict[str, int] = {n: 0 for n in reachable}
    for node in reachable:
        for child in children_map.get(node, ()):
            if child in reachable:
                parents_within[child].append(node)
                in_degree[child] += 1

    ready: deque[str] = deque(n for n in reachable if in_degree[n] == 0)
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for child in children_map.get(node, ()):
            if child in in_degree:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

    if len(order) != len(reachable):
        raise ValueError(
            "the children map contains a cycle: topological order covered "
            f"{len(order)} of {len(reachable)} reachable nodes"
        )
    return order, parents_within


#: Backwards-compatible alias: this function was private until the bottleneck sweep
#: needed it from another module.
_topological_order = topological_order


# ── Eq. (6): reachability recall ─────────────────────────────────────────────

def reachable_set(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
    expanded: Iterable[str],
) -> set[str]:
    """Nodes a breadth-first traversal reaches when only ``expanded`` nodes reveal their children.

    ``expanded`` is the set of nodes whose ``prune_score`` cleared :math:`\\tau_{\\text{prune}}`.
    Roots are always reached, the search starts there, whether or not they expand.
    """
    expanded = set(expanded)
    seen: set[str] = set()
    queue: deque[str] = deque()
    for r in roots:
        if r not in seen:
            seen.add(r)
            queue.append(r)
    while queue:
        node = queue.popleft()
        if node not in expanded:
            continue
        for child in children_map.get(node, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def reachability_recall(gold: Iterable[str], reachable: Iterable[str]) -> float | None:
    """Skeleton eq. (6) for one report: the fraction of annotated terms the traversal can still reach.

    :math:`R_{\\text{reach}}` is the ceiling on end-to-end recall, an annotated term the search never
    visits cannot be accepted no matter how well the identification stage would have scored it.
    Returns ``None`` for a report with no annotated terms, so an empty report cannot contribute a
    spurious ``1.0`` or ``0.0`` to the cohort average.
    """
    gold = set(gold)
    if not gold:
        return None
    return len(gold & set(reachable)) / len(gold)


def reachability_recall_cohort(
    gold_sets: Iterable[Iterable[str]],
    reachable_sets: Iterable[Iterable[str]],
) -> dict:
    """Reachability recall over a cohort, micro (pooled annotated terms) and macro (mean over reports)."""
    hits = total = 0
    per_report: list[float] = []
    for gold, reach in zip(gold_sets, reachable_sets):
        gold = set(gold)
        if not gold:
            continue
        found = len(gold & set(reach))
        hits += found
        total += len(gold)
        per_report.append(found / len(gold))
    return {
        "micro_reachability_recall": hits / total if total else None,
        "macro_reachability_recall": sum(per_report) / len(per_report) if per_report else None,
        "n_reports_with_gold": len(per_report),
        "n_gold_terms": total,
        "n_gold_reachable": hits,
    }


# ── Blocking depth ───────────────────────────────────────────────────────────

def blocking_depths(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
    expanded: Iterable[str],
    depths: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """For every node, the depth at which its last surviving root-to-node path was severed.

    The dynamic program, over a topological order so every parent is settled before its children:

    .. code-block:: text

        best(v) = INF                                  if v is a root
        best(v) = INF                                  if some parent p has best(p) = INF
                                                          and p was expanded
        best(v) = max over parents p of                otherwise
                    ( best(p)   if best(p) is finite
                      depth(p)  if best(p) is INF but p was not expanded )

    ``INF`` means *reachable*. A finite value is the blocking depth. The reasoning: if every path to
    a parent ``p`` is already severed then extending those paths to ``v`` cannot un-sever them, and
    the first non-expanded node on each is unchanged, so the value carries down as ``best(p)``. If
    ``p`` is itself reachable but refused to expand, then ``p`` *is* the first non-expanded node on
    every path through it, at ``depth(p)``. Taking the ``max`` over parents selects the path of
    least resistance, as decided for this thesis.

    Returns:
        ``{node: blocking_depth or inf}`` for every node reachable in the unpruned graph. Nodes not
        reachable from the roots even without pruning are absent from the mapping.
    """
    expanded = set(expanded)
    order, parents_within = topological_order(children_map, roots)
    if depths is None:
        depths = bfs_depths(children_map, roots)
    root_set = set(roots)

    best: dict[str, float] = {}
    for node in order:
        parents = parents_within.get(node, ())
        if node in root_set or not parents:
            best[node] = _INF
            continue
        if any(best[p] == _INF and p in expanded for p in parents):
            best[node] = _INF
            continue
        best[node] = max(
            best[p] if best[p] != _INF else float(depths[p])
            for p in parents
        )
    return best


def blocking_depth_distribution(
    gold_sets: Iterable[Iterable[str]],
    children_map: Mapping[str, Sequence[str]],
    roots: Sequence[str],
    expanded_sets: Iterable[Iterable[str]],
    depths: Mapping[str, int] | None = None,
) -> dict:
    """Histogram of the depth at which pruning severed each missed annotated term's path.

    "Concentration at shallow depths indicates that :math:`\\tau_{\\text{prune}}` is too aggressive
    near the ontology roots, where the surface expression of a broad term in a clinical narrative is
    weakest, whereas a flat distribution indicates diffuse identification error."

    Args:
        gold_sets: per-report annotated term iterables.
        children_map, roots: the traversal graph, shared across reports.
        expanded_sets: per-report sets of nodes that cleared :math:`\\tau_{\\text{prune}}`.
        depths: optional precomputed node depths; BFS depths from the roots are used otherwise.

    Returns:
        ``histogram`` (depth → count of blocked annotated terms), ``n_blocked``, ``n_reachable``,
        ``n_outside_graph`` (annotated terms not present in the traversal graph at all, an ontology
        hygiene issue, not a pruning failure) and the summary statistics ``mean``/``median``/
        ``fraction_at_depth_1``.
    """
    if depths is None:
        depths = bfs_depths(children_map, roots)

    blocked: list[int] = []
    n_reachable = n_outside = 0
    for gold, expanded in zip(gold_sets, expanded_sets):
        best = blocking_depths(children_map, roots, expanded, depths)
        for g in gold:
            if g not in best:
                n_outside += 1
            elif best[g] == _INF:
                n_reachable += 1
            else:
                blocked.append(int(best[g]))

    if not blocked:
        return {"histogram": {}, "n_blocked": 0, "n_reachable": n_reachable,
                "n_outside_graph": n_outside, "mean": float("nan"), "median": float("nan"),
                "fraction_at_depth_1": float("nan")}

    ordered = sorted(blocked)
    n = len(ordered)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "histogram": dict(sorted(Counter(ordered).items())),
        "n_blocked": n,
        "n_reachable": n_reachable,
        "n_outside_graph": n_outside,
        "mean": sum(ordered) / n,
        "median": median,
        "fraction_at_depth_1": sum(1 for d in ordered if d == 1) / n,
    }


#: Sentinel for "the iterator is finished", so an alignment check can tell a missing entry from a
#: legitimately falsy one.
_EXHAUSTED = object()


# ── §Scalability: traversal cost by depth ────────────────────────────────────

def depth_cost_table(
    visits_by_report: Iterable[Mapping[str, Mapping]],
    gold_sets: Iterable[Iterable[str]] | None = None,
) -> list[dict]:
    """The per-depth efficiency/recall exchange table of §Traversal cost by depth.

    "Efficiency in a hierarchical traversal is determined almost entirely at shallow depths, where
    the frontier is largest. We therefore tabulate, for each depth :math:`d`, the frontier size, the
    number of surviving candidates, the empirical survival rate :math:`p_d`, the cumulative SLM
    calls, and the cumulative reachability recall."

    Args:
        visits_by_report: one mapping per report, ``{hpo_id: {"depth", "expanded", "accepted",
            "n_slm_calls"}}``, the ``*_nodes.jsonl`` records for a single
            :math:`\\tau_{\\text{prune}}`. ``n_slm_calls`` may be absent, in which case the call
            columns are zero.
        gold_sets: optional aligned annotated terms. When supplied, the cumulative reachability recall
            column is filled in: the fraction of all annotated terms the search had already *visited* by
            depth :math:`d`.

    Returns:
        One row per depth, ascending, with ``depth``, ``frontier_size``, ``n_expanded``,
        ``survival_rate``, ``n_accepted``, ``n_slm_calls``, ``cumulative_slm_calls``,
        ``cumulative_frontier`` and, when ground truth is available, ``n_gold_visited``,
        ``cumulative_gold_visited`` and ``cumulative_reachability_recall``.
    """
    # Consumed in lockstep rather than materialised. Every accumulation below is per report and
    # independent, so nothing needs two reports to exist at once, and on a loose tau_prune the
    # caller's per-report mapping is thousands of entries, times a whole cohort. The alignment
    # check is preserved. It just counts as it goes instead of comparing two lengths.
    gold_iter = iter(gold_sets) if gold_sets is not None else None

    per_depth: dict[int, dict[str, int]] = {}
    n_gold_total = 0
    n_reports = 0

    for visits in visits_by_report:
        n_reports += 1
        if gold_iter is None:
            gold = set()
        else:
            try:
                gold = set(next(gold_iter))
            except StopIteration:
                raise ValueError(
                    f"visits_by_report and gold_sets must be aligned: "
                    f"visits_by_report has more than {n_reports - 1} entries"
                ) from None
        n_gold_total += len(gold)
        for hpo_id, rec in visits.items():
            d = int(rec["depth"])
            row = per_depth.setdefault(
                d, {"frontier_size": 0, "n_expanded": 0, "n_accepted": 0,
                    "n_slm_calls": 0, "n_gold_visited": 0}
            )
            row["frontier_size"] += 1
            row["n_expanded"] += int(bool(rec.get("expanded")))
            row["n_accepted"] += int(bool(rec.get("accepted")))
            row["n_slm_calls"] += int(rec.get("n_slm_calls") or 0)
            if hpo_id in gold:
                row["n_gold_visited"] += 1

    if gold_iter is not None and next(gold_iter, _EXHAUSTED) is not _EXHAUSTED:
        raise ValueError(
            f"visits_by_report and gold_sets must be aligned: "
            f"gold_sets has more than {n_reports} entries"
        )

    rows: list[dict] = []
    cum_calls = cum_frontier = cum_gold = 0
    for depth in sorted(per_depth):
        row = per_depth[depth]
        cum_calls += row["n_slm_calls"]
        cum_frontier += row["frontier_size"]
        cum_gold += row["n_gold_visited"]
        out = {
            "depth": depth,
            "frontier_size": row["frontier_size"],
            "n_expanded": row["n_expanded"],
            "survival_rate": row["n_expanded"] / row["frontier_size"],
            "n_accepted": row["n_accepted"],
            "n_slm_calls": row["n_slm_calls"],
            "cumulative_slm_calls": cum_calls,
            "cumulative_frontier": cum_frontier,
        }
        if gold_iter is not None:
            out["n_gold_visited"] = row["n_gold_visited"]
            out["cumulative_gold_visited"] = cum_gold
            out["cumulative_reachability_recall"] = (
                cum_gold / n_gold_total if n_gold_total else None
            )
        rows.append(out)
    return rows
