"""Where pruning went wrong, attributed to a node, not just to a depth.

``hpo_extraction.evaluation.metrics.traversal`` answers *how deep* pruning severed each missed annotated term's
path. That is the right summary for a thesis table and the wrong one for a debugging session: a
histogram peaking at depth 1 tells you the layer-1 gate is too aggressive, but not *which* of the
24 organ systems refused to open, in how many reports, or at what score. This module runs the same
dynamic program carrying the identity of the severing node, so every blocked annotated term can be
traced to the exact decision that cost it, and to the τ_prune at which that decision would have
gone the other way.

Three analyses, in the order a diagnosis actually proceeds:

:func:`blocking_culprits`
    The DP. Same recurrence and same numbers as ``traversal.blocking_depths``, plus the culprit.
    :func:`assert_matches_traversal` pins the equality, because a divergence here would move every
    downstream count silently.

:func:`prune_ledger`
    Every annotated term of every report given a fate, ``found`` / ``rejected`` / ``blocked`` /
    ``unreached`` / ``outside_graph``, and, for the blocked ones, the culprit. This is the recall
    decomposition the scorecard leads with: how much of the miss rate is traversal and how much is
    identification.

:func:`culprit_leaderboard` and :func:`fp_factory`
    The two "which node is responsible" rankings, one for recall lost to pruning, one for the
    precision lost to expansion. They are mirror images: the same decision that lets an annotated term
    through also lets a subtree of false positives through, and a UI that shows only one of them
    will always recommend moving τ in one direction.

Everything here is pure: dicts and DataFrames in, dicts out. No file is opened, no ``HPOTree`` is
built, which is what lets the whole module be tested against the ten-node toy ontology.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Iterable, Mapping, Sequence

_INF = float("inf")

#: An annotated term's cause of death. Ordered best → worst. Mirrored by ``theme.FATE_ORDER``.
FATES: tuple[str, ...] = ("found", "rejected", "blocked", "unreached", "outside_graph")


# ── the traversal graph, without the nltk import chain ───────────────────────

def children_map_from_tree(tree) -> tuple[dict[str, list[str]], list[str]]:
    """``(children, roots)`` over the phenotypic-abnormality subtree, direct edges only.

    A transcription of ``hpo_extraction.treephenorag.traversal.build_children_map`` that reads ``data[n]["Son"]``
    instead of going through ``HPO_class``. The only reason it exists: ``hpo_extraction.ontology.hpo_tree``
    imports ``nltk`` and ``stanza`` at module scope, so importing the original drags the whole NLP
    stack into a process that only wants the graph, and makes the analysis untestable anywhere
    those packages are not installed. :func:`assert_children_map_matches` pins it to the original
    wherever the original can be imported.
    """
    pa = set(tree.phenotypic_abnormality)
    root = tree.root
    children: dict[str, list[str]] = {}
    for node in pa:
        if node == root:
            continue
        sons = set(tree.data[node]["Son"].keys()) if node in tree.data else set()
        children[node] = sorted(s for s in sons if s in pa and s != root)
    roots = [r for r in tree.layer1 if r in children]
    return children, roots


def assert_children_map_matches(tree) -> None:
    """Fix :func:`children_map_from_tree` to ``hpo_extraction.treephenorag.traversal.build_children_map``.

    Raises ``AssertionError`` on any disagreement. Callers that cannot import the original (no
    nltk) should skip rather than swallow, an unrun fix is not a passing fix.
    """
    from hpo_extraction.treephenorag.traversal import build_children_map

    expected_children, expected_roots = build_children_map(tree)
    children, roots = children_map_from_tree(tree)
    assert roots == expected_roots, "roots drifted from hpo_extraction.treephenorag.traversal.build_children_map"
    assert children.keys() == expected_children.keys(), "node set drifted"
    for node, sons in expected_children.items():
        assert children[node] == sons, f"children of {node} drifted: {children[node]} != {sons}"


def bfs_depths(children_map: Mapping[str, Sequence[str]], roots: Iterable[str]) -> dict[str, int]:
    """Breadth-first depth from the roots, roots at depth 1.

    Delegates to ``thesis_metrics.traversal`` so the depths in this module and the depths in the
    thesis tables are the same numbers, computed once.
    """
    from hpo_extraction.evaluation.metrics.traversal import bfs_depths as _bfs

    return _bfs(children_map, roots)


#: ``id(children_map) -> (order, parents_within)``. The traversal graph is the whole ontology and
#: it is identical for every report of every run, but the DP needs its topological order once per
#: :func:`blocking_culprits` call, which is once per report. Recomputing it was 11% of a bundle
#: build. Keyed by identity because the registry builds one children map per process and a
#: 18k-entry dict is not hashable. A caller passing a fresh map each time simply misses the cache.
_ORDER_CACHE: dict[int, tuple[tuple[list[str], dict[str, list[str]]], object]] = {}


def _cached_topological_order(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """:func:`_topological_order`, memoised on the identity of ``children_map``.

    The cached value keeps a reference to the map it was built from, so the entry cannot outlive
    its graph and be handed to a different one that happened to reuse the address.
    """
    key = id(children_map)
    hit = _ORDER_CACHE.get(key)
    if hit is not None and hit[1] is children_map:
        return hit[0]
    computed = _topological_order(children_map, roots)
    _ORDER_CACHE[key] = (computed, children_map)
    return computed


_POSITION_CACHE: dict[int, tuple[dict[str, int], object]] = {}


def _cached_positions(children_map, order: Sequence[str]) -> dict[str, int]:
    """``{node: index in the topological order}``, memoised alongside the order itself."""
    key = id(children_map)
    hit = _POSITION_CACHE.get(key)
    if hit is not None and hit[1] is children_map:
        return hit[0]
    positions = {node: index for index, node in enumerate(order)}
    _POSITION_CACHE[key] = (positions, children_map)
    return positions


def _topological_order(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Kahn order over the nodes reachable from ``roots`` ignoring pruning, plus induced parents.

    duplicated from ``traversal._topological_order`` (a private function): importing
    a private name would couple this module to an implementation detail of another package, and
    :func:`assert_matches_traversal` catches any drift between the two far more cheaply than the
    coupling would cost.
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
    indegree: dict[str, int] = {n: 0 for n in reachable}
    for node in reachable:
        for child in children_map.get(node, ()):
            if child in reachable:
                parents_within[child].append(node)
                indegree[child] += 1

    order: list[str] = []
    ready: deque[str] = deque(sorted(n for n in reachable if indegree[n] == 0))
    while ready:
        node = ready.popleft()
        order.append(node)
        for child in children_map.get(node, ()):
            if child in reachable:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
    return order, parents_within


# ── the DP ───────────────────────────────────────────────────────────────────

def _ancestor_closure(
    parents_within: Mapping[str, Sequence[str]],
    targets: Iterable[str],
) -> set[str]:
    """``targets`` plus every node that can reach one of them, within the reachable subgraph.

    ``best(v)`` depends only on ``best(parent(v))``, recursively, so a target's value is fully
    determined by its own ancestor closure and nothing else in the graph can change it.
    """
    seen: set[str] = set()
    stack = [t for t in targets if t in parents_within]
    seen.update(stack)
    while stack:
        for parent in parents_within.get(stack.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def blocking_culprits(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
    expanded: Iterable[str],
    depths: Mapping[str, int] | None = None,
    targets: Iterable[str] | None = None,
) -> dict[str, tuple[float, str | None]]:
    """``{node: (blocking_depth, culprit)}`` over every node reachable in the unpruned graph.

    ``targets`` restricts the DP to the ancestor closure of the nodes actually being asked about.
    The result for those nodes is identical, the recurrence reads only ancestors, but a report
    with six annotated terms then settles a few hundred nodes instead of all 18k of the
    phenotypic-abnormality subtree, which is the difference between the ledger costing seconds
    per cohort and costing tens of seconds. Pass ``None`` to settle the whole graph;
    :func:`assert_matches_traversal` checks the two agree.

    ``(inf, None)`` means the node is reachable, the traversal could have got there. A finite
    depth means every root-to-node path was severed, and ``culprit`` names the node that severed
    the path of least resistance: the one whose refusal to expand is what a fix has to target.

    The recurrence is ``traversal.blocking_depths``'s, verbatim::

        best(v) = INF                        if v is a root, or has no parents
        best(v) = INF                        if some parent p is reachable and expanded
        best(v) = max over parents p of      otherwise
                    ( best(p)   if best(p) finite
                      depth(p)  if best(p) = INF but p was not expanded )

    ``max`` selects the path of least resistance, the one that got deepest before being cut, which is the path eq. (6) already privileges. Ties on depth are broken by the lexicographically
    smallest culprit id, so a rerun names the same node. The tie-break is arbitrary but fixed, and
    it never changes the depth, only which of two equally deep culprits is displayed.
    """
    expanded = set(expanded)
    order, parents_within = _cached_topological_order(children_map, roots)
    if depths is None:
        depths = bfs_depths(children_map, roots)
    root_set = set(roots)

    if targets is not None:
        needed = _ancestor_closure(parents_within, targets)
        # Reorder by the cached order's positions, not rescanning it: the scan is O(V) per
        # report, which is the cost this restriction exists to avoid.
        positions = _cached_positions(children_map, order)
        order = sorted(needed, key=positions.__getitem__)

    best: dict[str, tuple[float, str | None]] = {}
    for node in order:
        parents = parents_within.get(node, ())
        if node in root_set or not parents:
            best[node] = (_INF, None)
            continue

        # Every parent is either already severed, or reachable-but-closed (which makes the parent
        # itself the severing node). Pick the deepest such cut, breaking depth ties on the
        # lexicographically smallest culprit id so a rerun names the same node.
        #
        # This is one explicit loop, not max() over a comprehension with a sort key
        # because it is the innermost operation of the whole package, ~18k nodes per report per
        # τ, and the key-function version spent more time building throwaway tuples and
        # order-reversed strings than it did on the DP itself.
        best_depth = -1.0
        best_culprit: str | None = None
        reachable = False
        for parent in parents:
            parent_depth, parent_culprit = best[parent]
            if parent_depth == _INF:
                if parent in expanded:
                    reachable = True
                    break
                depth, culprit = float(depths[parent]), parent
            else:
                depth, culprit = parent_depth, parent_culprit
            if depth > best_depth or (depth == best_depth
                                      and culprit is not None
                                      and (best_culprit is None or culprit < best_culprit)):
                best_depth, best_culprit = depth, culprit

        best[node] = (_INF, None) if reachable else (best_depth, best_culprit)
    return best


def assert_matches_traversal(
    children_map: Mapping[str, Sequence[str]],
    roots: Iterable[str],
    expanded: Iterable[str],
    depths: Mapping[str, int] | None = None,
) -> None:
    """Fix :func:`blocking_culprits`' depths to ``traversal.blocking_depths``, node for node.

    The culprit is extra information. The depth is a published number. If the two ever disagree
    the UI would be attributing recall loss to depths the thesis does not report, so the equality
    is asserted, not assumed.
    """
    from hpo_extraction.evaluation.metrics.traversal import blocking_depths

    expected = blocking_depths(children_map, roots, expanded, depths)
    actual = blocking_culprits(children_map, roots, expanded, depths)
    assert actual.keys() == expected.keys(), "node set drifted from traversal.blocking_depths"
    for node, depth in expected.items():
        got, culprit = actual[node]
        assert got == depth, f"{node}: blocking depth {got} != {depth}"
        assert (culprit is None) == (depth == _INF), \
            f"{node}: culprit {culprit!r} disagrees with reachability (depth {depth})"

    # The restricted DP is what the ledger actually runs, so it is fixed too: it must return the
    # same value for every node it is asked about, having settled only their ancestors.
    restricted = blocking_culprits(children_map, roots, expanded, depths,
                                   targets=list(expected))
    for node in expected:
        assert restricted[node] == actual[node], (
            f"{node}: restricting the DP to its ancestors changed the answer "
            f"{restricted[node]} != {actual[node]}")


# ── the ledger ───────────────────────────────────────────────────────────────

def prune_ledger(
    nodes_by_report: Mapping[str, dict],
    gold: Mapping[str, Iterable[str]],
    children_map: Mapping[str, Sequence[str]],
    roots: Sequence[str],
    depths: Mapping[str, int] | None = None,
) -> dict:
    """Give every annotated term of every report a fate, and attribute the blocked ones.

    Args:
        nodes_by_report: ``{report_id: {"visited": set, "expanded": set, "accepted": set,
            "prune_score": {hpo: float}}}``, what :mod:`~apps.treephenorag_ui.nodes` derives from
            ``*_nodes.jsonl`` at one configuration.
        ground truth: ``{report_id: gold terms}``, joined from the chosen ground truth file. Never from the
            ``is_gold`` flag on the artifact, which records whichever ground-truth set was configured at
            *inference* time and may not be the one being scored against.
        children_map, roots: the traversal graph, shared across reports.
        depths: optional precomputed BFS depths.

    Returns:
        ``rows``, one dict per (report, annotated term) with its fate and, when blocked, the culprit
        and the culprit's own prune score; ``fate_counts``; ``recall_actual``. And
        ``recall_ceiling``, the recall this run could reach at *any* τ_accept given the pruning it
        performed. The gap between the two is identification error. The gap between the ceiling and
        1.0 is traversal error. That decomposition is the whole point of the ledger.
    """
    if depths is None:
        depths = bfs_depths(children_map, roots)

    rows: list[dict] = []
    counts: Counter = Counter()
    for report_id in sorted(gold):
        gold_terms = sorted(set(gold[report_id]))
        if not gold_terms:
            continue
        state = nodes_by_report.get(report_id) or {}
        visited: set[str] = set(state.get("visited") or ())
        expanded: set[str] = set(state.get("expanded") or ())
        accepted: set[str] = set(state.get("accepted") or ())
        scores: Mapping[str, float] = state.get("prune_score") or {}

        best: dict[str, tuple[float, str | None]] | None = None
        for term in gold_terms:
            if term in accepted:
                fate, culprit, depth = "found", None, depths.get(term)
            elif term in visited:
                fate, culprit, depth = "rejected", None, depths.get(term)
            elif term not in children_map:
                fate, culprit, depth = "outside_graph", None, None
            else:
                if best is None:
                    # Once per report, and restricted to this report's annotated terms: the DP
                    # over the whole 18k-node subtree is ~50 ms in Python, which across a
                    # cohort is most of the ledger's cost, and all but a few hundred of
                    # those nodes can have no bearing on the answer.
                    best = blocking_culprits(children_map, roots, expanded, depths,
                                             targets=gold_terms)
                block_depth, culprit = best.get(term, (_INF, None))
                if block_depth == _INF:
                    # Reachable in principle, yet never scored: the traversal stopped early
                    # (``max_nodes``), or this run's graph is not the graph being analysed. Either
                    # way it is not a pruning failure and must not be counted as one.
                    fate, depth = "unreached", depths.get(term)
                else:
                    fate, depth = "blocked", int(block_depth)

            counts[fate] += 1
            rows.append({
                "report_id": report_id,
                "hpo_id": term,
                "fate": fate,
                "depth": depth,
                "culprit": culprit,
                "culprit_depth": depths.get(culprit) if culprit else None,
                "culprit_prune_score": scores.get(culprit) if culprit else None,
            })

    n_gold = sum(counts.values())
    n_found = counts.get("found", 0)
    n_blocked = counts.get("blocked", 0)
    return {
        "rows": rows,
        "fate_counts": {fate: counts.get(fate, 0) for fate in FATES},
        "n_gold": n_gold,
        "recall_actual": n_found / n_gold if n_gold else 0.0,
        # Everything the traversal reached could in principle have been accepted at a low enough
        # τ_accept. Nothing it failed to reach could have been, at any τ_accept.
        "recall_ceiling": (n_gold - n_blocked - counts.get("unreached", 0)
                           - counts.get("outside_graph", 0)) / n_gold if n_gold else 0.0,
        "recall_lost_to_pruning": n_blocked / n_gold if n_gold else 0.0,
    }


def culprit_leaderboard(
    ledger: Mapping,
    *,
    tau_sweep: Sequence[float] = (),
    labels: Mapping[str, str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """The nodes whose refusal to expand cost the most annotated terms, worst first.

    Each row carries the culprit's own ``prune_score`` distribution across the reports where it
    blocked something, and ``tau_would_expand_all``, the largest τ in the run's sweep that sits at
    or below the *minimum* of those scores, i.e. The highest threshold at which this node would
    have opened in every report where it mattered. ``None`` there means no swept τ is low enough:
    the fix is not a threshold, it is the score.
    """
    labels = labels or {}
    by_culprit: dict[str, dict] = {}
    for row in ledger.get("rows", ()):
        culprit = row.get("culprit")
        if row.get("fate") != "blocked" or not culprit:
            continue
        entry = by_culprit.setdefault(culprit, {
            "hpo_id": culprit,
            "hpo_label": labels.get(culprit, culprit),
            "depth": row.get("culprit_depth"),
            "n_gold_lost": 0,
            "reports": set(),
            "scores": [],
            "gold_lost": [],
        })
        entry["n_gold_lost"] += 1
        entry["reports"].add(row["report_id"])
        score = row.get("culprit_prune_score")
        if score is not None:
            entry["scores"].append(float(score))
        entry["gold_lost"].append({"report_id": row["report_id"], "hpo_id": row["hpo_id"],
                                   "hpo_label": labels.get(row["hpo_id"], row["hpo_id"])})

    sweep = sorted(float(t) for t in tau_sweep)
    out: list[dict] = []
    for entry in by_culprit.values():
        scores = entry.pop("scores")
        reports = entry.pop("reports")
        score_min = min(scores) if scores else None
        candidates = [t for t in sweep if score_min is not None and t <= score_min]
        out.append({
            **entry,
            "n_reports": len(reports),
            "prune_score_min": score_min,
            "prune_score_max": max(scores) if scores else None,
            "prune_score_mean": sum(scores) / len(scores) if scores else None,
            "tau_would_expand_all": max(candidates) if candidates else None,
            "gold_lost": entry["gold_lost"][:20],
        })

    out.sort(key=lambda r: (-r["n_gold_lost"], -r["n_reports"], r["hpo_id"]))
    return out[:limit]


def fp_factory(
    nodes_by_report: Mapping[str, dict],
    false_positives: Mapping[str, Iterable[str]],
    children_map: Mapping[str, Sequence[str]],
    *,
    labels: Mapping[str, str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """The expanded nodes with the most accepted-but-wrong terms in their subtree, worst first.

    The mirror image of :func:`culprit_leaderboard`. A node that was expanded is a node whose
    subtree the traversal was allowed to explore, so every false positive below it is downstream
    of that one decision. Ranking by ``n_fp`` names the expansions that cost the most precision. Reading it beside the culprit leaderboard is what stops the obvious-but-wrong conclusion that
    τ_prune should simply be lowered.

    Attribution is to the FP's *direct expanded parents*, not to every ancestor: charging a
    layer-1 organ system for every false positive anywhere beneath it would rank the ontology's
    shape, not the run's decisions.
    """
    labels = labels or {}
    parents: dict[str, list[str]] = defaultdict(list)
    for node, children in children_map.items():
        for child in children:
            parents[child].append(node)

    tally: dict[str, dict] = {}
    for report_id, fps in false_positives.items():
        state = nodes_by_report.get(report_id) or {}
        expanded: set[str] = set(state.get("expanded") or ())
        scores: Mapping[str, float] = state.get("prune_score") or {}
        for fp in fps:
            for parent in parents.get(fp, ()):
                if parent not in expanded:
                    continue
                entry = tally.setdefault(parent, {
                    "hpo_id": parent,
                    "hpo_label": labels.get(parent, parent),
                    "n_fp": 0,
                    "reports": set(),
                    "scores": [],
                    "examples": [],
                })
                entry["n_fp"] += 1
                entry["reports"].add(report_id)
                if parent in scores:
                    entry["scores"].append(float(scores[parent]))
                entry["examples"].append({"report_id": report_id, "hpo_id": fp,
                                          "hpo_label": labels.get(fp, fp)})

    out: list[dict] = []
    for entry in tally.values():
        scores = entry.pop("scores")
        reports = entry.pop("reports")
        out.append({
            **entry,
            "n_reports": len(reports),
            "prune_score_mean": sum(scores) / len(scores) if scores else None,
            "examples": entry["examples"][:20],
        })
    out.sort(key=lambda r: (-r["n_fp"], -r["n_reports"], r["hpo_id"]))
    return out[:limit]


def blocking_depth_histogram(ledger: Mapping) -> dict:
    """``{depth: count}`` over blocked annotated terms, plus the summary stats the thesis reports.

    Concentration at depth 1 means τ_prune is too aggressive at the organ-system level, where a
    broad term's surface expression in a narrative is weakest. A flat distribution means diffuse
    identification error that no single threshold will fix.
    """
    depths = [int(r["depth"]) for r in ledger.get("rows", ())
              if r.get("fate") == "blocked" and r.get("depth") is not None]
    if not depths:
        return {"histogram": {}, "n_blocked": 0, "mean": None, "median": None,
                "fraction_at_depth_1": None}
    ordered = sorted(depths)
    n = len(ordered)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "histogram": dict(sorted(Counter(ordered).items())),
        "n_blocked": n,
        "mean": sum(ordered) / n,
        "median": median,
        "fraction_at_depth_1": sum(1 for d in ordered if d == 1) / n,
    }
