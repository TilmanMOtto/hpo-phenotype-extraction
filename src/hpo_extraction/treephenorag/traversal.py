"""Live breadth-first traversal of the HPO DAG, the TreePhenoRAG search engine (the earlier runs).

Prior tree experiments (the earlier runs/07/10) walked a *pre-expanded* target set (the ancestors of the
annotated terms), which presupposes the answer. This module instead descends the **real** ontology from
its roots, as deployment would: start with the children of ``HP:0000118`` and, at each
node, decide from the SLM signal whether to expand into the node's subtree.

The engine is split in two:

* :func:`traverse` owns only the frontier + threshold logic, enqueue roots, score each node once
  (DAG dedup), expand children when ``prune_score >= tau_prune``, report when
  ``accept_score >= tau_accept``, and record the trace. It is pure and I/O-free: the expensive
  per-node work is injected as ``evaluate_node``, so it unit-tests on a toy DAG with a dict of
  fixed scores.
* the retrieval + SLM + scoring that produces those scores lives in ``hpo_extraction.treephenorag.score_store``,
  which supplies ``evaluate_node``.

Two scores per node, by design (see ``hpo_extraction.treephenorag.pooling``): ``prune_score`` is *subtree-present*
and gates traversal; ``accept_score`` is *node-present* and gates reporting. They are thresholded
independently.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace


@dataclass
class NodeEval:
    """What :func:`traverse` needs back from ``evaluate_node`` for one node.

    ``calls`` is the list of per-(context-type, rank) SLM records for this node, already fully
    populated by the evaluator (the engine only collects them). ``n_slm_calls`` is the number of
    SLM forward passes the node cost, so the trace can report inference cost faithfully even when
    a mode issues two retrieval passes.
    """

    prune_score: float
    accept_score: float
    n_slm_calls: int = 0
    calls: list = field(default_factory=list)


@dataclass
class NodeVisit:
    """One visited term of a traversal.

    Attributes:
        hpo_id: the term.
        depth: distance from HP:0000118 along the path the traversal took (organ systems are 1).
        prune_score: expansion score, between 0 and 1.
        accept_score: acceptance score, between 0 and 1.
        expanded: whether its children were visited.
        accepted: whether it was output.
        n_slm_calls: verifier calls spent on it.
    """
    hpo_id: str
    depth: int
    prune_score: float
    accept_score: float
    expanded: bool
    accepted: bool
    n_slm_calls: int


@dataclass
class TraversalResult:
    """The full trace of one report's traversal, every quantity Step-2 metrics consume."""

    order: list[str]
    visits: dict[str, NodeVisit]
    calls: list[dict]
    accepted: set[str]
    total_frontier_insertions: int
    n_slm_calls: int

    @property
    def n_unique_nodes_visited(self) -> int:
        """Number of distinct terms visited."""
        return len(self.visits)

    def at_accept(self, tau_accept: float) -> "TraversalResult":
        """This same traversal with the accept decisions re-derived at a different ``tau_accept``.

        Expansion depends only on ``tau_prune``, so which nodes were visited, and therefore the
        SLM cost, is *invariant* to ``tau_accept``. Sweeping the accept threshold is thus exact
        post-processing over ``visits``, not a second traversal, and costs no SLM calls. That is
        what makes the accept sweep free: one BFS per ``tau_prune`` serves the whole accept grid.

        Returns a new result; ``self`` is untouched. ``calls`` is shared by reference because it is
        accept-independent and the caller writes it once per report.
        """
        visits = {
            h: replace(v, accepted=v.accept_score >= tau_accept) for h, v in self.visits.items()
        }
        return TraversalResult(
            order=self.order,
            visits=visits,
            calls=self.calls,
            accepted={h for h, v in visits.items() if v.accepted},
            total_frontier_insertions=self.total_frontier_insertions,
            n_slm_calls=self.n_slm_calls,
        )

    def depth_table(self) -> list[dict]:
        """Per-depth frontier stats: nodes scored, expanded, accepted, and cumulative visited.

        Feeds the "traversal cost by depth" table (thesis §Scalability) and, with the ground-truth set,
        the blocking-depth diagnostic, efficiency in a hierarchical traversal is decided almost
        entirely at shallow depths, so the trace is reported per depth rather than in aggregate.
        """
        by_depth: dict[int, dict] = {}
        for v in self.visits.values():
            d = by_depth.setdefault(
                v.depth, {"depth": v.depth, "n_visited": 0, "n_expanded": 0, "n_accepted": 0,
                          "n_slm_calls": 0}
            )
            d["n_visited"] += 1
            d["n_expanded"] += int(v.expanded)
            d["n_accepted"] += int(v.accepted)
            d["n_slm_calls"] += v.n_slm_calls
        rows = [by_depth[k] for k in sorted(by_depth)]
        cum = 0
        for r in rows:
            cum += r["n_visited"]
            r["cumulative_visited"] = cum
        return rows


def build_children_map(hpo_tree) -> tuple[dict[str, list[str]], list[str]]:
    """``(children, roots)`` over the phenotypic-abnormality subtree, direct edges only.

    ``children[n]`` are the direct children (``HPO_class.son``) of ``n`` inside the subtree, so
    expanding a node reveals one ontology level. Roots are ``hpo_tree.layer1``, the
    children of ``HP:0000118``, which is where the search starts. The root term itself is not
    scored (it is trivially present in every report).
    """
    from hpo_extraction.ontology.hpo_tree import HPO_class

    pa = hpo_tree.phenotypic_abnormality
    children: dict[str, list[str]] = {}
    for n in pa:
        if n == hpo_tree.root:
            continue
        sons = HPO_class(hpo_tree.data[n]).son if n in hpo_tree.data else set()
        children[n] = sorted(s for s in sons if s in pa and s != hpo_tree.root)
    roots = [r for r in hpo_tree.layer1 if r in children]
    return children, roots


def traverse(
    children_map: dict[str, list[str]],
    roots,
    evaluate_node,
    tau_prune: float,
    tau_accept: float,
    on_visit=None,
    max_nodes: int | None = None,
) -> TraversalResult:
    """Breadth-first calibrated traversal of the DAG.

    Args:
        children_map: ``{node: [direct children in the subtree]}``.
        roots: nodes to seed the frontier (the layer-1 organ systems).
        evaluate_node: ``hpo_id -> NodeEval``, retrieval + SLM + scoring for one node.
        tau_prune: expand a node's children iff ``prune_score >= tau_prune``.
        tau_accept: report a node iff ``accept_score >= tau_accept``.
        on_visit: optional callback invoked with each :class:`NodeVisit` (progress bars, logging).
        max_nodes: optional cap on unique nodes scored (pilots); ``None`` = no cap.

    Every node is scored at most once even when reached along several DAG paths. The number of
    times it was *offered* to the frontier is accumulated into ``total_frontier_insertions`` so
    the DAG deduplication is auditable (thesis §Per-report inference cost).
    """
    visits: dict[str, NodeVisit] = {}
    order: list[str] = []
    calls: list[dict] = []
    accepted: set[str] = set()
    n_slm_calls = 0
    total_insertions = 0

    depth_of: dict[str, int] = {}
    frontier: deque[str] = deque()
    enqueued: set[str] = set()
    for r in roots:
        depth_of[r] = 1
        enqueued.add(r)
        frontier.append(r)
        total_insertions += 1

    while frontier:
        node = frontier.popleft()
        if max_nodes is not None and len(visits) >= max_nodes:
            break

        ev = evaluate_node(node)
        expanded = ev.prune_score >= tau_prune
        is_accepted = ev.accept_score >= tau_accept

        visit = NodeVisit(
            hpo_id=node,
            depth=depth_of[node],
            prune_score=float(ev.prune_score),
            accept_score=float(ev.accept_score),
            expanded=bool(expanded),
            accepted=bool(is_accepted),
            n_slm_calls=int(ev.n_slm_calls),
        )
        visits[node] = visit
        order.append(node)
        calls.extend(ev.calls)
        n_slm_calls += ev.n_slm_calls
        if is_accepted:
            accepted.add(node)
        if on_visit is not None:
            on_visit(visit)

        if expanded:
            for child in children_map.get(node, ()):
                total_insertions += 1
                if child not in enqueued:
                    enqueued.add(child)
                    depth_of[child] = depth_of[node] + 1
                    frontier.append(child)

    return TraversalResult(
        order=order,
        visits=visits,
        calls=calls,
        accepted=accepted,
        total_frontier_insertions=total_insertions,
        n_slm_calls=n_slm_calls,
    )
