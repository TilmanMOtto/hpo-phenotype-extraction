"""One report's terms, placed back in the ontology they came from.

Every other view of a report is a list: the terms it got right, the terms it invented, the terms
it missed. A list cannot show the thing that actually distinguishes the earlier runs'errors from a flat
method's, *where* in the hierarchy they sit relative to each other. A false positive one edge
above a missed annotated term and a false positive in another organ system are the same row in a table
and completely different failures, and the traversal that produced them is a walk over
this graph.

So this module builds the **induced subgraph**: the report's true positives, false positives and
false negatives, plus the ancestor closure needed to connect them to a root. Ancestors are not
decoration, they are the path the traversal took (or would have taken), and the culprit that
severed a blocked annotated term is one of them. Nothing else is included: a full-ontology walk visits
thousands of nodes per report, and drawing them would bury the dozen that carry the story.

Three shapes come out of :func:`build`:

``nodes``   one entry per term with ``x``/``y``, its outcome, and everything the hover needs.
``edges``   ``(parent, child)`` pairs, split into the spanning ones and the extra DAG parents.
``orphans`` terms that resolve to nothing in the ontology, hallucinated codes and out-of-subtree
            ones. They have no place in the hierarchy and are parked in their own row rather than
            hung off a root they do not belong to.

The layout is the ordinary tidy-tree one, leaves get consecutive slots left to right, a parent
sits over the mean of its children, computed on a spanning tree taken from the DAG by giving each
node its lexicographically first in-set parent. HPO is a DAG and a term with two parents can only
be drawn under one of them. The edge to the other is kept and returned separately, so the reader
sees the second parent as an extra line, not as a missing one.
"""

from __future__ import annotations

#: More nodes than this and the figure is a smear, not a diagram. Real reports land far
#: below it, a ground-truth set is ~10 terms and its ancestor closure ~40, but a run at a permissive
#: τ_accept can predict hundreds of terms for one report, and that page must still open.
MAX_NODES = 500

#: Draw order and legend order. ``context`` is an ancestor that is itself neither predicted nor
#: annotated: structure, not a result.
GROUPS = ("TP", "FP", "FN", "context")


def build(terms: dict[str, str], view, *, max_nodes: int = MAX_NODES) -> dict:
    """Lay out one report's terms and their ancestors.

    Args:
        terms: ``{hpo_id: outcome}`` for the report, outcomes drawn from ``TP``/``FP``/``FN``.
            Raw ids as they appear in the artifacts, resolution happens here, and an id that
            resolves to nothing becomes an orphan, not disappearing.
        view: an ``OntologyView``.
        max_nodes: cap on the drawn set. When the closure exceeds it, the ancestors are kept and
            the *context* nodes are thinned first, since they are the ones carrying no result.

    Returns:
        ``{"nodes": [...], "edges": [...], "extra_edges": [...], "orphans": [...],
        "n_truncated": int}``. ``nodes`` entries carry ``hpo_id``, ``group``, ``depth``, ``x``,
        ``y`` and ``resolved``.
    """
    resolved_of: dict[str, str] = {}
    orphans: list[dict] = []
    outcome_of: dict[str, str] = {}
    for hpo_id, outcome in terms.items():
        resolved = view.resolve(hpo_id)
        if resolved is None:
            orphans.append({"hpo_id": hpo_id, "group": outcome, "resolved": None})
            continue
        resolved_of[hpo_id] = resolved
        # A raw id and its alt id resolve to the same node. Whichever outcome is worse wins, so a
        # node that is both predicted and annotated under two spellings is not drawn as a clean TP.
        outcome_of[resolved] = _worse(outcome_of.get(resolved), outcome)

    closure: set[str] = set()
    for resolved in resolved_of.values():
        closure |= set(view.ancestors(resolved))
    closure |= set(outcome_of)

    n_truncated = 0
    if len(closure) > max_nodes:
        closure, n_truncated = _thin(closure, set(outcome_of), view, max_nodes)

    parents = {n: sorted(p for p in view.direct_parents(n) if p in closure) for n in closure}
    spanning = {n: (p[0] if p else None) for n, p in parents.items()}
    extra_edges = [(p, n) for n, ps in parents.items() for p in ps[1:]]

    children: dict[str, list[str]] = {n: [] for n in closure}
    roots: list[str] = []
    for node, parent in spanning.items():
        if parent is None:
            roots.append(node)
        else:
            children[parent].append(node)

    depths = {n: view.depth(n) for n in closure}
    for node in children:
        children[node].sort(key=lambda n: (depths.get(n) or 0, n))
    positions = _positions(sorted(roots, key=lambda n: (depths.get(n) or 0, n)), children)

    nodes = [
        {
            "hpo_id": node,
            "group": outcome_of.get(node, "context"),
            "depth": depths.get(node),
            "x": positions[node],
            "y": -(depths.get(node) if depths.get(node) is not None else 0),
            "resolved": node,
        }
        for node in sorted(closure, key=lambda n: (positions[n], n))
    ]
    edges = [(parent, node) for node, parent in spanning.items() if parent is not None]
    return {
        "nodes": nodes,
        "edges": sorted(edges),
        "extra_edges": sorted(extra_edges),
        "orphans": sorted(orphans, key=lambda o: o["hpo_id"]),
        "n_truncated": n_truncated,
    }


#: Which outcome survives when two raw codes land on the same node. An error outranks a success,
#: so a node drawn green is one nothing went wrong with.
_SEVERITY = {"context": 0, "TN": 1, "TP": 2, "FN": 3, "FP": 4}


def _worse(current: str | None, candidate: str) -> str:
    if current is None:
        return candidate
    return candidate if _SEVERITY.get(candidate, 0) > _SEVERITY.get(current, 0) else current


def _thin(closure: set[str], keep: set[str], view, max_nodes: int) -> tuple[set[str], int]:
    """Drop context nodes, deepest first, until the closure fits.

    The result nodes are never dropped, they are the subject, and neither is anything on a path
    between two of them that a shallower node would leave dangling, which is why the deepest go
    first: a deep context node is a leaf of the closure, and removing a leaf disconnects nothing.
    """
    context = sorted(closure - keep, key=lambda n: (-(view.depth(n) or 0), n))
    dropped = 0
    for node in context:
        if len(closure) <= max_nodes:
            break
        closure.discard(node)
        dropped += 1
    return closure, dropped


def _positions(roots: list[str], children: dict[str, list[str]]) -> dict[str, float]:
    """Tidy-tree x coordinates: leaves take consecutive slots, parents sit over their children.

    Iterative, not recursive, the closure of a deep HPO branch is a few dozen levels, well
    inside Python's limit, but this runs inside a Dash callback where a ``RecursionError`` reads as
    a blank page, not as a stack trace.
    """
    positions: dict[str, float] = {}
    cursor = 0.0
    for root in roots:
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node in positions:
                continue
            kids = children.get(node) or []
            if not kids:
                positions[node] = cursor
                cursor += 1.0
                continue
            if not expanded:
                stack.append((node, True))
                stack.extend((kid, False) for kid in reversed(kids))
                continue
            placed = [positions[k] for k in kids if k in positions]
            positions[node] = sum(placed) / len(placed) if placed else cursor
        cursor += 1.0
    return positions


def terms_of_report(frame) -> dict[str, str]:
    """``{hpo_id: outcome}`` for one report's node-table slice, TP, FP and FN only.

    True negatives are excluded on purpose. At a permissive τ_prune a report has thousands of
    them, they are what the traversal *correctly* said nothing about, and including them would
    turn the diagram into the ontology.
    """
    if frame is None or frame.empty:
        return {}
    subset = frame.loc[frame["outcome"].isin(("TP", "FP", "FN"))]
    return {str(record.hpo_id): str(record.outcome) for record in subset.itertuples()}
