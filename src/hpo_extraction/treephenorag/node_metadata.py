"""Per-node structural covariates over the **full** HPO ontology.

These are the node-level features every pruning/calibration analysis conditions on, so they
are computed once here rather than re-derived per experiment:

* ``subtree_size``, transitive descendant count. The cost of exploring the node.
* ``n_direct_children``, branching factor.
* ``depth_bfs``, shortest path from ``HP:0000118`` (``HPOTree.depth_dict``).
* ``depth_long``, longest path from the root (``HPOTree.depth_long``).
* ``layer1_organ``, the top-level organ system(s) the node sits under.
* ``is_leaf``, no direct children.

**Why both depths.** ``depth_bfs`` is the shortest path, and HPO is a DAG, not a tree,
so it is *not* monotone along ancestry: a node reachable by a short branch can have an ancestor,
reachable only by a long branch, whose ``depth_bfs`` is larger than its own (see
``HPOTree._buildLongestDepth``, it holds for 1.4% of ancestor/descendant pairs). Any analysis
that conditions on depth *as a proxy for generality* needs ``depth_long``, which is monotone by
construction. ``depth_bfs`` is retained because existing findings are stratified by it.

**Nodes outside the phenotypic-abnormality subtree** get ``depth_bfs = depth_long = None`` and
``layer1_organ = []``, not a misleading 0, ``HPOTree`` restricts most of its operations to
that subtree, so such nodes have no defined depth. Callers must exclude them from
depth-stratified analysis; ``node_metadata_table`` flags them via ``in_phenotypic_abnormality``.
"""

from __future__ import annotations

from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree


def _ensure_depths(tree: HPOTree) -> None:
    """Populate ``depth_dict`` / ``depth_long`` if the caller hasn't run ``buildHPOTree``.

    Idempotent and cheap after the first call, mirroring
    ``hpo_extraction.evaluation.tree_metrics._ensure_depths``.
    """
    if not hasattr(tree, "depth_dict"):
        tree.buildHPOTree()


def node_metadata(hpo_id: str, tree: HPOTree) -> dict:
    """Structural covariates for one node. Returns a flat, JSON-serialisable dict."""
    _ensure_depths(tree)

    node = HPO_class(tree.data[hpo_id])
    in_pa = hpo_id in tree.phenotypic_abnormality

    return {
        "hpo_id": hpo_id,
        "hpo_label": node.name[0] if node.name else hpo_id,
        # ``child`` is the transitive descendant set, ``son`` the direct children.
        "subtree_size": len(node.child),
        "n_direct_children": len(node.son),
        "is_leaf": len(node.son) == 0,
        "depth_bfs": tree.depth_dict.get(hpo_id),
        "depth_long": tree.depth_long.get(hpo_id) if in_pa else None,
        "layer1_organ": sorted(tree.getLayer1HPOByHPO(hpo_id)) if in_pa else [],
        "in_phenotypic_abnormality": in_pa,
    }


def node_metadata_table(hpo_ids, tree: HPOTree) -> list[dict]:
    """``node_metadata`` for many nodes, skipping ids absent from ``hpo.json``."""
    return [node_metadata(h, tree) for h in hpo_ids if h in tree.data]


def estimated_node_cost(meta: dict, dual_prompt: bool) -> float:
    """Relative cost of scoring one node, used to balance shard assignment.

    Two terms with very different scales:

    * **Retrieval** grows with the union-context pool, which grows with ``subtree_size``
      (median 11, max 18,987 across the HCY target set, three orders of magnitude, so
      round-robin sharding would leave one job running far longer than the rest). Taken as
      ``log1p`` because the associative-max precomputation amortises the pool across nodes. The per-node marginal cost is the column slice, not a rescore.
    * **SLM calls** dominate wall-clock and are constant per node, doubling when the node
      also gets the category prompt.

    The constant 20.0 encodes "the S forward passes cost far more than one column-max". It
    only has to be roughly right for the shards to come out even.
    """
    import math

    retrieval = math.log1p(meta.get("subtree_size") or 0)
    slm = 20.0 * (2.0 if dual_prompt else 1.0)
    return retrieval + slm


def assign_shards(hpo_ids, tree: HPOTree, n_shards: int, dual_prompt_ids=None) -> list[list[str]]:
    """Partition nodes into ``n_shards`` balanced by estimated cost.

    Sorts by descending cost and deals snake-wise (0,1,..,n-1,n-1,..,1,0), the standard greedy
    partition heuristic, good enough here because the cost spread is dominated by the constant
    per-node SLM term.

    Deterministic: ties break on ``hpo_id``, so a given (id set, n_shards) always yields the
    same partition and a shard can be re-run reproducibly. The returned lists are a genuine
    partition, every input id appears in one shard.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")

    dual = set(dual_prompt_ids or ())
    metas = {m["hpo_id"]: m for m in node_metadata_table(hpo_ids, tree)}
    ordered = sorted(
        metas,
        key=lambda h: (-estimated_node_cost(metas[h], h in dual), h),
    )

    shards: list[list[str]] = [[] for _ in range(n_shards)]
    for i, hpo_id in enumerate(ordered):
        cycle, pos = divmod(i, n_shards)
        shards[pos if cycle % 2 == 0 else n_shards - 1 - pos].append(hpo_id)
    return shards
