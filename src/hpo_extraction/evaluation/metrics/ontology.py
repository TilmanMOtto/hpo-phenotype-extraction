"""Ontology graph primitives, the only module in this package that touches ``HPOTree``.

Every equation in ``thesis/sections/sceleton.tex`` that mentions :math:`\\mathcal{H} = (V, E)`,
:math:`\\mathrm{An}(v)`, a subtree, a depth or a path distance goes through :class:`OntologyView`.
Concentrating the graph access here means the *closure convention*, the single choice that decides
how inflated :math:`hF` looks next to :math:`F_1`, is defined in one place and can be
checked against the LaTeX in one read.

**The closure convention.** The skeleton defines

    ":math:`\\mathrm{An}(v)` for the set of ancestors of :math:`v` in :math:`\\mathcal{H}`
    including :math:`v` itself but excluding the root"

and this package reads "the root" as **both universal nodes**: ``HP:0000001`` (*All*) and
``HP:0000118`` (*Phenotypic abnormality*). Layer-1 organ systems are kept. This counts because the
repo's precomputed ``Father`` sets *do* contain both, ``Father`` of ``HP:0001250`` (*Seizure*) is
``{HP:0000001, HP:0000118, HP:0000707, HP:0012638}``, so without the subtraction every term in
every report would contribute two free true positives to :math:`hP` and :math:`hR`.

**Resolution vs hallucination.** :meth:`OntologyView.resolve` canonicalises a code (whitespace,
alt/obsolete id remap) and returns ``None`` for anything outside the phenotypic-abnormality subtree.
That is not enough for the §Severity-of-near-misses taxonomy, which needs *hallucinations* (terms
absent from the ontology altogether) separated from terms that merely fall outside the scored
subtree, so :meth:`in_ontology` is provided alongside and the two are used together in
``errors.classify_false_positive``.

Nothing here does IO and nothing here is specific to a dataset or a run.
"""

from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from typing import Iterable

# The two nodes every phenotypic-abnormality term descends from. Excluded from every ancestor
# closure (see the module docstring). They are shared by design and carry no information.
UNIVERSAL_NODES = frozenset({"HP:0000001", "HP:0000118"})


class OntologyView:
    """A read-only view over an :class:`~hpo_extraction.ontology.hpo_tree.HPOTree` with cached graph queries.

    Construct once per process and pass it to every metric function. All lookups are memoised, so
    the per-report loops in :mod:`~hpo_extraction.evaluation.metrics.hierarchy` and
    :mod:`~hpo_extraction.evaluation.metrics.errors` pay the graph cost once rather than once per report.

    Args:
        tree: an ``HPOTree`` (or any object exposing ``data``, ``alt_id_dict``,
            ``phenotypic_abnormality``, ``depth_dict``, ``layer1_set`` and ``root``).
            ``buildHPOTree()`` is called if ``depth_dict`` is missing, since ``HPOTree.__init__``
            does not populate it.

    Attributes:
        tree: the wrapped tree, for callers that need an escape hatch.
        scorable: the node set every metric is defined over, the phenotypic-abnormality subtree
            minus the universal nodes.
    """

    def __init__(self, tree):
        if not hasattr(tree, "depth_dict"):
            tree.buildHPOTree()
        self.tree = tree
        self._data = tree.data
        self._alt_id = getattr(tree, "alt_id_dict", {})
        self._subtree = set(tree.phenotypic_abnormality)
        self._layer1 = set(getattr(tree, "layer1_set", ()))
        self.scorable = self._subtree - UNIVERSAL_NODES

        # Bound methods cannot be decorated with lru_cache at class level without leaking the
        # instance into a module-level cache, so the caches are created per instance.
        self.ancestors = lru_cache(maxsize=None)(self._ancestors)
        self.descendants_or_self = lru_cache(maxsize=None)(self._descendants_or_self)
        self.direct_parents = lru_cache(maxsize=None)(self._direct_parents)
        self.direct_children = lru_cache(maxsize=None)(self._direct_children)
        self.layer1 = lru_cache(maxsize=None)(self._layer1_of)

    # ── Code canonicalisation ────────────────────────────────────────────────

    def resolve(self, code: str) -> str | None:
        """Canonical scorable node for a raw code, or ``None`` if it is not scorable.

        Handles surrounding whitespace and alt/obsolete ids (remapped through ``alt_id_dict``).
        Returns ``None`` for the universal nodes and for anything outside the
        phenotypic-abnormality subtree, including codes that are genuine HPO terms but sit
        elsewhere in the ontology (e.g. inheritance or clinical-modifier terms).
        """
        if not code:
            return None
        code = str(code).strip()
        if code not in self._data and code in self._alt_id:
            code = self._alt_id[code]
        return code if code in self.scorable else None

    def in_ontology(self, code: str) -> bool:
        """Whether a raw code names *any* term the ontology knows about.

        A code that fails :meth:`resolve` but passes this is out-of-subtree. One that fails both is
        a **hallucination** in the sense of §Severity of near-misses.
        """
        if not code:
            return False
        code = str(code).strip()
        return code in self._data or code in self._alt_id

    def resolve_set(self, codes: Iterable[str]) -> tuple[set[str], list[str]]:
        """``(scorable_codes, dropped_raw_codes)``, resolution with the losses kept visible.

        Metric functions call this, not silently discarding, so a caller can report how many
        ground truth or predicted codes were obsolete/out-of-subtree instead of absorbing them as misses.
        """
        keep: set[str] = set()
        dropped: list[str] = []
        for c in codes:
            r = self.resolve(c)
            if r is None:
                dropped.append(c)
            else:
                keep.add(r)
        return keep, dropped

    # ── Ancestry and descent ─────────────────────────────────────────────────

    def _ancestors(self, code: str) -> frozenset[str]:
        """:math:`\\mathrm{An}(v)`, ``v`` plus all transitive ancestors, minus the universal nodes.

        In a DAG this is the union over *all* root-to-``v`` paths, which is what the precomputed
        ``Father`` set already holds. Multi-parent terms therefore contribute proportionally more
        ancestors. That weighting is intentional (it is what the skeleton's §hier-metrics note is
        about) and is quantified by :func:`ancestor_count_distribution` in
        :mod:`~hpo_extraction.evaluation.metrics.hierarchy`.
        """
        r = self.resolve(code)
        if r is None:
            return frozenset()
        fathers = set(self._data[r].get("Father", {}))
        return frozenset(({r} | fathers) & self.scorable)

    def ancestors_of_set(self, codes: Iterable[str]) -> set[str]:
        """:math:`\\mathrm{An}(Y) = \\bigcup_{v \\in Y} \\mathrm{An}(v)`."""
        out: set[str] = set()
        for c in codes:
            out |= self.ancestors(c)
        return out

    def _descendants_or_self(self, code: str) -> frozenset[str]:
        """``{v}`` plus every transitive descendant, the subtree rooted at ``v``."""
        r = self.resolve(code)
        if r is None:
            return frozenset()
        children = set(self._data[r].get("Child", {}))
        return frozenset(({r} | children) & self.scorable)

    def _direct_parents(self, code: str) -> frozenset[str]:
        """Direct ``is_a`` parents inside the scorable subtree."""
        r = self.resolve(code)
        if r is None:
            return frozenset()
        return frozenset(p for p in self._data[r].get("Is_a", ()) if p in self.scorable)

    def _direct_children(self, code: str) -> frozenset[str]:
        """Direct children (``Son``) inside the scorable subtree."""
        r = self.resolve(code)
        if r is None:
            return frozenset()
        return frozenset(s for s in self._data[r].get("Son", {}) if s in self.scorable)

    def is_ancestor(self, candidate: str, of: str) -> bool:
        """Whether ``candidate`` is a strict ancestor of ``of``."""
        a, b = self.resolve(candidate), self.resolve(of)
        return a is not None and b is not None and a != b and a in self.ancestors(b)

    # ── Depth and organ system ───────────────────────────────────────────────

    def depth(self, code: str) -> int | None:
        """Shortest-path depth from ``HP:0000118``: layer-1 organ systems are depth 1.

        This is ``HPOTree.depth_dict`` (BFS), the depth used for depth-stratified reporting and for
        the blocking-depth diagnostic. It is *not* the longest-path depth
        ``HPOTree.depth_long``, which exists only to keep Wu-Palmer similarity below 1 and is not
        the quantity the skeleton's depth tables refer to.
        """
        r = self.resolve(code)
        if r is None:
            return None
        return self.tree.depth_dict.get(r)

    def _layer1_of(self, code: str) -> str | None:
        """A single layer-1 (organ-system) id for a term, or ``None``.

        A DAG term may sit under several organ systems. The lexicographically smallest is taken so
        that per-subtree grouping is stable and every term lands in one bucket.
        """
        r = self.resolve(code)
        if r is None:
            return None
        if r in self._layer1:
            return r
        shared = sorted(self._layer1 & set(self._data[r].get("Father", {})))
        return shared[0] if shared else None

    def layer1_ancestors(self, code: str) -> frozenset[str]:
        """**Every** depth-one (organ-system) ancestor of a term, not just the canonical one.

        :meth:`layer1` picks one so that grouping is a partition. The false-positive taxonomy needs
        the opposite: a prediction and some annotated term sharing *any* depth-one ancestor
        satisfies it, and a DAG term routinely sits under several organ systems at once.
        """
        r = self.resolve(code)
        if r is None:
            return frozenset()
        return frozenset(self.ancestors(r) & self._layer1)

    # ── Distance ─────────────────────────────────────────────────────────────

    def undirected_distance(self, a: str, b: str, max_hops: int = 64) -> int | None:
        """Shortest **undirected** path length between two terms, or ``None`` if unreachable.

        This is :math:`\\mathrm{dist}_{\\mathcal{H}}` of the skeleton's eq. (5). Edges are traversed
        in both directions, so a parent and a child are at distance 1 and two siblings sharing a
        parent are at distance 2. Bidirectional BFS, since the ontology is wide and a naive
        single-source BFS over ~19k nodes per false positive is the dominant cost of the error
        analysis.

        Args:
            max_hops: safety cap; ``None`` is returned if the two terms are further apart than
                this. The scorable subtree is connected through the layer-1 organ systems, so the
                true diameter is well under the default.
        """
        ra, rb = self.resolve(a), self.resolve(b)
        if ra is None or rb is None:
            return None
        if ra == rb:
            return 0

        # Two frontiers grown alternately. The distance is found when they touch.
        seen_a: dict[str, int] = {ra: 0}
        seen_b: dict[str, int] = {rb: 0}
        queue_a: deque[str] = deque([ra])
        queue_b: deque[str] = deque([rb])

        while queue_a or queue_b:
            # Always expand the smaller frontier, the standard bidirectional-BFS balance rule.
            if queue_a and (not queue_b or len(queue_a) <= len(queue_b)):
                queue, seen, other = queue_a, seen_a, seen_b
            else:
                queue, seen, other = queue_b, seen_b, seen_a

            for _ in range(len(queue)):
                node = queue.popleft()
                if seen[node] >= max_hops:
                    continue
                for nb in self.neighbours(node):
                    if nb in seen:
                        continue
                    seen[nb] = seen[node] + 1
                    if nb in other:
                        return seen[nb] + other[nb]
                    queue.append(nb)
        return None

    def neighbours(self, code: str) -> frozenset[str]:
        """Undirected neighbourhood: direct parents ∪ direct children."""
        return self.direct_parents(code) | self.direct_children(code)


# ── Cohort-level structural summaries ────────────────────────────────────────

def ancestor_count_distribution(gold_sets: Iterable[Iterable[str]], view: OntologyView) -> dict:
    """Distribution of :math:`|\\mathrm{An}(v)|` over the annotated terms of a cohort.

    §hier-metrics promises this explicitly, "we report the distribution of
    :math:`|\\mathrm{An}(v)|` over the ground-truth set so that this weighting is auditable", because a
    multi-parent term is weighted more heavily than a single-parent term at equal depth under
    ancestor closure.

    Terms are counted once per (report, term) occurrence, matching the unit the closure metrics
    actually pool over.

    Returns:
        ``{"n_terms", "mean", "median", "min", "max", "histogram", "multi_parent_fraction"}``.
        ``histogram`` maps :math:`|\\mathrm{An}(v)|` to its count.
    """
    counts: list[int] = []
    n_multi_parent = 0
    for gold in gold_sets:
        for code in gold:
            r = view.resolve(code)
            if r is None:
                continue
            counts.append(len(view.ancestors(r)))
            if len(view.direct_parents(r)) > 1:
                n_multi_parent += 1

    n = len(counts)
    if n == 0:
        return {"n_terms": 0, "mean": float("nan"), "median": float("nan"),
                "min": None, "max": None, "histogram": {}, "multi_parent_fraction": float("nan")}

    ordered = sorted(counts)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "n_terms": n,
        "mean": sum(counts) / n,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "histogram": dict(sorted(Counter(counts).items())),
        "multi_parent_fraction": n_multi_parent / n,
    }
