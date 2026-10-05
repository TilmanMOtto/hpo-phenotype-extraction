"""A ten-node ontology small enough to work every metric out by hand.

The whole point of ``hpo_extraction.evaluation.metrics`` is that its numbers can be checked against the
formulas in ``thesis/sections/sceleton.tex``. That check is only meaningful if the test inputs are
small enough to compute on paper, so every hierarchy, error and traversal test runs against this
graph rather than against the real 19k-node HPO.

::

                        HP:0000118  R   root (universal, excluded from every closure)
                        /                     \\
              HP:0000707  A                    HP:0000924  G      depth 1  (layer-1)
              /          \\                          |
     HP:0012638  B     HP:0002011  E          HP:0011842  H       depth 2
       /   |    \\             |                  /
 HP:0001250  HP:0002376  \\   HP:0002060  F      /                 depth 3
      C          D        \\       |            /
                           `--- HP:0007359  M -'                  depth 3, TWO parents (B and H)

Deliberate features, each exercising one convention the package had to settle:

``M`` has two parents
    Its ancestor closure ``{M, B, A, H, G}`` has five members against ``C``'s three, which is the
    multi-parent weighting §hier-metrics says must be auditable. It also creates a DAG shortcut:
    ``C -> B -> M -> H`` is three hops while the tree route ``C -> B -> A -> G -> H`` is four, so
    the undirected distance of eq. (5) cannot be derived from depths.
``HP:0009999``
    An alt/obsolete id remapping to ``C``, so code canonicalisation is exercised.
``HP:0000005``
    A real ontology term *outside* the phenotypic-abnormality subtree, so the taxonomy's
    ``out_of_subtree`` bucket is distinguishable from ``hallucination``.
``HP:9999999``
    Named by no test fixture at all, the hallucination case.

Reference values, computed by hand and relied on by the tests::

    An(C) = {C, B, A}           |An(C)| = 3
    An(D) = {D, B, A}           |An(D)| = 3
    An(M) = {M, B, A, H, G}     |An(M)| = 5
    An(F) = {F, E, A}           |An(F)| = 3
    depth: A=G=1, B=E=H=2, C=D=F=M=3
    dist(C, D) = 2   (siblings via B)
    dist(B, C) = 1
    dist(C, H) = 3   (via M, not 4 via A/G)
    dist(C, F) = 4   (C-B-A-E-F)
"""

from __future__ import annotations

from collections import deque

ROOT = "HP:0000118"

#: Human-readable aliases, so a failing assertion names a node, not a number.
A = "HP:0000707"   # layer-1: "Abnormality of the nervous system"
G = "HP:0000924"   # layer-1: "Abnormality of the skeletal system"
B = "HP:0012638"   # under A
E = "HP:0002011"   # under A
H = "HP:0011842"   # under G
C = "HP:0001250"   # under B, "Seizure"
D = "HP:0002376"   # under B
F = "HP:0002060"   # under E
M = "HP:0007359"   # under B *and* H, the multi-parent node

OUT_OF_SUBTREE = "HP:0000005"   # a real term outside the phenotypic-abnormality subtree
OBSOLETE_C = "HP:0009999"       # alt id remapping to C
HALLUCINATION = "HP:9999999"    # not in the ontology at all

#: ``{node: [direct parents]}``, the only hand-written structure. Everything else is derived.
PARENTS: dict[str, list[str]] = {
    A: [ROOT],
    G: [ROOT],
    B: [A],
    E: [A],
    H: [G],
    C: [B],
    D: [B],
    F: [E],
    M: [B, H],
}

NAMES: dict[str, str] = {
    ROOT: "Phenotypic abnormality",
    A: "Abnormality of the nervous system",
    G: "Abnormality of the skeletal system",
    B: "Abnormal nervous system physiology",
    E: "Abnormal nervous system morphology",
    H: "Abnormality of the vertebral column",
    C: "Seizure",
    D: "Abnormal fear/anxiety-related behavior",
    F: "Abnormal cerebral morphology",
    M: "Abnormality of the spinal cord",
    OUT_OF_SUBTREE: "Mode of inheritance",
}


class ToyTree:
    """The minimal surface :class:`~hpo_extraction.evaluation.metrics.ontology.OntologyView` consumes.

    ``OntologyView`` needs ``data``, ``alt_id_dict``, ``phenotypic_abnormality``,
    ``layer1_set``, ``root`` and ``depth_dict`` (or a ``buildHPOTree`` that produces it). Providing
    that surface here, not instantiating a real ``HPOTree`` keeps the metric tests free of
    the ``nltk``/``stanza`` import chain ``hpo_extraction.ontology.hpo_tree`` carries, and documents the contract
    the view actually depends on.

    ``data`` matches the ``hpo.json`` schema: ``Is_a`` holds direct parents, ``Son`` direct
    children, and ``Father`` / ``Child`` the transitive closures, all derived here so they cannot
    drift out of sync with :data:`PARENTS`.
    """

    def __init__(self):
        self.root = ROOT
        children: dict[str, list[str]] = {n: [] for n in {ROOT, *PARENTS}}
        for node, parents in PARENTS.items():
            for p in parents:
                children[p].append(node)

        ancestors = {n: self._closure(n, PARENTS) for n in children}
        descendants = {n: self._closure(n, children) for n in children}

        self.data: dict[str, dict] = {}
        for node in children:
            self.data[node] = {
                "Id": node,
                "Name": [NAMES.get(node, node)],
                "Alt_id": [OBSOLETE_C] if node == C else [],
                "Def": [], "Comment": [], "Synonym": [], "Xref": [],
                "Is_a": sorted(PARENTS.get(node, [])),
                "Father": {a: True for a in sorted(ancestors[node])},
                "Child": {d: True for d in sorted(descendants[node])},
                "Son": {s: True for s in sorted(children[node])},
            }
        # A real term outside the phenotypic-abnormality subtree: present in `data`, absent from
        # `phenotypic_abnormality`, which is what separates "out of subtree" from "hallucination".
        self.data[OUT_OF_SUBTREE] = {
            "Id": OUT_OF_SUBTREE, "Name": [NAMES[OUT_OF_SUBTREE]], "Alt_id": [],
            "Def": [], "Comment": [], "Synonym": [], "Xref": [],
            "Is_a": [], "Father": {}, "Child": {}, "Son": {},
        }

        self.alt_id_dict = {OBSOLETE_C: C}
        # Mirrors HPOTree: the subtree is the root's transitive children, *plus* the root itself.
        self.phenotypic_abnormality = set(descendants[ROOT]) | {ROOT}
        self.phenotypic_abnormalityNT = set(self.phenotypic_abnormality)
        self.layer1_set = set(children[ROOT])
        self.buildHPOTree()

    def getAllFatherHPOByHPO(self, hpo_num: str) -> set[str]:  # noqa: N802 - HPOTree's name
        """Transitive ancestors, *including* the two universal nodes, HPOTree's own semantics.

        ``OntologyView`` does not use this. The legacy ``hpo_extraction.evaluation.tree_metrics`` cross-check does,
        and it is the root-inclusive convention that check exists to contrast against.
        """
        if hpo_num not in self.phenotypic_abnormalityNT:
            return set()
        return set(self.data[hpo_num]["Father"])

    @staticmethod
    def _closure(start: str, edges: dict[str, list[str]]) -> set[str]:
        """Transitive closure of ``start`` under ``edges``, excluding ``start`` itself."""
        seen: set[str] = set()
        queue = deque(edges.get(start, []))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(edges.get(node, []))
        return seen

    def buildHPOTree(self) -> None:  # noqa: N802 - matches the HPOTree method name
        """Breadth-first depth from the root, root at 0, the same convention as ``HPOTree``."""
        self.depth_dict: dict[str, int] = {ROOT: 0}
        queue = deque([ROOT])
        while queue:
            node = queue.popleft()
            for child in self.data[node]["Son"]:
                if child not in self.depth_dict:
                    self.depth_dict[child] = self.depth_dict[node] + 1
                    queue.append(child)


def build_toy_tree() -> ToyTree:
    """A fresh :class:`ToyTree`. Cheap enough to build per test. No caching, no shared state."""
    return ToyTree()


def toy_children_map() -> tuple[dict[str, list[str]], list[str]]:
    """``(children_map, roots)`` for the traversal metrics, in ``hpo_extraction.treephenorag.traversal``'s shape.

    The root term itself is not a traversable node, the search starts at the layer-1 organ
    systems, as ``hpo_extraction.treephenorag.traversal.build_children_map`` does.
    """
    children: dict[str, list[str]] = {}
    for node, parents in PARENTS.items():
        children.setdefault(node, [])
        for p in parents:
            if p != ROOT:
                children.setdefault(p, []).append(node)
    return {k: sorted(v) for k, v in children.items()}, sorted([A, G])
