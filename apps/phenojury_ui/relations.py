"""How wrong is a false positive?, the FP taxonomy and near-miss distance.

An open-set method is scored on exact term identity, so predicting "Seizure" when the annotation
says "Focal seizure" is a false positive *and* a false negative, scoring identically to a term from
an unrelated organ system. That is the right scoring rule and the wrong summary: the two errors say
opposite things about the model. This module separates them.

Six classes, ordered near-miss → hallucination (the same order and vocabulary as
``app/tree_ui/theme.ERROR_ORDER``, so a reader moving between the two apps reads the same scale):

``ancestor``     the prediction is a parent of an annotated term, too general
``descendant``   the prediction is a child of an annotated term, too specific
``sibling``      shares a direct parent with an annotated term, right neighbourhood, wrong branch
``same_system``  shares a layer-1 organ system and nothing closer
``unrelated``    shares only the ontology root
``no_gt``        the report has no annotations at all, so nothing can be near

Only the ontology dictionaries are used (``Father``/``Child`` are the transitive closures,
``Is_a``/``Son`` the direct edges), which both :class:`~hpo_extraction.ontology.hpo_tree.HPOTree` and the test
suite's ``ToyTree`` expose identically, so the classification is unit-testable on a ten-node graph
instead of on 19k real terms.
"""

from __future__ import annotations

from collections import deque

#: Near-miss → hallucination. The order *is* the severity encoding. Charts must preserve it.
RELATION_ORDER = ("ancestor", "descendant", "sibling", "same_system", "unrelated", "no_gt")

RELATION_HELP = {
    "ancestor": "predicted a parent of an annotated term, too general",
    "descendant": "predicted a child of an annotated term, too specific",
    "sibling": "shares a direct parent with an annotated term, wrong branch",
    "same_system": "same layer-1 organ system, nothing closer",
    "unrelated": "shares only the ontology root",
    "no_gt": "the report has no annotations, nothing to be near",
}

#: Classes that mean "the model saw the right thing and named it at the wrong granularity".
NEAR_MISS = ("ancestor", "descendant", "sibling")

#: Hops beyond which the near-miss distance stops being informative, and the BFS stops paying for
#: it. Descending four levels from an annotated term already covers thousands of real HPO nodes.
MAX_HOPS = 4

_BUDGET = 60_000


def canonical(tree, hpo_id: str) -> str:
    """Resolve an alt/obsolete id to its current one, so a merged term is not read as unknown."""
    return getattr(tree, "alt_id_dict", {}).get(hpo_id, hpo_id)


def label(tree, hpo_id: str) -> str:
    """Human-readable term name, falling back to the id for anything outside the ontology."""
    node = tree.data.get(canonical(tree, hpo_id))
    if not node:
        return hpo_id
    names = node.get("Name") or []
    return names[0] if names else hpo_id


def definition(tree, hpo_id: str) -> str:
    """The HPO term's definition, or ``""`` where the ontology has none.

    ``hpo.json`` stores ``Def`` as a list of at most one string; 16.4k of 19.4k terms have one.
    """
    if tree is None:
        return ""
    node = tree.data.get(canonical(tree, hpo_id))
    defs = (node or {}).get("Def") or []
    return defs[0] if defs else ""


def synonyms(tree, hpo_id: str) -> list[str]:
    """The term's exact and related synonyms, or ``[]``."""
    if tree is None:
        return []
    node = tree.data.get(canonical(tree, hpo_id))
    syns = (node or {}).get("Synonym") or []
    return list(syns)


def tooltip(tree, hpo_id: str) -> str:
    """The hover text for a term: what it means and what else it is called.

    Plain text with newlines rather than markup, because it goes in a native ``title=`` attribute.
    That is this app's only tooltip mechanism, ``dash-bootstrap-components`` is not a dependency
    and adding one for a hover would be a poor trade, and it is what ``views/patient.py`` already
    uses on every highlighted PhenoBERT span.
    """
    if tree is None:
        return hpo_id
    name = label(tree, hpo_id)
    head = hpo_id if name == hpo_id else f"{hpo_id} · {name}"
    parts = [head]
    text = definition(tree, hpo_id)
    if text:
        parts.append(text)
    syns = synonyms(tree, hpo_id)
    if syns:
        parts.append("Also known as: " + ", ".join(syns))
    if len(parts) == 1:
        parts.append("(no definition or synonyms in this HPO release)")
    return "\n\n".join(parts)


def classify(tree, hpo_id: str, gold: set[str]) -> str:
    """The relation class of one false positive against one report's annotated set."""
    if not gold:
        return "no_gt"
    node_id = canonical(tree, hpo_id)
    node = tree.data.get(node_id)
    if node is None:
        return "unrelated"

    gold_ids = {canonical(tree, g) for g in gold}
    ancestors = set(node.get("Father") or ())
    descendants = set(node.get("Child") or ())
    if gold_ids & descendants:
        return "ancestor"
    if gold_ids & ancestors:
        return "descendant"

    parents = set(node.get("Is_a") or ())
    for gold_id in gold_ids:
        gold_node = tree.data.get(gold_id)
        if gold_node is not None and parents & set(gold_node.get("Is_a") or ()):
            return "sibling"

    systems = _systems(tree, node_id)
    if systems and any(systems & _systems(tree, g) for g in gold_ids):
        return "same_system"
    return "unrelated"


def _systems(tree, hpo_id: str) -> set[str]:
    """The layer-1 organ systems above a term (the term itself if it is one)."""
    layer1 = getattr(tree, "layer1_set", set())
    if hpo_id in layer1:
        return {hpo_id}
    node = tree.data.get(hpo_id)
    if node is None:
        return set()
    return layer1 & set(node.get("Father") or ())


def distances_from(tree, gold: set[str], max_hops: int = MAX_HOPS) -> dict[str, int]:
    """``{term: hops}`` for everything within *max_hops* of the annotated set, undirected.

    Multi-source BFS over ``Is_a``/``Son``, computed once per report and shared by every false
    positive in it. Undirected because the distance that counts to a reader is "how far off the
    mark", which has no direction, and because a DAG distance cannot be read off the depths anyway
    once a term has two parents.
    """
    dist: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for term in gold:
        node_id = canonical(tree, term)
        if node_id in tree.data and node_id not in dist:
            dist[node_id] = 0
            queue.append((node_id, 0))

    visited = 0
    while queue:
        node_id, hops = queue.popleft()
        if hops >= max_hops:
            continue
        node = tree.data.get(node_id)
        if node is None:
            continue
        for neighbour in list(node.get("Is_a") or ()) + list(node.get("Son") or ()):
            if neighbour in dist or neighbour not in tree.data:
                continue
            dist[neighbour] = hops + 1
            visited += 1
            if visited > _BUDGET:
                return dist
            queue.append((neighbour, hops + 1))
    return dist


def classify_report(tree, predicted: set[str], gold: set[str]) -> list[dict]:
    """One row per false positive in a report: ``hpo_id, label, relation, distance``.

    ``distance`` is ``None`` when the term is further than :data:`MAX_HOPS` from everything
    annotated, reported as "> 4", not as a number the BFS never actually measured.
    """
    false_positives = sorted(set(predicted) - set(gold))
    if not false_positives:
        return []
    dist = distances_from(tree, set(gold)) if gold else {}
    return [
        {
            "hpo_id": hpo_id,
            "label": label(tree, hpo_id),
            "relation": classify(tree, hpo_id, set(gold)),
            "distance": dist.get(canonical(tree, hpo_id)),
        }
        for hpo_id in false_positives
    ]
