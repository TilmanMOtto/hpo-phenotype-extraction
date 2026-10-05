"""The R3 query index, one vector per HPO term, straight from the ontology's own wording.

The retrieval-curve analysis measured four ways of representing a query term against the report's sentences. Two of
them need a generator model: ``R1`` is the term's ~36 LLM-written synthetic-sentence sentences and ``R1u``
pools those with every descendant's. The other two need none at all::

    R3    embed(label + " " + definition + " " + synonyms)            one vector per term
    R3u   R3(v) max-pooled over desc_closure(v)                       one vector per pooled term

This module is the R3 side of that 2x2, lifted out of
``experiments/04_treephenorag/retrieval_curves/query_sets.py`` so the retrieval experiment and the
TreePhenoRAG cache run (``treephenorag_scores_terminfo``) embed **the identical string**. The retrieval-curve analysis imports these names
rather than keeping a second copy: an R3 condition that drifted from the R3 index a traversal was built
on would make the two experiments incomparable while still looking fine in both.

``R3u`` itself is not a separate object. It is ``UnionScorer`` over this index with the same
descendant-closure pools ``R1u`` uses, so the pooling code path is shared and ``R3u - R3`` is
the intervention ``R1u - R1`` is.
"""

from __future__ import annotations

import numpy as np

from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree

#: The exact string R3 embeds. Recorded in provenance so the condition is reproducible.
R3_TEMPLATE = "{label} {definition} {synonyms}"
R3_SYNONYM_JOIN = " "


def r3_text(tree: HPOTree, hpo: str) -> str:
    """``label + " " + definition + " " + synonyms`` for one term, empty parts dropped.

    Every query term must get a string: a term skipped here would be scored as a guaranteed miss
    and the condition would be measuring ontology *coverage*, not ontology *wording*. Terms with
    no ``def:`` and no synonym fall back to the label alone, and both counts are recorded in the
    provenance dict so the condition's ceiling is visible.

    Note the shape, which is the whole point of the condition: **one** string, hence **one** vector.
    ``tree_experiment._ontology_context_sentences`` builds the same material as a *list* of
    name/synonym/definition strings, i.e. several vectors that a max then pools over, a different
    representation that happens to read from the same fields.
    """
    node = HPO_class(tree.data[hpo])
    label = node.name[0] if node.name else hpo
    definition = node.definition[0] if node.definition else ""
    synonyms = R3_SYNONYM_JOIN.join(s for s in node.synonym if s and s != label)
    return " ".join(p.strip() for p in (label, definition, synonyms) if p.strip())


def build_r3_index(tree: HPOTree, hpos: list[str], encode) -> tuple[np.ndarray, dict]:
    """``(n_hpos x dim)`` L2-normalised matrix aligned to *hpos*, plus provenance counts.

    *hpos* is the whole **closure universe**, not just the query terms: R3u pools a query's vector
    with every descendant's, so a descendant needs a vector even though it is never itself a query.
    Embedding ~18k short strings is one cheap GPU pass, and doing it over the universe is what lets
    R3/R3u come out of a single ``UnionScorer`` as R1/R1u do.
    """
    texts = [r3_text(tree, h) for h in hpos]
    nodes = [HPO_class(tree.data[h]) for h in hpos]
    n_no_def = sum(1 for n in nodes if not n.definition)
    n_no_syn = sum(1 for n in nodes if not n.synonym)
    n_label_only = sum(1 for n in nodes if not n.definition and not n.synonym)
    vectors = np.asarray(encode(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    prov = {
        "template": R3_TEMPLATE,
        "synonym_join": R3_SYNONYM_JOIN,
        "n_terms": len(hpos),
        "n_without_definition": n_no_def,
        "n_without_synonym": n_no_syn,
        "n_label_only": n_label_only,
        "median_chars": int(np.median([len(t) for t in texts])) if texts else 0,
        "examples": {h: texts[i] for i, h in enumerate(hpos[:3])},
    }
    return vectors, prov
