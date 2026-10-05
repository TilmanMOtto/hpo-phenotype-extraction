"""The label strings an HPO term can legally be written as, and the map back to its identifier.

This is the dictionary that makes constrained decoding (the earlier runs) a *linking* step rather than a
*normalisation* step. When the decoder's output space is restricted to strings that are already HPO
label strings, resolving a generation to an identifier is one lookup, and the whole
free-text-to-ontology error class disappears by design.

**Why this is not** ``HPOTree.p_phrase2HPO``. That map exists and looks like it would do, and it
would silently corrupt the candidate set. Its key is ``" ".join(sorted(processStr(phrase)))`` - a
*sorted bag of processed words* - so *Hepatic steatosis* and *Steatosis hepatic* collide, and it is
populated last-write-wins over the whole ontology, so a synonym shared by two terms drops one of
them without a warning. It was built for fuzzy phrase matching, where that is the right trade. Here
the string the model emitted is the string we scored, and it must resolve to what it names.

**What the fixed release supports.** ``resources/util/hpo.json`` carries a flat ``Synonym`` list
with no exact/broad/narrow scope distinction, so "all synonyms" is the only granularity available;
``labels_only=True`` is the other end of that axis. Measured on the fixed file (fingerprint
``8330ca317e71``): 18 354 phenotypic-abnormality terms, 18 354 distinct labels with **zero**
collisions, 40 335 distinct label strings with **one**, mean 2.20 forms per term, and 8 108 terms
(44 %) carrying no synonym at all.

That one collision is why :func:`build_surface_index` maps a surface to a *list*. A dict of single
strings would be right 40 334 times out of 40 335 and wrong silently on the last one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Surface:
    """One string an HPO term may be written as.

    ``text`` is verbatim from the release, because it is what gets scored and therefore what the
    model is asked to emit. Normalisation happens only in the lookup key.
    """

    text: str
    hpo_id: str
    is_label: bool  # True for the official Name of the term, False for a synonym


def surface_key(text: str) -> str:
    """The lookup key for a label string: whitespace-collapsed, case-folded.

    weaker than ``ontology_index.normalise_phrase``, which strips punctuation and
    lemmatises. The model emits one of the strings we scored, so the only variation that needs
    absorbing is case and stray whitespace. Anything more aggressive would merge terms the release
    keeps distinct.
    """
    return " ".join(text.split()).casefold()


def build_surface_index(
    tree: HPOTree, *, restrict_to_phenotypic_abnormality: bool = True
) -> tuple[dict[str, list[Surface]], dict[str, list[Surface]]]:
    """``(surface2hpo, hpo2surfaces)`` over the fixed ontology.

    ``surface2hpo`` is keyed by :func:`surface_key` and its values are lists - see the module
    docstring for the one real collision. ``hpo2surfaces`` is keyed by identifier, label first then
    synonyms in release order, deduplicated case-insensitively within the term (six terms in the
    release carry a form differing from another only in case).

    Iteration is sorted by identifier and stable within a node, so two runs over the same file
    produce byte-identical candidate lists. The driver's resume logic depends on that: it trusts
    that sentence *i* had the same candidates the last time the job ran.
    """
    scorable = (
        set(tree.phenotypic_abnormality)
        if restrict_to_phenotypic_abnormality
        else set(tree.data)
    )

    surface2hpo: dict[str, list[Surface]] = {}
    hpo2surfaces: dict[str, list[Surface]] = {}

    for hpo_id in sorted(scorable):
        node = HPO_class(tree.data[hpo_id])
        seen: set[str] = set()
        forms: list[Surface] = []
        pairs = [(n, True) for n in node.name] + [(s, False) for s in node.synonym]
        for text, is_label in pairs:
            if not text or not text.strip():
                continue
            key = surface_key(text)
            if key in seen:
                continue
            seen.add(key)
            forms.append(Surface(text=" ".join(text.split()), hpo_id=hpo_id, is_label=is_label))
        if not forms:
            continue
        hpo2surfaces[hpo_id] = forms
        for surface in forms:
            surface2hpo.setdefault(surface_key(surface.text), []).append(surface)

    n_collisions = sum(1 for v in surface2hpo.values() if len({s.hpo_id for s in v}) > 1)
    logger.info(
        "surface index: %d terms, %d distinct surfaces, %d colliding",
        len(hpo2surfaces), len(surface2hpo), n_collisions,
    )
    return surface2hpo, hpo2surfaces


def expand_candidates(
    hpo_ids, hpo2surfaces: dict[str, list[Surface]], *, labels_only: bool = False
) -> list[Surface]:
    """The candidate strings for a retrieved term list, in retrieval order.

    Order is essential twice over: it is the order the score rows are written in, which is what
    makes ``retrieval_rank`` recoverable offline, and it is the order the an earlier exploratory run menu renders in.

    A surface that two retrieved terms happen to share is emitted once per term. Aggregation needs
    the row attributed to *each* term, and the scorer deduplicates identical strings internally,
    not paying for the same forward pass twice.
    """
    out: list[Surface] = []
    for hpo_id in hpo_ids:
        for surface in hpo2surfaces.get(hpo_id, ()):
            if labels_only and not surface.is_label:
                continue
            out.append(surface)
    return out


def resolve_surface(text: str, surface2hpo: dict[str, list[Surface]]) -> list[str]:
    """Identifiers a generated string names, or ``[]``. Never guesses."""
    return [s.hpo_id for s in surface2hpo.get(surface_key(text), ())]


if __name__ == "__main__":  # pragma: no cover - GPU-free self-check
    import statistics

    _tree = HPOTree()
    _s2h, _h2s = build_surface_index(_tree)
    print(f"ok   {len(_h2s)} terms, {len(_s2h)} distinct surfaces")

    _colliding = {k: v for k, v in _s2h.items() if len({s.hpo_id for s in v}) > 1}
    assert len(_colliding) <= 2, _colliding
    print(f"ok   {len(_colliding)} colliding surface(s): " + "; ".join(
        f"{k!r} -> {sorted({s.hpo_id for s in v})}" for k, v in _colliding.items()))

    _label_clashes = [k for k, v in _s2h.items() if len({s.hpo_id for s in v if s.is_label}) > 1]
    assert not _label_clashes, _label_clashes
    print("ok   canonical labels are unique across the ontology")

    _counts = [len(v) for v in _h2s.values()]
    print(f"ok   surface forms per term: mean {statistics.mean(_counts):.2f} "
          f"median {statistics.median(_counts)} max {max(_counts)}")

    assert resolve_surface("Enlarged liver", _s2h) == ["HP:0002240"]
    assert resolve_surface("  enlarged   LIVER ", _s2h) == ["HP:0002240"]
    assert resolve_surface("not a phenotype at all", _s2h) == []
    print("ok   lookup resolves a synonym, ignores case/whitespace, and refuses to guess")

    _cands = expand_candidates(["HP:0002240", "HP:0001397"], _h2s)
    assert _cands[0].text == "Hepatomegaly" and _cands[0].is_label
    assert "Fatty liver" in [c.text for c in _cands]
    _lonly = expand_candidates(["HP:0002240", "HP:0001397"], _h2s, labels_only=True)
    assert [c.text for c in _lonly] == ["Hepatomegaly", "Hepatic steatosis"]
    print(f"ok   expansion: {len(_cands)} surfaces for 2 terms, {len(_lonly)} labels-only")
