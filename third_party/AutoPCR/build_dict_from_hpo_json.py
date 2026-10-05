"""Build AutoPCR's ontology dictionary from the HPO file this repository uses.

Added for this repository, not part of upstream AutoPCR. See PATCHES.md.

AutoPCR's ``build_dict.py`` parses an ``hp.obo`` download (upstream ships ``hp_20240208.obo``).
This repository fixes its ontology **by content hash**: ``resources/util/hpo.json``, fingerprint
``8330ca317e71``, and every recall ceiling in the thesis is a property of that file (see
``resources/util/PROVENANCE.md``). Letting AutoPCR answer from a different HPO release would put a
confound in the middle of the method comparison: it could win or lose terms that do not exist for
any other method.

So this module replaces one thing only: where the terms come from. Everything downstream
(tokenisation, lemmatisation, the abbreviation rule, the mapping files) is upstream's own code,
called directly, so the artifact formats cannot drift from what ``nn_model2`` / ``build_index``
expect.

It writes, into ``--output``:

===========================  ====================================================================
``obo.json``                 upstream's per-term record (name/synonyms as ``[surface, lemma]``)
``lable.vocab``              the HP:0000118 subtree, one id per line, plus upstream's ``HP:None``
``abbr.json``                all-caps short synonyms -> id (upstream's rule, applied verbatim)
``noabb_lemma.dic``          every surface/lemma form, ordered by token count
``word_id_map.json``         written by upstream ``build_dict.word_hpo_map``
``firstword_id_map.json``    ditto, restricted to the subtree
``id_word_map.json``         written by upstream ``build_dict.hpo_word_map``
``alt_hpoid.json``           written by upstream ``build_dict.alt_hpo``
===========================  ====================================================================

Then run upstream ``build_index.py`` over the same folder for the SapBERT/FAISS index.

Usage::

    python src/AutoPCR/build_dict_from_hpo_json.py -o src/AutoPCR/dict/HPO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_AUTOPCR_DIR = Path(__file__).resolve().parent
_SRC = _AUTOPCR_DIR.parents[1] / "src"
for _p in (str(_AUTOPCR_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nltk  # noqa: E402

import hpo_extraction.ontology.nltk_data  # noqa: E402,F401  (puts the vendored corpora on nltk.data.path)
import build_dict as upstream  # noqa: E402  (AutoPCR's own, via the sys.path insert above)
from hpo_extraction.ontology.hpo_tree import DEFAULT_HPO_JSON, HPOTree, ontology_fingerprint  # noqa: E402

PHENOTYPIC_ABNORMALITY = "HP:0000118"


def _forms(term: str) -> list[str]:
    """``[surface, lemma]`` for one term, from upstream's tokenise/POS/lemmatise chain, verbatim.

    Lifted from ``build_dict.build_dict``'s ``name:`` / ``synonym:`` branches so the strings landing
    in ``obo.json`` are what upstream would have produced for the same term.
    """
    tokens = nltk.tokenize.word_tokenize(
        term.strip().lower().replace("-", " - ").replace("/", " / ")
    )
    token_pos = nltk.pos_tag(tokens)
    lemmas = [upstream.lemmatizer.lemmatize(t[0], upstream.get_wordnet_pos(t[1])) for t in token_pos]
    return [" ".join(tokens), " ".join(lemmas)]


def subtree_ids(tree: HPOTree, root: str = PHENOTYPIC_ABNORMALITY) -> list[str]:
    """Every descendant of *root*, excluding *root* itself.

    The exclusion is upstream's: ``get_all_child`` seeds ``ALL_CLEAN_NODE`` by recursing into the
    root's children, so the root never enters the vocabulary.
    """
    return sorted(tree.data[root]["Child"])


def build(hpo_json: str, outpath: str, root: str = PHENOTYPIC_ABNORMALITY) -> dict:
    """Write the dictionary files; return upstream's ``hpo_obo`` mapping."""
    outpath = os.path.join(outpath, "")  # upstream concatenates, so it needs the trailing separator
    os.makedirs(outpath, exist_ok=True)

    tree = HPOTree(hpo_json)
    all_nodes = subtree_ids(tree, root)
    node_set = set(all_nodes)

    with open(outpath + "lable.vocab", "w", encoding="utf-8") as f:
        for hpoid in all_nodes:
            f.write(hpoid + "\n")
        f.write("HP:None\n")  # upstream's neg label

    hpo_obo: dict[str, dict] = {}
    abbr: dict[str, str] = {}
    hpo_dict: dict[str, int] = {}

    for hpoid, record in tree.data.items():
        names = record.get("Name") or []
        if not names:
            continue
        first_name = _forms(names[0])
        if hpoid in node_set:
            hpo_dict[first_name[0]] = len(first_name[0].split())
            hpo_dict[first_name[1]] = len(first_name[1].split())

        synonym_list: list[list[str]] = []
        for syn in record.get("Synonym") or []:
            # hpo.json occasionally repeats the label as a synonym; an .obo ``synonym:`` line never
            # does. Dropping it keeps the FAISS index free of a duplicate vector for one string.
            if syn.strip().lower() == names[0].strip().lower():
                continue
            if syn.isupper() and len(syn) < 10:  # upstream's abbreviation rule, verbatim
                if hpoid in node_set:
                    abbr[syn.lower()] = hpoid
                continue
            forms = _forms(syn)
            if forms not in synonym_list:
                synonym_list.append(forms)
            if hpoid in node_set:
                hpo_dict[forms[0]] = len(forms[0].split())
                hpo_dict[forms[1]] = len(forms[1].split())

        defs = record.get("Def") or []
        hpo_obo[hpoid] = {
            "name": first_name,
            "alt_id": list(record.get("Alt_id") or []),
            "def": defs[0] if defs else "",
            "synonym": synonym_list,
            "xref": list(record.get("Xref") or []),
            "is_a": list(record.get("Is_a") or []),
            # This release carries no obsolescence fields. Upstream leaves both '' for any
            # live term and ``alt_hpo`` keys off ``is_obsolete == ''``, so '' is the correct value
            # here, not a placeholder for something missing.
            "is_obsolete": "",
            "replace_id": "",
        }

    with open(outpath + "noabb_lemma.dic", "w", encoding="utf-8") as f:
        for term, _ in sorted(hpo_dict.items(), key=lambda kv: (kv[1], kv[0])):
            f.write(term + "\n")
    with open(outpath + "obo.json", "w", encoding="utf-8") as f:
        json.dump(hpo_obo, f, indent=2)
    with open(outpath + "abbr.json", "w", encoding="utf-8") as f:
        json.dump(abbr, f)

    # word_hpo_map reads ALL_CLEAN_NODE off the module to decide what goes in firstword_id_map.
    upstream.ALL_CLEAN_NODE = all_nodes
    upstream.word_hpo_map(hpo_obo, outpath)
    upstream.hpo_word_map(hpo_obo, outpath)
    upstream.alt_hpo(hpo_obo, outpath)

    return hpo_obo


def cross_check(outpath: str, hpo_json: str, root: str = PHENOTYPIC_ABNORMALITY) -> int:
    """The vocabulary must be the ontology every other method of the comparison is scored against.

    ``hpo_extraction.retrieval.surface_index.build_surface_index`` is the repo's own reading of the phenotypic-abnormality
    subtree: 18 355 ids, because it counts ``HP:0000118`` itself, where upstream's ``get_all_child``
    only ever recurses *into* the root's children. Compared as sets modulo that one id, so the check
    catches a wrong ontology rather than merely a wrong count.

    A disagreement means this dictionary was built from another release, which is the reason this
    script exists, so it aborts instead of warning.
    """
    from hpo_extraction.retrieval.surface_index import build_surface_index

    tree = HPOTree(hpo_json)
    _, hpo2surfaces = build_surface_index(tree)
    expected = set(hpo2surfaces) - {root}
    with open(os.path.join(outpath, "lable.vocab"), encoding="utf-8") as f:
        ids = {ln.strip() for ln in f if ln.strip() and ln.strip() != "HP:None"}
    if ids != expected:
        missing, extra = sorted(expected - ids)[:5], sorted(ids - expected)[:5]
        raise SystemExit(
            f"lable.vocab holds {len(ids)} terms, surface_index sees {len(expected)} in "
            f"{hpo_json} (excluding the root). Missing e.g. {missing}; unexpected e.g. {extra}. "
            "The AutoPCR dictionary is not built from resources/util/hpo.json, refusing to write it."
        )
    print(f"cross-check OK | {len(ids)} terms | ontology fingerprint {ontology_fingerprint()}")
    return len(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AutoPCR's dictionary from resources/util/hpo.json")
    parser.add_argument("--input", "-i", default=str(DEFAULT_HPO_JSON),
                        help="the ontology file (default: resources/util/hpo.json)")
    parser.add_argument("--output", "-o", default=str(_AUTOPCR_DIR / "dict" / "HPO"),
                        help="output dictionary folder")
    parser.add_argument("--rootnode", "-r", default=PHENOTYPIC_ABNORMALITY)
    args = parser.parse_args()

    print(f"building AutoPCR dictionary from {args.input} -> {args.output}")
    build(args.input, args.output, args.rootnode)
    cross_check(args.output, args.input)
    print(f"done. Next: python src/AutoPCR/build_index.py --ontology_dict {args.output}")
