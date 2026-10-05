"""Count the label and synonym strings the dictionary normaliser matches against (thesis section 5.3).

    python experiments/05_phenojury/count_surface_strings.py [--hpo_json resources/util/hpo.json]

The thesis gives 40,335 labels and synonyms over 18,354 terms. No analysis run wrote that number,
so this script recomputes it with the index the dictionary and SapBERT normalisers use
(:func:`hpo_extraction.retrieval.surface_index.build_surface_index`). Strings are counted after the
index's own normalisation (:func:`surface_key`), once each. The root *Phenotypic abnormality*
(HP:0000118) is in the index but is not one of the 18,354 terms the thesis counts, so its two
strings are reported separately.
"""
from __future__ import annotations

import argparse

from hpo_extraction.ontology.hpo_tree import HPOTree
from hpo_extraction.retrieval.surface_index import build_surface_index, surface_key

ROOT = "HP:0000118"


def count(tree: HPOTree) -> dict[str, int]:
    """Terms and distinct strings in the surface index, with and without the root."""
    _surface2hpo, hpo2surfaces = build_surface_index(tree)
    every = {surface_key(s.text) for surfaces in hpo2surfaces.values() for s in surfaces}
    below_root = {surface_key(s.text) for hpo_id, surfaces in hpo2surfaces.items()
                  if hpo_id != ROOT for s in surfaces}
    return {
        "terms_in_index": len(hpo2surfaces),
        "strings_in_index": len(every),
        "terms_below_root": len(hpo2surfaces) - (ROOT in hpo2surfaces),
        "strings_below_root": len(below_root),
    }


def main() -> int:
    """Print the counts. Exit status 0."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hpo_json", default=None, help="ontology file (default: resources/util/hpo.json)")
    args = parser.parse_args()
    tree = HPOTree(args.hpo_json) if args.hpo_json else HPOTree()
    tree.buildHPOTree()
    for key, value in count(tree).items():
        print(f"{key:<20} {value:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
