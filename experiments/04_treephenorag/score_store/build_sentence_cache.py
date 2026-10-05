"""Prewarm the whole-ontology context embedding cache, run ONCE before the earlier array.

Every earlier tree / flat job builds a contextual database over the full phenotypic-abnormality
subtree (~18k HPO terms), which means encoding ~18k ``.txt`` synthetic-sentence files into the shared
``context_dir/.cache``. Running the 12-job array cold makes all 12 processes encode-and-write the
same cache concurrently, which both wastes GPU time 12× and races on the ``.npy`` writes. This
script does it once, single-process, so the array then reads the cache read-only.

    python experiments/04_treephenorag/score_store/build_sentence_cache.py \
        context_dir=/…/context_HCY_llama_api/llama-3.3-70b-instruct \
        sent_transformer_dir=/…/all-mpnet-base-v2

``--rebuild`` deletes any existing ``.cache`` first (use it once to clear the corrupt/partial
files left by an earlier cold array run).
"""

import logging
import shutil
import sys
from pathlib import Path


from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hpo_extraction.data.loading import LazyContextDict
from hpo_extraction.ontology.hpo_tree import HPO_class, HPOTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("build_context_cache")


def _ontology_context_sentences(hpo_id, tree):
    node = HPO_class(tree.data[hpo_id])
    parts = list(node.name) + list(node.synonym) + list(node.definition)
    return [p for p in parts if p and p.strip()] or [hpo_id]


USAGE = (
    "usage: build_context_cache.py context_dir=<path> sent_transformer_dir=<path> "
    "[restrict_to_subtree=HP:xxxxxxx] [--rebuild]\n"
    "(Hydra-style key=value args.)"
)


def _parse_args(argv):
    """Accept the same key=value form the cluster scripts pass, plus a --rebuild flag."""
    kv, flags = {}, set()
    for a in argv:
        if a in ("--rebuild", "rebuild=true"):
            flags.add("rebuild")
        elif "=" in a:
            k, v = a.split("=", 1)
            kv[k.lstrip("-")] = v
        else:
            flags.add(a)
    missing = [k for k in ("context_dir", "sent_transformer_dir") if k not in kv]
    if missing:
        sys.exit(f"error: missing required arg(s): {', '.join(missing)}\n{USAGE}")
    return kv, flags


def main():
    """Encode every synthetic-sentence file once and store the embeddings beside it (key=value arguments)."""
    kv, flags = _parse_args(sys.argv[1:])

    # Encoding ~18k files is only fast on GPU. On CPU it can take hours. Fail fast (unless the
    # caller opts in with allow_cpu=true) so a node whose CUDA failed to init doesn't silently
    # burn the whole wall clock, just resubmit to land on a healthy GPU.
    try:
        import torch
        if not torch.cuda.is_available() and kv.get("allow_cpu") != "true":
            sys.exit("error: CUDA not available on this node (encoding would take hours on CPU). "
                     "Resubmit to get a GPU node, or pass allow_cpu=true to force it.")
    except ImportError:
        pass

    context_dir = Path(kv["context_dir"])
    cache = context_dir / ".cache"
    if "rebuild" in flags and cache.exists():
        logger.warning("Removing existing cache %s", cache)
        shutil.rmtree(cache)

    tree = HPOTree()
    tree.buildHPOTree()
    restrict = kv.get("restrict_to_subtree")
    if restrict:
        from hpo_extraction.evaluation.retrieval_analysis import descendants_with_hops
        subtree = set(descendants_with_hops(tree, restrict))
    else:
        subtree = set(tree.phenotypic_abnormality) - {tree.root}

    model = SentenceTransformer(kv["sent_transformer_dir"])
    ctx = LazyContextDict(str(context_dir), model)
    universe = sorted(h for h in subtree if h in tree.data)
    logger.info("Warming %d context embeddings into %s", len(universe), cache)

    n_txt = n_fallback = 0
    for h in tqdm(universe, desc="context cache", unit="hpo"):
        if ctx.is_file_backed(h):
            _ = ctx[h]  # loads (and heals) or encodes+persists atomically
            n_txt += 1
        else:
            # No .txt: nothing to cache to disk, but confirm it can be encoded (ontology-derived).
            model.encode(_ontology_context_sentences(h, tree))
            n_fallback += 1
    logger.info("Done | %d file-backed cached, %d ontology-derived (not cached)", n_txt, n_fallback)


if __name__ == "__main__":
    main()
