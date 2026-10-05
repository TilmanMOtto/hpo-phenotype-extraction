"""Link PhenoBERT's model weights and embeddings into ``third_party/PhenoBERT/phenobert/``.

    python third_party/phenobert_weights.py                 # list what would be linked
    python third_party/phenobert_weights.py --apply         # link
    python third_party/phenobert_weights.py --source <dir>  # another folder than phenobert.weights_dir

PhenoBERT reads its weights by paths relative to ``phenobert/utils`` (``../models/...``,
``../embeddings/...``). They are about 2.3 GB, published by the PhenoBERT authors, and not tracked.
This script takes them from a folder that holds ``models/`` and ``embeddings/`` in PhenoBERT's
layout (by default ``phenobert.weights_dir`` of the path file) and creates symbolic links to them,
so that one copy on disk serves every checkout. Nothing is copied, moved or deleted. A file that is
already in place is left as it is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "PhenoBERT" / "phenobert"

#: What annotation reads, relative to ``phenobert/``. A folder is linked as a whole.
ITEMS = [
    "models/HPOModel_H",
    "models/bert_model_max_triple.pkl",
    "embeddings/fasttext_pubmed.bin",
    "embeddings/biobert_v1.1_pubmed",
]
#: Files that must exist inside the linked folders.
REQUIRED = (["models/HPOModel_H/model_layer1.pkl"]
            + [f"models/HPOModel_H/model_l1_{i}.pkl" for i in range(25)]
            + [f"embeddings/biobert_v1.1_pubmed/{name}" for name in ("config.json", "pytorch_model.bin", "vocab.txt")])


def missing(root: Path) -> list[str]:
    """The items and required files that do not exist under *root*."""
    return [rel for rel in ITEMS + REQUIRED if not (root / rel).exists()]


def main() -> int:
    """Check the source, then list or create the links. Exit status 1 when something is missing."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=None,
                    help="folder with models/ and embeddings/ (default: phenobert.weights_dir of the path file)")
    ap.add_argument("--apply", action="store_true", help="create the links")
    args = ap.parse_args()
    if args.source is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from hpo_extraction.paths import lookup
        args.source = lookup("phenobert.weights_dir")
    source = Path(args.source).resolve()
    absent = missing(source)
    if absent:
        print(f"missing under {source}:\n  " + "\n  ".join(absent), file=sys.stderr)
        return 1
    for rel in ITEMS:
        link = TARGET / rel
        if link.exists() or link.is_symlink():
            print(f"in place: {link}")
            continue
        print(f"{'link' if args.apply else 'would link'}: {link} -> {source / rel}")
        if args.apply:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source / rel, target_is_directory=(source / rel).is_dir())
    still = missing(TARGET)
    if args.apply:
        print("all weights in place" if not still else f"still missing: {still}")
    return 1 if args.apply and still else 0


if __name__ == "__main__":
    raise SystemExit(main())
