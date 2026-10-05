"""Pre-warm the LazyContextDict embedding cache for a context directory.

Iterates over all .txt files in context_dir, encodes each via SentenceTransformer,
and saves the resulting .npy to context_dir/.cache/. Already-cached files are skipped
(cache is valid when the .npy is newer than the source .txt), so the script is safe
to interrupt and re-run.
"""

import argparse
import os
import time

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from hpo_extraction.data.loading import LazyContextDict


def main() -> None:
    """Pre-compute and cache the sentence embeddings of every synthetic-sentence file."""
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache sentence embeddings for all HPO context files."
    )
    parser.add_argument("--context_dir", required=True, help="Directory containing HP_*.txt files")
    parser.add_argument("--sent_transformer_dir", required=True, help="Path to SentenceTransformer model")
    args = parser.parse_args()

    print(f"Loading SentenceTransformer from {args.sent_transformer_dir} ...")
    model = SentenceTransformer(args.sent_transformer_dir)

    print(f"Scanning context directory: {args.context_dir}")
    ctx = LazyContextDict(args.context_dir, model)
    all_keys = sorted(ctx._key_to_stem.keys())
    total = len(all_keys)
    print(f"Found {total} context files.")

    cache_dir = os.path.join(args.context_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    skipped = 0
    processed = 0
    errors = 0
    t0 = time.time()

    for key in tqdm(all_keys, desc="Encoding", unit="HPO"):
        stem = ctx._key_to_stem[key]
        txt_path = os.path.join(args.context_dir, stem + ".txt")
        npy_path = os.path.join(cache_dir, stem + ".npy")

        if os.path.isfile(npy_path) and os.path.getmtime(npy_path) >= os.path.getmtime(txt_path):
            skipped += 1
            continue

        try:
            ctx[key]  # triggers parse → encode → save .npy
            processed += 1
        except Exception as exc:
            tqdm.write(f"ERROR encoding {key}: {exc}")
            errors += 1

    elapsed = time.time() - t0
    print(
        f"\nFinished in {elapsed:.1f}s  |  "
        f"encoded: {processed}  |  skipped (cached): {skipped}  |  errors: {errors}"
    )


if __name__ == "__main__":
    main()
