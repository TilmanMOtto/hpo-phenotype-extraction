"""Run AutoPCR's constituency-parse phrase extraction (``ee="neural+"/"neural++"``) on a staged corpus.

Added for this repository, not upstream's. Runs in the parser environment, with upstream's own versions of the parser stack
(``requirements_ee.txt``: benepar 0.2.0, spaCy 3.7.5, transformers 4.49, protobuf 6), which
PhenoRAG_marc_env cannot also hold. It writes the two files ``run_gsc_test_ner`` already reuses when
they exist beside the corpus:

    <corpus_dir>/phrases_benepar.json    constituents (neural+)
    <corpus_dir>/phrases_conjunct.json   coordination-decomposed spans (neural++)

The parse itself is ``HPO_evaluation.build_benepar_phrase_cache``, upstream's loop, lifted verbatim,
so what runs here is what upstream runs. This file only chooses where and verifies the result.

The linker runs (the two AutoPCR baselines, main environment) then load these and never import benepar.
``--copy_to`` hands the SAME parse to every other experiment staged on the same corpus, after
checking the corpus bytes are identical: the 8B-vs-70B comparison must differ in the linker only.

    python third_party/AutoPCR/extract_phrases.py \\
        --corpus output/baseline_autopcr_8b/gsc/autopcr_corpus/corpus_test.tsv \\
        --copy_to output/baseline_autopcr_70b/gsc/autopcr_corpus

Needs, offline: the ``en_core_web_trf`` package installed, and ``models/benepar_en3_large/`` on
``nltk.data.path`` (``NLTK_DATA``). Staged by scripts/stage_autopcr_ee_assets.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

_AUTOPCR_DIR = Path(__file__).resolve().parent
_SRC = _AUTOPCR_DIR.parents[1] / "src"
for _p in (str(_SRC), str(_AUTOPCR_DIR)):   # hpo_extraction (ee/utils.py), then AutoPCR's flat modules
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_FILES = ("phrases_benepar.json", "phrases_conjunct.json")
PROVENANCE_FILE = "phrases_benepar.provenance.json"
_VERSIONED = ("benepar", "spacy", "en_core_web_trf", "torch", "torch-struct", "transformers",
              "protobuf", "nltk")


def read_corpus(corpus_path: str) -> list[str]:
    """The corpus blocks, split the way ``run_gsc_test_ner`` splits them."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        return f.read().strip().split("\n\n")


def corpus_pmids(all_test: list[str]) -> set[str]:
    return {doc.split("\n")[0] for doc in all_test}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_covers(corpus_dir: str, pmids: set[str]) -> bool:
    """Both cache files present, readable, and keyed by the corpus's pmids and no others."""
    for name in CACHE_FILES:
        path = os.path.join(corpus_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                if set(json.load(f)) != pmids:
                    return False
        except (OSError, ValueError):
            return False
    return True


def phrase_counts(corpus_dir: str) -> dict[str, int]:
    counts = {}
    for name in CACHE_FILES:
        with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
            cached = json.load(f)
        counts[name] = sum(len(v["phrases"]) for v in cached.values())
    return counts


def copy_cache(corpus_path: str, targets: list[str]) -> None:
    """Copy the cache to every target corpus dir, refusing any whose corpus is not byte-identical."""
    src_dir = os.path.dirname(corpus_path)
    want = sha256(corpus_path)
    name = os.path.basename(corpus_path)
    for target in targets:
        other = os.path.join(target, name)
        if not os.path.isfile(other):
            raise SystemExit(f"--copy_to {target}: no staged {name} there, stage it first "
                             "(run.py stage_only=true)")
        got = sha256(other)
        if got != want:
            raise SystemExit(
                f"--copy_to {target}: {name} differs from {corpus_path} (sha256 {got[:12]} vs "
                f"{want[:12]}). The two runs would be linking different documents; re-stage both "
                "from the same cohort and max_patients."
            )
        for fname in (*CACHE_FILES, PROVENANCE_FILE):
            if fname == PROVENANCE_FILE and not os.path.isfile(os.path.join(src_dir, fname)):
                continue
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(target, fname))
        print(f"copied parser cache -> {target}")


def _versions() -> dict[str, str | None]:
    out = {}
    for dist in _VERSIONED:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", required=True, help="staged corpus_test.tsv")
    p.add_argument("--copy_to", nargs="*", default=[],
                   help="other autopcr_corpus dirs staged on the identical corpus")
    p.add_argument("--force", action="store_true", help="re-parse even if a covering cache exists")
    args = p.parse_args(argv)

    corpus_path = Path(args.corpus).as_posix()
    corpus_dir = os.path.dirname(corpus_path)
    all_test = read_corpus(corpus_path)
    pmids = corpus_pmids(all_test)
    print(f"corpus: {corpus_path} | {len(pmids)} document(s) | sha256 {sha256(corpus_path)[:12]}")

    if cache_covers(corpus_dir, pmids) and not args.force:
        print("parser cache already covers this corpus, reusing it (pass --force to re-parse)")
    else:
        from HPO_evaluation import build_benepar_phrase_cache  # AutoPCR's own

        t0 = time.time()
        # Same derivation as run_gsc_test_ner: "/".join(testfile.split("/")[:-1]).
        build_benepar_phrase_cache(all_test, "/".join(corpus_path.split("/")[:-1]))
        elapsed = time.time() - t0
        if not cache_covers(corpus_dir, pmids):
            raise SystemExit(f"parse finished but {CACHE_FILES} in {corpus_dir} do not cover the "
                             "corpus pmids, refusing to leave a partial cache")
        try:
            import torch
            device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:  # provenance only; never fail a finished parse over it
            device = None
        provenance = {
            "corpus_sha256": sha256(corpus_path),
            "n_documents": len(pmids),
            "parse_seconds": round(elapsed, 1),
            "device": device,
            "models": {"spacy": "en_core_web_trf", "benepar": "benepar_en3_large"},
            "versions": _versions(),
            "phrase_counts": phrase_counts(corpus_dir),
        }
        with open(os.path.join(corpus_dir, PROVENANCE_FILE), "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)
        print(f"parsed {len(pmids)} document(s) in {elapsed:.0f}s on {device}")

    print("phrase counts:", json.dumps(phrase_counts(corpus_dir)))
    if args.copy_to:
        copy_cache(corpus_path, args.copy_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
