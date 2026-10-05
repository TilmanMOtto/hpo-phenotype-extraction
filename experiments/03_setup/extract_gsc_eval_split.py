"""Regenerate ``resources/data/GSC_2024/eval_206_ids.txt`` from AutoPCR's released corpus.

Tao et al. (AutoPCR, Bioinformatics 2026;42(Suppl 1):btag304) evaluate on **206 of the 228**
GSC-2024 abstracts: the corpus is split 22 development / 206 evaluation, following PhenoTagger's
split. Their published table is therefore not comparable with a number computed over all 228, and
it is the only published row in ``comparison``'s literature table that we can be made comparable to
at all -- so the split has to be a vendored, reproducible resource rather than a guess.

Upstream ships it as two PubTator-style ground truth files inside one zip:

    https://github.com/yctao7/AutoPCR  ->  data/corpus.zip
      corpus/GSC-2024/GSC-2024_dev_gold.tsv    22 documents
      corpus/GSC-2024/GSC-2024_test_gold.tsv  206 documents

Each file is blank-line-separated blocks whose first line is the PubMed id. The rest of the block
is the abstract text and its ``start<TAB>end<TAB>span<TAB>HP:#######`` annotations, none of which
we take. **Only the document ids are vendored.** The annotations we score against stay
``resources/data/GSC_2024/Annotations``, which is the corpus's own ground truth -- pulling Tao et al.'s
copy in would silently change the ground truth standard as well as the document set, and then a difference
between our row and theirs would have two causes.

Usage:
    python experiments/03_setup/extract_gsc_eval_split.py                # download, write the defaults
    python experiments/03_setup/extract_gsc_eval_split.py --zip local.zip --out /tmp/check

Running it against an unchanged upstream must leave the vendored file byte-identical.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

CORPUS_URL = "https://github.com/yctao7/AutoPCR/raw/main/data/corpus.zip"
DEV_MEMBER = "corpus/GSC-2024/GSC-2024_dev_gold.tsv"
EVAL_MEMBER = "corpus/GSC-2024/GSC-2024_test_gold.tsv"

#: What the paper states. Both are asserted, not assumed: a silently different split would move
#: every number in the full-GSC+ comparison table without changing its caption.
PAPER_N_DEV = 22
PAPER_N_EVAL = 206


def document_ids(archive: zipfile.ZipFile, member: str) -> list[str]:
    """The PubMed ids of *member*, in file order.

    One document is one blank-line-separated block whose first line is the bare id. Reading only
    the first line of each block is what keeps this independent of the annotation format.
    """
    raw = archive.read(member).decode("utf-8", errors="replace")
    ids = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block.strip():
            continue
        head = block.splitlines()[0].strip()
        if not head.isdigit():
            raise SystemExit(f"{member}: block does not start with a document id: {head!r}")
        ids.append(head)
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{member}: duplicate document ids")
    return ids


def main() -> None:
    """Recover AutoPCR's 206 evaluation and 22 development document ids from its released corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None,
                        help="local copy of AutoPCR's data/corpus.zip; downloaded if omitted")
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "resources" / "data" / "GSC_2024")
    parser.add_argument("--gsc_dir", type=Path,
                        default=_REPO_ROOT / "resources" / "data" / "GSC_2024",
                        help="corpus to cross-check the ids against; '' to skip")
    args = parser.parse_args()

    path = args.zip
    if path is None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "_corpus.zip"
        print(f"downloading {CORPUS_URL}")
        urllib.request.urlretrieve(CORPUS_URL, path)

    with zipfile.ZipFile(path) as archive:
        dev = document_ids(archive, DEV_MEMBER)
        evaluation = document_ids(archive, EVAL_MEMBER)
    if args.zip is None:
        path.unlink()  # The zip is not vendored. Only the id list is

    if len(dev) != PAPER_N_DEV or len(evaluation) != PAPER_N_EVAL:
        raise SystemExit(
            f"upstream split is {len(dev)} dev / {len(evaluation)} eval, but the paper reports "
            f"{PAPER_N_DEV} / {PAPER_N_EVAL}. The published row cannot be read against a frame "
            f"this script no longer reproduces -- reconcile before regenerating.")
    if set(dev) & set(evaluation):
        raise SystemExit("the dev and evaluation sets overlap")

    # Every id must exist in the corpus we score. A silent miss would shrink the cohort while
    # still looking like a complete replication -- the same guard extract_raghpo_gsc_subset uses.
    if str(args.gsc_dir):
        from hpo_extraction.evaluation.datasets.gsc import _gsc_doc_ids

        corpus = set(_gsc_doc_ids(args.gsc_dir))
        missing = sorted(set(dev + evaluation) - corpus)
        if missing:
            raise SystemExit(f"{len(missing)} GSC-2024 document id(s) absent from "
                             f"{args.gsc_dir}: {missing[:10]}")
        extra = sorted(corpus - set(dev + evaluation))
        if extra:
            raise SystemExit(f"{len(extra)} corpus document(s) are in neither split: {extra[:10]}")

    args.out.mkdir(parents=True, exist_ok=True)
    # Sorted the way `hpo_extraction.evaluation.datasets.gsc._gsc_doc_ids` sorts the corpus -- lexicographic on
    # The string, so a set comparison against a loaded cohort needs no re-keying.
    (args.out / "eval_206_ids.txt").write_text(
        "\n".join(sorted(evaluation)) + "\n", encoding="utf-8")
    (args.out / "dev_22_ids.txt").write_text(
        "\n".join(sorted(dev)) + "\n", encoding="utf-8")

    print(f"{len(dev)} development documents (paper: {PAPER_N_DEV})")
    print(f"{len(evaluation)} evaluation documents (paper: {PAPER_N_EVAL})")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
