"""Regenerate ``resources/data/GSC_RAGHPO/`` from RAG-HPO's published analysis workbook.

The RAG-HPO paper (Garcia et al., Genome Medicine 2025;17:91) does **not** evaluate on the full
228-document GSC+ corpus. It evaluates on 114 of those documents, scored against the authors' own
manual re-annotation rather than the corpus annotations. Both live in the workbook they released:

    https://github.com/PoseyPod/RAG-HPO  →  "RAG-HPO Tests and Data Analysis copy.xlsx"
    (= the paper's Additional file 3)

      sheet "GSC Input"               patient_id | ID | clinical_note   → the 114 document ids
      sheet "GSC Manual Annotations "  Patient ID | ID | hpo_description | hpo_term | Category
                                                                       → their ground truth annotations
                                       (the trailing space in the sheet name is theirs)

This script pulls those two sheets out into the two small text files the result-table library reads, so the
vendored copy is reproducible, not a mystery blob. The workbook itself is not vendored:
it is 370 KB of spreadsheet we would never diff, and re-deriving it is one command.

Usage:
    python experiments/03_setup/extract_raghpo_subset.py                 # download, write the defaults
    python experiments/03_setup/extract_raghpo_subset.py --xlsx local.xlsx --out /tmp/check

Running it against an unchanged upstream must leave the vendored files byte-identical, that is
the check that the resource still matches what RAG-HPO published.
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from pathlib import Path

from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT  # noqa: E402

WORKBOOK_URL = (
    "https://github.com/PoseyPod/RAG-HPO/raw/main/"
    "RAG-HPO%20Tests%20and%20Data%20Analysis%20copy.xlsx"
)
ID_SHEET = "GSC Input"
ANNOTATION_SHEET = "GSC Manual Annotations "  # trailing space is upstream's
_HPO_RE = re.compile(r"HP:\d{7}")

# What the paper reports for this subset (Table 1). We reproduce the counts to within two
# annotation rows. The mismatch is recorded in PROVENANCE.md, not silently absorbed.
PAPER_N_DOCS = 114
PAPER_N_PAIRS = 1013
PAPER_N_UNIQUE = 415


def _header_index(header: tuple, name: str) -> int:
    """Column index of *name*, tolerant of the whitespace upstream sprinkles into headers."""
    want = name.strip().lower()
    for i, cell in enumerate(header):
        if cell is not None and str(cell).strip().lower() == want:
            return i
    raise KeyError(f"column {name!r} not found in header {header!r}")


def extract(xlsx_path: Path) -> tuple[list[str], list[tuple[str, str, str]]]:
    """``(document_ids, [(doc_id, hpo_id, description), ...])`` from the workbook.

    Ids are returned sorted the way :func:`hpo_extraction.evaluation.datasets.gsc._gsc_doc_ids` sorts the corpus
    (lexicographic on the string), so the two line up without a second sort anywhere downstream.
    """
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - dependency hint, not logic
        raise SystemExit(
            "openpyxl is required to read the RAG-HPO workbook: pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

    rows = list(wb[ID_SHEET].iter_rows(values_only=True))
    id_col = _header_index(rows[0], "ID")
    ids = sorted({str(r[id_col]).strip() for r in rows[1:] if r[id_col] is not None})

    rows = list(wb[ANNOTATION_SHEET].iter_rows(values_only=True))
    doc_col = _header_index(rows[0], "ID")
    term_col = _header_index(rows[0], "hpo_term")
    desc_col = _header_index(rows[0], "hpo_description")

    seen: set[tuple[str, str]] = set()
    annotations: list[tuple[str, str, str]] = []
    for row in rows[1:]:
        if row[doc_col] is None or row[term_col] is None:
            continue
        doc_id = str(row[doc_col]).strip()
        code = str(row[term_col]).strip()
        if not _HPO_RE.fullmatch(code) or (doc_id, code) in seen:
            continue
        seen.add((doc_id, code))
        desc = "" if row[desc_col] is None else str(row[desc_col]).strip()
        annotations.append((doc_id, code, desc))

    annotations.sort(key=lambda r: (r[0], r[1]))
    return ids, annotations


def main() -> None:
    """Recover the 114 GSC+ documents and their re-annotation from the RAG-HPO authors' workbook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=None,
                        help="local copy of the workbook; downloaded to a temp file if omitted")
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "resources" / "data" / "GSC_RAGHPO")
    parser.add_argument("--gsc_dir", type=Path,
                        default=_REPO_ROOT / "resources" / "data" / "GSC_2024",
                        help="corpus to cross-check the ids against; '' to skip")
    args = parser.parse_args()

    xlsx = args.xlsx
    if xlsx is None:
        xlsx = args.out / "_workbook.xlsx"
        xlsx.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {WORKBOOK_URL}")
        urllib.request.urlretrieve(WORKBOOK_URL, xlsx)

    ids, annotations = extract(xlsx)
    if args.xlsx is None:
        xlsx.unlink()  # The workbook is not vendored

    # Every id must exist in the corpus. A silent miss would shrink the cohort while still
    # looking like a complete replication.
    if str(args.gsc_dir):
        from hpo_extraction.evaluation.datasets.gsc import _gsc_doc_ids

        corpus = set(_gsc_doc_ids(args.gsc_dir))
        missing = [i for i in ids if i not in corpus]
        if missing:
            raise SystemExit(
                f"{len(missing)} RAG-HPO document id(s) are absent from {args.gsc_dir}: "
                f"{missing[:10]}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "document_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    with open(args.out / "annotations.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["doc_id", "hpo_id", "hpo_description"])
        writer.writerows(annotations)

    n_unique = len({code for _, code, _ in annotations})
    print(f"{len(ids)} documents (paper: {PAPER_N_DOCS})")
    print(f"{len(annotations)} doc-term pairs (paper: {PAPER_N_PAIRS})")
    print(f"{n_unique} unique HPO terms (paper: {PAPER_N_UNIQUE})")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
