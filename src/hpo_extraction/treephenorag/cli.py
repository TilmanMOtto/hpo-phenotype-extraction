"""Command line for TreePhenoRAG.

    hpo-treephenorag --config configs/apps/treephenorag.yaml --input reports/ --output terms.jsonl

``--input`` is one ``.txt`` file or a folder of ``.txt`` files, one report per file. The output is
JSON Lines with one report per line (``docs/data_formats.md``).
"""
from __future__ import annotations

import argparse
import logging
import sys

from hpo_extraction.results import read_reports, write_jsonl


def main(argv: list[str] | None = None) -> int:
    """Run TreePhenoRAG over the reports named on the command line. Returns the exit status."""
    ap = argparse.ArgumentParser(prog="hpo-treephenorag",
                                 description="Extract HPO terms from clinical reports with "
                                             "TreePhenoRAG.")
    ap.add_argument("--config", required=True, help="YAML file (configs/apps/treephenorag.yaml)")
    ap.add_argument("--input", required=True, help="a .txt report or a folder of .txt reports")
    ap.add_argument("--output", required=True, help="JSON Lines file to write")
    ap.add_argument("--encoding", default="utf-8", help="text encoding of the reports")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from hpo_extraction.treephenorag.pipeline import TreePhenoRAG

    reports = read_reports(args.input, encoding=args.encoding)
    method = TreePhenoRAG.from_config(args.config)
    results = method.extract_many(reports)
    path = write_jsonl(results, args.output)
    print(f"{len(results)} report(s) written to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
