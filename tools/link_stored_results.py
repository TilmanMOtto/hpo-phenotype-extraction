#!/usr/bin/env python
"""Make stored results readable under the new result directory names.

    python tools/link_stored_results.py --root <results directory> [--apply]

The stored thesis runs on LeoMed live in folders with their original names (for example
``exp13_22_tree_protocol``). The code now reads and writes the new names
(``treephenorag_protocol``). This tool walks the results directory and, for every directory with
an old name, creates a symbolic link with the new name beside it. Some runs nest an old name inside
another (``exp13_25_inputs/exp14_04_prompt_jury_protocol/``), so the walk goes four levels deep.

Nothing is moved, renamed or deleted. Without ``--apply`` the tool only lists what it would do.
The same mapping is used for the dated ground-truth directories
(``curated_gold_<date>`` -> ``curated_ground_truth_<date>``); pass the HCY data directory as
``--root`` for those.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

#: Stored result folders of the thesis runs (old name) and the name this repository uses for them.
RESULT_DIRS = {
    "exp13_21_tree_score_cache": "treephenorag_scores_synthetic",
    "exp13_24_tree_score_cache_r3u": "treephenorag_scores_terminfo",
    "exp13_22_tree_protocol": "treephenorag_protocol",
    "exp00_07_query_set_variants": "treephenorag_retrieval_curves",
    "exp13_06_slm_ensemble": "phenojury_generation_free_listing",
    "exp14_01_prompt_full": "phenojury_generation_other_prompts",
    "exp14_03_normaliser_grid": "phenojury_normalisation",
    "exp14_04_prompt_jury_protocol": "phenojury_protocol",
    "exp13_08_phenobert": "baseline_phenobert",
    "exp13_03_raghpo_llama8b": "baseline_raghpo_8b",
    "exp13_04_raghpo_llama70b": "baseline_raghpo_70b",
    "exp13_19_autopcr_llama8b": "baseline_autopcr_8b",
    "exp13_20_autopcr_llama70b": "baseline_autopcr_70b",
    "exp13_11_raghpo_paper70b": "raghpo_reproduction_published_code",
    "exp13_15_raghpo_paper70b_meta": "raghpo_reproduction_llama3_70b",
    "exp13_17_raghpo_figs1_prompt": "raghpo_reproduction_published_prompt",
    "exp13_25_final_comparison": "comparison",
    "exp13_25_inputs": "comparison_inputs",
    "exp13_18_curated_gold_eval": "hcy_ground_truth",
}
#: Generic folder names, renamed only where they appear as a folder of their own.
SEGMENT_ONLY = {
    "index_statistics": "synthetic_sentence_statistics",
    "compare_bundles": "compare_ui_bundles",
}


_DATED = re.compile(r"^curated_gold_(\d{4}-\d{2}-\d{2})$")


def new_name(name: str) -> str | None:
    if name in RESULT_DIRS:
        return RESULT_DIRS[name]
    if name in SEGMENT_ONLY:
        return SEGMENT_ONLY[name]
    m = _DATED.match(name)
    return f"curated_ground_truth_{m.group(1)}" if m else None


def plan(root: Path, depth: int = 4) -> list[tuple[Path, Path]]:
    links = []
    for current, dirs, _ in os.walk(root, followlinks=False):
        level = len(Path(current).relative_to(root).parts)
        if level >= depth:
            dirs[:] = []
            continue
        for d in sorted(dirs):
            target = new_name(d)
            if target:
                link = Path(current) / target
                if not link.exists() and not link.is_symlink():
                    links.append((link, Path(d)))
    return links


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="results directory (or the HCY data directory)")
    ap.add_argument("--apply", action="store_true", help="create the links")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    links = plan(root)
    for link, target in links:
        print(f"{'link' if args.apply else 'would link'} {link.relative_to(root)} -> {target}")
        if args.apply:
            link.symlink_to(target, target_is_directory=True)
    print(f"{len(links)} link(s){'' if args.apply else ' (dry run)'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
