r"""Regenerate every chapter-4 figure, table and inline number from the TreePhenoRAG protocol result tables.

    python experiments/figures/make_all_ch4.py

Everything is derived from ``output/treephenorag_protocol/tables/*.csv`` in a single pass, so a
number in a caption cannot disagree with the table printed beside it. Re-run this after any re-run
of the experiment. Never hand-edit the generated ``.tex``.

Blocked deliverables are skipped with a message rather than failing the build. Chapter 4's
specification includes artifacts whose source CSVs do not exist yet -- the retrieval-sufficiency
gate, the pooling sweep, the GSC+ transfer block and most of the inline numbers -- and a missing
CSV is an ordinary state of the world, not a build error. ``docs/thesis_map.md`` lists every one with
the stage that would produce it.
"""
from __future__ import annotations

import importlib
import sys
import traceback

import common as C

FIGURES = [
    "fig_ch4_pipeline",             # The schematic. Reads no CSV
    "fig_ch4_ablation_ladder",      # The V0-V3 schematic. Reads stages.LADDER, no CSV
    "fig_ch4_retrieval",            # The gate sweep -- the one ch4 figure reading the retrieval-curve analysis
    "fig_ch4_score_distributions",  # do the two scores separate what they threshold
    "fig_ch4_reliability",
    "fig_ch4_cost_coverage",
    "fig_ch4_recall_decomposition",  # where the missed annotated terms were lost, one stacked bar
    "fig_ch4_error_analysis",        # what the false positives are, one stacked bar
]

TABLES = [
    "tables_ch4",             # The eight T4.x tables, each skipping itself if its CSV is absent
    "num_ch4_inline",         # The inline-number macros
]


def main() -> int:
    """Build the chapter 4 artifacts."""
    failures: list[str] = []
    print(f"reading {C.TABLES}")
    print(f"figures -> {C.FIG_DIR}")
    print(f"latex   -> {C.TEX_DIR}\n")

    for name in FIGURES + TABLES:
        print(f"{name}:")
        try:
            importlib.import_module(name).main()
        except ModuleNotFoundError:
            print(f"  (no such script: {name}.py)")
        except SystemExit as exc:
            print(f"  skipped: {exc}")
        except Exception:
            traceback.print_exc()
            failures.append(name)

    print(f"\nresolved serif: {C.style.resolved_serif()}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
