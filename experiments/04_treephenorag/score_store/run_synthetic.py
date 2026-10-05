"""The synthetic-sentence score store, the instrumented permissive pass: one traversal, cached, re-run for every cell.

# The two settings that entails (S = 10, and the dominating union expansion rule) are required for
# the re-run to be valid, not independent variables. See experiment.md.

Expansion score = max(noisy_or(margins), LRGate.p_expand(meta, margins))
    The maximum over the two rule *families* the chapter reports. `noisy_or` (= P3_1) dominates
    every graded margin pooling, so caching with it makes the node set maximal for a given
    threshold; `LRGate` reads tree geometry that no margin bound constrains, so it has to be unioned
    in explicitly or it cannot be re-run at expansion at all.

Acceptance score = accept_confidence
    Unchanged, and immaterial to the cache: `traverse` expands on the prune score alone, so the
    acceptance rule and threshold never affect which nodes are scored. Both re-run for free.

This run produces no table row. It produces `{variant}_calls.jsonl`, which `treephenorag_protocol`
re-runs into every table row of the chapter.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.treephenorag.pooling import DEFAULT_GATE_PATH, LRGate, accept_confidence, noisy_or

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "treephenorag_scores_synthetic"


def build_scores(cfg: DictConfig):
    """The union expansion rule, so both rule families land inside one cache.

    ``LRGate`` is loaded once here rather than per node: it is a handful of floats read from JSON,
    and re-reading it 4 000 times per report would dominate the CPU side of a GPU-bound loop.
    """
    gate = LRGate(cfg.get("gate_json_path") or DEFAULT_GATE_PATH,
                  calibrated=bool(cfg.get("gate_calibrated", True)))

    def prune_score(margins, meta):
        return max(noisy_or(margins), gate.p_expand(meta, margins))

    return prune_score, accept_confidence, "tree_score_cache"


@hydra.main(version_base=None, config_path="../../../configs/experiments/04_treephenorag", config_name="score_store_synthetic")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    from hpo_extraction.treephenorag.score_store import execute
    execute(cfg, EXP_ID, build_scores)


if __name__ == "__main__":
    main()
