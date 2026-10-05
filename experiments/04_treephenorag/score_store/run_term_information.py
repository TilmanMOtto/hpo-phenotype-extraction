"""The term-information score store, the synthetic-sentence score store's cache run, retrieved by the retrieval-curve analysis's R3 instead of the synthetic-sentence corpus.


Why this run exists. The retrieval-curve analysis compared four query representations on the retrieval stage alone and
found that R3 beats R1 on the annotated terms while R1u's advantage over R1 is entirely an ancestor
effect. That was measured on recall curves, with no SLM downstream. TreePhenoRAG is where the
question actually bites: the traversal's retrieval is ``R1u`` (``UnionScorer``'s column-max
over the descendant closure), and every annotated term it can ever reach has to survive it first. This
run is the same traversal with ``R3u`` in that slot, the only end-to-end measurement of what the
synthetic-sentence corpus is worth once a verifier reads the sentences it retrieves.

Expansion score = max(noisy_or(margins), LRGate.p_expand(meta, margins))
    Unchanged from the synthetic-sentence score store, and it has to be: the threshold acts on SLM margins, not on retrieval
    similarity, so the containment argument that makes this offline evaluationable is untouched by the
    index swap.

Acceptance score = accept_confidence
    Unchanged, and immaterial to the cache.

Like the synthetic-sentence score store this run produces no table row. It produces ``{variant}_calls.jsonl``, which
``treephenorag_protocol`` re-runs, pointed at this cache instead, the whole chapter re-derives
on R3u retrieval at no further GPU cost.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.treephenorag.pooling import DEFAULT_GATE_PATH, LRGate, accept_confidence, noisy_or

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "treephenorag_scores_terminfo"


def build_scores(cfg: DictConfig):
    """The synthetic-sentence score store's union expansion rule, verbatim, the scoring side is not the variable here."""
    gate = LRGate(cfg.get("gate_json_path") or DEFAULT_GATE_PATH,
                  calibrated=bool(cfg.get("gate_calibrated", True)))

    def prune_score(margins, meta):
        return max(noisy_or(margins), gate.p_expand(meta, margins))

    return prune_score, accept_confidence, "tree_score_cache_r3u"


@hydra.main(version_base=None, config_path="../../../configs/experiments/04_treephenorag", config_name="score_store_term_information")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    from hpo_extraction.treephenorag.score_store import execute
    execute(cfg, EXP_ID, build_scores)


if __name__ == "__main__":
    main()
