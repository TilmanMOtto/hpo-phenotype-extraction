"""The RAG-HPO reproduction with the published prompt, RAG-HPO on the extraction prompt the *paper* publishes, not the one the repo ships.

#, `system_prompts_paper.json` (byte-faithful to upstream at 25c1ea7) → `system_prompts_figs1.json`
# (byte-faithful to Additional file 2, Fig. S1). Everything else, the Meta LLaMA-3-70B model,
# the paper-era code, the BGE-small/top-20 retrieval over the HPO v2024-08-13 + addons DB, sampled
# decoding at t=0.2, 8-bit, is held fixed.

The published prompt and the published code disagree, and they disagree in the place that
governs extraction volume. System Message II (mapping) is byte-identical between them; System
Message I is not. Fig. S1 adds a one-shot example demonstrating normalisation ("polyps had been
reported" → "nasal polyps"), the qualifier "containing only essential information directly relevant
to HPO determination", and a closing "Do not, under any circumstances, return anything else besides
the JSON object". It also asks for the key "finding" while its own example emits "findings".

We followed the repo. This follows the paper. It is the last place the missing recall can hide
inside our control: the note text was the other candidate, and it was refuted outright, their
`clinical_note` strings are byte-identical to ours in 109 of 114 documents and differ in the other
five only by mojibake (`resources/data/GSC_RAGHPO_NOTES/PROVENANCE.md`).

See core/rag_hpo_paper_experiment.py for the driver, shared with raghpo_reproduction_published_code/raghpo_reproduction_llama3_70b, unchanged.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.baselines.rag_hpo_published_experiment import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "raghpo_reproduction_published_prompt"


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="raghpo_reproduction_published_prompt")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
