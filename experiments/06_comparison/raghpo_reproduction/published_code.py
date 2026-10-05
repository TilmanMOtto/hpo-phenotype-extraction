"""The RAG-HPO reproduction with the published code, RAG-HPO as published (Garcia et al. 2025) with a local LLaMA-3-70B, HCY and GSC+.

# DB, sampled decoding) instead of upstream's current one, so the published table has a
# like-for-like row.

See core/rag_hpo_paper_experiment.py for the driver.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.baselines.rag_hpo_published_experiment import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "raghpo_reproduction_published_code"


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="raghpo_reproduction_published_code")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
