"""The RAG-HPO 8B baseline, RAG-HPO comparison baseline with a local LLaMA-8B, on HCY and GSC+.


See core/rag_hpo_experiment.py for the shared driver.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.baselines.rag_hpo_experiment import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "baseline_raghpo_8b"


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="baseline_raghpo_8b")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
