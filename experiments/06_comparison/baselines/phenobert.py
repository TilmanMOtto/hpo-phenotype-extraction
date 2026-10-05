"""The PhenoBERT baseline, PhenoBERT applied directly to the patient reports, on HCY and GSC+.


See core/phenobert_experiment.py for the driver.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.baselines.phenobert_baseline import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "baseline_phenobert"


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="baseline_phenobert")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
