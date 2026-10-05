"""The Free Listing generation run, Full SLM ensemble (generative extraction + PhenoBERT) in the earlier harness.


Two stages: `stage=extract model_key=<slm>` (one model, GPU, run as an 8-way job array) and
`stage=aggregate` (CPU-only, re-runs every rule over the cached detections).

See core/slm_ensemble_experiment.py for the driver.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.phenojury.generation import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "phenojury_generation_free_listing"


@hydra.main(version_base=None, config_path="../../../configs/experiments/05_phenojury", config_name="generate_free_listing")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
