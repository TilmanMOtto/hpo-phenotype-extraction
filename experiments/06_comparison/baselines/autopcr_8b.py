"""The AutoPCR 8B baseline, AutoPCR applied directly to the patient reports, on HCY and GSC+.

# PhenoBERT. Same cohorts, same ground-truth sets, same emitted artifacts.

See core/autopcr_experiment.py for the driver and third_party/AutoPCR/PATCHES.md for what differs from
upstream.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.baselines.autopcr_experiment import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "baseline_autopcr_8b"


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="baseline_autopcr_8b")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
