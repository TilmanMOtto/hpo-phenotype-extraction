"""The AutoPCR 70B baseline, AutoPCR with a 70B entity linker, on HCY and GSC+.

# LLaMA-3.3-70B instead of LLaMA-3.1-8B. Nothing else, same corpus staging, same retrieval
# encoder, same thresholds, same artifacts.

See core/autopcr_experiment.py for the driver and third_party/AutoPCR/PATCHES.md for what differs from
upstream.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.baselines.autopcr_experiment import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "baseline_autopcr_70b"


@hydra.main(version_base=None, config_path="../../../configs/experiments/06_comparison", config_name="baseline_autopcr_70b")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
