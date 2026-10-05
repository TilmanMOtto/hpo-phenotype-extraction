"""The generation run of the other prompts, The prompts an earlier exploratory run promoted, on the full HCY and GSC+ cohorts.


Same driver as an earlier exploratory run, only `prompt_keys`, `max_patients` and `dataset` differ, so a phase-B
number is comparable to its own screen row without any porting step. These are the runs whose
results enter the earlier comparison table.

Two stages here (no `select`: the choice was already made):
  stage=extract  array_index=<0..15>   one (prompt, model) cell, GPU
  stage=aggregate                      both prompts, CPU

See src/hpo_extraction/phenojury/generation_prompts.py for the driver and src/hpo_extraction/phenojury/prompts.py for the prompts.
"""

import hydra
from omegaconf import DictConfig


from hpo_extraction.phenojury.generation_prompts import execute

#: Result folder under output_dir (docs/cluster.md lists the stored ones). Named here because several scripts share
#: this source folder.
EXP_ID = "phenojury_generation_other_prompts"


@hydra.main(version_base=None, config_path="../../../configs/experiments/05_phenojury", config_name="generate_other_prompts")
def main(cfg: DictConfig) -> None:
    """Entry point: run with the Hydra config named in the decorator (see the config header for the command)."""
    execute(cfg, EXP_ID)


if __name__ == "__main__":
    main()
