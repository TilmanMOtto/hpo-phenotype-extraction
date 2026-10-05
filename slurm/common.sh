# Sourced by every template in slurm/. Submit from the repository root, with the conda environment
# of environment.yml active: sbatch passes the active environment on to the job.
#
#   mkdir -p logs
#   sbatch slurm/<template>.sbatch
#
# Locations come from configs/cluster_leomed.yaml (or the file HPO_PATHS names). Compute nodes have
# no outbound internet, so every model is read from a local folder.

set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HPO_PATHS="${HPO_PATHS:-$PWD/configs/cluster_leomed.yaml}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# p <key>: one value of the path file, for example $(p hcy.input_dir).
p() { python -m hpo_extraction.path_lookup "$1"; }

RESULTS=$(p results_dir)
OUTPUT=$(p output_dir)
echo "job ${SLURM_JOB_NAME:-local} ${SLURM_JOB_ID:-} on ${SLURMD_NODENAME:-$(hostname)} | results ${RESULTS}"
