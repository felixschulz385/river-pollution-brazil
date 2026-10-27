#!/bin/bash
#SBATCH --job-name=execute_notebook
#SBATCH --output=./log/notebooks/slurm-%j.log
#SBATCH --partition=scicore
#SBATCH --time=0-06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Execute notebooks in place, in order, so their outputs are committed with them:
#   sbatch setup/slurm_scripts/execute_notebook.sh output/notebooks/health/03_garimpo_gold.ipynb [...]
# Submit from the project root (the log directory is relative to it). Raise
# --time/--mem on the sbatch command line for heavier notebooks.

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch setup/slurm_scripts/execute_notebook.sh <notebook.ipynb> [...]" >&2
  exit 2
fi

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

for notebook in "$@"; do
  echo "Executing ${notebook}"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "${notebook}"
done
