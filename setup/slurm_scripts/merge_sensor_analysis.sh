#!/bin/bash
#SBATCH --job-name=sensor_analysis_merge
#SBATCH --output=./log/sensor_analysis/merge-%j.log
#SBATCH --partition=scicore
#SBATCH --time=0-01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <run-dir> <expected-shards>" >&2
  exit 2
fi

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb
cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
python -m src.cli analysis sensor-data merge --run-dir "$1" --expected-shards "$2"
