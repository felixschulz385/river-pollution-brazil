#!/bin/bash
#SBATCH --job-name=assemble_sensor_data
#SBATCH --output=./log/assemble_sensor_data/slurm-%j.log
#SBATCH --error=./log/assemble_sensor_data-slurm/error-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log

PYTHONPATH=code/data python code/data/cli.py water-quality assemble \
  --root-dir . \
  --river-network-path data/river_network \
  --water-quality-path data/sensor_data/water_quality.parquet \
  --streamflow-path data/sensor_data/streamflow.parquet \
  --stations-rivers-path data/sensor_data/stations_rivers.parquet \
  --output data/sensor_data/water_quality_assembled.parquet \
  --n_jobs "${SLURM_CPUS_PER_TASK:-4}"
