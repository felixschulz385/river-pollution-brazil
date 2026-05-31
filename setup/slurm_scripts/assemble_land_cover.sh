#!/bin/bash
#SBATCH --job-name=assemble_land_cover
#SBATCH --output=./log/assemble_land_cover/slurm-%j.log
#SBATCH --error=./log/assemble_land_cover/slurm-error-%j.err
#SBATCH --partition=scicore
#SBATCH --time=0-06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/assemble_land_cover

PYTHONPATH=code/data python code/data/cli.py land-cover assemble \
  --variant sensor_upstream_distance_buckets \
  --land-cover-path data/land_cover/land_cover.feather \
  --water-quality-path data/sensor_data/water_quality.parquet \
  --stations-rivers-path data/sensor_data/stations_rivers.parquet \
  --river-network-path data/river_network \
  --output data/land_cover/land_cover_sensor_upstream.parquet \
  --n_jobs "${SLURM_CPUS_PER_TASK:-4}"
