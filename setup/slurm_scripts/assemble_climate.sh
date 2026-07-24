#!/bin/bash
#SBATCH --job-name=assemble_climate
#SBATCH --output=./log/assemble_climate/slurm-%j.log
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/assemble_climate_era5_land

ROOT_DIR="${ROOT_DIR:-.}"
SENSOR_OUTPUT_PATH="${SENSOR_OUTPUT_PATH:-data/climate/processed/climate_assembled_sensor.parquet}"
ADM2_OUTPUT_PATH="${ADM2_OUTPUT_PATH:-data/climate/processed/climate_assembled_adm2.parquet}"

PYTHONPATH=code/data python code/data/cli.py climate assemble \
  --root-dir "${ROOT_DIR}" \
  --variant sensor_upstream_distance_buckets \
  --output "${SENSOR_OUTPUT_PATH}" \
  --n_jobs "${SLURM_CPUS_PER_TASK:-4}"

PYTHONPATH=code/data python code/data/cli.py climate assemble \
  --root-dir "${ROOT_DIR}" \
  --variant adm2_upstream_yearly \
  --output "${ADM2_OUTPUT_PATH}" \
  --n_jobs "${SLURM_CPUS_PER_TASK:-4}"
