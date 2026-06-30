#!/bin/bash
#SBATCH --job-name=extract_climate
#SBATCH --output=./log/extract_climate/slurm-%j.log
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/extract_climate

ROOT_DIR="${ROOT_DIR:-.}"

PYTHONPATH=code/data conda run -n 311 python code/data/cli.py climate preprocess \
  --root-dir "${ROOT_DIR}" \
  --subtype era5_land_daily \
  --stage parquet
