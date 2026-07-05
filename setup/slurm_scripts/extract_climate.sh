#!/bin/bash
#SBATCH --job-name=extract_climate
#SBATCH --output=./log/extract_climate/slurm-%j.log
#SBATCH --error=./log/extract_climate/error-%j.log
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/extract_climate

ROOT_DIR="${ROOT_DIR:-.}"

PYTHONPATH=code/data python code/data/cli.py climate preprocess \
  --root-dir "${ROOT_DIR}" \
  --subtype era5_land_daily \
  --stage parquet
