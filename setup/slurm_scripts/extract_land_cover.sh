#!/bin/bash
#SBATCH --job-name=extract_land_cover
#SBATCH --output=./log/extract_land_cover/slurm-%j.log
#SBATCH --error=./log/extract_land_cover/slurm-error-%j.log
#SBATCH --partition=scicore
#SBATCH --time=0-12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil

# Run land-cover preprocessing using the unified CLI
python code/cli.py data land-cover preprocess --n_jobs 4 --river-network-path='data/river_network' --output='data/land_cover/land_cover.feather'
