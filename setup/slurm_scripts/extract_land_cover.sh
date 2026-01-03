#!/bin/bash
#SBATCH --partition=scicore
#SBATCH --qos=1day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=64000mb
#SBATCH --job-name=extract_land_cover
#SBATCH --output=./log/extract_land_cover/slurm-%j.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb

# Run land-cover preprocessing using the CLI
python /scicore/home/meiera/schulz0022/projects/river-pollution-brazil/code/data/cli.py land-cover preprocess --n_jobs 4 --output land_cover_results.feather