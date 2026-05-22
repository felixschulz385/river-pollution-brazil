#!/bin/bash
#SBATCH --job-name=assemble_land_cover_adm2
#SBATCH --output=./log/assemble_land_cover_adm2/slurm-%j.log
#SBATCH --error=./log/assemble_land_cover_adm2/slurm-%j-error.log
#SBATCH --partition=scicore
#SBATCH --time=0-06:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/assemble_land_cover_adm2

python code/cli.py data land-cover assemble \
  --variant adm2 \
  --land-cover-path data/land_cover/land_cover.feather \
  --river-network-path data/river_network \
  --output data/land_cover/land_cover_assembled_adm2.parquet \
  --n_jobs "${SLURM_CPUS_PER_TASK:-4}"
