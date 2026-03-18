#!/bin/bash
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00
#SBATCH --mem=64000mb
#SBATCH --job-name=generate_river_network
#SBATCH --output=./log/generate_river_network/slurm-%j.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb

python code/data/cli.py river-network generate --gpkg-path='/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/river_network/raw/bho_2017_v_01_05_5k.gpkg' --gadm-path='/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/gadm/gadm41_BRA.gpkg' --output-dir='/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/river_network/'