#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=90000mb
#SBATCH --job-name=extract_drainage_polygons
#SBATCH --mail-type=END
#SBATCH --output=extract_drainage_polygons.log
#SBATCH --error=extract_drainage_polygons-error.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/main/02_02_extract_drainage_polygons.py
