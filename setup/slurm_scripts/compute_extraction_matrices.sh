#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32000mb
#SBATCH --job-name=compute_extraction_matrices
#SBATCH --mail-type=END
#SBATCH --output=compute_extraction_matrices.log
#SBATCH --error=compute_extraction_matrices-error.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/main/03_02_compute_extraction_matrices.py
