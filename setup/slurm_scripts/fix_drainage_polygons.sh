#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32000mb
#SBATCH --job-name=fix_drainage_polygons
#SBATCH --mail-type=END
#SBATCH --output=fix_drainage_polygons.log
#SBATCH --error=fix_drainage_polygons-error.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

# ORPHANED: this script no longer exists in the repo -- it predates the
# current GeoPackage-based RiverNetwork.generate() pipeline (see
# generate_river_network.sh), which now handles drainage-area loading
# directly. Left as-is pending a decision on whether to retire this script.
python /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/main/02_00_fix_drainage_polygons.py
