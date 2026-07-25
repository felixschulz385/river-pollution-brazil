#!/bin/bash
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --mem=64000mb
#SBATCH --job-name=compute_reachability_graph
#SBATCH --output=./log/compute_reachability_graph/slurm-%j.log
#SBATCH --error=./log/compute_reachability_graph/slurm-error-%j.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil

# NON-FUNCTIONAL against the current CLI/RiverNetwork: `compute-reachability` is
# not a registered river-network action (only `generate` is), and RiverNetwork
# has no method taking shapefile-path/topology-path/distance-path parquet
# inputs -- this predates the current GeoPackage-based RiverNetwork.generate()
# pipeline and was never updated. Left as-is pending a decision on whether this
# workflow should be reimplemented against the current API or retired.
python src/data/cli.py river-network compute-reachability --shapefile-path "data/river_network/shapefile.parquet" --topology-path "data/river_network/topology.parquet" --distance-path "data/river_network/distance_from_estuary.parquet" --output-dir "data/river_network/"
