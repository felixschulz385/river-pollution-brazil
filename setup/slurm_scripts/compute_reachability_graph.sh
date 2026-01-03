#!/bin/bash
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --mem=64000mb
#SBATCH --job-name=compute_reachability_graph
#SBATCH --output=./log/compute_reachability_graph/slurm-%j.log

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil

# Run land-cover preprocessing using the CLI
python code/data/cli.py river-network compute-reachability --shapefile-path "data/river_network/shapefile.parquet" --topology-path "data/river_network/topology.parquet" --distance-path "data/river_network/distance_from_estuary.parquet" --output-dir "data/river_network/"