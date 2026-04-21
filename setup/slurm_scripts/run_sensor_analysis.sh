#!/bin/bash
#SBATCH --job-name=sensor_analysis
#SBATCH --output=./log/sensor_analysis/slurm-%j.log
#SBATCH --error=./log/sensor_analysis/slurm-%j.err
#SBATCH --partition=scicore
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch setup/slurm_scripts/run_sensor_analysis.sh <group-kind> <group-name> [extra analysis args...]

Required:
  <group-kind>   Pollutant grouping kind. Must be one of: type, importance
  <group-name>   Pollutant group name within the selected kind, e.g. high or nutrients

Examples:
  sbatch setup/slurm_scripts/run_sensor_analysis.sh importance high
  sbatch setup/slurm_scripts/run_sensor_analysis.sh type nutrients --max-distance-step 3
  sbatch setup/slurm_scripts/run_sensor_analysis.sh importance medium --land-cover-subclasses c41,c42
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

GROUP_KIND="$1"
GROUP_NAME="$2"
shift 2

case "${GROUP_KIND}" in
  type|importance)
    ;;
  *)
    echo "Error: group kind must be one of: type, importance" >&2
    usage >&2
    exit 1
    ;;
esac

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate rpb

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/sensor_analysis

python code/cli.py analysis sensor-data run \
  --pollutant-group-kind "${GROUP_KIND}" \
  --pollutant-group "${GROUP_NAME}" \
  "$@"
