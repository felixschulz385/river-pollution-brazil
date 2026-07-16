#!/bin/bash
# Submit resumable sensor-analysis shards and a dependent merge job.
set -euo pipefail

SHARDS=8
MAX_CONCURRENT=4
RUN_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards) SHARDS="$2"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "--run-dir is required (the canonical output subdirectory, e.g. output/.../all_all)." >&2
  exit 2
fi

ARRAY_JOB=$(sbatch --parsable --array="0-$((SHARDS - 1))%${MAX_CONCURRENT}" \
  setup/slurm_scripts/run_sensor_analysis.sh --shard-count "${SHARDS}" --resume "${EXTRA[@]}")

sbatch --dependency="afterany:${ARRAY_JOB}" \
  setup/slurm_scripts/merge_sensor_analysis.sh "${RUN_DIR}" "${SHARDS}"

printf 'Submitted array job %s and dependent merge job for %s\n' "${ARRAY_JOB}" "${RUN_DIR}"
