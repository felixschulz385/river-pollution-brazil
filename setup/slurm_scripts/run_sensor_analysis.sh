#!/bin/bash
#SBATCH --job-name=sensor_analysis
#SBATCH --output=./log/sensor_analysis/slurm-%j.log
#SBATCH --error=./log/sensor_analysis/slurm-error-%j.log
#SBATCH --partition=scicore
#SBATCH --time=1-00:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch setup/slurm_scripts/run_sensor_analysis.sh [options] [-- extra analysis args]

Options:
  --group-kind <all|type|importance>   Pollutant grouping kind. Default: all
  --group-name <name>                  Pollutant group name. Default: all
  --pollutants <csv>                   Explicit pollutant list, e.g. ph,turbidity
  --land-cover-subclasses <csv>        Restrict land-cover subclasses, e.g. c41,c42
  --available-land-cover-subclasses <csv>
                                       Override the available land-cover subclasses in the input data
  --max-distance-step <n>              Maximum cumulative distance step
  --model-families <csv>               Model families to run. Default: crude_twfe,post_lasso
  --sensor-data-path <path>            Override the sensor panel parquet
  --land-cover-path <path>             Override the land-cover parquet
  --climate-data-path <path>           Override the climate parquet
  --transformations-path <path>        Override the transformation metadata json
  --trenches-path <path>               Override the trenches parquet
  --sensor-id-column <name>            Panel id column. Default: station_code
  --sensor-id-aliases <csv>            Fallback id aliases to normalize
  --datetime-column <name>             Datetime source column
  --date-column <name>                 Date column used for joins/FE
  --climate-join-keys <csv>            Climate join keys. Default: station_code,date
  --climate-column-prefix <prefix>     Climate autodiscovery prefix. Default: cl_
  --climate-count-suffix <suffix>      Climate count suffix. Default: _n
  --climate-interaction-mode <mode>    same_bucket, cumulative, or all
  --distance-buckets <csv>             Ordered distance buckets in the design matrix
  --land-cover-statistic <name>        Land-cover suffix, e.g. cnt or shr
  --cluster-variable <name>            Cluster variable for inference
  --min-observations <n>               Minimum observations per pollutant
  --map-tolerance <float>              MAP convergence tolerance
  --map-max-iterations <n>             MAP maximum iterations
  --lasso-jobs <n>                     Workers for LASSO CV (defaults to allocated CPUs)
  --shard-count <n>                    Total array shards. Default: 1
  --shard-index <n>                    Shard index; defaults to SLURM_ARRAY_TASK_ID
  --resume                             Skip completed checkpoint chunks
  --checkpoint-models <n>              Models per checkpoint chunk. Default: 25
  --output-dir <path>                  Override output directory
  --log-level <level>                  CLI log level. Default: INFO
  --dry-run                            Print the final command and exit
  --help                               Show this message

Examples:
  sbatch setup/slurm_scripts/run_sensor_analysis.sh
  sbatch setup/slurm_scripts/run_sensor_analysis.sh --group-kind importance --group-name high
  sbatch setup/slurm_scripts/run_sensor_analysis.sh --pollutants ph,turbidity --max-distance-step 3
  sbatch setup/slurm_scripts/run_sensor_analysis.sh --group-kind type --group-name nutrients --model-families post_lasso
EOF
}

GROUP_KIND="all"
GROUP_NAME="all"
POLLUTANTS=""
LAND_COVER_SUBCLASSES=""
AVAILABLE_LAND_COVER_SUBCLASSES=""
MAX_DISTANCE_STEP=""
MODEL_FAMILIES="crude_twfe,post_lasso"
SENSOR_DATA_PATH=""
LAND_COVER_PATH=""
CLIMATE_DATA_PATH=""
TRANSFORMATIONS_PATH=""
TRENCHES_PATH=""
SENSOR_ID_COLUMN=""
SENSOR_ID_ALIASES=""
DATETIME_COLUMN=""
DATE_COLUMN=""
CLIMATE_JOIN_KEYS=""
CLIMATE_COLUMN_PREFIX=""
CLIMATE_COUNT_SUFFIX=""
CLIMATE_INTERACTION_MODE="same_bucket"
DISTANCE_BUCKETS=""
LAND_COVER_STATISTIC=""
CLUSTER_VARIABLE=""
MIN_OBSERVATIONS=""
MAP_TOLERANCE=""
MAP_MAX_ITERATIONS=""
LASSO_JOBS=""
SHARD_COUNT="1"
SHARD_INDEX=""
RESUME=0
CHECKPOINT_MODELS="25"
OUTPUT_DIR=""
LOG_LEVEL="INFO"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group-kind)
      GROUP_KIND="$2"
      shift 2
      ;;
    --group-name)
      GROUP_NAME="$2"
      shift 2
      ;;
    --pollutants)
      POLLUTANTS="$2"
      shift 2
      ;;
    --land-cover-subclasses)
      LAND_COVER_SUBCLASSES="$2"
      shift 2
      ;;
    --available-land-cover-subclasses)
      AVAILABLE_LAND_COVER_SUBCLASSES="$2"
      shift 2
      ;;
    --max-distance-step)
      MAX_DISTANCE_STEP="$2"
      shift 2
      ;;
    --model-families)
      MODEL_FAMILIES="$2"
      shift 2
      ;;
    --sensor-data-path)
      SENSOR_DATA_PATH="$2"
      shift 2
      ;;
    --land-cover-path)
      LAND_COVER_PATH="$2"
      shift 2
      ;;
    --climate-data-path)
      CLIMATE_DATA_PATH="$2"
      shift 2
      ;;
    --transformations-path)
      TRANSFORMATIONS_PATH="$2"
      shift 2
      ;;
    --trenches-path)
      TRENCHES_PATH="$2"
      shift 2
      ;;
    --sensor-id-column)
      SENSOR_ID_COLUMN="$2"
      shift 2
      ;;
    --sensor-id-aliases)
      SENSOR_ID_ALIASES="$2"
      shift 2
      ;;
    --datetime-column)
      DATETIME_COLUMN="$2"
      shift 2
      ;;
    --date-column)
      DATE_COLUMN="$2"
      shift 2
      ;;
    --climate-join-keys)
      CLIMATE_JOIN_KEYS="$2"
      shift 2
      ;;
    --climate-column-prefix)
      CLIMATE_COLUMN_PREFIX="$2"
      shift 2
      ;;
    --climate-count-suffix)
      CLIMATE_COUNT_SUFFIX="$2"
      shift 2
      ;;
    --climate-interaction-mode)
      CLIMATE_INTERACTION_MODE="$2"
      shift 2
      ;;
    --distance-buckets)
      DISTANCE_BUCKETS="$2"
      shift 2
      ;;
    --land-cover-statistic)
      LAND_COVER_STATISTIC="$2"
      shift 2
      ;;
    --cluster-variable)
      CLUSTER_VARIABLE="$2"
      shift 2
      ;;
    --min-observations)
      MIN_OBSERVATIONS="$2"
      shift 2
      ;;
    --map-tolerance)
      MAP_TOLERANCE="$2"
      shift 2
      ;;
    --map-max-iterations)
      MAP_MAX_ITERATIONS="$2"
      shift 2
      ;;
    --lasso-jobs)
      LASSO_JOBS="$2"
      shift 2
      ;;
    --shard-count)
      SHARD_COUNT="$2"
      shift 2
      ;;
    --shard-index)
      SHARD_INDEX="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --checkpoint-models)
      CHECKPOINT_MODELS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Error: unknown option $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${GROUP_KIND}" in
  all|type|importance)
    ;;
  *)
    echo "Error: group kind must be one of: all, type, importance" >&2
    exit 1
    ;;
esac

case "${CLIMATE_INTERACTION_MODE}" in
  same_bucket|cumulative|all)
    ;;
  *)
    echo "Error: climate interaction mode must be one of: same_bucket, cumulative, all" >&2
    exit 1
    ;;
esac

COMMAND=(
  python -m src.cli analysis sensor-data
)

COMMAND+=(--log-level "${LOG_LEVEL}")

if [[ -n "${SENSOR_DATA_PATH}" ]]; then
  COMMAND+=(--sensor-data-path "${SENSOR_DATA_PATH}")
fi
if [[ -n "${LAND_COVER_PATH}" ]]; then
  COMMAND+=(--land-cover-path "${LAND_COVER_PATH}")
fi
if [[ -n "${CLIMATE_DATA_PATH}" ]]; then
  COMMAND+=(--climate-data-path "${CLIMATE_DATA_PATH}")
fi
if [[ -n "${TRANSFORMATIONS_PATH}" ]]; then
  COMMAND+=(--transformations-path "${TRANSFORMATIONS_PATH}")
fi
if [[ -n "${TRENCHES_PATH}" ]]; then
  COMMAND+=(--trenches-path "${TRENCHES_PATH}")
fi
if [[ -n "${SENSOR_ID_COLUMN}" ]]; then
  COMMAND+=(--sensor-id-column "${SENSOR_ID_COLUMN}")
fi
if [[ -n "${SENSOR_ID_ALIASES}" ]]; then
  COMMAND+=(--sensor-id-aliases "${SENSOR_ID_ALIASES}")
fi
if [[ -n "${DATETIME_COLUMN}" ]]; then
  COMMAND+=(--datetime-column "${DATETIME_COLUMN}")
fi
if [[ -n "${DATE_COLUMN}" ]]; then
  COMMAND+=(--date-column "${DATE_COLUMN}")
fi
if [[ -n "${CLIMATE_JOIN_KEYS}" ]]; then
  COMMAND+=(--climate-join-keys "${CLIMATE_JOIN_KEYS}")
fi
if [[ -n "${CLIMATE_COLUMN_PREFIX}" ]]; then
  COMMAND+=(--climate-column-prefix "${CLIMATE_COLUMN_PREFIX}")
fi
if [[ -n "${CLIMATE_COUNT_SUFFIX}" ]]; then
  COMMAND+=(--climate-count-suffix "${CLIMATE_COUNT_SUFFIX}")
fi
if [[ -n "${CLIMATE_INTERACTION_MODE}" ]]; then
  COMMAND+=(--climate-interaction-mode "${CLIMATE_INTERACTION_MODE}")
fi
if [[ -n "${DISTANCE_BUCKETS}" ]]; then
  COMMAND+=(--distance-buckets "${DISTANCE_BUCKETS}")
fi
if [[ -n "${LAND_COVER_STATISTIC}" ]]; then
  COMMAND+=(--land-cover-statistic "${LAND_COVER_STATISTIC}")
fi
if [[ -n "${CLUSTER_VARIABLE}" ]]; then
  COMMAND+=(--cluster-variable "${CLUSTER_VARIABLE}")
fi
if [[ -n "${AVAILABLE_LAND_COVER_SUBCLASSES}" ]]; then
  COMMAND+=(--available-land-cover-subclasses "${AVAILABLE_LAND_COVER_SUBCLASSES}")
fi

if [[ -n "${MIN_OBSERVATIONS}" ]]; then
  COMMAND+=(--min-observations "${MIN_OBSERVATIONS}")
fi
if [[ -n "${MAP_TOLERANCE}" ]]; then
  COMMAND+=(--map-tolerance "${MAP_TOLERANCE}")
fi
if [[ -n "${MAP_MAX_ITERATIONS}" ]]; then
  COMMAND+=(--map-max-iterations "${MAP_MAX_ITERATIONS}")
fi
if [[ -n "${LASSO_JOBS}" ]]; then
  COMMAND+=(--lasso-jobs "${LASSO_JOBS}")
fi

COMMAND+=(run --pollutant-group-kind "${GROUP_KIND}" --pollutant-group "${GROUP_NAME}")
if [[ -z "${SHARD_INDEX}" ]]; then
  SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
fi
COMMAND+=(--shard-count "${SHARD_COUNT}" --shard-index "${SHARD_INDEX}" --checkpoint-models "${CHECKPOINT_MODELS}")
if [[ "${RESUME}" -eq 1 ]]; then
  COMMAND+=(--resume)
fi

if [[ -n "${POLLUTANTS}" ]]; then
  COMMAND+=(--pollutants "${POLLUTANTS}")
fi
if [[ -n "${LAND_COVER_SUBCLASSES}" ]]; then
  COMMAND+=(--land-cover-subclasses "${LAND_COVER_SUBCLASSES}")
fi
if [[ -n "${MAX_DISTANCE_STEP}" ]]; then
  COMMAND+=(--max-distance-step "${MAX_DISTANCE_STEP}")
fi
if [[ -n "${MODEL_FAMILIES}" ]]; then
  COMMAND+=(--model-families "${MODEL_FAMILIES}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  COMMAND+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  COMMAND+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate 311

cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil
mkdir -p log/sensor_analysis

"${COMMAND[@]}"
