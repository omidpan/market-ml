set -euo pipefail

# market_ml configurable pipeline runner
#
# Default: use enabled/disabled stages from config/pipeline.yaml
#   ./scripts/market_pipeline_run.sh AMAT
#
# Run exactly one stage (useful for ConcourseCI):
#   ./scripts/market_pipeline_run.sh AMAT --stage validate
#   ./scripts/market_pipeline_run.sh AMAT --stage features
#   ./scripts/market_pipeline_run.sh NVDA --stage sequences
#
# Run a contiguous stage range:
#   ./scripts/market_pipeline_run.sh AMAT --from-stage regularize
#   ./scripts/market_pipeline_run.sh AMAT --from-stage resample --to-stage features
#
# Preview without executing:
#   ./scripts/market_pipeline_run.sh AMAT --dry-run
#
# Show stage selection:
#   ./scripts/market_pipeline_run.sh AMAT --list-stages

STAGES=(
  validate
  canonicalize
  observed_parquet
  regularize
  regularized_parquet
  resample
  features
  targets
  sequences
  model_matrix
  train_model
)

usage() {
  cat <<'EOF'
Usage:
  market_pipeline_run.sh SYMBOL [options]

Options:
  --stage NAME          Run exactly one stage, regardless of YAML enabled flag.
  --from-stage NAME     Run from NAME forward, regardless of YAML enabled flags.
  --to-stage NAME       Run through NAME. With no --from-stage, starts at validate.
  --dry-run             Print which stages would run without executing them.
  --list-stages         Print the resolved stage plan and exit.
  -h, --help            Show this help.

Stage names:
  validate
  canonicalize
  observed_parquet
  regularize
  regularized_parquet
  resample
  features
  targets
  sequences
  model_matrix
  train_model

Selection rules:
  1. No selector: pipeline.yaml pipeline.stages.*.enabled controls execution.
  2. --stage: runs exactly that stage.
  3. --from-stage/--to-stage: runs the selected contiguous range.
     Explicit CLI selectors override YAML enabled flags.

Important:
  Running a downstream stage alone assumes its upstream artifacts already exist.
EOF
}

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 2
fi

SYMBOL_INPUT="$1"
shift

SELECT_STAGE=""
FROM_STAGE=""
TO_STAGE=""
DRY_RUN=false
LIST_STAGES=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage)
      [ "$#" -ge 2 ] || {
        echo "ERROR: --stage requires a stage name." >&2
        exit 2
      }
      SELECT_STAGE="$2"
      shift 2
      ;;
    --from-stage)
      [ "$#" -ge 2 ] || {
        echo "ERROR: --from-stage requires a stage name." >&2
        exit 2
      }
      FROM_STAGE="$2"
      shift 2
      ;;
    --to-stage)
      [ "$#" -ge 2 ] || {
        echo "ERROR: --to-stage requires a stage name." >&2
        exit 2
      }
      TO_STAGE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --list-stages)
      LIST_STAGES=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "${SELECT_STAGE}" ] && {
  [ -n "${FROM_STAGE}" ] || [ -n "${TO_STAGE}" ];
}; then
  echo "ERROR: --stage cannot be combined with --from-stage/--to-stage." >&2
  exit 2
fi

SYMBOL_UPPER="$(printf '%s' "${SYMBOL_INPUT}" | tr '[:lower:]' '[:upper:]')"
SYMBOL_LOWER="$(printf '%s' "${SYMBOL_INPUT}" | tr '[:upper:]' '[:lower:]')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: .env file not found: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${DATA_PATH:?ERROR: DATA_PATH is not defined in ${ENV_FILE}}"
: "${CONFIG_PATH:?ERROR: CONFIG_PATH is not defined in ${ENV_FILE}}"

DATA_PATH="${DATA_PATH%/}"
CONFIG_PATH="${CONFIG_PATH%/}"
REPORTS_PATH="${REPORTS_PATH:-${DATA_PATH}/reports}"
REPORTS_PATH="${REPORTS_PATH%/}"

PYTHON="${PYTHON:-python}"

PIPELINE_CONFIG="${CONFIG_PATH}/pipeline.yaml"
INSTRUMENTS_CONFIG="${CONFIG_PATH}/instruments.csv"
SOURCES_CONFIG="${CONFIG_PATH}/sources.csv"

if [ ! -f "${PIPELINE_CONFIG}" ]; then
  echo "ERROR: pipeline config not found: ${PIPELINE_CONFIG}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Small YAML reader
# ---------------------------------------------------------------------------
#
# PyYAML is already a project dependency because common.py loads pipeline.yaml.
# Keeping the YAML parsing in Python avoids brittle grep/sed parsing.

yaml_value() {
  local dotted_path="$1"
  local default_value="${2-}"

  "${PYTHON}" - "${PIPELINE_CONFIG}" "${dotted_path}" "${default_value}" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
dotted = sys.argv[2]
default = sys.argv[3]

raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

value = raw
for part in dotted.split("."):
    if not isinstance(value, dict) or part not in value:
        value = default
        break
    value = value[part]

if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, list):
    print(" ".join(str(item) for item in value))
elif value is None:
    print("")
else:
    print(value)
PY
}

SOURCE_ID="$(yaml_value "pipeline.source_id" "primary_extended")"

RESAMPLE_TIMEFRAMES="$(yaml_value "pipeline.resample.timeframes" "5m 15m 30m 1h")"
FEATURE_TIMEFRAMES="$(yaml_value "pipeline.features.timeframes" "5m 15m 30m 1h")"
TARGET_HORIZONS="$(yaml_value "pipeline.targets.horizons_minutes" "5 15 30")"
TARGET_THRESHOLD_BPS="$(yaml_value "pipeline.targets.direction_threshold_bps" "0.0")"
TARGET_REQUIRE_SAME_SESSION="$(yaml_value "pipeline.targets.require_same_session" "false")"

CONFIG_ENABLED_STAGES=""
for stage in "${STAGES[@]}"; do
  enabled="$(yaml_value "pipeline.stages.${stage}.enabled" "false")"
  if [ "${enabled}" = "true" ]; then
    CONFIG_ENABLED_STAGES="${CONFIG_ENABLED_STAGES} ${stage}"
  elif [ "${enabled}" != "false" ]; then
    echo "ERROR: pipeline.stages.${stage}.enabled must be true or false." >&2
    exit 1
  fi
done

stage_index() {
  local wanted="$1"
  local i=0
  local stage

  for stage in "${STAGES[@]}"; do
    if [ "${stage}" = "${wanted}" ]; then
      printf '%s\n' "${i}"
      return 0
    fi
    i=$((i + 1))
  done

  return 1
}

validate_stage_name() {
  local stage="$1"
  if [ -z "${stage}" ]; then
    return 0
  fi

  if ! stage_index "${stage}" >/dev/null; then
    echo "ERROR: unknown pipeline stage: ${stage}" >&2
    echo "Valid stages: ${STAGES[*]}" >&2
    exit 2
  fi
}

validate_stage_name "${SELECT_STAGE}"
validate_stage_name "${FROM_STAGE}"
validate_stage_name "${TO_STAGE}"

FROM_INDEX=0
TO_INDEX=$((${#STAGES[@]} - 1))

if [ -n "${FROM_STAGE}" ]; then
  FROM_INDEX="$(stage_index "${FROM_STAGE}")"
fi

if [ -n "${TO_STAGE}" ]; then
  TO_INDEX="$(stage_index "${TO_STAGE}")"
fi

if [ "${FROM_INDEX}" -gt "${TO_INDEX}" ]; then
  echo "ERROR: --from-stage comes after --to-stage." >&2
  exit 2
fi

should_run_stage() {
  local stage="$1"
  local idx

  if [ -n "${SELECT_STAGE}" ]; then
    [ "${stage}" = "${SELECT_STAGE}" ]
    return
  fi

  if [ -n "${FROM_STAGE}" ] || [ -n "${TO_STAGE}" ]; then
    idx="$(stage_index "${stage}")"
    [ "${idx}" -ge "${FROM_INDEX}" ] && [ "${idx}" -le "${TO_INDEX}" ]
    return
  fi

  case " ${CONFIG_ENABLED_STAGES} " in
    *" ${stage} "*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

RAW_CSV="${DATA_PATH}/raw/bootstrap/${SYMBOL_LOWER}/${SYMBOL_LOWER}_1min_extended.csv"

OBSERVED_CSV="${DATA_PATH}/canonical/observed/${SYMBOL_LOWER}/${SYMBOL_LOWER}_1min_extended.csv"
REGULARIZED_CSV="${DATA_PATH}/canonical/regularized/${SYMBOL_LOWER}/${SYMBOL_LOWER}_1min_extended.csv"

OBSERVED_PARQUET_ROOT="${DATA_PATH}/parquet/ohlcv_1m_observed"
REGULARIZED_PARQUET_ROOT="${DATA_PATH}/parquet/ohlcv_1m_regularized"
RESAMPLED_PARQUET_ROOT="${DATA_PATH}/parquet/resampled"
FEATURES_PARQUET_ROOT="${DATA_PATH}/parquet/features_1m"
TARGETS_PARQUET_ROOT="${DATA_PATH}/parquet/targets_1m"
SEQUENCE_INDEX_ROOT="${DATA_PATH}/parquet/sequence_index"
MODEL_MATRIX_ROOT="${DATA_PATH}/parquet/model_matrix"
MODEL_OUTPUT_ROOT="${DATA_PATH}/models"

VALIDATION_REPORTS="${REPORTS_PATH}/validation"
CANONICALIZATION_REPORTS="${REPORTS_PATH}/canonicalization"
REGULARIZATION_REPORTS="${REPORTS_PATH}/regularization"
FEATURE_REPORTS="${REPORTS_PATH}/features"
TARGET_REPORTS="${REPORTS_PATH}/targets"
SEQUENCE_REPORTS="${REPORTS_PATH}/sequences"
MODEL_MATRIX_REPORTS="${REPORTS_PATH}/model_matrix"
TRAINING_REPORTS="${REPORTS_PATH}/training"

print_plan() {
  echo
  echo "Resolved pipeline plan for ${SYMBOL_UPPER}"
  echo "------------------------------------------------------------"

  local stage
  local state

  for stage in "${STAGES[@]}"; do
    if should_run_stage "${stage}"; then
      state="RUN"
    else
      state="SKIP"
    fi
    printf '  %-22s %s\n' "${stage}" "${state}"
  done

  echo "------------------------------------------------------------"
  echo "Selector mode: $(
    if [ -n "${SELECT_STAGE}" ]; then
      printf 'single stage: %s' "${SELECT_STAGE}"
    elif [ -n "${FROM_STAGE}" ] || [ -n "${TO_STAGE}" ]; then
      printf 'range: %s -> %s' \
        "${FROM_STAGE:-validate}" \
        "${TO_STAGE:-train_model}"
    else
      printf 'pipeline.yaml enabled flags'
    fi
  )"
  echo
}

print_plan

if [ "${LIST_STAGES}" = true ]; then
  exit 0
fi

if [ "${DRY_RUN}" = true ]; then
  echo "Dry run only; no stages executed."
  exit 0
fi

echo "============================================================"
echo "market_ml pipeline: ${SYMBOL_UPPER}"
echo "Python: $(${PYTHON} -c 'import sys; print(sys.executable)')"
echo "DATA_PATH: ${DATA_PATH}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "REPORTS_PATH: ${REPORTS_PATH}"
echo "SOURCE_ID: ${SOURCE_ID}"
echo "============================================================"
echo

run_stage() {
  local stage="$1"
  local ordinal="$2"
  local title="$3"
  shift 3

  if ! should_run_stage "${stage}"; then
    echo "[${ordinal}/${#STAGES[@]}] SKIP ${stage} - ${title}"
    return 0
  fi

  echo "[${ordinal}/${#STAGES[@]}] RUN  ${stage} - ${title}"
  "$@"
  echo
}

run_stage \
  validate 1 \
  "Validate raw bootstrap CSV" \
  "${PYTHON}" "${PROJECT_ROOT}/src/validate.py" \
    --input "${RAW_CSV}" \
    --symbol "${SYMBOL_UPPER}" \
    --source-id "${SOURCE_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --instruments "${INSTRUMENTS_CONFIG}" \
    --sources "${SOURCES_CONFIG}" \
    --reports "${VALIDATION_REPORTS}"

run_stage \
  canonicalize 2 \
  "Canonicalize observed source rows" \
  "${PYTHON}" "${PROJECT_ROOT}/src/canonicalize.py" \
    --input "${RAW_CSV}" \
    --output "${OBSERVED_CSV}" \
    --symbol "${SYMBOL_LOWER}" \
    --source-id "${SOURCE_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --sources "${SOURCES_CONFIG}" \
    --reports "${CANONICALIZATION_REPORTS}"

run_stage \
  observed_parquet 3 \
  "Write observed monthly Parquet" \
  "${PYTHON}" "${PROJECT_ROOT}/src/parquet_store.py" \
    --input "${OBSERVED_CSV}" \
    --root "${OBSERVED_PARQUET_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --config "${PIPELINE_CONFIG}"

run_stage \
  regularize 4 \
  "Build regularized 1-minute grid" \
  "${PYTHON}" "${PROJECT_ROOT}/src/regularize.py" \
    --input "${OBSERVED_CSV}" \
    --output "${REGULARIZED_CSV}" \
    --symbol "${SYMBOL_LOWER}" \
    --source-id "${SOURCE_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --instruments "${INSTRUMENTS_CONFIG}" \
    --sources "${SOURCES_CONFIG}" \
    --reports "${REGULARIZATION_REPORTS}"

run_stage \
  regularized_parquet 5 \
  "Write regularized monthly Parquet" \
  "${PYTHON}" "${PROJECT_ROOT}/src/parquet_store.py" \
    --input "${REGULARIZED_CSV}" \
    --root "${REGULARIZED_PARQUET_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --config "${PIPELINE_CONFIG}"

# shellcheck disable=SC2086
run_stage \
  resample 6 \
  "Resample higher timeframes" \
  "${PYTHON}" "${PROJECT_ROOT}/src/resample.py" \
    --input-root "${REGULARIZED_PARQUET_ROOT}" \
    --output-root "${RESAMPLED_PARQUET_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --timeframes ${RESAMPLE_TIMEFRAMES} \
    --config "${PIPELINE_CONFIG}"

# shellcheck disable=SC2086
run_stage \
  features 7 \
  "Build causal feature store" \
  "${PYTHON}" "${PROJECT_ROOT}/src/features.py" \
    --regularized-root "${REGULARIZED_PARQUET_ROOT}" \
    --resampled-root "${RESAMPLED_PARQUET_ROOT}" \
    --output-root "${FEATURES_PARQUET_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --timeframes ${FEATURE_TIMEFRAMES} \
    --config "${PIPELINE_CONFIG}" \
    --reports "${FEATURE_REPORTS}"

# Build the target command as a non-empty array.
#
# This avoids a portability bug with `set -u` on older Bash versions
# (notably the Bash 3.2 shipped with macOS), where expanding an empty
# array such as "${TARGET_SESSION_ARGS[@]}" can raise "unbound variable".
TARGET_CMD=(
  "${PYTHON}" "${PROJECT_ROOT}/src/targets.py"
  --features-root "${FEATURES_PARQUET_ROOT}"
  --output-root "${TARGETS_PARQUET_ROOT}"
  --symbol "${SYMBOL_LOWER}"
)

# TARGET_HORIZONS is read from YAML as a whitespace-separated list.
# shellcheck disable=SC2206
TARGET_HORIZON_ARGS=( ${TARGET_HORIZONS} )
TARGET_CMD+=(--horizons "${TARGET_HORIZON_ARGS[@]}")
TARGET_CMD+=(--direction-threshold-bps "${TARGET_THRESHOLD_BPS}")

if [ "${TARGET_REQUIRE_SAME_SESSION}" = "true" ]; then
  TARGET_CMD+=(--require-same-session)
elif [ "${TARGET_REQUIRE_SAME_SESSION}" != "false" ]; then
  echo "ERROR: pipeline.targets.require_same_session must be true or false." >&2
  exit 1
fi

TARGET_CMD+=(
  --config "${PIPELINE_CONFIG}"
  --reports "${TARGET_REPORTS}"
)

run_stage \
  targets 8 \
  "Build exact-horizon target store" \
  "${TARGET_CMD[@]}"

# sequences.py v1.1 resolves sequence settings with this precedence:
#   explicit Python CLI override > pipeline.yaml > sequences.py defaults.
# The pipeline runner intentionally passes no sequence-tuning flags, so normal
# pipeline execution uses pipeline.sequences.* from pipeline.yaml.
run_stage \
  sequences 9 \
  "Build leakage-safe sequence index" \
  "${PYTHON}" "${PROJECT_ROOT}/src/sequences.py" \
    --features-root "${FEATURES_PARQUET_ROOT}" \
    --targets-root "${TARGETS_PARQUET_ROOT}" \
    --output-root "${SEQUENCE_INDEX_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --config "${PIPELINE_CONFIG}" \
    --reports "${SEQUENCE_REPORTS}"

run_stage \
  model_matrix 10 \
  "Build train-scaled model feature matrix" \
  "${PYTHON}" "${PROJECT_ROOT}/src/model_matrix.py" \
    --features-root "${FEATURES_PARQUET_ROOT}" \
    --sequence-root "${SEQUENCE_INDEX_ROOT}" \
    --output-root "${MODEL_MATRIX_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --config "${PIPELINE_CONFIG}" \
    --reports "${MODEL_MATRIX_REPORTS}"

run_stage \
  train_model 11 \
  "Train LSTM using train/validation splits" \
  "${PYTHON}" "${PROJECT_ROOT}/src/train_model.py" \
    --model-matrix-root "${MODEL_MATRIX_ROOT}" \
    --output-root "${MODEL_OUTPUT_ROOT}" \
    --symbol "${SYMBOL_LOWER}" \
    --config "${PIPELINE_CONFIG}" \
    --reports "${TRAINING_REPORTS}"

echo "============================================================"
echo "Pipeline completed successfully for ${SYMBOL_UPPER}"
echo "============================================================"