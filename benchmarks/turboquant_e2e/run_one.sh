#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

usage() {
  cat >&2 <<'EOF'
Usage: run_one.sh
  --variant <vanilla|turboquant>
  --model <model-or-path>
  --engine-args <extra vllm serve args>
  --concurrency <int>
  --repeat <int>
  --input-len <int>
  --output-len <int>
  --num-prompts <int>
  --request-rate <float|inf>
  --host <host>
  --port <port>
  --results-dir <dir>
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  printf '[turboquant-e2e:%s] %s\n' "${VARIANT:-run_one}" "$*" >&2
}

quote_cmd() {
  local arg quoted out=""
  for arg in "$@"; do
    printf -v quoted '%q' "${arg}"
    out+="${quoted} "
  done
  printf '%s' "${out% }"
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

VARIANT=""
MODEL=""
ENGINE_ARGS=""
CONCURRENCY=""
REPEAT_INDEX=""
INPUT_LEN=""
OUTPUT_LEN=""
NUM_PROMPTS=""
REQUEST_RATE=""
HOST=""
PORT=""
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT=$2
      shift 2
      ;;
    --model)
      MODEL=$2
      shift 2
      ;;
    --engine-args)
      ENGINE_ARGS=$2
      shift 2
      ;;
    --concurrency)
      CONCURRENCY=$2
      shift 2
      ;;
    --repeat)
      REPEAT_INDEX=$2
      shift 2
      ;;
    --input-len)
      INPUT_LEN=$2
      shift 2
      ;;
    --output-len)
      OUTPUT_LEN=$2
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS=$2
      shift 2
      ;;
    --request-rate)
      REQUEST_RATE=$2
      shift 2
      ;;
    --host)
      HOST=$2
      shift 2
      ;;
    --port)
      PORT=$2
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown argument: $1"
      ;;
  esac
done

[[ "${VARIANT}" == "vanilla" || "${VARIANT}" == "turboquant" ]] \
  || die "--variant must be 'vanilla' or 'turboquant'."
[[ -n "${MODEL}" ]] || die "--model is required."
is_positive_int "${CONCURRENCY}" || die "--concurrency must be a positive integer."
is_positive_int "${REPEAT_INDEX}" || die "--repeat must be a positive integer."
is_positive_int "${INPUT_LEN}" || die "--input-len must be a positive integer."
is_positive_int "${OUTPUT_LEN}" || die "--output-len must be a positive integer."
is_positive_int "${NUM_PROMPTS}" || die "--num-prompts must be a positive integer."
[[ -n "${REQUEST_RATE}" ]] || die "--request-rate is required."
[[ -n "${HOST}" ]] || die "--host is required."
is_nonnegative_int "${PORT}" || die "--port must be a nonnegative integer."
[[ -n "${RESULTS_DIR}" ]] || die "--results-dir is required."

: "${DRY_RUN:=0}"
: "${VLLM_BIN:=vllm}"
: "${SERVER_READY_TIMEOUT:=600}"
: "${NUM_WARMUPS:=1}"
: "${WARMUP_PROMPTS:=}"
: "${IGNORE_EOS:=1}"

PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}

RAW_DIR="${RESULTS_DIR}/raw"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RAW_DIR}" "${LOG_DIR}"

RUN_BASENAME="${VARIANT}_c${CONCURRENCY}_r${REPEAT_INDEX}"
RESULT_JSON="${RAW_DIR}/${RUN_BASENAME}.json"
METADATA_JSON="${RAW_DIR}/${RUN_BASENAME}.metadata.json"
SERVER_LOG="${LOG_DIR}/${RUN_BASENAME}.server.log"
BENCH_STDOUT="${LOG_DIR}/${RUN_BASENAME}.bench.stdout"
BENCH_STDERR="${LOG_DIR}/${RUN_BASENAME}.bench.stderr"
WARMUP_STDOUT="${LOG_DIR}/${RUN_BASENAME}.warmup.stdout"
WARMUP_STDERR="${LOG_DIR}/${RUN_BASENAME}.warmup.stderr"
MODELS_JSON="${LOG_DIR}/${RUN_BASENAME}.models.json"
HELP_LOG="${LOG_DIR}/${RUN_BASENAME}.bench_help.txt"

URL_HOST="${HOST}"
if [[ "${URL_HOST}" == "0.0.0.0" || "${URL_HOST}" == "::" ]]; then
  URL_HOST="127.0.0.1"
fi
BASE_URL="http://${URL_HOST}:${PORT}"

ENGINE_ARGS_ARRAY=()
if [[ -n "${ENGINE_ARGS}" ]]; then
  # Extra engine args are trusted local input. eval preserves quoted env values
  # such as TURBOQUANT_ENGINE_ARGS='--json-arg "{\"k\": 1}"'.
  eval "ENGINE_ARGS_ARRAY=(${ENGINE_ARGS})"
fi

SERVE_CMD=("${VLLM_BIN}" serve "${MODEL}" --host "${HOST}" --port "${PORT}")
SERVE_CMD+=("${ENGINE_ARGS_ARRAY[@]}")

BENCH_CMD=()
WARMUP_CMD=()
SERVE_COMMAND_STR=$(quote_cmd "${SERVE_CMD[@]}")
BENCH_COMMAND_STR=""
WARMUP_COMMAND_STR=""
VLLM_VERSION=""
GIT_COMMIT=""
GIT_BRANCH=""
GPU_NAME=""
CUDA_VERSION=""
START_TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
END_TIMESTAMP=""
STATUS="failed"
EXIT_CODE=1
SERVER_PID=""
METADATA_READY=1
METADATA_WRITTEN=0

collect_vllm_version() {
  "${VLLM_BIN}" --version 2>/dev/null \
    || "${PYTHON_BIN}" -c 'import vllm; print(getattr(vllm, "__version__", "unknown"))' 2>/dev/null \
    || printf 'unknown\n'
}

collect_gpu_name() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_names
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
      | tr '\n' ';' \
      | sed 's/;$//' || true)
    printf '%s' "${gpu_names:-unknown}"
  else
    printf 'unknown'
  fi
}

collect_cuda_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_version
    cuda_version=$(nvidia-smi 2>/dev/null \
      | sed -n 's/.*CUDA Version: \([^ ]*\).*/\1/p' \
      | head -n 1 || true)
    printf '%s' "${cuda_version:-unknown}"
  else
    printf 'unknown'
  fi
}

write_metadata() {
  local status=$1
  local exit_code=$2

  END_TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

  export META_VARIANT="${VARIANT}"
  export META_MODEL="${MODEL}"
  export META_ENGINE_ARGS="${ENGINE_ARGS}"
  export META_CONCURRENCY="${CONCURRENCY}"
  export META_REPEAT_INDEX="${REPEAT_INDEX}"
  export META_INPUT_LEN="${INPUT_LEN}"
  export META_OUTPUT_LEN="${OUTPUT_LEN}"
  export META_NUM_PROMPTS="${NUM_PROMPTS}"
  export META_REQUEST_RATE="${REQUEST_RATE}"
  export META_HOST="${HOST}"
  export META_PORT="${PORT}"
  export META_BASE_URL="${BASE_URL}"
  export META_RESULT_JSON="${RESULT_JSON}"
  export META_METADATA_JSON="${METADATA_JSON}"
  export META_SERVER_LOG="${SERVER_LOG}"
  export META_BENCH_STDOUT="${BENCH_STDOUT}"
  export META_BENCH_STDERR="${BENCH_STDERR}"
  export META_WARMUP_STDOUT="${WARMUP_STDOUT}"
  export META_WARMUP_STDERR="${WARMUP_STDERR}"
  export META_MODELS_JSON="${MODELS_JSON}"
  export META_HELP_LOG="${HELP_LOG}"
  export META_SERVE_COMMAND="${SERVE_COMMAND_STR}"
  export META_BENCH_COMMAND="${BENCH_COMMAND_STR}"
  export META_WARMUP_COMMAND="${WARMUP_COMMAND_STR}"
  export META_GIT_COMMIT="${GIT_COMMIT}"
  export META_GIT_BRANCH="${GIT_BRANCH}"
  export META_VLLM_VERSION="${VLLM_VERSION}"
  export META_GPU_NAME="${GPU_NAME}"
  export META_CUDA_VERSION="${CUDA_VERSION}"
  export META_START_TIMESTAMP="${START_TIMESTAMP}"
  export META_END_TIMESTAMP="${END_TIMESTAMP}"
  export META_STATUS="${status}"
  export META_EXIT_CODE="${exit_code}"

  "${PYTHON_BIN}" - "${METADATA_JSON}" <<'PY'
import json
import os
import sys

path = sys.argv[1]


def int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


data = {
    "variant": os.environ["META_VARIANT"],
    "model": os.environ["META_MODEL"],
    "engine_args": os.environ["META_ENGINE_ARGS"],
    "concurrency": int_env("META_CONCURRENCY"),
    "repeat_index": int_env("META_REPEAT_INDEX"),
    "input_len": int_env("META_INPUT_LEN"),
    "output_len": int_env("META_OUTPUT_LEN"),
    "num_prompts": int_env("META_NUM_PROMPTS"),
    "request_rate": os.environ["META_REQUEST_RATE"],
    "host": os.environ["META_HOST"],
    "port": int_env("META_PORT"),
    "base_url": os.environ["META_BASE_URL"],
    "result_json": os.environ["META_RESULT_JSON"],
    "metadata_json": os.environ["META_METADATA_JSON"],
    "server_log": os.environ["META_SERVER_LOG"],
    "benchmark_stdout": os.environ["META_BENCH_STDOUT"],
    "benchmark_stderr": os.environ["META_BENCH_STDERR"],
    "warmup_stdout": os.environ["META_WARMUP_STDOUT"],
    "warmup_stderr": os.environ["META_WARMUP_STDERR"],
    "models_json": os.environ["META_MODELS_JSON"],
    "bench_help": os.environ["META_HELP_LOG"],
    "serve_command": os.environ["META_SERVE_COMMAND"],
    "benchmark_command": os.environ["META_BENCH_COMMAND"],
    "warmup_command": os.environ["META_WARMUP_COMMAND"],
    "git_commit": os.environ["META_GIT_COMMIT"],
    "git_branch": os.environ["META_GIT_BRANCH"],
    "vllm_version": os.environ["META_VLLM_VERSION"],
    "gpu_name": os.environ["META_GPU_NAME"],
    "cuda_version": os.environ["META_CUDA_VERSION"],
    "timestamp": os.environ["META_START_TIMESTAMP"],
    "start_timestamp": os.environ["META_START_TIMESTAMP"],
    "end_timestamp": os.environ["META_END_TIMESTAMP"],
    "status": os.environ["META_STATUS"],
    "exit_code": int_env("META_EXIT_CODE"),
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY

  METADATA_WRITTEN=1
}

cleanup() {
  local code=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${DRY_RUN}" != "1" && "${METADATA_READY}" == "1" && "${METADATA_WRITTEN}" != "1" ]]; then
    write_metadata "${STATUS}" "${code}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN serve: ${SERVE_COMMAND_STR}"
  log "DRY_RUN output: ${RESULT_JSON}"
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || die \
  "Python executable not found at '${PYTHON_BIN}'. Create .venv with uv or set PYTHON_BIN."
command -v curl >/dev/null 2>&1 || die "curl is required to poll /v1/models."

if ! BENCH_HELP=$("${VLLM_BIN}" bench serve --help 2>&1); then
  printf '%s\n' "${BENCH_HELP}" > "${HELP_LOG}"
  die "Failed to run '${VLLM_BIN} bench serve --help'. See ${HELP_LOG}."
fi
printf '%s\n' "${BENCH_HELP}" > "${HELP_LOG}"

help_has() {
  [[ "${BENCH_HELP}" == *"$1"* ]]
}

require_bench_option() {
  help_has "$1" || die "Installed 'vllm bench serve' does not support required option '$1'. See ${HELP_LOG}."
}

require_bench_option "--num-prompts"
require_bench_option "--request-rate"
require_bench_option "--max-concurrency"
require_bench_option "--model"

if ! help_has "--base-url" && { ! help_has "--host" || ! help_has "--port"; }; then
  die "Installed 'vllm bench serve' must support either --base-url or both --host/--port. See ${HELP_LOG}."
fi

if help_has "--input-len" && help_has "--output-len"; then
  LENGTH_ARGS=(--input-len "${INPUT_LEN}" --output-len "${OUTPUT_LEN}")
elif help_has "--random-input-len" && help_has "--random-output-len"; then
  LENGTH_ARGS=(--random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}")
else
  die "Installed 'vllm bench serve' does not support input/output length options. See ${HELP_LOG}."
fi

if help_has "--save-result-json"; then
  SAVE_ARGS=(--save-result-json "${RESULT_JSON}")
elif help_has "--save-result" && help_has "--result-filename"; then
  SAVE_ARGS=(--save-result --result-filename "${RESULT_JSON}")
elif help_has "--output-json"; then
  SAVE_ARGS=(--output-json "${RESULT_JSON}")
else
  die "Installed 'vllm bench serve' does not support a known JSON result output option. See ${HELP_LOG}."
fi

VLLM_VERSION=$(collect_vllm_version | tr '\n' ' ' | sed 's/[[:space:]]*$//')
GIT_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)
GIT_BRANCH=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
GPU_NAME=$(collect_gpu_name)
CUDA_VERSION=$(collect_cuda_version)

log "Starting server on ${BASE_URL}"
"${SERVE_CMD[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  if curl -fsS "${BASE_URL}/v1/models" > "${MODELS_JSON}" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    die "vLLM server exited before readiness. See ${SERVER_LOG}."
  fi
  sleep 2
done

[[ "${ready}" == "1" ]] || die "Timed out waiting for ${BASE_URL}/v1/models. See ${SERVER_LOG}."

BENCH_MODEL=$("${PYTHON_BIN}" - "${MODELS_JSON}" "${MODEL}" <<'PY'
import json
import sys

models_json, fallback = sys.argv[1], sys.argv[2]
try:
    with open(models_json, encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data") or []
    model_id = data[0].get("id") if data else None
    print(model_id or fallback)
except Exception:
    print(fallback)
PY
)

BENCH_BASE_CMD=("${VLLM_BIN}" bench serve --model "${BENCH_MODEL}")
if help_has "--base-url"; then
  BENCH_BASE_CMD+=(--base-url "${BASE_URL}")
else
  BENCH_BASE_CMD+=(--host "${URL_HOST}" --port "${PORT}")
fi

if help_has "--dataset-name"; then
  BENCH_BASE_CMD+=(--dataset-name random)
fi

BENCH_BASE_CMD+=(
  "${LENGTH_ARGS[@]}"
)

BENCH_OPTIONAL_ARGS=()
if help_has "--disable-tqdm"; then
  BENCH_OPTIONAL_ARGS+=(--disable-tqdm)
fi

if [[ "${IGNORE_EOS}" == "1" ]] && help_has "--ignore-eos"; then
  BENCH_OPTIONAL_ARGS+=(--ignore-eos)
fi

BENCH_CMD=(
  "${BENCH_BASE_CMD[@]}"
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --max-concurrency "${CONCURRENCY}"
  "${BENCH_OPTIONAL_ARGS[@]}"
)

if [[ "${NUM_WARMUPS}" != "0" ]] && help_has "--num-warmups"; then
  BENCH_CMD+=(--num-warmups "${NUM_WARMUPS}")
fi

BENCH_CMD+=("${SAVE_ARGS[@]}")
BENCH_COMMAND_STR=$(quote_cmd "${BENCH_CMD[@]}")

if [[ "${NUM_WARMUPS}" != "0" ]] && ! help_has "--num-warmups"; then
  if [[ -z "${WARMUP_PROMPTS}" ]]; then
    if (( NUM_PROMPTS < CONCURRENCY )); then
      WARMUP_PROMPTS="${NUM_PROMPTS}"
    else
      WARMUP_PROMPTS="${CONCURRENCY}"
    fi
  fi
  WARMUP_CMD=(
    "${BENCH_BASE_CMD[@]}"
    --num-prompts "${WARMUP_PROMPTS}"
    --request-rate "${REQUEST_RATE}"
    --max-concurrency "${CONCURRENCY}"
    "${BENCH_OPTIONAL_ARGS[@]}"
  )
  WARMUP_COMMAND_STR=$(quote_cmd "${WARMUP_CMD[@]}")
  log "Running warmup benchmark because --num-warmups is unavailable."
  if "${WARMUP_CMD[@]}" > "${WARMUP_STDOUT}" 2> "${WARMUP_STDERR}"; then
    :
  else
    EXIT_CODE=$?
    STATUS="failed"
    write_metadata "${STATUS}" "${EXIT_CODE}"
    exit "${EXIT_CODE}"
  fi
fi

log "Running benchmark: ${BENCH_COMMAND_STR}"
if "${BENCH_CMD[@]}" > "${BENCH_STDOUT}" 2> "${BENCH_STDERR}"; then
  STATUS="success"
  EXIT_CODE=0
else
  EXIT_CODE=$?
  STATUS="failed"
fi

if [[ "${STATUS}" == "success" && ! -s "${RESULT_JSON}" ]]; then
  STATUS="failed"
  EXIT_CODE=1
  printf 'Expected result JSON was not created: %s\n' "${RESULT_JSON}" >> "${BENCH_STDERR}"
fi

write_metadata "${STATUS}" "${EXIT_CODE}"
exit "${EXIT_CODE}"
