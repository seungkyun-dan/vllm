#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

CONDA_ENV=${CONDA_ENV:-"vllm"}
PYTHON_BIN=${PYTHON_BIN:-""}
PYTHON_ENV_KIND=""
PYTHON_CMD=()
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_ENV_KIND="python_bin"
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_ENV_KIND="venv"
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_ENV_KIND="conda"
  if ! command -v conda >/dev/null 2>&1; then
    echo "Missing required command: conda" >&2
    exit 1
  fi
  if ! PYTHON_BIN=$(conda run -n "${CONDA_ENV}" python -c "import sys; print(sys.executable)"); then
    echo "Unable to find Python for conda env ${CONDA_ENV}" >&2
    exit 1
  fi
  PYTHON_CMD=("${PYTHON_BIN}")
fi

TARGET_MODEL=${TARGET_MODEL:-"Qwen/Qwen3-8B"}
DRAFT_MODEL=${DRAFT_MODEL:-"Qwen/Qwen3-0.6B"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"qwen3-target"}

K_VALUES=${K_VALUES:-"0 1 2 3 4 6"}
LOAD_MODE=${LOAD_MODE:-"concurrency"}
LOAD_VALUE=${LOAD_VALUE:-"1"}

HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"8000"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
DRAFT_TENSOR_PARALLEL_SIZE=${DRAFT_TENSOR_PARALLEL_SIZE:-"${TENSOR_PARALLEL_SIZE}"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"2048"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"128"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-"8192"}
DTYPE=${DTYPE:-"auto"}
ENFORCE_EAGER=${ENFORCE_EAGER:-"0"}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-"0"}

DATASET_NAME=${DATASET_NAME:-"random"}
INPUT_LEN=${INPUT_LEN:-"128"}
OUTPUT_LEN=${OUTPUT_LEN:-"128"}
NUM_PROMPTS=${NUM_PROMPTS:-"32"}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-"0.0"}
TEMPERATURE=${TEMPERATURE:-"0"}
SEED=${SEED:-"0"}
SAVE_DETAILED=${SAVE_DETAILED:-"1"}

NSYS_TRACE=${NSYS_TRACE:-"cuda,nvtx,osrt,cublas,cudnn"}
NSYS_GPU_METRICS_DEVICES=${NSYS_GPU_METRICS_DEVICES:-"none"}
NSYS_EXTRA_ARGS=${NSYS_EXTRA_ARGS:-""}
SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC:-"900"}
COOLDOWN_SEC=${COOLDOWN_SEC:-"5"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d-%H%M%S)"}
RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/nsys_results/${RUN_ID}"}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-""}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS:-""}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-"INFO"}
export VLLM_NVTX_SCOPES_FOR_PROFILING=${VLLM_NVTX_SCOPES_FOR_PROFILING:-"1"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

NSYS_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
}

cleanup() {
  if [[ -n "${NSYS_PID}" ]] && kill -0 "${NSYS_PID}" >/dev/null 2>&1; then
    echo "Stopping nsys/server pid=${NSYS_PID}"
    kill "${NSYS_PID}" >/dev/null 2>&1 || true
    wait "${NSYS_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local server_log=$1
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SEC}"); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${NSYS_PID}" ]] && ! kill -0 "${NSYS_PID}" >/dev/null 2>&1; then
      echo "Server/nsys exited before becoming healthy. Log: ${server_log}" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for server. Log: ${server_log}" >&2
  return 1
}

write_run_config_json() {
  RUN_CONFIG_JSON="${RESULT_DIR}/run_config.json" "${PYTHON_CMD[@]}" - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "RUN_ID",
    "RESULT_DIR",
    "PYTHON_BIN",
    "PYTHON_CMD_DISPLAY",
    "PYTHON_ENV_KIND",
    "CONDA_ENV",
    "TARGET_MODEL",
    "DRAFT_MODEL",
    "SERVED_MODEL_NAME",
    "K_VALUES",
    "LOAD_MODE",
    "LOAD_VALUE",
    "HOST",
    "PORT",
    "TENSOR_PARALLEL_SIZE",
    "DRAFT_TENSOR_PARALLEL_SIZE",
    "GPU_MEMORY_UTILIZATION",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "DTYPE",
    "ENFORCE_EAGER",
    "TRUST_REMOTE_CODE",
    "DATASET_NAME",
    "INPUT_LEN",
    "OUTPUT_LEN",
    "NUM_PROMPTS",
    "RANDOM_RANGE_RATIO",
    "TEMPERATURE",
    "SEED",
    "SAVE_DETAILED",
    "NSYS_TRACE",
    "NSYS_GPU_METRICS_DEVICES",
    "NSYS_EXTRA_ARGS",
    "SERVER_READY_TIMEOUT_SEC",
    "COOLDOWN_SEC",
    "SERVER_EXTRA_ARGS",
    "BENCH_EXTRA_ARGS",
    "VLLM_LOGGING_LEVEL",
    "VLLM_NVTX_SCOPES_FOR_PROFILING",
    "CUDA_VISIBLE_DEVICES",
]
config = {key: os.environ.get(key, "") for key in keys}
Path(os.environ["RUN_CONFIG_JSON"]).write_text(json.dumps(config, indent=2) + "\n")
PY
}

if [[ ! -x "${PYTHON_CMD[0]}" ]]; then
  echo "Expected executable Python at ${PYTHON_CMD[0]}" >&2
  exit 1
fi

require_cmd curl
require_cmd nsys

case "${LOAD_MODE}" in
  concurrency|request_rate) ;;
  *)
    echo "LOAD_MODE must be 'concurrency' or 'request_rate', got ${LOAD_MODE}" >&2
    exit 1
    ;;
esac

mkdir -p \
  "${RESULT_DIR}/bench_results" \
  "${RESULT_DIR}/bench_logs" \
  "${RESULT_DIR}/server_logs" \
  "${RESULT_DIR}/nsys" \
  "${RESULT_DIR}/stats" \
  "${RESULT_DIR}/commands"

PYTHON_CMD_DISPLAY=$(quote_cmd "${PYTHON_CMD[@]}")
export \
  RUN_ID RESULT_DIR PYTHON_BIN PYTHON_CMD_DISPLAY PYTHON_ENV_KIND CONDA_ENV \
  TARGET_MODEL DRAFT_MODEL SERVED_MODEL_NAME K_VALUES LOAD_MODE LOAD_VALUE \
  HOST PORT TENSOR_PARALLEL_SIZE DRAFT_TENSOR_PARALLEL_SIZE \
  GPU_MEMORY_UTILIZATION MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
  DTYPE ENFORCE_EAGER TRUST_REMOTE_CODE DATASET_NAME INPUT_LEN OUTPUT_LEN \
  NUM_PROMPTS RANDOM_RANGE_RATIO TEMPERATURE SEED SAVE_DETAILED NSYS_TRACE \
  NSYS_GPU_METRICS_DEVICES NSYS_EXTRA_ARGS SERVER_READY_TIMEOUT_SEC \
  COOLDOWN_SEC SERVER_EXTRA_ARGS BENCH_EXTRA_ARGS VLLM_LOGGING_LEVEL \
  VLLM_NVTX_SCOPES_FOR_PROFILING CUDA_VISIBLE_DEVICES
write_run_config_json

cat > "${RESULT_DIR}/run_config.env" <<EOF
PYTHON_BIN=${PYTHON_BIN}
PYTHON_CMD_DISPLAY=${PYTHON_CMD_DISPLAY}
PYTHON_ENV_KIND=${PYTHON_ENV_KIND}
CONDA_ENV=${CONDA_ENV}
TARGET_MODEL=${TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
K_VALUES=${K_VALUES}
LOAD_MODE=${LOAD_MODE}
LOAD_VALUE=${LOAD_VALUE}
INPUT_LEN=${INPUT_LEN}
OUTPUT_LEN=${OUTPUT_LEN}
NUM_PROMPTS=${NUM_PROMPTS}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
DRAFT_TENSOR_PARALLEL_SIZE=${DRAFT_TENSOR_PARALLEL_SIZE}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
DTYPE=${DTYPE}
NSYS_TRACE=${NSYS_TRACE}
NSYS_GPU_METRICS_DEVICES=${NSYS_GPU_METRICS_DEVICES}
VLLM_NVTX_SCOPES_FOR_PROFILING=${VLLM_NVTX_SCOPES_FOR_PROFILING}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF

echo "Writing Nsight results to ${RESULT_DIR}"

for k in ${K_VALUES}; do
  sleep "${COOLDOWN_SEC}"

  server_log="${RESULT_DIR}/server_logs/k${k}.log"
  nsys_base="${RESULT_DIR}/nsys/k${k}_${LOAD_MODE}${LOAD_VALUE}"
  spec_config=""
  spec_enabled="0"
  spec_method="none"

  server_cmd=(
    "${PYTHON_CMD[@]}" -m vllm.entrypoints.cli.main serve "${TARGET_MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --dtype "${DTYPE}"
    --profiler-config '{"profiler":"cuda"}'
  )

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    server_cmd+=(--trust-remote-code)
  fi
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    server_cmd+=(--enforce-eager)
  fi
  if [[ "${k}" != "0" ]]; then
    spec_enabled="1"
    spec_method="draft_model"
    spec_config=$(
      printf '{"method":"draft_model","model":"%s","num_speculative_tokens":%s,"draft_tensor_parallel_size":%s}' \
        "${DRAFT_MODEL}" "${k}" "${DRAFT_TENSOR_PARALLEL_SIZE}"
    )
    server_cmd+=(--speculative-config "${spec_config}")
  fi
  if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then
    read -r -a server_extra_args <<< "${SERVER_EXTRA_ARGS}"
    server_cmd+=("${server_extra_args[@]}")
  fi

  nsys_cmd=(
    nsys profile
    --wait=all
    --trace="${NSYS_TRACE}"
    --sample=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop-shutdown
    --export=sqlite
    --force-overwrite=true
    --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES}"
    -o "${nsys_base}"
  )
  if [[ -n "${NSYS_EXTRA_ARGS}" ]]; then
    read -r -a nsys_extra_args <<< "${NSYS_EXTRA_ARGS}"
    nsys_cmd+=("${nsys_extra_args[@]}")
  fi
  nsys_cmd+=("${server_cmd[@]}")

  quote_cmd "${server_cmd[@]}" > "${RESULT_DIR}/commands/server_k${k}.cmd"
  quote_cmd "${nsys_cmd[@]}" > "${RESULT_DIR}/commands/nsys_server_k${k}.cmd"
  echo "Starting profiled server for K=${k}; log=${server_log}"
  "${nsys_cmd[@]}" > "${server_log}" 2>&1 &
  NSYS_PID=$!
  wait_for_server "${server_log}"

  bench_json="${RESULT_DIR}/bench_results/k${k}_${LOAD_MODE}${LOAD_VALUE}.json"
  bench_log="${RESULT_DIR}/bench_logs/k${k}_${LOAD_MODE}${LOAD_VALUE}.log"
  bench_cmd=(
    "${PYTHON_CMD[@]}" -m vllm.entrypoints.cli.main bench serve
    --backend openai
    --host "${HOST}"
    --port "${PORT}"
    --endpoint /v1/completions
    --model "${TARGET_MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tokenizer "${TARGET_MODEL}"
    --dataset-name "${DATASET_NAME}"
    --random-input-len "${INPUT_LEN}"
    --random-output-len "${OUTPUT_LEN}"
    --random-range-ratio "${RANDOM_RANGE_RATIO}"
    --num-prompts "${NUM_PROMPTS}"
    --seed "${SEED}"
    --temperature "${TEMPERATURE}"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,95,99
    --save-result
    --result-dir "${RESULT_DIR}/bench_results"
    --result-filename "$(basename -- "${bench_json}")"
    --metadata
    "run_id=${RUN_ID}"
    "result_dir=${RESULT_DIR}"
    "nsys_report_base=${nsys_base}"
    "python_bin=${PYTHON_BIN}"
    "python_cmd=${PYTHON_CMD_DISPLAY}"
    "python_env_kind=${PYTHON_ENV_KIND}"
    "conda_env=${CONDA_ENV}"
    "target_model=${TARGET_MODEL}"
    "draft_model=${DRAFT_MODEL}"
    "served_model_name=${SERVED_MODEL_NAME}"
    "speculative_enabled=${spec_enabled}"
    "speculative_method=${spec_method}"
    "num_speculative_tokens=${k}"
    "speculative_config_json=${spec_config}"
    "load_mode=${LOAD_MODE}"
    "load_value=${LOAD_VALUE}"
    "k_values=${K_VALUES}"
    "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    "draft_tensor_parallel_size=${DRAFT_TENSOR_PARALLEL_SIZE}"
    "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "max_model_len=${MAX_MODEL_LEN}"
    "max_num_seqs=${MAX_NUM_SEQS}"
    "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "dtype=${DTYPE}"
    "enforce_eager=${ENFORCE_EAGER}"
    "trust_remote_code=${TRUST_REMOTE_CODE}"
    "dataset_name=${DATASET_NAME}"
    "input_len=${INPUT_LEN}"
    "output_len=${OUTPUT_LEN}"
    "num_prompts=${NUM_PROMPTS}"
    "random_range_ratio=${RANDOM_RANGE_RATIO}"
    "temperature=${TEMPERATURE}"
    "seed=${SEED}"
    "vllm_nvtx_scopes_for_profiling=${VLLM_NVTX_SCOPES_FOR_PROFILING}"
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  )

  if [[ "${LOAD_MODE}" == "concurrency" ]]; then
    bench_cmd+=(--request-rate inf --max-concurrency "${LOAD_VALUE}")
  else
    bench_cmd+=(--request-rate "${LOAD_VALUE}")
  fi
  if [[ "${SAVE_DETAILED}" == "1" ]]; then
    bench_cmd+=(--save-detailed)
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    bench_cmd+=(--trust-remote-code)
  fi
  if [[ -n "${BENCH_EXTRA_ARGS}" ]]; then
    read -r -a bench_extra_args <<< "${BENCH_EXTRA_ARGS}"
    bench_cmd+=("${bench_extra_args[@]}")
  fi

  quote_cmd "${bench_cmd[@]}" > "${RESULT_DIR}/commands/bench_k${k}.cmd"
  echo "Starting CUDA profiler for K=${k}"
  curl -fsS -X POST "http://${HOST}:${PORT}/start_profile" >/dev/null

  echo "Benchmark K=${k} ${LOAD_MODE}=${LOAD_VALUE}; result=${bench_json}"
  "${bench_cmd[@]}" > "${bench_log}" 2>&1

  echo "Stopping CUDA profiler for K=${k}"
  curl -fsS -X POST "http://${HOST}:${PORT}/stop_profile" >/dev/null || true

  wait "${NSYS_PID}" >/dev/null 2>&1 || true
  NSYS_PID=""

  sqlite_path="${nsys_base}.sqlite"
  if [[ -f "${sqlite_path}" ]]; then
    stats_dir="${RESULT_DIR}/stats/k${k}_${LOAD_MODE}${LOAD_VALUE}"
    mkdir -p "${stats_dir}"
    nsys stats \
      --force-overwrite true \
      --report nvtx_sum \
      --report nvtx_gpu_proj_sum \
      --report nvtx_gpu_proj_trace \
      --report nvtx_kern_sum \
      --format csv \
      --output "${stats_dir}/nsys" \
      "${sqlite_path}" >/dev/null || true
    "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/extract_nsys_forward_ranges.py" \
      "${sqlite_path}" \
      --out-dir "${stats_dir}" \
      --prefix "k${k}_${LOAD_MODE}${LOAD_VALUE}"
  else
    echo "Missing expected SQLite export: ${sqlite_path}" >&2
  fi
done

"${SCRIPT_DIR}/summarize_results.sh" "${RESULT_DIR}" || true
echo "Done. Nsight summaries: ${RESULT_DIR}/stats"
