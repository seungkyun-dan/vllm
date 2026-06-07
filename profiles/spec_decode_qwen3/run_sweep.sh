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

K_VALUES=${K_VALUES:-"0 1 2 3 4 6 8"}
LOAD_MODE=${LOAD_MODE:-"concurrency"}
LOAD_VALUES=${LOAD_VALUES:-"1 2 4 8 16 32 64 128"}

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
NUM_PROMPTS=${NUM_PROMPTS:-"256"}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-"0.0"}
TEMPERATURE=${TEMPERATURE:-"0"}
SEED=${SEED:-"0"}
SAVE_DETAILED=${SAVE_DETAILED:-"0"}
PROFILE_BENCH=${PROFILE_BENCH:-"0"}

SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC:-"900"}
COOLDOWN_SEC=${COOLDOWN_SEC:-"5"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d-%H%M%S)"}
RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/results/${RUN_ID}"}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-""}
BENCH_EXTRA_ARGS=${BENCH_EXTRA_ARGS:-""}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-"INFO"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

SERVER_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Stopping vLLM server pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

cleanup() {
  stop_server
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local server_log=$1
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SEC}"); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Server exited before becoming healthy. Log: ${server_log}" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for server. Log: ${server_log}" >&2
  return 1
}

quote_cmd() {
  printf "%q " "$@"
  printf "\n"
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
    "LOAD_VALUES",
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
    "PROFILE_BENCH",
    "SERVER_READY_TIMEOUT_SEC",
    "COOLDOWN_SEC",
    "SERVER_EXTRA_ARGS",
    "BENCH_EXTRA_ARGS",
    "VLLM_LOGGING_LEVEL",
    "CUDA_VISIBLE_DEVICES",
]
config = {key: os.environ.get(key, "") for key in keys}
Path(os.environ["RUN_CONFIG_JSON"]).write_text(json.dumps(config, indent=2) + "\n")
PY
}

PYTHON_CMD_DISPLAY=$(quote_cmd "${PYTHON_CMD[@]}")
if [[ ! -x "${PYTHON_CMD[0]}" ]]; then
  echo "Expected executable Python at ${PYTHON_CMD[0]}" >&2
  echo "Set PYTHON_BIN, create the repo venv, or set CONDA_ENV to an existing conda env" >&2
  exit 1
fi
require_cmd curl

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
  "${RESULT_DIR}/metrics" \
  "${RESULT_DIR}/gpu" \
  "${RESULT_DIR}/commands"

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
LOAD_VALUES=${LOAD_VALUES}
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
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF

export \
  RUN_ID RESULT_DIR PYTHON_BIN PYTHON_CMD_DISPLAY PYTHON_ENV_KIND CONDA_ENV TARGET_MODEL DRAFT_MODEL SERVED_MODEL_NAME \
  K_VALUES LOAD_MODE LOAD_VALUES HOST PORT TENSOR_PARALLEL_SIZE \
  DRAFT_TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION MAX_MODEL_LEN \
  MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS DTYPE ENFORCE_EAGER TRUST_REMOTE_CODE \
  DATASET_NAME INPUT_LEN OUTPUT_LEN NUM_PROMPTS RANDOM_RANGE_RATIO TEMPERATURE \
  SEED SAVE_DETAILED PROFILE_BENCH SERVER_READY_TIMEOUT_SEC COOLDOWN_SEC \
  SERVER_EXTRA_ARGS BENCH_EXTRA_ARGS VLLM_LOGGING_LEVEL CUDA_VISIBLE_DEVICES
write_run_config_json

echo "Writing results to ${RESULT_DIR}"

for k in ${K_VALUES}; do
  stop_server
  sleep "${COOLDOWN_SEC}"

  server_log="${RESULT_DIR}/server_logs/k${k}.log"
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

  quote_cmd "${server_cmd[@]}" > "${RESULT_DIR}/commands/server_k${k}.cmd"
  echo "Starting server for K=${k}; log=${server_log}"
  "${server_cmd[@]}" > "${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"

  for load in ${LOAD_VALUES}; do
    load_tag="${LOAD_MODE}${load}"
    bench_json="${RESULT_DIR}/bench_results/k${k}_${load_tag}.json"
    bench_log="${RESULT_DIR}/bench_logs/k${k}_${load_tag}.log"
    metrics_file="${RESULT_DIR}/metrics/k${k}_${load_tag}.prom"
    gpu_file="${RESULT_DIR}/gpu/k${k}_${load_tag}.csv"
    server_cmd_file="${RESULT_DIR}/commands/server_k${k}.cmd"
    bench_cmd_file="${RESULT_DIR}/commands/bench_k${k}_${load_tag}.cmd"

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
      "result_file=${bench_json}"
      "server_log=${server_log}"
      "bench_log=${bench_log}"
      "metrics_file=${metrics_file}"
      "gpu_file=${gpu_file}"
      "server_command_file=${server_cmd_file}"
      "bench_command_file=${bench_cmd_file}"
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
      "load_value=${load}"
      "load_values=${LOAD_VALUES}"
      "k_values=${K_VALUES}"
      "host=${HOST}"
      "port=${PORT}"
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
      "save_detailed=${SAVE_DETAILED}"
      "profile_bench=${PROFILE_BENCH}"
      "server_ready_timeout_sec=${SERVER_READY_TIMEOUT_SEC}"
      "cooldown_sec=${COOLDOWN_SEC}"
      "server_extra_args=${SERVER_EXTRA_ARGS}"
      "bench_extra_args=${BENCH_EXTRA_ARGS}"
      "vllm_logging_level=${VLLM_LOGGING_LEVEL}"
      "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    )

    if [[ "${LOAD_MODE}" == "concurrency" ]]; then
      bench_cmd+=(--request-rate inf --max-concurrency "${load}")
    else
      bench_cmd+=(--request-rate "${load}")
    fi
    if [[ "${SAVE_DETAILED}" == "1" ]]; then
      bench_cmd+=(--save-detailed)
    fi
    if [[ "${PROFILE_BENCH}" == "1" ]]; then
      bench_cmd+=(--profile)
    fi
    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
      bench_cmd+=(--trust-remote-code)
    fi
    if [[ -n "${BENCH_EXTRA_ARGS}" ]]; then
      read -r -a bench_extra_args <<< "${BENCH_EXTRA_ARGS}"
      bench_cmd+=("${bench_extra_args[@]}")
    fi

    quote_cmd "${bench_cmd[@]}" > "${RESULT_DIR}/commands/bench_k${k}_${load_tag}.cmd"
    echo "Benchmark K=${k} ${LOAD_MODE}=${load}; result=${bench_json}"
    "${bench_cmd[@]}" > "${bench_log}" 2>&1

    curl -fsS "http://${HOST}:${PORT}/metrics" > "${metrics_file}" 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits > "${gpu_file}" 2>/dev/null || true
    fi
  done
done

stop_server
"${SCRIPT_DIR}/summarize_results.sh" "${RESULT_DIR}"
echo "Done. Summary: ${RESULT_DIR}/summary.csv"

