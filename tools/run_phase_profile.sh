#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

PHASE="${PHASE:-phase0}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
BACKEND="${BACKEND:-openai-chat}"
DATASET_NAME="${DATASET_NAME:-random}"
TEMPERATURE="${TEMPERATURE:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
NUM_PROMPTS_SMALL="${NUM_PROMPTS_SMALL:-128}"
NUM_PROMPTS_LARGE="${NUM_PROMPTS_LARGE:-512}"
NUM_PROMPTS_PHASE0="${NUM_PROMPTS_PHASE0:-8}"
NUM_PROMPTS_PHASE5_SPEC="${NUM_PROMPTS_PHASE5_SPEC:-256}"
NUM_PROMPTS_PHASE6="${NUM_PROMPTS_PHASE6:-1024}"
NUM_PROMPTS_PHASE7="${NUM_PROMPTS_PHASE7:-512}"
NUM_PROMPTS_PHASE7_SHORT_EASY="${NUM_PROMPTS_PHASE7_SHORT_EASY:-256}"
NUM_PROMPTS_PHASE7_AMORT="${NUM_PROMPTS_PHASE7_AMORT:-256}"
REQUEST_RATES="${REQUEST_RATES:-}"
CAPACITY_C="${CAPACITY_C:-}"
MAX_CONCURRENCY_CAP="${MAX_CONCURRENCY_CAP:-}"
DEFAULT_TOKEN_BUDGET="${DEFAULT_TOKEN_BUDGET:-8192}"
HOST="${HOST:-127.0.0.1}"
USE_VENV="${USE_VENV:-0}"
RUN_PLOTS="${RUN_PLOTS:-1}"
STRICT_MAX_MODEL_LEN="${STRICT_MAX_MODEL_LEN:-0}"
INCLUDE_PHASE4_OPTIONAL="${INCLUDE_PHASE4_OPTIONAL:-0}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-120}"
SERVER_READY_INTERVAL="${SERVER_READY_INTERVAL:-5}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
OUT_ROOT="${OUT_ROOT:-/tmp/vllm_phase_profile_runs/${TIMESTAMP}}"
SERVER_PID=""
PYTHON_BIN=""
SERVE_HELP_FILE=""
BENCH_HELP_FILE=""
VLLM_CMD=()

case "${BACKEND}" in
  openai-chat) BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/chat/completions}" ;;
  *) BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/completions}" ;;
esac

log() {
  printf '[phase-profile] %s\n' "$*" >&2
}

die() {
  printf '[phase-profile] ERROR: %s\n' "$*" >&2
  exit 1
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server pid ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${conda_base}/etc/profile.d/conda.sh"
      if conda env list | awk '{print $1}' | grep -qx profile; then
        conda activate profile
        log "activated conda environment: profile"
      fi
    fi
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" != "profile" ]]; then
    if [[ "${USE_VENV}" == "1" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "${REPO_ROOT}/.venv/bin/activate"
      log "activated .venv because USE_VENV=1 and conda profile was unavailable"
    else
      die "conda environment 'profile' is not active/available. Run: conda activate profile"
    fi
  fi

  PYTHON_BIN="$(command -v python || true)"
  [[ -n "${PYTHON_BIN}" ]] || die "python was not found in the active environment"
  if command -v vllm >/dev/null 2>&1; then
    VLLM_CMD=(vllm)
  else
    VLLM_CMD=("${PYTHON_BIN}" -m vllm.entrypoints.cli.main)
    log "vllm console script not found; using python -m vllm.entrypoints.cli.main"
  fi
}

write_env_report() {
  local report_path="${OUT_ROOT}/env_report.txt"
  {
    printf 'pwd %s\n' "${REPO_ROOT}"
    printf 'python %s\n' "${PYTHON_BIN}"
    "${PYTHON_BIN}" - <<'PYREPORT'
import sys

print("python_version", sys.version.replace("\n", " "))
try:
    import vllm.env_override  # Applies optional CUDA compatibility path.
    import vllm

    print("vllm_file", vllm.__file__)
    print("vllm_version", getattr(vllm, "__version__", "unknown"))
except Exception as exc:
    print("vllm_error", repr(exc))

try:
    import torch

    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_error", repr(exc))
PYREPORT
  } >"${report_path}" 2>&1
  log "wrote environment report to ${report_path}"
}

validate_cuda_ready() {
  if [[ "${SKIP_CUDA_PREFLIGHT:-0}" == "1" ]]; then
    log "skipping CUDA preflight because SKIP_CUDA_PREFLIGHT=1"
    return 0
  fi

  local preflight_path="${OUT_ROOT}/cuda_preflight.txt"
  if ! "${PYTHON_BIN}" - <<'PYCHECK' >"${preflight_path}" 2>&1; then
import sys

try:
    import vllm.env_override  # Applies optional CUDA compatibility path.
except Exception as exc:
    print("vllm_env_override_error", repr(exc))

try:
    import torch
except Exception as exc:
    raise SystemExit(f"CUDA preflight failed: cannot import torch: {exc!r}")

print("python", sys.executable)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
try:
    available = torch.cuda.is_available()
    print("cuda_available", available)
    print("device_count", torch.cuda.device_count())
    if not available:
        raise SystemExit(
            "CUDA preflight failed: torch.cuda.is_available() is False. "
            "Check the NVIDIA driver/PyTorch CUDA build or enable CUDA "
            "forward compatibility."
        )
    torch.cuda.set_device(0)
    print("device_name", torch.cuda.get_device_name(0))
except Exception as exc:
    raise SystemExit(f"CUDA preflight failed: {exc!r}") from exc
PYCHECK
    cat "${preflight_path}" >&2 || true
    die "CUDA preflight failed; see ${preflight_path}"
  fi
  log "CUDA preflight passed; details: ${preflight_path}"
}

capture_help() {
  SERVE_HELP_FILE="${OUT_ROOT}/vllm_serve_help.txt"
  BENCH_HELP_FILE="${OUT_ROOT}/vllm_bench_serve_help.txt"
  if ! "${VLLM_CMD[@]}" serve --help=all >"${SERVE_HELP_FILE}" 2>&1; then
    "${VLLM_CMD[@]}" serve --help >"${SERVE_HELP_FILE}" 2>&1 || true
  fi
  "${VLLM_CMD[@]}" bench serve --help >"${BENCH_HELP_FILE}" 2>&1 || true
}

help_has() {
  local pattern="$1"
  grep -q -- "${pattern}" "${SERVE_HELP_FILE}"
}

bench_help_has() {
  local pattern="$1"
  grep -q -- "${pattern}" "${BENCH_HELP_FILE}"
}

chunked_args() {
  local chunked="$1"
  case "${chunked}" in
    on)
      if help_has "--enable-chunked-prefill"; then
        printf '%s\n' "--enable-chunked-prefill"
      else
        log "WARNING: --enable-chunked-prefill is unavailable; using server default"
      fi
      ;;
    off)
      if help_has "--no-enable-chunked-prefill"; then
        printf '%s\n' "--no-enable-chunked-prefill"
      else
        log "WARNING: --no-enable-chunked-prefill is unavailable; using server default"
      fi
      ;;
    default) ;;
    *) die "unknown chunked_prefill value: ${chunked}" ;;
  esac
}

effective_max_model_len() {
  local token_budget="$1"
  local chunked="$2"
  local total_len="$3"
  local effective="${MAX_MODEL_LEN}"
  if [[ "${chunked}" == "off" && "${token_budget}" -lt "${MAX_MODEL_LEN}" ]]; then
    if [[ "${STRICT_MAX_MODEL_LEN}" == "1" ]]; then
      die "chunked_prefill=off with token_budget=${token_budget} is invalid when MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    fi
    effective="${token_budget}"
    log "using effective max_model_len=${effective} for chunked_prefill=off and token_budget=${token_budget}"
  fi
  if [[ "${total_len}" -gt "${effective}" ]]; then
    die "input_len + output_len (${total_len}) exceeds effective max_model_len (${effective})"
  fi
  printf '%s\n' "${effective}"
}

sanitize_request_rate() {
  local rate="$1"
  rate="${rate//./p}"
  rate="${rate//+/}"
  rate="${rate//-/m}"
  printf '%s\n' "${rate}"
}

phase6_request_rates() {
  if [[ -n "${REQUEST_RATES}" ]]; then
    printf '%s\n' ${REQUEST_RATES}
  elif [[ -n "${CAPACITY_C}" ]]; then
    awk -v c="${CAPACITY_C}" 'BEGIN {
      split("0.25 0.5 0.75 1.0 1.25", factors, " ");
      for (idx = 1; idx <= 5; idx++) {
        printf "%g\n", c * factors[idx];
      }
    }'
  else
    log "WARNING: REQUEST_RATES and CAPACITY_C are unset; using placeholder rates 1 2 4 8 16. Replace these after measuring capacity."
    printf '%s\n' 1 2 4 8 16
  fi
}

write_metadata() {
  local path="$1"
  local phase="$2"
  local case_name="$3"
  local k="$4"
  local input_len="$5"
  local output_len="$6"
  local concurrency="$7"
  local token_budget="$8"
  local chunked="$9"
  local num_prompts="${10}"
  local max_num_seqs="${11}"
  local effective_len="${12}"
  cat >"${path}" <<EOF
{
  "phase": "${phase}",
  "case": "${case_name}",
  "k": ${k},
  "input_len": ${input_len},
  "output_len": ${output_len},
  "max_concurrency": ${concurrency},
  "max_num_batched_tokens": ${token_budget},
  "chunked_prefill": "${chunked}",
  "num_prompts": ${num_prompts},
  "max_num_seqs": ${max_num_seqs},
  "max_model_len": ${MAX_MODEL_LEN},
  "effective_max_model_len": ${effective_len},
  "target_model": "${TARGET_MODEL}",
  "draft_model": "${DRAFT_MODEL}",
  "backend": "${BACKEND}",
  "dataset_name": "${DATASET_NAME}",
  "temperature": ${TEMPERATURE}
}
EOF
}

write_metadata_request_rate() {
  local path="$1"
  local phase="$2"
  local case_name="$3"
  local k="$4"
  local input_len="$5"
  local output_len="$6"
  local request_rate="$7"
  local token_budget="$8"
  local chunked="$9"
  local num_prompts="${10}"
  local max_num_seqs="${11}"
  local effective_len="${12}"
  local cap_value="null"
  if [[ -n "${MAX_CONCURRENCY_CAP}" ]]; then
    cap_value="${MAX_CONCURRENCY_CAP}"
  fi
  cat >"${path}" <<EOF
{
  "phase": "${phase}",
  "case": "${case_name}",
  "k": ${k},
  "input_len": ${input_len},
  "output_len": ${output_len},
  "requested_request_rate": ${request_rate},
  "max_concurrency_cap": ${cap_value},
  "max_num_batched_tokens": ${token_budget},
  "chunked_prefill": "${chunked}",
  "num_prompts": ${num_prompts},
  "max_num_seqs": ${max_num_seqs},
  "max_model_len": ${MAX_MODEL_LEN},
  "effective_max_model_len": ${effective_len},
  "target_model": "${TARGET_MODEL}",
  "draft_model": "${DRAFT_MODEL}",
  "backend": "${BACKEND}",
  "dataset_name": "${DATASET_NAME}",
  "temperature": ${TEMPERATURE}
}
EOF
}

wait_for_server() {
  local server_log="$1"
  local url="http://${HOST}:${PORT}/health"
  for _ in $(seq 1 "${SERVER_READY_ATTEMPTS}"); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 80 "${server_log}" >&2 || true
      die "server exited before becoming healthy; see ${server_log}"
    fi
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${SERVER_READY_INTERVAL}"
  done
  tail -n 80 "${server_log}" >&2 || true
  die "server did not become healthy at ${url}; see ${server_log}"
}

run_analyzers() {
  local phase="$1"
  local run_dir="$2"
  local trace_path="$3"
  local analyzer_log="${run_dir}/analyzer.log"
  : >"${analyzer_log}"
  if [[ ! -s "${trace_path}" ]]; then
    printf 'WARNING: trace file is missing or empty: %s\n' \
      "${trace_path}" >>"${analyzer_log}"
    return 0
  fi

  if [[ -f "${SCRIPT_DIR}/analyze_batch_composition_trace.py" ]]; then
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_batch_composition_trace.py" \
      --trace "${trace_path}" \
      --out-dir "${run_dir}" >>"${analyzer_log}" 2>&1; then
      log "WARNING: batch composition analyzer failed for ${run_dir}; see ${analyzer_log}"
    fi
  else
    printf 'WARNING: missing optional analyzer: analyze_batch_composition_trace.py\n' \
      >>"${analyzer_log}"
  fi

  if [[ "${phase}" == "phase7" ]]; then
    if [[ -f "${SCRIPT_DIR}/analyze_spec_ttft_trace.py" ]]; then
      if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_spec_ttft_trace.py" \
        --trace "${trace_path}" \
        --out-dir "${run_dir}" >>"${analyzer_log}" 2>&1; then
        log "WARNING: spec TTFT analyzer failed for ${run_dir}; see ${analyzer_log}"
      fi
    else
      printf 'WARNING: missing optional analyzer: analyze_spec_ttft_trace.py\n' \
        >>"${analyzer_log}"
    fi
  fi

  if [[ ( "${phase}" == "phase5" || "${phase}" == "phase7" ) \
      && -f "${run_dir}/request_lifecycle.csv" ]]; then
    if [[ -f "${SCRIPT_DIR}/analyze_remaining_lifetime.py" ]]; then
      if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_remaining_lifetime.py" \
        --request-csv "${run_dir}/request_lifecycle.csv" \
        --out-dir "${run_dir}" >>"${analyzer_log}" 2>&1; then
        log "WARNING: remaining lifetime analyzer failed for ${run_dir}; see ${analyzer_log}"
      fi
    else
      printf 'WARNING: missing optional analyzer: analyze_remaining_lifetime.py\n' \
        >>"${analyzer_log}"
    fi
  fi
}

run_one() {
  local phase="$1"
  local case_name="$2"
  local k="$3"
  local input_len="$4"
  local output_len="$5"
  local concurrency="$6"
  local token_budget="$7"
  local chunked="$8"
  local num_prompts="$9"
  local max_num_seqs="${10:-${MAX_NUM_SEQS}}"

  local total_len=$((input_len + output_len))
  local effective_len
  effective_len="$(effective_max_model_len "${token_budget}" "${chunked}" "${total_len}")"

  local run_name="${phase}_${case_name}_k${k}_in${input_len}_out${output_len}_c${concurrency}_tb${token_budget}_chunked_${chunked}"
  local phase_dir="${OUT_ROOT}/${phase}"
  local run_dir="${phase_dir}/${run_name}"
  local trace_path="${run_dir}/trace.jsonl"
  local server_log="${run_dir}/server.log"
  local benchmark_log="${run_dir}/benchmark.log"
  mkdir -p "${run_dir}"

  write_metadata "${run_dir}/metadata.json" "${phase}" "${case_name}" "${k}" \
    "${input_len}" "${output_len}" "${concurrency}" "${token_budget}" \
    "${chunked}" "${num_prompts}" "${max_num_seqs}" "${effective_len}"

  local server_cmd=(
    "${VLLM_CMD[@]}" serve "${TARGET_MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --max-model-len "${effective_len}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${max_num_seqs}"
    --max-num-batched-tokens "${token_budget}"
  )
  while IFS= read -r arg; do
    [[ -n "${arg}" ]] && server_cmd+=("${arg}")
  done < <(chunked_args "${chunked}")

  if [[ "${k}" -gt 0 ]]; then
    help_has "--speculative-config" || die "--speculative-config is unavailable in this vLLM CLI"
    server_cmd+=(
      --speculative-config
      "{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${k}}"
    )
  fi

  local bench_cmd=(
    "${VLLM_CMD[@]}" bench serve
    --backend "${BACKEND}"
    --endpoint "${BENCH_ENDPOINT}"
    --base-url "http://${HOST}:${PORT}"
    --model "${TARGET_MODEL}"
    --dataset-name "${DATASET_NAME}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
    --temperature "${TEMPERATURE}"
    --num-prompts "${num_prompts}"
    --max-concurrency "${concurrency}"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --metric-percentiles "50,95,99"
    --save-result
    --save-detailed
    --result-dir "${run_dir}"
    --result-filename "benchmark_result.json"
    --disable-tqdm
  )
  if bench_help_has "--request-id-prefix"; then
    bench_cmd+=(--request-id-prefix "${run_name}-")
  fi

  {
    printf 'server:\n'
    printf 'VLLM_USE_V1=1 VLLM_SPEC_TRACE=1 VLLM_SPEC_TRACE_FILE=%q VLLM_SPEC_TTFT_TRACE=1 VLLM_SPEC_TTFT_TRACE_FILE=%q ' "${trace_path}" "${trace_path}"
    quote_cmd "${server_cmd[@]}"
    printf '\nbenchmark:\n'
    quote_cmd "${bench_cmd[@]}"
  } >"${run_dir}/command.txt"

  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    die "port ${PORT} already has a healthy vLLM server; stop it or set PORT"
  fi

  log "starting ${run_name}"
  env \
    VLLM_USE_V1=1 \
    VLLM_SPEC_TRACE=1 \
    VLLM_SPEC_TRACE_FILE="${trace_path}" \
    VLLM_SPEC_TTFT_TRACE=1 \
    VLLM_SPEC_TTFT_TRACE_FILE="${trace_path}" \
    "${server_cmd[@]}" >"${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"

  "${bench_cmd[@]}" >"${benchmark_log}" 2>&1
  stop_server
  run_analyzers "${phase}" "${run_dir}" "${trace_path}"
  log "finished ${run_name}"
}


run_one_request_rate() {
  local phase="$1"
  local case_name="$2"
  local k="$3"
  local input_len="$4"
  local output_len="$5"
  local request_rate="$6"
  local token_budget="$7"
  local chunked="$8"
  local num_prompts="$9"
  local max_num_seqs="${10:-${MAX_NUM_SEQS}}"

  bench_help_has "--request-rate" || die "--request-rate is unavailable in this benchmark CLI"

  local total_len=$((input_len + output_len))
  local effective_len
  effective_len="$(effective_max_model_len "${token_budget}" "${chunked}" "${total_len}")"

  local safe_rate
  safe_rate="$(sanitize_request_rate "${request_rate}")"
  local run_name="${phase}_${case_name}_k${k}_in${input_len}_out${output_len}_rr${safe_rate}_tb${token_budget}_chunked_${chunked}"
  local phase_dir="${OUT_ROOT}/${phase}"
  local run_dir="${phase_dir}/${run_name}"
  local trace_path="${run_dir}/trace.jsonl"
  local server_log="${run_dir}/server.log"
  local benchmark_log="${run_dir}/benchmark.log"
  mkdir -p "${run_dir}"

  write_metadata_request_rate "${run_dir}/metadata.json" "${phase}" \
    "${case_name}" "${k}" "${input_len}" "${output_len}" \
    "${request_rate}" "${token_budget}" "${chunked}" "${num_prompts}" \
    "${max_num_seqs}" "${effective_len}"

  local server_cmd=(
    "${VLLM_CMD[@]}" serve "${TARGET_MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --max-model-len "${effective_len}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${max_num_seqs}"
    --max-num-batched-tokens "${token_budget}"
  )
  while IFS= read -r arg; do
    [[ -n "${arg}" ]] && server_cmd+=("${arg}")
  done < <(chunked_args "${chunked}")

  if [[ "${k}" -gt 0 ]]; then
    help_has "--speculative-config" || die "--speculative-config is unavailable in this vLLM CLI"
    server_cmd+=(
      --speculative-config
      "{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${k}}"
    )
  fi

  local bench_cmd=(
    "${VLLM_CMD[@]}" bench serve
    --backend "${BACKEND}"
    --endpoint "${BENCH_ENDPOINT}"
    --base-url "http://${HOST}:${PORT}"
    --model "${TARGET_MODEL}"
    --dataset-name "${DATASET_NAME}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
    --temperature "${TEMPERATURE}"
    --num-prompts "${num_prompts}"
    --request-rate "${request_rate}"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --metric-percentiles "50,95,99"
    --save-result
    --save-detailed
    --result-dir "${run_dir}"
    --result-filename "benchmark_result.json"
    --disable-tqdm
  )
  if [[ -n "${MAX_CONCURRENCY_CAP}" ]]; then
    if bench_help_has "--max-concurrency"; then
      bench_cmd+=(--max-concurrency "${MAX_CONCURRENCY_CAP}")
    else
      log "WARNING: MAX_CONCURRENCY_CAP is set but --max-concurrency is unavailable"
    fi
  fi
  if bench_help_has "--request-id-prefix"; then
    bench_cmd+=(--request-id-prefix "${run_name}-")
  fi

  {
    printf 'server:\n'
    printf 'VLLM_USE_V1=1 VLLM_SPEC_TRACE=1 VLLM_SPEC_TRACE_FILE=%q VLLM_SPEC_TTFT_TRACE=1 VLLM_SPEC_TTFT_TRACE_FILE=%q ' "${trace_path}" "${trace_path}"
    quote_cmd "${server_cmd[@]}"
    printf '\nbenchmark:\n'
    quote_cmd "${bench_cmd[@]}"
  } >"${run_dir}/command.txt"

  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    die "port ${PORT} already has a healthy vLLM server; stop it or set PORT"
  fi

  log "starting ${run_name}"
  env \
    VLLM_USE_V1=1 \
    VLLM_SPEC_TRACE=1 \
    VLLM_SPEC_TRACE_FILE="${trace_path}" \
    VLLM_SPEC_TTFT_TRACE=1 \
    VLLM_SPEC_TTFT_TRACE_FILE="${trace_path}" \
    "${server_cmd[@]}" >"${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"

  "${bench_cmd[@]}" >"${benchmark_log}" 2>&1
  stop_server
  run_analyzers "${phase}" "${run_dir}" "${trace_path}"
  log "finished ${run_name}"
}

aggregate_outputs() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_phase_profile_results.py" "${OUT_ROOT}"
  if [[ -f "${OUT_ROOT}/phase7_summary.csv" \
      && -f "${SCRIPT_DIR}/compare_phase7_spec_results.py" ]]; then
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_phase7_spec_results.py" \
      "${OUT_ROOT}/phase7_summary.csv" --out-dir "${OUT_ROOT}/phase7" \
      >"${OUT_ROOT}/phase7_compare.log" 2>&1; then
      log "WARNING: Phase 7 comparison failed; see ${OUT_ROOT}/phase7_compare.log"
    fi
  fi
  if [[ "${RUN_PLOTS}" == "1" ]]; then
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_phase_profile_results.py" \
      "${OUT_ROOT}" >"${OUT_ROOT}/plot.log" 2>&1; then
      log "WARNING: plotting failed; see ${OUT_ROOT}/plot.log"
    fi
  fi
}

run_phase0() {
  run_one phase0 smoke 0 128 16 1 2048 default "${NUM_PROMPTS_PHASE0}" "${MAX_NUM_SEQS}"
}

run_phase1() {
  for input_len in 128 512 1024 2048; do
    for batch_size in 1 2 4 8 16 32; do
      run_one phase1 prefill_like 0 "${input_len}" 1 "${batch_size}" \
        "${DEFAULT_TOKEN_BUDGET}" off "${NUM_PROMPTS_SMALL}" "${MAX_NUM_SEQS}"
    done
  done
  for input_len in 128 1024; do
    for concurrency in 1 8 16 32 64 128 256; do
      run_one phase1 decode_like 0 "${input_len}" 128 "${concurrency}" \
        "${DEFAULT_TOKEN_BUDGET}" default "${NUM_PROMPTS_LARGE}" "${MAX_NUM_SEQS}"
    done
  done
}

run_phase2() {
  for chunked in on off; do
    for concurrency in 1 8 32 64 128; do
      run_one phase2 short_prompt_baseline 0 128 128 "${concurrency}" \
        8192 "${chunked}" "${NUM_PROMPTS_LARGE}" 256
      run_one phase2 long_prompt_stress 0 1024 256 "${concurrency}" \
        2048 "${chunked}" "${NUM_PROMPTS_LARGE}" 256
    done
  done
}

run_phase3() {
  for concurrency in 32 64; do
    for token_budget in 2048 4096 8192 16384; do
      for chunked in on off; do
        run_one phase3 token_budget 0 1024 256 "${concurrency}" \
          "${token_budget}" "${chunked}" "${NUM_PROMPTS_LARGE}" 256
      done
    done
  done
}

run_phase4() {
  local concurrencies=(32)
  if [[ "${INCLUDE_PHASE4_OPTIONAL}" == "1" ]]; then
    concurrencies+=(64)
  fi
  for concurrency in "${concurrencies[@]}"; do
    for input_len in 128 512 1024 2048; do
      for token_budget in 2048 8192; do
        run_one phase4 input_length 0 "${input_len}" 256 "${concurrency}" \
          "${token_budget}" on "${NUM_PROMPTS_LARGE}" 256
      done
    done
  done
}

run_phase5() {
  for output_len in 16 32 64 128 256; do
    for token_budget in 2048 8192; do
      run_one phase5 vanilla_output_length 0 1024 "${output_len}" 32 \
        "${token_budget}" on "${NUM_PROMPTS_LARGE}" 256
    done
  done
  for k in 0 1 2 3; do
    for output_len in 16 64 256; do
      for token_budget in 2048 8192; do
        run_one phase5 speculative_k "${k}" 1024 "${output_len}" 32 \
          "${token_budget}" on "${NUM_PROMPTS_PHASE5_SPEC}" 256
      done
    done
  done
}

run_phase6() {
  local request_rates=()
  while IFS= read -r request_rate; do
    [[ -n "${request_rate}" ]] && request_rates+=("${request_rate}")
  done < <(phase6_request_rates)

  for request_rate in "${request_rates[@]}"; do
    run_one_request_rate phase6 short_prompt_open_loop 0 128 128 \
      "${request_rate}" 8192 on "${NUM_PROMPTS_PHASE6}" 256
    for token_budget in 2048 8192; do
      run_one_request_rate phase6 long_prompt_open_loop 0 1024 256 \
        "${request_rate}" "${token_budget}" on "${NUM_PROMPTS_PHASE6}" 256
    done
  done
}

run_phase7() {
  for k in 0 1 2 3 4; do
    for concurrency in 1 32; do
      run_one phase7 short_easy "${k}" 128 128 "${concurrency}" \
        8192 on "${NUM_PROMPTS_PHASE7_SHORT_EASY}" 256
    done
    for concurrency in 64 128; do
      run_one phase7 short_high_concurrency "${k}" 128 128 \
        "${concurrency}" 8192 on "${NUM_PROMPTS_PHASE7}" 256
    done
    for concurrency in 32 64; do
      run_one phase7 long_tight "${k}" 1024 256 "${concurrency}" \
        2048 on "${NUM_PROMPTS_PHASE7}" 256
      run_one phase7 long_relaxed "${k}" 1024 256 "${concurrency}" \
        8192 on "${NUM_PROMPTS_PHASE7}" 256
    done
  done

  for k in 0 1 2 3; do
    for output_len in 16 64 256; do
      for token_budget in 2048 8192; do
        run_one phase7 output_length_amortization "${k}" 1024 \
          "${output_len}" 32 "${token_budget}" on \
          "${NUM_PROMPTS_PHASE7_AMORT}" 256
      done
    done
  done
}

run_selected_phase() {
  local phase="$1"
  case "${phase}" in
    phase0) run_phase0 ;;
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    phase3) run_phase3 ;;
    phase4) run_phase4 ;;
    phase5) run_phase5 ;;
    phase6) run_phase6 ;;
    phase7) run_phase7 ;;
    all)
      run_phase0
      run_phase1
      run_phase2
      run_phase3
      run_phase4
      run_phase5
      run_phase6
      run_phase7
      ;;
    *) die "PHASE must be one of phase0, phase1, phase2, phase3, phase4, phase5, phase6, phase7, all" ;;
  esac
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "${OUT_ROOT}"
  activate_env
  write_env_report
  validate_cuda_ready
  capture_help
  log "outputs: ${OUT_ROOT}"
  run_selected_phase "${PHASE}"
  aggregate_outputs
  log "done. outputs: ${OUT_ROOT}"
}

main "$@"
