#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  printf '[turboquant-e2e] %s\n' "$*" >&2
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

: "${CONCURRENCY_VALUES:=1 2 4 8 16}"
: "${INPUT_LEN:=1024}"
: "${OUTPUT_LEN:=256}"
: "${NUM_PROMPTS:=128}"
: "${REQUEST_RATE:=inf}"
: "${REPEAT:=3}"
: "${HOST:=127.0.0.1}"
: "${BASE_PORT:=8000}"
: "${DRY_RUN:=0}"
: "${VLLM_BIN:=vllm}"
: "${SKIP_VANILLA:=0}"
: "${SKIP_TURBOQUANT:=0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN=$(command -v python || true)
  fi
fi

[[ "${SKIP_VANILLA}" == "1" || -n "${VANILLA_MODEL:-}" ]] || die "VANILLA_MODEL must be set (or set SKIP_VANILLA=1)."
[[ "${SKIP_TURBOQUANT}" == "1" || -n "${TURBOQUANT_MODEL:-}" ]] || die "TURBOQUANT_MODEL must be set (or set SKIP_TURBOQUANT=1)."

VANILLA_ENGINE_ARGS=${VANILLA_ENGINE_ARGS:-}
TURBOQUANT_ENGINE_ARGS=${TURBOQUANT_ENGINE_ARGS:-}

read -r -a CONCURRENCY_ARRAY <<< "${CONCURRENCY_VALUES}"
[[ ${#CONCURRENCY_ARRAY[@]} -gt 0 ]] || die "CONCURRENCY_VALUES is empty."

for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
  is_positive_int "${concurrency}" \
    || die "CONCURRENCY_VALUES must contain positive integers; got '${concurrency}'."
done

is_positive_int "${INPUT_LEN}" || die "INPUT_LEN must be a positive integer."
is_positive_int "${OUTPUT_LEN}" || die "OUTPUT_LEN must be a positive integer."
is_positive_int "${NUM_PROMPTS}" || die "NUM_PROMPTS must be a positive integer."
is_positive_int "${REPEAT}" || die "REPEAT must be a positive integer."
is_nonnegative_int "${BASE_PORT}" || die "BASE_PORT must be a nonnegative integer."

if [[ -z "${RESULTS_DIR:-}" ]]; then
  timestamp=$(date -u '+%Y%m%dT%H%M%SZ')
  RESULTS_DIR="${SCRIPT_DIR}/results/${timestamp}"
fi

mkdir -p "${RESULTS_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || die \
    "Python executable not found at '${PYTHON_BIN:-<unset>}'. Activate a Python env (conda/uv) or set PYTHON_BIN."
fi

CONFIG_FILE="${RESULTS_DIR}/run_config.env"
{
  printf 'VANILLA_MODEL=%q\n' "${VANILLA_MODEL}"
  printf 'TURBOQUANT_MODEL=%q\n' "${TURBOQUANT_MODEL}"
  printf 'VANILLA_ENGINE_ARGS=%q\n' "${VANILLA_ENGINE_ARGS}"
  printf 'TURBOQUANT_ENGINE_ARGS=%q\n' "${TURBOQUANT_ENGINE_ARGS}"
  printf 'CONCURRENCY_VALUES=%q\n' "${CONCURRENCY_VALUES}"
  printf 'INPUT_LEN=%q\n' "${INPUT_LEN}"
  printf 'OUTPUT_LEN=%q\n' "${OUTPUT_LEN}"
  printf 'NUM_PROMPTS=%q\n' "${NUM_PROMPTS}"
  printf 'REQUEST_RATE=%q\n' "${REQUEST_RATE}"
  printf 'REPEAT=%q\n' "${REPEAT}"
  printf 'HOST=%q\n' "${HOST}"
  printf 'BASE_PORT=%q\n' "${BASE_PORT}"
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'VLLM_BIN=%q\n' "${VLLM_BIN}"
  printf 'PYTHON_BIN=%q\n' "${PYTHON_BIN}"
} > "${CONFIG_FILE}"

log "Results directory: ${RESULTS_DIR}"
log "Config snapshot: ${CONFIG_FILE}"

failures=0
condition_index=0
RUN_ONE="${SCRIPT_DIR}/run_one.sh"

VARIANTS_TO_RUN=()
[[ "${SKIP_VANILLA}" != "1" ]] && VARIANTS_TO_RUN+=("vanilla")
[[ "${SKIP_TURBOQUANT}" != "1" ]] && VARIANTS_TO_RUN+=("turboquant")
[[ ${#VARIANTS_TO_RUN[@]} -gt 0 ]] || die "Both SKIP_VANILLA and SKIP_TURBOQUANT are set; nothing to run."

for variant in "${VARIANTS_TO_RUN[@]}"; do
  if [[ "${variant}" == "vanilla" ]]; then
    model="${VANILLA_MODEL}"
    engine_args="${VANILLA_ENGINE_ARGS}"
  else
    model="${TURBOQUANT_MODEL}"
    engine_args="${TURBOQUANT_ENGINE_ARGS}"
  fi

  for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
    for ((repeat_index = 1; repeat_index <= REPEAT; repeat_index++)); do
      port=$((BASE_PORT + condition_index))
      condition_index=$((condition_index + 1))

      log "variant=${variant} concurrency=${concurrency} repeat=${repeat_index}/${REPEAT} port=${port}"

      if ! DRY_RUN="${DRY_RUN}" \
        VLLM_BIN="${VLLM_BIN}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        bash "${RUN_ONE}" \
          --variant "${variant}" \
          --model "${model}" \
          --engine-args "${engine_args}" \
          --concurrency "${concurrency}" \
          --repeat "${repeat_index}" \
          --input-len "${INPUT_LEN}" \
          --output-len "${OUTPUT_LEN}" \
          --num-prompts "${NUM_PROMPTS}" \
          --request-rate "${REQUEST_RATE}" \
          --host "${HOST}" \
          --port "${port}" \
          --results-dir "${RESULTS_DIR}"; then
        failures=$((failures + 1))
        log "Condition failed: variant=${variant} concurrency=${concurrency} repeat=${repeat_index}"
      fi
    done
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  log "Dry run complete. No servers or benchmarks were started."
  exit 0
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize.py" --results-dir "${RESULTS_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot.py" \
  --summary-csv "${RESULTS_DIR}/summary.csv" \
  --out-dir "${RESULTS_DIR}/plots"

log "Summary: ${RESULTS_DIR}/summary.md"

if [[ "${failures}" -gt 0 ]]; then
  log "${failures} condition(s) failed; see summary.md and raw metadata for details."
  exit 1
fi
