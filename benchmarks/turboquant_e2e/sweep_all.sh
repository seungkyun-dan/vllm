#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-shot sweep over multiple workloads × {vanilla, all turboquant variants}.
# Activate the `vllm` conda env first, then run:
#
#   bash benchmarks/turboquant_e2e/sweep_all.sh
#
# Optional env overrides:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct
#   WORKLOADS="batch_baseline long_single"   # subset, space-separated
#   VARIANTS="turboquant_k8v4 turboquant_4bit_nc"   # subset
#   OUT_ROOT=/path/to/results
#   DRY_RUN=1     # print plan, don't run
#   SERVER_READY_TIMEOUT=1200

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL:=Qwen/Qwen3-8B}"
: "${OUT_ROOT:=${SCRIPT_DIR}/results}"
: "${DRY_RUN:=0}"
: "${SERVER_READY_TIMEOUT:=900}"
: "${VLLM_BIN:=vllm}"

# Auto-pick python from active conda env (or PATH).
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN=$(command -v python || true)
  fi
fi
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] \
  || { echo "ERROR: no python found. Activate the vllm conda env." >&2; exit 1; }
command -v "${VLLM_BIN}" >/dev/null 2>&1 \
  || { echo "ERROR: '${VLLM_BIN}' CLI not found on PATH." >&2; exit 1; }
export PYTHON_BIN VLLM_BIN SERVER_READY_TIMEOUT

# Workload matrix: name | input_len | output_len | concurrencies | num_prompts | repeat
ALL_WORKLOADS=(
  "batch_baseline  | 1024  | 256  | 1 2 4 8 16    | 64  | 3"
  "batch_large     | 1024  | 256  | 32 64 128 256 | 256 | 2"
  "batch_capacity  | 1024  | 256  | 256 512 1024  | 512 | 2"
  "long_single     | 16384 | 1024 | 1             | 8   | 2"
  "vlong_single    | 32768 | 1024 | 1             | 4   | 2"
  "xlong_single    | 65536 | 1024 | 1             | 4   | 2"
  "long_batch      | 4096  | 512  | 1 4 16 32     | 64  | 2"
  "xlong_batch     | 8192  | 512  | 1 4 16 32     | 32  | 2"
)

# Filter workloads via WORKLOADS env var (space-separated list of names).
SELECTED=()
if [[ -n "${WORKLOADS:-}" ]]; then
  for wl in ${WORKLOADS}; do
    found=0
    for row in "${ALL_WORKLOADS[@]}"; do
      name=$(awk -F'|' '{gsub(/[[:space:]]/,"",$1); print $1}' <<< "$row")
      if [[ "$name" == "$wl" ]]; then
        SELECTED+=("$row"); found=1; break
      fi
    done
    [[ "$found" == 1 ]] || echo "WARN: unknown workload '${wl}', skipping." >&2
  done
else
  SELECTED=("${ALL_WORKLOADS[@]}")
fi
[[ ${#SELECTED[@]} -gt 0 ]] || { echo "ERROR: no workloads selected." >&2; exit 1; }

ALL_VARIANTS=(turboquant_k8v4 turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc)
if [[ -n "${VARIANTS:-}" ]]; then
  read -ra VARIANT_ARRAY <<< "${VARIANTS}"
else
  VARIANT_ARRAY=("${ALL_VARIANTS[@]}")
fi

RUN_TS=$(date -u '+%Y%m%dT%H%M%SZ')
mkdir -p "${OUT_ROOT}"
INDEX_FILE="${OUT_ROOT}/sweep_${RUN_TS}.index.txt"

# head closing the pipe early triggers pipefail; capture into vars instead.
VLLM_VER=$("${VLLM_BIN}" --version 2>/dev/null || true)
VLLM_VER=$(printf '%s\n' "${VLLM_VER}" | head -n1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
GPU_NAME=$(printf '%s\n' "${GPU_NAME}" | head -n1)

{
  echo "TurboQuant sweep ${RUN_TS}"
  echo "model=${MODEL}"
  echo "python=${PYTHON_BIN}"
  echo "vllm=${VLLM_BIN} (${VLLM_VER:-unknown})"
  echo "gpu=${GPU_NAME:-unknown}"
  echo "workloads=${#SELECTED[@]} variants=${#VARIANT_ARRAY[@]}"
  echo ""
  echo "format: workload | variant | path"
} | tee "${INDEX_FILE}"

trim() { echo "$1" | awk '{$1=$1;print}'; }

run_workload() {
  local row="$1"
  local name input output concs nprompts repeat
  IFS='|' read -r name input output concs nprompts repeat <<< "$row"
  name=$(trim "$name"); input=$(trim "$input"); output=$(trim "$output")
  concs=$(trim "$concs"); nprompts=$(trim "$nprompts"); repeat=$(trim "$repeat")

  local max_model_len=$((input + output + 1024))

  echo ""
  echo "============================================================"
  printf "Workload: %-15s in=%-6s out=%-5s c=[%s] np=%s rep=%s\n" \
    "$name" "$input" "$output" "$concs" "$nprompts" "$repeat"
  echo "============================================================"

  export VANILLA_MODEL="${MODEL}" TURBOQUANT_MODEL="${MODEL}"
  export INPUT_LEN="${input}" OUTPUT_LEN="${output}"
  export CONCURRENCY_VALUES="${concs}" NUM_PROMPTS="${nprompts}" REPEAT="${repeat}"
  export VANILLA_ENGINE_ARGS="--dtype float16 --max-model-len ${max_model_len}"

  # 1) vanilla
  local TS VAN_DIR
  TS=$(date -u '+%Y%m%dT%H%M%SZ')
  VAN_DIR="${OUT_ROOT}/${TS}_${name}_vanilla"
  export RESULTS_DIR="${VAN_DIR}"
  export TURBOQUANT_ENGINE_ARGS="--dtype float16 --max-model-len ${max_model_len} --kv-cache-dtype turboquant_k8v4"
  unset SKIP_VANILLA
  export SKIP_TURBOQUANT=1
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] vanilla -> ${VAN_DIR}"
  else
    bash "${SCRIPT_DIR}/run_all.sh" || echo "WARN: vanilla had failures for ${name}"
  fi
  unset SKIP_TURBOQUANT
  printf "%-15s | %-22s | %s\n" "${name}" "vanilla" "${VAN_DIR}" >> "${INDEX_FILE}"

  # 2) variants (reuse vanilla)
  local variant_args=()
  export SKIP_VANILLA=1
  for VAR in "${VARIANT_ARRAY[@]}"; do
    TS=$(date -u '+%Y%m%dT%H%M%SZ')
    local DIR="${OUT_ROOT}/${TS}_${name}_${VAR}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${DIR}/raw" "${DIR}/logs"
      \cp -f "${VAN_DIR}"/raw/vanilla_*.json          "${DIR}/raw/" 2>/dev/null || true
      \cp -f "${VAN_DIR}"/raw/vanilla_*.metadata.json "${DIR}/raw/" 2>/dev/null || true
      \cp -f "${VAN_DIR}"/logs/vanilla_*              "${DIR}/logs/" 2>/dev/null || true
    fi
    export RESULTS_DIR="${DIR}"
    export TURBOQUANT_ENGINE_ARGS="--dtype float16 --max-model-len ${max_model_len} --kv-cache-dtype ${VAR}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] ${VAR} -> ${DIR}"
    else
      bash "${SCRIPT_DIR}/run_all.sh" || echo "WARN: ${VAR} had failures for ${name}"
    fi
    variant_args+=(--variant "${VAR#turboquant_}=${DIR}")
    printf "%-15s | %-22s | %s\n" "${name}" "${VAR}" "${DIR}" >> "${INDEX_FILE}"
  done
  unset SKIP_VANILLA

  # 3) comparison
  if [[ "${DRY_RUN}" != "1" && ${#variant_args[@]} -gt 0 ]]; then
    TS=$(date -u '+%Y%m%dT%H%M%SZ')
    local CMP="${OUT_ROOT}/comparison_${name}_${TS}"
    if "${PYTHON_BIN}" "${SCRIPT_DIR}/compare.py" \
        --vanilla-dir "${VAN_DIR}" \
        "${variant_args[@]}" \
        --out-dir "${CMP}" >/dev/null 2>&1; then
      printf "%-15s | %-22s | %s\n" "${name}" "comparison" "${CMP}" >> "${INDEX_FILE}"
      echo "  comparison: ${CMP}/comparison.md"
    else
      echo "  WARN: comparison failed for ${name}"
    fi
  fi
}

START_TS=$(date +%s)
for row in "${SELECTED[@]}"; do
  run_workload "$row" || echo "WARN: workload row failed: $row"
done
END_TS=$(date +%s)

echo ""
echo "============================================================"
echo "Sweep complete in $(( (END_TS - START_TS) / 60 ))m $(( (END_TS - START_TS) % 60 ))s."
echo "Index file: ${INDEX_FILE}"
echo "Per-workload comparisons: ${OUT_ROOT}/comparison_*_${RUN_TS%T*}*/"
echo "============================================================"
