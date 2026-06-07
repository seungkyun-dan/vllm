#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONDA_ENV=${CONDA_ENV:-"vllm"}
PYTHON_BIN=${PYTHON_BIN:-""}
PYTHON_CMD=()

if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  PYTHON_CMD=("${PYTHON_BIN}")
else
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

RESULT_DIR=${1:-"${SCRIPT_DIR}/results/latest"}

if [[ ! -f "${RESULT_DIR}/summary.csv" ]]; then
  "${SCRIPT_DIR}/summarize_results.sh" "${RESULT_DIR}"
fi

"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/plot_results.py" "${RESULT_DIR}"
