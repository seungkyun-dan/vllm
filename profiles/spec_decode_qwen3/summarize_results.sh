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

if [[ "${PYTHON_CMD[0]}" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Missing required command: conda" >&2
    exit 1
  fi
elif [[ ! -x "${PYTHON_CMD[0]}" ]]; then
  echo "Expected executable Python at ${PYTHON_CMD[0]}" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" - "${RESULT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
bench_dir = result_dir / "bench_results"
csv_path = result_dir / "summary.csv"
jsonl_path = result_dir / "summary.jsonl"

fields = [
    "run_id",
    "target_model",
    "draft_model",
    "served_model_name",
    "python_cmd",
    "python_env_kind",
    "conda_env",
    "speculative_enabled",
    "speculative_method",
    "num_speculative_tokens",
    "speculative_config_json",
    "load_mode",
    "load_value",
    "load_values",
    "k_values",
    "request_rate",
    "max_concurrency",
    "tensor_parallel_size",
    "draft_tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "dtype",
    "enforce_eager",
    "trust_remote_code",
    "dataset_name",
    "input_len",
    "output_len",
    "num_prompts",
    "random_range_ratio",
    "temperature",
    "seed",
    "cuda_visible_devices",
    "completed",
    "failed",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p50_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
    "spec_decode_num_drafts",
    "spec_decode_draft_tokens",
    "spec_decode_accepted_tokens",
    "spec_decode_per_position_acceptance_rates",
    "duration",
    "result_file",
    "server_log",
    "bench_log",
    "metrics_file",
    "gpu_file",
    "server_command_file",
    "bench_command_file",
]


def as_number(value):
    if value in (None, ""):
        return value
    try:
        if isinstance(value, str) and "." not in value:
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return value


rows = []
for path in sorted(bench_dir.glob("*.json")):
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"Skipping invalid JSON {path}: {exc}", file=sys.stderr)
        continue

    row = {field: data.get(field) for field in fields}
    numeric_fields = (
        "speculative_enabled",
        "num_speculative_tokens",
        "load_value",
        "tensor_parallel_size",
        "draft_tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enforce_eager",
        "trust_remote_code",
        "input_len",
        "output_len",
        "num_prompts",
        "random_range_ratio",
        "temperature",
        "seed",
    )
    for field in numeric_fields:
        row[field] = as_number(data.get(field))
    row["load_mode"] = data.get("load_mode")
    row["result_file"] = str(path.relative_to(result_dir))

    rates = row.get("spec_decode_per_position_acceptance_rates")
    if isinstance(rates, list):
        row["spec_decode_per_position_acceptance_rates"] = json.dumps(rates)

    rows.append(row)

rows.sort(
    key=lambda r: (
        r.get("num_speculative_tokens") if r.get("num_speculative_tokens") != "" else -1,
        r.get("load_value") if r.get("load_value") != "" else -1,
        r.get("result_file") or "",
    )
)

result_dir.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with jsonl_path.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")

print(f"Wrote {len(rows)} rows to {csv_path}")
print(f"Wrote {len(rows)} rows to {jsonl_path}")
PY
