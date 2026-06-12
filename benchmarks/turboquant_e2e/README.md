# TurboQuant E2E Serving Benchmark

This harness compares two vLLM serving variants end to end:

- `vanilla`: `VANILLA_MODEL` plus optional `VANILLA_ENGINE_ARGS`
- `turboquant`: `TURBOQUANT_MODEL` plus optional `TURBOQUANT_ENGINE_ARGS`

TurboQuant is treated as the same benchmark flow with a different served model
path and/or extra engine flags. This supports a TurboQuant model path, a
TurboQuant-enabled vLLM fork/plugin, or feature flags passed to `vllm serve`.

The benchmark uses `vllm serve` and `vllm bench serve` against the
OpenAI-compatible endpoint. It measures E2E serving behavior, so it can support
or reject the claim that TurboQuant is slower for a workload. It cannot by
itself prove that dequantization is the cause. Use Nsight Systems/Compute or
other kernel profiling as a follow-up if the E2E results show a slowdown.

## Setup

Install the Python packages used for summary and plotting in the repository
virtual environment:

```bash
uv pip install -r benchmarks/turboquant_e2e/requirements.txt
```

The wrapper defaults to `.venv/bin/python` for Python helper scripts. Set
`PYTHON_BIN=/path/to/python` if your vLLM environment uses a different Python
inside a managed virtual environment.

## Usage

```bash
export VANILLA_MODEL=/path/to/vanilla
export TURBOQUANT_MODEL=/path/to/turboquant
export TURBOQUANT_ENGINE_ARGS="..."
export CONCURRENCY_VALUES="1 2 4 8 16"
export INPUT_LEN=1024
export OUTPUT_LEN=256
export NUM_PROMPTS=128
export REPEAT=3

bash benchmarks/turboquant_e2e/run_all.sh
```

Optional variables:

- `VANILLA_ENGINE_ARGS`: extra args passed to `vllm serve` for vanilla
- `TURBOQUANT_ENGINE_ARGS`: extra args passed to `vllm serve` for TurboQuant
- `REQUEST_RATE`: request rate for `vllm bench serve`, default `inf`
- `HOST`: server bind host, default `127.0.0.1`
- `BASE_PORT`: first port to use, default `8000`
- `RESULTS_DIR`: output directory; defaults to a timestamped directory under
  `benchmarks/turboquant_e2e/results/`
- `VLLM_BIN`: vLLM CLI executable, default `vllm`
- `PYTHON_BIN`: Python executable for summary/plot helpers, default
  `.venv/bin/python`
- `NUM_WARMUPS`: warmup requests passed to `vllm bench serve` when supported,
  default `1`
- `IGNORE_EOS`: pass `--ignore-eos` when supported, default `1`
- `DRY_RUN=1`: print the planned runs without starting servers

## Workload Examples

Decode-heavy:

```bash
export INPUT_LEN=1024
export OUTPUT_LEN=512
bash benchmarks/turboquant_e2e/run_all.sh
```

Prefill-heavy:

```bash
export INPUT_LEN=8192
export OUTPUT_LEN=16
bash benchmarks/turboquant_e2e/run_all.sh
```

## Outputs

Each run writes raw files under `RESULTS_DIR`:

- `raw/*.json`: raw `vllm bench serve` JSON results
- `raw/*.metadata.json`: metadata, commands, environment, and log paths
- `raw_results.csv`: flattened per-run metrics and metadata
- `summary.csv`: mean/std/min/max by variant and concurrency
- `summary.md`: environment, config, comparison table, warnings, and conclusion
- `plots/*.png`: simple vanilla vs TurboQuant plots

`run_one.sh` checks `vllm bench serve --help` before running and fails with a
readable error if required options are missing. It prefers `--save-result-json`
when available, then `--save-result --result-filename`, then `--output-json`.

## Interpretation

Lower output throughput is worse. Higher TPOT is worse. Higher TTFT is worse.

The Markdown summary reports ratios:

- `turboquant_output_throughput / vanilla_output_throughput`
- `turboquant_mean_tpot_ms / vanilla_mean_tpot_ms`
- `turboquant_mean_ttft_ms / vanilla_mean_ttft_ms`

For concurrency `4`, the summary prints a PASS/FAIL-style conclusion. If
TurboQuant output throughput is lower than vanilla, or TurboQuant mean TPOT is
higher than vanilla, the conclusion is:

```text
Conclusion: TurboQuant is slower than vanilla at concurrency=4 for this workload.
```

If TurboQuant is worse at concurrency `4` consistently across repeats, the E2E
claim is supported. To prove dequantization overhead, follow up with kernel
profiling.
