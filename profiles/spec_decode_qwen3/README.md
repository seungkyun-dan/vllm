# Qwen3 Draft-Model Speculative Decoding Sweep

Experimental sweep workspace for Qwen/Qwen3-8B target serving with
Qwen/Qwen3-0.6B as the draft model. This is an online serving
benchmark: it launches `vllm serve` and drives it with `vllm bench serve`.
Offline one-batch profiling is useful for microbenchmarks, but it does not
exercise request arrival rate, queueing, scheduler pressure, or effective
batch-size changes under load.

## Sweep Axes

Default axes:

- Speculation length: `K_VALUES="0 1 2 3 4 6 8"`.
- Serving load: `LOAD_VALUES="1 2 4 8 16 32 64 128"`.
- Load interpretation: `LOAD_MODE=concurrency`, so load is passed to
  `vllm bench serve --max-concurrency` with `--request-rate inf`.

Alternative load axis:

- `LOAD_MODE=request_rate` treats `LOAD_VALUES` as offered request rate and
  passes each value to `--request-rate`.

Held fixed by default: target model, draft model, prompt length, output length,
number of prompts, scheduler limits, tensor parallelism, dtype, and GPU memory
utilization. Override them with environment variables when you want a second
controlled experiment.

## Python Environment

The scripts choose Python in this order:

1. `PYTHON_BIN=/path/to/python` if you set it explicitly.
2. `${REPO_ROOT}/.venv/bin/python` if that file exists.
3. `conda run -n ${CONDA_ENV:-vllm} python -c ...` to resolve the env Python,
   then that Python executable as the default fallback.

So this checkout can use your existing `vllm` conda env without creating a
repo `.venv`. Set `CONDA_ENV=other_name` if needed.

## Quick Smoke Test

From the repo root:

```bash
K_VALUES="0 4" LOAD_VALUES="1 2" NUM_PROMPTS=16 OUTPUT_LEN=32 \
  profiles/spec_decode_qwen3/run_sweep.sh
```

## Full Default Sweep

```bash
profiles/spec_decode_qwen3/run_sweep.sh
```

Defaults:

- `TARGET_MODEL=Qwen/Qwen3-8B`
- `DRAFT_MODEL=Qwen/Qwen3-0.6B`
- `K_VALUES="0 1 2 3 4 6 8"`
- `LOAD_MODE=concurrency`
- `LOAD_VALUES="1 2 4 8 16 32 64 128"`
- `INPUT_LEN=128`
- `OUTPUT_LEN=128`
- `NUM_PROMPTS=256`

`K=0` launches the target without speculative decoding. Other K values launch
with `--speculative-config` using draft-model speculation.

## Common Overrides

```bash
# Sweep request rates instead of max concurrency.
LOAD_MODE=request_rate LOAD_VALUES="1 2 4 8 16 32" \
  profiles/spec_decode_qwen3/run_sweep.sh

# Use a different GPU or a smaller server budget.
CUDA_VISIBLE_DEVICES=1 MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=4096 \
  profiles/spec_decode_qwen3/run_sweep.sh

# Keep detailed per-request benchmark data.
SAVE_DETAILED=1 profiles/spec_decode_qwen3/run_sweep.sh
```

## Output Layout

Each run writes a timestamped directory under
`profiles/spec_decode_qwen3/results/`:

- `run_config.env` and `run_config.json`: full run-level configuration.
- `bench_results/*.json`: `vllm bench serve` result JSON for each K/load point;
  each file also includes server, dataset, load, speculation, artifact-path,
  and environment metadata via `--metadata`.
- `bench_logs/*.log`: raw benchmark stdout/stderr.
- `server_logs/*.log`: server stdout/stderr for each K.
- `metrics/*.prom`: `/metrics` snapshots after each benchmark point.
- `gpu/*.csv`: `nvidia-smi` snapshots after each benchmark point, if available.
- `summary.csv`: normalized summary across all JSON result files.
- `summary.jsonl`: one normalized JSON object per benchmark point.

The summary includes latency percentiles, throughput, goodput when configured,
and existing speculative decoding acceptance metrics reported by `vllm bench
serve`.

