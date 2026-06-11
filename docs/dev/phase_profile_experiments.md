# Phase 0-6 Serving Profile Experiments

This suite profiles vLLM serving behavior and scheduler phase composition. It is
not an optimization, does not add a scheduling policy, and should not be used to
claim a performance result before the experiments are run and inspected.

The goal is to diagnose when serving-level TPOT/ITL degradation is associated
with prefill pressure, token budget, input length, output length, or speculative
decoding overhead. Use trace-derived scheduler composition to support any
interpretation; benchmark metrics alone are not causal evidence.

## Environment

Use the existing conda environment named `profile`. The environment may still be
under construction.

```bash
conda activate profile
# If uv is not available in this environment yet:
conda install -c conda-forge uv
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install matplotlib
```

Do not use `.venv` unless explicitly requested. The runner will use `.venv` only
when `USE_VENV=1` is set and the `profile` conda environment is unavailable.

If `torch_cuda` is newer than the installed NVIDIA driver supports, install CUDA
forward-compatibility libraries for the Torch CUDA build and export the vLLM
compatibility variables before running the phases. For example, `torch 2.11.0+cu130`
with a CUDA 12.8 driver needs CUDA 13 compatibility libraries:

```bash
conda activate profile
conda install -c conda-forge cuda-compat=13.0 -y
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="${CONDA_PREFIX}/cuda-compat"
```

The runner writes `<OUT_ROOT>/cuda_preflight.txt` and fails before server launch
when CUDA is not usable. Set `SKIP_CUDA_PREFLIGHT=1` only for diagnostics.

Use `uv pip` inside the already-created `profile` environment for dependency
management. Do not rebuild into `.venv` unless you intentionally decide to move
away from the `profile` environment.

## Phase Definitions

- Phase 0: environment and smoke test
- Phase 1: microbenchmark preparation
- Phase 2: vanilla concurrency sweep
- Phase 3: token budget sweep
- Phase 4: input length sweep
- Phase 5: output length and speculative amortization sweep
- Phase 6: open-loop request-rate sweep

Phase 1 uses serving-level approximations. The prefill-like mode uses short
outputs and varied closed-loop concurrency. The decode-like mode uses longer
outputs and varied closed-loop concurrency. These are not pure kernel-level
prefill/decode measurements.

## Commands

```bash
conda activate profile
PHASE=phase0 bash tools/run_phase_profile.sh
PHASE=phase1 bash tools/run_phase_profile.sh
PHASE=phase2 bash tools/run_phase_profile.sh
PHASE=phase3 bash tools/run_phase_profile.sh
PHASE=phase4 bash tools/run_phase_profile.sh
PHASE=phase5 bash tools/run_phase_profile.sh
PHASE=phase6 REQUEST_RATES="1 2 4 8 16" bash tools/run_phase_profile.sh
PHASE=phase6 CAPACITY_C=8 bash tools/run_phase_profile.sh
```

Recommended first sequence:

```bash
conda activate profile
PHASE=phase0 bash tools/run_phase_profile.sh
PHASE=phase2 bash tools/run_phase_profile.sh
PHASE=phase3 bash tools/run_phase_profile.sh
```

Outputs are written under `/tmp/vllm_phase_profile_runs/<timestamp>/` unless
`OUT_ROOT` is set.

This checkout's `benchmarks/benchmark_serving.py` is a compatibility shim that
exits with a deprecation message. The runner uses the current equivalent,
`vllm bench serve`, or falls back to
`python -m vllm.entrypoints.cli.main bench serve` if the console script is not
on `PATH`. For the default `openai-chat` backend it sets
`--endpoint /v1/chat/completions`.

## Interpretation Guide

If TPOT/ITL increases together with `prefill_token_ratio` and waiting requests,
suspect prefill-pressure-induced decode cadence degradation.

If increasing token budget reduces TPOT/ITL and prefill residence time, suspect
token-budget-induced prefill fragmentation.

If chunked prefill on/off changes TTFT and TPOT in opposite directions, interpret
it as a TTFT/TPOT scheduler trade-off.

If speculative `k` improves TPOT but hurts TTFT, interpret it as decode-cycle
reduction versus draft-proposal overhead.

Do not claim causality from benchmark metrics alone. Use trace-derived scheduler
composition to support interpretation.


## Phase 6

Phase 6 studies open-loop arrival pressure. Unlike the closed-loop concurrency
sweep, request arrival rate is externally controlled with `--request-rate`. This
is better for observing queueing, prefill backlog, and SLO collapse.

Run with explicit request rates:

```bash
conda activate profile
PHASE=phase6 REQUEST_RATES="1 2 4 8 16" bash tools/run_phase_profile.sh
```

Or run from an approximate sustainable capacity `C`:

```bash
conda activate profile
PHASE=phase6 CAPACITY_C=8 bash tools/run_phase_profile.sh
```

When `CAPACITY_C` is set, the runner expands it to `0.25C`, `0.5C`, `0.75C`,
`1.0C`, and `1.25C`. If neither `REQUEST_RATES` nor `CAPACITY_C` is set, the
runner uses placeholder rates `1 2 4 8 16` and prints a warning. Replace those
after measuring capacity.

`MAX_CONCURRENCY_CAP` can be set as an optional safety cap. If supported by the
benchmark CLI, the runner passes it as `--max-concurrency`; this changes pure
open-loop behavior because arrivals may then be bounded by the cap.

Interpret Phase 6 as follows:

- If achieved throughput tracks requested `request_rate` and latency stays
  stable, the system is below capacity.
- If achieved throughput stops increasing while TTFT, TPOT/ITL, or request
  latency grows, the system is near or beyond capacity.
- If TPOT/ITL increases with `prefill_token_ratio` and `num_waiting_reqs`,
  suspect prefill-pressure-induced decode cadence degradation.
- If long prompt + small token budget collapses earlier than short prompt +
  large token budget, token-budget-induced prefill fragmentation is likely.

Limitations specific to Phase 6:

- Open-loop experiments are sensitive to client overhead and timeout settings.
- Request-rate values should be calibrated using measured capacity `C`.
- `vllm bench serve` TPOT/ITL is serving-level observed cadence, not pure decode
  kernel time.
- Trace-derived causal evidence should be used to support benchmark metrics.

## Trace Fields

The trace is enabled by the runner with:

```bash
VLLM_SPEC_TRACE=1
VLLM_SPEC_TRACE_FILE=<per-run trace.jsonl>
```

For compatibility it also sets:

```bash
VLLM_SPEC_TTFT_TRACE=1
VLLM_SPEC_TTFT_TRACE_FILE=<per-run trace.jsonl>
```

The trace emits `scheduled_batch_summary`, `request_arrived`,
`request_first_prefill_scheduled`, `request_last_prefill_scheduled`,
`request_first_decode_scheduled`, `first_output_enqueued`, and
`request_finished` records.

Prefill and decode scheduled-token counts are exact scheduler-derived counts for
the target scheduler work. They are computed from each request's prompt length,
computed-token position before scheduling, and scheduled target token count.

Chunked prefill can be toggled in this repo with `--enable-chunked-prefill` and
`--no-enable-chunked-prefill`. vLLM rejects `chunked_prefill=off` when
`max_num_batched_tokens < max_model_len`; for those runs, the runner lowers the
effective server `max_model_len` to the token budget unless
`STRICT_MAX_MODEL_LEN=1` is set. The configured and effective values are recorded
in each run's `metadata.json`.

## Limitations

`vllm bench serve` TPOT/ITL is serving-level observed cadence, not pure kernel
decode time.

CPU timestamps do not measure CUDA kernel duration precisely.

Mixed batch classification is based on target scheduler metadata and does not
measure CUDA kernels directly.

Phase 6 covers open-loop request-rate sweeps, but those runs remain sensitive
to client overhead, network behavior, and timeout settings.
