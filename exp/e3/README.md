# E3: Target Decode Slowdown Under Concurrent Draft Prefill

Purpose: measure how much Qwen3-0.6B draft prefill slows a steady
Qwen3-8B TP1 decode-only serving batch by default, and compare overlap against
the trivial serial-insertion alternative.

## Rig

- Target A: `vllm serve Qwen/Qwen3-8B`, TP1, prefix caching off.
- Target load: 32 simultaneous streaming clients, `in=8192`, `out=4096`,
  greedy, `ignore_eos=True`.
- Draft B: offline `Qwen/Qwen3-0.6B`, TP1, repeated exact-token prefills of
  `C_d` tokens with `max_tokens=1`, prefix caching off.
- Modes:
  - `m0`: target alone.
  - `m1`: MPS shared, no draft cap.
  - `m2`: draft process `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25`.
  - `m3`: draft process `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=13`, the nearest
    integer MPS proxy for the requested 12.5%.

The overlap runner starts B only after A has entered the decode-only step
window. The analyzer verifies no prefill steps appear after that point.

## MPS Setup

Run this outside Python before the experiment:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
```

Stop MPS after the run:

```bash
echo quit | nvidia-cuda-mps-control
```

## Dry Run

```bash
conda run -n vllm python -m exp.e3.run \
  --cuda-visible-devices 0 \
  --mps-pipe-directory "$CUDA_MPS_PIPE_DIRECTORY" \
  --mps-log-directory "$CUDA_MPS_LOG_DIRECTORY" \
  --dry-run
```

## Full Run

```bash
conda run -n vllm python -m exp.e3.run \
  --cuda-visible-devices 0 \
  --mps-pipe-directory "$CUDA_MPS_PIPE_DIRECTORY" \
  --mps-log-directory "$CUDA_MPS_LOG_DIRECTORY" \
  --run-nsys-m1
```

Original large-target rig:

```bash
conda run -n vllm python -m exp.e3.run \
  --target-model Qwen/Qwen3-32B \
  --draft-model Qwen/Qwen3-1.7B \
  --target-tp 4 \
  --draft-tp 1 \
  --cuda-visible-devices 0,1,2,3 \
  --mps-pipe-directory "$CUDA_MPS_PIPE_DIRECTORY" \
  --mps-log-directory "$CUDA_MPS_LOG_DIRECTORY"
```

Useful smaller shakeout:

```bash
conda run -n vllm python -m exp.e3.run \
  --cuda-visible-devices 0 \
  --mps-pipe-directory "$CUDA_MPS_PIPE_DIRECTORY" \
  --mps-log-directory "$CUDA_MPS_LOG_DIRECTORY" \
  --chunk-tokens 256 \
  --modes m1 \
  --reps 1 \
  --draft-solo-prefills 4
```

## Outputs

The runner writes:

- `results/e3/<run_id>/config.json`
- `results/e3/<run_id>/raw/cell_manifest.jsonl`
- per-cell target step traces and client JSONL under `raw/`
- per-cell draft JSONL and draft summaries under `raw/`
- `parsed.csv`
- `summary.csv`
- `kill_rule.csv`
- `plots/pareto_draft_tps_vs_target_inflation.png`
- `plots/kill_rule.json`

`kill_rule.csv` reports pass/fail per `C_d`: pass means the best overlap mode
has target iteration inflation below 50% of the matched serial insertion
inflation.

Re-run analysis:

```bash
conda run -n vllm python -m exp.e3.analyze \
  --run-dir results/e3/<run_id>
```
