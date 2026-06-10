# E2: Decode-Only Bubble Room

Purpose: show whether decode-only execution leaves substantial idle compute and
how often serving iterations are decode-only as load rises.

## Part A: Offline Decode Regime Map

Default runnable grid:

- Model: `Qwen/Qwen3-8B`, TP1.
- Decode batch sizes: `1,2,4,8,16`.
- Context lengths: `2048,8192`.
- Analytic MFU uses `--n-params 8e9` by default.
- The worker prefills all prompts, then generates enough fixed-length output to
  skip warmup decode steps and measure at least 200 decode-only scheduler steps.

The original larger Qwen3-32B TP4 grid is still available explicitly:

```bash
conda run -n vllm python -m exp.e2.offline_decode_map \
  --model Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --n-params 32e9 \
  --batch-sizes 1,2,4,8,16,32,64,128,256 \
  --context-lens 2048,8192,32768 \
  --nsys-points 8:8192,64:8192
```

Run example:

```bash
conda run -n vllm python -m exp.e2.offline_decode_map \
  --peak-flops-per-gpu-tflops <per_gpu_peak> \
  --hbm-bw-per-gpu-gbps <per_gpu_hbm_bw> \
  --run-nsys
```

Nsight Systems profiling is only run when `--run-nsys` is set, defaulting to
`B=8,L=8192` and `B=16,L=8192`.

## Part B: Serving Decode-Only Fraction

The serving sweep launches `vllm serve` with `Qwen/Qwen3-8B` TP1 by default,
runs a quick saturation probe unless `--lambda-sat` is provided, then sweeps
`rho in {0.2,0.4,0.6,0.8,0.95}` using `vllm bench serve`.

```bash
conda run -n vllm python -m exp.e2.serving_sweep \
  --sharegpt-path /path/to/ShareGPT.json
```

Outputs land in `results/e2/<run_id>/`:

- `part_a_offline.csv`
- `part_b_saturation_probe.csv`
- `part_b_serving.csv`
- `plots/part_a_mfu_vs_B.png`
- `plots/part_a_bw_vs_B.png`
- `plots/part_b_decode_only_fraction_vs_rho.png`
- `plots/part_b_slack_vs_rho.png`
- `plots/bubble_verdict.json`
