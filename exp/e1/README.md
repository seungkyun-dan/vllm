# E1: Draft Prefill TTFT Inflation

Purpose: quantify whether synchronous draft prefill materially inflates TTFT
across input length and target/draft TP asymmetry.

Default runnable matrix:

- Qwen/Qwen3-8B TP1 + Qwen/Qwen3-0.6B TP1 draft.
- Input lengths: 1K, 4K, 16K, 64K. Qwen 64K is skipped unless
  `--include-qwen-64k` is set.
- Arms: target-only, integrated draft-model spec for same TP, and draft-only
  prefill decomposition.

The original full-scale pairs are still available explicitly as `qwen32` and
`llama70`:

```bash
conda run -n vllm python -m exp.e1.run \
  --pairs qwen32,llama70 \
  --tp-target 4 \
  --tp-drafts 4,1
```

In this vLLM build, integrated `draft_model` requires draft TP to equal target
TP, so `--tp-drafts 1` on a TP4 target is used only by the decomposition arm.

Run:

```bash
conda run -n vllm python -m exp.common.capability_probe
conda run -n vllm python -m exp.e1.run
```

Outputs land in `results/e1/<run_id>/`: raw request JSONL, full `config.json`,
aggregate `parsed.csv`, `plots/inflation.png`, and `plots/decision.json`.
