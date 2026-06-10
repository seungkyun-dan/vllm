from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _prompt_tokens(model: str, context_len: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    text = (
        "Decode-only bubble experiment prompt. This deterministic text is "
        "repeated to create exact token-length contexts for every batch item. "
    )
    base_ids = tokenizer.encode(text, add_special_tokens=False)
    if not base_ids:
        fallback = tokenizer.eos_token_id
        base_ids = [0 if fallback is None else int(fallback)]
    repeats = (context_len + len(base_ids) - 1) // len(base_ids)
    return (base_ids * repeats)[:context_len]


def run_point(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    prompt = _prompt_tokens(args.model, args.context_len)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,
        seed=args.seed,
    )
    prompts = [{"prompt_token_ids": prompt} for _ in range(args.batch_size)]
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    out_lens = [len(output.outputs[0].token_ids) for output in outputs]
    return {
        "model": args.model,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "output_len": args.output_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "wall_time_s": end - start,
        "output_lens": out_lens,
        "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_point(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
