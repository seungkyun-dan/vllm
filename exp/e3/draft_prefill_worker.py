from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from exp.e3.utils import prompt_hash, token_prompt


def run_prefills(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    prompt = token_prompt(
        model=args.model,
        length=args.chunk_tokens,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
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
        max_tokens=1,
        min_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        seed=args.seed,
    )
    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    args.records_out.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    deadline = None
    if args.duration_s is not None:
        deadline = start_all + args.duration_s
    iteration = 0
    while True:
        if args.num_prefills is not None and iteration >= args.num_prefills:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        start = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": prompt}], sampling_params)
        end = time.perf_counter()
        out_len = len(outputs[0].outputs[0].token_ids) if outputs else 0
        elapsed = end - start
        record = {
            "iteration": iteration,
            "start_perf": start,
            "end_perf": end,
            "elapsed_s": elapsed,
            "chunk_tokens": args.chunk_tokens,
            "prefill_tok_s": args.chunk_tokens / elapsed if elapsed > 0 else None,
            "out_len": out_len,
        }
        records.append(record)
        with args.records_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        iteration += 1
    end_all = time.perf_counter()
    total_elapsed = end_all - start_all
    total_tokens = args.chunk_tokens * len(records)
    summary = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "chunk_tokens": args.chunk_tokens,
        "num_prefills": len(records),
        "duration_s": total_elapsed,
        "draft_tps": total_tokens / total_elapsed if total_elapsed > 0 else None,
        "prompt_sha256": prompt_hash(prompt),
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--chunk-tokens", type=int, required=True)
    parser.add_argument("--num-prefills", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--records-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()
    if args.max_model_len is None:
        args.max_model_len = args.chunk_tokens + 16
    if args.num_prefills is None and args.duration_s is None:
        args.num_prefills = 32
    summary = run_prefills(args)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
