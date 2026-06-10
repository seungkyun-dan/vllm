from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp

from exp.common.client import run_completion
from exp.e3.utils import prompt_hash, token_prompt


async def run_fixed_decode_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = token_prompt(
        model=args.tokenizer_model,
        length=args.input_len,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_meta = {
        "tokenizer_model": args.tokenizer_model,
        "input_len": args.input_len,
        "prompt_sha256": prompt_hash(prompt),
        "seed": args.seed,
    }
    if args.prompt_meta_out:
        args.prompt_meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_meta_out.write_text(
            json.dumps(prompt_meta, indent=2, sort_keys=True) + "\n"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("", encoding="utf-8")
    timeout = aiohttp.ClientTimeout(total=None)
    records: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    start_gate = asyncio.Event()

    async def one(session: aiohttp.ClientSession, idx: int) -> None:
        await start_gate.wait()
        rec = await run_completion(
            session,
            base_url=args.base_url,
            model=args.model_name,
            prompt=prompt,
            output_len=args.output_len,
            request_id=f"e3-target-{idx}",
            seed=args.request_seed_base + idx,
            temperature=0.0,
            ignore_eos=True,
            metadata={
                "client_index": idx,
                "input_len_requested": args.input_len,
                "output_len_requested": args.output_len,
                "prompt_sha256": prompt_meta["prompt_sha256"],
            },
        )
        async with lock:
            records.append(rec)
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(one(session, idx))
            for idx in range(args.concurrency)
        ]
        await asyncio.sleep(0.1)
        started_perf = time.perf_counter()
        start_gate.set()
        await asyncio.gather(*tasks)
    summary = {
        "concurrency": args.concurrency,
        "elapsed_s": time.perf_counter() - started_perf,
        "n_records": len(records),
        **prompt_meta,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default="target")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-seed-base", type=int, default=100000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--prompt-meta-out", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    records = asyncio.run(run_fixed_decode_batch(args))
    print(json.dumps({"n_records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
