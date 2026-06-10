from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from exp.common.analyze import parsed_rows, summarize, write_parsed_csv
from exp.common.client import load_jsonl, run_open_loop
from exp.common.env_capture import write_config
from exp.common.server import (
    ManagedServer,
    build_vllm_serve_cmd,
    find_free_port,
    server_env,
    try_lock_gpu_clocks,
)


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--served-model-name", default="smoke-model")
    parser.add_argument("--exp", default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--no-launch-server", action="store_true")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--arrival", choices=["poisson", "gamma"], default="poisson")
    parser.add_argument("--gamma-shape", type=float, default=1.0)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-clock-mhz", type=int)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    args = parser.parse_args()

    run_id = args.run_id or _run_id()
    run_dir = Path("results") / args.exp / run_id
    raw_dir = run_dir / "raw"
    plots_dir = run_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    client_out = raw_dir / "client.jsonl"
    step_trace = raw_dir / "step_trace.jsonl"
    port = args.port or find_free_port(args.host)
    base_url = args.base_url or f"http://{args.host}:{port}"
    schedule = [(args.duration_s, args.rate)]
    server_args = build_vllm_serve_cmd(
        model=args.model,
        host=args.host,
        port=port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        served_model_name=args.served_model_name,
        seed=args.seed,
        extra_args=args.extra_server_arg,
    )
    client_args = [
        "--schedule-json",
        json.dumps(schedule),
        "--output-len",
        str(args.output_len),
    ]
    env = server_env(step_trace)
    lock_result = try_lock_gpu_clocks(args.gpu_clock_mhz)
    write_config(
        run_dir,
        exp=args.exp,
        run_id=run_id,
        cli_args=sys.argv,
        env_overrides={"VLLM_STEP_TRACE": str(step_trace)},
        server_args=server_args,
        client_args=client_args,
        extra={"gpu_clock_lock": lock_result},
    )

    async def run_client() -> None:
        await run_open_loop(
            base_url=base_url,
            model=args.served_model_name,
            prompt="Hello, my name is",
            output_len=args.output_len,
            schedule=schedule,
            arrival=args.arrival,
            gamma_shape=args.gamma_shape,
            seed=args.seed,
            out=client_out,
            request_seed_base=1000,
        )

    if args.no_launch_server:
        asyncio.run(run_client())
    else:
        with ManagedServer(
            cmd=server_args,
            env=env,
            log_path=raw_dir / "server.log",
            base_url=base_url,
        ):
            asyncio.run(run_client())

    rows = parsed_rows(load_jsonl(client_out), warmup_s=0.0)
    write_parsed_csv(rows, run_dir / "parsed.csv")
    summary = summarize(
        rows,
        ttft_slo_ms=1000.0,
        tpot_slo_ms=100.0,
        duration_s=args.duration_s,
        bootstrap_iters=500,
        seed=args.seed,
    )
    (plots_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
