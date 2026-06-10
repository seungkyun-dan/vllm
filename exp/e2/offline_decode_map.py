from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from exp.common.env_capture import write_config
from exp.e2.analyze import plot_offline
from exp.e2.nsys_metrics import extract_nsys_gpu_metrics
from exp.e2.utils import (
    analytic_utils,
    decode_measurement,
    kv_bytes_per_decode_token,
    parse_int_csv,
    read_jsonl,
    write_csv,
)


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _worker_cmd(
    args: argparse.Namespace, batch_size: int, context_len: int
) -> list[str]:
    output_len = args.output_len or args.warmup_decode_steps + args.measure_steps + 4
    max_model_len = args.max_model_len or context_len + output_len + 16
    return [
        sys.executable,
        "-m",
        "exp.e2.offline_decode_worker",
        "--model",
        args.model,
        "--batch-size",
        str(batch_size),
        "--context-len",
        str(context_len),
        "--output-len",
        str(output_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]


def _nsys_point(args: argparse.Namespace, batch_size: int, context_len: int) -> bool:
    return f"{batch_size}:{context_len}" in set(args.nsys_points.split(","))


def run_grid(args: argparse.Namespace) -> Path:
    run_id = args.run_id or _run_id()
    run_dir = Path("results") / "e2" / run_id
    raw_dir = run_dir / "raw" / "offline"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_config(
        run_dir,
        exp="e2",
        run_id=run_id,
        cli_args=sys.argv,
        extra={"part": "offline_decode_regime_map", "args": vars(args)},
    )

    rows: list[dict[str, Any]] = []
    for context_len in parse_int_csv(args.context_lens):
        kv_bytes, kv_shape = kv_bytes_per_decode_token(
            model=args.model, context_len=context_len, dtype_bytes=args.dtype_bytes
        )
        for batch_size in parse_int_csv(args.batch_sizes):
            stem = f"B{batch_size}_L{context_len}"
            trace_path = raw_dir / f"{stem}.step_trace.jsonl"
            meta_path = raw_dir / f"{stem}.meta.json"
            log_path = raw_dir / f"{stem}.log"
            cmd = _worker_cmd(args, batch_size, context_len) + ["--out", str(meta_path)]
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            if args.trust_remote_code:
                cmd.append("--trust-remote-code")
            env = os.environ.copy()
            env["VLLM_STEP_TRACE"] = str(trace_path)
            run_cmd = cmd
            nsys_report = None
            if args.run_nsys and _nsys_point(args, batch_size, context_len):
                nsys_report = raw_dir / f"nsys_{stem}"
                run_cmd = [
                    "nsys",
                    "profile",
                    "--gpu-metrics-devices=all",
                    "--trace=cuda,nvtx,osrt,cublas,cudnn",
                    "--sample=none",
                    "--force-overwrite=true",
                    "-o",
                    str(nsys_report),
                    *cmd,
                ]
            manifest = {
                "batch_size": batch_size,
                "context_len": context_len,
                "cmd": run_cmd,
                "trace_path": str(trace_path),
                "meta_path": str(meta_path),
            }
            if args.dry_run:
                (raw_dir / f"{stem}.manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                continue
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    run_cmd,
                    env=env,
                    check=False,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            trace_rows = read_jsonl(trace_path)
            measurement = decode_measurement(
                trace_rows,
                warmup_decode_steps=args.warmup_decode_steps,
                measure_steps=args.measure_steps,
            )
            util = analytic_utils(
                tokens_per_s=measurement.get("tokens_per_s"),
                n_params=args.n_params,
                tensor_parallel_size=args.tensor_parallel_size,
                peak_flops_per_gpu_tflops=args.peak_flops_per_gpu_tflops,
                kv_bytes_per_token=kv_bytes,
                hbm_bw_per_gpu_gbps=args.hbm_bw_per_gpu_gbps,
            )
            row = {
                "part": "offline",
                "model": args.model,
                "B": batch_size,
                "L_ctx": context_len,
                "returncode": proc.returncode,
                "kv_bytes_per_decode_token": kv_bytes,
                **kv_shape,
                **measurement,
                **util,
                "trace_path": str(trace_path),
                "log_path": str(log_path),
            }
            if nsys_report is not None:
                row.update(
                    extract_nsys_gpu_metrics(nsys_report.with_suffix(".nsys-rep"))
                )
            rows.append(row)
            write_csv(run_dir / "part_a_offline.csv", rows)
            plot_offline(run_dir / "part_a_offline.csv", run_dir / "plots")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--context-lens", default="2048,8192")
    parser.add_argument("--warmup-decode-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--n-params", type=float, default=8e9)
    parser.add_argument("--peak-flops-per-gpu-tflops", type=float)
    parser.add_argument("--hbm-bw-per-gpu-gbps", type=float)
    parser.add_argument("--run-nsys", action="store_true")
    parser.add_argument("--nsys-points", default="8:8192,16:8192")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = run_grid(args)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
