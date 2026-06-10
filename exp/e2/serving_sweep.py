from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from exp.common.env_capture import write_config
from exp.common.server import (
    ManagedServer,
    build_vllm_serve_cmd,
    find_free_port,
    server_env,
    try_lock_gpu_clocks,
)
from exp.e2.analyze import plot_serving, write_bubble_verdict
from exp.e2.utils import (
    parse_float_csv,
    read_trace_slice,
    serving_step_stats,
    trace_line_count,
    write_csv,
)


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _dataset_args(args: argparse.Namespace, dataset: str) -> list[str]:
    if dataset == "random":
        return [
            "--dataset-name",
            "random",
            "--random-input-len",
            str(args.random_input_len),
            "--random-output-len",
            str(args.output_len),
            "--random-range-ratio",
            "0.0",
        ]
    if dataset == "sharegpt":
        if not args.sharegpt_path:
            raise ValueError(
                "--sharegpt-path is required when dataset includes sharegpt"
            )
        return ["--dataset-name", "sharegpt", "--dataset-path", args.sharegpt_path]
    raise ValueError(f"Unknown dataset: {dataset}")


def _bench_cmd(
    *,
    args: argparse.Namespace,
    dataset: str,
    rate: float,
    num_prompts: int,
    host: str,
    port: int,
    result_dir: Path,
    filename: str,
    metadata: dict[str, Any],
) -> list[str]:
    meta_args = []
    for key, value in metadata.items():
        meta_args.extend(["--metadata", f"{key}={value}"])
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        host,
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer or args.model,
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(rate),
        "--temperature",
        "0",
        "--ignore-eos",
        "--seed",
        str(args.seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        filename,
        "--disable-tqdm",
        *_dataset_args(args, dataset),
        *meta_args,
    ]


def _run_bench(
    *,
    cmd: list[str],
    log_path: Path,
    result_path: Path,
) -> tuple[int, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result: dict[str, Any] = {}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return proc.returncode, result


def _quick_lambda_sat(
    *,
    args: argparse.Namespace,
    host: str,
    port: int,
    dataset: str,
    raw_dir: Path,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    previous_good: float | None = None
    for rate in parse_float_csv(args.probe_rates):
        num_prompts = max(1, math.ceil(rate * args.probe_duration_s))
        filename = f"probe_{dataset}_r{rate}.json"
        result_dir = raw_dir / "bench_results"
        result_path = result_dir / filename
        cmd = _bench_cmd(
            args=args,
            dataset=dataset,
            rate=rate,
            num_prompts=num_prompts,
            host=host,
            port=port,
            result_dir=result_dir,
            filename=filename,
            metadata={"phase": "probe", "dataset": dataset, "rate": rate},
        )
        rc, result = _run_bench(
            cmd=cmd,
            log_path=raw_dir / f"probe_{dataset}_r{rate}.log",
            result_path=result_path,
        )
        throughput = float(result.get("request_throughput") or 0.0)
        failed = int(result.get("failed") or 0)
        stable = rc == 0 and failed == 0 and throughput >= 0.9 * rate
        rows.append(
            {
                "phase": "probe",
                "dataset": dataset,
                "offered_rate": rate,
                "request_throughput": throughput,
                "failed": failed,
                "stable": stable,
                "result_path": str(result_path),
            }
        )
        if stable:
            previous_good = throughput
        elif previous_good is not None:
            return previous_good, rows
    if previous_good is not None:
        return previous_good, rows
    best = max((float(row["request_throughput"]) for row in rows), default=1.0)
    return max(best, 1e-6), rows


def run_sweep(args: argparse.Namespace) -> Path:
    run_id = args.run_id or _run_id()
    run_dir = Path("results") / "e2" / run_id
    raw_dir = run_dir / "raw" / "serving"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_path = raw_dir / "server.step_trace.jsonl"
    port = args.port or find_free_port(args.host)
    base_url = f"http://{args.host}:{port}"
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
    write_config(
        run_dir,
        exp="e2",
        run_id=run_id,
        cli_args=sys.argv,
        env_overrides={"VLLM_STEP_TRACE": str(trace_path)},
        server_args=server_args,
        extra={"part": "serving_decode_only_sweep", "args": vars(args)},
    )
    try_lock_gpu_clocks(args.gpu_clock_mhz)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    if args.dry_run:
        manifest = {
            "server_args": server_args,
            "base_url": base_url,
            "datasets": datasets,
        }
        (raw_dir / "dry_run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return run_dir
    with ManagedServer(
        cmd=server_args,
        env=server_env(trace_path),
        log_path=raw_dir / "server.log",
        base_url=base_url,
        ready_timeout_s=args.server_ready_timeout_s,
    ):
        lambda_sat = args.lambda_sat
        if lambda_sat is None:
            lambda_sat, probe_rows = _quick_lambda_sat(
                args=args,
                host=args.host,
                port=port,
                dataset=args.probe_dataset,
                raw_dir=raw_dir,
            )
            write_csv(run_dir / "part_b_saturation_probe.csv", probe_rows)
        for dataset in datasets:
            for rho in parse_float_csv(args.rhos):
                rate = rho * lambda_sat
                num_prompts = max(1, math.ceil(rate * args.duration_s))
                start_line = trace_line_count(trace_path)
                filename = f"sweep_{dataset}_rho{rho:.2f}.json"
                result_dir = raw_dir / "bench_results"
                result_path = result_dir / filename
                cmd = _bench_cmd(
                    args=args,
                    dataset=dataset,
                    rate=rate,
                    num_prompts=num_prompts,
                    host=args.host,
                    port=port,
                    result_dir=result_dir,
                    filename=filename,
                    metadata={
                        "phase": "sweep",
                        "dataset": dataset,
                        "rho": rho,
                        "lambda_sat": lambda_sat,
                    },
                )
                rc, result = _run_bench(
                    cmd=cmd,
                    log_path=raw_dir / f"sweep_{dataset}_rho{rho:.2f}.log",
                    result_path=result_path,
                )
                _next_line, trace_rows = read_trace_slice(trace_path, start_line)
                stats = serving_step_stats(
                    trace_rows, max_num_batched_tokens=args.max_num_batched_tokens
                )
                rows.append(
                    {
                        "part": "serving",
                        "dataset": dataset,
                        "rho": rho,
                        "lambda_sat_req_s": lambda_sat,
                        "lambda_req_s": rate,
                        "num_prompts": num_prompts,
                        "returncode": rc,
                        "request_throughput": result.get("request_throughput"),
                        "completed": result.get("completed"),
                        "failed": result.get("failed"),
                        "mean_ttft_ms": result.get("mean_ttft_ms"),
                        "p95_ttft_ms": result.get("p95_ttft_ms"),
                        "result_path": str(result_path),
                        **stats,
                    }
                )
                write_csv(run_dir / "part_b_serving.csv", rows)
                plot_serving(run_dir / "part_b_serving.csv", run_dir / "plots")
                write_bubble_verdict(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--served-model-name", default="e2-target")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-model-len", type=int, default=8720)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--datasets", default="random,sharegpt")
    parser.add_argument("--sharegpt-path")
    parser.add_argument("--lambda-sat", type=float)
    parser.add_argument("--probe-dataset", default="random")
    parser.add_argument("--probe-rates", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--probe-duration-s", type=float, default=60.0)
    parser.add_argument("--rhos", default="0.2,0.4,0.6,0.8,0.95")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--gpu-clock-mhz", type=int)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = run_sweep(args)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
