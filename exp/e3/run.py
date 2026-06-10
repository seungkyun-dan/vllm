from __future__ import annotations

import argparse
import json
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
from exp.e3.analyze import analyze_run
from exp.e3.nsys_nccl import extract_nccl_kernel_share
from exp.e3.utils import MODE_CAPS, OVERLAP_MODES, append_jsonl, parse_int_csv
from exp.e3.utils import parse_str_csv


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _env(args: argparse.Namespace, *, step_trace: Path | None = None) -> dict[str, str]:
    env = server_env(step_trace)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.mps_pipe_directory:
        env["CUDA_MPS_PIPE_DIRECTORY"] = args.mps_pipe_directory
    if args.mps_log_directory:
        env["CUDA_MPS_LOG_DIRECTORY"] = args.mps_log_directory
    return env


def _target_env(args: argparse.Namespace, step_trace: Path) -> dict[str, str]:
    env = _env(args, step_trace=step_trace)
    env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    return env


def _draft_env(args: argparse.Namespace, mode: str | None) -> dict[str, str]:
    env = _env(args)
    cap = None if mode is None else MODE_CAPS.get(mode)
    if cap is not None:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = cap
    else:
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    return env


def _serve_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    nsys_report_stem: Path | None = None,
) -> list[str]:
    max_model_len = args.target_max_model_len
    if max_model_len is None:
        max_model_len = args.input_len + args.output_len + 16
    extra_args = [
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--disable-log-requests",
    ]
    if args.enforce_eager:
        extra_args.append("--enforce-eager")
    if args.trust_remote_code:
        extra_args.append("--trust-remote-code")
    cmd = build_vllm_serve_cmd(
        model=args.target_model,
        host=host,
        port=port,
        tensor_parallel_size=args.target_tp,
        max_model_len=max_model_len,
        served_model_name=args.served_model_name,
        seed=args.seed,
        extra_args=extra_args,
    )
    if nsys_report_stem is None:
        return cmd
    return [
        "nsys",
        "profile",
        "--gpu-metrics-devices=all",
        "--trace=cuda,nvtx,osrt,cublas,cudnn",
        "--sample=none",
        "--force-overwrite=true",
        "-o",
        str(nsys_report_stem),
        *cmd,
    ]


def _target_client_cmd(
    args: argparse.Namespace,
    *,
    base_url: str,
    records_path: Path,
    summary_path: Path,
    prompt_meta_path: Path,
    rep: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "exp.e3.target_decode_load",
        "--base-url",
        base_url,
        "--model-name",
        args.served_model_name,
        "--tokenizer-model",
        args.target_model,
        "--concurrency",
        str(args.concurrency),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--seed",
        str(args.seed + rep),
        "--request-seed-base",
        str(args.request_seed_base + rep * 1000),
        "--out",
        str(records_path),
        "--summary-out",
        str(summary_path),
        "--prompt-meta-out",
        str(prompt_meta_path),
        *(["--trust-remote-code"] if args.trust_remote_code else []),
    ]


def _draft_cmd(
    args: argparse.Namespace,
    *,
    chunk_tokens: int,
    records_path: Path,
    summary_path: Path,
    rep: int,
    duration_s: float | None,
    num_prefills: int | None,
) -> list[str]:
    max_model_len = args.draft_max_model_len or chunk_tokens + 16
    cmd = [
        sys.executable,
        "-m",
        "exp.e3.draft_prefill_worker",
        "--model",
        args.draft_model,
        "--chunk-tokens",
        str(chunk_tokens),
        "--tensor-parallel-size",
        str(args.draft_tp),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--seed",
        str(args.seed + rep),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--records-out",
        str(records_path),
        "--summary-out",
        str(summary_path),
    ]
    if duration_s is not None:
        cmd.extend(["--duration-s", str(duration_s)])
    if num_prefills is not None:
        cmd.extend(["--num-prefills", str(num_prefills)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def _wait_for_decode_steps(
    trace_path: Path,
    *,
    min_decode_steps: int,
    timeout_s: float,
    watched_proc: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_decode = 0
    while time.monotonic() < deadline:
        if watched_proc is not None and watched_proc.poll() is not None:
            raise RuntimeError(
                "target client exited before the decode-only window was reached"
            )
        if trace_path.exists():
            decode = 0
            with trace_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (
                        int(row.get("prefill_tokens", 0) or 0) == 0
                        and int(row.get("total_scheduled_tokens", 0) or 0) > 0
                    ):
                        decode += 1
            last_decode = decode
            if decode >= min_decode_steps:
                return
        time.sleep(1.0)
    raise TimeoutError(
        "timed out waiting for target decode-only steps "
        f"({last_decode}/{min_decode_steps})"
    )


def _terminate(proc: subprocess.Popen[str] | None) -> int | None:
    if proc is None:
        return None
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        proc.wait(timeout=60.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30.0)
    return proc.returncode


def _write_dry_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_target_cell(
    args: argparse.Namespace,
    *,
    raw_dir: Path,
    mode: str,
    rep: int,
    chunk_tokens: int,
    nsys_report_stem: Path | None = None,
) -> dict[str, Any]:
    cell = raw_dir / f"{mode}_rep{rep}_Cd{chunk_tokens}"
    cell.mkdir(parents=True, exist_ok=True)
    trace_path = cell / "target.step_trace.jsonl"
    records_path = cell / "target.client.jsonl"
    summary_path = cell / "target.summary.json"
    prompt_meta_path = cell / "target.prompt.json"
    host = "127.0.0.1"
    port = find_free_port(host)
    base_url = f"http://{host}:{port}"
    serve_cmd = _serve_cmd(
        args, host=host, port=port, nsys_report_stem=nsys_report_stem
    )
    client_cmd = _target_client_cmd(
        args,
        base_url=base_url,
        records_path=records_path,
        summary_path=summary_path,
        prompt_meta_path=prompt_meta_path,
        rep=rep,
    )
    manifest = {
        "kind": "target",
        "mode": mode,
        "rep": rep,
        "C_d": chunk_tokens,
        "target_trace": str(trace_path),
        "target_records": str(records_path),
        "target_summary": str(summary_path),
        "serve_cmd": serve_cmd,
        "client_cmd": client_cmd,
    }
    if args.dry_run:
        _write_dry_manifest(cell / "manifest.json", manifest)
        return manifest
    with ManagedServer(
        cmd=serve_cmd,
        env=_target_env(args, trace_path),
        log_path=cell / "target.server.log",
        base_url=base_url,
        ready_timeout_s=args.server_ready_timeout_s,
    ):
        with (cell / "target.client.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                client_cmd,
                env=_env(args),
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
    manifest["client_returncode"] = proc.returncode
    if nsys_report_stem is not None:
        manifest.update(extract_nccl_kernel_share(nsys_report_stem))
    return manifest


def _run_draft_solo(
    args: argparse.Namespace,
    *,
    raw_dir: Path,
    rep: int,
    chunk_tokens: int,
) -> dict[str, Any]:
    cell = raw_dir / f"draft_solo_rep{rep}_Cd{chunk_tokens}"
    cell.mkdir(parents=True, exist_ok=True)
    records_path = cell / "draft.jsonl"
    summary_path = cell / "draft.summary.json"
    cmd = _draft_cmd(
        args,
        chunk_tokens=chunk_tokens,
        records_path=records_path,
        summary_path=summary_path,
        rep=rep,
        duration_s=None,
        num_prefills=args.draft_solo_prefills,
    )
    manifest = {
        "kind": "draft_solo",
        "mode": "draft_solo",
        "rep": rep,
        "C_d": chunk_tokens,
        "draft_records": str(records_path),
        "draft_summary": str(summary_path),
        "draft_cmd": cmd,
    }
    if args.dry_run:
        _write_dry_manifest(cell / "manifest.json", manifest)
        return manifest
    with (cell / "draft.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            env=_draft_env(args, None),
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    manifest["draft_returncode"] = proc.returncode
    return manifest


def _run_overlap_cell(
    args: argparse.Namespace,
    *,
    raw_dir: Path,
    mode: str,
    rep: int,
    chunk_tokens: int,
    nsys_report_stem: Path | None = None,
) -> dict[str, Any]:
    cell = raw_dir / f"{mode}_rep{rep}_Cd{chunk_tokens}"
    cell.mkdir(parents=True, exist_ok=True)
    trace_path = cell / "target.step_trace.jsonl"
    target_records = cell / "target.client.jsonl"
    target_summary = cell / "target.summary.json"
    prompt_meta = cell / "target.prompt.json"
    draft_records = cell / "draft.jsonl"
    draft_summary = cell / "draft.summary.json"
    host = "127.0.0.1"
    port = find_free_port(host)
    base_url = f"http://{host}:{port}"
    serve_cmd = _serve_cmd(
        args, host=host, port=port, nsys_report_stem=nsys_report_stem
    )
    client_cmd = _target_client_cmd(
        args,
        base_url=base_url,
        records_path=target_records,
        summary_path=target_summary,
        prompt_meta_path=prompt_meta,
        rep=rep,
    )
    draft_cmd = _draft_cmd(
        args,
        chunk_tokens=chunk_tokens,
        records_path=draft_records,
        summary_path=draft_summary,
        rep=rep,
        duration_s=args.draft_duration_s,
        num_prefills=None,
    )
    manifest = {
        "kind": "overlap",
        "mode": mode,
        "rep": rep,
        "C_d": chunk_tokens,
        "target_trace": str(trace_path),
        "target_records": str(target_records),
        "target_summary": str(target_summary),
        "draft_records": str(draft_records),
        "draft_summary": str(draft_summary),
        "serve_cmd": serve_cmd,
        "client_cmd": client_cmd,
        "draft_cmd": draft_cmd,
        "mps_cap": MODE_CAPS.get(mode),
    }
    if args.dry_run:
        _write_dry_manifest(cell / "manifest.json", manifest)
        return manifest
    draft_proc: subprocess.Popen[str] | None = None
    target_proc: subprocess.Popen[str] | None = None
    target_returncode: int | None = None
    draft_returncode: int | None = None
    with ManagedServer(
        cmd=serve_cmd,
        env=_target_env(args, trace_path),
        log_path=cell / "target.server.log",
        base_url=base_url,
        ready_timeout_s=args.server_ready_timeout_s,
    ):
        with (cell / "target.client.log").open("w", encoding="utf-8") as target_log:
            try:
                target_proc = subprocess.Popen(
                    client_cmd,
                    env=_env(args),
                    stdout=target_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                _wait_for_decode_steps(
                    trace_path,
                    min_decode_steps=args.start_draft_after_decode_steps,
                    timeout_s=args.decode_window_timeout_s,
                    watched_proc=target_proc,
                )
                with (cell / "draft.log").open("w", encoding="utf-8") as draft_log:
                    draft_proc = subprocess.Popen(
                        draft_cmd,
                        env=_draft_env(args, mode),
                        stdout=draft_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    target_returncode = target_proc.wait()
            finally:
                draft_returncode = _terminate(draft_proc)
                if target_proc is not None and target_proc.poll() is None:
                    target_returncode = _terminate(target_proc)
    manifest["client_returncode"] = target_returncode
    manifest["draft_returncode"] = draft_returncode
    if nsys_report_stem is not None:
        manifest.update(extract_nccl_kernel_share(nsys_report_stem))
    return manifest


def run_experiment(args: argparse.Namespace) -> Path:
    run_id = args.run_id or _run_id()
    run_dir = Path("results") / "e3" / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clock_lock = try_lock_gpu_clocks(args.gpu_clock_mhz)
    write_config(
        run_dir,
        exp="e3",
        run_id=run_id,
        cli_args=sys.argv,
        extra={
            "purpose": "target decode slowdown under concurrent draft prefill",
            "args": vars(args),
            "clock_lock": clock_lock,
            "mps_mode_caps": MODE_CAPS,
        },
    )
    manifest_path = raw_dir / "cell_manifest.jsonl"
    chunks = parse_int_csv(args.chunk_tokens)
    modes = parse_str_csv(args.modes)
    did_nsys = False
    for rep in range(args.reps):
        row = _run_target_cell(
            args,
            raw_dir=raw_dir,
            mode="m0",
            rep=rep,
            chunk_tokens=0,
        )
        append_jsonl(manifest_path, [row])
    for rep in range(args.reps):
        for chunk_tokens in chunks:
            row = _run_draft_solo(
                args, raw_dir=raw_dir, rep=rep, chunk_tokens=chunk_tokens
            )
            append_jsonl(manifest_path, [row])
    for rep in range(args.reps):
        for chunk_tokens in chunks:
            for mode in modes:
                if mode not in OVERLAP_MODES:
                    raise ValueError(f"Unknown E3 overlap mode: {mode}")
                nsys_stem = None
                if args.run_nsys_m1 and not did_nsys and mode == "m1":
                    if (
                        args.nsys_chunk_tokens is None
                        or chunk_tokens == args.nsys_chunk_tokens
                    ):
                        nsys_stem = raw_dir / f"nsys_{mode}_rep{rep}_Cd{chunk_tokens}"
                        did_nsys = True
                row = _run_overlap_cell(
                    args,
                    raw_dir=raw_dir,
                    mode=mode,
                    rep=rep,
                    chunk_tokens=chunk_tokens,
                    nsys_report_stem=nsys_stem,
                )
                append_jsonl(manifest_path, [row])
    if not args.dry_run:
        analyze_args = argparse.Namespace(
            warmup_itl_per_request=args.warmup_itl_per_request,
            warmup_decode_steps=args.warmup_decode_steps,
            measure_steps=args.measure_steps,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
        )
        analyze_run(run_dir, analyze_args)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--served-model-name", default="target")
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--chunk-tokens", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--modes", default="m1,m2,m3")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--target-max-model-len", type=int)
    parser.add_argument("--draft-max-model-len", type=int)
    parser.add_argument("--draft-solo-prefills", type=int, default=32)
    parser.add_argument("--draft-duration-s", type=float, default=7200.0)
    parser.add_argument("--start-draft-after-decode-steps", type=int, default=8)
    parser.add_argument("--decode-window-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--warmup-itl-per-request", type=int, default=32)
    parser.add_argument("--warmup-decode-steps", type=int, default=32)
    parser.add_argument("--measure-steps", type=int, default=300)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--request-seed-base", type=int, default=100000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gpu-clock-mhz", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--mps-pipe-directory")
    parser.add_argument("--mps-log-directory")
    parser.add_argument("--run-nsys-m1", action="store_true")
    parser.add_argument("--nsys-chunk-tokens", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = run_experiment(args)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
