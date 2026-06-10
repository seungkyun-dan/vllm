from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import aiohttp

from exp.common.capability_probe import probe_capabilities
from exp.common.client import run_completion, write_jsonl
from exp.common.env_capture import write_config
from exp.common.server import (
    ManagedServer,
    build_vllm_serve_cmd,
    find_free_port,
    server_env,
    try_lock_gpu_clocks,
)
from exp.e1.analyze import analyze_run

PAIRS = {
    "qwen": {
        "label": "qwen3_8b_0p6b",
        "target": "Qwen/Qwen3-8B",
        "draft": "Qwen/Qwen3-0.6B",
        "allow_64k_default": False,
    },
    "qwen8": {
        "label": "qwen3_8b_0p6b",
        "target": "Qwen/Qwen3-8B",
        "draft": "Qwen/Qwen3-0.6B",
        "allow_64k_default": False,
    },
    "qwen32": {
        "label": "qwen3_32b_1p7b",
        "target": "Qwen/Qwen3-32B",
        "draft": "Qwen/Qwen3-1.7B",
        "allow_64k_default": False,
    },
    "llama70": {
        "label": "llama3p1_70b_llama3p2_1b",
        "target": "meta-llama/Llama-3.1-70B-Instruct",
        "draft": "meta-llama/Llama-3.2-1B-Instruct",
        "allow_64k_default": True,
    },
}


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_or_probe_capabilities(path: Path) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "draft_tp_runtime_requires_equal_tp" in payload:
            return payload
    payload = probe_capabilities()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _make_prompt_tokens(model: str, length: int, out: Path) -> list[int]:
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        return list(payload["tokens"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    base = (
        "In this controlled serving benchmark, the prompt body is deterministic "
        "so every arm receives identical token ids for the same input length. "
    )
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    if not base_ids:
        fallback = tokenizer.eos_token_id
        base_ids = [0 if fallback is None else int(fallback)]
    repeats = (length + len(base_ids) - 1) // len(base_ids)
    ids = (base_ids * repeats)[:length]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"model": model, "length": length, "tokens": ids}) + "\n"
    )
    return ids


def _read_trace_slice(path: Path, start_line: int) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return start_line, []
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = lines[start_line:]
    parsed = [json.loads(line) for line in new_lines if line.strip()]
    return len(lines), parsed


def _annotate_trace_metrics(
    record: dict[str, Any], trace_lines: Sequence[dict[str, Any]]
) -> None:
    if not trace_lines:
        record["scheduler_overhead_ms"] = None
        record["draft_prefill_component_ms"] = None
        record["step_trace_lines"] = 0
        return
    first_ts = float(trace_lines[0].get("ts", record["send_ts"]))
    overhead_ms = max(0.0, (first_ts - float(record["send_ts"])) * 1000.0)
    record["scheduler_overhead_ms"] = overhead_ms
    record["step_trace_lines"] = len(trace_lines)
    if record.get("arm") == "draft_prefill" and record.get("ttft") is not None:
        record["draft_prefill_component_ms"] = max(
            0.0, float(record["ttft"]) * 1000.0 - overhead_ms
        )


async def _run_reps(
    *,
    base_url: str,
    served_model_name: str,
    prompt_tokens: list[int],
    output_len: int,
    reps: int,
    seed: int,
    metadata: dict[str, Any],
    trace_path: Path,
    records_path: Path,
) -> None:
    timeout = aiohttp.ClientTimeout(total=None)
    trace_line = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for rep in range(reps):
            request_id = (
                f"{metadata['pair']}-{metadata['L_in']}-{metadata['arm']}-"
                f"dtp{metadata['tp_draft']}-rep{rep}"
            )
            rec = await run_completion(
                session,
                base_url=base_url,
                model=served_model_name,
                prompt=prompt_tokens,
                output_len=output_len,
                request_id=request_id,
                seed=seed + rep,
                metadata={**metadata, "rep": rep},
            )
            trace_line, trace_slice = _read_trace_slice(trace_path, trace_line)
            _annotate_trace_metrics(rec, trace_slice)
            write_jsonl(records_path, [rec])


def _server_extra_args(args: argparse.Namespace) -> list[str]:
    extra = list(args.extra_server_arg or [])
    if args.gpu_memory_utilization is not None:
        extra.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    return extra


def _run_arm(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    pair_label: str,
    model: str,
    served_model_name: str,
    tp: int,
    max_model_len: int,
    prompt_tokens: list[int],
    output_len: int,
    metadata: dict[str, Any],
    speculative_config: dict[str, Any] | None = None,
) -> None:
    host = args.host
    port = find_free_port(host)
    base_url = f"http://{host}:{port}"
    trace_path = run_dir / "raw" / f"trace_{metadata['request_group']}.jsonl"
    log_path = run_dir / "raw" / f"server_{metadata['request_group']}.log"
    cmd = build_vllm_serve_cmd(
        model=model,
        host=host,
        port=port,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        served_model_name=served_model_name,
        seed=args.seed,
        speculative_config=speculative_config,
        extra_args=_server_extra_args(args),
    )
    env = server_env(trace_path)
    records_path = run_dir / "raw" / "records.jsonl"
    arm_config = {
        "request_group": metadata["request_group"],
        "cmd": cmd,
        "trace_path": str(trace_path),
        "log_path": str(log_path),
        "base_url": base_url,
        "pair": pair_label,
    }
    write_jsonl(run_dir / "raw" / "arm_manifest.jsonl", [arm_config])
    if args.dry_run:
        print(json.dumps(arm_config, indent=2))
        return
    with ManagedServer(
        cmd=cmd,
        env=env,
        log_path=log_path,
        base_url=base_url,
        ready_timeout_s=args.server_ready_timeout_s,
    ):
        asyncio.run(
            _run_reps(
                base_url=base_url,
                served_model_name=served_model_name,
                prompt_tokens=prompt_tokens,
                output_len=output_len,
                reps=args.reps,
                seed=args.seed,
                metadata=metadata,
                trace_path=trace_path,
                records_path=records_path,
            )
        )


def _iter_lengths(
    pair_name: str, lengths: Sequence[int], include_qwen_64k: bool
) -> list[int]:
    pair = PAIRS[pair_name]
    out = []
    for length in lengths:
        if length >= 65_536 and not pair["allow_64k_default"] and not include_qwen_64k:
            continue
        out.append(length)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pairs", default="qwen")
    parser.add_argument("--lengths", default="1024,4096,16384,65536")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--spec-k", type=int, default=4)
    parser.add_argument("--tp-target", type=int, default=1)
    parser.add_argument("--tp-drafts", default="1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--capabilities", type=Path, default=Path("results/capabilities.json")
    )
    parser.add_argument("--include-qwen-64k", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-clock-mhz", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--server-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    args = parser.parse_args()

    run_id = args.run_id or _run_id()
    run_dir = Path("results") / "e1" / run_id
    (run_dir / "raw" / "prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    capabilities = _load_or_probe_capabilities(args.capabilities)
    lock_result = try_lock_gpu_clocks(args.gpu_clock_mhz)
    write_config(
        run_dir,
        exp="e1",
        run_id=run_id,
        cli_args=sys.argv,
        env_overrides={},
        server_args=[],
        client_args=[],
        extra={
            "purpose": "draft prefill TTFT inflation decomposition",
            "capabilities": capabilities,
            "gpu_clock_lock": lock_result,
            "guards": {
                "prefix_caching": "off",
                "temperature": 0,
                "ignore_eos": True,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 256,
                "closed_loop_single_client": True,
            },
        },
    )

    pair_names = [p.strip() for p in args.pairs.split(",") if p.strip()]
    lengths = _parse_csv_ints(args.lengths)
    draft_tps = _parse_csv_ints(args.tp_drafts)
    draft_model_supported = bool(capabilities.get("draft_model_supported"))
    integrated_tp_asym_supported = bool(
        capabilities.get("supports_integrated_draft_model_tp4_tp1")
    )
    for pair_name in pair_names:
        pair = PAIRS[pair_name]
        pair_label = str(pair["label"])
        for L_in in _iter_lengths(pair_name, lengths, args.include_qwen_64k):
            prompt_path = run_dir / "raw" / "prompts" / f"{pair_label}_{L_in}.json"
            prompt_tokens = _make_prompt_tokens(str(pair["target"]), L_in, prompt_path)
            max_model_len = L_in + args.output_len + args.spec_k + 16
            target_meta = {
                "pair": pair_label,
                "L_in": L_in,
                "tp_target": args.tp_target,
                "tp_draft": 0,
                "arm": "target_only",
                "request_group": f"{pair_label}_{L_in}_target",
            }
            _run_arm(
                args=args,
                run_dir=run_dir,
                pair_label=pair_label,
                model=str(pair["target"]),
                served_model_name=f"{pair_label}-target",
                tp=args.tp_target,
                max_model_len=max_model_len,
                prompt_tokens=prompt_tokens,
                output_len=args.output_len,
                metadata=target_meta,
            )
            for tp_draft in draft_tps:
                same_tp = tp_draft == args.tp_target
                can_run_integrated = draft_model_supported and (
                    same_tp or integrated_tp_asym_supported
                )
                if can_run_integrated:
                    spec_config = {
                        "method": "draft_model",
                        "model": pair["draft"],
                        "num_speculative_tokens": args.spec_k,
                        "draft_tensor_parallel_size": tp_draft,
                    }
                    spec_meta = {
                        "pair": pair_label,
                        "L_in": L_in,
                        "tp_target": args.tp_target,
                        "tp_draft": tp_draft,
                        "arm": "integrated_spec",
                        "request_group": f"{pair_label}_{L_in}_spec_dtp{tp_draft}",
                    }
                    _run_arm(
                        args=args,
                        run_dir=run_dir,
                        pair_label=pair_label,
                        model=str(pair["target"]),
                        served_model_name=f"{pair_label}-spec-dtp{tp_draft}",
                        tp=args.tp_target,
                        max_model_len=max_model_len,
                        prompt_tokens=prompt_tokens,
                        output_len=args.output_len,
                        metadata=spec_meta,
                        speculative_config=spec_config,
                    )
                draft_meta = {
                    "pair": pair_label,
                    "L_in": L_in,
                    "tp_target": args.tp_target,
                    "tp_draft": tp_draft,
                    "arm": "draft_prefill",
                    "request_group": f"{pair_label}_{L_in}_draft_dtp{tp_draft}",
                }
                _run_arm(
                    args=args,
                    run_dir=run_dir,
                    pair_label=pair_label,
                    model=str(pair["draft"]),
                    served_model_name=f"{pair_label}-draft-dtp{tp_draft}",
                    tp=tp_draft,
                    max_model_len=max_model_len,
                    prompt_tokens=prompt_tokens,
                    output_len=1,
                    metadata=draft_meta,
                )
    if not args.dry_run:
        analyze_run(run_dir, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
