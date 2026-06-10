from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from exp.common.analyze import percentile


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trace_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_trace_slice(
    path: Path, start_line: int = 0
) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return start_line, []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[start_line:] if line.strip()]
    return len(lines), rows


def decode_measurement(
    trace_rows: Sequence[dict[str, Any]],
    *,
    warmup_decode_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    decode_rows = [
        row
        for row in trace_rows
        if int(row.get("prefill_tokens", 0)) == 0
        and int(row.get("total_scheduled_tokens", 0)) > 0
    ]
    selected = decode_rows[warmup_decode_steps : warmup_decode_steps + measure_steps]
    if len(selected) < 2:
        return {
            "decode_only_steps": len(decode_rows),
            "measured_steps": len(selected),
            "tokens_per_s": None,
            "mean_step_ms": None,
        }
    duration_s = float(selected[-1]["ts"]) - float(selected[0]["ts"])
    tokens = sum(int(row["total_scheduled_tokens"]) for row in selected[1:])
    tokens_per_s = tokens / duration_s if duration_s > 0 else None
    return {
        "decode_only_steps": len(decode_rows),
        "measured_steps": len(selected),
        "measured_duration_s": duration_s,
        "measured_tokens": tokens,
        "tokens_per_s": tokens_per_s,
        "mean_step_ms": duration_s / (len(selected) - 1) * 1000.0,
    }


def serving_step_stats(
    trace_rows: Sequence[dict[str, Any]],
    *,
    max_num_batched_tokens: int,
) -> dict[str, Any]:
    active = [
        row
        for row in trace_rows
        if int(row.get("total_scheduled_tokens", 0)) > 0
    ]
    if not active:
        return {
            "num_steps": 0,
            "decode_only_fraction": None,
            "median_token_slack": None,
            "p95_token_slack": None,
        }
    decode_only = [row for row in active if int(row.get("prefill_tokens", 0)) == 0]
    slack = [
        max_num_batched_tokens - int(row.get("total_scheduled_tokens", 0))
        for row in active
    ]
    return {
        "num_steps": len(active),
        "decode_only_steps": len(decode_only),
        "decode_only_fraction": len(decode_only) / len(active),
        "median_token_slack": median(slack),
        "p95_token_slack": percentile([float(v) for v in slack], 95),
        "mean_total_scheduled_tokens": sum(
            int(row.get("total_scheduled_tokens", 0)) for row in active
        )
        / len(active),
    }


def model_kv_shape(model: str) -> dict[str, int]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    num_attention_heads = int(getattr(cfg, "num_attention_heads"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", num_attention_heads))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = int(getattr(cfg, "hidden_size")) // num_attention_heads
    return {
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": int(head_dim),
    }


def kv_bytes_per_decode_token(
    *,
    model: str,
    context_len: int,
    dtype_bytes: int = 2,
) -> tuple[int, dict[str, int]]:
    shape = model_kv_shape(model)
    bytes_per_ctx_token = (
        shape["num_layers"]
        * shape["num_key_value_heads"]
        * shape["head_dim"]
        * 2
        * dtype_bytes
    )
    return bytes_per_ctx_token * context_len, shape


def analytic_utils(
    *,
    tokens_per_s: float | None,
    n_params: float,
    tensor_parallel_size: int,
    peak_flops_per_gpu_tflops: float | None,
    kv_bytes_per_token: int,
    hbm_bw_per_gpu_gbps: float | None,
) -> dict[str, float | None]:
    if tokens_per_s is None:
        return {"analytic_mfu_pct": None, "analytic_kv_bw_util_pct": None}
    mfu = None
    if peak_flops_per_gpu_tflops:
        peak = peak_flops_per_gpu_tflops * 1e12 * tensor_parallel_size
        mfu = tokens_per_s * 2.0 * n_params / peak * 100.0
    bw = None
    if hbm_bw_per_gpu_gbps:
        aggregate_bw = hbm_bw_per_gpu_gbps * 1e9 * tensor_parallel_size
        bw = tokens_per_s * kv_bytes_per_token / aggregate_bw * 100.0
    return {"analytic_mfu_pct": mfu, "analytic_kv_bw_util_pct": bw}
