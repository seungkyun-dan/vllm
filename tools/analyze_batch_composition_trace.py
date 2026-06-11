#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze VLLM_SPEC_TRACE JSONL batch-composition traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ITERATION_FIELDS = [
    "iteration_id",
    "ts_ns",
    "num_waiting_reqs",
    "num_running_reqs",
    "num_scheduled_reqs",
    "num_prefill_reqs_scheduled",
    "num_decode_reqs_scheduled",
    "num_prefill_tokens_scheduled",
    "num_decode_tokens_scheduled",
    "num_total_target_tokens_scheduled",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "token_budget_used",
    "token_budget_remaining",
    "mixed_batch",
    "chunked_prefill_enabled",
    "prefill_token_ratio",
    "prefill_req_ratio",
    "pure_decode_iteration",
    "decode_heavy_iteration",
    "mixed_iteration",
    "prefill_heavy_iteration",
]

REQUEST_FIELDS = [
    "request_id",
    "num_prompt_tokens",
    "request_arrived_ns",
    "first_prefill_scheduled_ns",
    "last_prefill_scheduled_ns",
    "first_decode_scheduled_ns",
    "first_output_enqueued_ns",
    "request_finished_ns",
    "first_prefill_iteration_id",
    "last_prefill_iteration_id",
    "first_decode_iteration_id",
    "first_output_iteration_id",
    "finish_iteration_id",
    "num_output_tokens_at_first_output",
    "num_output_tokens_finished",
    "finish_reason",
    "prefill_queue_delay_ms",
    "prefill_residence_time_ms",
    "time_to_first_decode_ms",
    "ttft_ms",
    "total_lifetime_ms",
]


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[int(rank)]
    weight = rank - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return None
    return statistics.fmean(clean)


def _ms_delta(start_ns: Any, end_ns: Any) -> float | None:
    start = _num(start_ns)
    end = _num(end_ns)
    if start is None or end is None:
        return None
    return (end - start) / 1_000_000.0


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as trace_file:
        for line_no, line in enumerate(trace_file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return events


def build_iteration_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "scheduled_batch_summary":
            continue
        total_tokens = _num(event.get("num_total_target_tokens_scheduled")) or 0.0
        total_reqs = _num(event.get("num_scheduled_reqs")) or 0.0
        prefill_tokens = _num(event.get("num_prefill_tokens_scheduled")) or 0.0
        decode_tokens = _num(event.get("num_decode_tokens_scheduled")) or 0.0
        prefill_reqs = _num(event.get("num_prefill_reqs_scheduled")) or 0.0

        prefill_token_ratio = (
            prefill_tokens / total_tokens if total_tokens > 0 else None
        )
        prefill_req_ratio = prefill_reqs / total_reqs if total_reqs > 0 else None
        scheduled = total_tokens > 0
        pure_decode = scheduled and prefill_tokens == 0 and decode_tokens > 0
        mixed = prefill_tokens > 0 and decode_tokens > 0
        prefill_heavy = scheduled and (prefill_token_ratio or 0.0) >= 0.5
        decode_heavy = scheduled and (prefill_token_ratio or 0.0) < 0.5

        row = {field: event.get(field, "") for field in ITERATION_FIELDS}
        row.update(
            {
                "prefill_token_ratio": prefill_token_ratio,
                "prefill_req_ratio": prefill_req_ratio,
                "pure_decode_iteration": pure_decode,
                "decode_heavy_iteration": decode_heavy,
                "mixed_iteration": mixed,
                "prefill_heavy_iteration": prefill_heavy,
            }
        )
        rows.append(row)
    return rows


def build_request_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    event_map = {
        "request_arrived": ("request_arrived_ns", None),
        "request_first_prefill_scheduled": (
            "first_prefill_scheduled_ns",
            "first_prefill_iteration_id",
        ),
        "request_last_prefill_scheduled": (
            "last_prefill_scheduled_ns",
            "last_prefill_iteration_id",
        ),
        "request_first_decode_scheduled": (
            "first_decode_scheduled_ns",
            "first_decode_iteration_id",
        ),
        "first_output_enqueued": (
            "first_output_enqueued_ns",
            "first_output_iteration_id",
        ),
        "request_finished": ("request_finished_ns", "finish_iteration_id"),
    }

    for event in events:
        req_id = event.get("request_id")
        if req_id is None or event.get("event") not in event_map:
            continue
        row = requests[str(req_id)]
        row["request_id"] = req_id
        if event.get("num_prompt_tokens") is not None:
            row.setdefault("num_prompt_tokens", event.get("num_prompt_tokens"))

        ts_field, iter_field = event_map[event["event"]]
        row.setdefault(ts_field, event.get("ts_ns"))
        if iter_field is not None:
            row.setdefault(iter_field, event.get("iteration_id"))

        if event["event"] == "first_output_enqueued":
            row.setdefault(
                "num_output_tokens_at_first_output",
                event.get("num_output_tokens"),
            )
        elif event["event"] == "request_finished":
            row.setdefault("num_output_tokens_finished", event.get("num_output_tokens"))
            row.setdefault("finish_reason", event.get("finish_reason"))

    rows = []
    for row in requests.values():
        row["prefill_queue_delay_ms"] = _ms_delta(
            row.get("request_arrived_ns"), row.get("first_prefill_scheduled_ns")
        )
        row["prefill_residence_time_ms"] = _ms_delta(
            row.get("first_prefill_scheduled_ns"),
            row.get("last_prefill_scheduled_ns"),
        )
        row["time_to_first_decode_ms"] = _ms_delta(
            row.get("request_arrived_ns"), row.get("first_decode_scheduled_ns")
        )
        row["ttft_ms"] = _ms_delta(
            row.get("request_arrived_ns"), row.get("first_output_enqueued_ns")
        )
        row["total_lifetime_ms"] = _ms_delta(
            row.get("request_arrived_ns"), row.get("request_finished_ns")
        )
        rows.append({field: row.get(field, "") for field in REQUEST_FIELDS})
    return sorted(rows, key=lambda r: str(r.get("request_id", "")))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _series(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [v for row in rows if (v := _num(row.get(field))) is not None]


def summarize(
    iteration_rows: list[dict[str, Any]], request_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    scheduled = [
        row for row in iteration_rows if (_num(row.get("token_budget_used")) or 0) > 0
    ]
    denom = len(scheduled) or 1

    summary: dict[str, Any] = {
        "num_iterations": len(iteration_rows),
        "num_scheduled_iterations": len(scheduled),
        "num_requests": len(request_rows),
    }
    for field, out_prefix in [
        ("prefill_token_ratio", "prefill_token_ratio"),
        ("num_waiting_reqs", "waiting_reqs"),
        ("prefill_residence_time_ms", "prefill_residence_time_ms"),
        ("time_to_first_decode_ms", "time_to_first_decode_ms"),
        ("ttft_ms", "trace_ttft_ms"),
    ]:
        rows = request_rows if field.endswith("_ms") else iteration_rows
        values = _series(rows, field)
        summary[f"mean_{out_prefix}"] = _mean(values)
        summary[f"p50_{out_prefix}"] = _percentile(values, 50)
        summary[f"p95_{out_prefix}"] = _percentile(values, 95)

    waiting_values = _series(iteration_rows, "num_waiting_reqs")
    summary["max_waiting_reqs"] = max(waiting_values) if waiting_values else None

    for field, out_name in [
        ("pure_decode_iteration", "pure_decode_iteration_fraction"),
        ("decode_heavy_iteration", "decode_heavy_iteration_fraction"),
        ("mixed_iteration", "mixed_iteration_fraction"),
        ("prefill_heavy_iteration", "prefill_heavy_iteration_fraction"),
    ]:
        summary[out_name] = sum(bool(row.get(field)) for row in scheduled) / denom
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = ["Batch composition trace summary", ""]
    for key in sorted(summary):
        value = _fmt(summary[key])
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = read_events(args.trace)
    iteration_rows = build_iteration_rows(events)
    request_rows = build_request_rows(events)
    summary = summarize(iteration_rows, request_rows)

    write_csv(args.out_dir / "iteration_metrics.csv", ITERATION_FIELDS, iteration_rows)
    write_csv(args.out_dir / "request_lifecycle.csv", REQUEST_FIELDS, request_rows)
    write_summary(args.out_dir / "trace_summary.txt", summary)
    (args.out_dir / "trace_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
