#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute remaining request lifetime after first output from trace CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

THRESHOLDS_MS = [10, 25, 50, 100, 250, 500]
OUTPUT_FIELDS = [
    "request_id",
    "ttft_ms",
    "total_lifetime_ms",
    "remaining_lifetime_after_first_output_ms",
    "num_output_tokens_at_first_output",
    "num_output_tokens_finished",
    "remaining_decode_steps_after_first_output",
]


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ns_delta_ms(start: Any, end: Any) -> float | None:
    start_num = _num(start)
    end_num = _num(end)
    if start_num is None or end_num is None:
        return None
    return (end_num - start_num) / 1_000_000.0


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.fmean(clean) if clean else None


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


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def compute(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        remaining_ms = _ns_delta_ms(
            row.get("first_output_enqueued_ns"), row.get("request_finished_ns")
        )
        total_ms = row.get("total_lifetime_ms") or _ns_delta_ms(
            row.get("request_arrived_ns"), row.get("request_finished_ns")
        )
        first_count = _num(row.get("num_output_tokens_at_first_output"))
        finish_count = _num(row.get("num_output_tokens_finished"))
        remaining_steps = None
        if first_count is not None and finish_count is not None:
            remaining_steps = max(0.0, finish_count - first_count)
        output_rows.append(
            {
                "request_id": row.get("request_id", ""),
                "ttft_ms": row.get("ttft_ms", ""),
                "total_lifetime_ms": total_ms,
                "remaining_lifetime_after_first_output_ms": remaining_ms,
                "num_output_tokens_at_first_output": row.get(
                    "num_output_tokens_at_first_output", ""
                ),
                "num_output_tokens_finished": row.get(
                    "num_output_tokens_finished", ""
                ),
                "remaining_decode_steps_after_first_output": remaining_steps,
            }
        )

    remaining_values = [
        value
        for row in output_rows
        if (value := _num(row["remaining_lifetime_after_first_output_ms"]))
        is not None
    ]
    total_values = [
        value
        for row in output_rows
        if (value := _num(row["total_lifetime_ms"])) is not None
    ]
    ttft_values = [
        value for row in output_rows if (value := _num(row["ttft_ms"])) is not None
    ]
    summary: dict[str, Any] = {
        "num_requests": len(output_rows),
        "num_requests_with_remaining_lifetime": len(remaining_values),
        "mean_remaining_lifetime_after_first_output_ms": _mean(remaining_values),
        "p50_remaining_lifetime_after_first_output_ms": _percentile(
            remaining_values, 50
        ),
        "p95_remaining_lifetime_after_first_output_ms": _percentile(
            remaining_values, 95
        ),
        "mean_total_lifetime_ms": _mean(total_values),
        "mean_ttft_ms": _mean(ttft_values),
    }
    denom = len(remaining_values) or 1
    for threshold in THRESHOLDS_MS:
        count = sum(value < threshold for value in remaining_values)
        summary[
            f"fraction_remaining_lifetime_below_{threshold}ms"
        ] = count / denom
    steps = [
        value
        for row in output_rows
        if (value := _num(row["remaining_decode_steps_after_first_output"]))
        is not None
    ]
    summary["mean_remaining_decode_steps_after_first_output"] = _mean(steps)
    return output_rows, summary


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "remaining_lifetime.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _fmt(row.get(field, "")) for field in OUTPUT_FIELDS}
            )

    (out_dir / "remaining_lifetime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["Remaining lifetime summary", ""]
    lines.extend(f"{key}: {_fmt(summary[key])}" for key in sorted(summary))
    (out_dir / "remaining_lifetime_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.request_csv)
    output_rows, summary = compute(rows)
    write_outputs(output_rows, summary, args.out_dir)


if __name__ == "__main__":
    main()
