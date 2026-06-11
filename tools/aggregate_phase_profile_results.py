#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate Phase 0-6 profiling run outputs into summary CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

CLOSED_LOOP_RUN_RE = re.compile(
    r"(?P<phase>phase\d)(?:_(?P<case>.+?))?_k(?P<k>\d+)"
    r"_in(?P<input_len>\d+)_out(?P<output_len>\d+)"
    r"_c(?P<max_concurrency>\d+)_tb(?P<max_num_batched_tokens>\d+)"
    r"_chunked_(?P<chunked_prefill>\w+)"
)
OPEN_LOOP_RUN_RE = re.compile(
    r"(?P<phase>phase\d)(?:_(?P<case>.+?))?_k(?P<k>\d+)"
    r"_in(?P<input_len>\d+)_out(?P<output_len>\d+)"
    r"_rr(?P<request_rate_name>[^_]+)_tb(?P<max_num_batched_tokens>\d+)"
    r"_chunked_(?P<chunked_prefill>\w+)"
)
RUN_RES = (CLOSED_LOOP_RUN_RE, OPEN_LOOP_RUN_RE)

BASE_FIELDS = [
    "phase",
    "case",
    "run_name",
    "k",
    "input_len",
    "output_len",
    "max_concurrency",
    "requested_request_rate",
    "request_rate_name",
    "max_concurrency_cap",
    "max_num_batched_tokens",
    "chunked_prefill",
    "num_prompts",
    "run_dir",
]

BENCH_FIELDS = [
    "duration",
    "completed",
    "failed",
    "timeout_count",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "max_concurrent_requests",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p50_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
]

TRACE_FIELDS = [
    "mean_prefill_token_ratio",
    "p50_prefill_token_ratio",
    "p95_prefill_token_ratio",
    "pure_decode_iteration_fraction",
    "decode_heavy_iteration_fraction",
    "mixed_iteration_fraction",
    "prefill_heavy_iteration_fraction",
    "mean_waiting_reqs",
    "p50_waiting_reqs",
    "p95_waiting_reqs",
    "max_waiting_reqs",
    "mean_prefill_residence_time_ms",
    "p95_prefill_residence_time_ms",
    "mean_time_to_first_decode_ms",
    "p95_time_to_first_decode_ms",
    "mean_trace_ttft_ms",
    "p95_trace_ttft_ms",
]

REMAINING_FIELDS = [
    "mean_remaining_lifetime_after_first_output_ms",
    "p50_remaining_lifetime_after_first_output_ms",
    "p95_remaining_lifetime_after_first_output_ms",
    "fraction_remaining_lifetime_below_10ms",
    "fraction_remaining_lifetime_below_25ms",
    "fraction_remaining_lifetime_below_50ms",
    "fraction_remaining_lifetime_below_100ms",
    "fraction_remaining_lifetime_below_250ms",
    "fraction_remaining_lifetime_below_500ms",
    "mean_remaining_decode_steps_after_first_output",
]

OPEN_LOOP_FIELDS = [
    "throughput_gap",
    "overload_indicator",
    "cadence_collapse_indicator",
]

PHASE_READMES = {
    "phase2": [
        "Whether short prompt + large token budget quickly becomes decode-heavy.",
        (
            "Whether long prompt + small token budget creates persistent "
            "prefill/mixed batches."
        ),
        "Whether TPOT/ITL tracks prefill_token_ratio and waiting request backlog.",
    ],
    "phase3": [
        "Does increasing max_num_batched_tokens reduce prefill fragmentation?",
        "Does vanilla TPOT/ITL improve as token budget increases?",
        "Does larger token budget hurt TTFT or throughput?",
        "Does chunked prefill on/off change the TTFT/TPOT trade-off?",
    ],
    "phase4": [
        "Does prefill pressure increase with input_len?",
        "Does TPOT/ITL increase with prefill pressure?",
        "Is the effect worse under smaller token budget?",
    ],
    "phase5": [
        "Do short outputs finish before speculation overhead can be amortized?",
        "Does speculative k improve TPOT but hurt TTFT?",
        "Does long output make speculative decoding more beneficial?",
        "Does small token budget amplify or reduce speculative benefit?",
    ],
    "phase6": [
        "Does achieved throughput track requested request_rate below capacity?",
        "Where do TTFT, TPOT/ITL, and request latency start growing sharply?",
        (
            "Does prefill_token_ratio and waiting backlog rise with cadence "
            "degradation?"
        ),
        (
            "Does long prompt + small token budget collapse earlier than short "
            "prompt + large token budget?"
        ),
    ],
}


def _num(value: Any) -> float | None:
    if value in (None, "", "inf"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: bool) -> str:
    return "true" if value else "false"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def parse_benchmark_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    mapping = {
        "Successful requests": "completed",
        "Failed requests": "failed",
        "Benchmark duration (s)": "duration",
        "Total input tokens": "total_input_tokens",
        "Total generated tokens": "total_output_tokens",
        "Request throughput (req/s)": "request_throughput",
        "Output token throughput (tok/s)": "output_throughput",
        "Total token throughput (tok/s)": "total_token_throughput",
        "Peak concurrent requests": "max_concurrent_requests",
        "Mean TTFT (ms)": "mean_ttft_ms",
        "Median TTFT (ms)": "median_ttft_ms",
        "P50 TTFT (ms)": "p50_ttft_ms",
        "P95 TTFT (ms)": "p95_ttft_ms",
        "P99 TTFT (ms)": "p99_ttft_ms",
        "Mean TPOT (ms)": "mean_tpot_ms",
        "Median TPOT (ms)": "median_tpot_ms",
        "P50 TPOT (ms)": "p50_tpot_ms",
        "P95 TPOT (ms)": "p95_tpot_ms",
        "P99 TPOT (ms)": "p99_tpot_ms",
        "Mean ITL (ms)": "mean_itl_ms",
        "Median ITL (ms)": "median_itl_ms",
        "P50 ITL (ms)": "p50_itl_ms",
        "P95 ITL (ms)": "p95_itl_ms",
        "P99 ITL (ms)": "p99_itl_ms",
        "Mean E2EL (ms)": "mean_e2el_ms",
        "Median E2EL (ms)": "median_e2el_ms",
        "P50 E2EL (ms)": "p50_e2el_ms",
        "P95 E2EL (ms)": "p95_e2el_ms",
        "P99 E2EL (ms)": "p99_e2el_ms",
    }
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = mapping.get(label.strip())
        if key is None:
            continue
        try:
            out[key] = float(value.strip().split()[0])
        except (IndexError, ValueError):
            pass
    return out


def _count_timeouts(result_json: dict[str, Any]) -> int | None:
    errors = result_json.get("errors")
    if not isinstance(errors, list):
        return None
    return sum(
        1
        for error in errors
        if error and "timeout" in str(error).lower()
    )


def _rate_from_name(rate_name: Any) -> float | None:
    if rate_name is None:
        return None
    return _num(str(rate_name).replace("p", "."))


def _parse_run_name(run_name: str) -> dict[str, Any]:
    for regex in RUN_RES:
        match = regex.search(run_name)
        if match:
            return match.groupdict()
    return {}


def parse_run_dir(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"run_name": run_dir.name, "run_dir": str(run_dir)}
    row.update(_parse_run_name(run_dir.name))
    row.update(load_json(run_dir / "metadata.json"))
    if row.get("requested_request_rate") is None:
        row["requested_request_rate"] = _rate_from_name(row.get("request_rate_name"))

    bench = parse_benchmark_log(run_dir / "benchmark.log")
    benchmark_json = load_json(run_dir / "benchmark_result.json")
    timeout_count = _count_timeouts(benchmark_json)
    bench.update(benchmark_json)
    if timeout_count is not None:
        bench["timeout_count"] = timeout_count
    for field in BENCH_FIELDS:
        if field in bench:
            row[field] = bench[field]

    for filename, source in [
        ("trace_summary.json", TRACE_FIELDS),
        ("remaining_lifetime_summary.json", REMAINING_FIELDS),
    ]:
        data = load_json(run_dir / filename)
        for key in source:
            if key in data:
                row[key] = data[key]
    return row


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(regex.search(path.name) for regex in RUN_RES)


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("phase*/*") if _is_run_dir(path))


def _cadence_value(row: dict[str, Any]) -> float | None:
    for field in [
        "p99_tpot_ms",
        "p99_itl_ms",
        "p95_tpot_ms",
        "p95_itl_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
    ]:
        value = _num(row.get(field))
        if value is not None:
            return value
    return None


def _phase6_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("case"),
        row.get("k"),
        row.get("input_len"),
        row.get("output_len"),
        row.get("max_num_batched_tokens"),
        row.get("chunked_prefill"),
    )


def annotate_phase6(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase") != "phase6":
            continue
        requested = _num(row.get("requested_request_rate"))
        achieved = _num(row.get("request_throughput"))
        if requested is not None and achieved is not None:
            row["throughput_gap"] = requested - achieved

        failed = _num(row.get("failed")) or 0.0
        timeouts = _num(row.get("timeout_count")) or 0.0
        waiting = _num(row.get("max_waiting_reqs")) or 0.0
        throughput_gap = _num(row.get("throughput_gap"))
        overload = failed > 0 or timeouts > 0
        if requested is not None and achieved is not None:
            overload = overload or achieved < requested * 0.9
        if waiting > (_num(row.get("max_num_seqs")) or 0.0):
            overload = overload or True
        if throughput_gap is not None and throughput_gap > 0:
            row["throughput_gap"] = throughput_gap
        row["overload_indicator"] = _bool(overload)
        groups.setdefault(_phase6_group_key(row), []).append(row)

    for group_rows in groups.values():
        ordered = sorted(
            group_rows,
            key=lambda row: _num(row.get("requested_request_rate")) or 0.0,
        )
        baseline_cadence: float | None = None
        baseline_waiting: float | None = None
        for row in ordered:
            cadence = _cadence_value(row)
            waiting = _num(row.get("max_waiting_reqs"))
            if baseline_cadence is None and cadence is not None:
                baseline_cadence = cadence
            if baseline_waiting is None and waiting is not None:
                baseline_waiting = waiting
            collapse = False
            if baseline_cadence and cadence is not None:
                collapse = collapse or cadence >= baseline_cadence * 2.0
            if baseline_waiting is not None and waiting is not None:
                collapse = collapse or (
                    waiting > 0 and waiting >= baseline_waiting * 2.0
                )
            row["cadence_collapse_indicator"] = _bool(collapse)
            if collapse:
                row["overload_indicator"] = _bool(True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        BASE_FIELDS
        + BENCH_FIELDS
        + TRACE_FIELDS
        + REMAINING_FIELDS
        + OPEN_LOOP_FIELDS
    )
    extra = sorted({key for row in rows for key in row} - set(fields))
    fieldnames = fields + extra
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_phase_readme(root: Path, phase: str, summary_path: Path) -> None:
    questions = PHASE_READMES.get(phase)
    if not questions:
        return
    phase_dir = root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {phase} Results Notes",
        "",
        "This file is generated by aggregate_phase_profile_results.py.",
        "It records what to inspect after the runs; it does not claim conclusions.",
        "",
        f"Summary CSV: {summary_path.name}",
        "",
        "Inspect:",
    ]
    lines.extend(f"{idx}. {question}" for idx, question in enumerate(questions, 1))
    (phase_dir / "README_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def aggregate(root: Path) -> list[dict[str, Any]]:
    rows = [parse_run_dir(path) for path in find_run_dirs(root)]
    annotate_phase6(rows)
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phase = str(row.get("phase") or "unknown")
        by_phase.setdefault(phase, []).append(row)

    for phase, phase_rows in sorted(by_phase.items()):
        if phase == "unknown":
            continue
        summary_path = root / f"{phase}_summary.csv"
        write_csv(summary_path, phase_rows)
        write_phase_readme(root, phase, summary_path)
    if rows:
        write_csv(root / "all_phases_summary.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()
    rows = aggregate(args.out_root)
    print(f"aggregated {len(rows)} runs under {args.out_root}")


if __name__ == "__main__":
    main()
