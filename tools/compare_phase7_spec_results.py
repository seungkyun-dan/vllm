#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Phase 7 fixed-k speculative runs against k=0 baselines."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COMPARE_FIELDS = [
    "case",
    "input_len",
    "output_len",
    "max_concurrency",
    "max_num_batched_tokens",
    "chunked_prefill",
    "k",
    "baseline_run_name",
    "run_name",
    "delta_TTFT_mean",
    "delta_TTFT_p95",
    "delta_TPOT_or_ITL_mean",
    "delta_TPOT_or_ITL_p95",
    "delta_request_latency_mean",
    "delta_throughput",
    "delta_output_token_throughput",
    "delta_prefill_token_ratio",
    "classification",
]

GROUP_FIELDS = [
    "case",
    "input_len",
    "output_len",
    "max_concurrency",
    "max_num_batched_tokens",
    "chunked_prefill",
]


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(row: dict[str, str], baseline: dict[str, str], field: str) -> float | None:
    value = _num(row.get(field))
    base = _num(baseline.get(field))
    if value is None or base is None:
        return None
    return value - base


def _first_metric(row: dict[str, str], *fields: str) -> tuple[str | None, float | None]:
    for field in fields:
        value = _num(row.get(field))
        if value is not None:
            return field, value
    return None, None


def _delta_first(
    row: dict[str, str], baseline: dict[str, str], *fields: str
) -> float | None:
    for field in fields:
        value = _num(row.get(field))
        base = _num(baseline.get(field))
        if value is not None and base is not None:
            return value - base
    return None


def _classify(
    delta_ttft: float | None,
    delta_cadence: float | None,
) -> str:
    if delta_ttft is None or delta_cadence is None:
        return "inconclusive_missing_metrics"
    ttft_better = delta_ttft < 0
    cadence_better = delta_cadence < 0
    if cadence_better and not ttft_better:
        return "TPOT_better_TTFT_worse"
    if cadence_better and ttft_better:
        return "both_better"
    if not cadence_better and not ttft_better:
        return "both_worse"
    return "TPOT_worse_TTFT_better"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def compare(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "phase7":
            continue
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)

    comparisons: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        baselines = [row for row in group_rows if str(row.get("k")) == "0"]
        if not baselines:
            continue
        baseline = baselines[0]
        for row in sorted(group_rows, key=lambda item: _num(item.get("k")) or 0):
            if str(row.get("k")) == "0":
                continue
            delta_ttft_mean = _delta(row, baseline, "mean_ttft_ms")
            delta_ttft_p95 = _delta_first(
                row, baseline, "p95_ttft_ms", "p95_trace_ttft_ms"
            )
            delta_cadence_mean = _delta_first(
                row, baseline, "mean_tpot_ms", "mean_itl_ms"
            )
            delta_cadence_p95 = _delta_first(
                row, baseline, "p95_tpot_ms", "p95_itl_ms"
            )
            comparison = {field: value for field, value in zip(GROUP_FIELDS, key)}
            comparison.update(
                {
                    "k": row.get("k", ""),
                    "baseline_run_name": baseline.get("run_name", ""),
                    "run_name": row.get("run_name", ""),
                    "delta_TTFT_mean": delta_ttft_mean,
                    "delta_TTFT_p95": delta_ttft_p95,
                    "delta_TPOT_or_ITL_mean": delta_cadence_mean,
                    "delta_TPOT_or_ITL_p95": delta_cadence_p95,
                    "delta_request_latency_mean": _delta_first(
                        row, baseline, "mean_e2el_ms"
                    ),
                    "delta_throughput": _delta(
                        row, baseline, "request_throughput"
                    ),
                    "delta_output_token_throughput": _delta(
                        row, baseline, "output_throughput"
                    ),
                    "delta_prefill_token_ratio": _delta(
                        row, baseline, "mean_prefill_token_ratio"
                    ),
                    "classification": _classify(
                        delta_ttft_mean, delta_cadence_mean
                    ),
                }
            )
            comparisons.append(comparison)
    return comparisons


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COMPARE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _fmt(row.get(field, "")) for field in COMPARE_FIELDS}
            )


def write_report(path: Path, rows: list[dict[str, Any]], summary_path: Path) -> None:
    counts = Counter(str(row.get("classification", "")) for row in rows)
    lines = [
        "# Phase 7 Speculative Decoding Comparisons",
        "",
        "This report compares each k>0 run against the matching k=0 baseline.",
        (
            "It summarizes observed trade-off categories and does not "
            "make research claims."
        ),
        "",
        f"Raw comparison CSV: `{summary_path.name}`",
        "",
        "## Classification Counts",
        "",
    ]
    if counts:
        lines.extend(f"- `{name}`: {count}" for name, count in sorted(counts.items()))
    else:
        lines.append("- No comparable k>0 rows were found.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Negative TTFT or TPOT/ITL deltas mean the k>0 run was lower than k=0.",
            "- Missing benchmark metrics produce `inconclusive_missing_metrics`.",
            (
                "- Inspect `phase7_summary.csv` and per-run traces "
                "before drawing conclusions."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase7_summary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.phase7_summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = compare(read_rows(args.phase7_summary))
    comparison_csv = out_dir / "phase7_comparisons.csv"
    write_csv(comparison_csv, rows)
    write_report(out_dir / "phase7_comparison_report.md", rows, comparison_csv)
    print(f"wrote {len(rows)} comparisons to {comparison_csv}")


if __name__ == "__main__":
    main()
