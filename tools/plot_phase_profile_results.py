#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot Phase 2-6 profiling summaries with matplotlib."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(rows: list[dict[str, str]], *candidates: str) -> str | None:
    for candidate in candidates:
        if any(_num(row.get(candidate)) is not None for row in rows):
            return candidate
    return None


def _group_key(row: dict[str, str], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={row.get(key, '')}" for key in keys if row.get(key, "") != ""]
    return ", ".join(parts) or "all"


def _plot(
    rows: list[dict[str, str]],
    *,
    x_field: str,
    y_field: str,
    group_keys: tuple[str, ...],
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    series: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _num(row.get(x_field))
        y = _num(row.get(y_field))
        if x is None or y is None:
            continue
        series[_group_key(row, group_keys)].append((x, y))
    if not series:
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, points in sorted(series.items()):
        points = sorted(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(x_field)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_phase2(root: Path, plots_dir: Path) -> None:
    rows = _load_csv(root / "phase2_summary.csv")
    if not rows:
        return
    cadence = _metric(rows, "mean_tpot_ms", "mean_itl_ms")
    if cadence:
        _plot(
            rows,
            x_field="max_concurrency",
            y_field=cadence,
            group_keys=("case", "chunked_prefill"),
            title="Phase 2 TPOT/ITL vs concurrency",
            ylabel=cadence,
            out_path=plots_dir / "phase2_tpot_itl_vs_concurrency.png",
        )
    for field, filename in [
        ("mean_ttft_ms", "phase2_ttft_vs_concurrency.png"),
        ("mean_prefill_token_ratio", "phase2_prefill_ratio_vs_concurrency.png"),
        (
            "pure_decode_iteration_fraction",
            "phase2_pure_decode_fraction_vs_concurrency.png",
        ),
    ]:
        if _metric(rows, field):
            _plot(
                rows,
                x_field="max_concurrency",
                y_field=field,
                group_keys=("case", "chunked_prefill"),
                title=f"Phase 2 {field} vs concurrency",
                ylabel=field,
                out_path=plots_dir / filename,
            )


def plot_phase3(root: Path, plots_dir: Path) -> None:
    rows = _load_csv(root / "phase3_summary.csv")
    if not rows:
        return
    for field, filename in [
        (_metric(rows, "mean_tpot_ms", "mean_itl_ms"), "phase3_tpot_itl_vs_tb.png"),
        ("mean_ttft_ms", "phase3_ttft_vs_tb.png"),
        ("mean_prefill_token_ratio", "phase3_prefill_ratio_vs_tb.png"),
    ]:
        if field:
            _plot(
                rows,
                x_field="max_num_batched_tokens",
                y_field=field,
                group_keys=("max_concurrency", "chunked_prefill"),
                title=f"Phase 3 {field} vs token budget",
                ylabel=field,
                out_path=plots_dir / filename,
            )


def plot_phase4(root: Path, plots_dir: Path) -> None:
    rows = _load_csv(root / "phase4_summary.csv")
    if not rows:
        return
    for field, filename in [
        (_metric(rows, "mean_tpot_ms", "mean_itl_ms"), "phase4_tpot_itl_vs_input.png"),
        ("mean_prefill_token_ratio", "phase4_prefill_ratio_vs_input.png"),
        (
            "mean_prefill_residence_time_ms",
            "phase4_prefill_residence_vs_input.png",
        ),
    ]:
        if field:
            _plot(
                rows,
                x_field="input_len",
                y_field=field,
                group_keys=("max_num_batched_tokens", "max_concurrency"),
                title=f"Phase 4 {field} vs input_len",
                ylabel=field,
                out_path=plots_dir / filename,
            )


def plot_phase5(root: Path, plots_dir: Path) -> None:
    rows = _load_csv(root / "phase5_summary.csv")
    if not rows:
        return
    for field, filename in [
        ("mean_ttft_ms", "phase5_ttft_vs_output_by_k.png"),
        (
            _metric(rows, "mean_tpot_ms", "mean_itl_ms"),
            "phase5_tpot_itl_vs_output_by_k.png",
        ),
        (
            "mean_remaining_lifetime_after_first_output_ms",
            "phase5_remaining_lifetime_vs_output.png",
        ),
    ]:
        if field:
            _plot(
                rows,
                x_field="output_len",
                y_field=field,
                group_keys=("k", "max_num_batched_tokens"),
                title=f"Phase 5 {field} vs output_len",
                ylabel=field,
                out_path=plots_dir / filename,
            )


def plot_phase6(root: Path, plots_dir: Path) -> None:
    rows = _load_csv(root / "phase6_summary.csv")
    if not rows:
        return
    group_keys = ("case", "max_num_batched_tokens")
    for field, filename, title in [
        (
            "request_throughput",
            "phase6_throughput_vs_request_rate.png",
            "Phase 6 achieved throughput vs requested request_rate",
        ),
        (
            _metric(rows, "p95_ttft_ms", "mean_ttft_ms", "p95_trace_ttft_ms"),
            "phase6_ttft_vs_request_rate.png",
            "Phase 6 TTFT vs requested request_rate",
        ),
        (
            _metric(rows, "p95_tpot_ms", "p95_itl_ms", "mean_tpot_ms", "mean_itl_ms"),
            "phase6_tpot_itl_vs_request_rate.png",
            "Phase 6 TPOT/ITL vs requested request_rate",
        ),
        (
            _metric(rows, "p95_e2el_ms", "mean_e2el_ms"),
            "phase6_latency_vs_request_rate.png",
            "Phase 6 request latency vs requested request_rate",
        ),
        (
            "mean_prefill_token_ratio",
            "phase6_prefill_ratio_vs_request_rate.png",
            "Phase 6 prefill_token_ratio vs requested request_rate",
        ),
        (
            "p95_waiting_reqs",
            "phase6_waiting_reqs_vs_request_rate.png",
            "Phase 6 p95 waiting requests vs requested request_rate",
        ),
        (
            "pure_decode_iteration_fraction",
            "phase6_pure_decode_fraction_vs_request_rate.png",
            "Phase 6 pure decode fraction vs requested request_rate",
        ),
    ]:
        if field and _metric(rows, field):
            _plot(
                rows,
                x_field="requested_request_rate",
                y_field=field,
                group_keys=group_keys,
                title=title,
                ylabel=field,
                out_path=plots_dir / filename,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()
    plots_dir = args.out_root / "plots"
    plot_phase2(args.out_root, plots_dir)
    plot_phase3(args.out_root, plots_dir)
    plot_phase4(args.out_root, plots_dir)
    plot_phase5(args.out_root, plots_dir)
    plot_phase6(args.out_root, plots_dir)
    print(f"plots written under {plots_dir}")


if __name__ == "__main__":
    main()
