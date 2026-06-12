#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


METRICS = [
    ("output_throughput", "Output throughput (tokens/s)"),
    ("request_throughput", "Request throughput (requests/s)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TurboQuant E2E summaries.")
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_dir: Path) -> bool:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in df.columns:
        return False

    usable = df[["variant", "concurrency", mean_column]].dropna()
    if usable.empty:
        return False

    plt.figure(figsize=(7.5, 4.5))
    for variant in ["vanilla", "turboquant"]:
        subset = df[df["variant"] == variant].copy()
        subset["concurrency"] = pd.to_numeric(subset["concurrency"], errors="coerce")
        subset[mean_column] = pd.to_numeric(subset[mean_column], errors="coerce")
        subset = subset.dropna(subset=["concurrency", mean_column]).sort_values(
            "concurrency"
        )
        if subset.empty:
            continue
        if std_column in subset.columns:
            yerr = pd.to_numeric(subset[std_column], errors="coerce")
        else:
            yerr = None
        plt.errorbar(
            subset["concurrency"],
            subset[mean_column],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=variant,
        )

    plt.xlabel("Max concurrency")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs max concurrency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{metric}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        print("summary.csv is empty; no plots were created.")
        return

    created = 0
    for metric, ylabel in METRICS:
        if plot_metric(df, metric, ylabel, args.out_dir):
            created += 1

    print(f"Created {created} plot(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
