#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-variant comparison plots and tables for TurboQuant E2E results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


METRICS = [
    ("output_throughput", "Output throughput (tokens/s)", "higher is better"),
    ("request_throughput", "Request throughput (req/s)", "higher is better"),
    ("mean_tpot_ms", "Mean TPOT (ms)", "lower is better"),
    ("mean_ttft_ms", "Mean TTFT (ms)", "lower is better"),
    ("p99_ttft_ms", "P99 TTFT (ms)", "lower is better"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-dir", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Repeat for each TurboQuant variant, e.g. --variant k8v4=path/to/dir",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_rows(summary_csv: Path, variant_filter: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df[df["variant"] == variant_filter].copy()
    df["label"] = label
    df["concurrency"] = pd.to_numeric(df["concurrency"], errors="coerce")
    df = df.dropna(subset=["concurrency"]).sort_values("concurrency")
    return df


def plot_metric(combined: pd.DataFrame, metric: str, ylabel: str, hint: str,
                out_dir: Path) -> bool:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in combined.columns:
        return False

    plt.figure(figsize=(8.5, 5))
    for label, group in combined.groupby("label", sort=False):
        sub = group[["concurrency", mean_col]].copy()
        sub[mean_col] = pd.to_numeric(sub[mean_col], errors="coerce")
        sub = sub.dropna(subset=[mean_col])
        if sub.empty:
            continue
        yerr = None
        if std_col in group.columns:
            yerr = pd.to_numeric(
                group.loc[sub.index, std_col], errors="coerce"
            )
        plt.errorbar(
            sub["concurrency"],
            sub[mean_col],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    plt.xlabel("Max concurrency")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs concurrency  ({hint})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}.png", dpi=160)
    plt.close()
    return True


def plot_ratio(combined: pd.DataFrame, metric: str, ylabel: str,
               out_dir: Path) -> bool:
    mean_col = f"{metric}_mean"
    if mean_col not in combined.columns:
        return False

    vanilla = combined[combined["label"] == "vanilla"][
        ["concurrency", mean_col]
    ].rename(columns={mean_col: "vanilla"})
    if vanilla.empty:
        return False

    plt.figure(figsize=(8.5, 5))
    plotted = False
    for label, group in combined.groupby("label", sort=False):
        if label == "vanilla":
            continue
        merged = group.merge(vanilla, on="concurrency", how="inner")
        merged["ratio"] = pd.to_numeric(
            merged[mean_col], errors="coerce"
        ) / pd.to_numeric(merged["vanilla"], errors="coerce")
        merged = merged.dropna(subset=["ratio"])
        if merged.empty:
            continue
        plt.plot(
            merged["concurrency"], merged["ratio"], marker="o", linewidth=2,
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    plt.xlabel("Max concurrency")
    plt.ylabel(f"{ylabel} / vanilla")
    plt.title(f"{ylabel} ratio (turboquant / vanilla)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_ratio.png", dpi=160)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_rows(args.vanilla_dir / "summary.csv", "vanilla", "vanilla")]
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant must be LABEL=DIR, got: {spec}")
        label, path = spec.split("=", 1)
        frames.append(load_rows(Path(path) / "summary.csv", "turboquant", label))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out_dir / "combined_summary.csv", index=False)

    created = 0
    for metric, ylabel, hint in METRICS:
        if plot_metric(combined, metric, ylabel, hint, args.out_dir):
            created += 1
        if plot_ratio(combined, metric, ylabel, args.out_dir):
            created += 1

    # Compact comparison tables (per metric, rows = concurrency, cols = labels)
    tables = []
    for metric, ylabel, hint in METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in combined.columns:
            continue
        pivot = combined.pivot_table(
            index="concurrency",
            columns="label",
            values=mean_col,
            sort=False,
        )
        ordered = [c for c in ["vanilla"] + [
            l for l in combined["label"].unique() if l != "vanilla"
        ] if c in pivot.columns]
        pivot = pivot[ordered]
        tables.append(f"### {ylabel}  ({hint})\n\n" + pivot.round(2).to_markdown() + "\n")

    (args.out_dir / "comparison.md").write_text(
        "# TurboQuant variant comparison\n\n" + "\n".join(tables)
    )

    print(f"Wrote {created} plot(s) and comparison.md to {args.out_dir}")


if __name__ == "__main__":
    main()
