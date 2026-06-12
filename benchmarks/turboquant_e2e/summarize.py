#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


KEY_METRICS = [
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
]

ALIASES = {
    "request_throughput": [
        "request_throughput",
        "requests_per_second",
        "requests_per_sec",
    ],
    "output_throughput": [
        "output_throughput",
        "output_token_throughput",
        "output_tokens_per_second",
        "output_tokens_per_sec",
    ],
    "total_token_throughput": [
        "total_token_throughput",
        "total_tokens_per_second",
        "total_tokens_per_sec",
    ],
    "mean_ttft_ms": ["mean_ttft_ms", "mean_ttft"],
    "median_ttft_ms": ["median_ttft_ms", "median_ttft"],
    "p99_ttft_ms": ["p99_ttft_ms", "p99_ttft"],
    "mean_tpot_ms": ["mean_tpot_ms", "mean_tpot"],
    "median_tpot_ms": ["median_tpot_ms", "median_tpot"],
    "p99_tpot_ms": ["p99_tpot_ms", "p99_tpot"],
    "mean_itl_ms": ["mean_itl_ms", "mean_itl"],
    "median_itl_ms": ["median_itl_ms", "median_itl"],
    "p99_itl_ms": ["p99_itl_ms", "p99_itl"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TurboQuant E2E benchmark results."
    )
    parser.add_argument("--results-dir", required=True, type=Path)
    return parser.parse_args()


def flatten_json(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_json(value, name))
        elif isinstance(value, list):
            if all(isinstance(item, (int, float)) for item in value):
                flat[f"{name}.count"] = len(value)
                if value:
                    flat[f"{name}.mean"] = sum(value) / len(value)
            else:
                flat[name] = json.dumps(value, sort_keys=True)
        else:
            flat[name] = value
    return flat


def read_json_or_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        return [payload] if isinstance(payload, dict) else []
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows


def metadata_by_result(
    results_dir: Path,
) -> tuple[dict[Path, dict[str, Any]], list[dict[str, Any]]]:
    by_result: dict[Path, dict[str, Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    for path in results_dir.rglob("*.metadata.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_metadata_path"] = str(path)
        metadata_rows.append(payload)
        result_path = payload.get("result_json")
        if result_path:
            by_result[Path(result_path).resolve()] = payload
    return by_result, metadata_rows


def result_json_paths(results_dir: Path) -> Iterable[Path]:
    for path in results_dir.rglob("*.json"):
        if path.name.endswith(".metadata.json"):
            continue
        if path.name.endswith(".pytorch.json"):
            continue
        yield path


def normalize_metric_aliases(row: dict[str, Any]) -> None:
    for canonical, aliases in ALIASES.items():
        if row.get(canonical) not in (None, ""):
            continue
        for alias in aliases:
            if alias in row and row[alias] not in (None, ""):
                row[canonical] = row[alias]
                break


def load_rows(results_dir: Path) -> pd.DataFrame:
    metadata_lookup, metadata_rows = metadata_by_result(results_dir)
    seen_metadata = set()
    rows: list[dict[str, Any]] = []

    for result_path in result_json_paths(results_dir):
        resolved = result_path.resolve()
        metadata = metadata_lookup.get(resolved, {})
        if metadata:
            seen_metadata.add(metadata.get("_metadata_path"))
        for payload in read_json_or_lines(result_path):
            row = dict(metadata)
            row.update(flatten_json(payload))
            row["result_path"] = str(result_path)
            row.setdefault("status", metadata.get("status", "success"))
            normalize_metric_aliases(row)
            rows.append(row)

    for metadata in metadata_rows:
        if metadata.get("_metadata_path") in seen_metadata:
            continue
        row = dict(metadata)
        row["result_path"] = metadata.get("result_json", "")
        normalize_metric_aliases(row)
        rows.append(row)

    return pd.DataFrame(rows)


def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    int_columns = [
        "concurrency",
        "repeat_index",
        "input_len",
        "output_len",
        "num_prompts",
    ]
    for column in int_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    for column in df.columns:
        if column in {"variant", "model", "status"}:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() > 0:
            df[column] = converted
    return df


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "concurrency",
        "repeat_index",
        "input_len",
        "output_len",
        "num_prompts",
        "port",
        "exit_code",
    }
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in skip and not column.startswith("_")
    ]


def status_series(df: pd.DataFrame) -> pd.Series:
    if "status" not in df:
        return pd.Series(["success"] * len(df), index=df.index)
    return df["status"].fillna("success")


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "variant" not in df or "concurrency" not in df:
        return pd.DataFrame()

    metric_columns = [metric for metric in KEY_METRICS if metric in df.columns]
    extra_numeric = [
        column
        for column in numeric_metric_columns(df)
        if column not in metric_columns
    ]
    metric_columns.extend(extra_numeric)

    statuses = status_series(df)
    successful = df[statuses == "success"].copy()
    if successful.empty:
        groups = df.groupby(["variant", "concurrency"], dropna=False)
        return groups.size().reset_index(name="run_count")

    agg = successful.groupby(["variant", "concurrency"], dropna=False)[
        metric_columns
    ].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    summary = agg.reset_index()

    run_counts = (
        df.groupby(["variant", "concurrency"], dropna=False)
        .size()
        .reset_index(name="run_count")
    )
    failed_counts = (
        df[statuses != "success"]
        .groupby(["variant", "concurrency"], dropna=False)
        .size()
        .reset_index(name="failed_count")
    )
    summary = summary.merge(run_counts, on=["variant", "concurrency"], how="outer")
    summary = summary.merge(failed_counts, on=["variant", "concurrency"], how="left")
    summary["failed_count"] = summary["failed_count"].fillna(0).astype(int)
    return summary.sort_values(["concurrency", "variant"]).reset_index(drop=True)


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if pd.isna(num) or pd.isna(den) or den == 0:
        return None
    return num / den


def get_mean(summary: pd.DataFrame, variant: str, concurrency: int, metric: str) -> Any:
    column = f"{metric}_mean"
    if column not in summary:
        return None
    row = summary[
        (summary["variant"] == variant)
        & (pd.to_numeric(summary["concurrency"], errors="coerce") == concurrency)
    ]
    if row.empty:
        return None
    return row.iloc[0][column]


def first_nonempty(df: pd.DataFrame, column: str) -> str:
    if column not in df:
        return "unknown"
    values = df[column].dropna().astype(str)
    values = values[values != ""]
    if values.empty:
        return "unknown"
    return values.iloc[0]


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_markdown(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines: list[str] = ["# TurboQuant E2E Serving Summary", ""]

    warnings: list[str] = []
    if df.empty:
        warnings.append("WARNING: No benchmark result or metadata rows were found.")
    elif "repeat_index" in df:
        repeats = pd.to_numeric(df["repeat_index"], errors="coerce").dropna().unique()
        if len(repeats) < 3:
            warnings.append("WARNING: Repeat count is less than 3.")

    statuses = status_series(df) if not df.empty else pd.Series(dtype=str)
    if not statuses.empty and (statuses != "success").any():
        failed = int((statuses != "success").sum())
        warnings.append(f"WARNING: {failed} condition run(s) failed.")

    missing_key_metrics = [
        metric
        for metric in ["output_throughput", "mean_tpot_ms", "mean_ttft_ms"]
        if metric not in df.columns
        or pd.to_numeric(df[metric], errors="coerce").isna().all()
    ]
    if missing_key_metrics:
        warnings.append(
            "WARNING: Missing key metric(s): " + ", ".join(missing_key_metrics) + "."
        )

    lines.extend(["## Environment", ""])
    env_rows = [
        ["git_commit", first_nonempty(df, "git_commit")],
        ["git_branch", first_nonempty(df, "git_branch")],
        ["vllm_version", first_nonempty(df, "vllm_version")],
        ["gpu_name", first_nonempty(df, "gpu_name")],
        ["cuda_version", first_nonempty(df, "cuda_version")],
        ["first_timestamp", first_nonempty(df, "timestamp")],
    ]
    lines.extend(markdown_table(["field", "value"], env_rows))
    lines.append("")

    lines.extend(["## Benchmark Config", ""])
    if not df.empty and "variant" in df:
        vanilla_model = first_nonempty(df[df["variant"] == "vanilla"], "model")
        turboquant_model = first_nonempty(
            df[df["variant"] == "turboquant"], "model"
        )
    else:
        vanilla_model = "unknown"
        turboquant_model = "unknown"

    if not df.empty:
        concurrency_series = pd.to_numeric(
            df.get("concurrency", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        concurrency_values = ", ".join(
            str(int(v)) for v in sorted(concurrency_series.unique())
        )
    else:
        concurrency_values = "unknown"

    config_rows = [
        ["vanilla_model", vanilla_model],
        ["turboquant_model", turboquant_model],
        ["concurrency_values", concurrency_values],
        ["input_len", first_nonempty(df, "input_len")],
        ["output_len", first_nonempty(df, "output_len")],
        ["num_prompts", first_nonempty(df, "num_prompts")],
        ["request_rate", first_nonempty(df, "request_rate")],
    ]
    lines.extend(markdown_table(["field", "value"], config_rows))
    lines.append("")

    lines.extend(["## Comparison", ""])
    comparison_rows: list[list[str]] = []
    if not summary.empty and "concurrency" in summary:
        concurrency_values = pd.to_numeric(
            summary["concurrency"], errors="coerce"
        ).dropna()
        concurrencies = sorted(int(v) for v in concurrency_values.unique())
        for concurrency in concurrencies:
            vanilla_output = get_mean(
                summary, "vanilla", concurrency, "output_throughput"
            )
            turbo_output = get_mean(
                summary, "turboquant", concurrency, "output_throughput"
            )
            vanilla_tpot = get_mean(summary, "vanilla", concurrency, "mean_tpot_ms")
            turbo_tpot = get_mean(summary, "turboquant", concurrency, "mean_tpot_ms")
            vanilla_ttft = get_mean(summary, "vanilla", concurrency, "mean_ttft_ms")
            turbo_ttft = get_mean(summary, "turboquant", concurrency, "mean_ttft_ms")
            comparison_rows.append(
                [
                    str(concurrency),
                    format_number(vanilla_output),
                    format_number(turbo_output),
                    format_number(ratio(turbo_output, vanilla_output)),
                    format_number(vanilla_tpot),
                    format_number(turbo_tpot),
                    format_number(ratio(turbo_tpot, vanilla_tpot)),
                    format_number(vanilla_ttft),
                    format_number(turbo_ttft),
                    format_number(ratio(turbo_ttft, vanilla_ttft)),
                ]
            )

    if comparison_rows:
        lines.extend(
            markdown_table(
                [
                    "concurrency",
                    "vanilla output tok/s",
                    "turbo output tok/s",
                    "output ratio",
                    "vanilla mean TPOT ms",
                    "turbo mean TPOT ms",
                    "TPOT ratio",
                    "vanilla mean TTFT ms",
                    "turbo mean TTFT ms",
                    "TTFT ratio",
                ],
                comparison_rows,
            )
        )
    else:
        lines.append("No successful comparison rows were available.")
    lines.append("")

    lines.extend(["## Conclusion", ""])
    c4_vanilla_output = get_mean(summary, "vanilla", 4, "output_throughput")
    c4_turbo_output = get_mean(summary, "turboquant", 4, "output_throughput")
    c4_vanilla_tpot = get_mean(summary, "vanilla", 4, "mean_tpot_ms")
    c4_turbo_tpot = get_mean(summary, "turboquant", 4, "mean_tpot_ms")

    has_output_pair = (
        c4_vanilla_output is not None
        and c4_turbo_output is not None
        and not pd.isna(c4_vanilla_output)
        and not pd.isna(c4_turbo_output)
    )
    has_tpot_pair = (
        c4_vanilla_tpot is not None
        and c4_turbo_tpot is not None
        and not pd.isna(c4_vanilla_tpot)
        and not pd.isna(c4_turbo_tpot)
    )

    output_slower = (
        c4_vanilla_output is not None
        and c4_turbo_output is not None
        and not pd.isna(c4_vanilla_output)
        and not pd.isna(c4_turbo_output)
        and float(c4_turbo_output) < float(c4_vanilla_output)
    )
    tpot_slower = (
        c4_vanilla_tpot is not None
        and c4_turbo_tpot is not None
        and not pd.isna(c4_vanilla_tpot)
        and not pd.isna(c4_turbo_tpot)
        and float(c4_turbo_tpot) > float(c4_vanilla_tpot)
    )

    if not has_output_pair and not has_tpot_pair:
        lines.append(
            "Conclusion: Unable to determine whether TurboQuant is slower than vanilla "
            "at concurrency=4 for this workload because required metrics are missing."
        )
    elif output_slower or tpot_slower:
        lines.append(
            "Conclusion: TurboQuant is slower than vanilla at "
            "concurrency=4 for this workload."
        )
    else:
        lines.append(
            "Conclusion: TurboQuant is not slower than vanilla at "
            "concurrency=4 for this workload."
        )
    lines.append("")

    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    df = coerce_columns(load_rows(results_dir))
    summary = make_summary(df)

    raw_csv = results_dir / "raw_results.csv"
    summary_csv = results_dir / "summary.csv"
    summary_md = results_dir / "summary.md"

    df.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(build_markdown(df, summary), encoding="utf-8")

    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
