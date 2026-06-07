#!/usr/bin/env python
"""Extract vLLM NVTX forward ranges from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


EXECUTE_RE = re.compile(
    r"execute_context_(?P<context_reqs>\d+)\((?P<context_tokens>\d+)\)"
    r"_generation_(?P<generation_reqs>\d+)\((?P<generation_tokens>\d+)\)"
)

RANGE_NAME_TO_COMPONENT = {
    "gpu_model_runner: forward": "target_forward",
    "gpu_model_runner: draft": "draft_forward",
    "gpu_model_runner: sample": "sample",
    "gpu_model_runner: bookkeep": "bookkeep",
    "gpu_model_runner: preprocess": "preprocess",
    "gpu_model_runner: postprocess": "postprocess",
    "gpu_model_runner: ModelRunnerOutput": "model_runner_output",
    "schedule: allocate_slots": "schedule_allocate_slots",
    "schedule: get_num_common_prefix_blocks": "schedule_common_prefix",
    "schedule: make_cached_request_data": "schedule_cached_request_data",
    "schedule: update_after_schedule": "schedule_update_after_schedule",
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def pick_column(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for name in candidates:
        if name in available:
            return name
    lower_to_name = {name.lower(): name for name in available}
    for name in candidates:
        found = lower_to_name.get(name.lower())
        if found is not None:
            return found
    return None


def get_string_table(conn: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in table_names(conn):
        return {}
    cols = set(columns(conn, "StringIds"))
    id_col = pick_column(("id", "Id", "ID"), cols)
    value_col = pick_column(("value", "Value", "string", "String"), cols)
    if id_col is None or value_col is None:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in conn.execute(f"SELECT {id_col}, {value_col} FROM StringIds")
    }


def read_nvtx_events(conn: sqlite3.Connection) -> list[dict[str, object]]:
    names = table_names(conn)
    if "NVTX_EVENTS" not in names:
        raise RuntimeError(f"NVTX_EVENTS table not found. Tables: {sorted(names)}")

    cols = columns(conn, "NVTX_EVENTS")
    col_set = set(cols)
    start_col = pick_column(("start", "startTime", "Start"), col_set)
    end_col = pick_column(("end", "endTime", "End"), col_set)
    text_col = pick_column(("text", "Text", "message", "Message"), col_set)
    text_id_col = pick_column(("textId", "text_id", "TextId"), col_set)
    global_tid_col = pick_column(("globalTid", "globalTidId"), col_set)
    domain_col = pick_column(("domainId", "domain_id"), col_set)

    if start_col is None or end_col is None:
        raise RuntimeError(f"Could not identify NVTX start/end columns: {cols}")
    if text_col is None and text_id_col is None:
        raise RuntimeError(f"Could not identify NVTX text columns: {cols}")

    string_table = get_string_table(conn)
    wanted_cols = [start_col, end_col]
    if text_col is not None:
        wanted_cols.append(text_col)
    if text_id_col is not None:
        wanted_cols.append(text_id_col)
    if global_tid_col is not None:
        wanted_cols.append(global_tid_col)
    if domain_col is not None:
        wanted_cols.append(domain_col)

    query = f"SELECT {', '.join(wanted_cols)} FROM NVTX_EVENTS"
    events = []
    for row in conn.execute(query):
        values = dict(zip(wanted_cols, row, strict=True))
        start = values[start_col]
        end = values[end_col]
        if start is None or end is None:
            continue
        name = ""
        if text_col is not None and values.get(text_col) not in (None, ""):
            name = str(values[text_col])
        elif text_id_col is not None and values.get(text_id_col) is not None:
            name = string_table.get(int(values[text_id_col]), "")
        if not name:
            continue
        start_ns = int(start)
        end_ns = int(end)
        if end_ns <= start_ns:
            continue
        events.append(
            {
                "name": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
                "global_tid": values.get(global_tid_col) if global_tid_col else "",
                "domain_id": values.get(domain_col) if domain_col else "",
            }
        )
    events.sort(key=lambda event: (event["start_ns"], event["end_ns"]))
    return events


def classify_execute(name: str) -> dict[str, object]:
    match = EXECUTE_RE.fullmatch(name)
    if match is None:
        return {}
    context_reqs = int(match.group("context_reqs"))
    context_tokens = int(match.group("context_tokens"))
    generation_reqs = int(match.group("generation_reqs"))
    generation_tokens = int(match.group("generation_tokens"))
    if context_tokens and generation_tokens:
        phase = "mixed"
    elif context_tokens:
        phase = "prefill"
    elif generation_tokens:
        phase = "decode"
    else:
        phase = "empty"
    return {
        "phase": phase,
        "context_reqs": context_reqs,
        "context_tokens": context_tokens,
        "generation_reqs": generation_reqs,
        "generation_tokens": generation_tokens,
    }


def annotate_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    execute_ranges = [
        event | classify_execute(str(event["name"]))
        for event in events
        if EXECUTE_RE.fullmatch(str(event["name"]))
    ]
    execute_ranges.sort(key=lambda event: (event["start_ns"], event["end_ns"]))

    annotated = []
    for event in events:
        name = str(event["name"])
        component = RANGE_NAME_TO_COMPONENT.get(name)
        if component is None and not name.startswith("gpu_model_runner:"):
            continue
        row = {
            "name": name,
            "component": component or "other_gpu_model_runner",
            "start_ns": event["start_ns"],
            "end_ns": event["end_ns"],
            "duration_ns": event["duration_ns"],
            "duration_ms": int(event["duration_ns"]) / 1_000_000.0,
            "global_tid": event["global_tid"],
            "domain_id": event["domain_id"],
            "phase": "unknown",
            "context_reqs": "",
            "context_tokens": "",
            "generation_reqs": "",
            "generation_tokens": "",
        }
        row["phase_source"] = ""
        containing = [
            execute
            for execute in execute_ranges
            if execute["start_ns"] <= event["start_ns"]
            and event["end_ns"] <= execute["end_ns"]
        ]
        if containing:
            execute = min(
                containing,
                key=lambda item: int(item["end_ns"]) - int(item["start_ns"]),
            )
            row["phase_source"] = "enclosing_execute"
        else:
            previous = [
                execute
                for execute in execute_ranges
                if execute["end_ns"] <= event["start_ns"]
                and (
                    not execute.get("global_tid")
                    or not event.get("global_tid")
                    or execute.get("global_tid") == event.get("global_tid")
                )
            ]
            execute = max(previous, key=lambda item: int(item["end_ns"]), default=None)
            if (
                execute is not None
                and int(event["start_ns"]) - int(execute["end_ns"]) <= 5_000_000_000
            ):
                row["phase_source"] = "previous_execute"
            else:
                execute = None
        if execute is not None:
            for key in (
                "phase",
                "context_reqs",
                "context_tokens",
                "generation_reqs",
                "generation_tokens",
            ):
                row[key] = execute.get(key, "")
        annotated.append(row)
    return annotated


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["component"]), str(row["name"]), str(row["phase"]))
        grouped[key].append(float(row["duration_ms"]))

    summaries = []
    for (component, name, phase), values in sorted(grouped.items()):
        summaries.append(
            {
                "component": component,
                "name": name,
                "phase": phase,
                "count": len(values),
                "total_ms": sum(values),
                "mean_ms": mean(values),
                "median_ms": median(values),
                "p90_ms": percentile(values, 90),
                "p95_ms": percentile(values, 95),
                "p99_ms": percentile(values, 99),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    prefix = f"{args.prefix}_" if args.prefix else ""
    with sqlite3.connect(args.sqlite_path) as conn:
        events = read_nvtx_events(conn)

    rows = annotate_events(events)
    summary_rows = summarize(rows)

    event_fields = [
        "component",
        "name",
        "phase",
        "duration_ms",
        "duration_ns",
        "start_ns",
        "end_ns",
        "context_reqs",
        "context_tokens",
        "generation_reqs",
        "generation_tokens",
        "phase_source",
        "global_tid",
        "domain_id",
    ]
    summary_fields = [
        "component",
        "name",
        "phase",
        "count",
        "total_ms",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
    ]
    write_csv(args.out_dir / f"{prefix}nvtx_events.csv", rows, event_fields)
    write_csv(args.out_dir / f"{prefix}nvtx_summary.csv", summary_rows, summary_fields)

    print(f"Wrote {len(rows)} NVTX rows to {args.out_dir}")


if __name__ == "__main__":
    main()
