from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[low]
    return clean[low] + (clean[high] - clean[low]) * (rank - low)


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float | None],
    *,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 1,
) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None, None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(iters):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        stat = stat_fn(sample)
        if stat is not None and math.isfinite(stat):
            stats.append(stat)
    return percentile(stats, 100 * alpha / 2), percentile(stats, 100 * (1 - alpha / 2))


def tpot_seconds(row: dict[str, Any]) -> float | None:
    itl = row.get("itl_list") or []
    if itl:
        return mean(float(v) for v in itl)
    ttft = row.get("ttft")
    e2e = row.get("e2e")
    out_len = int(row.get("out_len") or 0)
    if ttft is None or e2e is None or out_len <= 1:
        return None
    return max(0.0, (float(e2e) - float(ttft)) / (out_len - 1))


def parsed_rows(
    records: Iterable[dict[str, Any]], warmup_s: float
) -> list[dict[str, Any]]:
    raw_rows = list(records)
    ok_send_times = [float(r["send_ts"]) for r in raw_rows if r.get("send_ts")]
    first_send = min(ok_send_times) if ok_send_times else 0.0
    rows = []
    for row in raw_rows:
        send_ts = row.get("send_ts")
        if send_ts is None or float(send_ts) < first_send + warmup_s:
            continue
        ttft = row.get("ttft")
        e2e = row.get("e2e")
        tpot = tpot_seconds(row)
        rows.append(
            {
                **row,
                "relative_send_s": float(send_ts) - first_send,
                "ttft_ms": None if ttft is None else float(ttft) * 1000.0,
                "tpot_ms": None if tpot is None else tpot * 1000.0,
                "e2e_ms": None if e2e is None else float(e2e) * 1000.0,
            }
        )
    return rows


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    duration_s: float | None,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("ttft_ms")]
    ttft = [float(r["ttft_ms"]) for r in ok_rows]
    tpot = [float(r["tpot_ms"]) for r in ok_rows if r.get("tpot_ms") is not None]
    e2e = [float(r["e2e_ms"]) for r in ok_rows if r.get("e2e_ms") is not None]
    good = [
        r
        for r in ok_rows
        if r.get("ttft_ms") is not None
        and r.get("tpot_ms") is not None
        and float(r["ttft_ms"]) <= ttft_slo_ms
        and float(r["tpot_ms"]) <= tpot_slo_ms
    ]
    sends = [float(r["send_ts"]) for r in ok_rows if r.get("send_ts")]
    measured_duration = duration_s
    if measured_duration is None and len(sends) >= 2:
        measured_duration = max(sends) - min(sends)
    if not measured_duration or measured_duration <= 0:
        measured_duration = None
    p99_ci = bootstrap_ci(
        ttft,
        lambda sample: percentile(sample, 99),
        iters=bootstrap_iters,
        seed=seed,
    )
    return {
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "ttft_ms_p50": percentile(ttft, 50),
        "ttft_ms_p95": percentile(ttft, 95),
        "ttft_ms_p99": percentile(ttft, 99),
        "ttft_ms_p99_ci_low": p99_ci[0],
        "ttft_ms_p99_ci_high": p99_ci[1],
        "tpot_ms_p50": percentile(tpot, 50),
        "tpot_ms_p95": percentile(tpot, 95),
        "e2e_ms_p50": percentile(e2e, 50),
        "e2e_ms_p95": percentile(e2e, 95),
        "goodput_fraction": len(good) / len(ok_rows) if ok_rows else None,
        "goodput_req_s": len(good) / measured_duration if measured_duration else None,
        "duration_s": measured_duration,
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
    }


def write_parsed_csv(rows: Sequence[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "request_id",
        "send_ts",
        "relative_send_s",
        "status",
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "in_len",
        "out_len",
        "error",
    ]
    extra = sorted({key for row in rows for key in row if key not in base_fields})
    fields = base_fields + extra
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parsed-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rows = parsed_rows(read_jsonl(args.input), args.warmup_s)
    write_parsed_csv(rows, args.parsed_out)
    summary = summarize(
        rows,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        duration_s=args.duration_s,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
