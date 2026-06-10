from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from exp.common.analyze import bootstrap_ci, percentile
from exp.e3.utils import OVERLAP_MODES, read_json, read_jsonl, write_csv


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(row: dict[str, Any], key: str) -> int | None:
    value = _float(row, key)
    return None if value is None else int(value)


def _finite(values: Sequence[float | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _target_client_stats(
    records_path: Path,
    *,
    warmup_itl_per_request: int,
) -> dict[str, Any]:
    records = read_jsonl(records_path)
    itl_ms: list[float] = []
    ok_records = [row for row in records if row.get("status") == "ok"]
    for row in ok_records:
        values = [
            float(value) * 1000.0
            for value in row.get("itl_list", [])
            if value is not None
        ]
        itl_ms.extend(values[warmup_itl_per_request:])
    return {
        "target_requests": len(records),
        "target_requests_ok": len(ok_records),
        "client_itl_samples": len(itl_ms),
        "client_itl_ms_p50": percentile(itl_ms, 50),
        "client_itl_ms_p99": percentile(itl_ms, 99),
    }


def _step_trace_stats(
    trace_path: Path,
    *,
    warmup_decode_steps: int,
    measure_steps: int | None,
) -> dict[str, Any]:
    rows = read_jsonl(trace_path)
    active = [
        row
        for row in rows
        if int(row.get("total_scheduled_tokens", 0) or 0) > 0
    ]
    decode = [
        row
        for row in active
        if int(row.get("prefill_tokens", 0) or 0) == 0
    ]
    first_decode_step = int(decode[0].get("step_id", -1)) if decode else None
    post_decode_prefill = 0
    if first_decode_step is not None:
        post_decode_prefill = sum(
            1
            for row in active
            if int(row.get("step_id", -1)) > first_decode_step
            and int(row.get("prefill_tokens", 0) or 0) > 0
        )
    selected = decode[warmup_decode_steps:]
    if measure_steps is not None:
        selected = selected[:measure_steps]
    durations_ms = [
        (float(selected[idx]["ts"]) - float(selected[idx - 1]["ts"])) * 1000.0
        for idx in range(1, len(selected))
    ]
    decode_fraction = len(decode) / len(active) if active else None
    return {
        "trace_active_steps": len(active),
        "trace_decode_only_steps": len(decode),
        "trace_decode_only_fraction": decode_fraction,
        "trace_measured_steps": len(selected),
        "post_decode_prefill_steps": post_decode_prefill,
        "decode_only_ok": post_decode_prefill == 0 and len(decode) > 0,
        "step_ms_p50": percentile(durations_ms, 50),
        "step_ms_p99": percentile(durations_ms, 99),
    }


def _target_stats(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    client = _target_client_stats(
        Path(manifest["target_records"]),
        warmup_itl_per_request=args.warmup_itl_per_request,
    )
    trace = _step_trace_stats(
        Path(manifest["target_trace"]),
        warmup_decode_steps=args.warmup_decode_steps,
        measure_steps=args.measure_steps,
    )
    p50 = trace.get("step_ms_p50") or client.get("client_itl_ms_p50")
    p99 = trace.get("step_ms_p99") or client.get("client_itl_ms_p99")
    return {
        **client,
        **trace,
        "target_iter_ms_p50": p50,
        "target_iter_ms_p99": p99,
    }


def _draft_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(Path(manifest["draft_summary"]))
    if summary.get("draft_tps") is not None:
        return summary
    rows = read_jsonl(Path(manifest["draft_records"]))
    elapsed = sum(float(row.get("elapsed_s", 0.0) or 0.0) for row in rows)
    tokens = sum(int(row.get("chunk_tokens", 0) or 0) for row in rows)
    summary["draft_tps"] = tokens / elapsed if elapsed > 0 else None
    summary["num_prefills"] = len(rows)
    return summary


def _serial_projection(
    *,
    baseline_iter_ms: float | None,
    chunk_tokens: int,
    draft_tps: float | None,
    draft_solo_tps: float | None,
) -> dict[str, Any]:
    if (
        baseline_iter_ms is None
        or draft_tps is None
        or draft_solo_tps is None
        or chunk_tokens <= 0
        or baseline_iter_ms <= 0
        or draft_solo_tps <= 0
    ):
        return {
            "chunks_per_iteration": None,
            "serial_extra_ms": None,
            "serial_iter_ms": None,
            "serial_inflation_pct": None,
        }
    baseline_iter_s = baseline_iter_ms / 1000.0
    baseline_iters_per_s = 1.0 / baseline_iter_s
    chunks_per_s = draft_tps / chunk_tokens
    chunks_per_iteration = chunks_per_s / baseline_iters_per_s
    serial_extra_ms = chunks_per_iteration * chunk_tokens / draft_solo_tps * 1000.0
    serial_iter_ms = baseline_iter_ms + serial_extra_ms
    return {
        "chunks_per_iteration": chunks_per_iteration,
        "serial_extra_ms": serial_extra_ms,
        "serial_iter_ms": serial_iter_ms,
        "serial_inflation_pct": serial_extra_ms / baseline_iter_ms * 100.0,
    }


def _aggregate(
    rows: Sequence[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("mode") not in (*OVERLAP_MODES, "m_serial"):
            continue
        source_mode = None
        if row.get("mode") == "m_serial":
            source_mode = None if row.get("source_mode") is None else str(
                row.get("source_mode")
            )
        key = (int(row.get("C_d", 0) or 0), str(row.get("mode")), source_mode)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (chunk_tokens, mode, source_mode), values in sorted(groups.items()):
        inflation = _finite([_float(row, "inflation_pct_vs_m0") for row in values])
        p50_vals = _finite([_float(row, "target_iter_ms_p50") for row in values])
        p99_vals = _finite([_float(row, "target_iter_ms_p99") for row in values])
        draft_tps = _finite([_float(row, "draft_tps") for row in values])
        ci = bootstrap_ci(
            inflation,
            lambda sample: percentile(sample, 50),
            iters=args.bootstrap_iters,
            seed=args.seed,
        )
        out.append(
            {
                "C_d": chunk_tokens,
                "mode": mode,
                "n": len(values),
                "target_iter_ms_p50": median(p50_vals) if p50_vals else None,
                "target_iter_ms_p99": median(p99_vals) if p99_vals else None,
                "inflation_pct_vs_m0": median(inflation) if inflation else None,
                "inflation_pct_ci_low": ci[0],
                "inflation_pct_ci_high": ci[1],
                "draft_tps": median(draft_tps) if draft_tps else None,
                "source_mode": source_mode,
            }
        )
    return out


def _kill_rule(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    chunks = sorted(
        {
            int(row.get("C_d", 0) or 0)
            for row in summary_rows
            if row.get("mode") in OVERLAP_MODES
        }
    )
    for chunk_tokens in chunks:
        overlap = [
            row
            for row in summary_rows
            if int(row.get("C_d", 0) or 0) == chunk_tokens
            and row.get("mode") in OVERLAP_MODES
        ]
        best: dict[str, Any] | None = None
        best_ratio: float | None = None
        for row in overlap:
            serial = _matched_serial(summary_rows, row)
            inflation = _float(row, "inflation_pct_vs_m0")
            serial_inflation = _float(serial, "inflation_pct_vs_m0") if serial else None
            if inflation is None or serial_inflation in (None, 0.0):
                continue
            ratio = max(0.0, inflation) / serial_inflation
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best = {
                    "C_d": chunk_tokens,
                    "best_mode": row.get("mode"),
                    "best_inflation_pct": inflation,
                    "serial_inflation_pct_at_best": serial_inflation,
                    "overlap_to_serial_ratio": ratio,
                    "kill_rule_pass": ratio < 0.5,
                }
        if best is None:
            best = {
                "C_d": chunk_tokens,
                "best_mode": None,
                "best_inflation_pct": None,
                "serial_inflation_pct_at_best": None,
                "overlap_to_serial_ratio": None,
                "kill_rule_pass": None,
            }
        out.append(best)
    return out


def _matched_serial(
    summary_rows: Sequence[dict[str, Any]], overlap_row: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in summary_rows
        if row.get("mode") == "m_serial"
        and int(row.get("C_d", 0) or 0) == int(overlap_row.get("C_d", 0) or 0)
        and row.get("source_mode") == overlap_row.get("mode")
    ]
    return candidates[0] if candidates else None


def _plot_pareto(summary_rows: Sequence[dict[str, Any]], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "pareto_plot_error.txt").write_text(repr(exc) + "\n")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for mode in OVERLAP_MODES:
        rows = [row for row in summary_rows if row.get("mode") == mode]
        rows = sorted(rows, key=lambda row: _float(row, "draft_tps") or 0.0)
        xs = [_float(row, "draft_tps") for row in rows]
        ys = [_float(row, "inflation_pct_vs_m0") for row in rows]
        if any(x is not None and y is not None for x, y in zip(xs, ys)):
            ax.plot(xs, ys, marker="o", label=mode)
    serial_rows = [row for row in summary_rows if row.get("mode") == "m_serial"]
    serial_rows = sorted(serial_rows, key=lambda row: _float(row, "draft_tps") or 0.0)
    xs = [_float(row, "draft_tps") for row in serial_rows]
    ys = [_float(row, "inflation_pct_vs_m0") for row in serial_rows]
    if any(x is not None and y is not None for x, y in zip(xs, ys)):
        ax.plot(xs, ys, color="black", linestyle="--", label="m_serial matched")
    ax.set_xlabel("Draft prefill throughput (tokens/s)")
    ax.set_ylabel("Target decode iteration inflation vs m0 (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "pareto_draft_tps_vs_target_inflation.png", dpi=160)
    plt.close(fig)


def analyze_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = run_dir / "raw" / "cell_manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    detailed: list[dict[str, Any]] = []
    baseline_iters = []
    for item in manifest:
        if item.get("kind") != "target" or item.get("mode") != "m0":
            continue
        stats = _target_stats(item, args)
        row = {**item, **stats, "draft_tps": None, "inflation_pct_vs_m0": 0.0}
        detailed.append(row)
        if row.get("target_iter_ms_p50") is not None:
            baseline_iters.append(float(row["target_iter_ms_p50"]))
    baseline_iter_ms = median(baseline_iters) if baseline_iters else None

    solo_by_chunk: dict[int, list[float]] = {}
    for item in manifest:
        if item.get("kind") != "draft_solo":
            continue
        draft = _draft_stats(item)
        row = {**item, **draft, "inflation_pct_vs_m0": None}
        detailed.append(row)
        tps = _float(row, "draft_tps")
        if tps is not None:
            solo_by_chunk.setdefault(int(item["C_d"]), []).append(tps)

    solo_median = {
        chunk: median(values) for chunk, values in solo_by_chunk.items() if values
    }
    for item in manifest:
        if item.get("kind") != "overlap":
            continue
        target = _target_stats(item, args)
        draft = _draft_stats(item)
        chunk_tokens = int(item["C_d"])
        draft_tps = _float(draft, "draft_tps")
        target_iter_ms = _float(target, "target_iter_ms_p50")
        inflation = None
        if baseline_iter_ms and target_iter_ms is not None:
            inflation = (target_iter_ms / baseline_iter_ms - 1.0) * 100.0
        serial = _serial_projection(
            baseline_iter_ms=baseline_iter_ms,
            chunk_tokens=chunk_tokens,
            draft_tps=draft_tps,
            draft_solo_tps=solo_median.get(chunk_tokens),
        )
        serial_inflation = serial.get("serial_inflation_pct")
        ratio = None
        if inflation is not None and serial_inflation not in (None, 0.0):
            ratio = max(0.0, inflation) / float(serial_inflation)
        row = {
            **item,
            **target,
            **draft,
            "baseline_iter_ms_p50": baseline_iter_ms,
            "draft_solo_prefill_tps": solo_median.get(chunk_tokens),
            "inflation_pct_vs_m0": inflation,
            "overlap_to_serial_ratio": ratio,
            **serial,
        }
        detailed.append(row)
        detailed.append(
            {
                "kind": "serial_projection",
                "mode": "m_serial",
                "source_mode": item.get("mode"),
                "rep": item.get("rep"),
                "C_d": chunk_tokens,
                "baseline_iter_ms_p50": baseline_iter_ms,
                "target_iter_ms_p50": serial.get("serial_iter_ms"),
                "target_iter_ms_p99": None,
                "inflation_pct_vs_m0": serial_inflation,
                "draft_tps": draft_tps,
                "draft_solo_prefill_tps": solo_median.get(chunk_tokens),
                **serial,
            }
        )

    summary = _aggregate(detailed, args)
    kill_rows = _kill_rule(summary)
    write_csv(run_dir / "parsed.csv", detailed)
    write_csv(run_dir / "summary.csv", summary)
    write_csv(run_dir / "kill_rule.csv", kill_rows)
    _plot_pareto(summary, run_dir / "plots")
    verdict = {
        "baseline_iter_ms_p50": baseline_iter_ms,
        "kill_rule": kill_rows,
    }
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots" / "kill_rule.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmup-itl-per-request", type=int, default=32)
    parser.add_argument("--warmup-decode-steps", type=int, default=32)
    parser.add_argument("--measure-steps", type=int, default=300)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    verdict = analyze_run(args.run_dir, args)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
