from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from exp.common.analyze import percentile, read_jsonl

SUMMARY_FIELDS = [
    "pair",
    "L_in",
    "tp_target",
    "tp_draft",
    "arm",
    "n",
    "ttft_ms_p50",
    "ttft_ms_p50_ci_low",
    "ttft_ms_p50_ci_high",
    "ttft_ms_p95",
    "draft_prefill_ms",
    "scheduler_overhead_ms",
    "inflation_pct",
    "inflation_pct_ci_low",
    "inflation_pct_ci_high",
    "arm2_vs_arm3_discrepancy_pct",
]


def _finite(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _ci(
    rows: Sequence[Any],
    stat: Callable[[Sequence[Any]], float | None],
    *,
    iters: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = stat(sample)
        if value is not None and math.isfinite(value):
            vals.append(value)
    return percentile(vals, 2.5), percentile(vals, 97.5)


def _ttft_ms(row: dict[str, Any]) -> float | None:
    ttft = row.get("ttft")
    return None if ttft is None else float(ttft) * 1000.0


def _draft_prefill_ms(row: dict[str, Any]) -> float | None:
    if row.get("draft_prefill_component_ms") is not None:
        return float(row["draft_prefill_component_ms"])
    return _ttft_ms(row)


def _sched_overhead_ms(row: dict[str, Any]) -> float:
    value = row.get("scheduler_overhead_ms")
    return 0.0 if value is None else float(value)


def _group_records(
    records: Sequence[dict[str, Any]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("status") != "ok" or row.get("ttft") is None:
            continue
        key = (
            row.get("pair"),
            int(row.get("L_in")),
            int(row.get("tp_target")),
            int(row.get("tp_draft")),
            row.get("arm"),
        )
        groups[key].append(row)
    return groups


def _paired_inflation(
    sample: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> float | None:
    arm = _finite([_ttft_ms(a) for a, _ in sample])
    target = _finite([_ttft_ms(t) for _, t in sample])
    arm_p50 = percentile(arm, 50)
    target_p50 = percentile(target, 50)
    if arm_p50 is None or not target_p50:
        return None
    return (arm_p50 - target_p50) / target_p50 * 100.0


def _summary_row(
    key: tuple[Any, ...],
    rows: Sequence[dict[str, Any]],
    target_rows: Sequence[dict[str, Any]] | None,
    *,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    ttfts = _finite([_ttft_ms(r) for r in rows])
    p50 = percentile(ttfts, 50)
    p95 = percentile(ttfts, 95)
    p50_ci = _ci(
        rows,
        lambda sample: percentile(_finite([_ttft_ms(r) for r in sample]), 50),
        iters=bootstrap_iters,
        seed=seed,
    )
    draft_prefill = percentile(_finite([_draft_prefill_ms(r) for r in rows]), 50)
    sched = percentile(_finite([_sched_overhead_ms(r) for r in rows]), 50)
    inflation = None
    inflation_ci = (None, None)
    if target_rows:
        target_ttfts = _finite([_ttft_ms(r) for r in target_rows])
        target_p50 = percentile(target_ttfts, 50)
        if target_p50 and p50 is not None:
            inflation = (p50 - target_p50) / target_p50 * 100.0
            paired = list(zip(rows, target_rows, strict=False))
            inflation_ci = _ci(
                paired,
                lambda sample: _paired_inflation(sample),
                iters=bootstrap_iters,
                seed=seed + 17,
            )
    pair, L_in, tp_target, tp_draft, arm = key
    return {
        "pair": pair,
        "L_in": L_in,
        "tp_target": tp_target,
        "tp_draft": tp_draft,
        "arm": arm,
        "n": len(rows),
        "ttft_ms_p50": p50,
        "ttft_ms_p50_ci_low": p50_ci[0],
        "ttft_ms_p50_ci_high": p50_ci[1],
        "ttft_ms_p95": p95,
        "draft_prefill_ms": draft_prefill if arm == "draft_prefill" else None,
        "scheduler_overhead_ms": sched,
        "inflation_pct": inflation,
        "inflation_pct_ci_low": inflation_ci[0],
        "inflation_pct_ci_high": inflation_ci[1],
        "arm2_vs_arm3_discrepancy_pct": None,
    }


def _synthetic_rows(
    target_rows: Sequence[dict[str, Any]],
    draft_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for idx, (target, draft) in enumerate(zip(target_rows, draft_rows, strict=False)):
        target_ttft = _ttft_ms(target)
        draft_prefill = _draft_prefill_ms(draft)
        if target_ttft is None or draft_prefill is None:
            continue
        overhead = _sched_overhead_ms(draft)
        rows.append(
            {
                **target,
                "request_id": f"synthetic-{idx}",
                "arm": "decomp_synthetic",
                "tp_draft": draft.get("tp_draft"),
                "ttft": (target_ttft + draft_prefill + overhead) / 1000.0,
                "draft_prefill_component_ms": draft_prefill,
                "scheduler_overhead_ms": overhead,
            }
        )
    return rows


def _add_discrepancy(rows: list[dict[str, Any]]) -> None:
    by_key = {(r["pair"], r["L_in"], r["tp_draft"], r["arm"]): r for r in rows}
    for row in rows:
        if row["arm"] not in ("integrated_spec", "decomp_synthetic"):
            continue
        other_arm = (
            "decomp_synthetic"
            if row["arm"] == "integrated_spec"
            else "integrated_spec"
        )
        other = by_key.get((row["pair"], row["L_in"], row["tp_draft"], other_arm))
        if not other:
            continue
        base = other.get("ttft_ms_p50")
        current = row.get("ttft_ms_p50")
        if base and current is not None:
            row["arm2_vs_arm3_discrepancy_pct"] = (current - base) / base * 100.0


def analyze_run(
    run_dir: Path,
    *,
    bootstrap_iters: int = 1000,
    seed: int = 1,
) -> list[dict[str, Any]]:
    records = read_jsonl(run_dir / "raw" / "records.jsonl")
    groups = _group_records(records)
    summaries: list[dict[str, Any]] = []
    target_by_pair_len: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for key, rows in groups.items():
        pair, L_in, tp_target, _tp_draft, arm = key
        if arm == "target_only":
            target_by_pair_len[(pair, L_in, tp_target)] = rows

    for key, rows in sorted(groups.items()):
        pair, L_in, tp_target, _tp_draft, arm = key
        target_rows = None if arm in ("target_only", "draft_prefill") else (
            target_by_pair_len.get((pair, L_in, tp_target))
        )
        summaries.append(
            _summary_row(
                key,
                rows,
                target_rows,
                bootstrap_iters=bootstrap_iters,
                seed=seed,
            )
        )

    for key, draft_rows in groups.items():
        pair, L_in, tp_target, tp_draft, arm = key
        if arm != "draft_prefill":
            continue
        target_rows = target_by_pair_len.get((pair, L_in, tp_target))
        if not target_rows:
            continue
        synth = _synthetic_rows(target_rows, draft_rows)
        synth_key = (pair, L_in, tp_target, tp_draft, "decomp_synthetic")
        summaries.append(
            _summary_row(
                synth_key,
                synth,
                target_rows,
                bootstrap_iters=bootstrap_iters,
                seed=seed + 101,
            )
        )
    _add_discrepancy(summaries)
    out = run_dir / "parsed.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    _write_plot(run_dir, summaries)
    _write_decision(run_dir, summaries)
    return summaries


def _write_plot(run_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    plot_rows = [
        r
        for r in rows
        if r.get("arm") in ("integrated_spec", "decomp_synthetic")
        and r.get("inflation_pct") is not None
    ]
    if not plot_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        (run_dir / "plots" / "plot_error.txt").write_text(repr(exc) + "\n")
        return
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    series: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in plot_rows:
        series[(row["pair"], row["tp_draft"], row["arm"])].append(row)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, values in sorted(series.items()):
        values = sorted(values, key=lambda item: int(item["L_in"]))
        label = f"{key[0]} draftTP{key[1]} {key[2]}"
        ax.plot(
            [int(v["L_in"]) for v in values],
            [float(v["inflation_pct"]) for v in values],
            marker="o",
            label=label,
        )
    ax.axhline(5.0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(12.0, color="red", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("TTFT inflation (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "inflation.png", dpi=160)
    plt.close(fig)


def _write_decision(run_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    inflation_rows = [
        r
        for r in rows
        if r.get("arm") in ("integrated_spec", "decomp_synthetic")
        and r.get("inflation_pct") is not None
    ]
    inflations = [float(r["inflation_pct"]) for r in inflation_rows]
    kill = bool(inflations) and max(inflations) < 5.0
    go = any(
        int(r["tp_draft"]) == 1
        and int(r["L_in"]) >= 16_384
        and float(r["inflation_pct"]) > 12.0
        for r in inflation_rows
    )
    payload = {
        "motivation_kill_signal": kill,
        "tp4_tp1_long_context_go_signal": go,
        "max_inflation_pct": max(inflations) if inflations else None,
    }
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots" / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rows = analyze_run(
        args.run_dir, bootstrap_iters=args.bootstrap_iters, seed=args.seed
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
