from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(row: dict[str, Any], key: str) -> int | None:
    value = _float(row, key)
    return None if value is None else int(value)


def plot_offline(csv_path: Path, plots_dir: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "offline_plot_error.txt").write_text(repr(exc) + "\n")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        ("analytic_mfu_pct", "Analytic MFU (%)", "part_a_mfu_vs_B.png"),
        ("analytic_kv_bw_util_pct", "Analytic KV-BW util (%)", "part_a_bw_vs_B.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for L_ctx in sorted({_int(row, "L_ctx") for row in rows if _int(row, "L_ctx")}):
            values = [row for row in rows if _int(row, "L_ctx") == L_ctx]
            values = sorted(values, key=lambda row: _int(row, "B") or 0)
            xs = [_int(row, "B") for row in values]
            ys = [_float(row, metric) for row in values]
            if any(y is not None for y in ys):
                ax.plot(xs, ys, marker="o", label=f"L_ctx={L_ctx}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Decode batch size B")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)


def plot_serving(csv_path: Path, plots_dir: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "serving_plot_error.txt").write_text(repr(exc) + "\n")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        (
            "decode_only_fraction",
            "Decode-only step fraction",
            "part_b_decode_only_fraction_vs_rho.png",
        ),
        ("median_token_slack", "Median token slack", "part_b_slack_vs_rho.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        datasets = sorted({row.get("dataset", "unknown") for row in rows})
        for dataset in datasets:
            values = [row for row in rows if row.get("dataset") == dataset]
            values = sorted(values, key=lambda row: _float(row, "rho") or 0.0)
            xs = [_float(row, "rho") for row in values]
            ys = [_float(row, metric) for row in values]
            ax.plot(xs, ys, marker="o", label=dataset)
        ax.set_xlabel("rho = lambda / lambda_sat")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)


def write_bubble_verdict(run_dir: Path) -> dict[str, Any]:
    offline = _read_csv(run_dir / "part_a_offline.csv")
    serving = _read_csv(run_dir / "part_b_serving.csv")
    low_rho = [
        row
        for row in serving
        if (_float(row, "rho") is not None and (_float(row, "rho") or 0.0) <= 0.5)
        and _float(row, "decode_only_fraction") is not None
    ]
    low_rho_min = None
    if low_rho:
        low_rho_min = min(_float(row, "decode_only_fraction") or 0.0 for row in low_rho)
    rho_04 = [
        row
        for row in serving
        if _float(row, "rho") is not None
        and abs((_float(row, "rho") or 0.0) - 0.4) <= 0.05
        and _float(row, "decode_only_fraction") is not None
    ]
    kill = bool(rho_04) and all(
        (_float(row, "decode_only_fraction") or 0.0) < 0.15 for row in rho_04
    )
    sm_values = [
        _float(row, "sm_active_pct")
        for row in offline
        if _int(row, "L_ctx") == 8192 and _int(row, "B") in (8, 64)
    ]
    sm_values = [value for value in sm_values if value is not None]
    sm_active_max = max(sm_values) if sm_values else None
    verdict = {
        "low_rho_decode_only_fraction_min": low_rho_min,
        "nsys_sm_active_pct_max_B8_B64_L8K": sm_active_max,
        "kill_signal_decode_only_lt_15pct_at_rho_0p4": kill,
    }
    if low_rho_min is None:
        sentence = "Bubble verdict unavailable: no serving rows at rho <= 0.5."
    else:
        x_pct = low_rho_min * 100.0
        if sm_active_max is None:
            sentence = (
                f"At rho <= 0.5, decode-only steps are at least {x_pct:.1f}% "
                "of active steps; SM-active Nsight data is unavailable."
            )
        else:
            sentence = (
                f"At rho <= 0.5, decode-only steps are at least {x_pct:.1f}% "
                f"of active steps with profiled SM-active <= {sm_active_max:.1f}%."
            )
    verdict["sentence"] = sentence
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "bubble_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    )
    (plots_dir / "bubble_verdict.txt").write_text(sentence + "\n")
    return verdict


def analyze_run(run_dir: Path) -> dict[str, Any]:
    plots_dir = run_dir / "plots"
    plot_offline(run_dir / "part_a_offline.csv", plots_dir)
    plot_serving(run_dir / "part_b_serving.csv", plots_dir)
    return write_bubble_verdict(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verdict = analyze_run(args.run_dir)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
