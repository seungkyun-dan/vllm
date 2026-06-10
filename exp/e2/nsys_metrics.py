from __future__ import annotations

import csv
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any


def extract_nsys_gpu_metrics(report_path: Path) -> dict[str, Any]:
    """Best-effort extraction of SM Active and DRAM BW percentages.

    Nsight Systems report names vary by version. This helper keeps the raw CSV
    next to the report and scans for metric rows containing SM active or DRAM.
    """
    if report_path.suffix != ".nsys-rep":
        report_path = report_path.with_suffix(".nsys-rep")
    result: dict[str, Any] = {
        "nsys_report": str(report_path),
        "sm_active_pct": None,
        "dram_bw_pct": None,
        "errors": [],
    }
    if not report_path.exists():
        result["errors"].append("missing nsys report")
        return result
    for report in ("gpumetssum", "gpumets"):
        cmd = ["nsys", "stats", "--report", report, "--format", "csv", str(report_path)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        raw_path = report_path.with_suffix(f".{report}.csv")
        raw_path.write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            result["errors"].append(f"{report}: {proc.stderr.strip()}")
            continue
        _scan_metrics(proc.stdout, result)
    return result


def _scan_metrics(csv_text: str, result: dict[str, Any]) -> None:
    lines = [line for line in csv_text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.lower().startswith("metric") or "metric name" in line.lower():
            reader = csv.DictReader(StringIO("\n".join(lines[idx:])))
            for row in reader:
                joined = " ".join(str(v) for v in row.values()).lower()
                numeric = _first_number(row)
                if numeric is None:
                    continue
                if "sm active" in joined and result.get("sm_active_pct") is None:
                    result["sm_active_pct"] = numeric
                if (
                    "dram" in joined
                    and "%" in joined
                    and result.get("dram_bw_pct") is None
                ):
                    result["dram_bw_pct"] = numeric
            return


def _first_number(row: dict[str, Any]) -> float | None:
    for value in row.values():
        try:
            return float(str(value).strip().rstrip("%"))
        except ValueError:
            continue
    return None
