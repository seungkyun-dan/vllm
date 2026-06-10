from __future__ import annotations

import csv
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any


def extract_nccl_kernel_share(report_path: Path) -> dict[str, Any]:
    if report_path.suffix != ".nsys-rep":
        report_path = report_path.with_suffix(".nsys-rep")
    result: dict[str, Any] = {
        "nsys_report": str(report_path),
        "nccl_kernel_time_pct": None,
        "nccl_kernel_time_ns": None,
        "total_kernel_time_ns": None,
        "errors": [],
    }
    if not report_path.exists():
        result["errors"].append("missing nsys report")
        return result
    cmd = ["nsys", "stats", "--report", "gpukernsum", "--format", "csv"]
    proc = subprocess.run(
        [*cmd, str(report_path)], check=False, capture_output=True, text=True
    )
    report_path.with_suffix(".gpukernsum.csv").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        result["errors"].append(proc.stderr.strip())
        return result
    total_ns, nccl_ns = _scan_gpukernsum(proc.stdout)
    result["total_kernel_time_ns"] = total_ns
    result["nccl_kernel_time_ns"] = nccl_ns
    if total_ns > 0:
        result["nccl_kernel_time_pct"] = nccl_ns / total_ns * 100.0
    return result


def _scan_gpukernsum(csv_text: str) -> tuple[float, float]:
    lines = [line for line in csv_text.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "name" in lowered and ("time" in lowered or "duration" in lowered):
            reader = csv.DictReader(StringIO("\n".join(lines[idx:])))
            total_ns = 0.0
            nccl_ns = 0.0
            for row in reader:
                name = _name_value(row).lower()
                time_ns = _time_ns(row)
                total_ns += time_ns
                if "nccl" in name:
                    nccl_ns += time_ns
            return total_ns, nccl_ns
    return 0.0, 0.0


def _name_value(row: dict[str, Any]) -> str:
    for key, value in row.items():
        if key and key.strip().lower() == "name":
            return str(value)
    return " ".join(str(value) for value in row.values())


def _time_ns(row: dict[str, Any]) -> float:
    for key, value in row.items():
        norm = (key or "").strip().lower()
        if "total time" not in norm and "duration" not in norm:
            continue
        try:
            numeric = float(str(value).replace(",", "").strip())
        except ValueError:
            continue
        if "(ns)" in norm:
            return numeric
        if "(us)" in norm:
            return numeric * 1000.0
        if "(ms)" in norm:
            return numeric * 1_000_000.0
        return numeric
    return 0.0
