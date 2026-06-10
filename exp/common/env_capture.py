from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def _run(cmd: Sequence[str], timeout_s: float = 60.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "cmd": list(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # noqa: BLE001 - env capture should not fail runs.
        return {"cmd": list(cmd), "error": repr(exc)}


def _collect_python_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version}
    try:
        import vllm

        versions["vllm"] = getattr(vllm, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        versions["vllm_error"] = repr(exc)
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = getattr(torch.version, "cuda", None)
        versions["torch_git_version"] = getattr(torch.version, "git_version", None)
        versions["cuda_available"] = torch.cuda.is_available()
        versions["cuda_device_count"] = torch.cuda.device_count()
        versions["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # noqa: BLE001
        versions["torch_error"] = repr(exc)
    return versions


def collect_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,pstate,clocks.current.graphics,"
        "clocks.current.sm,clocks.current.memory,clocks.max.graphics,"
        "clocks.max.sm,clocks.max.memory",
        "--format=csv,noheader,nounits",
    ]
    data: dict[str, Any] = {
        "captured_ts": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "versions": _collect_python_versions(),
        "env": dict(sorted(os.environ.items())),
        "nvidia_smi_query": _run(query_cmd),
        "nvidia_smi_q_clocks": _run(["nvidia-smi", "-q", "-d", "CLOCK"]),
        "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"], 120.0),
    }
    if extra:
        data["extra"] = extra
    return data


def write_config(
    run_dir: Path | str,
    *,
    exp: str,
    run_id: str,
    cli_args: Sequence[str],
    env_overrides: dict[str, str] | None = None,
    server_args: Sequence[str] | None = None,
    client_args: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payload = collect_environment(
        {
            "exp": exp,
            "run_id": run_id,
            "cli_args": list(cli_args),
            "env_overrides": env_overrides or {},
            "server_args": list(server_args or []),
            "client_args": list(client_args or []),
            **(extra or {}),
        }
    )
    out = run_path / "config.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cli-json", default="[]")
    parser.add_argument("--extra-json", default="{}")
    args = parser.parse_args()
    cli_args = json.loads(args.cli_json)
    extra = json.loads(args.extra_json)
    payload = collect_environment({"exp": args.exp, "run_id": args.run_id, **extra})
    payload["extra"]["cli_args"] = cli_args
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
