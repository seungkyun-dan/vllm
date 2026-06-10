from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MAX_NUM_SEQS = 256


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_http(url: str, timeout_s: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error!r}")


def wait_for_server_http(
    url: str,
    *,
    proc: subprocess.Popen[str],
    log_path: Path,
    timeout_s: float = 900.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _tail_text(log_path, max_lines=80)
            raise RuntimeError(
                f"Server exited before {url} became ready "
                f"(returncode={proc.returncode}).\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error!r}")


def _tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return f"{path} does not exist."
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def build_vllm_serve_cmd(
    *,
    model: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    max_model_len: int,
    served_model_name: str,
    seed: int,
    speculative_config: dict[str, Any] | None = None,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens",
        str(DEFAULT_MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(DEFAULT_MAX_NUM_SEQS),
        "--seed",
        str(seed),
    ]
    if speculative_config is not None:
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])
    cmd.extend(extra_args or [])
    return cmd


def try_lock_gpu_clocks(clock_mhz: int | None) -> dict[str, Any] | None:
    if clock_mhz is None:
        return None
    cmd = ["nvidia-smi", "-lgc", str(clock_mhz)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@dataclass
class ManagedServer:
    cmd: list[str]
    env: dict[str, str]
    log_path: Path
    base_url: str
    ready_timeout_s: float = 900.0
    proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "ManagedServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("a", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server_http(
                f"{self.base_url}/health",
                proc=self.proc,
                log_path=self.log_path,
                timeout_s=self.ready_timeout_s,
            )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30.0)


def server_env(step_trace_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if step_trace_path is not None:
        env["VLLM_STEP_TRACE"] = str(step_trace_path)
        step_trace_path.parent.mkdir(parents=True, exist_ok=True)
    return env
