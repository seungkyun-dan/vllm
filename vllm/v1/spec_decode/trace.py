# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated JSONL trace helpers for serving phase profiling.

The trace is disabled unless VLLM_SPEC_TRACE or the legacy
VLLM_SPEC_TTFT_TRACE environment variable is set to a truthy value. When
enabled, records are appended to VLLM_SPEC_TRACE_FILE, falling back to
VLLM_SPEC_TTFT_TRACE_FILE and then to a process-local file under /tmp.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}

_ENABLED = (
    os.getenv("SPEC_TRACE", "").lower() in _TRUE_VALUES
    or os.getenv("SPEC_TTFT_TRACE", "").lower() in _TRUE_VALUES
    or os.getenv("VLLM_SPEC_TRACE", "").lower() in _TRUE_VALUES
    or os.getenv("VLLM_SPEC_TTFT_TRACE", "").lower() in _TRUE_VALUES
)

_TRACE_PATH = (
    os.getenv("SPEC_TRACE_FILE")
    or os.getenv("SPEC_TTFT_TRACE_FILE")
    or os.getenv("VLLM_SPEC_TRACE_FILE")
    or os.getenv("VLLM_SPEC_TTFT_TRACE_FILE")
    or f"/tmp/vllm_spec_trace_{os.getpid()}.jsonl"
)

_lock = threading.Lock()
_trace_file: Any | None = None

_first_prefill_seen: set[str] = set()
_last_prefill_seen: set[str] = set()
_first_decode_seen: set[str] = set()
_first_output_seen: set[str] = set()
_finished_seen: set[str] = set()


def enabled() -> bool:
    return _ENABLED


def _get_trace_file() -> Any | None:
    global _trace_file
    if not _ENABLED:
        return None
    if _trace_file is None:
        path = Path(_TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        _trace_file = path.open("a", encoding="utf-8", buffering=1)
    return _trace_file


def _close_trace_file() -> None:
    global _trace_file
    if _trace_file is not None:
        _trace_file.close()
        _trace_file = None


atexit.register(_close_trace_file)


def emit(event: str, **payload: Any) -> None:
    if not _ENABLED:
        return

    try:
        record = {
            "schema_version": 1,
            "event": event,
            "ts_ns": time.time_ns(),
            **payload,
        }
        line = json.dumps(record, default=str, separators=(",", ":"), sort_keys=True)

        with _lock:
            trace_file = _get_trace_file()
            if trace_file is not None:
                trace_file.write(line + "\n")
    except Exception:
        # Tracing is best-effort and must never affect serving.
        return


def emit_request_arrived(request: "Request") -> None:
    if not _ENABLED:
        return
    try:
        emit(
            "request_arrived",
            request_id=request.request_id,
            num_prompt_tokens=request.num_prompt_tokens,
            num_output_tokens=request.num_output_tokens,
        )
    except Exception:
        return


def emit_scheduled_batch_summary(
    *,
    iteration_id: int,
    scheduler_output: "SchedulerOutput",
    requests: dict[str, "Request"],
    num_waiting_reqs: int,
    num_running_reqs: int,
    max_num_batched_tokens: int,
    max_num_scheduled_tokens: int,
    token_budget_remaining: int,
    chunked_prefill_enabled: bool,
) -> None:
    if not _ENABLED:
        return

    try:
        _emit_scheduled_batch_summary_impl(
            iteration_id=iteration_id,
            scheduler_output=scheduler_output,
            requests=requests,
            num_waiting_reqs=num_waiting_reqs,
            num_running_reqs=num_running_reqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            token_budget_remaining=token_budget_remaining,
            chunked_prefill_enabled=chunked_prefill_enabled,
        )
    except Exception:
        return


def _emit_scheduled_batch_summary_impl(
    *,
    iteration_id: int,
    scheduler_output: "SchedulerOutput",
    requests: dict[str, "Request"],
    num_waiting_reqs: int,
    num_running_reqs: int,
    max_num_batched_tokens: int,
    max_num_scheduled_tokens: int,
    token_budget_remaining: int,
    chunked_prefill_enabled: bool,
) -> None:
    num_prefill_reqs = 0
    num_decode_reqs = 0
    num_prefill_tokens = 0
    num_decode_tokens = 0

    for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        request = requests.get(req_id)
        if request is None:
            continue

        computed_before = request.num_computed_tokens
        prompt_tokens = request.num_prompt_tokens
        prefill_tokens = max(
            0,
            min(computed_before + scheduled_tokens, prompt_tokens)
            - min(computed_before, prompt_tokens),
        )
        decode_tokens = max(0, scheduled_tokens - prefill_tokens)
        num_prefill_tokens += prefill_tokens
        num_decode_tokens += decode_tokens

        if prefill_tokens > 0:
            num_prefill_reqs += 1
            if req_id not in _first_prefill_seen:
                _first_prefill_seen.add(req_id)
                emit(
                    "request_first_prefill_scheduled",
                    request_id=req_id,
                    iteration_id=iteration_id,
                    num_prompt_tokens=prompt_tokens,
                    num_prefill_tokens_scheduled=prefill_tokens,
                )
            if (
                computed_before + prefill_tokens >= prompt_tokens
                and req_id not in _last_prefill_seen
            ):
                _last_prefill_seen.add(req_id)
                emit(
                    "request_last_prefill_scheduled",
                    request_id=req_id,
                    iteration_id=iteration_id,
                    num_prompt_tokens=prompt_tokens,
                    num_prefill_tokens_scheduled=prefill_tokens,
                )

        if decode_tokens > 0:
            num_decode_reqs += 1
            if req_id not in _first_decode_seen:
                _first_decode_seen.add(req_id)
                emit(
                    "request_first_decode_scheduled",
                    request_id=req_id,
                    iteration_id=iteration_id,
                    num_prompt_tokens=prompt_tokens,
                    num_decode_tokens_scheduled=decode_tokens,
                )

    num_scheduled_reqs = len(scheduler_output.num_scheduled_tokens)
    emit(
        "scheduled_batch_summary",
        iteration_id=iteration_id,
        num_waiting_reqs=num_waiting_reqs,
        num_running_reqs=num_running_reqs,
        num_scheduled_reqs=num_scheduled_reqs,
        num_prefill_reqs_scheduled=num_prefill_reqs,
        num_decode_reqs_scheduled=num_decode_reqs,
        num_prefill_tokens_scheduled=num_prefill_tokens,
        num_decode_tokens_scheduled=num_decode_tokens,
        num_total_target_tokens_scheduled=(
            scheduler_output.total_num_scheduled_tokens
        ),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        token_budget_used=scheduler_output.total_num_scheduled_tokens,
        token_budget_remaining=token_budget_remaining,
        mixed_batch=num_prefill_reqs > 0 and num_decode_reqs > 0,
        chunked_prefill_enabled=chunked_prefill_enabled,
    )


def emit_first_output_enqueued(
    request: "Request",
    *,
    iteration_id: int,
    num_new_token_ids: int,
) -> None:
    if not _ENABLED:
        return
    try:
        req_id = request.request_id
        if req_id in _first_output_seen:
            return
        _first_output_seen.add(req_id)
        emit(
            "first_output_enqueued",
            request_id=req_id,
            iteration_id=iteration_id,
            num_prompt_tokens=request.num_prompt_tokens,
            num_new_token_ids=num_new_token_ids,
            num_output_tokens=request.num_output_tokens,
        )
    except Exception:
        return


def emit_request_finished(request: "Request", *, iteration_id: int) -> None:
    if not _ENABLED:
        return
    try:
        req_id = request.request_id
        if req_id in _finished_seen:
            return
        _finished_seen.add(req_id)
        emit(
            "request_finished",
            request_id=req_id,
            iteration_id=iteration_id,
            num_prompt_tokens=request.num_prompt_tokens,
            num_output_tokens=request.num_output_tokens,
            finish_reason=request.get_finished_reason(),
        )
    except Exception:
        return
