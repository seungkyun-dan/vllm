from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import aiohttp


JsonPrompt = str | list[int]


def parse_schedule(value: str) -> list[tuple[float, float]]:
    raw = json.loads(value)
    schedule: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict):
            duration = float(item["duration_s"])
            rate = float(item["rate"])
        else:
            duration = float(item[0])
            rate = float(item[1])
        if duration < 0 or rate < 0:
            raise ValueError(f"Invalid schedule item: {item!r}")
        schedule.append((duration, rate))
    return schedule


def arrival_offsets(
    schedule: Sequence[tuple[float, float]],
    *,
    arrival: str,
    gamma_shape: float,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    offsets: list[float] = []
    segment_start = 0.0
    for duration_s, rate in schedule:
        segment_end = segment_start + duration_s
        if rate <= 0.0:
            segment_start = segment_end
            continue
        t = segment_start
        while True:
            if arrival == "poisson":
                t += rng.expovariate(rate)
            elif arrival == "gamma":
                t += rng.gammavariate(gamma_shape, 1.0 / (rate * gamma_shape))
            else:
                raise ValueError(f"Unknown arrival process: {arrival}")
            if t >= segment_end:
                break
            offsets.append(t)
        segment_start = segment_end
    return offsets


async def run_completion(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    prompt: JsonPrompt,
    output_len: int,
    request_id: str,
    seed: int,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len if ignore_eos else 0,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "seed": seed,
        "stream": True,
        "return_token_ids": True,
        "request_id": request_id,
        "stream_options": {"include_usage": True},
    }
    send_ts = time.time()
    send_perf = time.perf_counter()
    token_times: list[float] = []
    prompt_token_ids: list[int] | None = None
    text_chunks = 0
    url = f"{base_url.rstrip('/')}/v1/completions"
    record: dict[str, Any] = {
        "request_id": request_id,
        "send_ts": send_ts,
        "ttft": None,
        "itl_list": [],
        "e2e": None,
        "in_len": len(prompt) if isinstance(prompt, list) else None,
        "out_len": 0,
        "status": "ok",
    }
    if metadata:
        record.update(metadata)
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                record["status"] = "error"
                record["error"] = await response.text()
                record["http_status"] = response.status
                record["e2e"] = time.perf_counter() - send_perf
                return record
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("prompt_token_ids") is not None:
                    prompt_token_ids = list(choice["prompt_token_ids"])
                token_ids = choice.get("token_ids") or []
                token_count = len(token_ids)
                if token_count == 0 and choice.get("text"):
                    token_count = 1
                    text_chunks += 1
                if token_count:
                    now = time.perf_counter()
                    token_times.extend([now] * token_count)
    except Exception as exc:  # noqa: BLE001 - no retries by design.
        record["status"] = "error"
        record["error"] = repr(exc)
    e2e = time.perf_counter() - send_perf
    if prompt_token_ids is not None:
        record["in_len"] = len(prompt_token_ids)
    if token_times:
        record["ttft"] = token_times[0] - send_perf
        record["itl_list"] = [
            token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
        ]
    record["e2e"] = e2e
    record["out_len"] = len(token_times)
    if text_chunks:
        record["text_chunk_fallbacks"] = text_chunks
    return record


async def run_open_loop(
    *,
    base_url: str,
    model: str,
    prompt: JsonPrompt,
    output_len: int,
    schedule: Sequence[tuple[float, float]],
    arrival: str,
    gamma_shape: float,
    seed: int,
    out: Path,
    request_seed_base: int,
) -> list[dict[str, Any]]:
    offsets = arrival_offsets(
        schedule, arrival=arrival, gamma_shape=gamma_shape, seed=seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=None)
    records: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def one(session: aiohttp.ClientSession, idx: int, offset: float) -> None:
        await asyncio.sleep(max(0.0, start + offset - time.perf_counter()))
        rec = await run_completion(
            session,
            base_url=base_url,
            model=model,
            prompt=prompt,
            output_len=output_len,
            request_id=f"open-loop-{idx}",
            seed=request_seed_base + idx,
            metadata={"arrival_offset_s": offset},
        )
        async with lock:
            records.append(rec)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        tasks = [
            asyncio.create_task(one(session, idx, offset))
            for idx, offset in enumerate(offsets)
        ]
        if tasks:
            await asyncio.gather(*tasks)
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_prompt(args: argparse.Namespace) -> JsonPrompt:
    if args.prompt_tokens_json:
        return list(json.loads(args.prompt_tokens_json))
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schedule-json", default="[[30, 2]]")
    parser.add_argument("--arrival", choices=["poisson", "gamma"], default="poisson")
    parser.add_argument("--gamma-shape", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-seed-base", type=int, default=1000)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-tokens-json")
    args = parser.parse_args()
    schedule = parse_schedule(args.schedule_json)
    prompt = _parse_prompt(args)
    asyncio.run(
        run_open_loop(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            output_len=args.output_len,
            schedule=schedule,
            arrival=args.arrival,
            gamma_shape=args.gamma_shape,
            seed=args.seed,
            out=args.out,
            request_seed_base=args.request_seed_base,
        )
    )


if __name__ == "__main__":
    main()
