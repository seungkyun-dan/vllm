from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


OVERLAP_MODES = ("m1", "m2", "m3")
MODE_CAPS = {
    "m1": None,
    "m2": "25",
    # MPS expects an integer percentage; 13 is the nearest usable proxy for 12.5.
    "m3": "13",
}


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def token_prompt(
    *,
    model: str,
    length: int,
    seed: int,
    trust_remote_code: bool = True,
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    text = (
        f"Seed {seed}. Measure decode interference with deterministic filler. "
        "The following content is intentionally repetitive but tokenized by the "
        "model tokenizer so every run uses exact prompt_token_ids. "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer for {model!r} produced an empty prompt")
    while len(ids) < length:
        ids.extend(ids[: max(1, length - len(ids))])
    return ids[:length]


def prompt_hash(prompt_token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(prompt_token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
