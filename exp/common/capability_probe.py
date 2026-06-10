from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any, get_args


def probe_capabilities() -> dict[str, Any]:
    result: dict[str, Any] = {
        "captured_ts": time.time(),
        "probe_mode": "static_import",
        "draft_model_supported": False,
        "draft_tp_asymmetry_supported": False,
        "errors": [],
    }
    try:
        import vllm
        from vllm.config.speculative import SpeculativeConfig, SpeculativeMethod

        result["vllm_version"] = getattr(vllm, "__version__", "unknown")
        methods = [str(value) for value in get_args(SpeculativeMethod)]
        result["speculative_methods"] = methods
        result["draft_model_supported"] = "draft_model" in methods
        config_source = inspect.getsource(SpeculativeConfig._verify_and_get_draft_tp)
        repo_root = Path(__file__).resolve().parents[2]
        proposer_source = (
            repo_root / "vllm" / "v1" / "spec_decode" / "draft_model.py"
        ).read_text(encoding="utf-8")
        config_allows_asymmetry = (
            "target_parallel_config.tensor_parallel_size" in config_source
            and "1," in config_source
        )
        runtime_requires_equal_tp = (
            "draft_tp != tgt_tp" in proposer_source
            and "raise ValueError" in proposer_source
        )
        result["draft_tp_config_allows_1_or_target_tp"] = config_allows_asymmetry
        result["draft_tp_runtime_requires_equal_tp"] = runtime_requires_equal_tp
        result["draft_tp_asymmetry_supported"] = (
            config_allows_asymmetry and not runtime_requires_equal_tp
        )
        result["draft_tp_supported_values"] = ["target_tp"]
        if result["draft_tp_asymmetry_supported"]:
            result["draft_tp_supported_values"].insert(0, 1)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(repr(exc))
    result["supports_integrated_draft_model_tp4_tp1"] = bool(
        result["draft_model_supported"] and result["draft_tp_asymmetry_supported"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/capabilities.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = probe_capabilities()
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
