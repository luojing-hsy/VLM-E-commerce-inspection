from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.common import load_yaml
from src.rewards.composite import compute_reward


@lru_cache(maxsize=None)
def _load_reward_config(path: str) -> dict[str, Any]:
    return load_yaml(Path(path))


def _ground_truth(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("veRL ground_truth must decode to a JSON object")
    return parsed


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    reward_config_path: str = "configs/grpo.yaml",
) -> dict[str, float]:
    """Adapt the project's rule reward to veRL's custom reward API."""
    if data_source != "vlm_product_audit":
        raise ValueError(f"unsupported data_source: {data_source}")
    result = compute_reward(_ground_truth(ground_truth), solution_str, _load_reward_config(reward_config_path))
    return {
        "score": float(result["reward"]),
        **{f"reward_{name}": float(value) for name, value in result["components"].items()},
        "evidence_gate_applied": float(result["evidence_gate_applied"]),
        "protocol_valid": float(result["protocol_valid"]),
    }
