from __future__ import annotations

from typing import Any

from src.rewards.action_cost import action_reward
from src.rewards.evidence_reward import evidence_reward
from src.rewards.parser import tolerant_parse
from src.rewards.type_reward import violation_type_reward
from src.rewards.value_reward import value_reward

DEFAULT_WEIGHTS = {"action": 0.40, "type": 0.15, "evidence": 0.25, "value": 0.20}


def compute_reward(sample: dict[str, Any], prediction: str | dict, config: dict[str, Any]) -> dict[str, Any]:
    parsed = tolerant_parse(prediction)
    data = parsed.data
    weights = config.get("reward_weights", DEFAULT_WEIGHTS)
    has_evidence = bool(sample.get("evidence"))
    has_values = sample.get("listed_value") is not None or sample.get("observed_value") is not None
    components = {
        "action": (float(weights["action"]), action_reward(sample["decision"], data.get("decision"), config["cost_matrix"]), True),
        "type": (float(weights["type"]), violation_type_reward(sample.get("violation_type"), data.get("violation_type")), True),
        "evidence": (float(weights["evidence"]), evidence_reward(sample, data), has_evidence),
        "value": (float(weights["value"]), value_reward(sample, data), has_values),
    }
    numerator = sum(weight * score for weight, score, enabled in components.values() if enabled)
    denominator = sum(weight for weight, _, enabled in components.values() if enabled)
    total = numerator / denominator
    gate_threshold = float(config.get("evidence_gate_threshold", 0.3))
    gate_applied = has_evidence and components["evidence"][1] < gate_threshold
    if gate_applied:
        total = min(total, float(config.get("evidence_gate_cap", 0.45)))
    return {
        "reward": total,
        "components": {name: score for name, (_, score, _) in components.items()},
        "masks": {name: enabled for name, (_, _, enabled) in components.items()},
        "evidence_gate_applied": gate_applied,
        "protocol_valid": parsed.protocol_valid,
        "parse_error": parsed.error,
    }

