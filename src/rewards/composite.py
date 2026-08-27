from __future__ import annotations

from typing import Any

from src.rewards.action_cost import action_reward
from src.rewards.evidence_reward import evidence_reward
from src.rewards.parser import tolerant_parse
from src.rewards.type_reward import violation_type_reward

DEFAULT_WEIGHTS = {"action": 0.40, "type": 0.20, "evidence": 0.25, "subtype": 0.15}


def compute_reward(sample: dict[str, Any], prediction: str | dict, config: dict[str, Any]) -> dict[str, Any]:
    parsed = tolerant_parse(prediction)
    data = parsed.data
    protocol_data = data if parsed.protocol_valid else {}
    weights = config.get("reward_weights", DEFAULT_WEIGHTS)
    has_evidence = sample.get("violation_type") in {"image_quality", "wrong_image"}
    has_subtype = sample.get("violation_type") == "image_quality"
    components = {
        "action": (float(weights["action"]), action_reward(sample["decision"], data.get("decision"), config["cost_matrix"]), True),
        "type": (float(weights["type"]), violation_type_reward(sample.get("violation_type"), protocol_data.get("violation_type")), True),
        "evidence": (float(weights["evidence"]), evidence_reward(sample, protocol_data), has_evidence),
        "subtype": (float(weights["subtype"]), float(parsed.protocol_valid and sample.get("issue_subtype") == protocol_data.get("issue_subtype")), has_subtype),
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

