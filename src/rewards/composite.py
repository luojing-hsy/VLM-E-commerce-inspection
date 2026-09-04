from __future__ import annotations

from typing import Any

from src.models.audit_protocol import IMAGE_EVIDENCE_TYPES
from src.rewards.action_cost import action_reward
from src.rewards.evidence_reward import evidence_reward
from src.rewards.parser import tolerant_parse
from src.rewards.type_reward import violation_type_reward

DEFAULT_WEIGHTS = {"action": 0.30, "type": 0.35, "evidence": 0.20, "subtype": 0.15}


def compute_reward(sample: dict[str, Any], prediction: str | dict, config: dict[str, Any]) -> dict[str, Any]:
    parsed = tolerant_parse(prediction)
    data = parsed.data
    protocol_data = data if parsed.protocol_valid else {}
    weights = config.get("reward_weights", DEFAULT_WEIGHTS)
    gold_type = sample.get("violation_type")
    has_evidence = gold_type in IMAGE_EVIDENCE_TYPES
    has_subtype = gold_type == "image_quality"

    type_score = (
        violation_type_reward(gold_type, protocol_data.get("violation_type"))
        if parsed.protocol_valid
        else 0.0
    )
    type_correct = type_score == 1.0
    evidence_score = (
        evidence_reward(sample, protocol_data)
        if parsed.protocol_valid and type_correct and has_evidence
        else 0.0
    )
    subtype_score = float(
        parsed.protocol_valid
        and type_correct
        and has_subtype
        and sample.get("issue_subtype") == protocol_data.get("issue_subtype")
    )
    components = {
        "action": (
            float(weights["action"]),
            action_reward(sample["decision"], data.get("decision"), config["cost_matrix"]),
            True,
        ),
        "type": (float(weights["type"]), type_score, True),
        "evidence": (float(weights["evidence"]), evidence_score, has_evidence),
        "subtype": (float(weights["subtype"]), subtype_score, has_subtype),
    }
    numerator = sum(weight * score for weight, score, enabled in components.values() if enabled)
    denominator = sum(weight for weight, _, enabled in components.values() if enabled)
    if denominator <= 0.0:
        raise ValueError("at least one reward component must be enabled")
    pre_gate_reward = numerator / denominator

    gate_threshold = float(config.get("evidence_gate_threshold", 0.3))
    gate_multiplier = float(config.get("evidence_gate_multiplier", 0.70))
    if not 0.0 <= gate_multiplier <= 1.0:
        raise ValueError("evidence gate multiplier must be in [0, 1]")
    gate_applied = has_evidence and evidence_score < gate_threshold
    total = (
        pre_gate_reward * gate_multiplier
        if gate_applied and pre_gate_reward > 0.0
        else pre_gate_reward
    )
    return {
        "reward": total,
        "pre_gate_reward": pre_gate_reward,
        "components": {name: score for name, (_, score, _) in components.items()},
        "masks": {name: enabled for name, (_, _, enabled) in components.items()},
        "evidence_gate_applied": gate_applied,
        "protocol_valid": parsed.protocol_valid,
        "parse_error": parsed.error,
    }
