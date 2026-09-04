from __future__ import annotations


def violation_type_reward(true_type: str | None, predicted_type: str | None) -> float:
    """Return credit only for an exact V2 violation-type match."""
    return float(true_type is not None and true_type == predicted_type)
