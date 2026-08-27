from __future__ import annotations

from typing import Mapping


def action_reward(true_decision: str, predicted_decision: str | None, cost_matrix: Mapping[str, Mapping[str, float]]) -> float:
    if predicted_decision not in {"pass", "reject"}:
        return -1.0
    cost = float(cost_matrix[true_decision][predicted_decision])
    if not 0.0 <= cost <= 1.0:
        raise ValueError("cost matrix values must be in [0, 1]")
    return 1.0 - 2.0 * cost

