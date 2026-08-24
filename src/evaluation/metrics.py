from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.rewards.evidence_reward import evidence_reward
from src.rewards.value_reward import normalize_value


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def classification_metrics(samples: list[dict], predictions: dict[str, dict], config: dict[str, Any]) -> dict[str, Any]:
    available = [sample for sample in samples if sample["sample_id"] in predictions]
    if not available:
        raise ValueError("no prediction sample_ids matched the manifest")
    true_decisions = [sample["decision"] for sample in available]
    pred_decisions = [predictions[sample["sample_id"]].get("decision", "INVALID") for sample in available]
    true_types = [sample["violation_type"] for sample in available]
    pred_types = [predictions[sample["sample_id"]].get("violation_type") or "PASS" for sample in available]
    total = len(available)
    severe = [index for index, value in enumerate(true_decisions) if value == "reject"]
    passes = [index for index, value in enumerate(true_decisions) if value == "pass"]

    by_true: dict[str, Counter] = defaultdict(Counter)
    for true, predicted in zip(true_decisions, pred_decisions):
        by_true[true][predicted] += 1
    risk = 0.0
    for true, prior in config["target_class_prior"].items():
        count = sum(by_true[true].values())
        if not count:
            continue
        class_cost = sum(
            amount / count * float(config["cost_matrix"][true].get(action, 1.0))
            for action, amount in by_true[true].items()
        )
        risk += float(prior) * class_cost

    return {
        "num_evaluated": total,
        "decision_accuracy": sum(a == b for a, b in zip(true_decisions, pred_decisions)) / total,
        "violation_macro_f1": _macro_f1(true_types, pred_types),
        "severe_violation_miss_rate": sum(pred_decisions[index] == "pass" for index in severe) / len(severe) if severe else None,
        "false_reject_rate": sum(pred_decisions[index] == "reject" for index in passes) / len(passes) if passes else None,
        "unnecessary_review_rate": sum(pred_decisions[index] == "review" for index in passes) / len(passes) if passes else None,
        "business_risk": risk,
        "decision_confusion": {true: dict(counter) for true, counter in sorted(by_true.items())},
    }


def perception_metrics(samples: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    observed, listed, evidence_scores = [], [], []
    for sample in samples:
        prediction = predictions.get(sample["sample_id"])
        if prediction is None:
            continue
        if sample.get("observed_value") is not None:
            observed.append(normalize_value(sample["observed_value"]) == normalize_value(prediction.get("observed_value")))
        if sample.get("listed_value") is not None:
            listed.append(normalize_value(sample["listed_value"]) == normalize_value(prediction.get("listed_value")))
        if sample.get("evidence"):
            evidence_scores.append(evidence_reward(sample, prediction))
    return {
        "observed_value_exact_match": sum(observed) / len(observed) if observed else None,
        "listed_value_exact_match": sum(listed) / len(listed) if listed else None,
        "evidence_mean_score": sum(evidence_scores) / len(evidence_scores) if evidence_scores else None,
        "evidence_recall_at_0_5": sum(score >= 0.5 for score in evidence_scores) / len(evidence_scores) if evidence_scores else None,
    }

