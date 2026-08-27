from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.rewards.evidence_reward import evidence_reward


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


def _complete_inspection_success(sample: dict, prediction: dict) -> bool:
    if prediction.get("decision") != sample.get("decision"):
        return False
    violation_type = sample.get("violation_type")
    if prediction.get("violation_type") != violation_type:
        return False
    if violation_type == "image_quality":
        return (
            prediction.get("issue_subtype") == sample.get("issue_subtype")
            and evidence_reward(sample, prediction) == 1.0
        )
    if violation_type == "wrong_image":
        return evidence_reward(sample, prediction) == 1.0
    return True


def classification_metrics(samples: list[dict], predictions: dict[str, dict], config: dict[str, Any]) -> dict[str, Any]:
    matched = [sample for sample in samples if sample["sample_id"] in predictions]
    if not matched:
        raise ValueError("no prediction sample_ids matched the manifest")
    available = samples
    true_decisions = [sample["decision"] for sample in available]
    pred_decisions = [predictions.get(sample["sample_id"], {}).get("decision", "INVALID") for sample in available]
    true_types = [sample["violation_type"] for sample in available]
    pred_types = [predictions.get(sample["sample_id"], {}).get("violation_type", "INVALID") for sample in available]
    total = len(available)
    severe = [index for index, value in enumerate(true_decisions) if value == "reject"]
    passes = [index for index, value in enumerate(true_decisions) if value == "pass"]
    type_successes = [a == b for a, b in zip(true_types, pred_types)]
    complete_successes = [
        _complete_inspection_success(sample, predictions.get(sample["sample_id"], {})) for sample in available
    ]

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
        "detection_success_rate": sum(a == b for a, b in zip(true_decisions, pred_decisions)) / total,
        "violation_detection_success_rate": (
            sum(pred_decisions[index] == "reject" for index in severe) / len(severe) if severe else None
        ),
        "type_inspection_success_probability": sum(type_successes) / total,
        "complete_inspection_success_probability": sum(complete_successes) / total,
        "decision_accuracy": sum(a == b for a, b in zip(true_decisions, pred_decisions)) / total,
        "violation_macro_f1": _macro_f1(true_types, pred_types),
        "severe_violation_miss_rate": sum(pred_decisions[index] == "pass" for index in severe) / len(severe) if severe else None,
        "false_reject_rate": sum(pred_decisions[index] == "reject" for index in passes) / len(passes) if passes else None,
        "business_risk": risk,
        "decision_confusion": {true: dict(counter) for true, counter in sorted(by_true.items())},
    }


def perception_metrics(samples: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    subtype_scores, evidence_scores = [], []
    for sample in samples:
        prediction = predictions.get(sample["sample_id"])
        if prediction is None:
            continue
        if sample.get("violation_type") == "image_quality":
            subtype_scores.append(sample.get("issue_subtype") == prediction.get("issue_subtype"))
        if sample.get("violation_type") in {"image_quality", "wrong_image"}:
            evidence_scores.append(evidence_reward(sample, prediction))
    return {
        "quality_subtype_accuracy": sum(subtype_scores) / len(subtype_scores) if subtype_scores else None,
        "image_ref_accuracy": sum(evidence_scores) / len(evidence_scores) if evidence_scores else None,
        "evidence_mean_score": sum(evidence_scores) / len(evidence_scores) if evidence_scores else None,
        "evidence_recall_at_0_5": sum(score >= 0.5 for score in evidence_scores) / len(evidence_scores) if evidence_scores else None,
    }

