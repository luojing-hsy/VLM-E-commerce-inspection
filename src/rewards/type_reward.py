from __future__ import annotations

GROUPS = {
    "PRODUCT_MISMATCH": "consistency",
    "ATTRIBUTE_CONFLICT": "consistency",
    "TEXT_LABEL_CONFLICT": "consistency",
    "MISSING_REQUIRED_FIELD": "completeness",
    "IMAGE_QUALITY": "quality",
    "IRRELEVANT_IMAGE": "quality",
    "DUPLICATE_IMAGE": "quality",
    "PASS": "normal",
    None: "normal",
}


def violation_type_reward(true_type: str | None, predicted_type: str | None) -> float:
    canonical_true = "PASS" if true_type is None else true_type
    canonical_predicted = "PASS" if predicted_type is None else predicted_type
    if canonical_true == canonical_predicted:
        return 1.0
    if GROUPS.get(canonical_true) == GROUPS.get(canonical_predicted):
        return 0.25
    return 0.0

