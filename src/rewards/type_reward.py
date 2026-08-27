from __future__ import annotations

GROUPS = {
    "duplicate_detail_image": "image",
    "image_quality": "quality",
    "wrong_image": "image",
    "category_mismatch": "consistency",
    "color_mismatch": "consistency",
    "material_mismatch": "consistency",
    "title_mismatch": "consistency",
    "pass": "normal",
}


def violation_type_reward(true_type: str | None, predicted_type: str | None) -> float:
    canonical_true = true_type
    canonical_predicted = predicted_type
    if canonical_true == canonical_predicted:
        return 1.0
    if GROUPS.get(canonical_true) == GROUPS.get(canonical_predicted):
        return 0.25
    return 0.0

