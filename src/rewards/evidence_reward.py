from __future__ import annotations

from src.models.audit_protocol import IMAGE_REFS


def _single_image_ref(value: dict) -> str | None:
    evidence = value.get("evidence")
    if isinstance(evidence, str) and evidence in IMAGE_REFS:
        return evidence
    target_image_ref = value.get("target_image_ref")
    return target_image_ref if target_image_ref in IMAGE_REFS else None


def evidence_reward(sample: dict, prediction: dict) -> float:
    expected_ref = _single_image_ref(sample)
    predicted_ref = _single_image_ref(prediction)
    return float(expected_ref is not None and predicted_ref == expected_ref)
