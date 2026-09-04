from __future__ import annotations

from typing import Any

from src.models.audit_protocol import IMAGE_REFS


def _valid_image_ref(value: Any) -> str | None:
    return value if isinstance(value, str) and value in IMAGE_REFS else None


def _expected_image_ref(sample: dict[str, Any]) -> str | None:
    target_image_ref = _valid_image_ref(sample.get("target_image_ref"))
    if target_image_ref is not None:
        return target_image_ref
    return _valid_image_ref(sample.get("evidence"))


def evidence_reward(sample: dict[str, Any], prediction: dict[str, Any]) -> float:
    expected_ref = _expected_image_ref(sample)
    predicted_ref = _valid_image_ref(prediction.get("evidence"))
    return float(expected_ref is not None and predicted_ref == expected_ref)
