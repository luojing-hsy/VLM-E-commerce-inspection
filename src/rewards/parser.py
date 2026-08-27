from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.models.schema import AuditPrediction


@dataclass(frozen=True)
class ParseResult:
    data: dict[str, Any]
    protocol_valid: bool
    error: str | None = None


def tolerant_parse(prediction: str | dict[str, Any]) -> ParseResult:
    if isinstance(prediction, dict):
        raw = prediction
    else:
        try:
            raw = json.loads(prediction.strip())
        except (json.JSONDecodeError, TypeError) as exc:
            return ParseResult({}, False, str(exc))
    if not isinstance(raw, dict):
        return ParseResult({}, False, "prediction is not a JSON object")
    try:
        parsed = AuditPrediction.model_validate(raw)
        return ParseResult(parsed.model_dump(mode="json"), True)
    except ValidationError as exc:
        # Reward logic can still use an executable decision from a partially valid object.
        return ParseResult(raw, False, str(exc))

