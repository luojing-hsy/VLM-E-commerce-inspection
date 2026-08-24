from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.models.schema import AuditPrediction


@dataclass(frozen=True)
class ParseResult:
    data: dict[str, Any]
    protocol_valid: bool
    error: str | None = None


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def tolerant_parse(prediction: str | dict[str, Any]) -> ParseResult:
    if isinstance(prediction, dict):
        raw = prediction
    else:
        try:
            raw = json.loads(_extract_json(prediction))
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

