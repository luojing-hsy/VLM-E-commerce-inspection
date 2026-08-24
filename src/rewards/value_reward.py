from __future__ import annotations

import re
import unicodedata

COLOR_ALIASES = {
    "navyblue": "navy",
    "deepblue": "navy",
    "深蓝": "navy",
}


def normalize_value(value: str | None) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", value).strip().lower()
    compact = re.sub(r"[\s_\-]+", "", text)
    if compact in COLOR_ALIASES:
        return COLOR_ALIASES[compact]
    measurement = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m)", text)
    if measurement:
        amount, unit = float(measurement.group(1)), measurement.group(2)
        cm_value = amount / 10 if unit == "mm" else amount * 100 if unit == "m" else amount
        return f"{cm_value:g}cm"
    return compact


def value_reward(sample: dict, prediction: dict) -> float:
    expected_listed = normalize_value(sample.get("listed_value"))
    expected_observed = normalize_value(sample.get("observed_value"))
    predicted_listed = normalize_value(prediction.get("listed_value"))
    predicted_observed = normalize_value(prediction.get("observed_value"))
    listed_ok = expected_listed is not None and expected_listed == predicted_listed
    observed_ok = expected_observed is not None and expected_observed == predicted_observed
    if listed_ok and observed_ok:
        return 1.0
    if listed_ok or observed_ok:
        return 0.5
    if prediction.get("decision") in {"review", "reject"} and prediction.get("violation_type") == sample.get("violation_type"):
        return 0.3
    return 0.0

