from __future__ import annotations

import re
from typing import Sequence


FIELD_WEIGHTS = {
    "decision": 1.0,
    "violation_type": 1.5,
    "issue_subtype": 1.5,
}


def _value_spans(text: str) -> list[tuple[int, int, float]]:
    spans: list[tuple[int, int, float]] = []
    for field, weight in FIELD_WEIGHTS.items():
        pattern = re.compile(
            rf'"{re.escape(field)}"\s*:\s*(null|"(?:\\.|[^"\\])*")',
            re.DOTALL,
        )
        match = pattern.search(text)
        if match is not None:
            spans.append((match.start(1), match.end(1), weight))
    return spans


def semantic_token_weights(tokenizer, token_ids: Sequence[int], enabled: bool) -> list[float]:
    """Map generated JSON value spans to OPD weights.

    Prefix decoding keeps the mapping tied to the exact generated token ids. JSON
    keys, punctuation, evidence values and malformed completions receive zero.
    """
    ids = [int(token_id) for token_id in token_ids]
    if not enabled or not ids:
        return [0.0] * len(ids)

    prefixes = [""]
    for end in range(1, len(ids) + 1):
        prefixes.append(tokenizer.decode(ids[:end], skip_special_tokens=False))
    text = prefixes[-1]
    spans = _value_spans(text)
    if not spans:
        return [0.0] * len(ids)

    weights: list[float] = []
    for index in range(len(ids)):
        start, end = len(prefixes[index]), len(prefixes[index + 1])
        weight = max(
            (
                span_weight
                for span_start, span_end, span_weight in spans
                if start < span_end and end > span_start
            ),
            default=0.0,
        )
        weights.append(weight)
    return weights
