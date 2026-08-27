from __future__ import annotations

import re
from typing import Sequence


SEMANTIC_FIELDS = ("decision", "violation_type", "issue_subtype", "evidence")


def _semantic_value_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for field in SEMANTIC_FIELDS:
        pattern = re.compile(
            rf'"{re.escape(field)}"\s*:\s*(?:"((?:\\.|[^"\\])*)"|(null))',
            re.DOTALL,
        )
        match = pattern.search(text)
        if match is not None:
            group = 1 if match.group(1) is not None else 2
            spans.append((match.start(group), match.end(group)))
    return spans


def semantic_completion_mask(tokenizer, token_ids: Sequence[int]) -> list[int]:
    """Return a token mask for JSON semantic values, excluding keys and punctuation."""
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return []

    prefixes = [""]
    for end in range(1, len(ids) + 1):
        prefixes.append(tokenizer.decode(ids[:end], skip_special_tokens=False))
    spans = _semantic_value_spans(prefixes[-1])
    return [
        int(any(start < span_end and end > span_start for span_start, span_end in spans))
        for start, end in (
            (len(prefixes[index]), len(prefixes[index + 1]))
            for index in range(len(ids))
        )
    ]
