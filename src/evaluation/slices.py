from __future__ import annotations

from collections import Counter


def slice_summary(samples: list[dict]) -> dict:
    return {
        "by_split": dict(Counter(row["split"] for row in samples)),
        "by_violation_type": dict(Counter(row["violation_type"] for row in samples)),
        "by_template": dict(
            Counter(row.get("template_id") for row in samples if row.get("template_id") is not None)
        ),
        "with_small_local_evidence": 0,
    }

