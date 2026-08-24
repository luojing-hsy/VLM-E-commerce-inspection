from __future__ import annotations

from collections import Counter


def slice_summary(samples: list[dict]) -> dict:
    return {
        "by_split": dict(Counter(row["split"] for row in samples)),
        "by_violation_type": dict(Counter(row["violation_type"] for row in samples)),
        "by_template": dict(Counter(row["template_id"] for row in samples)),
        "with_small_local_evidence": sum(
            any(
                item.get("region_type") == "bbox"
                and (item["bbox_norm"][2] - item["bbox_norm"][0]) * (item["bbox_norm"][3] - item["bbox_norm"][1]) < 300_000
                for item in row["evidence"]
            )
            for row in samples
        ),
    }

