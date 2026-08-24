from __future__ import annotations

import argparse
from pathlib import Path

from src.common import load_yaml, read_jsonl
from src.data.export_sft import PROMPT, target_from_sample
from src.data.split_manifest import SPLITS, lineage_from_sample, manifest_path, write_split_manifests

ALLOWED_TYPES = {"PRODUCT_MISMATCH", "ATTRIBUTE_CONFLICT", "TEXT_LABEL_CONFLICT"}


def export(config: dict, split: str) -> list[dict]:
    rows = read_jsonl(manifest_path(config, "samples", split))
    return [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "full_image": row["image"],
            "crop_images": [crop["path"] for crop in row.get("crops", [])],
            "lineage": lineage_from_sample(row),
            "crop_lineage": [
                {
                    "derived_image_id": crop["derived_image_id"],
                    "parent_derived_image_id": crop["parent_derived_image_id"],
                    "source_image_ids": crop["source_image_ids"],
                }
                for crop in row.get("crops", [])
            ],
            "prompt": PROMPT,
            "target": target_from_sample(row),
            "teacher_filter_status": "pending_model_inference",
        }
        for row in rows
        if row["violation_type"] in ALLOWED_TYPES and row.get("crops")
    ]


def write_exports(config: dict) -> dict[str, Path]:
    rows = [row for split in SPLITS for row in export(config, split)]
    return write_split_manifests(config, "opd", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export regional-to-global OPD inputs")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    targets = write_exports(config)
    print(f"wrote split OPD candidates to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
