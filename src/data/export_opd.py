from __future__ import annotations

import argparse
from pathlib import Path

from src.common import load_yaml, read_jsonl
from src.models.audit_protocol import VIOLATION_TYPES, product_prompt, target_from_sample
from src.data.split_manifest import TRAIN_SPLITS, lineage_from_sample, manifest_path, write_split_manifests

ALLOWED_TYPES = VIOLATION_TYPES


def export(config: dict, split: str) -> list[dict]:
    rows = read_jsonl(manifest_path(config, "samples", split))
    exported = []
    for row in rows:
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"OPD row must contain main and two detail images: {row.get('sample_id')}")
        if row["dataset_stage"] != "opd" or row["violation_type"] not in ALLOWED_TYPES:
            continue
        exported.append(
            {
                "sample_id": row["sample_id"],
                "dataset_stage": row["dataset_stage"],
                "split": row["split"],
                "images": images,
                "title": row.get("title"),
                "category": row.get("category"),
                "color": row.get("color"),
                "material": row.get("material"),
                "lineage": lineage_from_sample(row),
                "prompt_text": product_prompt(
                    row.get("title"),
                    row.get("category"),
                    row.get("color"),
                    row.get("material"),
                    image_placeholders=False,
                ),
                "target": target_from_sample(row),
                "teacher_filter_status": "pending_model_inference",
            }
        )
    return exported


def write_exports(config: dict) -> dict[str, Path]:
    rows = [row for split in TRAIN_SPLITS for row in export(config, split)]
    return write_split_manifests(config, "opd", rows, splits=TRAIN_SPLITS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export regional-to-global OPD inputs")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    targets = write_exports(config)
    print(f"wrote split OPD candidates to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
