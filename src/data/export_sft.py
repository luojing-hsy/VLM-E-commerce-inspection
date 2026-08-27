from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import load_yaml, read_jsonl, write_jsonl
from src.data.split_manifest import TRAIN_SPLITS, lineage_from_sample, manifest_path, write_split_manifests
from src.models.audit_protocol import prompt_with_image_token, target_from_sample



def export(config: dict, split: str) -> list[dict]:
    rows = [row for row in read_jsonl(manifest_path(config, "samples", split)) if row["dataset_stage"] == "sft"]
    counterfactual_path = manifest_path(config, "counterfactuals", split)
    if counterfactual_path.exists():
        rows += [row for row in read_jsonl(counterfactual_path) if row["dataset_stage"] == "sft"]
    exported = []
    for sample in rows:
        images = sample.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"SFT row must contain main and two detail images: {sample.get('sample_id')}")
        prompt = prompt_with_image_token(
            sample.get("title"),
            sample.get("category"),
            sample.get("color"),
            sample.get("material"),
        )
        exported.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_stage": sample["dataset_stage"],
                "split": sample["split"],
                "images": images,
                "lineage": lineage_from_sample(sample),
                "conversations": [
                    {"from": "human", "value": prompt},
                    {
                        "from": "gpt",
                        "value": json.dumps(
                            target_from_sample(sample),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
            }
        )
    return exported


def write_exports(config: dict) -> dict[str, Path]:
    rows = [row for split in TRAIN_SPLITS for row in export(config, split)]
    return write_split_manifests(config, "sft", rows, splits=TRAIN_SPLITS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen-style SFT conversations")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    targets = write_exports(config)
    print(f"wrote split SFT records to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
