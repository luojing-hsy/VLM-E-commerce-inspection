from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import load_yaml, read_jsonl, write_jsonl
from src.data.export_sft import PROMPT, target_from_sample
from src.data.split_manifest import TRAIN_SPLITS, lineage_from_sample, manifest_path, write_split_manifests

DATA_SOURCE = "vlm_product_audit"


def export(config: dict, split: str) -> list[dict]:
    rows = [row for row in read_jsonl(manifest_path(config, "samples", split)) if row["dataset_stage"] == "grpo"]
    return [
        {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"<image>\n{PROMPT}"}],
            "images": [row["image"]],
            "ability": "product_audit",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(target_from_sample(row), ensure_ascii=False, separators=(",", ":")),
            },
            "extra_info": {
                "dataset_stage": row["dataset_stage"],
                "split": row["split"],
                "index": index,
                "sample_id": row["sample_id"],
                "lineage": lineage_from_sample(row),
            },
        }
        for index, row in enumerate(rows)
    ]


def write_exports(config: dict) -> dict[str, Path]:
    rows = [row for split in TRAIN_SPLITS for row in export(config, split)]
    return write_split_manifests(
        config,
        "grpo",
        rows,
        split_getter=lambda row: row["extra_info"]["split"],
        splits=TRAIN_SPLITS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prompts and rule-verifiable GRPO targets")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    targets = write_exports(config)
    print(f"wrote veRL GRPO records to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
