from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import load_yaml, read_jsonl, write_jsonl
from src.data.split_manifest import TRAIN_SPLITS, lineage_from_sample, manifest_path, write_split_manifests

PROMPT = "检查商品页是否存在违规，并输出规定字段。"


def target_from_sample(sample: dict) -> dict:
    return {
        "schema_version": "1.0",
        "decision": sample["decision"],
        "violation_type": None if sample["violation_type"] == "PASS" else sample["violation_type"],
        "field": sample["field"],
        "listed_value": sample["listed_value"],
        "observed_value": sample["observed_value"],
        "evidence": [
            {key: value for key, value in item.items() if key not in {"value", "evidence_source", "source_field"}}
            for item in sample["evidence"]
        ],
    }


def export(config: dict, split: str) -> list[dict]:
    rows = [row for row in read_jsonl(manifest_path(config, "samples", split)) if row["dataset_stage"] == "sft"]
    counterfactual_path = manifest_path(config, "counterfactuals", split)
    if counterfactual_path.exists():
        rows += [row for row in read_jsonl(counterfactual_path) if row["dataset_stage"] == "sft"]
    return [
        {
            "sample_id": sample["sample_id"],
            "dataset_stage": sample["dataset_stage"],
            "split": sample["split"],
            "image": sample["image"],
            "lineage": lineage_from_sample(sample),
            "conversations": [
                {"from": "human", "value": f"<image>\n{PROMPT}"},
                {"from": "gpt", "value": json.dumps(target_from_sample(sample), ensure_ascii=False, separators=(",", ":"))},
            ],
        }
        for sample in rows
    ]


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
