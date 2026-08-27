from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import load_yaml, read_jsonl
from src.data.export_grpo import DATA_SOURCE
from src.data.split_manifest import TRAIN_SPLITS, lineage_from_sample, manifest_path, write_split_manifests
from src.models.audit_protocol import product_prompt, structured_prompt


def _message(image_paths: list[str], row: dict) -> list[dict]:
    if not isinstance(image_paths, list) or len(image_paths) != 3:
        raise ValueError(f"joint row must contain main and two detail images: {row.get('sample_id')}")
    text = product_prompt(
        row.get("title"),
        row.get("category"),
        row.get("color"),
        row.get("material"),
        image_placeholders=False,
    )
    return structured_prompt(image_paths, text)


def _grpo_rows(config: dict, split: str) -> list[dict]:
    rows = read_jsonl(manifest_path(config, "grpo", split))
    exported = []
    for row in rows:
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"invalid GRPO image list: {row.get('extra_info', {}).get('sample_id')}")
        extra_info = dict(row["extra_info"])
        extra_info["training_stage"] = "joint"
        exported.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": _message(images, row),
                "images": images,
                "opd_enabled": False,
                "reward_model": row["reward_model"],
                "ability": row.get("ability", "product_audit"),
                "extra_info": extra_info,
            }
        )
    return exported


def _opd_rows(config: dict, split: str) -> list[dict]:
    rows = read_jsonl(manifest_path(config, "opd", split))
    exported = []
    for row in rows:
        if row.get("teacher_filter_status") != "approved":
            continue
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"invalid OPD image list: {row.get('sample_id')}")
        extra_info = {
            "dataset_stage": "opd",
            "training_stage": "joint",
            "split": split,
            "sample_id": row["sample_id"],
            "lineage": row["lineage"],
            "teacher_filter_status": row.get("teacher_filter_status"),
        }
        prompt = _message(images, row)
        exported.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": prompt,
                "teacher_prompt": prompt,
                "images": images,
                "opd_enabled": True,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        row["target"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
                "ability": "product_audit",
                "extra_info": extra_info,
            }
        )
    return exported


def export(config: dict, split: str) -> list[dict]:
    rows = _grpo_rows(config, split) + _opd_rows(config, split)
    for index, row in enumerate(rows):
        row["extra_info"]["index"] = index
    return rows


def write_exports(config: dict) -> dict[str, Path]:
    rows = [row for split in TRAIN_SPLITS for row in export(config, split)]
    return write_split_manifests(
        config,
        "joint",
        rows,
        split_getter=lambda row: row["extra_info"]["split"],
        splits=TRAIN_SPLITS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export veRL joint GRPO + regional OPD records")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    targets = write_exports(load_yaml(args.config))
    print(f"wrote joint veRL records to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
