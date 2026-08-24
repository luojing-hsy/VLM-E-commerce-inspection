from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import load_yaml, read_jsonl
from src.data.export_grpo import DATA_SOURCE
from src.data.export_sft import PROMPT
from src.data.split_manifest import TRAIN_SPLITS, manifest_path, write_split_manifests


TEACHER_PROMPT = (
    "你是冻结的区域增强教师。结合完整商品页与后续证据裁剪，"
    "在学生已经生成的前缀上判断下一个结构化输出 token。"
)


def _message(image_paths: list[str], text: str) -> list[dict]:
    content = [{"type": "image", "image": path} for path in image_paths]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _grpo_rows(config: dict, split: str) -> list[dict]:
    rows = read_jsonl(manifest_path(config, "grpo", split))
    exported = []
    for row in rows:
        image_paths = row.get("images")
        if not isinstance(image_paths, list) or len(image_paths) != 1:
            raise ValueError(f"invalid GRPO image list: {row.get('extra_info', {}).get('sample_id')}")
        extra_info = dict(row["extra_info"])
        extra_info["training_stage"] = "joint"
        exported.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": _message(image_paths, PROMPT),
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
        crops = list(row.get("crop_images") or [])
        if not crops:
            continue
        extra_info = {
            "dataset_stage": "opd",
            "training_stage": "joint",
            "split": split,
            "sample_id": row["sample_id"],
            "lineage": row["lineage"],
            "teacher_filter_status": row.get("teacher_filter_status"),
        }
        exported.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": _message([row["full_image"]], PROMPT),
                "teacher_prompt": _message([row["full_image"], *crops], TEACHER_PROMPT),
                "opd_enabled": True,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(row["target"], ensure_ascii=False, separators=(",", ":")),
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
