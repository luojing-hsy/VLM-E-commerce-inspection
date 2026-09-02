from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.common import load_yaml, read_jsonl, write_jsonl


def _to_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"dataset path must be a non-empty string: {value!r}")
    return Path(value)


def convert_rows(rows: list[dict], expected_split: str) -> list[dict]:
    exported: list[dict] = []
    for index, row in enumerate(rows):
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError(f"joint row is missing extra_info: {index}")
        if extra_info.get("split") != expected_split:
            raise ValueError(
                f"joint row {extra_info.get('sample_id')} has split={extra_info.get('split')}; "
                f"expected {expected_split}"
            )
        if extra_info.get("dataset_stage") not in {"grpo", "opd"}:
            raise ValueError(
                f"joint row {extra_info.get('sample_id')} has invalid dataset_stage="
                f"{extra_info.get('dataset_stage')}"
            )
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            raise ValueError(f"joint row is missing a structured prompt: {extra_info.get('sample_id')}")
        images = row.get("images")
        if images is None:
            content = prompt[0].get("content", []) if prompt else []
            images = [
                part.get("image")
                for part in content
                if isinstance(part, dict) and part.get("type") == "image"
            ]
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"joint row must contain three images: {extra_info.get('sample_id')}")
        if not isinstance(row.get("reward_model"), dict):
            raise ValueError(f"joint row is missing reward_model: {extra_info.get('sample_id')}")

        grpo_extra_info = dict(extra_info)
        grpo_extra_info["dataset_stage"] = "grpo"
        grpo_extra_info["training_stage"] = "grpo_on_joint"
        grpo_extra_info["joint_dataset_stage"] = extra_info["dataset_stage"]
        if "opd_enabled" in row:
            if not isinstance(row["opd_enabled"], bool):
                raise ValueError(f"joint row has non-boolean opd_enabled: {extra_info.get('sample_id')}")
            grpo_extra_info["joint_opd_enabled"] = row["opd_enabled"]
        grpo_extra_info["index"] = index

        exported.append(
            {
                "data_source": row.get("data_source"),
                "prompt": row["prompt"],
                "images": images,
                "ability": row.get("ability", "product_audit"),
                "reward_model": row["reward_model"],
                "extra_info": grpo_extra_info,
            }
        )
    return exported

def validate_image_files(rows: list[dict]) -> None:
    checked: set[Path] = set()
    for row in rows:
        sample_id = row["extra_info"].get("sample_id", "<unknown>")
        for value in row["images"]:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{sample_id} has invalid image path: {value!r}")
            image_path = Path(value)
            if image_path in checked:
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                raise ValueError(f"{sample_id} has invalid image {image_path}: {exc}") from exc
            checked.add(image_path)


def write_exports(config: dict) -> dict[str, Path]:
    pairs = {
        "train": ("joint_source_dataset", "dataset"),
        "validation": ("joint_source_validation_dataset", "validation_dataset"),
    }
    targets: dict[str, Path] = {}
    for split, (source_key, target_key) in pairs.items():
        source = _to_path(config[source_key])
        target = _to_path(config[target_key])
        if not source.is_file():
            raise FileNotFoundError(f"joint source dataset does not exist: {source}")
        rows = convert_rows(read_jsonl(source), split)
        validate_image_files(rows)
        write_jsonl(target, rows)
        targets[split] = target
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GRPO view from the joint runtime dataset")
    parser.add_argument("--config", default="configs/grpo_on_joint.yaml")
    args = parser.parse_args()
    targets = write_exports(load_yaml(args.config))
    print(f"wrote GRPO-on-joint records to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
