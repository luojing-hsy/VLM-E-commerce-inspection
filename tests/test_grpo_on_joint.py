import json
from pathlib import Path

import pytest
from PIL import Image

from src.common import load_yaml, write_jsonl
from src.data.export_grpo_from_joint import convert_rows, write_exports
from src.training.runtime import build_verl_command


ROOT = Path(__file__).resolve().parents[1]


def _joint_row(tmp_path: Path, dataset_stage: str) -> dict:
    images = [(tmp_path / f"image_{index}.png").as_posix() for index in range(3)]
    for image_path in images:
        Image.new("RGB", (8, 8), "white").save(image_path)
    return {
        "data_source": "vlm_product_audit",
        "prompt": [{"role": "user", "content": []}],
        "teacher_prompt": [{"role": "user", "content": []}],
        "images": images,
        "opd_enabled": dataset_stage == "opd",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"decision": "reject", "violation_type": "TITLE_MISMATCH", "issue_subtype": None, "evidence": None}
            ),
        },
        "extra_info": {
            "dataset_stage": dataset_stage,
            "training_stage": "joint",
            "split": "train",
            "sample_id": f"sample-{dataset_stage}",
            "lineage": {
                "source_product_ids": [f"product-{dataset_stage}"],
                "source_image_ids": [f"image-{dataset_stage}"],
                "derived_image_id": f"page-{dataset_stage}",
            },
        },
    }


def test_convert_rows_removes_teacher_fields_and_preserves_joint_task_data(tmp_path: Path) -> None:
    rows = convert_rows([_joint_row(tmp_path, "grpo"), _joint_row(tmp_path, "opd")], "train")

    assert all(set(row) == {"data_source", "prompt", "images", "ability", "reward_model", "extra_info"} for row in rows)
    assert all(row["extra_info"]["dataset_stage"] == "grpo" for row in rows)
    assert [row["extra_info"]["joint_dataset_stage"] for row in rows] == ["grpo", "opd"]
    assert [row["extra_info"]["joint_opd_enabled"] for row in rows] == [False, True]
    assert all("teacher_prompt" not in row and "opd_enabled" not in row for row in rows)


def test_write_exports_uses_joint_train_and_validation_sources(tmp_path: Path) -> None:
    source_train = tmp_path / "joint_train.jsonl"
    source_validation = tmp_path / "joint_validation.jsonl"
    target_train = tmp_path / "grpo_train.jsonl"
    target_validation = tmp_path / "grpo_validation.jsonl"
    train_row = _joint_row(tmp_path, "grpo")
    validation_row = _joint_row(tmp_path, "grpo")
    validation_row["extra_info"]["split"] = "validation"
    write_jsonl(source_train, [train_row])
    write_jsonl(source_validation, [validation_row])

    targets = write_exports(
        {
            "joint_source_dataset": str(source_train),
            "joint_source_validation_dataset": str(source_validation),
            "dataset": str(target_train),
            "validation_dataset": str(target_validation),
        }
    )

    assert targets == {"train": target_train, "validation": target_validation}
    assert json.loads(target_train.read_text(encoding="utf-8").splitlines()[0])["extra_info"]["split"] == "train"
    assert json.loads(target_validation.read_text(encoding="utf-8").splitlines()[0])["extra_info"]["split"] == "validation"


def test_write_exports_rejects_corrupt_images_before_training(tmp_path: Path) -> None:
    source_train = tmp_path / "joint_train.jsonl"
    source_validation = tmp_path / "joint_validation.jsonl"
    train_row = _joint_row(tmp_path, "grpo")
    validation_row = _joint_row(tmp_path, "grpo")
    validation_row["extra_info"]["split"] = "validation"
    corrupt_path = Path(train_row["images"][0])
    corrupt_path.write_bytes(b"not an image")
    write_jsonl(source_train, [train_row])
    write_jsonl(source_validation, [validation_row])

    with pytest.raises(ValueError, match=r"sample-grpo.*invalid image"):
        write_exports(
            {
                "joint_source_dataset": str(source_train),
                "joint_source_validation_dataset": str(source_validation),
                "dataset": str(tmp_path / "grpo_train.jsonl"),
                "validation_dataset": str(tmp_path / "grpo_validation.jsonl"),
            }
        )

def test_grpo_on_joint_config_does_not_add_distillation() -> None:
    command = build_verl_command(load_yaml(ROOT / "configs" / "grpo_on_joint.yaml"))

    assert "algorithm.adv_estimator=grpo" in command
    assert not any("distillation" in item for item in command)
    assert not any("teacher" in item for item in command)
