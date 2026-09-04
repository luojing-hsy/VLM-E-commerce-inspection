from pathlib import Path

import pytest
import yaml

from src.common import write_jsonl
from src.training.runtime import validate_stage_config
from src.models.audit_protocol import prompt_with_image_token


def _sft_row(
    image: Path,
    split: str,
    sample_id: str,
    product_id: str,
    source_image_id: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "dataset_stage": "sft",
        "split": split,
        "images": [image.as_posix()] * 3,
        "lineage": {
            "dataset_stage": "sft",
            "source_product_ids": [product_id],
            "source_image_ids": [source_image_id],
            "derived_image_id": f"sample:{sample_id}",
        },
        "conversations": [
            {"from": "human", "value": prompt_with_image_token("Sample title", "sample_category", None, None)},
            {"from": "gpt", "value": '{"decision":"pass","violation_type":"pass","issue_subtype":null,"evidence":null}'},
        ],
    }


def _write_sft_config(tmp_path: Path, train: Path, validation: Path) -> Path:
    config = {
        "stage": "sft",
        "framework": "verl",
        "dataset": train.as_posix(),
        "validation_dataset": validation.as_posix(),
        "output_dir": (tmp_path / "output").as_posix(),
        "precision": "bf16",
        "quantization": "none",
        "global_train_batch_size": 1,
        "per_device_train_batch_size": 1,
        "n_gpus_per_node": 1,
        "nnodes": 1,
    }
    path = tmp_path / "sft.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_sft_entry_rejects_test_row_in_train_export(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    train = tmp_path / "sft_train.jsonl"
    validation = tmp_path / "sft_validation.jsonl"
    write_jsonl(
        train,
        [
            _sft_row(image, "train", "train-1", "product-a", "image-a"),
            _sft_row(image, "test", "test-1", "product-b", "image-b"),
        ],
    )
    write_jsonl(validation, [_sft_row(image, "validation", "val-1", "product-c", "image-c")])

    with pytest.raises(ValueError, match="expected split=train"):
        validate_stage_config(_write_sft_config(tmp_path, train, validation), "sft")


def test_sft_entry_rejects_source_product_overlap(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    train = tmp_path / "sft_train.jsonl"
    validation = tmp_path / "sft_validation.jsonl"
    write_jsonl(train, [_sft_row(image, "train", "train-1", "shared-product", "image-a")])
    write_jsonl(
        validation,
        [_sft_row(image, "validation", "val-1", "shared-product", "image-b")],
    )

    with pytest.raises(ValueError, match="source product IDs overlap"):
        validate_stage_config(_write_sft_config(tmp_path, train, validation), "sft")


