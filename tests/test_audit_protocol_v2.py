import json
from pathlib import Path

import pytest

from src.common import write_jsonl
from src.data.export_verl_sft import build_rows
from src.models.audit_protocol import (
    OUTPUT_KEYS,
    product_prompt,
    prompt_with_image_token,
    structured_prompt,
    validate_prediction_dict,
)
from src.rewards.parser import tolerant_parse
from src.training.runtime import validate_joint_export


def _pass_target() -> dict:
    return {
        "decision": "pass",
        "violation_type": "pass",
        "issue_subtype": None,
        "evidence": None,
    }


def test_protocol_is_strict_four_field_json() -> None:
    assert tuple(validate_prediction_dict(_pass_target())) == OUTPUT_KEYS
    with pytest.raises(ValueError, match="exactly"):
        validate_prediction_dict({**_pass_target(), "sample_id": "secret"})
    assert not tolerant_parse(f"```json\n{json.dumps(_pass_target())}\n```").protocol_valid


def test_sft_parquet_rows_expose_four_fields_and_three_images(tmp_path: Path) -> None:
    images = [tmp_path / f"image_{index}.png" for index in range(3)]
    for image in images:
        image.write_bytes(b"image")
    source = tmp_path / "sft_train.jsonl"
    prompt = product_prompt("Sample title", "sample_category", None, "fabric", image_placeholders=True)
    write_jsonl(
        source,
        [
            {
                "sample_id": "do-not-expose-this-id",
                "dataset_stage": "sft",
                "split": "train",
                "images": [image.as_posix() for image in images],
                "lineage": {"transform": "do-not-expose-this-transform"},
                "conversations": [
                    {"from": "human", "value": prompt},
                    {
                        "from": "gpt",
                        "value": json.dumps(_pass_target(), ensure_ascii=False, separators=(",", ":")),
                    },
                ],
            }
        ],
    )
    rows = build_rows(source)
    assert set(rows[0]) == {"messages", "images"}
    assert rows[0]["images"] == [image.as_posix() for image in images]
    assert rows[0]["messages"][0]["content"].count("<image>") == 3
    visible = json.dumps(rows[0]["messages"], ensure_ascii=False)
    assert "do-not-expose-this-id" not in visible
    assert "do-not-expose-this-transform" not in visible


def _lineage(prefix: str) -> dict:
    return {
        "source_product_ids": [f"{prefix}-product"],
        "source_image_ids": [f"{prefix}-image"],
        "derived_image_id": f"{prefix}-page",
    }


def test_joint_student_and_teacher_receive_same_three_images(tmp_path: Path) -> None:
    images = [tmp_path / f"image_{index}.png" for index in range(3)]
    for image in images:
        image.write_bytes(b"image")
    text = product_prompt("Sample title", "sample_category", None, "fabric", image_placeholders=False)
    prompt = structured_prompt([image.as_posix() for image in images], text)
    export_path = tmp_path / "joint_train.jsonl"
    write_jsonl(
        export_path,
        [
            {
                "data_source": "vlm_product_audit",
                "prompt": prompt,
                "opd_enabled": False,
                "reward_model": {"style": "rule", "ground_truth": json.dumps(_pass_target())},
                "extra_info": {
                    "dataset_stage": "grpo",
                    "training_stage": "joint",
                    "split": "train",
                    "sample_id": "grpo-1",
                    "lineage": _lineage("grpo"),
                },
            },
            {
                "data_source": "vlm_product_audit",
                "prompt": prompt,
                "teacher_prompt": prompt,
                "opd_enabled": True,
                "reward_model": {"style": "rule", "ground_truth": json.dumps(_pass_target())},
                "extra_info": {
                    "dataset_stage": "opd",
                    "training_stage": "joint",
                    "split": "train",
                    "sample_id": "opd-1",
                    "lineage": _lineage("opd"),
                },
            },
        ],
    )
    assert validate_joint_export(export_path, "train").sample_ids == {"grpo-1", "opd-1"}
