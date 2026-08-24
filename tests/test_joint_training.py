import json
from pathlib import Path

import pytest

from src.common import load_yaml, write_jsonl
from src.data.export_verl_sft import build_rows
from src.training.opd_tokens import semantic_token_weights
from src.training.runtime import (
    assert_joint_config,
    build_verl_command,
    build_verl_sft_command,
    validate_joint_export,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_semantic_opd_weights_ignore_protocol_and_bbox_tokens() -> None:
    text = (
        '{"decision":"reject","violation_type":"ATTRIBUTE_CONFLICT",'
        '"field":"model","listed_value":"Model Y","observed_value":"Model X",'
        '"evidence":[{"bbox_norm":[1,2,3,4]}]}'
    )
    weights = semantic_token_weights(CharacterTokenizer(), [ord(char) for char in text], True)
    assert max(weights[text.index("Model X") : text.index("Model X") + len("Model X")]) == 2.0
    assert max(weights[text.index("Model Y") : text.index("Model Y") + len("Model Y")]) == 1.5
    assert not any(weights[text.index("bbox_norm") :])
    assert not any(semantic_token_weights(CharacterTokenizer(), [ord(char) for char in text], False))


def test_verl_commands_enable_real_sft_and_joint_opd() -> None:
    sft_command = build_verl_sft_command(load_yaml(PROJECT_ROOT / "configs" / "sft.yaml"))
    assert "-m" in sft_command
    assert "verl.trainer.sft_trainer" in sft_command
    assert "checkpoint.save_contents=[\"model\",\"optimizer\",\"extra\",\"hf_model\"]" in sft_command

    joint_command = build_verl_command(load_yaml(PROJECT_ROOT / "configs" / "joint.yaml"))
    assert "algorithm.adv_estimator=grpo" in joint_command
    assert "distillation.enabled=True" in joint_command
    assert "distillation.distillation_loss.loss_mode=forward_kl_topk" in joint_command
    assert "distillation.distillation_loss.topk=64" in joint_command
    assert "distillation.distillation_loss.use_task_rewards=True" in joint_command
    assert not any("lora_adapter_path" in item for item in joint_command)


def test_joint_config_rejects_distillation_without_task_rewards() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "joint.yaml")
    config["opd"]["use_task_rewards"] = False
    with pytest.raises(ValueError, match="task rewards"):
        assert_joint_config(config)


def _message(images: list[Path]) -> list[dict]:
    content = [{"type": "image", "image": image.as_posix()} for image in images]
    content.append({"type": "text", "text": "check"})
    return [{"role": "user", "content": content}]


def _joint_row(
    full_page: Path,
    crop: Path,
    dataset_stage: str,
    split: str,
    sample_id: str,
) -> dict:
    row = {
        "data_source": "vlm_product_audit",
        "prompt": _message([full_page]),
        "opd_enabled": dataset_stage == "opd",
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"decision": "reject"})},
        "extra_info": {
            "dataset_stage": dataset_stage,
            "training_stage": "joint",
            "split": split,
            "sample_id": sample_id,
            "lineage": {
                "dataset_stage": dataset_stage,
                "source_product_ids": [f"product-{sample_id}"],
                "source_image_ids": [f"image-{sample_id}"],
                "derived_image_id": f"page-{sample_id}",
            },
        },
    }
    if dataset_stage == "opd":
        row["teacher_prompt"] = _message([full_page, crop])
    return row


def test_joint_export_keeps_student_full_page_and_teacher_privileged_crop(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    crop = tmp_path / "crop.png"
    page.write_bytes(b"page")
    crop.write_bytes(b"crop")
    export_path = tmp_path / "joint_train.jsonl"
    write_jsonl(
        export_path,
        [
            _joint_row(page, crop, "grpo", "train", "rl-1"),
            _joint_row(page, crop, "opd", "train", "opd-1"),
        ],
    )
    ids = validate_joint_export(export_path, "train")
    assert ids.sample_ids == {"rl-1", "opd-1"}


def test_verl_sft_converter_preserves_image_placeholder_and_target(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"page")
    source = tmp_path / "sft_train.jsonl"
    write_jsonl(
        source,
        [
            {
                "sample_id": "sft-1",
                "dataset_stage": "sft",
                "split": "train",
                "image": image.as_posix(),
                "conversations": [
                    {"from": "human", "value": "<image>\ncheck"},
                    {"from": "gpt", "value": '{"decision":"pass"}'},
                ],
            }
        ],
    )
    rows = build_rows(source)
    assert rows[0]["images"] == [image.as_posix()]
    assert rows[0]["messages"][0]["content"].startswith("<image>")
    assert rows[0]["messages"][1]["role"] == "assistant"
