import json
from pathlib import Path

import pytest

from src.common import load_yaml, write_jsonl
from src.data.export_verl_sft import build_rows
from src.training.opd_tokens import semantic_token_weights
from src.models.audit_protocol import product_prompt, structured_prompt
from src.training.runtime import (
    assert_joint_config,
    assert_verl_sft_config,
    build_verl_command,
    build_verl_sft_command,
    validate_joint_export,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_semantic_opd_weights_only_distill_three_semantic_fields() -> None:
    text = (
        '{"decision":"reject","violation_type":"image_quality",'
        '"issue_subtype":"blur","evidence":"main"}'
    )
    weights = semantic_token_weights(CharacterTokenizer(), [ord(char) for char in text], True)
    assert max(weights[text.index("image_quality") : text.index("image_quality") + len("image_quality")]) == 1.5
    assert max(weights[text.index("blur") : text.index("blur") + len("blur")]) == 1.5
    assert not any(weights[text.index("evidence") :])
    assert not any(semantic_token_weights(CharacterTokenizer(), [ord(char) for char in text], False))


def test_verl_commands_enable_real_sft_and_joint_opd() -> None:
    sft_command = build_verl_sft_command(load_yaml(PROJECT_ROOT / "configs" / "sft.yaml"))
    assert "-m" in sft_command
    assert "verl.trainer.sft_trainer" in sft_command
    assert "checkpoint.save_contents=[\"model\",\"optimizer\",\"extra\",\"hf_model\"]" in sft_command
    assert "+model.freeze_vision_encoder=True" in sft_command
    assert "+model.train_mm_projector=True" in sft_command
    assert "data.custom_cls.name=SemanticMultiTurnSFTDataset" in sft_command
    assert "trainer.resume_mode=auto" in sft_command
    assert "trainer.logger=[\"console\",\"file\"]" in sft_command
    assert "checkpoint.load_contents=[\"model\",\"optimizer\",\"extra\",\"hf_model\"]" in sft_command

    joint_command = build_verl_command(load_yaml(PROJECT_ROOT / "configs" / "joint.yaml"))
    assert "algorithm.adv_estimator=grpo" in joint_command
    assert "distillation.enabled=True" in joint_command
    assert "distillation.distillation_loss.loss_mode=forward_kl_topk" in joint_command
    assert "distillation.distillation_loss.topk=64" in joint_command
    assert "distillation.distillation_loss.use_task_rewards=True" in joint_command
    assert "trainer.resume_mode=auto" in joint_command
    assert "trainer.logger=[\"console\",\"file\"]" in joint_command
    assert "trainer.max_actor_ckpt_to_keep=3" in joint_command
    assert not any("lora_adapter_path" in item for item in joint_command)


def test_joint_config_rejects_distillation_without_task_rewards() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "joint.yaml")
    config["opd"]["use_task_rewards"] = False
    with pytest.raises(ValueError, match="task rewards"):
        assert_joint_config(config)


def test_sft_projector_requires_frozen_vision_encoder() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "sft.yaml")
    config["freeze_vision_encoder"] = False
    with pytest.raises(ValueError, match="freeze_vision_encoder"):
        assert_verl_sft_config(config)

def _message(images: list[Path]) -> list[dict]:
    text = product_prompt("Sample title", "sample_category", None, "fabric", image_placeholders=False)
    return structured_prompt([image.as_posix() for image in images], text)


def _joint_row(
    images: list[Path],
    dataset_stage: str,
    split: str,
    sample_id: str,
) -> dict:
    row = {
        "data_source": "vlm_product_audit",
        "prompt": _message(images),
        "opd_enabled": dataset_stage == "opd",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"decision": "reject", "violation_type": "title_mismatch", "issue_subtype": None, "evidence": None}
            ),
        },
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
        row["teacher_prompt"] = _message(images)
    return row


def test_joint_export_gives_student_and_teacher_the_same_three_images(tmp_path: Path) -> None:
    images = [tmp_path / f"image_{index}.png" for index in range(3)]
    for image in images:
        image.write_bytes(b"image")
    export_path = tmp_path / "joint_train.jsonl"
    write_jsonl(
        export_path,
        [
            _joint_row(images, "grpo", "train", "rl-1"),
            _joint_row(images, "opd", "train", "opd-1"),
        ],
    )
    ids = validate_joint_export(export_path, "train")
    assert ids.sample_ids == {"rl-1", "opd-1"}


def test_verl_sft_converter_preserves_three_image_placeholders_and_target(tmp_path: Path) -> None:
    images = [tmp_path / f"image_{index}.png" for index in range(3)]
    for image in images:
        image.write_bytes(b"image")
    source = tmp_path / "sft_train.jsonl"
    write_jsonl(
        source,
        [
            {
                "sample_id": "sft-1",
                "dataset_stage": "sft",
                "split": "train",
                "images": [image.as_posix() for image in images],
                "conversations": [
                    {
                        "from": "human",
                        "value": product_prompt(
                            "Sample title",
                            "sample_category",
                            None,
                            "fabric",
                            image_placeholders=True,
                        ),
                    },
                    {
                        "from": "gpt",
                        "value": '{"decision":"pass","violation_type":"pass","issue_subtype":null,"evidence":null}',
                    },
                ],
            }
        ],
    )
    rows = build_rows(source)
    assert rows[0]["images"] == [image.as_posix() for image in images]
    assert rows[0]["messages"][0]["content"].count("<image>") == 3
    assert rows[0]["messages"][1]["role"] == "assistant"
    assert set(rows[0]) == {"messages", "images"}
