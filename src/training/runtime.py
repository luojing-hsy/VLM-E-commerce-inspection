from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Iterable

from packaging.version import Version
from src.data.export_grpo_from_joint import validate_image_files

from src.models.audit_protocol import PROMPT, assert_model_text_is_sanitized, validate_prediction_dict, validate_product_prompt
from src.models.hf_loader import QWEN35_MODEL_TYPES, require_fast_gated_deltanet_kernels
from src.common import load_yaml, read_jsonl, sha256_file
from src.training.checkpoint_export import normalize_hf_checkpoint


@dataclass
class ExportIds:
    sample_ids: set[str]
    source_product_ids: set[str]
    source_image_ids: set[str]
    derived_image_ids: set[str]


def _empty_export_ids() -> ExportIds:
    return ExportIds(set(), set(), set(), set())


def _record_lineage(ids: ExportIds, row: dict, sample_id: str, path: str | Path) -> None:
    lineage = row.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError(f"missing lineage in {path}: {sample_id}")
    product_ids = lineage.get("source_product_ids")
    image_ids = lineage.get("source_image_ids")
    derived_image_id = lineage.get("derived_image_id")
    if not isinstance(product_ids, list) or not product_ids:
        raise ValueError(f"missing source_product_ids in {path}: {sample_id}")
    if not isinstance(image_ids, list) or not image_ids:
        raise ValueError(f"missing source_image_ids in {path}: {sample_id}")
    if not isinstance(derived_image_id, str) or not derived_image_id:
        raise ValueError(f"missing derived_image_id in {path}: {sample_id}")
    ids.source_product_ids.update(str(value) for value in product_ids)
    ids.source_image_ids.update(str(value) for value in image_ids)
    if derived_image_id in ids.derived_image_ids:
        raise ValueError(f"duplicate derived_image_id in {path}: {derived_image_id}")
    ids.derived_image_ids.add(derived_image_id)


def _record_sample_id(ids: ExportIds, sample_id: object, path: str | Path) -> str:
    if not isinstance(sample_id, str) or not sample_id or sample_id in ids.sample_ids:
        raise ValueError(f"duplicate or missing sample_id in {path}: {sample_id}")
    ids.sample_ids.add(sample_id)
    return sample_id


def _assert_export_name(path: str | Path, expected_split: str) -> None:
    if not Path(path).stem.endswith(f"_{expected_split}"):
        raise ValueError(f"export path must end with _{expected_split}.jsonl: {path}")


def assert_disjoint_exports(train: ExportIds, validation: ExportIds, stage: str) -> None:
    checks = (
        ("sample IDs", train.sample_ids & validation.sample_ids),
        ("source product IDs", train.source_product_ids & validation.source_product_ids),
        ("source image IDs", train.source_image_ids & validation.source_image_ids),
        ("derived image IDs", train.derived_image_ids & validation.derived_image_ids),
    )
    for label, overlap in checks:
        if overlap:
            raise ValueError(f"{stage} train/validation {label} overlap: {sorted(overlap)[:5]}")


def validate_sft_export(path: str | Path, expected_split: str) -> ExportIds:
    _assert_export_name(path, expected_split)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"SFT {expected_split} export is empty: {path}")
    ids = _empty_export_ids()
    for row in rows:
        sample_id = _record_sample_id(ids, row.get("sample_id"), path)
        if row.get("dataset_stage") != "sft":
            raise ValueError(f"SFT row {sample_id} has wrong dataset_stage={row.get('dataset_stage')}")
        if row.get("split") != expected_split:
            raise ValueError(
                f"SFT row {sample_id} has split={row.get('split')}; expected split={expected_split}"
            )
        images = row.get("images")
        conversations = row.get("conversations")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"SFT row must contain main and two detail images: {sample_id}")
        missing_images = [path for path in images if not isinstance(path, str) or not Path(path).exists()]
        if missing_images:
            raise FileNotFoundError(f"SFT image does not exist: {missing_images[0]}")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"invalid SFT conversations: {sample_id}")
        if conversations[0].get("from") != "human":
            raise ValueError(f"SFT prompt is not canonical: {sample_id}")
        validate_product_prompt(conversations[0].get("value"), image_placeholders=3)
        if conversations[1].get("from") != "gpt" or not isinstance(conversations[1].get("value"), str):
            raise ValueError(f"invalid SFT target message: {sample_id}")
        validate_prediction_dict(json.loads(conversations[1]["value"]))
        _record_lineage(ids, row, sample_id, path)
    return ids


def validate_opd_export(path: str | Path, expected_split: str) -> ExportIds:
    _assert_export_name(path, expected_split)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"OPD {expected_split} export is empty: {path}")
    ids = _empty_export_ids()
    for row in rows:
        sample_id = _record_sample_id(ids, row.get("sample_id"), path)
        if row.get("dataset_stage") != "opd":
            raise ValueError(f"OPD row {sample_id} has wrong dataset_stage={row.get('dataset_stage')}")
        if row.get("split") != expected_split:
            raise ValueError(
                f"OPD row {sample_id} has split={row.get('split')}; expected split={expected_split}"
            )
        full_image = row.get("full_image")
        if not isinstance(full_image, str) or not Path(full_image).exists():
            raise FileNotFoundError(f"OPD full image does not exist: {full_image}")
        _record_lineage(ids, row, sample_id, path)
    return ids


def assert_verl_grpo_config(config: dict) -> None:
    if config.get("framework") != "verl":
        raise ValueError("GRPO requires framework=verl")
    rollout_n = int(config["rollout_n"])
    train_batch_size = int(config["train_batch_size"])
    mini_batch_size = int(config["ppo_mini_batch_size"])
    trajectories = rollout_n * train_batch_size
    if rollout_n <= 1:
        raise ValueError("veRL GRPO requires rollout_n > 1")
    if train_batch_size <= 0 or mini_batch_size <= 0 or trajectories % mini_batch_size:
        raise ValueError("ppo_mini_batch_size must divide train_batch_size * rollout_n")
    if int(config["rollout_tensor_model_parallel_size"]) > int(config["n_gpus_per_node"]):
        raise ValueError("rollout tensor parallel size cannot exceed GPUs per node")


def assert_verl_sft_config(config: dict) -> None:
    if config.get("framework") != "verl":
        raise ValueError("SFT requires framework=verl")
    freeze_vision_encoder = config.get("freeze_vision_encoder", True)
    train_mm_projector = config.get("train_mm_projector", False)
    if not isinstance(freeze_vision_encoder, bool) or not isinstance(train_mm_projector, bool):
        raise ValueError("SFT vision/projector flags must be boolean")
    if train_mm_projector and not freeze_vision_encoder:
        raise ValueError("SFT MM projector training requires freeze_vision_encoder=true")
    global_batch = int(config.get("global_train_batch_size", 0))
    micro_batch = int(config.get("per_device_train_batch_size", 0))
    world_size = int(config.get("n_gpus_per_node", 0)) * int(config.get("nnodes", 0))
    if min(global_batch, micro_batch, world_size) < 1:
        raise ValueError("veRL SFT batch sizes and world size must be positive")
    if global_batch % (micro_batch * world_size):
        raise ValueError("global SFT batch size must divide into micro batch size * world size")


def assert_joint_config(config: dict) -> None:
    _assert_resume_config(config)

def _assert_resume_config(config: dict) -> None:
    resume_mode = config.get("resume_mode", "auto")
    if resume_mode not in {"auto", "disable", "resume_path"}:
        raise ValueError("resume_mode must be auto, disable, or resume_path")
    if resume_mode == "resume_path":
        resume_path = config.get("resume_from_path")
        if not resume_path or not Path(resume_path).is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    assert_verl_grpo_config(config)
    opd = config.get("opd")
    if not isinstance(opd, dict) or opd.get("loss_mode") != "forward_kl_topk":
        raise ValueError("joint training requires opd.loss_mode=forward_kl_topk")
    if int(opd.get("top_k_logits", 0)) < 1:
        raise ValueError("joint training requires a positive opd.top_k_logits")
    if float(opd.get("loss_coefficient", 0.0)) <= 0:
        raise ValueError("joint training requires a positive opd.loss_coefficient")
    if not opd.get("use_task_rewards") or opd.get("use_policy_gradient"):
        raise ValueError("joint training requires direct OPD KL combined with task rewards")
    teacher_world_size = int(config["teacher_n_gpus_per_node"]) * int(config["teacher_nnodes"])
    teacher_tp = int(config["teacher_tensor_model_parallel_size"])
    if teacher_world_size < 1 or teacher_world_size % teacher_tp:
        raise ValueError("teacher world size must be positive and divisible by teacher tensor parallel size")
    if "file" in config.get("loggers", []) and not config.get("log_file"):
        raise ValueError("joint file logging requires log_file")


def assert_lora_targets(module_names: Iterable[str], targets: Iterable[str]) -> list[str]:
    target_suffixes = tuple(targets)
    matches = [name for name in module_names if name.endswith(target_suffixes)]
    forbidden = ("vision", "visual", "merger", "projector", "mm_projector")
    bad = [name for name in matches if any(part in name.lower() for part in forbidden)]
    if not matches:
        raise ValueError(f"no modules matched LoRA targets {list(targets)}")
    if bad:
        raise ValueError(f"LoRA targets unexpectedly include visual/MM modules: {bad[:5]}")
    return matches


def assert_standard_lora_config(config: dict) -> None:
    if config.get("load_in_4bit"):
        raise ValueError("standard LoRA must not enable 4-bit loading")
    if str(config.get("quantization", "none")).lower() != "none":
        raise ValueError("standard LoRA requires quantization=none")
    if str(config.get("precision", "")).lower() != "bf16":
        raise ValueError("standard LoRA runs must explicitly use precision=bf16")


def dependency_report(stage: str) -> dict[str, bool]:
    required = ["torch", "transformers", "accelerate", "peft"]
    if stage in {"grpo", "joint"}:
        required.extend(["verl", "ray", "vllm"])
    return {name: importlib.util.find_spec(name) is not None for name in required}


def validate_verl_export(path: str | Path, expected_split: str) -> ExportIds:
    _assert_export_name(path, expected_split)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"veRL {expected_split} export is empty: {path}")
    ids = _empty_export_ids()
    for row in rows:
        extra_info = row.get("extra_info", {})
        sample_id = _record_sample_id(ids, extra_info.get("sample_id"), path)
        if extra_info.get("dataset_stage") != "grpo":
            raise ValueError(f"GRPO row {sample_id} has wrong dataset_stage={extra_info.get('dataset_stage')}")
        if row.get("data_source") != "vlm_product_audit" or extra_info.get("split") != expected_split:
            raise ValueError(f"invalid veRL {expected_split} row: {sample_id}")
        prompt = row.get("prompt")
        images = row.get("images")
        reward_model = row.get("reward_model", {})
        prompt_images = _structured_image_paths(prompt)
        if not isinstance(images, list) or len(images) != 3 or prompt_images != images:
            raise ValueError(f"veRL row must contain the same three product images in prompt and images: {sample_id}")
        missing_images = [path for path in images if not isinstance(path, str) or not Path(path).exists()]
        if missing_images:
            raise FileNotFoundError(f"veRL image does not exist: {missing_images[0]}")
        validate_prediction_dict(json.loads(reward_model.get("ground_truth", "")))
        _record_lineage(ids, extra_info, sample_id, path)
    validate_image_files(rows)
    return ids


def _structured_image_paths(messages: object) -> list[str]:
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("multimodal prompt must contain exactly one user message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("multimodal prompt must be a user message")
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("multimodal prompt content must be structured")
    expected_labels = ("main: ", "\ndetail:1: ", "\ndetail:2: ")
    if len(content) != 7:
        raise ValueError("multimodal prompt must contain three role-tagged images and one product prompt")
    paths: list[object] = []
    for index, label in enumerate(expected_labels):
        text_part = content[index * 2]
        image_part = content[index * 2 + 1]
        if (
            not isinstance(text_part, dict)
            or text_part.get("type") != "text"
            or text_part.get("text") != label
            or not isinstance(image_part, dict)
            or image_part.get("type") != "image"
        ):
            raise ValueError("multimodal prompt image roles must be main, detail:1, detail:2 in order")
        paths.append(image_part.get("image"))
    final_text = content[-1]
    if not isinstance(final_text, dict) or final_text.get("type") != "text":
        raise ValueError("multimodal prompt must end with one product prompt")
    if not all(isinstance(path, str) and path for path in paths):
        raise ValueError("multimodal prompt must contain image paths")
    validate_product_prompt(final_text.get("text"), image_placeholders=0)
    assert_model_text_is_sanitized(messages)
    return [path for path in paths if isinstance(path, str)]


def validate_joint_export(path: str | Path, expected_split: str) -> ExportIds:
    _assert_export_name(path, expected_split)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"joint {expected_split} export is empty: {path}")
    ids = _empty_export_ids()
    subpools: set[str] = set()
    for row in rows:
        extra_info = row.get("extra_info", {})
        sample_id = _record_sample_id(ids, extra_info.get("sample_id"), path)
        dataset_stage = extra_info.get("dataset_stage")
        if dataset_stage not in {"grpo", "opd"} or extra_info.get("training_stage") != "joint":
            raise ValueError(f"joint row {sample_id} has invalid dataset/training stage")
        if extra_info.get("split") != expected_split:
            raise ValueError(f"joint row {sample_id} has split={extra_info.get('split')}")
        subpools.add(str(dataset_stage))

        student_images = _structured_image_paths(row.get("prompt"))
        if len(student_images) != 3:
            raise ValueError(f"joint student must see main and two detail images: {sample_id}")
        for image_path in student_images:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"joint student image does not exist: {image_path}")

        opd_enabled = row.get("opd_enabled")
        if not isinstance(opd_enabled, bool):
            raise ValueError(f"joint row {sample_id} is missing boolean opd_enabled")
        teacher_images = _structured_image_paths(row.get("teacher_prompt"))
        if teacher_images != student_images or row.get("teacher_prompt") != row.get("prompt"):
            raise ValueError(f"teacher and student must receive the same three images and prompt: {sample_id}")
        if opd_enabled:
            if dataset_stage != "opd":
                raise ValueError(f"only OPD subpool rows may enable distillation: {sample_id}")
        elif dataset_stage != "grpo":
            raise ValueError(f"OPD subpool row must enable distillation: {sample_id}")

        validate_prediction_dict(json.loads(row.get("reward_model", {}).get("ground_truth", "")))
        _record_lineage(ids, extra_info, sample_id, path)

    if "grpo" not in subpools:
        raise ValueError(f"joint {expected_split} export must include a GRPO subpool")
    return ids


def assert_pipeline_stage_isolation(manifest_root: Path) -> None:
    stage_image_ids: dict[str, set[str]] = {}
    for stage in ("sft", "grpo", "opd"):
        image_ids: set[str] = set()
        for split in ("train", "validation"):
            path = manifest_root / f"{stage}_{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing {stage} split export: {path}")
            for row in read_jsonl(path):
                container = row.get("extra_info", row)
                lineage = container.get("lineage", {})
                if lineage.get("dataset_stage") != stage:
                    raise ValueError(f"{path} contains lineage from another dataset stage")
                image_ids.update(str(value) for value in lineage.get("source_image_ids", []))
        stage_image_ids[stage] = image_ids

    test_path = manifest_root / "samples_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing fixed test manifest: {test_path}")
    test_rows = read_jsonl(test_path)
    if any(row.get("dataset_stage") != "test" or row.get("split") != "test" for row in test_rows):
        raise ValueError("fixed test manifest contains non-test rows")
    stage_image_ids["test"] = {
        str(image_id)
        for row in test_rows
        for image_id in row.get("source_image_ids", [])
    }

    stages = tuple(stage_image_ids)
    for index, left in enumerate(stages):
        for right in stages[index + 1 :]:
            overlap = stage_image_ids[left] & stage_image_ids[right]
            if overlap:
                raise ValueError(
                    f"{left}/{right} source_image_id overlap: {sorted(overlap)[:5]}"
                )


def validate_stage_config(config_path: str | Path, stage: str) -> dict:
    config = load_yaml(config_path)
    if config.get("stage") != stage:
        raise ValueError(f"expected stage={stage!r} in {config_path}")
    if stage in {"sft", "opd", "grpo", "joint"}:
        assert_standard_lora_config(config)
    if stage == "sft":
        assert_verl_sft_config(config)
    if stage == "grpo":
        assert_verl_grpo_config(config)
    if stage == "joint":
        assert_joint_config(config)
    dataset = Path(config["dataset"])
    if not dataset.exists():
        raise FileNotFoundError(f"dataset export does not exist: {dataset}; run the data pipeline first")
    if stage in {"sft", "opd", "grpo", "joint"}:
        validation_dataset_value = config.get("validation_dataset")
        if not validation_dataset_value:
            raise ValueError(f"{stage} requires a split-specific validation_dataset")
        validation_dataset = Path(validation_dataset_value)
        if not validation_dataset.exists():
            raise FileNotFoundError(
                f"validation export does not exist: {validation_dataset}; run the data pipeline first"
            )
        validators = {
            "sft": validate_sft_export,
            "opd": validate_opd_export,
            "grpo": validate_verl_export,
            "joint": validate_joint_export,
        }
        validator = validators[stage]
        train_ids = validator(dataset, "train")
        validation_ids = validator(validation_dataset, "validation")
        assert_disjoint_exports(train_ids, validation_ids, stage)
        if (dataset.parent / "source_images.jsonl").exists():
            assert_pipeline_stage_isolation(dataset.parent)
    return config

def assert_qwen35_stack(model_type: str, *, require_rollout: bool) -> None:

    if model_type not in QWEN35_MODEL_TYPES:
        return
    minimum_versions = {
        "transformers": "5.16.0",
        "verl": "0.8.0",
    }
    if require_rollout:
        minimum_versions["vllm"] = "0.20.2"
    for package, minimum in minimum_versions.items():
        actual = package_version(package)
        if Version(actual) < Version(minimum):
            raise RuntimeError(
                f"Qwen3.5 {'GRPO' if require_rollout else 'SFT'} requires "
                f"{package}>={minimum}, got {package}=={actual}; run bash scripts/setup.sh"
            )


def assert_training_model_environment(config: dict, *, require_rollout: bool) -> str:
    from transformers import AutoConfig

    model_path = str(config["model_name_or_path"])
    hf_config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=Path(model_path).exists(),
    )
    model_type = str(hf_config.model_type)
    assert_qwen35_stack(model_type, require_rollout=require_rollout)
    if config.get("require_gated_deltanet_kernels", True):
        require_fast_gated_deltanet_kernels(model_type)
    return model_type



def _hydra_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{key}:{_hydra_value(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_verl_sft_command(config: dict, executable: str = "python") -> list[str]:
    assert_verl_sft_config(config)
    overrides = {
        "data.train_files": config["dataset_parquet"],
        "data.val_files": config["validation_dataset_parquet"],
        "data.train_batch_size": config["global_train_batch_size"],
        # VLM_PRODUCT_AUDIT_SFT_REQUEST_CONFIG_V1
        "data.train_max_samples": config.get("train_max_samples", -1),
        "data.val_max_samples": config.get("validation_max_samples", -1),
        "data.enable_thinking_key": config.get("enable_thinking_key", "__disabled_enable_thinking__"),
        "data.enable_thinking_default": config.get("enable_thinking_default"),
        "+data.apply_chat_template_kwargs": config.get("apply_chat_template_kwargs", {"enable_thinking": False}),
        "data.micro_batch_size_per_gpu": config["per_device_train_batch_size"],
        "data.max_length": config["max_sequence_length"],
        "data.max_token_len_per_gpu": config["max_token_len_per_gpu"],
        "data.pad_mode": "no_padding",
        "data.custom_cls.path": config["semantic_dataset_class_path"],
        "data.custom_cls.name": "SemanticMultiTurnSFTDataset",
        "data.truncation": "error",
        "data.use_dynamic_bsz": True,
        "model.path": config["model_name_or_path"],
        "+model.override_config.attn_implementation": config.get("override_config", {}).get("attn_implementation", "sdpa"),
        "+model.override_config.min_pixels": config.get("override_config", {}).get("min_pixels", config.get("min_pixels", 784)),
        "+model.override_config.max_pixels": config.get("override_config", {}).get("max_pixels", config.get("max_pixels", 50176)),
        "model.enable_gradient_checkpointing": config["gradient_checkpointing"],
        "+model.freeze_vision_encoder": config.get("freeze_vision_encoder", True),
        "+model.train_mm_projector": config.get("train_mm_projector", False),
        "model.use_remove_padding": True,
        "model.lora_rank": config["lora_r"],
        "model.lora_alpha": config["lora_alpha"],
        "model.target_modules": config["lora_target_modules"],
        "model.exclude_modules": config["lora_exclude_modules"],
        "engine": "fsdp",
        "optim": "fsdp",
        "engine.strategy": "fsdp2",
        "engine.fsdp_size": config["fsdp_size"],
        "engine.dtype": "bfloat16",
        "optim.lr": config["learning_rate"],
        "optim.clip_grad": 1.0,
        "trainer.n_gpus_per_node": config["n_gpus_per_node"],
        "trainer.nnodes": config["nnodes"],
        "trainer.default_local_dir": config["output_dir"],
        "hydra.run.dir": config["output_dir"],
        "trainer.save_freq": config["save_freq"],
        "trainer.test_freq": config["test_freq"],
        "trainer.max_ckpt_to_keep": config["max_ckpt_to_keep"],
        "trainer.resume_mode": config["resume_mode"],
        "trainer.resume_from_path": config.get("resume_from_path"),
        "trainer.logger": config["loggers"],
        "trainer.project_name": config["project_name"],
        "trainer.experiment_name": config["experiment_name"],
        "trainer.total_epochs": config["num_train_epochs"],
        "checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "checkpoint.load_contents": ["model", "optimizer", "extra", "hf_model"],
    }
    distributed = [
        executable, "-m", "torch.distributed.run", "--standalone",
        f"--nnodes={config['nnodes']}", f"--nproc-per-node={config['n_gpus_per_node']}",
        "-m", config["entrypoint"],
    ]
    return [*distributed, *(f"{key}={_hydra_value(value)}" for key, value in overrides.items())]


def build_verl_command(config: dict, executable: str = "python") -> list[str]:
    assert_verl_grpo_config(config)
    overrides = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": config["dataset"],
        "data.val_files": config["validation_dataset"],
        "data.image_key": "images",
        "data.train_batch_size": config["train_batch_size"],
        "data.max_prompt_length": config["max_prompt_length"],
        "data.max_response_length": config["max_response_length"],
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "+data.apply_chat_template_kwargs": config.get("apply_chat_template_kwargs", {"enable_thinking": False}),
        "+data.mm_processor_kwargs": config.get("mm_processor_kwargs", {"min_pixels": config.get("min_pixels", 784), "max_pixels": config.get("max_pixels", 65536)}),
        "+actor_rollout_ref.model.override_config.min_pixels": config.get("min_pixels", 784),
        "+actor_rollout_ref.model.override_config.max_pixels": config.get("max_pixels", 65536),
        "+actor_rollout_ref.model.override_config.attn_implementation": config.get("override_config", {}).get("attn_implementation", "sdpa"),
        "actor_rollout_ref.model.path": config["model_name_or_path"],
        "actor_rollout_ref.model.lora_rank": config["lora_r"],
        "actor_rollout_ref.model.lora_alpha": config["lora_alpha"],
        "actor_rollout_ref.model.target_modules": config["lora_target_modules"],
        "actor_rollout_ref.model.use_remove_padding": config.get("actor_use_remove_padding", True),
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": config["actor_strategy"],
        "actor_rollout_ref.actor.optim.lr": config["learning_rate"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": config["ppo_mini_batch_size"],
        "actor_rollout_ref.actor.ppo_epochs": config["ppo_epochs"],
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": config["ppo_max_token_len_per_gpu"],
        "actor_rollout_ref.actor.grad_clip": config["max_grad_norm"],
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": config["beta"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload": config["actor_param_offload"],
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": config["actor_optimizer_offload"],
        "actor_rollout_ref.rollout.name": config["rollout_backend"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": config["rollout_tensor_model_parallel_size"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": config["rollout_gpu_memory_utilization"],
        "actor_rollout_ref.rollout.n": config["rollout_n"],
        "actor_rollout_ref.rollout.temperature": config["temperature"],
        "actor_rollout_ref.rollout.top_p": config["top_p"],
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.layered_summon": True,
        "actor_rollout_ref.ref.fsdp_config.param_offload": config["ref_param_offload"],
        "reward.custom_reward_function.path": config["reward_function_path"],
        "reward.custom_reward_function.name": config["reward_function_name"],
        "+reward.custom_reward_function.reward_kwargs.reward_config_path": config.get(
            "reward_config_path", "configs/joint.yaml"
        ),
        "trainer.logger": config.get("loggers", ["console"]),
        "trainer.resume_mode": config.get("resume_mode", "auto"),
        "trainer.resume_from_path": config.get("resume_from_path"),
        "trainer.max_actor_ckpt_to_keep": config.get("max_actor_ckpt_to_keep"),
        "trainer.project_name": config["project_name"],
        "trainer.experiment_name": config["experiment_name"],
        "trainer.n_gpus_per_node": config["n_gpus_per_node"],
        "trainer.nnodes": config["nnodes"],
        "trainer.default_local_dir": config["output_dir"],
        "trainer.save_freq": config["save_freq"],
        "trainer.test_freq": config["test_freq"],
        "trainer.total_epochs": config["total_epochs"],
    }
    for config_key, override_key in (
        ("rollout_enforce_eager", "actor_rollout_ref.rollout.enforce_eager"),
        ("rollout_max_model_len", "actor_rollout_ref.rollout.max_model_len"),
        ("rollout_max_num_seqs", "actor_rollout_ref.rollout.max_num_seqs"),
        ("rollout_max_num_batched_tokens", "actor_rollout_ref.rollout.max_num_batched_tokens"),
        ("rollout_agent_num_workers", "actor_rollout_ref.rollout.agent.num_workers"),
    ):
        if config_key in config:
            overrides[override_key] = config[config_key]
    if config.get("stage") == "joint":
        opd = config["opd"]
        overrides.update(
            {
                "distillation.enabled": True,
                "distillation.n_gpus_per_node": config["teacher_n_gpus_per_node"],
                "distillation.nnodes": config["teacher_nnodes"],
                "distillation.teacher_models.teacher_model.model_path": config["teacher_model_path"],
                "distillation.teacher_models.teacher_model.num_replicas": 1,
                "distillation.teacher_models.teacher_model.inference.name": config["rollout_backend"],
                "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size": config[
                    "teacher_tensor_model_parallel_size"
                ],
                "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization": config[
                    "teacher_gpu_memory_utilization"
                ],
                "distillation.distillation_loss.loss_mode": opd["loss_mode"],
                "distillation.distillation_loss.topk": opd["top_k_logits"],
                "distillation.distillation_loss.use_task_rewards": opd["use_task_rewards"],
                "distillation.distillation_loss.distillation_loss_coef": opd["loss_coefficient"],
                "distillation.distillation_loss.use_policy_gradient": opd["use_policy_gradient"],
            }
        )
    if config.get("lora_adapter_path"):
        overrides["actor_rollout_ref.model.lora_adapter_path"] = config["lora_adapter_path"]
    return [executable, "-m", config["entrypoint"], *(f"{key}={_hydra_value(value)}" for key, value in overrides.items())]


def apply_resume_options(config: dict, resume_from: str | None, restart: bool) -> None:
    if resume_from:
        config["resume_mode"] = "resume_path"
        config["resume_from_path"] = str(Path(resume_from).resolve())
    elif restart:
        config["resume_mode"] = "disable"
        config["resume_from_path"] = None


def _tracking_environment(config: dict) -> dict[str, str]:
    env = os.environ.copy()
    if "file" in config.get("loggers", []):
        log_path = Path(config["log_file"]).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env["VERL_FILE_LOGGER_PATH"] = str(log_path)
    return env


def launch_verl(config: dict) -> Path | None:
    if platform.system() != "Linux":
        raise RuntimeError("veRL GPU training must be launched from a supported Linux CUDA environment")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("veRL is not installed; prepare-only does not install or download it")
    model_type = assert_training_model_environment(config, require_rollout=True)
    print(f"training model preflight: model_type={model_type}, rollout=vllm")
    adapter_value = config.get("lora_adapter_path")
    if adapter_value and not Path(adapter_value).exists():
        raise FileNotFoundError(f"SFT LoRA adapter does not exist: {adapter_value}")
    if config.get("stage") == "joint":
        student = Path(config["model_name_or_path"])
        teacher = Path(config["teacher_model_path"])
        if not student.exists():
            raise FileNotFoundError(f"SFT student checkpoint does not exist: {student}")
        if not teacher.exists():
            raise FileNotFoundError(f"frozen SFT teacher checkpoint does not exist: {teacher}")
    subprocess.run(build_verl_command(config, sys.executable), check=True, env=_tracking_environment(config))
    if config.get("stage") == "joint":
        return export_joint_hf_checkpoint(config)
    return None


def record_latest_sft_checkpoint(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    candidates = [
        path
        for path in root.glob("global_step_*")
        if path.is_dir() and (path / "huggingface").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"veRL SFT produced no Hugging Face checkpoint under {root}")

    def step(path: Path) -> int:
        try:
            return int(path.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    latest_checkpoint = max(candidates, key=step)
    normalize_hf_checkpoint(latest_checkpoint / "huggingface")
    alias = root / "latest"
    if alias.is_symlink():
        alias.unlink()
    elif alias.exists():
        raise FileExistsError(f"refusing to replace non-symlink SFT alias: {alias}")
    alias.symlink_to(latest_checkpoint.name, target_is_directory=True)
    return alias / "huggingface"


def launch_verl_sft(config: dict) -> Path:
    if platform.system() != "Linux":
        raise RuntimeError("veRL GPU training must be launched from a supported Linux CUDA environment")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("veRL is not installed; run scripts/setup.sh first")
    model_type = assert_training_model_environment(config, require_rollout=False)
    print(f"training model preflight: model_type={model_type}, gated_deltanet=fast")
    from src.data.export_verl_sft import write_parquet_if_needed

    parquet_pairs = [
        (
            config.get("full_dataset", config["dataset"]),
            config.get("full_dataset_parquet", config["dataset_parquet"]),
        ),
        (
            config.get("full_validation_dataset", config["validation_dataset"]),
            config.get("full_validation_dataset_parquet", config["validation_dataset_parquet"]),
        ),
        (config["dataset"], config["dataset_parquet"]),
        (config["validation_dataset"], config["validation_dataset_parquet"]),
    ]
    seen_pairs = set()
    for source, target in parquet_pairs:
        pair = (str(source), str(target))
        if pair in seen_pairs:
            continue
        write_parquet_if_needed(source, target)
        seen_pairs.add(pair)
    from src.training.sft_checkpoint_policy import ensure_sft_latest_only_checkpoint_hook
    from src.training.sft_progress import ensure_sft_progress_hook

    ensure_sft_progress_hook()
    ensure_sft_latest_only_checkpoint_hook()
    subprocess.run(build_verl_sft_command(config, sys.executable), check=True, env=_tracking_environment(config))
    return record_latest_sft_checkpoint(config["output_dir"])


def write_run_manifest(config_path: str | Path, config: dict) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(config["dataset"])
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "stage": config["stage"],
        "config_sha256": sha256_file(config_path),
        "dataset_sha256": sha256_file(dataset),
        "validation_dataset_sha256": sha256_file(config["validation_dataset"]),
        "dependencies_available": dependency_report(config["stage"]),
        "config": config,
    }
    target = output_dir / "run_manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _latest_joint_actor_checkpoint(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    candidates = [path for path in root.glob("global_step_*/actor") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"veRL Joint produced no actor checkpoint under {root}")
    return max(candidates, key=lambda path: _checkpoint_step(path.parent))


def export_joint_hf_checkpoint(config: dict) -> Path:
    """Merge the newest FSDP actor once and update a loadable ``latest`` alias."""

    output_dir = Path(config["output_dir"]).resolve()
    actor = _latest_joint_actor_checkpoint(output_dir)
    if not (actor / "huggingface").is_dir():
        raise FileNotFoundError(f"Joint actor checkpoint has no Hugging Face metadata: {actor}")
    export_root = Path(config.get("hf_export_root", output_dir / "hf_exports"))
    if not export_root.is_absolute():
        export_root = (Path.cwd() / export_root).resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    target = export_root / actor.parent.name
    if target.exists():
        if not target.is_dir() or not (target / "config.json").is_file():
            raise FileExistsError(f"refusing to reuse incomplete Joint HF export: {target}")
    else:
        command = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            str(config.get("hf_merge_backend", "fsdp")),
            "--local_dir",
            str(actor),
            "--target_dir",
            str(target),
        ]
        subprocess.run(command, check=True, env=_tracking_environment(config))
    weights = [
        path
        for path in target.iterdir()
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".pt"}
    ]
    if not (target / "config.json").is_file() or not weights:
        raise FileNotFoundError(f"Joint HF export is incomplete: {target}")
    alias = Path(config.get("latest_alias", output_dir / "latest"))
    if not alias.is_absolute():
        alias = (Path.cwd() / alias).resolve()
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        alias.unlink()
    elif alias.exists():
        raise FileExistsError(f"refusing to replace non-symlink Joint alias: {alias}")
    alias.symlink_to(Path(os.path.relpath(target, alias.parent)), target_is_directory=True)
    return alias
