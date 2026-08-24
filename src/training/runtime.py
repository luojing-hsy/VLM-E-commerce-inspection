from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.common import load_yaml, read_jsonl, sha256_file


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
        image = row.get("image")
        conversations = row.get("conversations")
        if not isinstance(image, str) or not Path(image).exists():
            raise FileNotFoundError(f"SFT image does not exist: {image}")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"invalid SFT conversations: {sample_id}")
        if "<image>" not in str(conversations[0].get("value", "")):
            raise ValueError(f"SFT prompt is missing image token: {sample_id}")
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
        crop_images = row.get("crop_images")
        if not isinstance(full_image, str) or not Path(full_image).exists():
            raise FileNotFoundError(f"OPD full image does not exist: {full_image}")
        if not isinstance(crop_images, list) or not crop_images:
            raise ValueError(f"OPD row has no crop images: {sample_id}")
        missing_crops = [value for value in crop_images if not Path(value).exists()]
        if missing_crops:
            raise FileNotFoundError(f"OPD crop image does not exist: {missing_crops[0]}")
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
    global_batch = int(config.get("global_train_batch_size", 0))
    micro_batch = int(config.get("per_device_train_batch_size", 0))
    world_size = int(config.get("n_gpus_per_node", 0)) * int(config.get("nnodes", 0))
    if min(global_batch, micro_batch, world_size) < 1:
        raise ValueError("veRL SFT batch sizes and world size must be positive")
    if global_batch % (micro_batch * world_size):
        raise ValueError("global SFT batch size must divide into micro batch size * world size")


def assert_joint_config(config: dict) -> None:
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
        if not prompt or "<image>" not in prompt[0].get("content", "") or not images or len(images) != 1:
            raise ValueError(f"invalid multimodal veRL row: {sample_id}")
        if not Path(images[0]).exists():
            raise FileNotFoundError(f"veRL image does not exist: {images[0]}")
        ground_truth = json.loads(reward_model.get("ground_truth", ""))
        if not isinstance(ground_truth, dict) or "decision" not in ground_truth:
            raise ValueError(f"invalid veRL ground truth: {sample_id}")
        _record_lineage(ids, extra_info, sample_id, path)
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
    paths = [part.get("image") for part in content if isinstance(part, dict) and part.get("type") == "image"]
    if not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("multimodal prompt must contain image paths")
    if not any(isinstance(part, dict) and part.get("type") == "text" for part in content):
        raise ValueError("multimodal prompt must contain text")
    return paths


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
        if len(student_images) != 1:
            raise ValueError(f"joint student must see exactly one full page: {sample_id}")
        for image_path in student_images:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"joint student image does not exist: {image_path}")

        opd_enabled = row.get("opd_enabled")
        if not isinstance(opd_enabled, bool):
            raise ValueError(f"joint row {sample_id} is missing boolean opd_enabled")
        if opd_enabled:
            if dataset_stage != "opd":
                raise ValueError(f"only OPD subpool rows may enable distillation: {sample_id}")
            teacher_images = _structured_image_paths(row.get("teacher_prompt"))
            if len(teacher_images) < 2 or teacher_images[0] != student_images[0]:
                raise ValueError(f"teacher prompt must contain the full page followed by crops: {sample_id}")
            for image_path in teacher_images:
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"joint teacher image does not exist: {image_path}")
        elif dataset_stage != "grpo":
            raise ValueError(f"OPD subpool row must enable distillation: {sample_id}")

        ground_truth = json.loads(row.get("reward_model", {}).get("ground_truth", ""))
        if not isinstance(ground_truth, dict) or "decision" not in ground_truth:
            raise ValueError(f"invalid joint ground truth: {sample_id}")
        _record_lineage(ids, extra_info, sample_id, path)

    if subpools != {"grpo", "opd"}:
        raise ValueError(f"joint {expected_split} export must include GRPO and OPD subpools")
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


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_verl_sft_command(config: dict, executable: str = "python") -> list[str]:
    assert_verl_sft_config(config)
    overrides = {
        "data.train_files": config["dataset_parquet"],
        "data.val_files": config["validation_dataset_parquet"],
        "data.train_batch_size": config["global_train_batch_size"],
        "data.micro_batch_size_per_gpu": config["per_device_train_batch_size"],
        "data.max_length": config["max_sequence_length"],
        "data.max_token_len_per_gpu": config["max_token_len_per_gpu"],
        "data.pad_mode": "no_padding",
        "data.truncation": "error",
        "data.use_dynamic_bsz": True,
        "model.path": config["model_name_or_path"],
        "model.enable_gradient_checkpointing": config["gradient_checkpointing"],
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
        "trainer.save_freq": config["save_freq"],
        "trainer.test_freq": config["test_freq"],
        "trainer.logger": ["console"],
        "trainer.project_name": config["project_name"],
        "trainer.experiment_name": config["experiment_name"],
        "trainer.total_epochs": config["num_train_epochs"],
        "checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
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
        "actor_rollout_ref.model.path": config["model_name_or_path"],
        "actor_rollout_ref.model.lora_rank": config["lora_r"],
        "actor_rollout_ref.model.lora_alpha": config["lora_alpha"],
        "actor_rollout_ref.model.target_modules": config["lora_target_modules"],
        "actor_rollout_ref.model.use_remove_padding": True,
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
        "reward.custom_reward_function.reward_kwargs.reward_config_path": config.get(
            "reward_config_path", "configs/joint.yaml"
        ),
        "trainer.logger": ["console"],
        "trainer.project_name": config["project_name"],
        "trainer.experiment_name": config["experiment_name"],
        "trainer.n_gpus_per_node": config["n_gpus_per_node"],
        "trainer.nnodes": config["nnodes"],
        "trainer.default_local_dir": config["output_dir"],
        "trainer.save_freq": config["save_freq"],
        "trainer.test_freq": config["test_freq"],
        "trainer.total_epochs": config["total_epochs"],
    }
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


def launch_verl(config: dict) -> None:
    if platform.system() != "Linux":
        raise RuntimeError("veRL GPU training must be launched from a supported Linux CUDA environment")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("veRL is not installed; prepare-only does not install or download it")
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
    subprocess.run(build_verl_command(config, sys.executable), check=True)


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
    from src.data.export_verl_sft import write_parquet

    write_parquet(config["dataset"], config["dataset_parquet"])
    write_parquet(config["validation_dataset"], config["validation_dataset_parquet"])
    subprocess.run(build_verl_sft_command(config, sys.executable), check=True)
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
