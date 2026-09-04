from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open
from safetensors.torch import save_file

from src.training.checkpoint_export import merge_peft_state_dict, normalize_hf_checkpoint


def test_merge_peft_state_dict_produces_standard_hf_keys_and_applies_lora() -> None:
    base_key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight"
    lora_prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    state_dict = {
        base_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        f"{lora_prefix}.lora_A.default.weight": torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        f"{lora_prefix}.lora_B.default.weight": torch.tensor([[0.5], [1.0]], dtype=torch.bfloat16),
        "base_model.model.model.visual.merger.weight": torch.tensor([[7.0]], dtype=torch.bfloat16),
    }

    converted = merge_peft_state_dict(state_dict, lora_rank=1, lora_alpha=2)

    assert set(converted) == {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.visual.merger.weight",
    }
    expected = torch.tensor([[2.0, 2.0], [5.0, 4.0]], dtype=torch.bfloat16)
    assert torch.equal(converted["model.language_model.layers.0.self_attn.q_proj.weight"], expected)
    assert torch.equal(converted["model.visual.merger.weight"], torch.tensor([[7.0]], dtype=torch.bfloat16))


def test_merge_peft_state_dict_leaves_standard_state_dict_unchanged() -> None:
    state_dict = {"model.layer.weight": torch.tensor([1.0])}

    converted = merge_peft_state_dict(state_dict, lora_rank=16, lora_alpha=32)

    assert converted == state_dict


def test_normalize_hf_checkpoint_removes_raw_export_by_default(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_50"
    source = checkpoint / "huggingface"
    source.mkdir(parents=True)
    prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    state_dict = {
        f"{prefix}.base_layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        f"{prefix}.lora_A.default.weight": torch.tensor([[1.0, 0.0]]),
        f"{prefix}.lora_B.default.weight": torch.tensor([[0.5], [1.0]]),
    }
    shard = "model.safetensors"
    save_file(state_dict, str(source / shard))
    index = {"metadata": {}, "weight_map": {key: shard for key in state_dict}}
    (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (source / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "lora_train_meta.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 2}), encoding="utf-8"
    )

    normalize_hf_checkpoint(source)

    assert not source.with_name("huggingface.peft_raw").exists()
    assert not source.with_name("huggingface.replace_tmp").exists()
    converted_index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert list(converted_index["weight_map"]) == [
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ]
    with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
        expected = torch.tensor([[2.0, 2.0], [5.0, 4.0]])
        assert torch.equal(
            handle.get_tensor("model.language_model.layers.0.self_attn.q_proj.weight"), expected
        )


def test_normalize_hf_checkpoint_handles_unindexed_single_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "global_step_50"
    source = checkpoint / "huggingface"
    source.mkdir(parents=True)
    prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    state_dict = {
        f"{prefix}.base_layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        f"{prefix}.lora_A.default.weight": torch.tensor([[1.0, 0.0]]),
        f"{prefix}.lora_B.default.weight": torch.tensor([[0.5], [1.0]]),
    }
    save_file(state_dict, str(source / "model.safetensors"))
    (source / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "lora_train_meta.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 2}), encoding="utf-8"
    )

    normalize_hf_checkpoint(source)

    converted_index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert list(converted_index["weight_map"]) == [
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ]
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as handle:
        expected = torch.tensor([[2.0, 2.0], [5.0, 4.0]])
        assert torch.equal(
            handle.get_tensor("model.language_model.layers.0.self_attn.q_proj.weight"), expected
        )
