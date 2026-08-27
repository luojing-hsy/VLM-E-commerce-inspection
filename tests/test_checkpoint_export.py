from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.checkpoint_export import merge_peft_state_dict


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
