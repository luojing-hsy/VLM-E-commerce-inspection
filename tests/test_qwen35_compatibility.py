from __future__ import annotations

from pathlib import Path

import pytest

from src.common import load_yaml
from src.models import hf_loader

from src.training import runtime

ROOT = Path(__file__).resolve().parents[1]


def test_default_training_configs_use_qwen35_checkpoints() -> None:
    eval_config = load_yaml(ROOT / "configs" / "eval.yaml")
    sft = load_yaml(ROOT / "configs" / "sft.yaml")
    grpo = load_yaml(ROOT / "configs" / "grpo.yaml")
    smoke = load_yaml(ROOT / "configs" / "grpo_smoke.yaml")

    assert sft["model_name_or_path"].endswith("/models/Qwen3.5-4B")
    assert sft["output_dir"] == "outputs/sft_qwen35_4b"
    assert grpo["model_name_or_path"] == "outputs/sft_qwen35_4b/latest/huggingface"
    assert grpo["source_dataset"] == "data/GRPO/train.jsonl"

    assert eval_config["require_gated_deltanet_kernels"] is True
    assert grpo["override_config"]["attn_implementation"] == "sdpa"
    assert smoke["override_config"]["attn_implementation"] == "sdpa"

    assert eval_config["target_class_prior"]["pass"] == 0.60

def test_qwen35_requires_both_fast_gated_deltanet_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"fla"}
    monkeypatch.setattr(
        hf_loader,
        "find_spec",
        lambda package: object() if package in available else None,
    )

    with pytest.raises(RuntimeError, match="causal_conv1d"):
        hf_loader.require_fast_gated_deltanet_kernels("qwen3_5")


def test_non_qwen35_model_does_not_require_gated_deltanet_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_loader, "find_spec", lambda package: None)

    hf_loader.require_fast_gated_deltanet_kernels("qwen3_vl")


def test_generic_loader_uses_auto_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls.append((path, kwargs))
            return object()

    monkeypatch.setattr(hf_loader, "AutoModelForImageTextToText", FakeAutoModel)
    model = hf_loader.load_multimodal_model("checkpoint", dtype="bf16", device_map="auto")


def test_evaluation_entrypoints_do_not_hardcode_qwen3vl_model_class() -> None:
    for relative in (
        "src/evaluation/predict.py",
        "src/evaluation/evaluate_direct.py",
    ):
        assert "Qwen3VLForConditionalGeneration" not in (ROOT / relative).read_text(encoding="utf-8")


def test_qwen35_grpo_rejects_legacy_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "transformers": "5.16.1",
        "verl": "0.8.0",
        "vllm": "0.12.0",
    }
    monkeypatch.setattr(runtime, "package_version", versions.__getitem__)

    with pytest.raises(RuntimeError, match="vllm>=0.20.2"):
        runtime.assert_qwen35_stack("qwen3_5", require_rollout=True)

    runtime.assert_qwen35_stack("qwen3_vl", require_rollout=True)


def test_training_stack_is_pinned_to_qwen35_compatible_versions() -> None:
    requirements = (ROOT / "scripts" / "training-requirements.txt").read_text(encoding="utf-8")
    kernels = (ROOT / "scripts" / "training-kernel-requirements.txt").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_training_stack.py").read_text(encoding="utf-8")

    assert "torch==2.11.0" in requirements
    assert "transformers==5.16.1" in requirements
    assert "verl==0.8.0" in requirements
    assert "vllm==0.20.2" in requirements
    assert "flash-linear-attention==0.4.2" in kernels
    assert "causal-conv1d==1.7.0" in kernels
    assert "Qwen3_5ForConditionalGeneration" in verifier
    assert "Qwen3VLForConditionalGeneration" not in verifier
    assert 'torch.version.cuda != "12.9"' in verifier
