from pathlib import Path

from src.common import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_configs_use_current_dataset_directories() -> None:
    sft = load_yaml(PROJECT_ROOT / "configs" / "sft.yaml")
    assert sft["source_dataset"] == "data/sft/train.jsonl"
    assert sft["validation_source_dataset"] == "data/sft/valid.jsonl"
    assert sft["dataset"].startswith("outputs/sft/")
    assert sft["validation_dataset"].startswith("outputs/sft/")

    grpo = load_yaml(PROJECT_ROOT / "configs" / "grpo.yaml")
    assert grpo["source_dataset"] == "data/GRPO/train.jsonl"
    assert grpo["validation_source_dataset"] == "data/GRPO/valid.jsonl"
    assert grpo["dataset"].startswith("outputs/grpo/")
    assert grpo["validation_dataset"].startswith("outputs/grpo/")

    evaluation = load_yaml(PROJECT_ROOT / "configs" / "eval.yaml")
    assert evaluation["dataset_split"] == "test"
    assert evaluation["source_dataset"] == "data/test/test.jsonl"
    assert evaluation["manifest"].endswith("manifest_test.jsonl")
