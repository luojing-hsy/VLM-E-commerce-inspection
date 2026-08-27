from pathlib import Path

from src.common import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_configs_use_current_dataset_directories() -> None:
    sft = load_yaml(PROJECT_ROOT / "configs" / "sft.yaml")
    assert sft["source_dataset"] == "data/sft/train.jsonl"
    assert sft["validation_source_dataset"] == "data/sft/valid.jsonl"
    assert sft["dataset"].startswith("outputs/sft/")
    assert sft["validation_dataset"].startswith("outputs/sft/")

    joint = load_yaml(PROJECT_ROOT / "configs" / "joint.yaml")
    assert joint["source_dataset"] == "data/joint/train.jsonl"
    assert joint["validation_source_dataset"] == "data/joint/valid.jsonl"
    assert joint["dataset"].startswith("outputs/joint/")
    assert joint["validation_dataset"].startswith("outputs/joint/")

    evaluation = load_yaml(PROJECT_ROOT / "configs" / "eval.yaml")
    assert evaluation["dataset_split"] == "test"
    assert evaluation["source_dataset"] == "data/test/test.jsonl"
    assert evaluation["manifest"].endswith("samples_test.jsonl")
