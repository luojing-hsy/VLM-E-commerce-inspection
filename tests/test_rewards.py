import json
from pathlib import Path

from src.common import load_yaml
from src.rewards.composite import compute_reward
from src.rewards.evidence_reward import bbox_iou
from src.rewards.parser import tolerant_parse
from src.rewards.verl_reward import compute_score as verl_compute_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRPO_CONFIG = PROJECT_ROOT / "configs" / "grpo.yaml"


def _sample() -> dict:
    return {
        "decision": "reject",
        "violation_type": "ATTRIBUTE_CONFLICT",
        "field": "model",
        "listed_value": "Model Y",
        "observed_value": "Model X",
        "evidence": [
            {"role": "listed_value", "image_ref": "page", "region_type": "bbox", "bbox_norm": [100, 100, 300, 200]},
            {"role": "observed_value", "image_ref": "page", "region_type": "bbox", "bbox_norm": [500, 500, 700, 600]},
        ],
    }


def _prediction() -> dict:
    return {"schema_version": "1.0", **_sample()}


def test_perfect_prediction_gets_full_reward() -> None:
    result = compute_reward(_sample(), _prediction(), load_yaml(GRPO_CONFIG))
    assert result["reward"] == 1.0
    assert result["protocol_valid"]


def test_missing_evidence_activates_gate() -> None:
    prediction = _prediction()
    prediction["evidence"] = []
    result = compute_reward(_sample(), prediction, load_yaml(GRPO_CONFIG))
    assert result["reward"] <= 0.45
    assert result["evidence_gate_applied"]


def test_invalid_output_has_non_executable_action() -> None:
    result = compute_reward(_sample(), "not json", load_yaml(GRPO_CONFIG))
    assert result["components"]["action"] == -1.0
    assert not result["protocol_valid"]


def test_parser_accepts_code_fence_and_key_order() -> None:
    text = "result:\n```json\n" + json.dumps(_prediction()) + "\n```"
    assert tolerant_parse(text).protocol_valid


def test_iou_is_continuous() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert 0 < bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) < 1


def test_verl_reward_adapter_preserves_components() -> None:
    result = verl_compute_score(
        data_source="vlm_product_audit",
        solution_str=json.dumps(_prediction()),
        ground_truth=json.dumps(_sample()),
        reward_config_path=str(GRPO_CONFIG),
    )
    assert result["score"] == 1.0
    assert result["reward_action"] == 1.0
    assert result["protocol_valid"] == 1.0
