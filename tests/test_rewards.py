import json
from pathlib import Path

from src.common import load_yaml
from src.rewards.composite import compute_reward
from src.rewards.evidence_reward import evidence_reward
from src.rewards.parser import tolerant_parse
from src.rewards.verl_reward import compute_score as verl_compute_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT_CONFIG = PROJECT_ROOT / "configs" / "joint.yaml"


def _sample() -> dict:
    return {
        "decision": "reject",
        "violation_type": "image_quality",
        "issue_subtype": "blur",
        "evidence": "main",
    }


def _prediction() -> dict:
    return dict(_sample())


def test_perfect_prediction_gets_full_reward() -> None:
    result = compute_reward(_sample(), _prediction(), load_yaml(JOINT_CONFIG))
    assert result["reward"] == 1.0
    assert result["protocol_valid"]


def test_missing_evidence_activates_gate() -> None:
    prediction = _prediction()
    prediction["evidence"] = None
    result = compute_reward(_sample(), prediction, load_yaml(JOINT_CONFIG))
    assert result["reward"] <= 0.45
    assert result["evidence_gate_applied"]


def test_invalid_output_has_non_executable_action() -> None:
    result = compute_reward(_sample(), "not json", load_yaml(JOINT_CONFIG))
    assert result["components"]["action"] == -1.0
    assert not result["protocol_valid"]


def test_parser_rejects_code_fence_but_accepts_key_order() -> None:
    text = "result:\n```json\n" + json.dumps(_prediction()) + "\n```"
    assert not tolerant_parse(text).protocol_valid
    reordered = {key: _prediction()[key] for key in reversed(_prediction())}
    assert tolerant_parse(reordered).protocol_valid


def test_evidence_requires_exact_image_ref() -> None:
    assert evidence_reward(_sample(), _prediction()) == 1.0
    wrong = {**_prediction(), "evidence": "detail:1"}
    assert evidence_reward(_sample(), wrong) == 0.0


def test_verl_reward_adapter_preserves_components() -> None:
    result = verl_compute_score(
        data_source="vlm_product_audit",
        solution_str=json.dumps(_prediction()),
        ground_truth=json.dumps(_sample()),
        reward_config_path=str(JOINT_CONFIG),
    )
    assert result["score"] == 1.0
    assert result["reward_action"] == 1.0
    assert result["protocol_valid"] == 1.0
