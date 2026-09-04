import json
from pathlib import Path

import pytest

from src.common import load_yaml
from src.rewards.action_cost import action_reward
from src.rewards.composite import compute_reward
from src.rewards.evidence_reward import evidence_reward
from src.rewards.parser import tolerant_parse
from src.rewards.type_reward import violation_type_reward
from src.rewards.verl_reward import compute_score as verl_compute_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRPO_CONFIG = PROJECT_ROOT / "configs" / "grpo.yaml"


def _sample(
    violation_type: str = "image_quality",
    *,
    issue_subtype: str | None = None,
    evidence: str | None = None,
) -> dict:
    if violation_type == "image_quality":
        issue_subtype = issue_subtype or "blur"
        evidence = evidence or "main"
    elif violation_type == "wrong_image":
        issue_subtype = None
        evidence = evidence or "main"
    else:
        issue_subtype = None
        evidence = None
    return {
        "decision": "pass" if violation_type == "pass" else "reject",
        "violation_type": violation_type,
        "issue_subtype": issue_subtype,
        "evidence": evidence,
    }


def _prediction(
    violation_type: str = "image_quality",
    *,
    issue_subtype: str | None = None,
    evidence: str | None = None,
) -> dict:
    return _sample(
        violation_type,
        issue_subtype=issue_subtype,
        evidence=evidence,
    )


def test_perfect_prediction_gets_full_reward() -> None:
    result = compute_reward(_sample(), _prediction(), load_yaml(GRPO_CONFIG))
    assert result["reward"] == 1.0
    assert result["protocol_valid"]


def test_missing_evidence_activates_soft_gate() -> None:
    prediction = _prediction()
    prediction["evidence"] = None
    result = compute_reward(_sample(), prediction, load_yaml(GRPO_CONFIG))
    assert result["pre_gate_reward"] == pytest.approx(0.30)
    assert result["reward"] == pytest.approx(0.21)
    assert result["evidence_gate_applied"]


def test_invalid_output_has_non_executable_action() -> None:
    result = compute_reward(_sample(), "not json", load_yaml(GRPO_CONFIG))
    assert result["components"]["action"] == -1.0
    assert not result["protocol_valid"]


def test_parser_rejects_code_fence_but_accepts_key_order() -> None:
    fence = chr(96) * 3
    text = "result:\n" + fence + "json\n" + json.dumps(_prediction()) + "\n" + fence
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
        reward_config_path=str(GRPO_CONFIG),
    )
    assert result["score"] == 1.0
    assert result["reward_action"] == 1.0
    assert result["protocol_valid"] == 1.0


def test_pass_to_reject_is_mildly_penalized() -> None:
    config = load_yaml(GRPO_CONFIG)
    assert action_reward("pass", "reject", config["cost_matrix"]) == pytest.approx(-0.5)


def test_false_reject_uses_the_mild_action_penalty() -> None:
    prediction = _prediction("pass")
    prediction["decision"] = "reject"
    result = compute_reward(_sample("pass"), prediction, load_yaml(GRPO_CONFIG))
    assert result["components"]["action"] == pytest.approx(-0.5)
    assert result["reward"] == pytest.approx(-0.5 * 0.30 / 0.65)
    assert not result["protocol_valid"]


def test_reject_to_pass_remains_fully_penalized() -> None:
    config = load_yaml(GRPO_CONFIG)
    assert action_reward("reject", "pass", config["cost_matrix"]) == pytest.approx(-1.0)


def test_type_reward_requires_exact_match() -> None:
    assert violation_type_reward("category_mismatch", "color_mismatch") == 0.0
    assert violation_type_reward("category_mismatch", "category_mismatch") == 1.0


def test_type_mismatch_does_not_get_group_partial_credit() -> None:
    result = compute_reward(
        _sample("category_mismatch"),
        _prediction("color_mismatch"),
        load_yaml(GRPO_CONFIG),
    )
    assert result["components"]["type"] == 0.0
    assert result["reward"] == pytest.approx(0.30 / 0.65)


def test_wrong_image_evidence_gate_soft_discounts_wrong_ref() -> None:
    sample = _sample("wrong_image", evidence="detail:2")
    prediction = _prediction("wrong_image", evidence="detail:1")
    result = compute_reward(sample, prediction, load_yaml(GRPO_CONFIG))
    assert result["pre_gate_reward"] == pytest.approx(0.65 / 0.85)
    assert result["reward"] == pytest.approx((0.65 / 0.85) * 0.70)
    assert result["evidence_gate_applied"]


def test_quality_wrong_ref_soft_discounts_but_wrong_subtype_is_not_gated() -> None:
    sample = _sample("image_quality", issue_subtype="blur", evidence="main")
    wrong_ref = _prediction("image_quality", issue_subtype="blur", evidence="detail:1")
    wrong_ref_result = compute_reward(sample, wrong_ref, load_yaml(GRPO_CONFIG))
    assert wrong_ref_result["pre_gate_reward"] == pytest.approx(0.80)
    assert wrong_ref_result["reward"] == pytest.approx(0.56)
    assert wrong_ref_result["evidence_gate_applied"]

    wrong_subtype = _prediction("image_quality", issue_subtype="occlusion", evidence="main")
    wrong_subtype_result = compute_reward(sample, wrong_subtype, load_yaml(GRPO_CONFIG))
    assert wrong_subtype_result["pre_gate_reward"] == pytest.approx(0.85)
    assert wrong_subtype_result["reward"] == pytest.approx(0.85)
    assert not wrong_subtype_result["evidence_gate_applied"]


def test_negative_reward_is_not_softened_by_evidence_gate() -> None:
    result = compute_reward(_sample(), "not json", load_yaml(GRPO_CONFIG))
    assert result["pre_gate_reward"] == pytest.approx(-0.30)
    assert result["reward"] == pytest.approx(-0.30)
    assert result["evidence_gate_applied"]


def test_invalid_type_on_image_sample_cannot_avoid_evidence_gate() -> None:
    sample = _sample("wrong_image", evidence="detail:2")
    prediction = _prediction("image_quality", issue_subtype="blur", evidence="detail:2")
    result = compute_reward(sample, prediction, load_yaml(GRPO_CONFIG))
    assert result["components"]["type"] == 0.0
    assert result["components"]["evidence"] == 0.0
    assert result["evidence_gate_applied"]


def test_prediction_target_image_ref_is_not_model_evidence() -> None:
    prediction = {**_prediction(), "evidence": None, "target_image_ref": "main"}
    assert evidence_reward(_sample(), prediction) == 0.0


def test_reward_config_uses_selected_pass_reject_cost() -> None:
    config = load_yaml(GRPO_CONFIG)
    assert config["cost_matrix"]["pass"]["reject"] == pytest.approx(0.75)
    assert config["evidence_gate_multiplier"] == pytest.approx(0.70)
    assert config["reward_weights"] == {
        "action": 0.30,
        "type": 0.35,
        "evidence": 0.20,
        "subtype": 0.15,
    }
