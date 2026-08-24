from src.data.approve_opd import approve


def _candidate() -> dict:
    target = {
        "schema_version": "1.0",
        "decision": "reject",
        "violation_type": "ATTRIBUTE_CONFLICT",
        "field": "model",
        "listed_value": "Model Y",
        "observed_value": "Model X",
        "evidence": [
            {
                "role": "observed_value",
                "image_ref": "page",
                "region_type": "bbox",
                "bbox_norm": [100, 100, 300, 200],
            }
        ],
    }
    return {"target": target}


def test_teacher_filter_approves_only_rule_verified_prediction() -> None:
    row = _candidate()
    accepted, reason = approve(row, row["target"], 0.5)
    assert accepted
    assert reason == "approved"

    wrong = {**row["target"], "observed_value": "Model Z"}
    accepted, reason = approve(row, wrong, 0.5)
    assert not accepted
    assert reason == "observed_value_mismatch"


def test_teacher_filter_rejects_unlocalized_evidence() -> None:
    row = _candidate()
    prediction = {
        **row["target"],
        "evidence": [
            {
                "role": "observed_value",
                "image_ref": "page",
                "region_type": "bbox",
                "bbox_norm": [700, 700, 800, 800],
            }
        ],
    }
    accepted, reason = approve(row, prediction, 0.5)
    assert not accepted
    assert reason == "evidence_mismatch"
