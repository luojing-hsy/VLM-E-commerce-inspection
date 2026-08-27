from src.data.approve_opd import approve


def _candidate() -> dict:
    target = {
        "decision": "reject",
        "violation_type": "image_quality",
        "issue_subtype": "blur",
        "evidence": "main",
    }
    return {"target": target}


def test_teacher_filter_approves_only_rule_verified_prediction() -> None:
    row = _candidate()
    accepted, reason = approve(row, row["target"], 0.5)
    assert accepted
    assert reason == "approved"

    wrong = {**row["target"], "issue_subtype": "occlusion"}
    accepted, reason = approve(row, wrong, 0.5)
    assert not accepted
    assert reason == "issue_subtype_mismatch"


def test_teacher_filter_rejects_unlocalized_evidence() -> None:
    row = _candidate()
    prediction = {
        **row["target"],
        "evidence": "detail:2",
    }
    accepted, reason = approve(row, prediction, 0.5)
    assert not accepted
    assert reason == "evidence_mismatch"
