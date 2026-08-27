from src.evaluation.metrics import classification_metrics


CONFIG = {
    "target_class_prior": {"pass": 0.5, "reject": 0.5},
    "cost_matrix": {
        "pass": {"pass": 0.0, "reject": 1.0, "INVALID": 1.0},
        "reject": {"pass": 1.0, "reject": 0.0, "INVALID": 1.0},
    },
}


def _sample(sample_id, decision, violation_type, issue_subtype=None, evidence=None):
    return {
        "sample_id": sample_id,
        "decision": decision,
        "violation_type": violation_type,
        "issue_subtype": issue_subtype,
        "evidence": evidence,
    }


def test_complete_success_requires_quality_subtype_and_image_ref() -> None:
    samples = [
        _sample("pass", "pass", "pass"),
        _sample("quality", "reject", "image_quality", "blur", "detail:1"),
        _sample("wrong", "reject", "wrong_image", evidence="detail:2"),
        _sample("category", "reject", "category_mismatch"),
    ]
    predictions = {
        "pass": _sample("unused", "pass", "pass"),
        "quality": _sample(
            "unused", "reject", "image_quality", "low_resolution", "detail:1"
        ),
        "wrong": _sample("unused", "reject", "wrong_image", evidence="main"),
        "category": _sample("unused", "reject", "category_mismatch"),
    }

    metrics = classification_metrics(samples, predictions, CONFIG)

    assert metrics["detection_success_rate"] == 1.0
    assert metrics["type_inspection_success_probability"] == 1.0
    assert metrics["complete_inspection_success_probability"] == 0.5
    assert metrics["violation_detection_success_rate"] == 1.0


def test_invalid_prediction_fails_all_success_levels() -> None:
    samples = [_sample("quality", "reject", "image_quality", "blur", "main")]
    metrics = classification_metrics(samples, {"quality": {}}, CONFIG)

    assert metrics["detection_success_rate"] == 0.0
    assert metrics["type_inspection_success_probability"] == 0.0
    assert metrics["complete_inspection_success_probability"] == 0.0


def test_missing_prediction_is_counted_as_failure() -> None:
    samples = [
        _sample("pass", "pass", "pass"),
        _sample("category", "reject", "category_mismatch"),
    ]
    predictions = {"pass": _sample("unused", "pass", "pass")}

    metrics = classification_metrics(samples, predictions, CONFIG)

    assert metrics["num_evaluated"] == 2
    assert metrics["detection_success_rate"] == 0.5
    assert metrics["type_inspection_success_probability"] == 0.5
    assert metrics["complete_inspection_success_probability"] == 0.5
