from __future__ import annotations


def counterfactual_metrics(originals: list[dict], counterfactuals: list[dict], predictions: dict[str, dict]) -> dict:
    original_by_id = {row["sample_id"]: row for row in originals}
    pair_scores: list[float] = []
    flips: list[bool] = []
    for cf in counterfactuals:
        original = original_by_id.get(cf["counterfactual_of"])
        original_prediction = predictions.get(cf["counterfactual_of"])
        cf_prediction = predictions.get(cf["sample_id"])
        if original is None or original_prediction is None or cf_prediction is None:
            continue
        original_ok = (
            original_prediction.get("decision") == original["decision"]
            and original_prediction.get("violation_type") == original["violation_type"]
        )
        cf_type = cf_prediction.get("violation_type") or "PASS"
        cf_ok = cf_prediction.get("decision") == "pass" and cf_type == "PASS"
        pair_scores.append(1.0 if original_ok and cf_ok else 0.2 if original_ok or cf_ok else 0.0)
        flips.append(original_prediction.get("decision") != cf_prediction.get("decision"))
    return {
        "num_counterfactual_pairs": len(pair_scores),
        "counterfactual_pair_accuracy": sum(pair_scores) / len(pair_scores) if pair_scores else None,
        "counterfactual_flip_consistency": sum(flips) / len(flips) if flips else None,
    }

