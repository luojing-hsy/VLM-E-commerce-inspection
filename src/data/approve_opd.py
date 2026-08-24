from __future__ import annotations

import argparse
from pathlib import Path

from src.common import load_yaml, read_jsonl, write_jsonl
from src.data.export_joint import write_exports as write_joint_exports
from src.data.split_manifest import TRAIN_SPLITS, manifest_path
from src.rewards.evidence_reward import evidence_reward
from src.rewards.parser import tolerant_parse
from src.rewards.value_reward import normalize_value


def _prediction_map(path: str | Path) -> dict[str, object]:
    predictions = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in predictions:
            raise ValueError(f"invalid or duplicate teacher prediction sample_id: {sample_id}")
        predictions[sample_id] = row.get("prediction")
    return predictions


def approve(row: dict, prediction: object, evidence_threshold: float) -> tuple[bool, str]:
    parsed = tolerant_parse(prediction)
    if not parsed.protocol_valid:
        return False, "protocol_invalid"
    target = row["target"]
    candidate = parsed.data
    if candidate.get("decision") != target.get("decision"):
        return False, "decision_mismatch"
    if candidate.get("violation_type") != target.get("violation_type"):
        return False, "violation_type_mismatch"
    if normalize_value(candidate.get("observed_value")) != normalize_value(target.get("observed_value")):
        return False, "observed_value_mismatch"
    if evidence_reward(target, candidate) < evidence_threshold:
        return False, "evidence_mismatch"
    return True, "approved"


def apply_predictions(config: dict, predictions_path: str | Path, evidence_threshold: float = 0.5) -> dict:
    if not 0 < evidence_threshold <= 1:
        raise ValueError("evidence threshold must be in (0, 1]")
    predictions = _prediction_map(predictions_path)
    counts = {"approved": 0, "rejected": 0}
    split_rows = {
        split: read_jsonl(manifest_path(config, "opd", split))
        for split in TRAIN_SPLITS
    }
    expected_ids = {
        row["sample_id"]
        for rows in split_rows.values()
        for row in rows
    }
    missing = sorted(expected_ids - set(predictions))
    extras = sorted(set(predictions) - expected_ids)
    if missing:
        raise ValueError(f"missing teacher predictions: {missing[:5]}")
    if extras:
        raise ValueError(f"teacher predictions contain unknown sample IDs: {extras[:5]}")

    for rows in split_rows.values():
        for row in rows:
            sample_id = row["sample_id"]
            accepted, reason = approve(row, predictions[sample_id], evidence_threshold)
            row["teacher_filter_status"] = "approved" if accepted else "rejected"
            row["teacher_filter_reason"] = reason
            counts[row["teacher_filter_status"]] += 1
    for split, rows in split_rows.items():
        write_jsonl(manifest_path(config, "opd", split), rows)
    write_joint_exports(config)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve OPD candidates using frozen-SFT predictions and rules")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--evidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    counts = apply_predictions(load_yaml(args.config), args.predictions, args.evidence_threshold)
    print(f"OPD teacher filter complete: {counts}")


if __name__ == "__main__":
    main()
