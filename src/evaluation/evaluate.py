from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import load_yaml, read_jsonl
from src.evaluation.counterfactual import counterfactual_metrics
from src.evaluation.metrics import classification_metrics, perception_metrics
from src.evaluation.slices import slice_summary
from src.rewards.parser import tolerant_parse


def _oracle_prediction(sample: dict) -> dict:
    return {
        "schema_version": "1.0",
        "decision": sample["decision"],
        "violation_type": sample["violation_type"],
        "field": sample.get("field"),
        "listed_value": sample.get("listed_value"),
        "observed_value": sample.get("observed_value"),
        "evidence": sample.get("evidence", []),
    }


def _load_prediction_map(path: Path) -> tuple[dict[str, dict], float]:
    rows = read_jsonl(path)
    parsed: dict[str, dict] = {}
    valid = 0
    for row in rows:
        result = tolerant_parse(row.get("prediction", {key: value for key, value in row.items() if key != "sample_id"}))
        parsed[row["sample_id"]] = result.data
        valid += result.protocol_valid
    return parsed, valid / len(rows) if rows else 0.0


def evaluate(config: dict, predictions_path: str | None = None, oracle_smoke: bool = False) -> dict:
    manifest = Path(config["manifest"])
    expected_split = config["dataset_split"]
    if not manifest.stem.endswith(f"_{expected_split}"):
        raise ValueError(f"evaluation manifest must end with _{expected_split}.jsonl: {manifest}")
    samples = read_jsonl(manifest)
    wrong_split = [row.get("sample_id") for row in samples if row.get("split") != expected_split]
    if wrong_split:
        raise ValueError(f"evaluation manifest mixes splits: {wrong_split[:5]}")
    counterfactual_path = Path(config["counterfactual_manifest"])
    counterfactuals = read_jsonl(counterfactual_path) if counterfactual_path.exists() else []
    wrong_cf_split = [row.get("sample_id") for row in counterfactuals if row.get("split") != expected_split]
    if wrong_cf_split:
        raise ValueError(f"counterfactual manifest mixes splits: {wrong_cf_split[:5]}")
    if oracle_smoke:
        predictions = {row["sample_id"]: _oracle_prediction(row) for row in samples + counterfactuals}
        parse_rate = 1.0
    else:
        path = Path(predictions_path or config["predictions"])
        predictions, parse_rate = _load_prediction_map(path)
    report = {
        **classification_metrics(samples, predictions, config),
        **perception_metrics(samples, predictions),
        **counterfactual_metrics(samples, counterfactuals, predictions),
        "parse_rate": parse_rate,
        "slices": slice_summary(samples),
        "mode": "oracle_smoke" if oracle_smoke else "model_predictions",
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions without an LLM judge")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--predictions")
    parser.add_argument("--oracle-smoke", action="store_true", help="verify the metric path using manifest labels")
    args = parser.parse_args()
    report = evaluate(load_yaml(args.config), args.predictions, args.oracle_smoke)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
