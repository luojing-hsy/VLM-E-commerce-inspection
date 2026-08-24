from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common import load_yaml, read_jsonl
from src.data.split_manifest import SPLITS, read_split_manifests
from src.rewards.parser import tolerant_parse

CONSISTENCY_TYPES = {"PRODUCT_MISMATCH", "ATTRIBUTE_CONFLICT", "TEXT_LABEL_CONFLICT"}
V1_TYPES = {
    "PASS",
    "PRODUCT_MISMATCH",
    "ATTRIBUTE_CONFLICT",
    "TEXT_LABEL_CONFLICT",
    "MISSING_REQUIRED_FIELD",
    "IMAGE_QUALITY",
    "IRRELEVANT_IMAGE",
    "DUPLICATE_IMAGE",
}


def _check(status: str, detail: str, **metrics: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **metrics}


def audit(config: dict, validation_report: dict | None = None) -> dict[str, Any]:
    manifest_root = Path(config["paths"]["manifests"])
    products = read_split_manifests(config, "products")
    samples = read_split_manifests(config, "samples")
    counterfactuals = read_split_manifests(config, "counterfactuals")
    sft_rows = read_split_manifests(config, "sft")
    grpo_rows = [
        row
        for split in SPLITS
        for row in read_jsonl(manifest_root / f"grpo_{split}.jsonl")
    ]
    opd_rows = read_split_manifests(config, "opd")
    if validation_report is None:
        validation_path = manifest_root / "validation_report.json"
        validation_report = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None

    checks: dict[str, dict[str, Any]] = {}
    raw_ok = 3000 <= len(products) <= 5000
    checks["raw_product_scale"] = _check(
        "pass" if raw_ok else "fail",
        "Agents.md target is 3,000-5,000 raw products.",
        actual=len(products),
    )
    final_ok = 8000 <= len(sft_rows) <= 12000
    checks["final_sft_scale"] = _check(
        "pass" if final_ok else "fail",
        "Counts main pages plus SFT counterfactual augmentation against the 8,000-12,000 target.",
        actual=len(sft_rows),
        main_samples=len(samples),
        counterfactuals=len(counterfactuals),
    )
    present_types = set(Counter(row["violation_type"] for row in samples))
    checks["all_v1_types"] = _check(
        "pass" if present_types == V1_TYPES else "fail",
        "All eight V1 labels must be present.",
        present=sorted(present_types),
        missing=sorted(V1_TYPES - present_types),
    )
    checks["programmatic_validation"] = _check(
        "pass" if validation_report and validation_report.get("valid") else "fail",
        "Image, schema, bbox, crop, split and counterfactual checks are delegated to validate_dataset.",
        validation_errors=[] if not validation_report else validation_report.get("errors", []),
    )
    reproducibility_path = manifest_root / "reproducibility.json"
    reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8")) if reproducibility_path.exists() else None
    reproducibility_status = "pass" if reproducibility and reproducibility.get("matched") is True else "fail" if reproducibility and reproducibility.get("matched") is False else "pending"
    checks["fixed_seed_reproducibility"] = _check(
        reproducibility_status,
        "A complete second generation must match product, manifest and generated-image tree hashes.",
        matched=None if reproducibility is None else reproducibility.get("matched"),
    )

    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        for product_id in row["source_product_ids"]:
            source_splits[product_id].add(row["split"])
    overlaps = {key: sorted(value) for key, value in source_splits.items() if len(value) > 1}
    checks["source_split_isolation"] = _check(
        "pass" if not overlaps else "fail",
        "No source or donor product may cross train/validation/test.",
        overlap_count=len(overlaps),
    )

    split_export_errors: list[str] = []
    for stem in ("samples", "counterfactuals", "sft", "opd"):
        for split in SPLITS:
            path = manifest_root / f"{stem}_{split}.jsonl"
            for row in read_jsonl(path):
                if row.get("split") != split:
                    split_export_errors.append(f"{path.name}:{row.get('sample_id')}")
    for split in SPLITS:
        path = manifest_root / f"grpo_{split}.jsonl"
        for row in read_jsonl(path):
            if row.get("extra_info", {}).get("split") != split:
                split_export_errors.append(f"{path.name}:{row.get('extra_info', {}).get('sample_id')}")
    checks["split_specific_exports"] = _check(
        "pass" if not split_export_errors else "fail",
        "Every stage export must contain exactly the split declared by its filename.",
        errors=split_export_errors[:20],
    )

    train_templates = {row["template_id"] for row in samples if row["split"] == "train"}
    test_templates = {row["template_id"] for row in samples if row["split"] == "test"}
    checks["heldout_test_template"] = _check(
        "pass" if train_templates.isdisjoint(test_templates) else "fail",
        "Test uses a layout not observed in training.",
        train=sorted(train_templates),
        test=sorted(test_templates),
    )

    required_cf = {row["sample_id"] for row in samples if row["violation_type"] in CONSISTENCY_TYPES}
    actual_cf = {row["counterfactual_of"] for row in counterfactuals}
    checks["counterfactual_coverage"] = _check(
        "pass" if required_cf == actual_cf else "fail",
        "Every consistency violation must have exactly one restored PASS counterpart.",
        required=len(required_cf),
        actual=len(actual_cf),
    )

    valid_sft = 0
    for row in sft_rows:
        answer = row["conversations"][-1]["value"]
        valid_sft += tolerant_parse(answer).protocol_valid
    parse_rate = valid_sft / len(sft_rows) if sft_rows else 0.0
    checks["sft_parseability"] = _check(
        "pass" if parse_rate >= 0.98 else "fail",
        "Program-generated SFT targets must exceed 98% protocol parseability.",
        parse_rate=parse_rate,
    )

    grpo_cf = [
        row["extra_info"]["sample_id"]
        for row in grpo_rows
        if row["extra_info"]["sample_id"].endswith("_cf")
    ]
    checks["grpo_excludes_counterfactuals"] = _check(
        "pass" if not grpo_cf else "fail",
        "V1 counterfactuals are evaluation/SFT data and must not enter ordinary GRPO.",
        offending_count=len(grpo_cf),
    )

    approved_opd = sum(row.get("teacher_filter_status") == "approved" for row in opd_rows)
    checks["opd_candidates"] = _check(
        "pass" if opd_rows else "fail",
        "Full-page plus renderer-derived crop candidates can be generated before teacher inference.",
        candidates=len(opd_rows),
    )
    checks["opd_teacher_filter"] = _check(
        "pass" if approved_opd else "pending",
        "Final OPD training rows require frozen-GRPO teacher inference and rule-based approval; they are not fabricated by the generator.",
        approved=approved_opd,
        pending=len(opd_rows) - approved_opd,
    )

    checks["source_graph_and_phash"] = _check(
        "pending",
        "The synthetic portfolio build isolates source IDs/families, but does not yet construct the full listing-image-donor graph or cross-split pHash near-duplicate components required for real ABO data.",
    )
    checks["real_source_license"] = _check(
        "not_applicable",
        "Current source_mode is synthetic_demo; ABO archive license capture is required only after real data is connected.",
        source_mode=config.get("source_mode"),
    )
    checks["trained_model_acceptance_metrics"] = _check(
        "pending",
        "SFT/GRPO/OPD acceptance metrics and paired bootstrap CIs require actual model runs.",
    )

    failing = [name for name, value in checks.items() if value["status"] == "fail"]
    pending = [name for name, value in checks.items() if value["status"] == "pending"]
    report = {
        "overall": "fail" if failing else "partial" if pending else "pass",
        "scope": "Agents.md data-pipeline and training-readiness audit",
        "checks": checks,
        "failing_checks": failing,
        "pending_checks": pending,
    }
    target = manifest_root / "agents_compliance.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated artifacts against the applicable Agents.md requirements")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    report = audit(load_yaml(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
