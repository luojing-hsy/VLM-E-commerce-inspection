from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common import load_yaml, read_jsonl
from src.data.split_manifest import DATASET_STAGES, SPLITS, TRAIN_SPLITS, read_split_manifests
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
    sft_rows = [
        row for split in TRAIN_SPLITS for row in read_jsonl(manifest_root / f"sft_{split}.jsonl")
    ]
    grpo_rows = [
        row
        for split in TRAIN_SPLITS
        for row in read_jsonl(manifest_root / f"grpo_{split}.jsonl")
    ]
    opd_rows = [
        row for split in TRAIN_SPLITS for row in read_jsonl(manifest_root / f"opd_{split}.jsonl")
    ]
    joint_rows = [
        row for split in TRAIN_SPLITS for row in read_jsonl(manifest_root / f"joint_{split}.jsonl")
    ]
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
    sft_main_samples = [row for row in samples if row["dataset_stage"] == "sft"]
    sft_counterfactuals = [row for row in counterfactuals if row["dataset_stage"] == "sft"]
    final_ok = 8000 <= len(sft_rows) <= 12000
    checks["final_sft_scale"] = _check(
        "pass" if final_ok else "fail",
        "Counts main pages plus SFT counterfactual augmentation against the 8,000-12,000 target.",
        actual=len(sft_rows),
        main_samples=len(sft_main_samples),
        counterfactuals=len(sft_counterfactuals),
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
    source_stages: dict[str, set[str]] = defaultdict(set)
    image_stages: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        for product_id in row["source_product_ids"]:
            source_splits[product_id].add(row["split"])
            source_stages[product_id].add(row["dataset_stage"])
        for image_id in row["source_image_ids"]:
            image_stages[image_id].add(row["dataset_stage"])
    overlaps = {key: sorted(value) for key, value in source_splits.items() if len(value) > 1}
    product_stage_overlaps = {key: sorted(value) for key, value in source_stages.items() if len(value) > 1}
    image_stage_overlaps = {key: sorted(value) for key, value in image_stages.items() if len(value) > 1}
    checks["source_split_isolation"] = _check(
        "pass" if not overlaps else "fail",
        "No source or donor product may cross train/validation/test.",
        overlap_count=len(overlaps),
    )
    checks["source_stage_isolation"] = _check(
        "pass" if not product_stage_overlaps and not image_stage_overlaps else "fail",
        "SFT, GRPO, OPD and fixed test must have disjoint source products and image IDs.",
        product_overlap_count=len(product_stage_overlaps),
        image_overlap_count=len(image_stage_overlaps),
    )

    split_export_errors: list[str] = []
    for stem in ("samples", "counterfactuals"):
        for split in SPLITS:
            path = manifest_root / f"{stem}_{split}.jsonl"
            for row in read_jsonl(path):
                if row.get("split") != split:
                    split_export_errors.append(f"{path.name}:{row.get('sample_id')}")
    for stem in ("sft", "opd"):
        for split in TRAIN_SPLITS:
            path = manifest_root / f"{stem}_{split}.jsonl"
            for row in read_jsonl(path):
                if row.get("split") != split or row.get("dataset_stage") != stem:
                    split_export_errors.append(f"{path.name}:{row.get('sample_id')}")
    for split in TRAIN_SPLITS:
        path = manifest_root / f"grpo_{split}.jsonl"
        for row in read_jsonl(path):
            info = row.get("extra_info", {})
            if info.get("split") != split or info.get("dataset_stage") != "grpo":
                split_export_errors.append(f"{path.name}:{info.get('sample_id')}")
    for split in TRAIN_SPLITS:
        path = manifest_root / f"joint_{split}.jsonl"
        for row in read_jsonl(path):
            info = row.get("extra_info", {})
            stage = info.get("dataset_stage")
            if (
                info.get("split") != split
                or info.get("training_stage") != "joint"
                or stage not in {"grpo", "opd"}
                or row.get("opd_enabled") != (stage == "opd")
            ):
                split_export_errors.append(f"{path.name}:{info.get('sample_id')}")
    unexpected_test_exports = [
        str(manifest_root / f"{stage}_test.jsonl")
        for stage in ("sft", "grpo", "opd", "joint")
        if (manifest_root / f"{stage}_test.jsonl").exists()
    ]
    split_export_errors.extend(unexpected_test_exports)
    checks["split_specific_exports"] = _check(
        "pass" if not split_export_errors else "fail",
        "Stage exports contain only their own train/validation rows; evaluation uses one canonical test.",
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

    required_cf = {
        row["sample_id"]
        for row in samples
        if row["dataset_stage"] in {"sft", "test"}
        and row["violation_type"] in CONSISTENCY_TYPES
    }
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
        "Joint OPD rows require frozen-SFT teacher inference and rule-based approval; they are not fabricated by the generator.",
        approved=approved_opd,
        pending=len(opd_rows) - approved_opd,
    )
    joint_stages = {row.get("extra_info", {}).get("dataset_stage") for row in joint_rows}
    checks["joint_manifest"] = _check(
        "pass" if joint_stages == {"grpo", "opd"} else "pending",
        "Stage 2 combines disjoint GRPO and OPD subpools while preserving per-row loss masks.",
        rows=len(joint_rows),
        stages=sorted(stage for stage in joint_stages if stage),
    )

    source_assets = read_jsonl(manifest_root / "source_images.jsonl")
    source_components = read_jsonl(manifest_root / "source_components.jsonl")
    expected_source_images = len({image_id for product in products for image_id in product.get("image_ids", [])})
    source_graph_ok = (
        config.get("source_mode") == "abo_rendered_audit"
        and len(source_assets) == expected_source_images
        and bool(source_components)
        and all(
            row.get("source_component_id")
            and row.get("dataset_stage") in DATASET_STAGES
            and row.get("split") in SPLITS
            for row in source_components
        )
        and all(
            len(str(row.get("phash64") or "")) == 16
            and row.get("source_image_id")
            and row.get("dataset_stage") in DATASET_STAGES
            and Path(row.get("path", "")).parent == Path(config["paths"]["abo_root"]) / "images"
            for row in source_assets
        )
    )
    checks["source_graph_and_phash"] = _check(
        "pass" if source_graph_ok else "fail",
        "ABO listing-image relations and pHash near-duplicate edges are merged before component-level splitting.",
        source_images=len(source_assets),
        source_components=len(source_components),
    )
    source_audit_path = manifest_root / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8")) if source_audit_path.exists() else {}
    license_ok = (
        source_audit.get("effective_policy") == "CC-BY-NC-4.0-conservative"
        and bool(source_audit.get("listing_archive_sha256"))
        and bool(source_audit.get("license_sha256"))
    )
    checks["real_source_license"] = _check(
        "pass" if license_ok else "fail",
        "The official-source license conflict is recorded and the stricter non-commercial policy is enforced.",
        source_mode=config.get("source_mode"),
        effective_policy=source_audit.get("effective_policy"),
    )
    checks["trained_model_acceptance_metrics"] = _check(
        "pending",
        "SFT/Joint acceptance metrics and paired bootstrap CIs require actual model runs.",
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
