from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.common import load_yaml, read_jsonl, sha256_file, stable_hash
from src.data.prepare_abo import _source_image_path
from src.data.split_manifest import DATASET_STAGES, SPLITS, assert_stage_source_isolation, manifest_path, read_split_manifests
from src.models.schema import AuditPrediction


def _protocol_payload(sample: dict) -> dict:
    return {key: sample.get(key) for key in ("schema_version", "decision", "violation_type", "field", "listed_value", "observed_value", "evidence")}


def validate(config: dict) -> dict:
    samples = read_split_manifests(config, "samples")
    counterfactuals = read_split_manifests(config, "counterfactuals")
    products = {row["product_id"]: row for row in read_split_manifests(config, "products")}
    errors: list[str] = []
    assert_stage_source_isolation(products.values())
    if config.get("source_mode") == "abo_rendered_audit":
        image_root = Path(config["paths"]["abo_root"]) / "images"
        source_assets = read_jsonl(Path(config["paths"]["manifests"]) / "source_images.jsonl")
        source_asset_ids: set[str] = set()
        for asset in source_assets:
            source_image_id = str(asset.get("source_image_id") or "")
            if not source_image_id or source_image_id in source_asset_ids:
                errors.append(f"missing or duplicate source_image_id: {source_image_id}")
                continue
            source_asset_ids.add(source_image_id)
            image_path = Path(asset["path"])
            expected_path = _source_image_path(
                Path(config["paths"]["abo_root"]),
                source_image_id,
                str(asset["object_path"]),
            )
            if image_path != expected_path or image_path.parent != image_root:
                errors.append(f"source image is not in the flat ID pool: {source_image_id} -> {image_path}")
            if not image_path.exists():
                errors.append(f"missing source image: {source_image_id} -> {image_path}")
            elif asset.get("sha256") != sha256_file(image_path):
                errors.append(f"source image hash mismatch: {source_image_id}")
            if not re.fullmatch(r"[0-9a-f]{16}", str(asset.get("phash64") or "")):
                errors.append(f"invalid source image pHash: {source_image_id}")
        for partial in image_root.glob("*.part"):
            errors.append(f"incomplete source image download remains: {partial}")

        component_splits: dict[str, set[str]] = defaultdict(set)
        component_stages: dict[str, set[str]] = defaultdict(set)
        product_image_splits: dict[str, set[str]] = defaultdict(set)
        product_image_stages: dict[str, set[str]] = defaultdict(set)
        product_type_counts = Counter()
        for product in products.values():
            dataset_stage = str(product.get("dataset_stage") or "")
            if dataset_stage not in DATASET_STAGES:
                errors.append(f"invalid product dataset_stage: {product['product_id']} -> {dataset_stage}")
            component_id = str(product.get("source_component_id") or "")
            if not component_id:
                errors.append(f"missing source_component_id: {product['product_id']}")
            else:
                component_splits[component_id].add(product["split"])
                component_stages[component_id].add(dataset_stage)
            product_type_counts[str(product.get("product_type") or "")] += 1
            for source_image_id in product.get("image_ids", []):
                if source_image_id not in source_asset_ids:
                    errors.append(f"product references unknown source image: {product['product_id']} -> {source_image_id}")
                product_image_splits[str(source_image_id)].add(product["split"])
                product_image_stages[str(source_image_id)].add(dataset_stage)
        for component_id, splits in component_splits.items():
            if len(splits) > 1:
                errors.append(f"source component crosses splits: {component_id} -> {sorted(splits)}")
        for source_image_id, splits in product_image_splits.items():
            if len(splits) > 1:
                errors.append(f"product source image crosses splits: {source_image_id} -> {sorted(splits)}")
        for component_id, stages in component_stages.items():
            if len(stages) > 1:
                errors.append(f"source component crosses dataset stages: {component_id} -> {sorted(stages)}")
        for source_image_id, stages in product_image_stages.items():
            if len(stages) > 1:
                errors.append(f"source image crosses dataset stages: {source_image_id} -> {sorted(stages)}")
        per_type_cap = int(config["abo"]["max_products_per_type"])
        if product_type_counts and max(product_type_counts.values()) > per_type_cap:
            errors.append(f"ABO product_type cap exceeded: {max(product_type_counts.values())} > {per_type_cap}")

    seen_ids: set[str] = set()

    source_splits: dict[str, set[str]] = defaultdict(set)
    source_stages: dict[str, set[str]] = defaultdict(set)
    source_image_splits: dict[str, set[str]] = defaultdict(set)
    source_image_stages: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    family_stages: dict[str, set[str]] = defaultdict(set)
    derived_image_ids: set[str] = set()
    counts = Counter(sample["violation_type"] for sample in samples)
    for sample in samples + counterfactuals:
        sample_id = sample["sample_id"]
        sample_stage = sample.get("dataset_stage")
        if sample_stage not in DATASET_STAGES:
            errors.append(f"invalid sample dataset_stage: {sample_id} -> {sample_stage}")
        elif (sample_stage == "test") != (sample["split"] == "test"):
            errors.append(f"test stage/split mismatch: {sample_id} -> {sample_stage}/{sample['split']}")
        if sample_id in seen_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        image_path = Path(sample["image"])
        derived_image_id = sample.get("derived_image_id")
        source_image_ids = sample.get("source_image_ids")
        if not derived_image_id or derived_image_id in derived_image_ids:
            errors.append(f"missing or duplicate derived_image_id: {sample_id}")
        else:
            derived_image_ids.add(derived_image_id)
        if not isinstance(source_image_ids, list) or not source_image_ids:
            errors.append(f"missing source_image_ids: {sample_id}")
        else:
            for source_image_id in source_image_ids:
                source_image_splits[str(source_image_id)].add(sample["split"])
                source_image_stages[str(source_image_id)].add(str(sample_stage))
        if image_path.parent.name != "pages" or image_path.name != f"{sample_id}.png":
            errors.append(f"page does not use a flat sample ID filename: {sample_id}")
        if not image_path.exists():
            errors.append(f"missing image: {image_path}")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:  # Pillow exposes several decoder-specific exceptions.
            errors.append(f"undecodable image {image_path}: {exc}")
        try:
            AuditPrediction.model_validate(_protocol_payload(sample))
        except Exception as exc:
            errors.append(f"invalid protocol for {sample_id}: {exc}")

        for product_id in sample["source_product_ids"]:
            source_splits[product_id].add(sample["split"])
            source_stages[product_id].add(str(sample_stage))
            product = products.get(product_id)
            if product is None:
                errors.append(f"unknown source product {product_id} in {sample_id}")
            else:
                family_splits[product["family_id"]].add(sample["split"])
                family_stages[product["family_id"]].add(str(sample_stage))
                if product.get("dataset_stage") != sample_stage:
                    errors.append(f"sample uses product from another dataset stage: {sample_id} -> {product_id}")

        if sample["violation_type"] == "PASS":
            if sample.get("transform") is not None or sample.get("changed_fields") or sample["evidence"]:
                errors.append(f"PASS sample contains a transform or evidence: {sample_id}")
        elif not sample.get("transform"):
            errors.append(f"violation sample has no transform: {sample_id}")

        for evidence in sample["evidence"]:
            if evidence["region_type"] == "bbox":
                x1, y1, x2, y2 = evidence["bbox_norm"]
                if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                    errors.append(f"out-of-range bbox in {sample_id}")
                if evidence.get("evidence_source") not in {"rendered_text", "catalog_image"}:
                    errors.append(f"invalid bbox evidence source in {sample_id}")

        for crop in sample.get("crops", []):
            crop_path = Path(crop["path"])
            if crop_path.parent.name != "crops" or not crop_path.name.startswith(f"{sample_id}__crop_"):
                errors.append(f"crop does not use a flat sample ID filename: {sample_id}")
            if not crop.get("derived_image_id") or crop.get("parent_derived_image_id") != derived_image_id:
                errors.append(f"invalid crop lineage in {sample_id}")
            if not crop_path.exists():
                errors.append(f"missing crop: {crop_path}")
                continue
            width, height = sample["image_size"]
            tx1, ty1, tx2, ty2 = [
                round(crop["target_bbox_norm"][0] * width / 1000),
                round(crop["target_bbox_norm"][1] * height / 1000),
                round(crop["target_bbox_norm"][2] * width / 1000),
                round(crop["target_bbox_norm"][3] * height / 1000),
            ]
            cx1, cy1, cx2, cy2 = crop["page_bbox_px"]
            if not (cx1 <= tx1 + 1 and cy1 <= ty1 + 1 and cx2 >= tx2 - 1 and cy2 >= ty2 - 1):
                errors.append(f"crop does not cover evidence in {sample_id}")

    for product_id, splits in source_splits.items():
        if len(splits) > 1:
            errors.append(f"source product crosses splits: {product_id} -> {sorted(splits)}")
    for source_image_id, splits in source_image_splits.items():
        if len(splits) > 1:
            errors.append(f"source image crosses splits: {source_image_id} -> {sorted(splits)}")
    for family_id, splits in family_splits.items():
        if len(splits) > 1:
            errors.append(f"product family crosses splits: {family_id} -> {sorted(splits)}")
    for product_id, stages in source_stages.items():
        if len(stages) > 1:
            errors.append(f"source product crosses dataset stages: {product_id} -> {sorted(stages)}")
    for source_image_id, stages in source_image_stages.items():
        if len(stages) > 1:
            errors.append(f"source image crosses dataset stages: {source_image_id} -> {sorted(stages)}")
    for family_id, stages in family_stages.items():
        if len(stages) > 1:
            errors.append(f"product family crosses dataset stages: {family_id} -> {sorted(stages)}")
    for violation, minimum in config["class_minimums"].items():
        if counts[violation] < int(minimum):
            errors.append(f"class {violation} has {counts[violation]} samples; minimum is {minimum}")

    cf_by_parent = {row["counterfactual_of"]: row for row in counterfactuals}
    for sample in samples:
        if (
            sample["dataset_stage"] in {"sft", "test"}
            and sample["violation_type"] in {"PRODUCT_MISMATCH", "ATTRIBUTE_CONFLICT", "TEXT_LABEL_CONFLICT"}
        ):
            cf = cf_by_parent.get(sample["sample_id"])
            if cf is None:
                errors.append(f"missing counterfactual for {sample['sample_id']}")
            elif cf["decision"] != "pass" or cf["violation_type"] != "PASS":
                errors.append(f"counterfactual was not restored to PASS: {sample['sample_id']}")

    report = {
        "valid": not errors,
        "num_samples": len(samples),
        "num_counterfactuals": len(counterfactuals),
        "split_counts": dict(Counter(sample["split"] for sample in samples)),
        "class_counts": dict(sorted(counts.items())),
        "manifest_sha256": {
            split: sha256_file(manifest_path(config, "samples", split)) for split in SPLITS
        },
        "content_hash": stable_hash(samples),
        "errors": errors,
    }
    report_path = Path(config["paths"]["manifests"]) / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        raise ValueError(f"dataset validation failed with {len(errors)} error(s):\n{preview}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated pages, evidence, splits and counterfactuals")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    report = validate(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    from src.data.audit_agents_compliance import audit

    compliance = audit(config, report)
    print(f"Agents.md compliance: {compliance['overall']} (pending: {', '.join(compliance['pending_checks']) or 'none'})")


if __name__ == "__main__":
    main()
