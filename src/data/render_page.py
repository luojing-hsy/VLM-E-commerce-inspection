from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from src.common import load_yaml, read_jsonl, stable_hash, write_jsonl
from src.data.split_manifest import DATASET_STAGES, TRAIN_SPLITS, read_split_manifests, stable_split_for

OPD_VIOLATIONS = ("PRODUCT_MISMATCH", "ATTRIBUTE_CONFLICT", "TEXT_LABEL_CONFLICT")

VIOLATIONS = (
    "PASS",
    "PRODUCT_MISMATCH",
    "ATTRIBUTE_CONFLICT",
    "TEXT_LABEL_CONFLICT",
    "MISSING_REQUIRED_FIELD",
    "IMAGE_QUALITY",
    "IRRELEVANT_IMAGE",
    "DUPLICATE_IMAGE",
)


def _split_for(product_id: str, seed: int, ratios: dict[str, float]) -> str:
    return stable_split_for(product_id, seed, ratios)


def _norm(box: tuple[int, int, int, int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [round(x1 * 1000 / width), round(y1 * 1000 / height), round(x2 * 1000 / width), round(y2 * 1000 / height)]


def _layout(template_id: int, width: int, height: int) -> dict[str, Any]:
    layouts = [
        {"title": (35, 25, width - 35, 100), "main": (35, 125, 455, 500), "attrs": (495, 125, width - 35, 335), "label": (495, 355, width - 35, 500)},
        {"title": (35, 25, width - 35, 100), "main": (width - 455, 125, width - 35, 500), "attrs": (35, 125, 465, 335), "label": (35, 355, 465, 500)},
        {"title": (35, 25, 550, 115), "main": (35, 135, 455, 510), "attrs": (495, 35, width - 35, 260), "label": (495, 280, width - 35, 510)},
        {"title": (410, 25, width - 35, 115), "main": (width - 455, 135, width - 35, 510), "attrs": (35, 35, 375, 260), "label": (35, 280, 375, 510)},
    ]
    layout = layouts[template_id % len(layouts)]
    gap = 18
    gallery_width = (width - 70 - 2 * gap) // 3
    layout["gallery"] = [(35 + i * (gallery_width + gap), 545, 35 + i * (gallery_width + gap) + gallery_width, height - 25) for i in range(3)]
    return layout


def _template_for_split(split: str, sample_index: int) -> int:
    # Template 03 is a layout holdout and never appears in training.
    return 3 if split == "test" else sample_index % 3


def _fit_image(path: str, box: tuple[int, int, int, int], blur: bool = False) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image = ImageOps.contain(image, (box[2] - box[0], box[3] - box[1]))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=10))
    return image


def _source_image_id(product: dict, index: int) -> str:
    image_ids = product.get("image_ids")
    if not isinstance(image_ids, list) or index >= len(image_ids):
        raise ValueError(f"product {product.get('product_id')} is missing source image IDs")
    return str(image_ids[index])


def _paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x = box[0] + (box[2] - box[0] - image.width) // 2
    y = box[1] + (box[3] - box[1] - image.height) // 2
    canvas.paste(image, (x, y))


def _replace_value(value: str, product_id: str) -> str:
    suffix = hashlib.sha256(f"{product_id}:{value}".encode()).hexdigest()[:8].upper()
    return f"ALT-{suffix}"


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[:middle] + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def render_one(
    product: dict,
    donor: dict,
    violation: str,
    split: str,
    sample_id: str,
    template_id: int,
    config: dict,
    dataset_stage: str | None = None,
) -> dict:
    dataset_stage = dataset_stage or str(product.get("dataset_stage") or ("test" if split == "test" else "sft"))
    width, height = int(config["page_width"]), int(config["page_height"])
    layout = _layout(template_id, width, height)
    background = (248 - 3 * template_id, 249 - template_id, 252)
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rounded_rectangle((20, 15, width - 20, height - 15), radius=18, outline="#c7ccd5", width=2)

    audit_field = str(product.get("audit_field", "model"))
    if audit_field not in product["attributes"]:
        raise ValueError(f"product {product['product_id']} is missing audit_field={audit_field}")
    true_value = str(product["attributes"][audit_field])
    wrong_value = _replace_value(true_value, product["product_id"])
    title = product["title"]
    promo_value = wrong_value if violation == "TEXT_LABEL_CONFLICT" else true_value
    promo_text = f"Verified {audit_field.replace('_', ' ').title()}: {promo_value}"
    display_attrs = dict(product["attributes"])
    label_product = donor if violation == "PRODUCT_MISMATCH" else product
    main_product = donor if violation == "PRODUCT_MISMATCH" else product
    missing_field = None
    if violation == "ATTRIBUTE_CONFLICT":
        display_attrs[audit_field] = wrong_value
    elif violation == "MISSING_REQUIRED_FIELD":
        missing_field = next(
            (field for field in product.get("required_fields", []) if field in display_attrs),
            None,
        )
        if missing_field is None:
            raise ValueError(f"product {product['product_id']} has no eligible required field")
        display_attrs.pop(missing_field)

    title_x, title_y = layout["title"][0] + 12, layout["title"][1] + 10
    title_width = layout["title"][2] - layout["title"][0] - 24
    rendered_title = _truncate_text(draw, title, font, title_width)
    rendered_promo = _truncate_text(draw, promo_text, font, title_width)
    draw.text((title_x, title_y), rendered_title, fill="#172033", font=font)
    promo_y = title_y + 26
    draw.text((title_x, promo_y), rendered_promo, fill="#3e4b66", font=font)
    promo_raw_box = draw.textbbox((title_x, promo_y), rendered_promo, font=font)
    promo_bbox = (promo_raw_box[0] - 3, promo_raw_box[1] - 2, promo_raw_box[2] + 3, promo_raw_box[3] + 2)
    title_bbox = layout["title"]
    draw.rounded_rectangle(layout["main"], radius=12, fill="white", outline="#d6dae3")
    _paste_center(canvas, _fit_image(main_product["images"][0], layout["main"]), layout["main"])
    source_image_ids = {_source_image_id(main_product, 0)}

    draw.rounded_rectangle(layout["attrs"], radius=12, fill="white", outline="#d6dae3")
    draw.text((layout["attrs"][0] + 12, layout["attrs"][1] + 10), "SPECIFICATIONS", fill="#3e4b66", font=font)
    attribute_value_bboxes: dict[str, tuple[int, int, int, int]] = {}
    y = layout["attrs"][1] + 38
    ordered_fields = []
    for key in (audit_field, *product.get("required_fields", []), *display_attrs):
        if key in display_attrs and key not in ordered_fields:
            ordered_fields.append(key)
    for key in ordered_fields[:5]:
        value = display_attrs[key]
        label = f"{key.replace('_', ' ').title()}:"
        draw.text((layout["attrs"][0] + 14, y), label, fill="#596579", font=font)
        value_x = layout["attrs"][0] + 130
        value_text = str(value)
        draw.text((value_x, y), value_text, fill="#111827", font=font)
        raw_box = draw.textbbox((value_x, y), value_text, font=font)
        attribute_value_bboxes[key] = (raw_box[0] - 3, raw_box[1] - 2, raw_box[2] + 3, raw_box[3] + 2)
        y += 31

    draw.rounded_rectangle(layout["label"], radius=12, fill="#fff8dc", outline="#d7b85c", width=2)
    lx, ly = layout["label"][0] + 14, layout["label"][1] + 12
    draw.text((lx, ly), "PACKAGE LABEL", fill="#5d4612", font=font)
    label_rows = {"brand": label_product["brand"]}
    label_audit_field = str(label_product.get("audit_field", audit_field))
    if label_audit_field in label_product["attributes"]:
        label_rows[label_audit_field] = label_product["attributes"][label_audit_field]
    for key in ("model", "color", "material"):
        if key in label_product["attributes"] and key not in label_rows and len(label_rows) < 3:
            label_rows[key] = label_product["attributes"][key]
    label_value_bboxes: dict[str, tuple[int, int, int, int]] = {}
    for row_index, (key, value) in enumerate(label_rows.items(), start=1):
        row_y = ly + row_index * 28
        draw.text((lx, row_y), f"{key.title()}:", fill="#755b1e", font=font)
        value_x = lx + 105
        draw.text((value_x, row_y), value, fill="#2f250d", font=font)
        raw_box = draw.textbbox((value_x, row_y), value, font=font)
        label_value_bboxes[key] = (raw_box[0] - 3, raw_box[1] - 2, raw_box[2] + 3, raw_box[3] + 2)

    gallery_products = [product, product, product]
    gallery_variants = [0, 1, 2]
    if violation == "IRRELEVANT_IMAGE":
        gallery_products[2] = donor
    if violation == "DUPLICATE_IMAGE":
        gallery_variants[2] = 0
    for gallery_index, box in enumerate(layout["gallery"]):
        draw.rounded_rectangle(box, radius=9, fill="white", outline="#d6dae3")
        gallery_product = gallery_products[gallery_index]
        path = gallery_product["images"][gallery_variants[gallery_index]]
        source_image_ids.add(_source_image_id(gallery_product, gallery_variants[gallery_index]))
        image = _fit_image(path, box, blur=violation == "IMAGE_QUALITY" and gallery_index == 0)
        _paste_center(canvas, image, box)

    generated_root = Path(config["paths"]["generated"])
    image_path = generated_root / dataset_stage / split / "pages" / f"{sample_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path)

    evidence: list[dict] = []
    field = listed_value = observed_value = None
    decision = "pass"
    changed_fields: dict[str, Any] = {}
    transform = None
    if violation == "PRODUCT_MISMATCH":
        decision, field = "reject", "product_identity"
        listed_value, observed_value = rendered_title, donor["title"]
        transform = "replace_product_images"
        evidence = [
            {"role": "listed_value", "image_ref": "page", "region_type": "bbox", "bbox_norm": _norm(title_bbox, width, height), "evidence_source": "rendered_text", "source_field": "title"},
            {"role": "observed_value", "image_ref": "page", "region_type": "bbox", "bbox_norm": _norm(layout["main"], width, height), "evidence_source": "catalog_image", "source_field": "product_id"},
        ]
    elif violation == "ATTRIBUTE_CONFLICT":
        decision, field = "reject", audit_field
        listed_value, observed_value = wrong_value, true_value
        transform = "replace_attribute"
        changed_fields = {audit_field: {"original": true_value, "modified": wrong_value}}
        evidence = [
            {"role": "listed_value", "value": wrong_value, "image_ref": "page", "region_type": "bbox", "bbox_norm": _norm(attribute_value_bboxes[audit_field], width, height), "evidence_source": "rendered_text", "source_field": audit_field},
            {"role": "observed_value", "value": true_value, "image_ref": "page", "region_type": "bbox", "bbox_norm": _norm(label_value_bboxes[audit_field], width, height), "evidence_source": "rendered_text", "source_field": audit_field},
        ]
    elif violation == "TEXT_LABEL_CONFLICT":
        decision, field = "reject", audit_field
        listed_value, observed_value = wrong_value, true_value
        transform = "replace_title_value"
        changed_fields = {audit_field: {"original": true_value, "modified": wrong_value}}
        evidence = [
            {"role": "listed_value", "value": wrong_value, "image_ref": "page", "region_type": "bbox", "bbox_norm": _norm(promo_bbox, width, height), "evidence_source": "rendered_text", "source_field": audit_field},
            {"role": "observed_value", "value": true_value, "image_ref": "page", "region_type": "bbox", "bbox_norm": _norm(label_value_bboxes[audit_field], width, height), "evidence_source": "rendered_text", "source_field": audit_field},
        ]
    elif violation == "MISSING_REQUIRED_FIELD":
        decision, field, transform = "review", missing_field, "remove_required_field"
        changed_fields = {missing_field: {"original": product["attributes"][missing_field], "modified": None}}
        evidence = [{"role": "missing", "region_type": "missing_field", "field": missing_field, "evidence_source": "rendered_text", "source_field": missing_field}]
    elif violation == "IMAGE_QUALITY":
        decision, transform = "review", "gaussian_blur"
        evidence = [{"role": "damaged_image", "region_type": "image_ref", "image_ref": "gallery:0", "evidence_source": "gallery_relation"}]
    elif violation == "IRRELEVANT_IMAGE":
        decision, transform = "review", "insert_irrelevant_image"
        evidence = [{"role": "irrelevant_image", "region_type": "image_ref", "image_ref": "gallery:2", "evidence_source": "gallery_relation"}]
    elif violation == "DUPLICATE_IMAGE":
        decision, transform = "review", "duplicate_image"
        evidence = [{"role": "duplicate_pair", "region_type": "image_pair", "image_refs": ["gallery:0", "gallery:2"], "evidence_source": "gallery_relation"}]

    source_ids = [product["product_id"]]
    if violation in {"PRODUCT_MISMATCH", "IRRELEVANT_IMAGE"}:
        source_ids.append(donor["product_id"])
    derived_image_id = f"page:{sample_id}"
    return {
        "sample_id": sample_id,
        "image": image_path.as_posix(),
        "derived_image_id": derived_image_id,
        "source_image_ids": sorted(source_image_ids),
        "image_size": [width, height],
        "source_product_ids": source_ids,
        "dataset_stage": dataset_stage,
        "split": split,
        "template_id": f"template_{template_id:02d}",
        "seed": int(config["seed"]),
        "schema_version": "1.0",
        "violation_type": violation,
        "decision": decision,
        "field": field,
        "listed_value": listed_value,
        "observed_value": observed_value,
        "transform": transform,
        "changed_fields": changed_fields,
        "evidence": evidence,
        "regions": {
            "title_bbox": _norm(title_bbox, width, height),
            "product_image_bbox": _norm(layout["main"], width, height),
            "attribute_table_bbox": _norm(layout["attrs"], width, height),
            "label_bbox": _norm(layout["label"], width, height),
            "gallery_image_bboxes": [_norm(box, width, height) for box in layout["gallery"]],
        },
    }


def build_pages(config: dict) -> list[dict]:
    products = read_split_manifests(config, "products")
    by_partition: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for product in products:
        by_partition[(product["dataset_stage"], product["split"])].append(product)

    expected = {(stage, split) for stage in DATASET_STAGES[:-1] for split in TRAIN_SPLITS}
    expected.add(("test", "test"))
    missing = sorted(partition for partition in expected if len(by_partition[partition]) < 2)
    if missing:
        raise ValueError(f"product pool is too small for stage/split donors: {missing}")

    sample_budgets = config.get("samples_per_product")
    if not isinstance(sample_budgets, dict) or set(sample_budgets) != set(DATASET_STAGES):
        raise ValueError(f"samples_per_product must define {DATASET_STAGES}")

    rows: list[dict] = []
    for stage in DATASET_STAGES:
        violations = OPD_VIOLATIONS if stage == "opd" else VIOLATIONS
        splits = ("test",) if stage == "test" else TRAIN_SPLITS
        samples_per_product = int(sample_budgets[stage])
        if samples_per_product < 1:
            raise ValueError(f"samples_per_product.{stage} must be positive")
        for split in splits:
            items = sorted(by_partition[(stage, split)], key=lambda item: item["product_id"])
            rng = random.Random(f"{config['seed']}:{stage}:{split}")
            sample_index = 0
            for product in items:
                candidates = [
                    item
                    for item in items
                    if item["product_id"] != product["product_id"]
                    and item.get("source_component_id", item["family_id"])
                    != product.get("source_component_id", product["family_id"])
                ]
                if not candidates:
                    raise ValueError(f"stage={stage} split={split} has no donor outside the source component")
                for _ in range(samples_per_product):
                    donor = candidates[rng.randrange(len(candidates))]
                    violation = violations[sample_index % len(violations)]
                    sample_id = f"{stage}_{split}_{sample_index:05d}"
                    template_id = _template_for_split(split, sample_index)
                    rows.append(
                        render_one(
                            product,
                            donor,
                            violation,
                            split,
                            sample_id,
                            template_id,
                            config,
                            dataset_stage=stage,
                        )
                    )
                    sample_index += 1
    return rows

def _reset_derived_outputs(config: dict) -> None:
    workspace = Path.cwd().resolve()
    generated_root = Path(config["paths"]["generated"]).resolve()
    if generated_root == workspace or workspace not in generated_root.parents:
        raise ValueError(f"generated path must stay below the workspace: {generated_root}")
    if generated_root.exists():
        shutil.rmtree(generated_root)
    generated_root.mkdir(parents=True)

    manifest_root = Path(config["paths"]["manifests"])
    for stage in ("sft", "grpo", "opd"):
        (manifest_root / f"{stage}_test.jsonl").unlink(missing_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Render product pages and inject one deterministic violation")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    _reset_derived_outputs(config)
    rows = build_pages(config)
    from src.data.build_counterfactual import build_counterfactuals
    from src.data.build_crops import build_crops
    from src.data.export_grpo import write_exports as export_grpo
    from src.data.export_joint import write_exports as export_joint
    from src.data.export_opd import write_exports as export_opd
    from src.data.export_sft import write_exports as export_sft
    from src.data.split_manifest import write_split_manifests

    rows = build_crops(config, rows)
    sample_targets = write_split_manifests(config, "samples", rows)
    counterfactuals = build_counterfactuals(config, rows)
    write_split_manifests(config, "counterfactuals", counterfactuals)
    export_sft(config)
    export_opd(config)
    export_grpo(config)
    export_joint(config)
    counts = Counter(row["violation_type"] for row in rows)
    print(f"wrote {len(rows)} pages to {', '.join(str(path) for path in sample_targets.values())}")
    print("class counts:", dict(sorted(counts.items())))
    print("manifest content hash:", stable_hash(rows)[:16])


if __name__ == "__main__":
    main()
