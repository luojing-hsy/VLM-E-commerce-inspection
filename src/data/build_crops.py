from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.common import load_yaml, read_jsonl, write_jsonl
from src.data.split_manifest import read_split_manifests, write_split_manifests


def _denorm(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box[0] * width / 1000),
        round(box[1] * height / 1000),
        round(box[2] * width / 1000),
        round(box[3] * height / 1000),
    )


def _expand(box: tuple[int, int, int, int], width: int, height: int, scale: float = 1.7) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box_width, box_height = (x2 - x1) * scale, (y2 - y1) * scale
    return (
        max(0, round(cx - box_width / 2)),
        max(0, round(cy - box_height / 2)),
        min(width, round(cx + box_width / 2)),
        min(height, round(cy + box_height / 2)),
    )


def build_crops(config: dict, samples: list[dict] | None = None) -> list[dict]:
    samples = samples or read_split_manifests(config, "samples")
    generated_root = Path(config["paths"]["generated"])
    for sample in samples:
        bbox_evidence = [item for item in sample["evidence"] if item["region_type"] == "bbox"]
        sample["crops"] = []
        if not bbox_evidence:
            continue
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB")
            for index, evidence in enumerate(bbox_evidence):
                pixel_box = _denorm(evidence["bbox_norm"], image.width, image.height)
                crop_box = _expand(pixel_box, image.width, image.height)
                crop_path = generated_root / sample["split"] / "crops" / sample["sample_id"] / f"crop_{index}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop(crop_box).save(crop_path)
                sample["crops"].append(
                    {
                        "path": crop_path.as_posix(),
                        "derived_image_id": f"crop:{sample['sample_id']}:{index}",
                        "parent_derived_image_id": sample["derived_image_id"],
                        "source_image_ids": list(sample["source_image_ids"]),
                        "role": evidence["role"],
                        "page_bbox_px": list(crop_box),
                        "target_bbox_norm": evidence["bbox_norm"],
                    }
                )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Create renderer-derived evidence crops")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows = build_crops(config)
    targets = write_split_manifests(config, "samples", rows)
    print(
        f"updated {', '.join(str(path) for path in targets.values())} "
        f"with {sum(len(row['crops']) for row in rows)} crops"
    )


if __name__ == "__main__":
    main()
