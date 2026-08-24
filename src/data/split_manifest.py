from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Callable, Iterable

from src.common import read_jsonl, write_jsonl

SPLITS = ("train", "validation", "test")


def stable_split_for(component_id: str, seed: int, ratios: dict[str, float]) -> str:
    value = int(hashlib.sha256(f"{seed}:{component_id}".encode()).hexdigest()[:12], 16) / 16**12
    if value < ratios["train"]:
        return "train"
    if value < ratios["train"] + ratios["validation"]:
        return "validation"
    return "test"


def manifest_path(config: dict, stem: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return Path(config["paths"]["manifests"]) / f"{stem}_{split}.jsonl"


def write_split_manifests(
    config: dict,
    stem: str,
    rows: Iterable[dict],
    split_getter: Callable[[dict], str] = lambda row: row["split"],
) -> dict[str, Path]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        split = split_getter(row)
        if split not in SPLITS:
            raise ValueError(f"invalid split in {stem} export: {split}")
        grouped[split].append(row)
    targets: dict[str, Path] = {}
    for split in SPLITS:
        target = manifest_path(config, stem, split)
        write_jsonl(target, grouped[split])
        targets[split] = target
    return targets


def read_split_manifests(config: dict, stem: str) -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        path = manifest_path(config, stem, split)
        split_rows = read_jsonl(path)
        bad = [row.get("sample_id") for row in split_rows if row.get("split") != split]
        if bad:
            raise ValueError(f"{path} contains rows outside split={split}: {bad[:5]}")
        rows.extend(split_rows)
    return rows


def lineage_from_sample(sample: dict) -> dict:
    source_product_ids = sample.get("source_product_ids")
    source_image_ids = sample.get("source_image_ids")
    derived_image_id = sample.get("derived_image_id")
    if not source_product_ids or not source_image_ids or not derived_image_id:
        raise ValueError(f"sample is missing lineage IDs: {sample.get('sample_id')}")
    return {
        "source_product_ids": list(source_product_ids),
        "source_image_ids": list(source_image_ids),
        "derived_image_id": str(derived_image_id),
        "parent_sample_id": sample.get("counterfactual_of"),
    }
