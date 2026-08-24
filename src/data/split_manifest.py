from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Callable, Iterable

from src.common import read_jsonl, write_jsonl

SPLITS = ("train", "validation", "test")
TRAIN_SPLITS = ("train", "validation")
DATASET_STAGES = ("sft", "grpo", "opd", "test")


def stable_split_for(component_id: str, seed: int, ratios: dict[str, float]) -> str:
    value = int(hashlib.sha256(f"{seed}:{component_id}".encode()).hexdigest()[:12], 16) / 16**12
    if value < ratios["train"]:
        return "train"
    if value < ratios["train"] + ratios["validation"]:
        return "validation"
    return "test"


def _stable_fraction(namespace: str, component_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{namespace}:{component_id}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def stage_assignment_for_component(
    component_id: str,
    seed: int,
    ratios: dict[str, float],
) -> str:
    if set(ratios) != set(DATASET_STAGES):
        raise ValueError(f"dataset_stage_ratios must define {DATASET_STAGES}")
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("dataset_stage_ratios must sum to 1")
    value = _stable_fraction("dataset-stage", component_id, seed)
    cumulative = 0.0
    for stage in DATASET_STAGES:
        cumulative += float(ratios[stage])
        if value < cumulative:
            return stage
    return DATASET_STAGES[-1]


def stage_split_for_component(
    component_id: str,
    stage: str,
    seed: int,
    validation_ratio: float,
) -> str:
    if stage == "test":
        return "test"
    if stage not in DATASET_STAGES[:-1]:
        raise ValueError(f"unsupported training stage: {stage}")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("stage_validation_ratio must be between 0 and 1")
    value = _stable_fraction(f"{stage}-split", component_id, seed)
    return "validation" if value < validation_ratio else "train"


def assert_stage_source_isolation(products: Iterable[dict]) -> None:
    image_stage: dict[str, str] = {}
    for product in products:
        stage = product.get("dataset_stage")
        if stage not in DATASET_STAGES:
            raise ValueError(f"invalid dataset_stage for {product.get('product_id')}: {stage}")
        for image_id in product.get("image_ids", []):
            previous = image_stage.setdefault(str(image_id), str(stage))
            if previous != stage:
                raise ValueError(
                    f"source_image_id crosses dataset stages: {image_id} -> {previous}, {stage}"
                )

def manifest_path(config: dict, stem: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return Path(config["paths"]["manifests"]) / f"{stem}_{split}.jsonl"


def write_split_manifests(
    config: dict,
    stem: str,
    rows: Iterable[dict],
    split_getter: Callable[[dict], str] = lambda row: row["split"],
    splits: Iterable[str] = SPLITS,
) -> dict[str, Path]:
    splits = tuple(splits)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        split = split_getter(row)
        if split not in splits:
            raise ValueError(f"invalid split in {stem} export: {split}")
        grouped[split].append(row)
    targets: dict[str, Path] = {}
    for split in splits:
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
        "dataset_stage": str(sample["dataset_stage"]),
        "source_product_ids": list(source_product_ids),
        "source_image_ids": list(source_image_ids),
        "derived_image_id": str(derived_image_id),
        "parent_sample_id": sample.get("counterfactual_of"),
    }
