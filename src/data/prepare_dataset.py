"""Prepare direct-image datasets for SFT and GRPO training plus evaluation.

The runtime consumes the three source images directly.  No composite images or
rendered page artifacts are created.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.common import load_yaml, read_jsonl, sha256_file, write_jsonl
from src.models.audit_protocol import IMAGE_REFS, product_prompt, structured_prompt, target_from_sample


DEFAULT_CONFIGS = {
    "sft": Path("configs/sft.yaml"),
    "grpo": Path("configs/grpo.yaml"),
    "eval": Path("configs/eval.yaml"),
}


def _image_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images")
    if isinstance(images, list):
        paths = images
    elif isinstance(images, dict):
        main = images.get("main")
        details = images.get("detail")
        main_path = main.get("image_id") if isinstance(main, dict) else None
        paths = [
            main_path,
            *(item.get("image_id") if isinstance(item, dict) else None for item in details)
        ] if isinstance(details, list) else []
    else:
        paths = []
    if len(paths) != 3 or not all(isinstance(path, str) and Path(path).is_file() for path in paths):
        raise FileNotFoundError(f"expected three existing dataset images: {row.get('sample_id') or row.get('product_id')} -> {paths}")
    return [str(path) for path in paths]


@lru_cache(maxsize=None)
def _source_image_id(path: str) -> str:
    return f"sha256:{sha256_file(path)}"


def _sample_id(row: dict[str, Any], dataset_stage: str, split: str) -> str:
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    product_id = row.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("dataset row requires sample_id or product_id")
    dataset = str(row.get("dataset") or f"{dataset_stage}_{split}")
    return f"{dataset}_{product_id}"


def _normalize(
    row: dict[str, Any],
    image_paths: list[str],
    dataset_stage: str,
    split: str,
) -> dict[str, Any]:
    violation_type = row.get("violation_type")
    evidence = None
    if violation_type in {"image_quality", "wrong_image"}:
        target_image_ref = row.get("target_image_ref")
        if target_image_ref not in IMAGE_REFS:
            raise ValueError(f"{violation_type} requires target_image_ref: {row.get('sample_id') or row.get('product_id')}")
        evidence = target_image_ref

    sample_id = _sample_id(row, dataset_stage, split)
    source_product_id = str(row.get("source_product_id") or row.get("product_id") or sample_id)
    source_image_ids = [_source_image_id(path) for path in image_paths]
    sample = {
        "sample_id": sample_id,
        "images": image_paths,
        "derived_image_id": f"sample:{sample_id}",
        "source_product_ids": [source_product_id],
        "source_image_ids": source_image_ids,
        "dataset_stage": dataset_stage,
        "split": split,
        "decision": "pass" if violation_type == "pass" else "reject",
        "violation_type": violation_type,
        "issue_subtype": row.get("issue_subtype") if violation_type == "image_quality" else None,
        "evidence": evidence,
        "target_image_ref": row.get("target_image_ref"),
        "title": row.get("title"),
        "category": row.get("category"),
        "color": row.get("color"),
        "material": row.get("material"),
        "difficulty": row.get("difficulty"),
    }
    sample["lineage"] = {
        "dataset_stage": dataset_stage,
        "source_product_ids": sample["source_product_ids"],
        "source_image_ids": source_image_ids,
        "derived_image_id": sample["derived_image_id"],
    }
    audit = {
        key: row[key]
        for key in ("changed_field", "title_audit", "wrong_image_audit")
        if key in row
    }
    if audit:
        sample["changed_field"] = row.get("changed_field")
        sample["title_audit"] = row.get("title_audit")
        sample["wrong_image_audit"] = row.get("wrong_image_audit")
        sample["lineage"]["audit"] = audit
    target_from_sample(sample)
    return sample


def prepare_partition(
    source: str | Path,
    dataset_stage: str,
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(source)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"direct dataset input is empty: {source}")
    return [
        _normalize(row, _image_paths(row), dataset_stage, split)
        for row in rows
    ]


def _sft_rows(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample["sample_id"],
            "dataset_stage": "sft",
            "split": sample["split"],
            "images": sample["images"],
            "lineage": sample["lineage"],
            "conversations": [
                {
                    "from": "human",
                    "value": product_prompt(
                        sample["title"],
                        sample["category"],
                        sample["color"],
                        sample["material"],
                        image_placeholders=True,
                    ),
                },
                {
                    "from": "gpt",
                    "value": json.dumps(
                        target_from_sample(sample),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        for sample in samples
    ]


def _grpo_rows(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, sample in enumerate(samples):
        text = product_prompt(
            sample["title"],
            sample["category"],
            sample["color"],
            sample["material"],
            image_placeholders=False,
        )
        rows.append(
            {
                "data_source": "vlm_product_audit",
                "prompt": structured_prompt(sample["images"], text),
                "images": sample["images"],
                "ability": "product_audit",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        target_from_sample(sample),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
                "extra_info": {
                    "dataset_stage": "grpo",
                    "training_stage": "grpo",
                    "split": sample["split"],
                    "index": index,
                    "sample_id": sample["sample_id"],
                    "lineage": sample["lineage"],
                },
            }
        )
    return rows



def _prepare_pair(config: dict[str, Any], stage: str, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = prepare_partition(config["source_dataset"], stage, "train", limit)
    validation = prepare_partition(
        config["validation_source_dataset"],
        stage,
        "validation",
        limit,
    )
    return train, validation


def _write_sft_variants(
    config: dict[str, Any],
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> list[Path]:
    train_rows = _sft_rows(train)
    validation_rows = _sft_rows(validation)
    full_train_path = Path(config.get("full_dataset", config["dataset"]))
    full_validation_path = Path(config.get("full_validation_dataset", config["validation_dataset"]))
    active_train_path = Path(config["dataset"])
    active_validation_path = Path(config["validation_dataset"])

    active_train_limit = config.get("active_train_max_samples")
    active_validation_limit = config.get("active_validation_max_samples")
    if active_train_limit is not None:
        active_train_limit = int(active_train_limit)
        if not 1 <= active_train_limit <= len(train_rows):
            raise ValueError("active_train_max_samples must be within the full SFT train size")
    if active_validation_limit is not None:
        active_validation_limit = int(active_validation_limit)
        if not 1 <= active_validation_limit <= len(validation_rows):
            raise ValueError("active_validation_max_samples must be within the full SFT validation size")

    active_train_rows = train_rows[:active_train_limit] if active_train_limit else train_rows
    active_validation_rows = (
        validation_rows[:active_validation_limit] if active_validation_limit else validation_rows
    )
    write_jsonl(full_train_path, train_rows)
    write_jsonl(full_validation_path, validation_rows)
    if active_train_path != full_train_path:
        write_jsonl(active_train_path, active_train_rows)
    if active_validation_path != full_validation_path:
        write_jsonl(active_validation_path, active_validation_rows)
    return [active_train_path, active_validation_path]


def prepare_stage(config_path: str | Path, stage: str, limit: int | None = None) -> list[Path]:
    config = load_yaml(config_path)
    if stage == "sft":
        train, validation = _prepare_pair(config, "sft", limit)
        return _write_sft_variants(config, train, validation)
    if stage == "grpo":
        train, validation = _prepare_pair(config, "grpo", limit)
        outputs = [Path(config["dataset"]), Path(config["validation_dataset"])]
        write_jsonl(outputs[0], _grpo_rows(train))
        write_jsonl(outputs[1], _grpo_rows(validation))
        return outputs
    if stage == "eval":
        source = config.get("source_dataset", config.get("dataset"))
        if not source:
            raise ValueError("eval config requires source_dataset or dataset")
        split = str(config.get("dataset_split", "test"))
        samples = prepare_partition(source, "eval", split, limit)
        output = Path(config["manifest"])
        write_jsonl(output, samples)
        return [output]
    raise ValueError(f"unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare direct-image JSONL data without creating composite images"
    )
    parser.add_argument("--stage", choices=("sft", "grpo", "eval", "all"), default="all")
    parser.add_argument("--config", help="override the default config for one explicit stage")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    stages = tuple(DEFAULT_CONFIGS) if args.stage == "all" else (args.stage,)
    if args.config and len(stages) != 1:
        raise ValueError("--config requires one explicit stage")
    for stage in stages:
        config_path = Path(args.config) if args.config else DEFAULT_CONFIGS[stage]
        if not config_path.exists():
            continue
        outputs = prepare_stage(config_path, stage, args.limit)
        print(f"prepared {stage}: {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()

