"""Adapt current dataset rows to rendered veRL consumer inputs without reinjection."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.common import load_yaml, read_jsonl, sha256_file, write_jsonl
from src.evaluation.evaluate_synthesis import render_page
from src.models.audit_protocol import IMAGE_REFS, product_prompt, structured_prompt, target_from_sample


DEFAULT_CONFIGS = {
    "sft": Path("configs/sft.yaml"),
    "joint": Path("configs/joint.yaml"),
    "eval": Path("configs/eval.yaml"),
}


def _image_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"missing images object: {row.get('product_id')}")
    main = images.get("main")
    details = images.get("detail")
    main_path = main.get("image_id") if isinstance(main, dict) else None
    if not isinstance(details, list) or len(details) != 2:
        raise ValueError(f"expected exactly two detail images: {row.get('product_id')}")
    paths = [
        main_path,
        *(item.get("image_id") if isinstance(item, dict) else None for item in details),
    ]
    if not all(isinstance(path, str) and Path(path).is_file() for path in paths):
        raise FileNotFoundError(f"missing dataset image: {row.get('product_id')} -> {paths}")
    return [str(path) for path in paths]


@lru_cache(maxsize=None)
def _source_image_id(path: str) -> str:
    return f"sha256:{sha256_file(path)}"


def _normalize(
    row: dict[str, Any],
    image_paths: list[str],
    page: Path | None,
    dataset_stage: str,
    split: str,
) -> dict[str, Any]:
    violation_type = row.get("violation_type")
    evidence = None
    if violation_type in {"image_quality", "wrong_image"}:
        target_image_ref = row.get("target_image_ref")
        if target_image_ref not in IMAGE_REFS:
            raise ValueError(f"{violation_type} requires target_image_ref: {row.get('product_id')}")
        evidence = target_image_ref

    dataset_name = str(row.get("dataset") or f"{dataset_stage}_{split}")
    sample_id = f"{dataset_name}_{row['product_id']}"
    source_product_id = str(row.get("source_product_id") or row["product_id"])
    sample = {
        "sample_id": sample_id,
        "images": image_paths,
        "page": page.as_posix() if page else None,
        "derived_image_id": f"sample:{sample_id}",
        "source_product_ids": [source_product_id],
        "source_image_ids": [_source_image_id(path) for path in image_paths],
        "dataset_stage": dataset_stage,
        "split": split,
        "template_id": "synthesis_v2",
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
        "opd_enabled": row.get("opd_enabled") is True,
    }
    sample["lineage"] = {
        "dataset_stage": dataset_stage,
        "source_product_ids": sample["source_product_ids"],
        "source_image_ids": sample["source_image_ids"],
        "derived_image_id": sample["derived_image_id"],
    }
    target_from_sample(sample)
    return sample


def prepare_partition(
    source: str | Path,
    output_root: str | Path,
    dataset_stage: str,
    split: str,
    config: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(source)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"synthesis input is empty: {source}")

    width = int(config.get("page_width", 960))
    height = int(config.get("page_height", 720))
    samples = []
    for row in rows:
        paths = _image_paths(row)
        page = None
        if dataset_stage == "eval":
            dataset_name = str(row.get("dataset") or f"{dataset_stage}_{split}")
            sample_id = f"{dataset_name}_{row['product_id']}"
            page = Path(output_root) / split / "pages" / f"{sample_id}.png"
            render_page(row, page, width, height)
        samples.append(_normalize(row, paths, page, dataset_stage, split))
    return samples


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


def _joint_rows(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, sample in enumerate(samples):
        opd_enabled = sample["opd_enabled"]
        subpool = "opd" if opd_enabled else "grpo"
        lineage = dict(sample["lineage"])
        lineage["dataset_stage"] = subpool
        text = product_prompt(
            sample["title"],
            sample["category"],
            sample["color"],
            sample["material"],
            image_placeholders=False,
        )
        prompt = structured_prompt(sample["images"], text)
        row = {
            "data_source": "vlm_product_audit",
            "prompt": prompt,
            "opd_enabled": opd_enabled,
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(
                    target_from_sample(sample),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            "ability": "product_audit",
            "extra_info": {
                "dataset_stage": subpool,
                "training_stage": "joint",
                "split": sample["split"],
                "index": index,
                "sample_id": sample["sample_id"],
                "lineage": lineage,
            },
        }
        if opd_enabled:
            row["teacher_prompt"] = prompt
        rows.append(row)
    return rows


def _prepare_pair(config: dict[str, Any], stage: str, limit: int | None) -> tuple[list[dict], list[dict]]:
    root = Path(config["pages_root"])
    train = prepare_partition(config["source_dataset"], root, stage, "train", config, limit)
    validation = prepare_partition(
        config["validation_source_dataset"],
        root,
        stage,
        "validation",
        config,
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
    if stage == "joint":
        train, validation = _prepare_pair(config, "joint", limit)
        outputs = [Path(config["dataset"]), Path(config["validation_dataset"])]
        write_jsonl(outputs[0], _joint_rows(train))
        write_jsonl(outputs[1], _joint_rows(validation))
        return outputs
    if stage == "eval":
        split = str(config.get("dataset_split", "validation"))
        samples = prepare_partition(
            config["source_dataset"],
            config["pages_root"],
            "eval",
            split,
            config,
            limit,
        )
        output = Path(config["manifest"])
        write_jsonl(output, samples)
        return [output]
    raise ValueError(f"unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render authoritative V2 synthesis rows and export consumer formats"
    )
    parser.add_argument("--stage", choices=("sft", "joint", "eval", "all"), default="all")
    parser.add_argument("--config", help="override the default config for one explicit stage")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    stages = tuple(DEFAULT_CONFIGS) if args.stage == "all" else (args.stage,)
    if args.config and len(stages) != 1:
        raise ValueError("--config requires one explicit stage")
    for stage in stages:
        config_path = Path(args.config) if args.config else DEFAULT_CONFIGS[stage]
        outputs = prepare_stage(config_path, stage, args.limit)
        print(f"prepared {stage}: {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
