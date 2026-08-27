from __future__ import annotations

import argparse
from pathlib import Path

from src.common import load_yaml, read_jsonl, write_jsonl
from src.data.render_page import render_one
from src.data.split_manifest import read_split_manifests, write_split_manifests

CONSISTENCY_TYPES = {"PRODUCT_MISMATCH", "ATTRIBUTE_CONFLICT", "TEXT_LABEL_CONFLICT"}


def build_counterfactuals(config: dict, samples: list[dict] | None = None) -> list[dict]:
    samples = samples or read_split_manifests(config, "samples")
    products = {item["product_id"]: item for item in read_split_manifests(config, "products")}
    counterfactuals: list[dict] = []
    for sample in samples:
        if (
            sample["dataset_stage"] not in {"sft", "test"}
            or sample["violation_type"] not in CONSISTENCY_TYPES
        ):
            continue
        product = products[sample["source_product_ids"][0]]
        template_id = int(sample["template_id"].split("_")[-1])
        row = render_one(
            product,
            product,
            "pass",
            sample["split"],
            f"{sample['sample_id']}_cf",
            template_id,
            config,
            dataset_stage=sample["dataset_stage"],
        )
        row["counterfactual_of"] = sample["sample_id"]
        row["pair_id"] = f"pair:{sample['sample_id']}"
        counterfactuals.append(row)
    return counterfactuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimal consistency counterfactuals")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows = build_counterfactuals(config)
    targets = write_split_manifests(config, "counterfactuals", rows)
    print(f"wrote {len(rows)} split counterfactuals to {', '.join(str(path) for path in targets.values())}")


if __name__ == "__main__":
    main()
