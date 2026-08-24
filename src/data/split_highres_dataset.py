"""Create the fixed high-resolution SFT, joint-training, and test datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from PIL import Image, ImageFile

from src.common import read_jsonl, sha256_file, stable_hash, write_jsonl
from src.data.prepare_abo import UnionFind, _BKTree, _phash64


TARGET_SIZES = {
    "sft_train": 1400,
    "sft_valid": 120,
    "joint_train": 2000,
    "joint_valid": 180,
    "test": 286,
}

TARGET_PATHS = {
    "sft_train": Path("SFT/train"),
    "sft_valid": Path("SFT/valid"),
    "joint_train": Path("GRPO+OPD/train"),
    "joint_valid": Path("GRPO+OPD/valid"),
    "test": Path("test"),
}


@dataclass(frozen=True)
class Component:
    component_id: str
    member_indices: tuple[int, ...]
    strata: Counter[str]

    @property
    def size(self) -> int:
        return len(self.member_indices)


def _stable_fraction(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def _metadata_family_key(product: dict) -> tuple[str, str, str] | None:
    model = str(product.get("attributes", {}).get("model", "")).casefold().strip()
    brand = str(product.get("brand") or "").casefold().strip()
    product_type = str(product.get("product_type") or product.get("category") or "")
    if len(model) < 3 or len(brand) < 2:
        return None
    return product_type, brand, model


def _phash64_with_compatibility(path: Path) -> tuple[int, bool]:
    try:
        return _phash64(path), False
    except OSError as error:
        is_jpeg = path.suffix.casefold() in {".jpg", ".jpeg"}
        if not is_jpeg or "truncated" not in str(error).casefold():
            raise
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            return _phash64(path), True
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _build_components(
    products: list[dict],
    source_root: Path,
    phash_distance: int,
) -> tuple[list[tuple[int, ...]], dict[str, str], set[str]]:
    union = UnionFind(len(products))
    relation_owner: dict[tuple[str, object], int] = {}
    sha_owner: dict[str, int] = {}
    compatibility_shas: set[str] = set()
    phash_by_sha: dict[str, int] = {}
    tree = _BKTree()

    for index, product in enumerate(products):
        relations: list[tuple[str, object]] = [("source_product", product["source_product_id"])]
        family_key = _metadata_family_key(product)
        if family_key is not None:
            relations.append(("metadata_family", family_key))
        for relation in relations:
            owner = relation_owner.setdefault(relation, index)
            union.union(index, owner)

        for asset in product["highres_images"]:
            sha256 = str(asset["sha256"])
            owner = sha_owner.setdefault(sha256, index)
            union.union(index, owner)
            if sha256 in phash_by_sha:
                continue
            path = source_root / asset["high_resolution_path"]
            phash, used_compatibility = _phash64_with_compatibility(path)
            phash_by_sha[sha256] = phash
            if used_compatibility:
                compatibility_shas.add(sha256)
            for near_owner in tree.query(phash, phash_distance):
                union.union(index, near_owner)
            tree.add(phash, index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(products)):
        grouped[union.find(index)].append(index)
    components = [tuple(sorted(indices)) for indices in grouped.values()]
    components.sort(key=lambda indices: products[indices[0]]["product_id"])
    return (
        components,
        {sha: f"{value:016x}" for sha, value in phash_by_sha.items()},
        compatibility_shas,
    )


def _as_components(
    grouped_indices: list[tuple[int, ...]],
    products: list[dict],
    rare_threshold: int,
) -> tuple[list[Component], dict[int, str], Counter[str]]:
    category_counts = Counter(str(row["category"]) for row in products)
    stratum_by_index = {
        index: category if category_counts[category] >= rare_threshold else "__rare__"
        for index, row in enumerate(products)
        for category in (str(row["category"]),)
    }
    components = []
    for indices in grouped_indices:
        product_ids = sorted(products[index]["product_id"] for index in indices)
        components.append(
            Component(
                component_id=f"source:{stable_hash(product_ids)[:20]}",
                member_indices=indices,
                strata=Counter(stratum_by_index[index] for index in indices),
            )
        )
    return components, stratum_by_index, Counter(stratum_by_index.values())


def assign_components(
    components: list[Component],
    global_strata: Counter[str],
    seed: int,
    target_sizes: dict[str, int] = TARGET_SIZES,
) -> dict[str, str]:
    if sum(component.size for component in components) != sum(target_sizes.values()):
        raise ValueError("component and target totals differ")
    if max(component.size for component in components) > max(target_sizes.values()):
        raise ValueError("a leakage component is larger than every target dataset")

    multi = [component for component in components if component.size > 1]
    single = [component for component in components if component.size == 1]
    all_count = sum(target_sizes.values())

    for attempt in range(128):
        remaining = dict(target_sizes)
        assigned_counts = {target: Counter() for target in target_sizes}
        assignment: dict[str, str] = {}
        ordered_multi = sorted(
            multi,
            key=lambda item: (
                -item.size,
                _stable_fraction(seed + attempt, "multi-order", item.component_id),
            ),
        )
        failed = False
        for component in ordered_multi:
            candidates = [
                target for target, count in remaining.items() if count >= component.size
            ]
            if not candidates:
                failed = True
                break

            def score(target: str) -> tuple[float, float, float]:
                category_need = 0.0
                for stratum, count in component.strata.items():
                    goal = global_strata[stratum] * target_sizes[target] / all_count
                    current = assigned_counts[target][stratum]
                    category_need += count * (goal - current) / (goal + 1.0)
                category_need /= component.size
                capacity_need = remaining[target] / target_sizes[target]
                jitter = _stable_fraction(
                    seed + attempt, f"multi-target:{component.component_id}", target
                )
                return category_need, capacity_need, jitter

            target = max(candidates, key=score)
            assignment[component.component_id] = target
            remaining[target] -= component.size
            assigned_counts[target].update(component.strata)
        if failed:
            continue

        ordered_single = sorted(
            single,
            key=lambda item: _stable_fraction(seed, "single-order", item.component_id),
        )
        for component in ordered_single:
            stratum = next(iter(component.strata))
            candidates = [target for target, count in remaining.items() if count]

            def singleton_score(target: str) -> tuple[float, float, float]:
                goal = global_strata[stratum] * target_sizes[target] / all_count
                category_need = (goal - assigned_counts[target][stratum]) / (goal + 1.0)
                capacity_need = remaining[target] / target_sizes[target]
                jitter = _stable_fraction(
                    seed, f"single-target:{component.component_id}", target
                )
                return category_need, capacity_need, jitter

            target = max(candidates, key=singleton_score)
            assignment[component.component_id] = target
            remaining[target] -= 1
            assigned_counts[target][stratum] += 1
        if all(count == 0 for count in remaining.values()):
            return assignment

    raise RuntimeError("could not pack leakage components into the exact target sizes")


def _dataset_fields(target: str) -> tuple[str, str]:
    if target == "test":
        return "test", "test"
    stage = "sft" if target.startswith("sft_") else "joint"
    split = "validation" if target.endswith("_valid") else "train"
    return stage, split


def _distribution(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def _total_variation(rows: list[dict], all_rows: list[dict]) -> float:
    local = Counter(str(row["category"]) for row in rows)
    global_counts = Counter(str(row["category"]) for row in all_rows)
    return 0.5 * sum(
        abs(local[category] / len(rows) - count / len(all_rows))
        for category, count in global_counts.items()
    )


def _write_dataset(
    staging: Path,
    source_root: Path,
    products: list[dict],
    components: list[Component],
    component_targets: dict[str, str],
    phash_by_sha: dict[str, str],
) -> tuple[dict[str, list[dict]], Counter[str]]:
    component_by_index = {
        index: component
        for component in components
        for index in component.member_indices
    }
    rows_by_target: dict[str, list[dict]] = defaultdict(list)
    link_methods: Counter[str] = Counter()

    for index, product in enumerate(products):
        component = component_by_index[index]
        target = component_targets[component.component_id]
        stage, split = _dataset_fields(target)
        target_root = staging / TARGET_PATHS[target]
        target_root.mkdir(parents=True, exist_ok=True)
        output = dict(product)
        output.update(
            {
                "dataset_stage": stage,
                "split": split,
                "source_component_id": component.component_id,
            }
        )
        output_assets = []
        for asset in product["highres_images"]:
            source = source_root / asset["high_resolution_path"]
            destination = target_root / source.name
            try:
                os.link(source, destination)
                link_methods["hardlink"] += 1
            except OSError:
                shutil.copy2(source, destination)
                link_methods["copy_fallback"] += 1
            output_asset = dict(asset)
            output_asset["source_high_resolution_path"] = str(source.as_posix())
            output_asset["high_resolution_path"] = source.name
            output_asset["phash64"] = phash_by_sha[str(asset["sha256"])]
            output_assets.append(output_asset)
        output["highres_images"] = output_assets
        rows_by_target[target].append(output)

    for target, expected in TARGET_SIZES.items():
        rows = sorted(rows_by_target[target], key=lambda row: row["product_id"])
        if len(rows) != expected:
            raise AssertionError(f"{target}: expected {expected}, got {len(rows)}")
        rows_by_target[target] = rows
        write_jsonl(staging / TARGET_PATHS[target] / "manifest.jsonl", rows)
    return rows_by_target, link_methods


def _validate_output(
    staging: Path,
    rows_by_target: dict[str, list[dict]],
    component_targets: dict[str, str],
) -> None:
    product_targets: dict[str, str] = {}
    sha_targets: dict[str, str] = {}
    for target, rows in rows_by_target.items():
        for row in rows:
            product_id = str(row["product_id"])
            if product_id in product_targets:
                raise AssertionError(f"product appears twice: {product_id}")
            product_targets[product_id] = target
            if component_targets[row["source_component_id"]] != target:
                raise AssertionError(f"component crosses targets: {row['source_component_id']}")
            for asset in row["highres_images"]:
                image_path = staging / TARGET_PATHS[target] / asset["high_resolution_path"]
                with Image.open(image_path) as image:
                    image.verify()
                sha256 = str(asset["sha256"])
                previous = sha_targets.setdefault(sha256, target)
                if previous != target:
                    raise AssertionError(f"exact duplicate crosses targets: {sha256}")
    if len(product_targets) != sum(TARGET_SIZES.values()):
        raise AssertionError("not every product was written exactly once")


def build_split(
    source_root: Path,
    output_root: Path,
    seed: int,
    phash_distance: int,
    rare_threshold: int,
) -> dict:
    source_manifest = source_root / "highres_products.jsonl"
    products = read_jsonl(source_manifest)
    products.sort(key=lambda row: row["product_id"])
    if len(products) != sum(TARGET_SIZES.values()):
        raise ValueError(f"expected 3986 products, found {len(products)}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    grouped_indices, phash_by_sha, compatibility_shas = _build_components(
        products, source_root, phash_distance
    )
    components, _, global_strata = _as_components(
        grouped_indices, products, rare_threshold
    )
    component_targets = assign_components(components, global_strata, seed)

    staging = output_root.with_name(f"{output_root.name}.building-{uuid.uuid4().hex}")
    try:
        rows_by_target, link_methods = _write_dataset(
            staging,
            source_root,
            products,
            components,
            component_targets,
            phash_by_sha,
        )
        _validate_output(staging, rows_by_target, component_targets)
        assignments = []
        for target, rows in rows_by_target.items():
            assignments.extend(
                {
                    "category": row["category"],
                    "dataset_stage": row["dataset_stage"],
                    "product_id": row["product_id"],
                    "source_component_id": row["source_component_id"],
                    "source_product_id": row["source_product_id"],
                    "split": row["split"],
                    "target": target,
                }
                for row in rows
            )
        assignments.sort(key=lambda row: row["product_id"])
        write_jsonl(staging / "assignments.jsonl", assignments)

        component_sizes = Counter(component.size for component in components)
        report = {
            "schema_version": "1.0",
            "seed": seed,
            "source_manifest": str(source_manifest.as_posix()),
            "source_manifest_sha256": sha256_file(source_manifest),
            "phash_distance_threshold": phash_distance,
            "rare_category_threshold": rare_threshold,
            "truncated_compatible_image_count": sum(
                str(asset["sha256"]) in compatibility_shas
                for product in products
                for asset in product["highres_images"]
            ),
            "truncated_compatible_unique_sha_count": len(compatibility_shas),
            "product_count": len(products),
            "component_count": len(components),
            "largest_component": max(component.size for component in components),
            "component_size_histogram": dict(sorted(component_sizes.items())),
            "image_materialization": dict(link_methods),
            "targets": {
                target: {
                    "count": len(rows),
                    "path": str(TARGET_PATHS[target].as_posix()),
                    "category_total_variation_from_full": _total_variation(rows, products),
                    "category_counts": _distribution(rows),
                }
                for target, rows in rows_by_target.items()
            },
        }
        (staging / "split_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw_clean_highres"))
    parser.add_argument("--output", type=Path, default=Path("data/highres_split"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--rare-threshold", type=int, default=5)
    args = parser.parse_args()
    report = build_split(
        args.source.resolve(),
        args.output.resolve(),
        args.seed,
        args.phash_distance,
        args.rare_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
