from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Lock

from PIL import Image

from src.common import sha256_file, stable_hash, write_jsonl
from src.data.split_manifest import stable_split_for, write_split_manifests


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _download(url: str, target: Path) -> int:
    if target.exists():
        return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vlm-qwen3vl-research/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    partial.replace(target)
    return target.stat().st_size


def _extract_listing_metadata(archive: Path, metadata_root: Path) -> tuple[list[Path], Path]:
    listing_dir = metadata_root / "listings"
    license_path = metadata_root / "LICENSE-CC-BY-4.0.txt"
    expected = [listing_dir / f"listings_{value:x}.json.gz" for value in range(16)]
    if all(path.exists() for path in expected) and license_path.exists():
        return expected, license_path
    listing_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            name = member.name.replace("\\", "/")
            if name == "LICENSE-CC-BY-4.0.txt":
                target = license_path
            elif name.startswith("listings/metadata/") and name.endswith(".json.gz"):
                target = listing_dir / Path(name).name
            else:
                continue
            stream = source.extractfile(member)
            if stream is None:
                continue
            target.write_bytes(stream.read())
    missing = [str(path) for path in [*expected, license_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ABO listing archive is incomplete: {missing}")
    return expected, license_path


def _load_image_metadata(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["image_id"]] = {
                "source_image_id": row["image_id"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "object_path": row["path"],
            }
    return result


def _unique_text(row: dict, key: str, locale: str | None = None, max_length: int = 240) -> str | None:
    values = {
        str(value.get("value", "")).strip()
        for value in row.get(key) or []
        if isinstance(value, dict)
        and (locale is None or value.get("language_tag") == locale)
        and 0 < len(str(value.get("value", "")).strip()) <= max_length
    }
    return next(iter(values)) if len(values) == 1 else None


def _product_type(row: dict) -> str | None:
    value = row.get("product_type")
    if isinstance(value, list):
        values = {str(item.get("value", "")).strip() for item in value if isinstance(item, dict)}
        return next(iter(values)) if len(values) == 1 else None
    value = str(value or "").strip()
    return value or None


def _format_dimensions(value: dict | None) -> str | None:
    if not isinstance(value, dict):
        return None
    parts = []
    for key in ("width", "length", "height"):
        item = value.get(key)
        if not isinstance(item, dict) or item.get("value") is None or not item.get("unit"):
            continue
        parts.append(f"{key[0].upper()} {item['value']:g} {item['unit']}")
    return " x ".join(parts) if parts else None


def _format_weight(value: list | None) -> str | None:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        return None
    item = value[0]
    if item.get("value") is None or not item.get("unit"):
        return None
    return f"{item['value']:g} {item['unit']}"


def _normalized_candidate(
    row: dict,
    image_metadata: dict[str, dict],
    locale: str,
    min_width: int,
    min_height: int,
) -> dict | None:
    title = _unique_text(row, "item_name", locale)
    product_type = _product_type(row)
    brand = _unique_text(row, "brand", locale, 80) or _unique_text(row, "brand", None, 80)
    image_ids = [row.get("main_image_id"), *(row.get("other_image_id") or [])]
    image_ids = [str(value) for value in image_ids if value]
    usable_image_ids = []
    for image_id in image_ids:
        metadata = image_metadata.get(image_id)
        if metadata and metadata["width"] >= min_width and metadata["height"] >= min_height:
            usable_image_ids.append(image_id)
    if not title or not product_type or not brand or len(usable_image_ids) < 3:
        return None
    attributes = {
        "brand": brand,
        "color": _unique_text(row, "color", locale, 80),
        "material": _unique_text(row, "material", locale, 100),
        "model": _unique_text(row, "model_number", None, 100)
        or _unique_text(row, "model_name", locale, 100),
        "dimensions": _format_dimensions(row.get("item_dimensions")),
        "weight": _format_weight(row.get("item_weight")),
    }
    attributes = {key: value for key, value in attributes.items() if value is not None}
    audit_field = next(
        (key for key in ("model", "color", "material", "dimensions", "weight", "brand") if key in attributes),
        None,
    )
    if audit_field is None:
        return None
    domain = str(row.get("domain_name") or "unknown-domain")
    item_id = str(row["item_id"])
    product_id = f"abo:{domain}:{item_id}"
    return {
        "product_id": product_id,
        "item_id": item_id,
        "domain_name": domain,
        "title": title,
        "original_title": title,
        "category": product_type.lower(),
        "product_type": product_type,
        "brand": brand,
        "attributes": attributes,
        "audit_field": audit_field,
        "required_fields": ["brand"],
        "image_ids": usable_image_ids[:3],
        "spin_id": row.get("spin_id"),
        "3dmodel_id": row.get("3dmodel_id"),
        "source": "abo",
        "source_record_id": {"item_id": item_id, "domain_name": domain},
    }


def _metadata_family_key(product: dict) -> tuple[str, str, str] | None:
    model = str(product["attributes"].get("model", "")).casefold().strip()
    brand = str(product["brand"]).casefold().strip()
    if len(model) < 3 or len(brand) < 2:
        return None
    return product["product_type"], brand, model


def _build_metadata_components(products: list[dict]) -> list[list[int]]:
    union = UnionFind(len(products))
    relation_owner: dict[tuple[str, str], int] = {}
    for index, product in enumerate(products):
        relations = [("image", value) for value in product["image_ids"]]
        relations.extend(
            (kind, str(value))
            for kind, value in (("spin", product.get("spin_id")), ("3d", product.get("3dmodel_id")))
            if value
        )
        family_key = _metadata_family_key(product)
        if family_key:
            relations.append(("family", "\x1f".join(family_key)))
        for relation in relations:
            owner = relation_owner.setdefault(relation, index)
            union.union(index, owner)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(products)):
        groups[union.find(index)].append(index)
    return list(groups.values())


def _select_products(products: list[dict], seed: int, target: int, per_type_cap: int) -> list[dict]:
    representatives = []
    for component in _build_metadata_components(products):
        representative = min(
            component,
            key=lambda index: (
                -len(products[index]["attributes"]),
                stable_hash([seed, products[index]["product_id"]]),
            ),
        )
        product = dict(products[representative])
        component_ids = sorted(products[index]["product_id"] for index in component)
        product["metadata_component_id"] = f"metadata:{stable_hash(component_ids)[:20]}"
        representatives.append(product)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for product in representatives:
        by_type[product["product_type"]].append(product)
    for values in by_type.values():
        values.sort(key=lambda product: stable_hash([seed, product["product_id"]]))
        del values[per_type_cap:]
    type_order = sorted(by_type, key=lambda value: stable_hash([seed, value]))
    selected: list[dict] = []
    for offset in range(per_type_cap):
        for product_type in type_order:
            values = by_type[product_type]
            if offset < len(values):
                selected.append(values[offset])
                if len(selected) == target:
                    return selected
    raise ValueError(
        f"only {len(selected)} ABO products satisfy target={target} with max_products_per_type={per_type_cap}"
    )


def _safe_image_id(image_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", image_id)
    if safe != image_id:
        safe += f"__{hashlib.sha256(image_id.encode()).hexdigest()[:8]}"
    return safe


def _phash64(path: Path) -> int:
    with Image.open(path) as source:
        pixels = list(source.convert("L").resize((32, 32), Image.Resampling.LANCZOS).getdata())
    cosine = [[math.cos(math.pi * (2 * x + 1) * frequency / 64) for x in range(32)] for frequency in range(8)]
    row_dct = [
        [sum(pixels[y * 32 + x] * cosine[u][x] for x in range(32)) for u in range(8)]
        for y in range(32)
    ]
    values = [
        sum(row_dct[y][u] * cosine[v][y] for y in range(32))
        for v in range(8)
        for u in range(8)
    ]
    median = sorted(values[1:])[len(values[1:]) // 2]
    result = 0
    for index, value in enumerate(values):
        if value > median:
            result |= 1 << index
    return result


class _BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, int, dict] | None = None

    def add(self, value: int, owner: int) -> None:
        if self.root is None:
            self.root = (value, owner, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, owner, {})
                return
            node = child

    def query(self, value: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        result: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                result.append(node[1])
            stack.extend(
                child
                for edge, child in node[2].items()
                if distance - threshold <= edge <= distance + threshold
            )
        return result


def _download_selected_images(
    products: list[dict],
    image_metadata: dict[str, dict],
    config: dict,
    raw_root: Path,
) -> tuple[list[dict], int]:
    base_url = config["abo"]["small_image_base_url"].rstrip("/") + "/"
    image_root = raw_root / "images" / "by_id"
    assets: dict[str, dict] = {}
    for product in products:
        for role_index, image_id in enumerate(product["image_ids"]):
            metadata = dict(image_metadata[image_id])
            suffix = Path(metadata["object_path"]).suffix or ".jpg"
            local_path = image_root / _safe_image_id(image_id) / f"image{suffix}"
            metadata.update(
                {
                    "path": local_path.as_posix(),
                    "url": base_url + urllib.parse.quote(metadata["object_path"], safe="/"),
                    "role": "main" if role_index == 0 else f"gallery:{role_index - 1}",
                }
            )
            assets[image_id] = metadata
    total_lock = Lock()
    total_bytes = 0

    def download_one(asset: dict) -> None:
        nonlocal total_bytes
        size = _download(asset["url"], Path(asset["path"]))
        with total_lock:
            total_bytes += size

    with ThreadPoolExecutor(max_workers=int(config["abo"].get("download_workers", 12))) as pool:
        list(pool.map(download_one, assets.values()))
    budget = int(config["abo"]["max_selected_image_bytes"])
    if total_bytes > budget:
        raise ValueError(f"selected ABO images use {total_bytes} bytes, exceeding budget {budget}")
    for asset in assets.values():
        path = Path(asset["path"])
        with Image.open(path) as image:
            image.verify()
        asset["sha256"] = sha256_file(path)
        asset["phash64"] = f"{_phash64(path):016x}"
        asset["download_bytes"] = path.stat().st_size
    for product in products:
        product["images"] = [assets[value]["path"] for value in product["image_ids"]]
        product["image_assets"] = [assets[value] for value in product["image_ids"]]
    return list(assets.values()), total_bytes


def _merge_phash_components(products: list[dict], distance_threshold: int) -> None:
    union = UnionFind(len(products))
    exact_owner: dict[str, int] = {}
    tree = _BKTree()
    for product_index, product in enumerate(products):
        for asset in product["image_assets"]:
            sha_owner = exact_owner.setdefault(asset["sha256"], product_index)
            union.union(product_index, sha_owner)
            phash = int(asset["phash64"], 16)
            for near_owner in tree.query(phash, distance_threshold):
                union.union(product_index, near_owner)
            tree.add(phash, product_index)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(products)):
        groups[union.find(index)].append(index)
    for group in groups.values():
        component_ids = sorted(products[index]["metadata_component_id"] for index in group)
        component_id = f"source:{stable_hash(component_ids)[:20]}"
        for index in group:
            products[index]["source_component_id"] = component_id


def prepare_abo_products(config: dict) -> list[dict]:
    raw_root = Path(config["paths"].get("abo_root", "data/raw/abo"))
    metadata_root = raw_root / "metadata"
    listing_archive = metadata_root / "abo-listings.tar"
    image_metadata_path = metadata_root / "images.csv.gz"
    abo_config = config["abo"]
    metadata_bytes = _download(abo_config["listing_archive_url"], listing_archive)
    metadata_bytes += _download(abo_config["image_metadata_url"], image_metadata_path)
    if metadata_bytes > int(abo_config["max_metadata_bytes"]):
        raise ValueError("ABO metadata download exceeded the configured byte budget")
    listing_paths, license_path = _extract_listing_metadata(listing_archive, metadata_root)
    image_metadata = _load_image_metadata(image_metadata_path)
    candidates: list[dict] = []
    for path in listing_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                candidate = _normalized_candidate(
                    json.loads(line),
                    image_metadata,
                    str(abo_config["locale"]),
                    int(abo_config["min_image_width"]),
                    int(abo_config["min_image_height"]),
                )
                if candidate:
                    candidates.append(candidate)
    products = _select_products(
        candidates,
        int(config["seed"]),
        int(config["num_products"]),
        int(abo_config["max_products_per_type"]),
    )
    assets, selected_image_bytes = _download_selected_images(products, image_metadata, config, raw_root)
    _merge_phash_components(products, int(abo_config["phash_distance_threshold"]))
    for product in products:
        product["family_id"] = product["source_component_id"]
        product["split"] = stable_split_for(
            product["source_component_id"], int(config["seed"]), config["split_ratios"]
        )
        product["license"] = "CC-BY-NC-4.0-conservative"
        product["license_status"] = "official_source_conflict"
        product["source_archive_sha256"] = sha256_file(listing_archive)
    write_split_manifests(config, "products", products)
    manifest_root = Path(config["paths"]["manifests"])
    write_jsonl(manifest_root / "source_images.jsonl", assets)
    component_rows = [
        {
            "source_component_id": component_id,
            "split": rows[0]["split"],
            "source_product_ids": sorted(row["product_id"] for row in rows),
            "source_image_ids": sorted({value for row in rows for value in row["image_ids"]}),
        }
        for component_id, rows in _group_products(products).items()
    ]
    write_jsonl(manifest_root / "source_components.jsonl", component_rows)
    report = {
        "source_mode": "abo_rendered_audit",
        "accessed_at": date.today().isoformat(),
        "num_metadata_candidates": len(candidates),
        "num_selected_products": len(products),
        "num_selected_images": len(assets),
        "metadata_download_bytes": metadata_bytes,
        "selected_image_download_bytes": selected_image_bytes,
        "listing_archive_sha256": sha256_file(listing_archive),
        "license_sha256": sha256_file(license_path),
        "license_status": "official_source_conflict",
        "effective_policy": "CC-BY-NC-4.0-conservative",
        "product_type_counts": dict(sorted(Counter(row["product_type"] for row in products).items())),
        "max_products_per_type": max(Counter(row["product_type"] for row in products).values()),
        "content_hash": stable_hash(products),
    }
    target = manifest_root / "source_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return products


def _group_products(products: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        result[product["source_component_id"]].append(product)
    return result
