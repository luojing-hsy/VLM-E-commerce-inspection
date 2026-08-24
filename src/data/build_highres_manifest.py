"""Build product-level text and image records for the high-resolution image set."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tarfile
from pathlib import Path


def _unique_text(row: dict, key: str, locale: str | None = None, limit: int = 240) -> str | None:
    values = {
        str(value.get("value", "")).strip()
        for value in row.get(key) or []
        if isinstance(value, dict)
        and (locale is None or value.get("language_tag") == locale)
        and 0 < len(str(value.get("value", "")).strip()) <= limit
    }
    return next(iter(values)) if len(values) == 1 else None



def _first_text(row: dict, key: str, locale: str | None = None, limit: int = 240) -> str | None:
    for value in row.get(key) or []:
        if not isinstance(value, dict):
            continue
        if locale is not None and value.get("language_tag") != locale:
            continue
        text = str(value.get("value", "")).strip()
        if 0 < len(text) <= limit:
            return text
    return None

def _product_type(row: dict) -> str | None:
    values = row.get("product_type")
    if isinstance(values, list):
        values = {str(item.get("value", "")).strip() for item in values if isinstance(item, dict)}
        return next(iter(values)) if len(values) == 1 else None

    value = str(values or "").strip()
    return value or None


def _dimensions(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    parts = []
    for key in ("width", "length", "height"):
        item = value.get(key)
        if isinstance(item, dict) and item.get("value") is not None and item.get("unit"):
            parts.append(f"{key[0].upper()} {item['value']:g} {item['unit']}")
    return " x ".join(parts) or None


def _weight(value: object) -> str | None:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        return None
    item = value[0]
    if item.get("value") is None or not item.get("unit"):
        return None
    return f"{item['value']:g} {item['unit']}"


def _text_from_listing(row: dict, source: str) -> dict:
    brand = (
        _unique_text(row, "brand", "en_US", 80)
        or _first_text(row, "brand", "en_US", 80)
        or _first_text(row, "brand", None, 80)
    )
    attributes = {
        "brand": brand,
        "color": _unique_text(row, "color", "en_US", 80),
        "material": _unique_text(row, "material", "en_US", 100),
        "model": _unique_text(row, "model_number", None, 100)
        or _unique_text(row, "model_name", "en_US", 100),
        "dimensions": _dimensions(row.get("item_dimensions")),
        "weight": _weight(row.get("item_weight")),
    }
    return {
        "title": _unique_text(row, "item_name", "en_US")
        or _first_text(row, "item_name", "en_US")
        or _first_text(row, "item_name"),
        "brand": brand,
        "category": (_product_type(row) or "").lower() or None,
        "product_type": _product_type(row),
        "attributes": {key: value for key, value in attributes.items() if value is not None},
        "text_source": source,
    }


def _load_existing_text(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for relative in (
        "data/manifests/products_train.jsonl",
        "data/manifests/products_validation.jsonl",
        "data/manifests/products_test.jsonl",
    ):
        path = root / relative
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                item_id = row.get("item_id")
                if item_id:
                    result[str(item_id)] = {
                        "title": row.get("title") or row.get("original_title"),
                        "brand": row.get("brand"),
                        "category": row.get("category"),
                        "product_type": row.get("product_type"),
                        "attributes": row.get("attributes") or {},
                        "text_source": "products_manifest",
                    }
    return result


def _load_listing_text(archive_path: Path, needed: set[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not needed:
        return result
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".json.gz"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            with gzip.open(handle, "rt", encoding="utf-8") as listings:
                for line in listings:
                    row = json.loads(line)
                    item_id = str(row.get("item_id", ""))
                    if item_id in needed:
                        result[item_id] = _text_from_listing(row, "abo_listing_archive")
            if needed.issubset(result):
                break
    return result


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--highres", type=Path, default=Path("data/raw_clean_highres"))
    args = parser.parse_args()
    root = args.root.resolve()
    highres = (root / args.highres).resolve()

    lowres_rows = _read_jsonl(root / "data/raw_clean/manifest.jsonl")
    correspondence_path = highres / "correspondence.jsonl"
    correspondence = _read_jsonl(correspondence_path)
    by_name = {row["low_resolution_name"]: row for row in correspondence}
    for row in correspondence:
        row["high_resolution_path"] = row["high_resolution_path"].removeprefix("images/")
        if not (highres / row["high_resolution_path"]).is_file():
            raise FileNotFoundError(row["high_resolution_path"])
    _write_jsonl(correspondence_path, correspondence)

    text = _load_existing_text(root)
    source_ids = {row["source_product_id"] for row in lowres_rows}
    text.update(
        _load_listing_text(
            root / "data/raw/abo/metadata/abo-listings.tar",
            source_ids - text.keys(),
        )
    )

    products = []
    for row in lowres_rows:
        names = [row["main_image"], *row["detail_images"]]
        source_paths = [row["source_main_image"], *row["source_detail_images"]]
        assets = []
        for role, name, source_path in zip(("main", "detail:1", "detail:2"), names, source_paths, strict=True):
            asset = dict(by_name[name])
            asset.update({"role": role, "source_path": source_path})
            assets.append(asset)
        products.append(
            {
                "schema_version": "1.0",
                "product_id": row["product_id"],
                "source_product_id": row["source_product_id"],
                "title": text.get(row["source_product_id"], {}).get("title"),
                "brand": text.get(row["source_product_id"], {}).get("brand"),
                "category": text.get(row["source_product_id"], {}).get("category"),
                "product_type": text.get(row["source_product_id"], {}).get("product_type"),
                "attributes": text.get(row["source_product_id"], {}).get("attributes", {}),
                "text_source": text.get(row["source_product_id"], {}).get("text_source"),
                "highres_images": assets,
            }
        )

    _write_jsonl(highres / "highres_products.jsonl", products)
    print(
        json.dumps(
            {
                "product_count": len(products),
                "text_record_count": sum(bool(row["text_source"]) for row in products),
                "correspondence_count": len(correspondence),
                "output": str(highres / "highres_products.jsonl"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
