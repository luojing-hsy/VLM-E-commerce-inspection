"""Build a deterministic, flat 3,000-product image set from ``data/raw``.

The program only assigns image roles when the source metadata or filenames make
them explicit.  It never treats an arbitrary first image as the main image.

Example::

    python -m src.data.build_raw_clean \
        --source data/raw --output data/raw_clean --count 3000

Use ``--manifest`` when automatic manifest discovery is ambiguous.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PRODUCT_KEYS = ("product_id", "item_id", "listing_id", "source_record_id", "id")
MAIN_KEYS = (
    "main_image",
    "main_image_path",
    "main_image_id",
    "primary_image",
    "primary_image_path",
    "primary_image_id",
    "cover_image",
)
DETAIL_KEYS = (
    "detail_images",
    "detail_image_paths",
    "detail_image_ids",
    "other_images",
    "other_image_paths",
    "other_image_id",
    "gallery_images",
    "additional_images",
)
MAIN_NAME_RE = re.compile(r"(?:^|[_\-.])(main|primary|cover)(?:$|[_\-.])", re.IGNORECASE)
DETAIL_NAME_RE = re.compile(
    r"(?:^|[_\-.])(detail|gallery|other|additional)(?:$|[_\-.0-9])", re.IGNORECASE
)


@dataclass(frozen=True)
class Product:
    source_id: str
    main: Path
    details: tuple[Path, ...]


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _data_suffix(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        return Path(path.stem).suffix.lower()
    return path.suffix.lower()


def _records_from_json_value(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        return (item for item in value if isinstance(item, dict))
    if isinstance(value, dict):
        for key in ("products", "items", "listings", "records", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return (item for item in nested if isinstance(item, dict))
        return (value,)
    return ()


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = _data_suffix(path)
    if suffix == ".csv":
        with _open_text(path) as handle:
            yield from csv.DictReader(handle)
        return

    with _open_text(path) as handle:
        if suffix == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value
        elif suffix == ".json":
            yield from _records_from_json_value(json.load(handle))
        else:
            raise ValueError(f"Unsupported manifest format: {path}")


def _find_key(record: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    lower_to_original = {str(key).lower(): key for key in record}
    for alias in aliases:
        original = lower_to_original.get(alias)
        if original is not None:
            return original
    return None


def _flatten_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[:1] in "[{":
            try:
                return _flatten_refs(json.loads(text))
            except json.JSONDecodeError:
                pass
        if ";" in text or "|" in text:
            return [part.strip() for part in re.split(r"[;|]", text) if part.strip()]
        return [text]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        refs: list[str] = []
        for item in value:
            refs.extend(_flatten_refs(item))
        return refs
    if isinstance(value, dict):
        for key in ("path", "file", "filename", "image_path", "image_id", "id", "value"):
            if key in value:
                return _flatten_refs(value[key])
        refs: list[str] = []
        for item in value.values():
            refs.extend(_flatten_refs(item))
        return refs
    return []


def _manifest_score(path: Path) -> int:
    try:
        record = next(iter_records(path))
    except (OSError, ValueError, json.JSONDecodeError, StopIteration, UnicodeDecodeError):
        return -1
    return (
        int(_find_key(record, PRODUCT_KEYS) is not None)
        + 3 * int(_find_key(record, MAIN_KEYS) is not None)
        + 2 * int(_find_key(record, DETAIL_KEYS) is not None)
    )


def discover_manifest(source: Path) -> Path | None:
    candidates = [
        path
        for path in source.rglob("*")
        if path.is_file() and _data_suffix(path) in {".json", ".jsonl", ".csv"}
    ]
    scored = sorted(((_manifest_score(path), path) for path in candidates), reverse=True)
    if not scored or scored[0][0] < 4:
        return None
    best_score = scored[0][0]
    tied = [path for score, path in scored if score == best_score]
    if len(tied) > 1:
        names = "\n  ".join(str(path) for path in tied[:10])
        raise ValueError(f"Multiple equally suitable manifests found; pass --manifest:\n  {names}")
    return tied[0]


def build_image_index(source: Path) -> dict[str, Path | None]:
    index: dict[str, Path | None] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(source).as_posix()
        keys = {relative.lower(), path.name.lower(), path.stem.lower()}
        for key in keys:
            previous = index.get(key)
            if previous is None and key in index:
                continue
            index[key] = path if previous is None else None
    return index


def resolve_image(ref: str, source: Path, index: dict[str, Path | None]) -> Path | None:
    normalized = ref.strip().replace("\\", "/").lstrip("./")
    direct = source / normalized
    if direct.is_file() and direct.suffix.lower() in IMAGE_SUFFIXES:
        return direct.resolve()
    for key in (normalized.lower(), Path(normalized).name.lower(), Path(normalized).stem.lower()):
        match = index.get(key)
        if match is not None:
            return match.resolve()
    return None


def products_from_manifest(source: Path, manifest: Path) -> tuple[list[Product], list[dict[str, Any]]]:
    image_index = build_image_index(source)
    products: list[Product] = []
    skipped: list[dict[str, Any]] = []
    for record_number, record in enumerate(iter_records(manifest), 1):
        product_key = _find_key(record, PRODUCT_KEYS)
        main_key = _find_key(record, MAIN_KEYS)
        detail_key = _find_key(record, DETAIL_KEYS)
        source_id = str(record.get(product_key, record_number)).strip()
        main_refs = _flatten_refs(record.get(main_key)) if main_key else []
        if len(main_refs) != 1:
            skipped.append({"source_id": source_id, "reason": "main image is absent or ambiguous"})
            continue
        main = resolve_image(main_refs[0], source, image_index)
        if main is None:
            skipped.append({"source_id": source_id, "reason": f"main image not found: {main_refs[0]}"})
            continue
        detail_refs = _flatten_refs(record.get(detail_key)) if detail_key else []
        details: list[Path] = []
        missing_details: list[str] = []
        for ref in detail_refs:
            detail = resolve_image(ref, source, image_index)
            if detail is None:
                missing_details.append(ref)
            elif detail != main and detail not in details:
                details.append(detail)
        products.append(Product(source_id=source_id, main=main, details=tuple(details)))
        if missing_details:
            skipped.append(
                {"source_id": source_id, "reason": "detail images not found", "refs": missing_details}
            )
    products.sort(key=lambda product: product.source_id)
    return products, skipped


def products_from_directories(source: Path) -> tuple[list[Product], list[dict[str, Any]]]:
    products: list[Product] = []
    skipped: list[dict[str, Any]] = []
    for directory in sorted((path for path in source.iterdir() if path.is_dir()), key=lambda p: p.name):
        images = sorted(
            (path.resolve() for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: path.as_posix(),
        )
        explicit_main = [path for path in images if MAIN_NAME_RE.search(path.stem)]
        if len(images) == 1:
            main = images[0]
            details: list[Path] = []
        elif len(explicit_main) == 1:
            main = explicit_main[0]
            details = [path for path in images if path != main]
        else:
            skipped.append(
                {"source_id": directory.name, "reason": "main image is absent or ambiguous"}
            )
            continue
        products.append(Product(directory.name, main, tuple(details)))
    return products, skipped


def _copy_name(base: str, role: str, source: Path) -> str:
    return f"{base}_{role}{source.suffix.lower()}"


def write_dataset(
    products: list[Product],
    skipped: list[dict[str, Any]],
    output: Path,
    count: int,
    source: Path,
    manifest: Path | None,
) -> None:
    if len(products) < count:
        raise ValueError(f"Only {len(products)} valid products found; {count} required")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        mapping_path = staging / "manifest.jsonl"
        with mapping_path.open("w", encoding="utf-8", newline="\n") as mapping:
            for number, product in enumerate(products[:count], 1):
                base = f"product_{number:04d}"
                main_name = _copy_name(base, "main", product.main)
                shutil.copy2(product.main, staging / main_name)
                detail_names: list[str] = []
                for detail_number, detail in enumerate(product.details, 1):
                    name = _copy_name(base, f"detail_{detail_number}", detail)
                    shutil.copy2(detail, staging / name)
                    detail_names.append(name)
                row = {
                    "product_id": base,
                    "source_product_id": product.source_id,
                    "main_image": main_name,
                    "detail_images": detail_names,
                    "source_main_image": product.main.relative_to(source.resolve()).as_posix(),
                    "source_detail_images": [
                        path.relative_to(source.resolve()).as_posix() for path in product.details
                    ],
                }
                mapping.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        summary = {
            "product_count": count,
            "image_count": sum(1 + len(product.details) for product in products[:count]),
            "source": str(source.resolve()),
            "source_manifest": str(manifest.resolve()) if manifest else None,
            "skipped_or_missing_detail_count": len(skipped),
            "skipped_or_missing_details": skipped,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/raw_clean"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--count", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of it")

    manifest = args.manifest.resolve() if args.manifest else discover_manifest(source)
    if manifest:
        products, skipped = products_from_manifest(source, manifest)
    else:
        products, skipped = products_from_directories(source)
    write_dataset(products, skipped, output, args.count, source, manifest)
    print(f"Created {args.count} products in {output}")


if __name__ == "__main__":
    main()
