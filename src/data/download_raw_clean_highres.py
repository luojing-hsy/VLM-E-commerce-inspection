"""Download the original-resolution images referenced by ``data/raw_clean``."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Asset:
    image_id: str
    object_path: str
    width: int
    height: int


def _read_selection(manifest_path: Path) -> tuple[dict[str, str], set[str]]:
    business_to_source: dict[str, str] = {}
    source_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pairs = [(row["main_image"], row["source_main_image"])]
            pairs.extend(zip(row["detail_images"], row["source_detail_images"], strict=True))
            for business_name, source_path in pairs:
                source_id = Path(source_path).stem
                previous = business_to_source.setdefault(business_name, source_id)
                if previous != source_id:
                    raise ValueError(f"{business_name} maps to multiple source images")
                source_ids.add(source_id)
    return business_to_source, source_ids


def _read_assets(metadata_path: Path, selected_ids: set[str]) -> dict[str, Asset]:
    assets: dict[str, Asset] = {}
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if image_id in selected_ids:
                assets[image_id] = Asset(
                    image_id=image_id,
                    object_path=row["path"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
    missing = selected_ids - assets.keys()
    if missing:
        raise ValueError(f"missing metadata for {len(missing)} selected images")
    return assets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(path: Path, asset: Asset) -> None:
    with Image.open(path) as image:
        size = image.size
        image.verify()
    if size != (asset.width, asset.height):
        raise ValueError(
            f"dimension mismatch for {asset.image_id}: {size} != "
            f"{(asset.width, asset.height)}"
        )


def _download_once(url: str, partial_path: Path) -> None:
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": "vlm-qwen3vl-data-preparation/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with partial_path.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def _download_asset(asset: Asset, base_url: str, image_root: Path, target_name: str) -> dict:
    suffix = Path(asset.object_path).suffix.lower() or ".jpg"
    target = image_root / target_name
    partial = target.with_suffix(target.suffix + ".part")
    url = f"{base_url.rstrip('/')}/{urllib.parse.quote(asset.object_path, safe='/')}"

    if target.exists():
        _verify_image(target, asset)
    else:
        for attempt in range(1, 6):
            try:
                _download_once(url, partial)
                _verify_image(partial, asset)
                os.replace(partial, target)
                break
            except (OSError, ValueError, urllib.error.URLError):
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 20))

    return {
        "image_id": asset.image_id,
        "object_path": asset.object_path,
        "url": url,
        "path": target.relative_to(image_root.parent).as_posix(),
        "width": asset.width,
        "height": asset.height,
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _materialize_copies_and_write_correspondence(
    output_root: Path,
    business_to_source: dict[str, str],
    source_rows: list[dict],
) -> None:
    sources = {row["image_id"]: row for row in source_rows}
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows = []
    items = sorted(business_to_source.items())
    for completed, (low_resolution_name, source_id) in enumerate(items, start=1):
        source = sources[source_id]
        source_path = output_root / source["path"]
        destination = image_root / low_resolution_name
        if destination.exists():
            if destination.stat().st_size != source["bytes"] or _sha256(destination) != source["sha256"]:
                raise FileExistsError(f"refusing to replace mismatched {destination}")
        else:
            shutil.copy2(source_path, destination)
        rows.append(
            {
                "low_resolution_name": low_resolution_name,
                "source_image_id": source_id,
                "high_resolution_path": destination.relative_to(output_root).as_posix(),
                "width": source["width"],
                "height": source["height"],
                "bytes": source["bytes"],
                "sha256": source["sha256"],
            }
        )
        if completed % 1000 == 0 or completed == len(items):
            print(f"materialized_and_verified={completed}/{len(items)}", flush=True)
    _write_jsonl(output_root / "correspondence.jsonl", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw_clean/manifest.jsonl"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/abo/metadata/images.csv.gz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default="https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    business_to_source, selected_ids = _read_selection(args.manifest)
    assets = _read_assets(args.metadata, selected_ids)
    plan = {
        "business_image_count": len(business_to_source),
        "unique_source_image_count": len(assets),
        "duplicate_reference_count": len(business_to_source) - len(assets),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    if args.plan_only:
        return

    output_root = args.output.resolve()
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    primary_names: dict[str, str] = {}
    for business_name, source_id in sorted(business_to_source.items()):
        primary_names.setdefault(source_id, business_name)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_asset, asset, args.base_url, image_root, primary_names[asset.image_id]): asset.image_id
            for asset in assets.values()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 1000 == 0 or completed == len(futures):
                print(f"downloaded_and_verified={completed}/{len(futures)}", flush=True)

    rows.sort(key=lambda row: row["image_id"])
    _write_jsonl(output_root / "source_manifest.jsonl", rows)
    shutil.copy2(args.manifest, output_root / "manifest.jsonl")
    _materialize_copies_and_write_correspondence(output_root, business_to_source, rows)
    print(
        json.dumps(
            {
                **plan,
                "downloaded_bytes": sum(row["bytes"] for row in rows),
                "source_manifest": str(output_root / "source_manifest.jsonl"),
                "correspondence_manifest": str(output_root / "correspondence.jsonl"),
                "business_image_root": str(output_root / "images"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
