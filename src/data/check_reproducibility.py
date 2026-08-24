from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.common import load_yaml, sha256_file
from src.data.split_manifest import SPLITS, manifest_path


def _tree_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(files)


def snapshot(config: dict) -> dict:
    manifest_root = Path(config["paths"]["manifests"])
    tracked = {}
    for stem in ("products", "samples", "counterfactuals", "sft", "opd", "grpo"):
        for split in SPLITS:
            tracked[f"{stem}_{split}"] = manifest_path(config, stem, split)
    missing = [str(path) for path in tracked.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing generated artifacts: {missing}")
    image_hash, image_count = _tree_hash(Path(config["paths"]["generated"]))
    return {
        "seed": int(config["seed"]),
        "files": {name: sha256_file(path) for name, path in tracked.items()},
        "generated_image_tree_sha256": image_hash,
        "generated_image_count": image_count,
    }


def check(config: dict, record: bool = False) -> dict:
    target = Path(config["paths"]["manifests"]) / "reproducibility.json"
    current = snapshot(config)
    if record or not target.exists():
        report = {"status": "recorded", "baseline": current, "current": current, "matched": None}
    else:
        previous = json.loads(target.read_text(encoding="utf-8"))["baseline"]
        matched = previous == current
        report = {"status": "pass" if matched else "fail", "baseline": previous, "current": current, "matched": matched}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or compare deterministic generated-data hashes")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--record", action="store_true", help="replace the stored baseline snapshot")
    args = parser.parse_args()
    report = check(load_yaml(args.config), args.record)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
