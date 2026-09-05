#!/usr/bin/env python3
"""Check configured JSONL/image inputs and print commands without launching models."""
import argparse
import hashlib
import json
import os
import shlex
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["baseline", "sft", "grpo", "all"], default="all")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.config and args.stage == "all":
        parser.error("--config requires one stage")
    os.chdir(ROOT)
    stages = ["baseline", "sft", "grpo"] if args.stage == "all" else [args.stage]
    checked = set()
    for stage in stages:
        config_path = args.config or f"configs/{stage}.yaml"
        config = yaml.safe_load(Path(config_path).read_text())
        sources = [config["source_dataset"]]
        if stage != "baseline":
            sources.append(config["validation_source_dataset"])
        for source in sources:
            path = Path(source)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not rows:
                raise ValueError(f"Empty dataset: {source}")
            for row in rows:
                images = row["images"]
                paths = images if isinstance(images, list) else [
                    images["main"]["image_id"], *[x["image_id"] for x in images["detail"]]
                ]
                if len(paths) != 3:
                    raise ValueError(f"Expected three image references: {source}")
                for image in paths:
                    if image not in checked:
                        with Image.open(image) as im:
                            im.verify()
                        checked.add(image)
            print(json.dumps({
                "source": source, "samples": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "classes": dict(Counter(row["violation_type"] for row in rows)),
            }, ensure_ascii=False))
        if stage == "baseline":
            local = ROOT.parent / "models/Qwen3.5-4B"
            model = os.environ.get("BASEMODEL_MODEL", str(local) if local.is_dir() else "Qwen/Qwen3.5-4B")
            predictions = os.environ.get("BASEMODEL_PREDICTIONS", config["predictions"])
            for module, options in [
                ("src.data.prepare_dataset", ["--stage", "eval", "--config", config_path]),
                ("src.evaluation.predict", ["--config", config_path, "--model", model, "--output", predictions]),
                ("src.evaluation.evaluate", ["--config", config_path, "--predictions", predictions]),
            ]:
                print(shlex.join([str(ROOT / ".venv/bin/python"), "-m", module, *options]))
    print(f"Verified {len(checked)} unique image files")

if __name__ == "__main__":
    main()
