from __future__ import annotations

import argparse

from src.training.runtime import validate_stage_config, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and stage a Qwen3-VL BF16 LoRA SFT run")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="validate inputs and write run metadata")
    args = parser.parse_args()
    config = validate_stage_config(args.config, "sft")
    manifest = write_run_manifest(args.config, config)
    print(f"SFT run inputs validated; metadata: {manifest}")
    if not args.prepare_only:
        raise SystemExit("Full 4B training is intentionally opt-in in this portfolio build. Re-run with --prepare-only, or connect this validated export to a pinned Qwen3-VL training environment.")


if __name__ == "__main__":
    main()
