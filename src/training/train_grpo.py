from __future__ import annotations

import argparse
import subprocess

from src.training.runtime import build_verl_command, launch_verl, validate_stage_config, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, inspect, or launch cost-sensitive GRPO with veRL")
    parser.add_argument("--config", default="configs/grpo.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="validate without importing or installing veRL")
    mode.add_argument("--print-command", action="store_true", help="print the veRL command without running it")
    args = parser.parse_args()
    config = validate_stage_config(args.config, "grpo")
    manifest = write_run_manifest(args.config, config)
    print(f"veRL GRPO run inputs validated; metadata: {manifest}")
    if args.prepare_only:
        return
    if args.print_command:
        print(subprocess.list2cmdline(build_verl_command(config)))
        return
    launch_verl(config)


if __name__ == "__main__":
    main()
