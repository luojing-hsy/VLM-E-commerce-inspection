from __future__ import annotations

import argparse
import subprocess

from src.data.prepare_synthesis import prepare_stage
from src.training.runtime import (
    apply_resume_options,
    assert_joint_config,
    build_verl_command,
    launch_verl,
    validate_stage_config,
    write_run_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch joint cost-sensitive GRPO + regional OPD with veRL")
    parser.add_argument("--config", default="configs/joint.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--print-command", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-from", help="resume from an explicit veRL global_step checkpoint directory")
    resume.add_argument("--restart", action="store_true", help="disable automatic resume and start a new run")
    args = parser.parse_args()
    prepare_stage(args.config, "joint")
    config = validate_stage_config(args.config, "joint")
    apply_resume_options(config, args.resume_from, args.restart)
    assert_joint_config(config)
    manifest = write_run_manifest(args.config, config)
    print(f"veRL joint run inputs validated; metadata: {manifest}")
    if args.prepare_only:
        return
    if args.print_command:
        print(subprocess.list2cmdline(build_verl_command(config)))
        return
    launch_verl(config)


if __name__ == "__main__":
    main()
