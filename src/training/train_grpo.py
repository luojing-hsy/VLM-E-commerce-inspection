from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from src.common import load_yaml
from src.data.prepare_dataset import prepare_stage
from src.training.grpo_checkpoint_policy import ensure_grpo_latest_only_checkpoint_hook
from src.training.runtime import (
    apply_resume_options,
    build_verl_command,
    export_grpo_hf_checkpoint,
    launch_verl,
    validate_stage_config,
    write_run_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch pure GRPO with veRL")
    parser.add_argument("--config", default="configs/grpo.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--print-command", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-from", help="resume from an explicit veRL global_step checkpoint directory")
    resume.add_argument("--restart", action="store_true", help="disable automatic resume and start a new run")
    args = parser.parse_args()

    source_config = load_yaml(args.config)
    targets = prepare_stage(args.config, "grpo", source_config.get("prepare_max_samples"))
    print(f"GRPO data prepared: {', '.join(str(path) for path in targets)}")

    config = validate_stage_config(args.config, "grpo")
    model_path = Path(config["model_name_or_path"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    sft_roots = {parent for parent in model_path.parents if parent.name == "sft"}
    if output_dir.name == "sft" or any(
        sft_root == output_dir or sft_root in output_dir.parents for sft_root in sft_roots
    ):
        raise RuntimeError(
            "GRPO output_dir must not be the SFT checkpoint root or its child: "
            f"output_dir={output_dir}, model={model_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"GRPO student checkpoint does not exist: {model_path}")
    apply_resume_options(config, args.resume_from, args.restart)
    manifest = write_run_manifest(args.config, config)
    print(f"veRL GRPO inputs validated; metadata: {manifest}")

    if args.prepare_only:
        return
    if args.print_command:
        print(shlex.join(build_verl_command(config, sys.executable)))
        return

    hook_path = ensure_grpo_latest_only_checkpoint_hook()
    print(f"GRPO latest-only checkpoint hook: {hook_path}")

    started = time.perf_counter()
    launch_verl(config)
    training_seconds = time.perf_counter() - started
    if config.get("export_hf_checkpoint", True):
        export_config = dict(config)
        export_config["latest_alias"] = config.get(
            "hf_latest_alias",
            str(Path(config["output_dir"]) / "hf_latest"),
        )
        checkpoint = export_grpo_hf_checkpoint(export_config)
        print(f"latest GRPO Hugging Face checkpoint: {checkpoint}")
    print(f"GRPO training seconds: {training_seconds:.3f}")


if __name__ == "__main__":
    main()
