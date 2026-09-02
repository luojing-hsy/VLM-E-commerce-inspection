from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from src.data.prepare_synthesis import prepare_stage
from src.evaluation.evaluate_synthesis import run as run_evaluation
from src.training.runtime import (
    apply_resume_options,
    assert_joint_config,
    build_verl_command,
    launch_verl,
    validate_stage_config,
    write_run_manifest,
)


def _run_validation(config: dict, checkpoint: Path, training_seconds: float) -> None:
    predictions = Path(config.get("validation_predictions", "outputs/joint/validation_predictions.jsonl"))
    metrics = Path(config.get("validation_metrics", "outputs/joint/validation_metrics.json"))
    manifest = Path(config.get("validation_manifest", "outputs/joint/validation_manifest.jsonl"))
    if predictions.exists():
        predictions.unlink()
    args = argparse.Namespace(
        config=config.get("validation_eval_config", "configs/eval.yaml"),
        dataset=config["validation_source_dataset"],
        model=str(checkpoint),
        pages=config.get("pages_root", "outputs/joint/runtime"),
        predictions=str(predictions),
        metrics=str(metrics),
        page_width=int(config.get("page_width", 960)),
        page_height=int(config.get("page_height", 720)),
        batch_size=int(config.get("validation_batch_size", 8)),
        limit=config.get("post_training_validation_max_samples"),
        split="validation",
        mode="joint_validation",
        manifest=str(manifest),
        min_pixels=config.get("min_pixels"),
        max_pixels=config.get("max_pixels"),
    )
    report = run_evaluation(args)
    report["checkpoint"] = str(checkpoint)
    report["training_seconds"] = round(training_seconds, 3)
    report["stage"] = "joint"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Joint validation report: {metrics}")


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
    started = time.perf_counter()
    checkpoint = launch_verl(config)
    training_seconds = time.perf_counter() - started
    if checkpoint is None:
        raise RuntimeError("Joint training finished without an exported Hugging Face checkpoint")
    print(f"latest Joint Hugging Face checkpoint: {checkpoint}")
    if config.get("run_validation_after_training", True):
        _run_validation(config, checkpoint, training_seconds)


if __name__ == "__main__":
    main()
