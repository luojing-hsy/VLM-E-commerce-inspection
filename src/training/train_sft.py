from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from src.data.prepare_dataset import prepare_stage
from src.evaluation.evaluate_direct import run as run_evaluation
from src.training.runtime import (
    apply_resume_options,
    assert_verl_sft_config,
    build_verl_sft_command,
    launch_verl_sft,
    validate_stage_config,
    write_run_manifest,
)


def _run_validation(config: dict, checkpoint: Path, training_seconds: float) -> None:
    predictions = Path(config.get("validation_predictions", "outputs/sft/validation_predictions.jsonl"))
    metrics = Path(config.get("validation_metrics", "outputs/sft/validation_metrics.json"))
    manifest = Path(config.get("validation_manifest", "outputs/sft/validation_manifest.jsonl"))
    if predictions.exists():
        predictions.unlink()
    args = argparse.Namespace(
        config=config.get("validation_eval_config", "configs/eval.yaml"),
        dataset=config["validation_source_dataset"],
        model=str(checkpoint),
        predictions=str(predictions),
        metrics=str(metrics),
        batch_size=int(config.get("validation_batch_size", 8)),
        limit=config.get("post_training_validation_max_samples"),
        split="validation",
        mode="sft_validation",
        manifest=str(manifest),
        min_pixels=config.get("min_pixels"),
        max_pixels=config.get("max_pixels"),
    )
    report = run_evaluation(args)
    report["checkpoint"] = str(checkpoint)
    report["training_seconds"] = round(training_seconds, 3)
    metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SFT validation report: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Qwen3.5 BF16 LoRA + MM projector SFT with veRL")
    parser.add_argument("--config", default="configs/sft.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="validate JSONL inputs and write run metadata")
    mode.add_argument("--print-command", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-from", help="resume from an explicit veRL global_step checkpoint directory")
    resume.add_argument("--restart", action="store_true", help="disable automatic resume and start a new run")
    args = parser.parse_args()
    prepare_stage(args.config, "sft")
    config = validate_stage_config(args.config, "sft")
    apply_resume_options(config, args.resume_from, args.restart)
    assert_verl_sft_config(config)
    manifest = write_run_manifest(args.config, config)
    print(f"SFT run inputs validated; metadata: {manifest}")
    if args.prepare_only:
        return
    if args.print_command:
        print(subprocess.list2cmdline(build_verl_sft_command(config)))
        return
    started = time.perf_counter()
    checkpoint = launch_verl_sft(config)
    training_seconds = time.perf_counter() - started
    print(f"latest SFT Hugging Face checkpoint: {checkpoint}")
    if config.get("run_validation_after_training", True):
        _run_validation(config, checkpoint, training_seconds)


if __name__ == "__main__":
    main()
