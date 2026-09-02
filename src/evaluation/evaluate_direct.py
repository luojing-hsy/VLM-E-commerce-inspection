from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.common import load_yaml, read_jsonl
from src.evaluation.metrics import classification_metrics, perception_metrics
from src.models.audit_protocol import (
    IMAGE_REFS,
    PROMPT,
    assert_model_text_is_sanitized,
    product_prompt,
    structured_prompt,
    target_from_sample,
)
from src.models.hf_loader import load_multimodal_components
from src.rewards.parser import tolerant_parse


def _image_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images")
    if isinstance(images, list):
        paths = images
    elif isinstance(images, dict):
        main = images.get("main")
        details = images.get("detail")
        main_path = main.get("image_id") if isinstance(main, dict) else None
        paths = [
            main_path,
            *(item.get("image_id") if isinstance(item, dict) else None for item in details)
        ] if isinstance(details, list) else []
    else:
        paths = []
    if len(paths) != 3 or not all(isinstance(path, str) and Path(path).is_file() for path in paths):
        raise FileNotFoundError(
            f"expected three existing evaluation images: {row.get('sample_id') or row.get('product_id')} -> {paths}"
        )
    return [str(path) for path in paths]


def normalize_sample(row: dict[str, Any], split: str = "test") -> dict[str, Any]:
    image_paths = _image_paths(row)
    violation_type = row.get("violation_type")
    evidence = row.get("evidence")
    target_image_ref = row.get("target_image_ref")
    if violation_type in {"image_quality", "wrong_image"}:
        if target_image_ref not in IMAGE_REFS:
            target_image_ref = evidence if evidence in IMAGE_REFS else None
        if target_image_ref not in IMAGE_REFS:
            raise ValueError(
                f"{violation_type} requires target_image_ref: {row.get('sample_id') or row.get('product_id')}"
            )
        evidence = target_image_ref
    else:
        evidence = None
        target_image_ref = row.get("target_image_ref")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        product_id = row.get("product_id")
        if not isinstance(product_id, str) or not product_id:
            raise ValueError("evaluation row requires sample_id or product_id")
        sample_id = f"{row.get('dataset', 'test')}_{product_id}"
    sample = {
        "sample_id": sample_id,
        "images": image_paths,
        "split": split,
        "decision": row.get("decision") or ("pass" if violation_type == "pass" else "reject"),
        "violation_type": violation_type,
        "issue_subtype": row.get("issue_subtype") if violation_type == "image_quality" else None,
        "evidence": evidence,
        "target_image_ref": target_image_ref,
        "title": row.get("title", ""),
        "category": row.get("category"),
        "color": row.get("color"),
        "material": row.get("material"),
        "difficulty": row.get("difficulty"),
    }
    target_from_sample(sample)
    return sample


def _direct_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    text = product_prompt(
        sample["title"],
        sample["category"],
        sample["color"],
        sample["material"],
        image_placeholders=False,
    )
    messages = structured_prompt(sample["images"], text)
    assert_model_text_is_sanitized(messages)
    return messages


def _prediction_map(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    predictions: dict[str, object] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in predictions:
            raise ValueError(f"invalid or duplicate prediction sample_id: {sample_id}")
        predictions[sample_id] = row.get("prediction")
    return predictions


def _append_prediction(path: Path, sample_id: str, prediction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps({"sample_id": sample_id, "prediction": prediction}, ensure_ascii=False) + "\n"
        )
        handle.flush()


def _report(
    samples: list[dict[str, Any]],
    raw_predictions: dict[str, object],
    config: dict[str, Any],
    mode: str = "base_model_test",
) -> dict[str, Any]:
    parsed: dict[str, dict[str, Any]] = {}
    protocol_valid = 0
    per_class: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "correct": 0}
    )
    for sample in samples:
        result = tolerant_parse(raw_predictions.get(sample["sample_id"], ""))
        prediction = result.data if result.protocol_valid else {}
        parsed[sample["sample_id"]] = prediction
        protocol_valid += result.protocol_valid
        label = sample["violation_type"]
        per_class[label]["count"] += 1
        if result.protocol_valid and prediction == target_from_sample(sample):
            per_class[label]["correct"] += 1
    total = len(samples)
    report = {
        **classification_metrics(samples, parsed, config),
        **perception_metrics(samples, parsed),
        "parse_rate": protocol_valid / total if total else 0.0,
        "exact_protocol_accuracy": (
            sum(item["correct"] for item in per_class.values()) / total if total else 0.0
        ),
        "per_class_exact_accuracy": {
            label: {**values, "accuracy": values["correct"] / values["count"]}
            for label, values in sorted(per_class.items())
        },
        "label_counts": dict(sorted(Counter(sample["violation_type"] for sample in samples).items())),
        "mode": mode,
    }
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    split = getattr(args, "split", "test")
    mode = getattr(args, "mode", "base_model_test")
    if getattr(args, "min_pixels", None) is not None:
        config["min_pixels"] = args.min_pixels
    if getattr(args, "max_pixels", None) is not None:
        config["max_pixels"] = args.max_pixels

    rows = read_jsonl(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    samples = [normalize_sample(row, split) for row in rows]
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("evaluation dataset contains duplicate sample_id values")

    predictions_path = Path(args.predictions)
    raw_predictions = _prediction_map(predictions_path)
    pending = [sample for sample in samples if sample["sample_id"] not in raw_predictions]
    inference_seconds = None
    peak_allocated_gib = None
    processor_load_seconds = None
    model_load_seconds = None
    model_type = None
    gated_deltanet_backend = None

    if pending:
        import torch

        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        components = load_multimodal_components(
            args.model,
            dtype=torch.bfloat16,
            processor_kwargs={
                "min_pixels": int(config.get("min_pixels", 784)),
                "max_pixels": int(config.get("max_pixels", 50176)),
            },
            require_fast_kernels=bool(config.get("require_gated_deltanet_kernels", True)),
            use_hub_kernels=bool(config.get("use_hub_kernels", False)),
        )
        processor = components.processor
        model = components.model
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "left"
        processor_load_seconds = components.processor_load_seconds
        model_load_seconds = components.model_load_seconds
        model_type = components.model_type
        gated_deltanet_backend = components.gated_deltanet_backend
        print(
            f"loaded {model_type}: processor={processor_load_seconds:.3f}s "
            f"model={model_load_seconds:.3f}s gated_deltanet={gated_deltanet_backend}",
            flush=True,
        )
        model.eval()
        use_cuda = bool(torch.cuda.is_available())
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            prompts: list[str] = []
            images: list[Image.Image] = []
            for sample in batch:
                messages = _direct_messages(sample)
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if PROMPT not in prompt or any(path in prompt for path in sample["images"]):
                    raise ValueError("processor exposed an image path or dropped the canonical prompt")
                prompts.append(prompt)
                for image_path in sample["images"]:
                    with Image.open(image_path) as source:
                        images.append(source.convert("RGB"))
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            for image in images:
                image.close()
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(config.get("max_new_tokens", 256)),
                )
            completion = generated[:, inputs["input_ids"].shape[1] :]
            predictions = processor.batch_decode(
                completion,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for sample, prediction in zip(batch, predictions):
                prediction = prediction.strip()
                _append_prediction(predictions_path, sample["sample_id"], prediction)
                raw_predictions[sample["sample_id"]] = prediction
            elapsed = time.perf_counter() - started
            completed = batch_start + len(batch)
            peak_gib = (
                torch.cuda.max_memory_allocated() / 1024**3
                if use_cuda
                else None
            )
            print(
                f"[{completed}/{len(pending)}] batch={len(batch)} elapsed={elapsed:.1f}s "
                f"peak_allocated={peak_gib:.1f}GiB" if peak_gib is not None
                else f"[{completed}/{len(pending)}] batch={len(batch)} elapsed={elapsed:.1f}s",
                flush=True,
            )
        inference_seconds = time.perf_counter() - started
        if use_cuda:
            peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3

    report = _report(samples, raw_predictions, config, mode)
    report["dataset"] = str(args.dataset)
    report["model"] = str(args.model)
    report["split"] = split
    report["inference_seconds"] = (
        round(inference_seconds, 3) if inference_seconds is not None else None
    )
    report["seconds_per_sample"] = (
        round(inference_seconds / len(pending), 3)
        if inference_seconds is not None and pending
        else None
    )
    report["peak_allocated_gib"] = (
        round(peak_allocated_gib, 3) if peak_allocated_gib is not None else None
    )
    report["processor_load_seconds"] = (
        round(processor_load_seconds, 3) if processor_load_seconds is not None else None
    )
    report["model_load_seconds"] = (
        round(model_load_seconds, 3) if model_load_seconds is not None else None
    )
    report["model_type"] = model_type
    report["gated_deltanet_backend"] = gated_deltanet_backend

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
                for sample in samples
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen multimodal checkpoint on direct product images")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--dataset", default="data/test/test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--predictions", default="outputs/test/predictions.jsonl")
    parser.add_argument("--metrics", default="outputs/test/metrics.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", default="base_model_test")
    parser.add_argument("--manifest")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

