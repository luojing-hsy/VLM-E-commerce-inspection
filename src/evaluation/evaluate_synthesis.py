from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.common import load_yaml, read_jsonl
from src.evaluation.metrics import classification_metrics, perception_metrics
from src.models.audit_protocol import PROMPT, product_prompt, structured_prompt, target_from_sample
from src.models.hf_loader import load_multimodal_components
from src.rewards.parser import tolerant_parse


IMAGE_REFS = ("main", "detail:1", "detail:2")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_image(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        lines.append(current.rstrip())
        current = character.lstrip()
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current.rstrip())
    truncated = "".join(lines) != text
    if truncated and lines:
        suffix = "..."
        last = lines[-1]
        while last and draw.textlength(last + suffix, font=font) > max_width:
            last = last[:-1]
        lines[-1] = last.rstrip() + suffix
    return lines


def _image_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"missing images object: {row.get('product_id')}")
    main = images.get("main", {}).get("image_id")
    details = images.get("detail")
    if not isinstance(main, str) or not isinstance(details, list) or len(details) != 2:
        raise ValueError(f"expected one main and two detail images: {row.get('product_id')}")
    paths = [main, *(item.get("image_id") for item in details)]
    if not all(isinstance(path, str) and Path(path).is_file() for path in paths):
        raise FileNotFoundError(f"missing synthesis image: {row.get('product_id')} -> {paths}")
    return paths


def render_page(row: dict[str, Any], target: Path, width: int = 960, height: int = 720) -> Path:
    paths = _image_paths(row)
    canvas = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    field_font = _font(18)
    label_font = _font(17)

    draw.rounded_rectangle((20, 18, width - 20, 200), radius=16, fill="white", outline="#cbd5e1", width=2)
    title_lines = _wrapped_lines(draw, row.get("title", ""), title_font, width - 70, 2)
    y = 32
    for line in title_lines:
        draw.text((35, y), line, fill="#111827", font=title_font)
        y += 29
    fields = (
        ("category", row.get("category")),
        ("color", row.get("color")),
        ("material", row.get("material")),
    )
    y = 103
    for name, value in fields:
        display = "N/A" if value is None else str(value)
        line = f"{name}: {display}"
        line = _wrapped_lines(draw, line, field_font, width - 70, 1)[0]
        draw.text((35, y), line, fill="#334155", font=field_font)
        y += 29

    gap = 18
    left = 24
    panel_top = 238
    panel_bottom = height - 24
    panel_width = (width - 2 * left - 2 * gap) // 3
    for index, (role, path) in enumerate(zip(IMAGE_REFS, paths)):
        x1 = left + index * (panel_width + gap)
        x2 = x1 + panel_width
        draw.text((x1 + 8, panel_top - 28), role, fill="#334155", font=label_font)
        draw.rounded_rectangle((x1, panel_top, x2, panel_bottom), radius=12, fill="white", outline="#cbd5e1", width=2)
        image = _fit_image(path, (panel_width - 16, panel_bottom - panel_top - 16))
        x = x1 + (panel_width - image.width) // 2
        y = panel_top + (panel_bottom - panel_top - image.height) // 2
        canvas.paste(image, (x, y))

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return target


def _normalized_sample(row: dict[str, Any], split: str = "test") -> dict[str, Any]:
    violation_type = row.get("violation_type")
    evidence = None
    image_paths = _image_paths(row)
    if violation_type in {"image_quality", "wrong_image"}:
        image_ref = row.get("target_image_ref")
        if image_ref not in IMAGE_REFS:
            raise ValueError(f"{violation_type} requires target_image_ref: {row.get('product_id')}")
        evidence = image_ref
    sample = {
        "sample_id": f"{row.get('dataset', 'test')}_{row['product_id']}",
        "images": image_paths,
        "split": split,
        "decision": "pass" if violation_type == "pass" else "reject",
        "violation_type": violation_type,
        "issue_subtype": row.get("issue_subtype") if violation_type == "image_quality" else None,
        "evidence": evidence,
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
    return structured_prompt(sample["images"], text)


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
        handle.write(json.dumps({"sample_id": sample_id, "prediction": prediction}, ensure_ascii=False) + "\n")
        handle.flush()


def _report(
    samples: list[dict],
    raw_predictions: dict[str, object],
    config: dict,
    mode: str = "base_model_test",
) -> dict:
    parsed: dict[str, dict] = {}
    protocol_valid = 0
    per_class = defaultdict(lambda: {"count": 0, "correct": 0})
    for sample in samples:
        result = tolerant_parse(raw_predictions.get(sample["sample_id"], ""))
        prediction = result.data if result.protocol_valid else {}
        parsed[sample["sample_id"]] = prediction
        protocol_valid += result.protocol_valid
        label = sample["violation_type"]
        per_class[label]["count"] += 1
        if result.protocol_valid and prediction == target_from_sample(sample):
            per_class[label]["correct"] += 1
    report = {
        **classification_metrics(samples, parsed, config),
        **perception_metrics(samples, parsed),
        "parse_rate": protocol_valid / len(samples) if samples else 0.0,
        "exact_protocol_accuracy": sum(item["correct"] for item in per_class.values()) / len(samples),
        "per_class_exact_accuracy": {
            label: {**values, "accuracy": values["correct"] / values["count"]}
            for label, values in sorted(per_class.items())
        },
        "label_counts": dict(sorted(Counter(sample["violation_type"] for sample in samples).items())),
        "mode": mode,
    }
    return report


def run(args: argparse.Namespace) -> dict:
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
    samples = [_normalized_sample(row, split) for row in rows]

    raw_predictions = _prediction_map(Path(args.predictions))
    pending = [sample for sample in samples if sample["sample_id"] not in raw_predictions]
    inference_seconds = None
    peak_allocated_gib = None
    processor_load_seconds = None
    model_load_seconds = None
    model_type = None
    gated_deltanet_backend = None
    if pending:
        import torch

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
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            prompts = []
            images = []
            for sample in batch:
                messages = _direct_messages(sample)
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if PROMPT not in prompt or any(path in prompt for path in sample["images"]):
                    raise ValueError("processor exposed a path or dropped the canonical prompt")
                prompts.append(prompt)
                for image_path in sample["images"]:
                    with Image.open(image_path) as source:
                        images.append(source.convert("RGB"))
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
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
                _append_prediction(Path(args.predictions), sample["sample_id"], prediction)
                raw_predictions[sample["sample_id"]] = prediction
            elapsed = time.perf_counter() - started
            completed = batch_start + len(batch)
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"[{completed}/{len(pending)}] batch={len(batch)} elapsed={elapsed:.1f}s peak_allocated={peak_gib:.1f}GiB",
                flush=True,
            )
        inference_seconds = time.perf_counter() - started
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3

    report = _report(samples, raw_predictions, config, mode)
    report["dataset"] = str(args.dataset)
    report["model"] = str(args.model)
    report["split"] = split
    report["inference_seconds"] = round(inference_seconds, 3) if inference_seconds is not None else None
    report["seconds_per_sample"] = round(inference_seconds / len(pending), 3) if inference_seconds is not None else None
    report["peak_allocated_gib"] = round(peak_allocated_gib, 3) if peak_allocated_gib is not None else None
    report["processor_load_seconds"] = round(processor_load_seconds, 3) if processor_load_seconds is not None else None
    report["model_load_seconds"] = round(model_load_seconds, 3) if model_load_seconds is not None else None
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
    parser = argparse.ArgumentParser(description="Evaluate a multimodal Qwen checkpoint on product-audit JSONL")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--dataset", default="data/test/test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pages", default="data/test")
    parser.add_argument("--predictions", default="outputs/baseline/test_predictions.jsonl")
    parser.add_argument("--metrics", default="outputs/baseline/test_metrics.json")
    parser.add_argument("--page-width", type=int, default=960)
    parser.add_argument("--page-height", type=int, default=720)
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
