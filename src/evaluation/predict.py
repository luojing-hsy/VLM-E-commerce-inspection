from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.common import load_yaml, read_jsonl, write_jsonl
from src.models.audit_protocol import PROMPT, assert_model_text_is_sanitized, product_prompt, structured_prompt


def _load_samples(config: dict) -> list[dict]:
    paths = [Path(config["manifest"]), Path(config["counterfactual_manifest"])]
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            images = row.get("images")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"invalid or duplicate evaluation sample_id: {sample_id}")
            if not isinstance(images, list) or len(images) != 3:
                raise ValueError(f"evaluation row must contain main and two detail images: {sample_id}")
            missing = [path for path in images if not isinstance(path, str) or not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"evaluation image does not exist: {missing[0]}")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError("evaluation manifests contain no samples")
    return rows


def predict(config: dict, model_name_or_path: str, output_path: str | Path) -> Path:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("torch and transformers with Qwen3-VL support are required for prediction") from exc

    processor_kwargs = {}
    if config.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(config["min_pixels"])
    if config.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(config["max_pixels"])
    processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    predictions = []
    for sample in _load_samples(config):
        image_paths = sample["images"]
        text = product_prompt(
            sample.get("title"),
            sample.get("category"),
            sample.get("color"),
            sample.get("material"),
            image_placeholders=False,
        )
        messages = structured_prompt(image_paths, text)
        assert_model_text_is_sanitized(messages)
        rendered_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if PROMPT not in rendered_prompt or any(path in rendered_prompt for path in image_paths):
            raise ValueError("processor exposed an image path or dropped the canonical audit prompt")
        with Image.open(image_paths[0]) as main, Image.open(image_paths[1]) as detail1, Image.open(image_paths[2]) as detail2:
            images = [source.convert("RGB") for source in (main, detail1, detail2)]
            inputs = processor(text=[rendered_prompt], images=images, return_tensors="pt")
            for image in images:
                image.close()
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(config.get("max_new_tokens", 256)),
            )
        completion_ids = generated[:, inputs["input_ids"].shape[1] :]
        prediction = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        predictions.append({"sample_id": sample["sample_id"], "prediction": prediction})

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(target, predictions)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Base/SFT/Joint checkpoints with the fixed audit prompt")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"wrote predictions: {predict(load_yaml(args.config), args.model, args.output)}")


if __name__ == "__main__":
    main()
