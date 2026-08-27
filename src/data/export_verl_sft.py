from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import read_jsonl, sha256_file
from src.models.audit_protocol import validate_prediction_dict, validate_product_prompt


def build_rows(jsonl_path: str | Path) -> list[dict]:
    rows = []
    for row in read_jsonl(jsonl_path):
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"invalid SFT conversation: {row.get('sample_id')}")
        user_message, assistant_message = conversations
        if user_message.get("from") != "human":
            raise ValueError(f"SFT user prompt is not canonical: {row.get('sample_id')}")
        validate_product_prompt(user_message.get("value"), image_placeholders=3)
        if assistant_message.get("from") != "gpt" or not isinstance(assistant_message.get("value"), str):
            raise ValueError(f"invalid SFT assistant message: {row.get('sample_id')}")
        try:
            target = validate_prediction_dict(json.loads(assistant_message["value"]))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid SFT target: {row.get('sample_id')}") from exc
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"SFT row must contain main and two detail images: {row.get('sample_id')}")
        missing = [path for path in images if not isinstance(path, str) or not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"SFT image does not exist: {missing[0]}")
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user_message["value"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                "images": images,
            }
        )
    if not rows:
        raise ValueError(f"SFT export is empty: {jsonl_path}")
    return rows


def write_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> Path:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets/pyarrow are required to write veRL SFT Parquet") from exc
    target = Path(parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(build_rows(jsonl_path)).to_parquet(str(target))
    return target


def write_parquet_if_needed(jsonl_path: str | Path, parquet_path: str | Path) -> Path:
    source = Path(jsonl_path)
    target = Path(parquet_path)
    stamp = Path(str(target) + ".source_sha256")
    source_hash = sha256_file(source)
    if target.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == source_hash:
        print(f"reusing current SFT parquet: {target}")
        return target

    result = write_parquet(source, target)
    stamp.write_text(source_hash + "\n", encoding="utf-8")
    print(f"wrote refreshed SFT parquet: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SFT JSONL to veRL MultiTurnSFTDataset Parquet")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"wrote veRL SFT parquet: {write_parquet(args.input, args.output)}")


if __name__ == "__main__":
    main()
