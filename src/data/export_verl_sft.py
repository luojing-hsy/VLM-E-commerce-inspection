from __future__ import annotations

import argparse
from pathlib import Path

from src.common import read_jsonl


def build_rows(jsonl_path: str | Path) -> list[dict]:
    rows = []
    for row in read_jsonl(jsonl_path):
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"invalid SFT conversation: {row.get('sample_id')}")
        image_path = row.get("image")
        if not isinstance(image_path, str) or not Path(image_path).is_file():
            raise FileNotFoundError(f"SFT image does not exist: {image_path}")
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": conversations[0]["value"]},
                    {"role": "assistant", "content": conversations[1]["value"]},
                ],
                "images": [image_path],
                "sample_id": row["sample_id"],
                "dataset_stage": "sft",
                "split": row["split"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SFT JSONL to veRL MultiTurnSFTDataset Parquet")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"wrote veRL SFT parquet: {write_parquet(args.input, args.output)}")


if __name__ == "__main__":
    main()
