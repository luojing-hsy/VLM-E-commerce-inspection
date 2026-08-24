#!/usr/bin/env python3
"""Verify the direct package pins used by scripts/setup.sh."""

from __future__ import annotations

import importlib.metadata
import json
import sys


EXPECTED = {
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "numpy": "2.2.6",
    "peft": "0.20.0",
    "Pillow": "12.3.0",
    "pydantic": "2.13.4",
    "PyYAML": "6.0.3",
    "qwen-vl-utils": "0.0.14",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.15.0.dev0",
    "verl": "0.8.0",
    "vllm": "0.25.1",
}
TRANSFORMERS_COMMIT = "7ea2320c76117e6742364808a666ef6f2fb40a67"


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"expected Python 3.12, got {sys.version.split()[0]}")

    mismatches = []
    for distribution, expected in EXPECTED.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            mismatches.append(f"{distribution}: expected {expected}, got {actual}")
    if mismatches:
        raise SystemExit("training stack version mismatch:\n" + "\n".join(mismatches))

    direct_url_text = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if direct_url_text is None:
        raise SystemExit("transformers direct_url.json is missing; cannot verify the pinned commit")
    direct_url = json.loads(direct_url_text)
    actual_commit = direct_url.get("vcs_info", {}).get("commit_id")
    if actual_commit != TRANSFORMERS_COMMIT:
        raise SystemExit(
            f"transformers commit mismatch: expected {TRANSFORMERS_COMMIT}, got {actual_commit}"
        )

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401

    print("verified fixed SFT/veRL/vLLM package versions")
    print(f"transformers commit: {actual_commit}")


if __name__ == "__main__":
    main()
