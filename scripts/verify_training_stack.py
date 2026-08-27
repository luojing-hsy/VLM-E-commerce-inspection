#!/usr/bin/env python3
"""Verify the direct package pins used by scripts/setup.sh."""

from __future__ import annotations

import importlib.metadata
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
    "torch": "2.9.0",
    "torchaudio": "2.9.0",
    "torchvision": "0.24.0",
    "transformers": "4.57.1",
    "verl": "0.8.0",
    "vllm": "0.12.0",
}


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"expected Python 3.12, got {sys.version.split()[0]}")

    mismatches = []
    for distribution, expected in EXPECTED.items():
        actual = importlib.metadata.version(distribution)
        if actual.split("+", maxsplit=1)[0] != expected:
            mismatches.append(f"{distribution}: expected {expected}, got {actual}")
    if mismatches:
        raise SystemExit("training stack version mismatch:\n" + "\n".join(mismatches))

    import torch

    if torch.version.cuda != "12.8":
        raise SystemExit(f"expected a CUDA 12.8 PyTorch build, got CUDA {torch.version.cuda}")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401

    print("verified fixed SFT/veRL/vLLM package versions")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")


if __name__ == "__main__":
    main()
