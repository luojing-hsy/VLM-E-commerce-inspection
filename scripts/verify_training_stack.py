#!/usr/bin/env python3
"""Verify the direct package pins used by scripts/setup.sh."""

from __future__ import annotations

import importlib.metadata
import sys


EXPECTED = {
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "causal-conv1d": "1.7.0",
    "numpy": "2.2.6",
    "flash-linear-attention": "0.4.2",
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
    "transformers": "5.16.1",
    "verl": "0.8.0",
    "vllm": "0.20.2",
}


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"expected Python 3.12, got {sys.version.split()[0]}")

    mismatches = []
    for distribution, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{distribution}: expected {expected}, got <missing>")
            continue
        if actual.split("+", maxsplit=1)[0] != expected:
            mismatches.append(f"{distribution}: expected {expected}, got {actual}")
    if mismatches:
        raise SystemExit("training stack version mismatch:\n" + "\n".join(mismatches))

    import torch

    if torch.version.cuda != "12.9":
        raise SystemExit(f"expected a CUDA 12.9 PyTorch build, got CUDA {torch.version.cuda}")

    import causal_conv1d  # noqa: F401
    import fla  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForImageTextToText,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
    )

    print("verified fixed Qwen3.5 SFT/veRL/vLLM package and Gated DeltaNet kernel versions")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")


if __name__ == "__main__":
    main()
