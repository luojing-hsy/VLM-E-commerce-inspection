from __future__ import annotations

import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}


@dataclass(frozen=True)
class MultimodalComponents:
    processor: Any
    model: Any
    model_type: str
    processor_load_seconds: float
    model_load_seconds: float
    gated_deltanet_backend: str


def gated_deltanet_backend(model_type: str) -> str:
    if model_type not in QWEN35_MODEL_TYPES:
        return "not_applicable"
    if find_spec("fla") is not None and find_spec("causal_conv1d") is not None:
        return "fla+causal_conv1d"
    return "pytorch_fallback"


def require_fast_gated_deltanet_kernels(model_type: str) -> None:
    if model_type not in QWEN35_MODEL_TYPES:
        return
    missing = [
        package
        for package in ("fla", "causal_conv1d")
        if find_spec(package) is None
    ]
    if missing:
        raise RuntimeError(
            "Qwen3.5 fast Gated DeltaNet kernels are required but missing: "
            + ", ".join(missing)
            + ". Run bash scripts/setup.sh; use --allow-slow-kernels only for diagnostics."
        )


def load_multimodal_model(
    model_name_or_path: str,
    *,
    dtype: Any,
    device_map: str = "auto",
    use_hub_kernels: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if use_hub_kernels:
        kwargs.update(use_kernels=True, trust_remote_code=True)
    return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)


def load_multimodal_components(
    model_name_or_path: str,
    *,
    dtype: Any,
    processor_kwargs: dict[str, Any] | None = None,
    require_fast_kernels: bool = True,
    use_hub_kernels: bool = False,
) -> MultimodalComponents:
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = str(config.model_type)
    if require_fast_kernels:
        require_fast_gated_deltanet_kernels(model_type)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_name_or_path, **(processor_kwargs or {}))
    processor_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = load_multimodal_model(
        model_name_or_path,
        dtype=dtype,
        use_hub_kernels=use_hub_kernels,
    )
    model_seconds = time.perf_counter() - started
    return MultimodalComponents(
        processor=processor,
        model=model,
        model_type=model_type,
        processor_load_seconds=processor_seconds,
        model_load_seconds=model_seconds,
        gated_deltanet_backend=gated_deltanet_backend(model_type),
    )
