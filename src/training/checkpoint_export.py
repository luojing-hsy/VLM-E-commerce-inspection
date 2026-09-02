from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


_PEFT_PREFIX = "base_model.model."
_LORA_A_MARKER = ".lora_A."
_LORA_B_MARKER = ".lora_B."
_BASE_LAYER_MARKER = ".base_layer."


def _lora_target(key: str, marker: str) -> tuple[str, str]:
    prefix, adapter_and_suffix = key.split(marker, maxsplit=1)
    if not adapter_and_suffix.endswith(".weight"):
        raise ValueError(f"unsupported LoRA key: {key}")
    adapter = adapter_and_suffix[: -len(".weight")]
    if not adapter:
        raise ValueError(f"missing LoRA adapter name: {key}")
    return f"{prefix}.weight", adapter


def merge_peft_state_dict(
    state_dict: dict[str, Any], *, lora_rank: int, lora_alpha: int
) -> dict[str, Any]:
    """Convert veRL's full PeftModel state dict into ordinary HF model keys."""
    if not any(
        _LORA_A_MARKER in key or _LORA_B_MARKER in key or _BASE_LAYER_MARKER in key
        for key in state_dict
    ):
        return dict(state_dict)
    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")

    base_layers: dict[str, Any] = {}
    lora_a: dict[tuple[str, str], Any] = {}
    lora_b: dict[tuple[str, str], Any] = {}
    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not key.startswith(_PEFT_PREFIX):
            raise ValueError(f"mixed PEFT and ordinary state_dict keys: {key}")
        ordinary_key = key[len(_PEFT_PREFIX) :]
        if _LORA_A_MARKER in ordinary_key:
            target, adapter = _lora_target(ordinary_key, _LORA_A_MARKER)
            lora_a[(target, adapter)] = value
        elif _LORA_B_MARKER in ordinary_key:
            target, adapter = _lora_target(ordinary_key, _LORA_B_MARKER)
            lora_b[(target, adapter)] = value
        elif _BASE_LAYER_MARKER in ordinary_key:
            target = ordinary_key.replace(_BASE_LAYER_MARKER, ".", 1)
            base_layers[target] = value
        else:
            converted[ordinary_key] = value

    import torch

    scaling = float(lora_alpha) / float(lora_rank)
    for target, base in base_layers.items():
        adapters = {adapter for key, adapter in lora_a if key == target}
        adapters.update(adapter for key, adapter in lora_b if key == target)
        if not adapters:
            converted[target] = base
            continue
        if adapters != {"default"}:
            raise ValueError(f"only the default LoRA adapter is supported: {target} -> {adapters}")
        a = lora_a.get((target, "default"))
        b = lora_b.get((target, "default"))
        if a is None or b is None:
            raise ValueError(f"incomplete LoRA pair for {target}")
        if base.ndim != 2 or a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"only 2-D LoRA weights are supported: {target}")
        delta = torch.matmul(b.float(), a.float()).to(dtype=base.dtype) * scaling
        converted[target] = base + delta

    dangling = (set(lora_a) | set(lora_b)) - {(target, "default") for target in base_layers}
    if dangling:
        raise ValueError(f"LoRA weights have no matching base layer: {sorted(dangling)[:3]}")
    return converted


def _copy_non_weight_files(source: Path, target: Path) -> None:
    for path in source.iterdir():
        if path.name == "model.safetensors.index.json" or path.name.endswith(".safetensors"):
            continue
        destination = target / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)


def _convert_sharded_checkpoint(source: Path, target: Path, lora_rank: int, lora_alpha: int) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        keys_by_shard[shard].append(key)

    lora_values: dict[tuple[str, str, str], Any] = {}
    for shard, keys in keys_by_shard.items():
        selected = [key for key in keys if _LORA_A_MARKER in key or _LORA_B_MARKER in key]
        if not selected:
            continue
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            for key in selected:
                lora_values[(shard, key, "value")] = handle.get_tensor(key)

    converted_index: dict[str, str] = {}
    total_parameters = 0
    total_size = 0
    for shard, keys in keys_by_shard.items():
        raw: dict[str, Any] = {}
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                if _LORA_A_MARKER in key or _LORA_B_MARKER in key:
                    raw[key] = lora_values[(shard, key, "value")]
                else:
                    raw[key] = handle.get_tensor(key)
        converted = merge_peft_state_dict(raw, lora_rank=lora_rank, lora_alpha=lora_alpha)
        output_shard = target / shard
        save_file(
            {key: value.contiguous() for key, value in converted.items()},
            str(output_shard),
            metadata={"format": "pt"},
        )
        for key, value in converted.items():
            converted_index[key] = shard
            total_parameters += value.numel()
            total_size += value.numel() * value.element_size()

    output_index = {
        "metadata": {"total_parameters": total_parameters, "total_size": total_size},
        "weight_map": converted_index,
    }
    (target / "model.safetensors.index.json").write_text(
        json.dumps(output_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _copy_non_weight_files(source, target)


def normalize_hf_checkpoint(
    checkpoint_dir: str | Path, *, keep_raw_backup: bool = False
) -> Path:
    """Make a veRL LoRA HF export loadable as a normal merged HF model."""
    source = Path(checkpoint_dir)
    backup = source.with_name(source.name + ".peft_raw")
    if not keep_raw_backup and backup.exists():
        shutil.rmtree(backup)
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        return source
    index = json.loads(index_path.read_text(encoding="utf-8"))
    keys = list(index.get("weight_map", {}))
    if not any(key.startswith(_PEFT_PREFIX) for key in keys):
        return source

    meta_path = source.parent / "lora_train_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"LoRA metadata is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lora_rank = int(meta["r"])
    lora_alpha = int(meta["lora_alpha"])
    if keep_raw_backup and backup.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint backup: {backup}")

    replaced = source.with_name(source.name + ".replace_tmp")
    if replaced.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint swap directory: {replaced}")
    temporary = Path(tempfile.mkdtemp(prefix=f"{source.name}.standard.", dir=source.parent))
    try:
        _convert_sharded_checkpoint(source, temporary, lora_rank, lora_alpha)
        if keep_raw_backup:
            source.rename(backup)
            temporary.rename(source)
        else:
            source.rename(replaced)
            try:
                temporary.rename(source)
            except Exception:
                replaced.rename(source)
                raise
            shutil.rmtree(replaced)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return source
