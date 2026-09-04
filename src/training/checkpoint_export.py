from __future__ import annotations

import json
import math
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


_PEFT_PREFIX = "base_model.model."
_LORA_A_MARKER = ".lora_A."
_LORA_B_MARKER = ".lora_B."
_BASE_LAYER_MARKER = ".base_layer."
_MAX_IN_MEMORY_TENSOR_BYTES = 128 * 1024 * 1024


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


def _checkpoint_index(source: Path) -> dict[str, Any] | None:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))

    # veRL can emit a complete PEFT state dict as one model.safetensors file.
    # Transformers does not need an index for a single file, but the exporter
    # needs the key map to detect and normalize PEFT names.
    single_file = source / "model.safetensors"
    if not single_file.is_file():
        return None
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to inspect a single-file HF checkpoint") from exc
    with safe_open(str(single_file), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if not keys:
        raise ValueError(f"single-file HF checkpoint has no tensors: {single_file}")
    return {
        "metadata": {"total_size": single_file.stat().st_size},
        "weight_map": {key: single_file.name for key in keys},
    }


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors file header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8 or header_length > 64 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path}")
        raw_header = stream.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    return header, 8 + header_length, file_size


def _safetensor_entry(
    info: tuple[dict[str, Any], int, int], key: str
) -> tuple[dict[str, Any], int]:
    header, data_offset, file_size = info
    entry = header.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"tensor is missing from safetensors shard: {key}")
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
    ):
        raise ValueError(f"invalid safetensors tensor entry: {key}")
    start, end = offsets
    if start < 0 or end < start or data_offset + end > file_size:
        raise ValueError(f"invalid safetensors tensor offsets: {key}")
    return entry, end - start


def _copy_safetensor_tensor(
    source_path: Path,
    target_path: Path,
    info: tuple[dict[str, Any], int, int],
    source_key: str,
    output_key: str,
) -> tuple[int, int]:
    entry, tensor_size = _safetensor_entry(info, source_key)
    header, data_offset, _ = info
    output_header = {
        "__metadata__": {"format": "pt"},
        output_key: {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [0, tensor_size],
        },
    }
    header_bytes = json.dumps(
        output_header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    source_start = data_offset + entry["data_offsets"][0]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source_stream, target_path.open("wb") as target_stream:
        target_stream.write(struct.pack("<Q", len(header_bytes)))
        target_stream.write(header_bytes)
        source_stream.seek(source_start)
        remaining = tensor_size
        while remaining:
            chunk = source_stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"unexpected end of safetensors tensor: {source_key}")
            target_stream.write(chunk)
            remaining -= len(chunk)
    return tensor_size, math.prod(entry["shape"])


class _StreamingShardWriter:
    def __init__(
        self, target: Path, *, max_in_memory_bytes: int = _MAX_IN_MEMORY_TENSOR_BYTES
    ) -> None:
        self.target = target
        self.max_in_memory_bytes = max_in_memory_bytes
        self._pending: dict[str, Any] = {}
        self._pending_bytes = 0
        self._parts: list[tuple[Path, list[str]]] = []
        self._seen: set[str] = set()
        self._total_parameters = 0
        self._total_size = 0

    def _reserve(self, key: str) -> None:
        if not key or key in self._seen:
            raise ValueError(f"duplicate converted tensor key: {key}")
        self._seen.add(key)

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        from safetensors.torch import save_file

        temporary = self.target / f".converted-{len(self._parts) + 1:05d}.safetensors"
        save_file(
            {key: value.contiguous() for key, value in self._pending.items()},
            str(temporary),
            metadata={"format": "pt"},
        )
        self._parts.append((temporary, list(self._pending)))
        self._pending.clear()
        self._pending_bytes = 0

    def add(self, key: str, value: Any) -> None:
        tensor_size = int(value.numel()) * int(value.element_size())
        if tensor_size > self.max_in_memory_bytes:
            raise ValueError(f"tensor is too large for in-memory conversion: {key}")
        self._reserve(key)
        if self._pending and self._pending_bytes + tensor_size > self.max_in_memory_bytes:
            self._flush_pending()
        tensor = value.contiguous()
        self._pending[key] = tensor
        self._pending_bytes += tensor_size
        self._total_parameters += int(tensor.numel())
        self._total_size += tensor_size

    def add_external(
        self,
        key: str,
        source_path: Path,
        info: tuple[dict[str, Any], int, int],
        source_key: str,
    ) -> None:
        self._reserve(key)
        self._flush_pending()
        temporary = self.target / f".converted-{len(self._parts) + 1:05d}.safetensors"
        tensor_size, parameters = _copy_safetensor_tensor(
            source_path, temporary, info, source_key, key
        )
        self._parts.append((temporary, [key]))
        self._total_parameters += parameters
        self._total_size += tensor_size

    def finish(self) -> None:
        self._flush_pending()
        if not self._parts:
            raise ValueError("checkpoint contains no tensors")
        weight_map: dict[str, str] = {}
        part_count = len(self._parts)
        for part_number, (temporary, keys) in enumerate(self._parts, start=1):
            if part_count == 1:
                final_name = "model.safetensors"
            else:
                final_name = f"model-{part_number:05d}-of-{part_count:05d}.safetensors"
            temporary.rename(self.target / final_name)
            for key in keys:
                weight_map[key] = final_name
        output_index = {
            "metadata": {
                "total_parameters": self._total_parameters,
                "total_size": self._total_size,
            },
            "weight_map": weight_map,
        }
        (self.target / "model.safetensors.index.json").write_text(
            json.dumps(output_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
def _convert_sharded_checkpoint(
    source: Path,
    target: Path,
    lora_rank: int,
    lora_alpha: int,
    *,
    index: dict[str, Any] | None = None,
) -> None:
    from safetensors import safe_open

    if index is None:
        index = _checkpoint_index(source)
    if index is None:
        raise ValueError(f"no safetensors weights found in checkpoint: {source}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index for checkpoint: {source}")
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise ValueError(f"invalid weight map entry in checkpoint: {source}")
        keys_by_shard[shard].append(key)

    source_infos: dict[str, tuple[dict[str, Any], int, int]] = {}
    for shard, keys in keys_by_shard.items():
        source_path = source / shard
        if not source_path.is_file():
            raise FileNotFoundError(f"safetensors shard is missing: {source_path}")
        info = _read_safetensors_header(source_path)
        source_infos[shard] = info
        for key in keys:
            _safetensor_entry(info, key)

    peft_markers = (_LORA_A_MARKER, _LORA_B_MARKER, _BASE_LAYER_MARKER)
    has_peft = any(any(marker in key for marker in peft_markers) for key in weight_map)
    if has_peft and any(not key.startswith(_PEFT_PREFIX) for key in weight_map):
        raise ValueError("mixed PEFT and ordinary state_dict keys in checkpoint")

    lora_a: dict[tuple[str, str], tuple[str, Any]] = {}
    lora_b: dict[tuple[str, str], tuple[str, Any]] = {}
    for shard, keys in keys_by_shard.items():
        selected = [key for key in keys if _LORA_A_MARKER in key or _LORA_B_MARKER in key]
        if not selected:
            continue
        source_path = source / shard
        info = source_infos[shard]
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            for key in selected:
                ordinary_key = key[len(_PEFT_PREFIX) :]
                marker = _LORA_A_MARKER if _LORA_A_MARKER in ordinary_key else _LORA_B_MARKER
                target_key, adapter = _lora_target(ordinary_key, marker)
                _, tensor_size = _safetensor_entry(info, key)
                if tensor_size > _MAX_IN_MEMORY_TENSOR_BYTES:
                    raise ValueError(f"LoRA tensor is too large for streaming conversion: {key}")
                values = lora_a if marker == _LORA_A_MARKER else lora_b
                slot = (target_key, adapter)
                if slot in values:
                    raise ValueError(f"duplicate LoRA tensor for {slot}")
                values[slot] = (key, handle.get_tensor(key))

    base_targets = {
        key[len(_PEFT_PREFIX) :].replace(_BASE_LAYER_MARKER, ".", 1)
        for key in weight_map
        if _BASE_LAYER_MARKER in key
    }
    dangling = {target for target, _ in (set(lora_a) | set(lora_b))} - base_targets
    if dangling:
        raise ValueError(f"LoRA weights have no matching base layer: {sorted(dangling)[:3]}")

    writer = _StreamingShardWriter(target)
    for shard, keys in keys_by_shard.items():
        source_path = source / shard
        info = source_infos[shard]
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            for key in keys:
                if _LORA_A_MARKER in key or _LORA_B_MARKER in key:
                    continue
                _, tensor_size = _safetensor_entry(info, key)
                if has_peft:
                    ordinary_key = key[len(_PEFT_PREFIX) :]
                    if _BASE_LAYER_MARKER in ordinary_key:
                        target_key = ordinary_key.replace(_BASE_LAYER_MARKER, ".", 1)
                        adapters = {
                            adapter for target, adapter in lora_a if target == target_key
                        }
                        adapters.update(
                            adapter for target, adapter in lora_b if target == target_key
                        )
                        if adapters:
                            if tensor_size > _MAX_IN_MEMORY_TENSOR_BYTES:
                                raise ValueError(
                                    f"LoRA target is too large for streaming conversion: {key}"
                                )
                            fragment: dict[str, Any] = {key: handle.get_tensor(key)}
                            for adapter in adapters:
                                a = lora_a.get((target_key, adapter))
                                b = lora_b.get((target_key, adapter))
                                if a is not None:
                                    fragment[a[0]] = a[1]
                                if b is not None:
                                    fragment[b[0]] = b[1]
                            converted = merge_peft_state_dict(
                                fragment, lora_rank=lora_rank, lora_alpha=lora_alpha
                            )
                            for output_key, value in converted.items():
                                writer.add(output_key, value)
                        elif tensor_size > _MAX_IN_MEMORY_TENSOR_BYTES:
                            writer.add_external(
                                target_key, source_path, info, key
                            )
                        else:
                            writer.add(target_key, handle.get_tensor(key))
                    elif tensor_size > _MAX_IN_MEMORY_TENSOR_BYTES:
                        writer.add_external(
                            ordinary_key, source_path, info, key
                        )
                    else:
                        writer.add(ordinary_key, handle.get_tensor(key))
                elif tensor_size > _MAX_IN_MEMORY_TENSOR_BYTES:
                    writer.add_external(key, source_path, info, key)
                else:
                    writer.add(key, handle.get_tensor(key))
    writer.finish()
    _copy_non_weight_files(source, target)

def normalize_hf_checkpoint(
    checkpoint_dir: str | Path, *, keep_raw_backup: bool = False
) -> Path:
    """Make a veRL LoRA HF export loadable as a normal merged HF model."""
    source = Path(checkpoint_dir)
    backup = source.with_name(source.name + ".peft_raw")
    if not keep_raw_backup and backup.exists():
        shutil.rmtree(backup)
    index = _checkpoint_index(source)
    if index is None:
        return source
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
        _convert_sharded_checkpoint(source, temporary, lora_rank, lora_alpha, index=index)
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
