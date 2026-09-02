from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import shutil
from pathlib import Path

from src.training.checkpoint_export import normalize_hf_checkpoint


MARKER = "VLM_PRODUCT_AUDIT_SFT_LATEST_ONLY_V1"
BACKUP_SUFFIX = ".vlm-sft-latest-only.orig"
_CHECKPOINT_NAME = re.compile(r"global_step_\d+")


def _validated_paths(output_dir: str | Path, checkpoint_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir).resolve()
    checkpoint = Path(checkpoint_dir).resolve(strict=False)
    if checkpoint.parent != root or not _CHECKPOINT_NAME.fullmatch(checkpoint.name):
        raise ValueError(f"checkpoint must be a global_step directory directly under {root}: {checkpoint}")
    return root, checkpoint


def prepare_latest_checkpoint(output_dir: str | Path, checkpoint_dir: str | Path) -> Path:
    """Remove prior SFT checkpoints before writing the next one."""
    root, checkpoint = _validated_paths(output_dir, checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("global_step_*"):
        if not _CHECKPOINT_NAME.fullmatch(path.name):
            continue
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    alias = root / "latest"
    if alias.is_symlink():
        alias.unlink()
    elif alias.exists():
        raise FileExistsError(f"refusing to replace non-symlink SFT alias: {alias}")

    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.is_file() or tracker.is_symlink():
        tracker.unlink()
    elif tracker.exists():
        raise FileExistsError(f"refusing to replace non-file checkpoint tracker: {tracker}")
    return checkpoint


def finalize_latest_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Publish one normalized checkpoint through the outputs/sft_qwen35_4b/latest alias."""
    checkpoint = Path(checkpoint_dir).resolve()
    root, checkpoint = _validated_paths(checkpoint.parent, checkpoint)
    huggingface = normalize_hf_checkpoint(checkpoint / "huggingface", keep_raw_backup=False)

    alias = root / "latest"
    if alias.is_symlink():
        alias.unlink()
    elif alias.exists():
        raise FileExistsError(f"refusing to replace non-symlink SFT alias: {alias}")
    alias.symlink_to(checkpoint.name, target_is_directory=True)
    return huggingface


def _handler_path() -> Path:
    if importlib.metadata.version("verl") != "0.8.0":
        raise RuntimeError("SFT latest-only checkpoint hook is pinned to verl==0.8.0")
    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate installed verl")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "utils"
        / "checkpoint"
        / "checkpoint_handler.py"
    )
    if not path.is_file():
        raise RuntimeError(f"veRL checkpoint handler is missing: {path}")
    return path.resolve()


def ensure_sft_latest_only_checkpoint_hook() -> Path:
    """Patch veRL so validation is followed by pre-save cleanup and latest publication."""
    path = _handler_path()
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return path

    prepare_anchor = (
        '        local_global_step_folder = os.path.join(self.default_local_dir, f"global_step_{step}")\n'
        "        if self.rank == 0:\n"
        '            print(f"Saving checkpoint to: {local_global_step_folder}")\n'
    )
    finalize_anchor = (
        "        if self.rank == 0:\n"
        "            # Update latest checkpoint tracker (atomic write)\n"
    )
    if prepare_anchor not in text or finalize_anchor not in text:
        raise RuntimeError("veRL checkpoint handler layout is not recognized; refusing latest-only patch")

    backup = Path(str(path) + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)

    prepare_patch = (
        '        local_global_step_folder = os.path.join(self.default_local_dir, f"global_step_{step}")\n'
        f"        # {MARKER}\n"
        "        from src.training.sft_checkpoint_policy import prepare_latest_checkpoint\n"
        "        if self.rank == 0:\n"
        "            prepare_latest_checkpoint(self.default_local_dir, local_global_step_folder)\n"
        "        if self.mode == OrchestrationMode.SPMD:\n"
        "            torch.distributed.barrier()\n"
        "        if self.rank == 0:\n"
        '            print(f"Saving checkpoint to: {local_global_step_folder}")\n'
    )
    finalize_patch = (
        "        if self.mode == OrchestrationMode.SPMD:\n"
        "            torch.distributed.barrier()\n"
        "        if self.rank == 0:\n"
        "            from src.training.sft_checkpoint_policy import finalize_latest_checkpoint\n"
        "            finalize_latest_checkpoint(local_global_step_folder)\n"
        "        if self.mode == OrchestrationMode.SPMD:\n"
        "            torch.distributed.barrier()\n\n"
        "        if self.rank == 0:\n"
        "            # Update latest checkpoint tracker (atomic write)\n"
    )
    text = text.replace(prepare_anchor, prepare_patch, 1)
    text = text.replace(finalize_anchor, finalize_patch, 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")
    return path
