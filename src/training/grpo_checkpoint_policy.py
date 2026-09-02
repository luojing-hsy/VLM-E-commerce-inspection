from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import shutil
from pathlib import Path

_MARKER = "VLM_PRODUCT_AUDIT_GRPO_LATEST_ONLY_V1"
_CHECKPOINT_NAME = re.compile(r"global_step_\d+\Z")
_BACKUP_SUFFIX = ".vlm-grpo-latest-only.orig"


def _validated_paths(output_dir: str | Path, checkpoint_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir).resolve()
    if root.name == "sft":
        raise RuntimeError(
            f"refusing to use an SFT checkpoint root for GRPO cleanup: {root}"
        )
    checkpoint = Path(checkpoint_dir).resolve()
    if _CHECKPOINT_NAME.fullmatch(checkpoint.name) is None or checkpoint.parent != root:
        raise ValueError(
            f"checkpoint must be a direct global_step_N child of output_dir: "
            f"output_dir={root}, checkpoint={checkpoint}"
        )
    return root, checkpoint


def _remove_alias_if_safe(path: Path, description: str) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"refusing to remove non-symlink {description}: {path}")


def _remove_tracker_if_safe(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if not path.is_file():
            raise FileExistsError(f"refusing to remove non-file checkpoint tracker: {path}")
        path.unlink()


def prepare_latest_checkpoint(
    output_dir: str | Path, checkpoint_dir: str | Path
) -> Path:
    """Remove the previous raw GRPO checkpoint before the next save."""
    root, checkpoint = _validated_paths(output_dir, checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)

    for old_checkpoint in root.iterdir():
        if _CHECKPOINT_NAME.fullmatch(old_checkpoint.name) is None:
            continue
        if old_checkpoint.is_symlink():
            old_checkpoint.unlink()
        elif old_checkpoint.is_dir():
            shutil.rmtree(old_checkpoint)
        else:
            raise FileExistsError(
                f"refusing to remove non-directory checkpoint: {old_checkpoint}"
            )

    _remove_alias_if_safe(root / "latest", "latest checkpoint alias")
    _remove_tracker_if_safe(root / "latest_checkpointed_iteration.txt")
    return checkpoint


def finalize_latest_checkpoint(
    output_dir: str | Path, checkpoint_dir: str | Path
) -> Path:
    """Point output_dir/latest at a fully written raw GRPO checkpoint."""
    root, checkpoint = _validated_paths(output_dir, checkpoint_dir)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "actor").is_dir():
        raise FileNotFoundError(f"actor checkpoint is incomplete: {checkpoint / 'actor'}")
    if not (checkpoint / "data.pt").is_file():
        raise FileNotFoundError(f"dataloader state is incomplete: {checkpoint / 'data.pt'}")

    alias = root / "latest"
    _remove_alias_if_safe(alias, "latest checkpoint alias")
    alias.symlink_to(checkpoint.name, target_is_directory=True)
    return alias


def _ray_trainer_path() -> Path:
    try:
        version = importlib.metadata.version("verl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("veRL is not installed in the active environment") from exc
    if version != "0.8.0":
        raise RuntimeError(
            f"GRPO checkpoint hook is pinned to veRL 0.8.0, found veRL {version}"
        )

    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate the installed veRL package")
    return Path(spec.origin).resolve().parent / "trainer" / "ppo" / "ray_trainer.py"


def ensure_grpo_latest_only_checkpoint_hook() -> Path:
    """Install the idempotent veRL GRPO pre-delete/latest symlink hook."""
    ray_trainer_path = _ray_trainer_path()
    original = ray_trainer_path.read_text(encoding="utf-8")
    if _MARKER in original:
        return ray_trainer_path

    before_save = (
        '        print(f"local_global_step_folder: {local_global_step_folder}")\n'
        '        actor_local_path = os.path.join(local_global_step_folder, "actor")\n'
    )
    after_data_state = (
        "        torch.save(dataloader_state_dict, dataloader_local_path)\n\n"
        "        # latest checkpointed iteration tracker (for atomic usage)\n"
    )
    before_save_replacement = (
        '        print(f"local_global_step_folder: {local_global_step_folder}")\n'
        "        # VLM_PRODUCT_AUDIT_GRPO_LATEST_ONLY_V1: delete old raw checkpoints first.\n"
        "        from src.training.grpo_checkpoint_policy import prepare_latest_checkpoint\n"
        "        prepare_latest_checkpoint(\n"
        "            self.config.trainer.default_local_dir,\n"
        "            local_global_step_folder,\n"
        "        )\n"
        '        actor_local_path = os.path.join(local_global_step_folder, "actor")\n'
    )
    after_data_state_replacement = (
        "        torch.save(dataloader_state_dict, dataloader_local_path)\n\n"
        "        # VLM_PRODUCT_AUDIT_GRPO_LATEST_ONLY_V1: publish only a complete checkpoint.\n"
        "        from src.training.grpo_checkpoint_policy import finalize_latest_checkpoint\n"
        "        finalize_latest_checkpoint(\n"
        "            self.config.trainer.default_local_dir,\n"
        "            local_global_step_folder,\n"
        "        )\n\n"
        "        # latest checkpointed iteration tracker (for atomic usage)\n"
    )

    if original.count(before_save) != 1:
        raise RuntimeError("veRL ray_trainer.py checkpoint save anchor was not unique")
    if original.count(after_data_state) != 1:
        raise RuntimeError("veRL ray_trainer.py dataloader save anchor was not unique")

    updated = original.replace(before_save, before_save_replacement)
    updated = updated.replace(after_data_state, after_data_state_replacement)
    compile(updated, str(ray_trainer_path), "exec")

    backup = Path(str(ray_trainer_path) + _BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(ray_trainer_path, backup)
    ray_trainer_path.write_text(updated, encoding="utf-8")
    return ray_trainer_path
