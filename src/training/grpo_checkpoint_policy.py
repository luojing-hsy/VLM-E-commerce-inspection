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

def ensure_grpo_final_checkpoint_hook() -> Path:
    """Install a final-save fallback for natural loop exhaustion."""
    path = _ray_trainer_path()
    original = path.read_text(encoding="utf-8")
    marker = "VLM_PRODUCT_AUDIT_GRPO_FINAL_CHECKPOINT_V1"
    if marker in original:
        return path

    anchor = (
        "        # Ensure dump executor is shut down when training loop ends without reaching is_last_step\n"
        "        self._shutdown_dump_executor()\n"
    )
    replacement = (
        "        # Ensure dump executor is shut down when training loop ends without reaching is_last_step\n"
        "        self._shutdown_dump_executor()\n"
        f"        # {marker}: save the actual final processed step.\n"
        "        final_step = self.global_steps - 1\n"
        "        if final_step > 0 and (\n"
        "            self.config.trainer.save_freq <= 0\n"
        "            or final_step % self.config.trainer.save_freq != 0\n"
        "        ):\n"
        "            previous_global_step = self.global_steps\n"
        "            self.global_steps = final_step\n"
        "            try:\n"
        "                self._save_checkpoint()\n"
        "            finally:\n"
        "                self.global_steps = previous_global_step\n"
    )
    if original.count(anchor) != 1:
        raise RuntimeError("veRL ray_trainer.py final-save anchor was not unique")
    updated = original.replace(anchor, replacement)
    compile(updated, str(path), "exec")
    backup = Path(str(path) + _BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    return path

def ensure_grpo_filter_reward_mean_hook() -> Path:
    """Install dynamic-filter reward mean logging in veRL."""
    path = _ray_trainer_path()
    original = path.read_text(encoding="utf-8")
    marker = "VLM_PRODUCT_AUDIT_GRPO_FILTER_REWARD_MEAN_V1"
    if marker in original:
        return path

    stats_anchor = (
        '        filter_pending_stats = {"generated": 0, "kept": 0, "filtered": 0, "fallback": 0}\n'
    )
    stats_replacement = (
        f"        # {marker}\n"
        '        filter_pending_stats = {"generated": 0, "kept": 0, "filtered": 0, "fallback": 0, '
        '"filtered_reward_sum": 0.0, "filtered_reward_count": 0}\n'
    )
    if original.count(stats_anchor) != 2:
        raise RuntimeError("veRL ray_trainer.py filter stats anchor was not unique twice")
    updated = original.replace(stats_anchor, stats_replacement)

    reward_anchor = (
        "                        prompt_uid2metric_vals = defaultdict(list)\n"
        "                        for uid, metric_value in zip(\n"
        "                            new_batch.non_tensor_batch[\"uid\"], metric_values, strict=True\n"
        "                        ):\n"
        "                            prompt_uid2metric_vals[str(uid)].append(float(metric_value))\n"
    )
    reward_replacement = (
        "                        reward_values = (\n"
        "                            new_batch.batch[\"token_level_scores\"]\n"
        "                            .sum(dim=-1)\n"
        "                            .detach()\n"
        "                            .cpu()\n"
        "                            .numpy()\n"
        "                        )\n"
        "                        prompt_uid2metric_vals = defaultdict(list)\n"
        "                        prompt_uid2reward_vals = defaultdict(list)\n"
        "                        for uid, metric_value, reward_value in zip(\n"
        "                            new_batch.non_tensor_batch[\"uid\"], metric_values, reward_values, strict=True\n"
        "                        ):\n"
        "                            prompt_uid2metric_vals[str(uid)].append(float(metric_value))\n"
        "                            prompt_uid2reward_vals[str(uid)].append(float(reward_value))\n"
    )
    if updated.count(reward_anchor) != 1:
        raise RuntimeError("veRL ray_trainer.py reward grouping anchor was not unique")
    updated = updated.replace(reward_anchor, reward_replacement)

    filter_anchor = (
        "                        kept_prompt_uids = []\n"
        "                        for uid, values in prompt_uid2metric_vals.items():\n"
        "                            std = float(np.std(values))\n"
        "                            if std > filter_groups_zero_variance_eps or len(values) == 1:\n"
        "                                kept_prompt_uids.append(uid)\n"
    )
    filter_replacement = (
        "                        kept_prompt_uids = []\n"
        "                        for uid, values in prompt_uid2metric_vals.items():\n"
        "                            std = float(np.std(values))\n"
        "                            if std > filter_groups_zero_variance_eps or len(values) == 1:\n"
        "                                kept_prompt_uids.append(uid)\n"
        "                            else:\n"
        "                                filter_pending_stats[\"filtered_reward_sum\"] += float(np.mean(prompt_uid2reward_vals[uid]))\n"
        "                                filter_pending_stats[\"filtered_reward_count\"] += 1\n"
    )
    if updated.count(filter_anchor) != 1:
        raise RuntimeError("veRL ray_trainer.py filter classification anchor was not unique")
    updated = updated.replace(filter_anchor, filter_replacement)

    metric_anchor = (
        "                            \"grpo/filter_zero_variance_ratio\": (\n"
        "                                filter_pending_stats[\"filtered\"] / generated_groups\n"
        "                                if generated_groups\n"
        "                                else 0.0\n"
        "                            ),\n"
    )
    metric_replacement = metric_anchor + (
        "                            \"grpo/filter_zero_variance_group_reward_mean\": (\n"
        "                                filter_pending_stats[\"filtered_reward_sum\"] / filter_pending_stats[\"filtered_reward_count\"]\n"
        "                                if filter_pending_stats[\"filtered_reward_count\"]\n"
        "                                else 0.0\n"
        "                            ),\n"
    )
    if updated.count(metric_anchor) != 1:
        raise RuntimeError("veRL ray_trainer.py filter metrics anchor was not unique")
    updated = updated.replace(metric_anchor, metric_replacement)

    compile(updated, str(path), "exec")
    backup = Path(str(path) + _BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    return path
