from __future__ import annotations

from pathlib import Path

import pytest

from src.training.grpo_checkpoint_policy import (
    finalize_latest_checkpoint,
    prepare_latest_checkpoint,
)


def test_prepare_deletes_old_checkpoint_and_alias(tmp_path: Path) -> None:
    root = tmp_path / "grpo"
    old = root / "global_step_200"
    old.mkdir(parents=True)
    (old / "stale").write_text("old", encoding="utf-8")
    (root / "latest").symlink_to(old.name, target_is_directory=True)
    (root / "latest_checkpointed_iteration.txt").write_text("200\n", encoding="utf-8")

    new = root / "global_step_400"
    prepared = prepare_latest_checkpoint(root, new)

    assert prepared == new.resolve()
    assert not old.exists()
    assert not (root / "latest").exists()
    assert not (root / "latest_checkpointed_iteration.txt").exists()


def test_finalize_points_latest_to_complete_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "grpo"
    checkpoint = root / "global_step_400"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"state")

    alias = finalize_latest_checkpoint(root, checkpoint)

    assert alias.is_symlink()
    assert alias.resolve() == checkpoint.resolve()


def test_prepare_refuses_real_latest_directory(tmp_path: Path) -> None:
    root = tmp_path / "grpo"
    (root / "latest").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        prepare_latest_checkpoint(root, root / "global_step_200")
