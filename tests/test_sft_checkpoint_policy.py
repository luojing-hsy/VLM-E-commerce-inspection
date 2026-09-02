from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.training import sft_checkpoint_policy


def test_prepare_latest_checkpoint_removes_only_prior_checkpoints(tmp_path: Path) -> None:
    old = tmp_path / "global_step_50"
    current = tmp_path / "global_step_100"
    unrelated = tmp_path / "runtime"
    old.mkdir()
    current.mkdir()
    unrelated.mkdir()
    (old / "model.pt").write_text("old", encoding="utf-8")
    (current / "partial.pt").write_text("partial", encoding="utf-8")
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("50", encoding="utf-8")

    result = sft_checkpoint_policy.prepare_latest_checkpoint(tmp_path, current)

    assert result == current.resolve(strict=False)
    assert not old.exists()
    assert not current.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


@pytest.mark.skipif(os.name == "nt", reason="directory symlink creation is restricted on Windows")
def test_finalize_latest_checkpoint_updates_alias_without_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "global_step_100"
    huggingface = checkpoint / "huggingface"
    huggingface.mkdir(parents=True)
    calls = []

    def fake_normalize(path: str | Path, *, keep_raw_backup: bool) -> Path:
        calls.append((Path(path), keep_raw_backup))
        return Path(path)

    monkeypatch.setattr(sft_checkpoint_policy, "normalize_hf_checkpoint", fake_normalize)

    result = sft_checkpoint_policy.finalize_latest_checkpoint(checkpoint)

    assert result == huggingface
    assert calls == [(huggingface, False)]
    assert (tmp_path / "latest").resolve() == checkpoint.resolve()
