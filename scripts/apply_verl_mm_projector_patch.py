#!/usr/bin/env python3
"""Apply the verified veRL 0.8.0 Qwen3-VL MM projector SFT patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import py_compile
import shutil
import subprocess
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_PATCH_SHA256 = "36590c96c43b6e55bb49538033c970ece8dc0e33f87f96898c239fd525efe454"
PATCH_MARKER = "VLM_PRODUCT_AUDIT_MM_PROJECTOR_PATCH_V1"
BACKUP_SUFFIX = ".vlm-mm-projector.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches" / "verl-0.8.0-mm-projector.patch"
FILE_HASHES = {
    "workers/config/model.py": (
        "4b382b0c06edf133c332822b61c5cf57f973f244d0f602f3f4e161956e6a775c",
        "3bc331d8cd0b9d6bf74a27bfac7a75a6807efa1791c1e52f105ec815722cd8f8",
    ),
    "workers/engine/fsdp/transformer_impl.py": (
        "b4f6243471f22c08dbfa472d13f708e8de609705ecff13f23713258160cf7902",
        "d4d2c088a889314544f3dbe8f2398f0ee226c40f8833fa3cbbe4301923bc80f7",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_verl_dir() -> Path:
    version = importlib.metadata.version("verl")
    if version != EXPECTED_VERL_VERSION:
        raise RuntimeError(f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={version}")
    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the installed verl package")
    verl_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_dir.is_relative_to(expected_environment):
        raise RuntimeError(f"verl is not installed in {expected_environment}: {verl_dir}")
    return verl_dir


def targets(verl_dir: Path) -> dict[Path, tuple[str, str]]:
    resolved = {verl_dir / relative: hashes for relative, hashes in FILE_HASHES.items()}
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise RuntimeError(f"veRL MM projector patch targets are missing: {missing}")
    return resolved


def verify_patch_file() -> None:
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"MM projector patch file is missing: {PATCH_FILE}")
    actual = sha256(PATCH_FILE)
    if actual != EXPECTED_PATCH_SHA256:
        raise RuntimeError(
            f"MM projector patch SHA-256 mismatch: expected {EXPECTED_PATCH_SHA256}, got {actual}"
        )


def verify_patched(target_map: dict[Path, tuple[str, str]]) -> None:
    for path, (_, expected_patched) in target_map.items():
        if sha256(path) != expected_patched:
            raise RuntimeError(f"patched target SHA-256 mismatch: {path}")
        py_compile.compile(str(path), doraise=True)
    if not any(PATCH_MARKER in path.read_text(encoding="utf-8") for path in target_map):
        raise RuntimeError(f"patched targets are missing marker {PATCH_MARKER}")


def apply_patch(verl_dir: Path, target_map: dict[Path, tuple[str, str]]) -> None:
    verify_patch_file()
    hashes = {path: sha256(path) for path in target_map}
    if all(hashes[path] == expected_patched for path, (_, expected_patched) in target_map.items()):
        verify_patched(target_map)
        print("veRL MM projector patch already applied")
        return

    unknown = [
        str(path)
        for path, (expected_original, _) in target_map.items()
        if hashes[path] != expected_original
    ]
    if unknown:
        raise RuntimeError(f"refusing to patch unknown veRL files: {unknown}")

    backups: dict[Path, Path] = {}
    for path, (expected_original, _) in target_map.items():
        backup = Path(str(path) + BACKUP_SUFFIX)
        if backup.exists() and sha256(backup) != expected_original:
            raise RuntimeError(f"refusing to overwrite invalid backup: {backup}")
        if not backup.exists():
            shutil.copy2(path, backup)
        backups[path] = backup

    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("the system 'patch' executable is required")
    try:
        with PATCH_FILE.open("rb") as patch_stream:
            subprocess.run(
                [patch_program, "--dry-run", "-p1", "-d", str(verl_dir.parent)],
                stdin=patch_stream,
                check=True,
            )
        with PATCH_FILE.open("rb") as patch_stream:
            subprocess.run(
                [patch_program, "-p1", "-d", str(verl_dir.parent)],
                stdin=patch_stream,
                check=True,
            )
        verify_patched(target_map)
    except Exception:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
        raise

    print("applied verified veRL MM projector patch")
    for path in target_map:
        print(f"patched: {path} ({sha256(path)})")


def restore_patch(target_map: dict[Path, tuple[str, str]]) -> None:
    for path, (expected_original, _) in target_map.items():
        if sha256(path) == expected_original:
            continue
        backup = Path(str(path) + BACKUP_SUFFIX)
        if not backup.is_file() or sha256(backup) != expected_original:
            raise RuntimeError(f"verified original backup is unavailable: {backup}")
        shutil.copy2(backup, path)
        py_compile.compile(str(path), doraise=True)
    print("restored original veRL MM projector targets")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    try:
        verl_dir = resolve_verl_dir()
        target_map = targets(verl_dir)
        if args.check:
            verify_patch_file()
            verify_patched(target_map)
            print("verified veRL MM projector patch")
        elif args.restore:
            restore_patch(target_map)
        else:
            apply_patch(verl_dir, target_map)
    except (
        OSError,
        RuntimeError,
        importlib.metadata.PackageNotFoundError,
        subprocess.CalledProcessError,
        py_compile.PyCompileError,
    ) as exc:
        raise SystemExit(f"veRL MM projector patch error: {exc}") from exc


if __name__ == "__main__":
    main()
