#!/usr/bin/env python3
"""Apply the verified veRL 0.8.0 privileged-teacher joint OPD patch."""

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
EXPECTED_PATCH_SHA256 = "6e8c0af00b7e1aca017099968db9878797754432df923061aeb0aa5dd4c1bb93"
PATCH_MARKER = "VLM_PRODUCT_AUDIT_JOINT_OPD_PATCH_V1"
BACKUP_SUFFIX = ".vlm-joint-opd.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches" / "verl-0.8.0-joint-opd.patch"
FILE_HASHES = {
    "experimental/agent_loop/agent_loop.py": (
        "aa4a7b0e58996e119265092b0460ffb2b617d57c416d118d78e2b40aa75e8a7c",
        "9caaa7239081089f0fae0f758f85765721790a3702989423ce9ee70263dfde6a",
    ),
    "experimental/agent_loop/single_turn_agent_loop.py": (
        "4412fefd8d0039b5e033188bed56fb093d6927826f975bb8f84d5a60d32031cc",
        "325d2636d697becf98dd1308583c28ed43370d316f7be561f983a6446e14be2b",
    ),
    "experimental/teacher_loop/teacher_manager.py": (
        "5df4b1ff20eb02f6d39d1478c02d8fee1615e67f624ac8f8a4a9cfdd54aa51a0",
        "e80b58fdf515c7eb60a5c270e823292ae58a085ccc178cdfb57a27cc9cfb76b1",
    ),
    "trainer/distillation/losses.py": (
        "aa1034da42d15a3980b2383b6fc7a53f9ae6e1f7b7ee75f4f58d8b2c40d4db03",
        "539abaf765dcd46eb02af3085d428809e2fd6847580d2c179ec79c01005f894b",
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
        raise RuntimeError(f"veRL patch targets are missing: {missing}")
    return resolved


def verify_patch_file() -> None:
    if not PATCH_FILE.is_file() or sha256(PATCH_FILE) != EXPECTED_PATCH_SHA256:
        raise RuntimeError("joint OPD patch file is missing or its SHA-256 does not match")


def verify_patched(target_map: dict[Path, tuple[str, str]]) -> None:
    for path, (_, expected_patched) in target_map.items():
        if sha256(path) != expected_patched:
            raise RuntimeError(f"patched target SHA-256 mismatch: {path}")
        if PATCH_MARKER not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"patched target is missing marker: {path}")
        py_compile.compile(str(path), doraise=True)


def apply_patch(verl_dir: Path, target_map: dict[Path, tuple[str, str]]) -> None:
    verify_patch_file()
    hashes = {path: sha256(path) for path in target_map}
    if all(hashes[path] == expected_patched for path, (_, expected_patched) in target_map.items()):
        verify_patched(target_map)
        print("veRL joint OPD patch already applied")
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

    git_program = shutil.which("git")
    if git_program is None:
        raise RuntimeError("the system 'git' executable is required")
    try:
        subprocess.run([git_program, "apply", "--check", str(PATCH_FILE)], cwd=verl_dir.parent, check=True)
        subprocess.run([git_program, "apply", str(PATCH_FILE)], cwd=verl_dir.parent, check=True)
        verify_patched(target_map)
    except Exception:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
        raise
    print("applied verified veRL joint OPD patch")


def restore_patch(target_map: dict[Path, tuple[str, str]]) -> None:
    for path, (expected_original, _) in target_map.items():
        if sha256(path) == expected_original:
            continue
        backup = Path(str(path) + BACKUP_SUFFIX)
        if not backup.is_file() or sha256(backup) != expected_original:
            raise RuntimeError(f"verified original backup is unavailable: {backup}")
        shutil.copy2(backup, path)
        py_compile.compile(str(path), doraise=True)
    print("restored original veRL joint OPD targets")


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
            print("verified veRL joint OPD patch")
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
        raise SystemExit(f"veRL joint OPD patch error: {exc}") from exc


if __name__ == "__main__":
    main()
