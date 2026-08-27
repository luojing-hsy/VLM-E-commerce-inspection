#!/usr/bin/env python3
"""Apply the verified veRL 0.8.0 SFT semantic/full loss metrics patch."""

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
EXPECTED_PATCH_SHA256 = "31e291b7b21bc602e4ce62498886ceb9d8753262db67049863d0b6b946da7b26"
PATCH_MARKER = "VLM_PRODUCT_AUDIT_SFT_METRICS_PATCH_V1"
BACKUP_SUFFIX = ".vlm-sft-metrics.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches" / "verl-0.8.0-sft-metrics.patch"
FILE_HASHES = {
    "workers/engine/fsdp/transformer_impl.py": (
        "d4d2c088a889314544f3dbe8f2398f0ee226c40f8833fa3cbbe4301923bc80f7",
        "9a4168f87853822e1937ca8e471eccc433d50c7a2404cb6a8242372d92e59b15",
    ),
    "workers/utils/losses.py": (
        "701a0bf7cd7c0fa3cde670379d44cce5755bcd44222e8866716e69d27a400b5d",
        "3faf0888a59d7799e51c4aa0314a4c3d51ef5a40b3eba5a203878b4d385e1e80",
    ),
    "trainer/sft_trainer.py": (
        "22e21f4a69ec473116454a6e97cd3462ad63a47c580740d7b5b9a31c00587e84",
        "fed32f50094d23d8b70c1fbb3a1773f713a639018ecc3ff5a9a4587c3be4a867",
    ),
    "utils/tracking.py": (
        "a96d48404c53425d4c6f44eb164e72d7a55edfea3337e4279ad9fd8f4695db77",
        "ce816e09cd861f0835ef57e1e9375be0d93b448af054f517f86b828bbf138e6b",
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


def target_map(verl_dir: Path) -> dict[Path, tuple[str, str]]:
    targets = {verl_dir / relative: hashes for relative, hashes in FILE_HASHES.items()}
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise RuntimeError(f"veRL SFT metrics patch targets are missing: {missing}")
    return targets


def verify_patch_file() -> None:
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"SFT metrics patch file is missing: {PATCH_FILE}")
    actual = sha256(PATCH_FILE)
    if actual != EXPECTED_PATCH_SHA256:
        raise RuntimeError(
            f"SFT metrics patch SHA-256 mismatch: expected {EXPECTED_PATCH_SHA256}, got {actual}"
        )


def verify_patched(targets: dict[Path, tuple[str, str]]) -> None:
    for path, (_, expected_patched) in targets.items():
        if sha256(path) != expected_patched:
            raise RuntimeError(f"patched target SHA-256 mismatch: {path}")
        py_compile.compile(str(path), doraise=True)
    if not all(PATCH_MARKER in path.read_text(encoding="utf-8") for path in targets):
        raise RuntimeError(f"patched targets are missing marker {PATCH_MARKER}")


def apply_patch(verl_dir: Path, targets: dict[Path, tuple[str, str]]) -> None:
    verify_patch_file()
    hashes = {path: sha256(path) for path in targets}
    if all(hashes[path] == expected_patched for path, (_, expected_patched) in targets.items()):
        verify_patched(targets)
        print("veRL SFT metrics patch already applied")
        return

    unknown = [
        str(path)
        for path, (expected_original, _) in targets.items()
        if hashes[path] != expected_original
    ]
    if unknown:
        raise RuntimeError(f"refusing to patch unknown veRL files: {unknown}")

    backups: dict[Path, Path] = {}
    for path, (expected_original, _) in targets.items():
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
        verify_patched(targets)
    except Exception:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
        raise

    print("applied verified veRL SFT metrics patch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        verl_dir = resolve_verl_dir()
        targets = target_map(verl_dir)
        if args.check:
            verify_patch_file()
            verify_patched(targets)
            print("verified veRL SFT metrics patch")
        else:
            apply_patch(verl_dir, targets)
    except (
        OSError,
        RuntimeError,
        importlib.metadata.PackageNotFoundError,
        subprocess.CalledProcessError,
        py_compile.PyCompileError,
    ) as exc:
        raise SystemExit(f"veRL SFT metrics patch error: {exc}") from exc


if __name__ == "__main__":
    main()
