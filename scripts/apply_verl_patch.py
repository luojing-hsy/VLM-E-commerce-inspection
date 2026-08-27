#!/usr/bin/env python3
"""Apply the verified veRL 0.8.0 JSONL image-path patch."""

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
EXPECTED_PATCH_SHA256 = "6e607722659f1ab398a7613e819f412c48eaafcadbfdab89298049a977147991"
EXPECTED_ORIGINAL_SHA256 = "7463b89114625d736e376a10622253be120bfe8c3f6b7d7b54b49beb790525b8"
EXPECTED_PATCHED_SHA256 = "ec2624116df27f4a762cafcbcdbac025c81268ecae4cb1daec80d105f3c5e095"
PATCH_MARKER = "VLM_PRODUCT_AUDIT_JSONL_IMAGE_PATH_PATCH_V1"
BACKUP_SUFFIX = ".vlm-product-audit.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches" / "verl-0.8.0-jsonl-image-path.patch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_target() -> Path:
    installed_version = importlib.metadata.version("verl")
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )

    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the installed verl package")
    verl_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_dir.is_relative_to(expected_environment):
        raise RuntimeError(f"verl is not installed in {expected_environment}: {verl_dir}")

    target = verl_dir / "utils" / "dataset" / "rl_dataset.py"
    if not target.is_file():
        raise RuntimeError(f"veRL target file is missing: {target}")
    return target


def verify_patch_file() -> None:
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")
    actual = sha256(PATCH_FILE)
    if actual != EXPECTED_PATCH_SHA256:
        raise RuntimeError(
            f"patch SHA-256 mismatch: expected {EXPECTED_PATCH_SHA256}, got {actual}"
        )


def verify_patched(target: Path) -> None:
    actual = sha256(target)
    if actual != EXPECTED_PATCHED_SHA256:
        raise RuntimeError(
            f"patched target SHA-256 mismatch: expected {EXPECTED_PATCHED_SHA256}, got {actual}"
        )
    if PATCH_MARKER not in target.read_text(encoding="utf-8"):
        raise RuntimeError(f"patched target is missing marker {PATCH_MARKER}")
    py_compile.compile(str(target), doraise=True)


def apply_patch(target: Path) -> None:
    verify_patch_file()
    actual = sha256(target)
    if actual == EXPECTED_PATCHED_SHA256:
        verify_patched(target)
        print(f"veRL patch already applied: {target}")
        return
    if actual != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch an unknown rl_dataset.py: "
            f"expected {EXPECTED_ORIGINAL_SHA256}, got {actual}"
        )

    git_program = shutil.which("git")
    if git_program is None:
        raise RuntimeError("the system 'git' executable is required")

    backup = Path(str(target) + BACKUP_SUFFIX)
    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to overwrite an invalid backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)

    try:
        subprocess.run(
            [
                git_program,
                "apply",
                "--check",
                str(PATCH_FILE),
            ],
            check=True,
            cwd=target.parents[3],
        )
        subprocess.run(
            [git_program, "apply", str(PATCH_FILE)],
            check=True,
            cwd=target.parents[3],
        )
        verify_patched(target)
    except Exception:
        shutil.copy2(backup, target)
        raise

    print(f"applied verified veRL patch: {target}")
    print(f"backup: {backup}")
    print(f"patched SHA-256: {sha256(target)}")


def restore_patch(target: Path) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    if sha256(target) == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL target is already original: {target}")
        return
    if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"verified original backup is unavailable: {backup}")
    shutil.copy2(backup, target)
    py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    try:
        target = resolve_target()
        if args.check:
            verify_patch_file()
            verify_patched(target)
            print(f"verified veRL patch: {target}")
        elif args.restore:
            restore_patch(target)
        else:
            apply_patch(target)
    except (
        OSError,
        RuntimeError,
        importlib.metadata.PackageNotFoundError,
        subprocess.CalledProcessError,
        py_compile.PyCompileError,
    ) as exc:
        raise SystemExit(f"veRL patch error: {exc}") from exc


if __name__ == "__main__":
    main()
