#!/usr/bin/env python3
"""Install the audited server veRL patch bundle without overwriting unknown edits."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches/verl-0.8.0-server.json"
PATCH = ROOT / "patches/verl-0.8.0-server.patch"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def apply_bundle(site, check=False):
    meta = json.loads(MANIFEST.read_text())
    if digest(PATCH) != meta["patch_sha256"]:
        raise RuntimeError("Patch bundle checksum mismatch")
    pending = []
    for item in meta["files"]:
        path = site / item["path"]
        actual = digest(path)
        if actual == item["patched_sha256"]:
            continue
        if check or actual != item["original_sha256"]:
            raise RuntimeError(f"Unknown or unpatched source: {path}")
        pending.append(item)
    if not pending:
        print(f"Verified {len(meta['files'])} patched veRL files")
        return
    # Replay and validate all changes outside the environment before replacing files.
    with tempfile.TemporaryDirectory(prefix="verl-patch-") as directory:
        stage = Path(directory)
        for item in pending:
            target = stage / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(site / item["path"], target)
        args = ["git", "apply", *["--include=" + x["path"] for x in pending], str(PATCH)]
        subprocess.run(args[:2] + ["--check"] + args[2:], cwd=stage, check=True)
        subprocess.run(args, cwd=stage, check=True)
        for item in pending:
            if digest(stage / item["path"]) != item["patched_sha256"]:
                raise RuntimeError(f"Replay checksum mismatch: {item['path']}")
        for item in pending:
            target = site / item["path"]
            backup = target.with_name(target.name + ".before_server_bundle")
            if not backup.exists():
                shutil.copy2(target, backup)
            shutil.copy2(stage / item["path"], target)
    print(f"Applied {len(pending)} veRL file patches")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("verl") != "0.8.0":
        raise SystemExit("This patch requires verl==0.8.0")
    site = Path(importlib.util.find_spec("verl").origin).resolve().parent.parent
    apply_bundle(site, check=args.check)

if __name__ == "__main__":
    main()
