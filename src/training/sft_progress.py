from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import time
from pathlib import Path

MARKER = "VLM_PRODUCT_AUDIT_SFT_PROGRESS_V1"
BACKUP_SUFFIX = ".vlm-sft-progress.orig"


def _trainer_path() -> Path:
    if importlib.metadata.version("verl") != "0.8.0":
        raise RuntimeError("SFT progress hook is pinned to verl==0.8.0")
    spec = importlib.util.find_spec("verl")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate installed verl")
    path = Path(next(iter(spec.submodule_search_locations))) / "trainer" / "sft_trainer.py"
    if not path.is_file():
        raise RuntimeError(f"veRL SFT trainer is missing: {path}")
    return path.resolve()


def ensure_sft_progress_hook() -> Path:
    path = _trainer_path()
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return path
    import_anchor = "import os\n"
    state_anchor = "        total_tokens = 0\n"
    step_anchor = "                    total_tokens += metrics[\"train/global_tokens\"]\n"
    if import_anchor not in text or state_anchor not in text or step_anchor not in text:
        raise RuntimeError("veRL SFT trainer layout is not recognized; refusing progress patch")
    backup = Path(str(path) + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    text = text.replace(import_anchor, import_anchor + "import time\n", 1)
    text = text.replace(
        state_anchor,
        state_anchor
        + "        # " + MARKER + "\n"
        + "        progress_started = time.perf_counter()\n"
        + "        samples_seen = self.resume_global_step * self.global_batch_size\n"
        + "        next_progress_sample = ((samples_seen // 20) + 1) * 20\n",
        1,
    )
    text = text.replace(
        step_anchor,
        step_anchor
        + "                    samples_seen += len(batch_seqlens)\n"
        + "                    if samples_seen >= next_progress_sample and is_logging:\n"
        + "                        elapsed = max(time.perf_counter() - progress_started, 1e-6)\n"
        + "                        rate = samples_seen / elapsed\n"
        + "                        total_samples = self.total_training_steps * self.global_batch_size\n"
        + "                        remaining = max(total_samples - samples_seen, 0)\n"
        + "                        eta = remaining / rate if rate > 0 else float(\"inf\")\n"
        + "                        print(\n"
        + "                            f\"[SFT progress] samples={samples_seen}/{total_samples} \"\n"
        + "                            f\"step={global_step}/{self.total_training_steps} \"\n"
        + "                            f\"rate={rate:.2f} samples/s elapsed={elapsed:.1f}s ETA={eta:.1f}s\",\n"
        + "                            flush=True,\n"
        + "                        )\n"
        + "                        while next_progress_sample <= samples_seen:\n"
        + "                            next_progress_sample += 20\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return path
