from __future__ import annotations

import argparse

from src.common import read_jsonl
from src.training.runtime import validate_stage_config, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and stage a regional-to-global OPD run")
    parser.add_argument("--config", default="configs/opd.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = validate_stage_config(args.config, "opd")
    manifest = write_run_manifest(args.config, config)
    rows = read_jsonl(config["dataset"])
    approved = sum(row.get("teacher_filter_status") == "approved" for row in rows)
    if not approved:
        print(
            f"OPD candidates validated ({len(rows)} rows), but final training data is pending "
            f"frozen-GRPO teacher inference and rule filtering; metadata: {manifest}"
        )
        return
    print(f"OPD run inputs validated ({approved} approved rows); metadata: {manifest}")
    if not args.prepare_only:
        raise SystemExit("The repository provides the tested OPD loss and data contract; model rollout wiring remains environment-specific. Use --prepare-only for the reproducibility check.")


if __name__ == "__main__":
    main()
