#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${GRPO_CONFIG:-${ROOT}/configs/grpo.yaml}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Training environment is missing; run: bash scripts/setup.sh" >&2
  exit 1
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  set -- --print-command "$@"
fi

cd "${ROOT}"
exec "${PYTHON}" -m src.training.train_grpo \
  --config "${CONFIG}" \
  "$@"
