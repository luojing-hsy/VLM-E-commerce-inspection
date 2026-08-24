#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE="${ROOT}/.venv/bin/accelerate"
CONFIG="${SFT_CONFIG:-${ROOT}/configs/sft.yaml}"

if [[ ! -x "${ACCELERATE}" ]]; then
  echo "Training environment is missing; run: bash scripts/setup.sh" >&2
  exit 1
fi

cd "${ROOT}"
exec "${ACCELERATE}" launch -m src.training.train_sft \
  --config "${CONFIG}" \
  "$@"
