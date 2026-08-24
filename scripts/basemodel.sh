#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${EVAL_CONFIG:-${ROOT}/configs/eval.yaml}"
PREDICTIONS="${BASEMODEL_PREDICTIONS:-${ROOT}/outputs/baseline/predictions.jsonl}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Training environment is missing; run: bash scripts/setup.sh" >&2
  exit 1
fi
if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "Base-model predictions are missing: ${PREDICTIONS}" >&2
  echo "Generate predictions with Qwen/Qwen3-VL-4B-Instruct before running fixed evaluation." >&2
  exit 1
fi

cd "${ROOT}"
exec "${PYTHON}" -m src.evaluation.evaluate \
  --config "${CONFIG}" \
  --predictions "${PREDICTIONS}" \
  "$@"
