#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${EVAL_CONFIG:-${ROOT}/configs/eval.yaml}"
PREDICTIONS="${BASEMODEL_PREDICTIONS:-${ROOT}/outputs/baseline/predictions.jsonl}"
MODEL="${BASEMODEL_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Training environment is missing; run: bash scripts/setup.sh" >&2
  exit 1
fi
MANIFEST="$ROOT/outputs/evaluation/samples_test.jsonl"
if [[ ! -f "$MANIFEST" ]]; then
  cd "$ROOT"
  "$PYTHON" -m src.data.prepare_synthesis --stage eval --config "$CONFIG"
fi

if [[ ! -f "${PREDICTIONS}" ]]; then
  cd "${ROOT}"
  "${PYTHON}" -m src.evaluation.predict \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --output "${PREDICTIONS}"
fi

cd "${ROOT}"
exec "${PYTHON}" -m src.evaluation.evaluate \
  --config "${CONFIG}" \
  --predictions "${PREDICTIONS}" \
  "$@"
