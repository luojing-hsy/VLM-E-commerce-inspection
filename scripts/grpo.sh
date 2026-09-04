#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${GRPO_CONFIG:-${ROOT}/configs/grpo.yaml}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Training environment is missing; run: bash scripts/setup.sh" >&2
  exit 1
fi
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
  export OMP_NUM_THREADS=1
fi
cd "${ROOT}"
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--print-command" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "${1} does not accept additional arguments in launcher dry-run mode." >&2
    exit 2
  fi
  exec "${PYTHON}" -m src.training.train_grpo --config "${CONFIG}" --print-command
fi
if pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[s]ft" >/dev/null 2>&1 \
  || pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[g]rpo" >/dev/null 2>&1; then
  echo "Another project training process is already running under ${ROOT}; refusing a concurrent launch." >&2
  exit 1
fi
LOG_FILE="${GRPO_LOG_FILE:-${ROOT}/outputs/grpo/grpo_$(date +%Y%m%d_%H%M%S)_$$.log}"
if [[ "${GRPO_NO_LOG:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "GRPO launcher log: ${LOG_FILE}"
fi
exec "${PYTHON}" -m src.training.train_grpo --config "${CONFIG}" "$@"
