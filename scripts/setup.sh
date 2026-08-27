#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv"
PYTHON="${VENV_DIR}/bin/python"
REQUIREMENTS="${ROOT}/scripts/training-requirements.txt"
OVERRIDES="${ROOT}/scripts/training-overrides.txt"
UV_VERSION="${UV_VERSION:-0.12.5}"

export PATH="${HOME}/.local/bin:${HOME}/miniconda3/bin:${PATH}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The Qwen3-VL/veRL/vLLM training stack is supported by this project on Linux only." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Required command is unavailable: git" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python)"
  else
    echo "Cannot install uv: Python is unavailable." >&2
    exit 1
  fi
  "${BOOTSTRAP_PYTHON}" -m pip install --user "uv==${UV_VERSION}"
fi

cd "${ROOT}"
if [[ ! -x "${PYTHON}" ]]; then
  uv venv --python 3.12 --seed --clear "${VENV_DIR}"
fi

if ! "${PYTHON}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "${VENV_DIR} must use Python 3.12; remove it and rerun setup.sh." >&2
  exit 1
fi

uv pip install \
  --python "${PYTHON}" \
  --requirements "${REQUIREMENTS}" \
  --overrides "${OVERRIDES}" \
  --torch-backend cu128 \
  --compile-bytecode

uv pip install \
  --python "${PYTHON}" \
  --no-deps \
  --editable "${ROOT}"

"${PYTHON}" "${ROOT}/scripts/verify_training_stack.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_patch.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_patch.py" --check
"${PYTHON}" "${ROOT}/scripts/apply_verl_joint_patch.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_joint_patch.py" --check
"${PYTHON}" "${ROOT}/scripts/apply_verl_sft_metrics_patch.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_sft_metrics_patch.py" --check
"${PYTHON}" "${ROOT}/scripts/apply_verl_mm_projector_patch.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_mm_projector_patch.py" --check

echo "Qwen3-VL SFT and joint veRL GRPO+OPD environment is ready: ${VENV_DIR}"
echo "Run launchers from the repository root, for example: bash scripts/joint.sh --dry-run"
