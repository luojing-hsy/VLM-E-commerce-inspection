#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv"
PYTHON="${VENV_DIR}/bin/python"
REQUIREMENTS="${ROOT}/scripts/training-requirements.txt"
OVERRIDES="${ROOT}/scripts/training-overrides.txt"
KERNEL_REQUIREMENTS="${ROOT}/scripts/training-kernel-requirements.txt"
UV_VERSION="${UV_VERSION:-0.12.5}"

if [[ "${1:-}" == "--check" ]]; then
  cd "${ROOT}"
  "${PYTHON}" "${ROOT}/scripts/verify_training_stack.py"
  exec "${PYTHON}" "${ROOT}/scripts/apply_verl_server_patch.py" --check
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/setup.sh [--check]" >&2
  exit 2
fi

export PATH="${HOME}/.local/bin:${HOME}/miniconda3/bin:${PATH}"
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
  export OMP_NUM_THREADS=1
fi

if pgrep -f '([s]rc.training.train_(sft|grpo)|[v]erl.trainer.(main_ppo|sft_trainer))' >/dev/null 2>&1; then
  echo "A project training process is already running under ${ROOT}; refusing to modify the environment." >&2
  exit 1
fi


if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The Qwen3.5/veRL/vLLM training stack is supported by this project on Linux only." >&2
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
  uv venv --python 3.12 --seed "${VENV_DIR}"
fi

if ! "${PYTHON}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "${VENV_DIR} must use Python 3.12; remove it and rerun setup.sh." >&2
  exit 1
fi

uv pip install \
  --python "${PYTHON}" \
  --requirements "${REQUIREMENTS}" \
  --overrides "${OVERRIDES}" \
  --torch-backend cu129 \
  --compile-bytecode

MAX_JOBS="${MAX_JOBS:-4}" uv pip install \
  --python "${PYTHON}" \
  --requirements "${KERNEL_REQUIREMENTS}" \
  --no-build-isolation \
  --compile-bytecode
uv pip install \
  --python "${PYTHON}" \
  --no-deps \
  --editable "${ROOT}"

"${PYTHON}" "${ROOT}/scripts/verify_training_stack.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_server_patch.py"
"${PYTHON}" "${ROOT}/scripts/apply_verl_server_patch.py" --check

echo "Qwen3.5 SFT, validation, test, and GRPO environment is ready: ${VENV_DIR}"
echo "Run launchers from the repository root, for example: bash scripts/grpo.sh --dry-run"
