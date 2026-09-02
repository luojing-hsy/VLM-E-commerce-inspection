#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${JOINT_CONFIG:-${ROOT}/configs/joint.yaml}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Training environment is missing; run: bash scripts/setup.sh" >&2
  exit 1
fi
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
  export OMP_NUM_THREADS=1
fi
cd "${ROOT}"
read_config_value() {
  "${PYTHON}" - "$1" "$2" <<'PY'
from pathlib import Path
import sys
import yaml
value = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
if value is not None:
    print(value)
PY
}
resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${ROOT}/${value#./}"
  fi
}
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "--print-command" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "${1} does not accept additional arguments in launcher dry-run mode." >&2
    exit 2
  fi
  exec "${PYTHON}" - "${CONFIG}" <<'PY'
import shlex
import sys
from src.training.runtime import build_verl_command, validate_stage_config
config = validate_stage_config(sys.argv[1], "joint")
print(shlex.join(build_verl_command(config, sys.executable)))
PY
fi
if pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[s]ft" >/dev/null 2>&1 || pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[j]oint" >/dev/null 2>&1; then
  echo "Another project training process is already running under ${ROOT}; refusing a concurrent launch." >&2
  exit 1
fi
PREPARE_ONLY=0
if [[ "${1:-}" == "--prepare-only" ]]; then
  PREPARE_ONLY=1
fi
check_local_config_path() {
  local role="$1"
  local key="$2"
  local value
  value="$(read_config_value "${CONFIG}" "${key}")"
  case "${value}" in
    /*|./*|../*|outputs/*|data/*)
      local path
      path="$(resolve_path "${value}")"
      if [[ ! -d "${path}" ]]; then
        echo "${role} checkpoint directory does not exist: ${path}" >&2
        exit 1
      fi
      ;;
  esac
}
if [[ "${PREPARE_ONLY}" -eq 0 ]]; then
  check_local_config_path "student" "model_name_or_path"
  check_local_config_path "teacher" "teacher_model_path"
fi
LOG_FILE="${JOINT_LOG_FILE:-${ROOT}/outputs/joint/joint_$(date +%Y%m%d_%H%M%S)_$$.log}"
if [[ "${JOINT_NO_LOG:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Joint launcher log: ${LOG_FILE}"
fi
exec "${PYTHON}" -m src.training.train_joint --config "${CONFIG}" "$@"
