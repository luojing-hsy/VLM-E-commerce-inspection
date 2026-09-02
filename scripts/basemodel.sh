#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${EVAL_CONFIG:-${ROOT}/configs/eval.yaml}"
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
if pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[s]ft" >/dev/null 2>&1 \
  || pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[j]oint" >/dev/null 2>&1 \
  || pgrep -f "${ROOT}/.venv/bin/python -m src.training.train_[g]rpo_on_joint" >/dev/null 2>&1; then
  echo "A project training process is already running under ${ROOT}; refusing a concurrent evaluation." >&2
  exit 1
fi
MANIFEST="$(resolve_path "$(read_config_value "${CONFIG}" "manifest")")"
COUNTERFACTUAL_MANIFEST="$(resolve_path "$(read_config_value "${CONFIG}" "counterfactual_manifest")")"
CONFIG_PREDICTIONS="$(resolve_path "$(read_config_value "${CONFIG}" "predictions")")"
PREDICTIONS="${BASEMODEL_PREDICTIONS:-${CONFIG_PREDICTIONS}}"
if [[ "${PREDICTIONS}" != /* ]]; then
  PREDICTIONS="$(resolve_path "${PREDICTIONS}")"
fi
if [[ -d "${ROOT}/../models/Qwen3.5-4B" ]]; then
  DEFAULT_MODEL="${ROOT}/../models/Qwen3.5-4B"
else
  DEFAULT_MODEL="Qwen/Qwen3.5-4B"
fi
MODEL="${BASEMODEL_MODEL:-${DEFAULT_MODEL}}"
case "${MODEL}" in
  /*|./*|../*|outputs/*|data/*)
    MODEL="$(resolve_path "${MODEL}")"
    if [[ ! -d "${MODEL}" ]]; then
      echo "Local base model directory does not exist: ${MODEL}" >&2
      exit 1
    fi
    ;;
esac
LOG_FILE="${BASEMODEL_LOG_FILE:-${ROOT}/outputs/test/basemodel_$(date +%Y%m%d_%H%M%S)_$$.log}"
if [[ "${BASEMODEL_NO_LOG:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Base-model evaluation log: ${LOG_FILE}"
fi
if [[ ! -f "${MANIFEST}" ]]; then
  "${PYTHON}" -m src.data.prepare_synthesis --stage eval --config "${CONFIG}"
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Evaluation manifest was not created: ${MANIFEST}" >&2
  exit 1
fi
prediction_matches_manifest() {
  "${PYTHON}" - "${MANIFEST}" "${COUNTERFACTUAL_MANIFEST}" "${PREDICTIONS}" <<'PY'
import json
import sys
from pathlib import Path

def ids(path):
    target = Path(path)
    if not target.is_file():
        return []
    result = []
    seen = set()
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise SystemExit(1)
        seen.add(sample_id)
        result.append(sample_id)
    return result
expected = ids(sys.argv[1]) + ids(sys.argv[2])
actual = ids(sys.argv[3])
if not expected or len(expected) != len(actual) or set(expected) != set(actual):
    raise SystemExit(1)
PY
}
if [[ "${BASEMODEL_FORCE:-0}" == "1" ]] || [[ ! -f "${PREDICTIONS}" ]] || ! prediction_matches_manifest; then
  mkdir -p "$(dirname "${PREDICTIONS}")"
  "${PYTHON}" -m src.evaluation.predict \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --output "${PREDICTIONS}"
fi
exec "${PYTHON}" -m src.evaluation.evaluate \
  --config "${CONFIG}" \
  --predictions "${PREDICTIONS}" \
  "$@"
