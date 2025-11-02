#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  # Prefer the universal python3 installed via system or Homebrew; fallback handled by PYTHON_BIN.
  if /usr/bin/which python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(/usr/bin/which python3)"
  fi
fi

if [[ -d "${VENV_DIR}" ]]; then
  echo "Virtual environment already exists at ${VENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  echo "Created venv at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate

echo "Environment setup complete. Activate with 'source .venv/bin/activate'."
