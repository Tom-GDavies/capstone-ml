#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEST_ROOT="${SCRIPT_DIR}/deliverables/omtad_xiv"
mkdir -p "${DEST_ROOT}"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  local pattern="$3"

  if [[ -d "${src}" ]]; then
    mkdir -p "${dest}"
    shopt -s nullglob
    local matched=(${src}/${pattern})
    shopt -u nullglob
    if [[ ${#matched[@]} -gt 0 ]]; then
      cp ${matched[@]} "${dest}/"
    else
      echo "Warning: no files matching ${pattern} in ${src}" >&2
    fi
  else
    echo "Warning: missing directory ${src}" >&2
  fi
}

copy_if_exists "${DEST_ROOT}/models" "${DEST_ROOT}/models" "*.pkl"
copy_if_exists "${DEST_ROOT}/models" "${DEST_ROOT}/models" "*.meta.json"
copy_if_exists "${DEST_ROOT}/charts" "${DEST_ROOT}/charts" "*.png"
copy_if_exists "${DEST_ROOT}/samples" "${DEST_ROOT}/samples" "track_features_sample.csv"
copy_if_exists "${DEST_ROOT}/samples" "${DEST_ROOT}/samples" "demo_tracks.csv"

if [[ -f "${DEST_ROOT}/metrics/report.txt" ]]; then
  :
else
  if [[ -f "${SCRIPT_DIR}/metrics/report.txt" ]]; then
    mkdir -p "${DEST_ROOT}/metrics"
    cp "${SCRIPT_DIR}/metrics/report.txt" "${DEST_ROOT}/metrics/"
  else
    echo "Warning: metrics/report.txt not found" >&2
  fi
fi

if [[ -f "${SCRIPT_DIR}/DEMO.md" ]]; then
  cp "${SCRIPT_DIR}/DEMO.md" "${DEST_ROOT}/"
else
  echo "Warning: DEMO.md not found" >&2
fi

export DEST_ROOT

python - <<'PY'
from pathlib import Path
import os

dest = Path(os.environ["DEST_ROOT"])
print("Exported files:")
for path in sorted(dest.rglob("*")):
    if path.is_file():
        rel = path.relative_to(dest)
        size = path.stat().st_size
        print(f"  {rel} ({size} bytes)")
PY
