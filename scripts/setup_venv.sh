#!/usr/bin/env bash
# filename: scripts/setup_venv.sh
set -euo pipefail

REQS_FILE="${REQS_FILE:-requirements.txt}"

pick_py() {
  if command -v py >/dev/null 2>&1; then
    echo "py -3.12"
  elif command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
  elif command -v python >/dev/null 2>&1 && python -c 'import sys;exit(0 if sys.version_info[:2]==(3,12) else 1)'; then
    echo "python"
  else
    echo ""
  fi
}

PYTHON_BIN=$(pick_py)
if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Hittade ingen Python 3.12. Se README för installation." >&2
  exit 1
fi

echo "[setup] Creating .venv with ${PYTHON_BIN}"

echo "✅ Python 3.12 confirmed: $($PYTHON_BIN -V)"
$PYTHON_BIN -m venv .venv
# Aktivera venv i bash/Git Bash
# shellcheck disable=SC1091
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r "$REQS_FILE"
pip install -e .
echo "✅ Venv klar. Kör: bash run_lesson.sh"
