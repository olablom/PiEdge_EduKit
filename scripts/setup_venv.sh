#!/usr/bin/env bash
# filename: scripts/setup_venv.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
REQS_FILE="${REQS_FILE:-requirements.txt}"

echo "[setup] Creating .venv with ${PYTHON_BIN}"

# Hård Python 3.12-kontroll
PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "NOT_FOUND")
if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo "❌ ERROR: Python 3.12 required, got $PYTHON_VERSION"
    echo ""
    echo "Install Python 3.12:"
    echo "  Windows: winget install -e --id Python.Python.3.12"
    echo "  macOS:   brew install python@3.12"
    echo "  Ubuntu:  sudo apt install python3.12 python3.12-venv"
    echo ""
    echo "Then run: PYTHON_BIN=python3.12 bash scripts/setup_venv.sh"
    exit 1
fi

echo "✅ Python 3.12 confirmed: $($PYTHON_BIN -V)"
$PYTHON_BIN -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
# Prefer pip; uv is allowed but not required by the spec
pip install -r "$REQS_FILE"

echo "[setup] .venv ready. To activate: source .venv/bin/activate"
