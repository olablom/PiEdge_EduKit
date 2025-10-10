#!/usr/bin/env bash
set -euo pipefail

echo -e "🚀 PiEdge EduKit - Setting up virtual environment\n=================================================="

# --- 1) Välj python ---
PY_EXE=""
for CAND in python3 python py; do
  if command -v "$CAND" >/dev/null 2>&1; then
    PY_EXE="$CAND"
    break
  fi
done
if [[ -z "${PY_EXE}" ]]; then
  echo "❌ Could not find python. Please ensure Python 3.12 is on PATH."
  exit 1
fi
echo "✅ Using Python: ${PY_EXE}"

# --- 2) Skapa .venv om saknas ---
if [[ ! -d ".venv" ]]; then
  # Windows py-launcher: prefer 3.12, then 3.11
  if command -v py >/dev/null 2>&1; then
    PYEXE=""
    for v in 3.12 3.11; do
      if py -$v -c "import sys; print()" >/dev/null 2>&1; then PYEXE="py -$v"; break; fi
    done
    : "${PYEXE:=py -3.12}"
    eval "$PYEXE -m venv .venv"
  else
    "${PY_EXE}" -m venv .venv
  fi
  echo "✅ Created .venv"
else
  echo "ℹ️  .venv already exists"
fi

# --- 3) Aktivera (bash på Linux/Mac eller Git-Bash på Windows) ---
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  echo "❌ Could not find venv activation script."
  exit 1
fi

# --- 4) Uppgradera pip & installera krav ---
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .

echo "✅ Environment ready"
