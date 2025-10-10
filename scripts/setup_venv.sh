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
if [[ -x ".venv/bin/pip" ]]; then
  PIP_EXE=".venv/bin/pip"
elif [[ -x ".venv/Scripts/pip.exe" ]]; then
  PIP_EXE=".venv/Scripts/pip.exe"
elif command -v pip >/dev/null 2>&1; then
  PIP_EXE="pip"
else
  echo "❌ Could not find pip in virtual environment"
  exit 1
fi

"${PIP_EXE}" install -U pip setuptools wheel
"${PIP_EXE}" install -r requirements.txt
"${PIP_EXE}" install -e .

echo "✅ Environment ready"
