#!/usr/bin/env bash
set -euo pipefail

# Pick the venv python on any OS
if [ -x ".venv/Scripts/python.exe" ]; then
  VENV_PY=".venv/Scripts/python.exe"   # Windows
elif [ -x ".venv/bin/python" ]; then
  VENV_PY=".venv/bin/python"           # Linux/macOS
else
  echo "❌ .venv not found. Run: bash scripts/setup_venv.sh" >&2
  exit 1
fi

# Enforce we're inside the venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "❌ Not running inside .venv."
  echo "   Run: bash scripts/setup_venv.sh && source .venv/bin/activate   (Git Bash)"
  echo "        OR .\\.venv\\Scripts\\Activate.ps1                        (PowerShell)"
  exit 1
fi

# Preflight: Python 3.12
"$VENV_PY" - <<'PY'
import sys
v = sys.version_info
ok = (v.major, v.minor) == (3, 12)
print(f"Python: {sys.version.split()[0]}")
if not ok:
    raise SystemExit("❌ Requires Python 3.12")
print("✅ Preflight OK")
PY

echo "➡️  Open notebooks in order: notebooks/00_run_everything.ipynb → 01 → 02 → 03 → 04"

# Run preflight check
echo "[run] Running preflight check..."
python scripts/preflight.py

echo "[run] Starting micro-lesson (see index.html for instructions)"

# Training
echo "[run] Training model..."
python -m piedge_edukit.train --ci-fast --no-pretrained || echo "[run] Training failed, continuing..."

# Benchmark
echo "[run] Starting latency benchmark..."
python -m piedge_edukit.benchmark --ci-fast || echo "[run] Benchmark failed, continuing..."

# Quantization
echo "[run] Starting quantization benchmark..."
python -m piedge_edukit.quantization --ci-fast || echo "[run] Quantization failed, continuing..."

# Evaluation
echo "[run] Running evaluation..."
python scripts/evaluate_onnx.py --limit 16 || echo "[run] Evaluation failed, continuing..."

# Verification
echo "[run] Running verification..."
python verify.py