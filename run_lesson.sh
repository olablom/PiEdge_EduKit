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

# Ensure the package is available inside this venv
echo "[run] Installing package..."
python -m pip install --upgrade pip >/dev/null 2>&1 || true
# Prefer editable install (works with pyproject.toml)
if ! python -m pip install -e . >/dev/null 2>&1; then
  # Fallback: use source directly
  export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
  echo "[run] Using PYTHONPATH fallback"
else
  echo "[run] Package installed successfully"
fi

# Training (Smoke Test mode)
echo "[run] Training model..."
python -m piedge_edukit.train --fakedata --no-pretrained --epochs 1 --batch-size 256 --output-dir ./models || echo "[run] Training failed, continuing..."

# Benchmark (Smoke Test mode)
echo "[run] Starting latency benchmark..."
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 1 --runs 3 --providers CPUExecutionProvider || echo "[run] Benchmark failed, continuing..."

# Quantization (may fail, that's OK)
echo "[run] Starting quantization benchmark..."

# Auto-fallback: create synthetic calibration data if data/train is missing
if [ ! -d "data/train" ]; then
  echo "[run] data/train saknas → skapar syntetiskt kalibreringsset"
  python - <<'PY'
from PIL import Image
import numpy as np, os
for cls in ['class0','class1']:
    d=os.path.join('data','train',cls); os.makedirs(d, exist_ok=True)
    for i in range(16):
        arr=(np.random.rand(64,64,3)*255).astype('uint8')
        Image.fromarray(arr).save(os.path.join(d,f'{i}.png'))
print("Created synthetic calibration set (32 imgs)")
PY
fi

python -m piedge_edukit.quantization --data-path data/train --model-path ./models/model.onnx --calib-size 32 || echo "[run] Quantization failed (this is OK), continuing..."

# Evaluation (Smoke Test mode)
echo "[run] Running evaluation..."
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 32 || echo "[run] Evaluation failed, continuing..."

# Verification
echo "[run] Running verification..."
python verify.py