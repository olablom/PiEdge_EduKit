#!/usr/bin/env bash
# filename: run_lesson.sh
set -euo pipefail

# --- Python 3.12 gate ---
need="3.12"
have=$(python -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' 2>/dev/null || true)
if [ "$have" != "$need" ]; then
  echo "❌ Python $need krävs (du har $have eller saknas)."
  if command -v py >/dev/null 2>&1; then
    echo "➡️  Windows: kör i Git Bash:"
    echo "    winget install --id Python.Python.3.12 -e && exec bash"
  else
    echo "➡️  Se README för installationskommandon (macOS/Linux)."
  fi
  exit 1
fi

# 1) Ensure venv
if [ ! -d ".venv" ]; then
  echo "[run] .venv not found -> running scripts/setup_venv.sh"
  bash scripts/setup_venv.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Hård Python 3.12-kontroll efter aktivering
python -c "import sys; assert sys.version_info[:2]==(3,12), f'Python 3.12 required, got {sys.version_info.major}.{sys.version_info.minor}'"
echo "✅ Python 3.12 confirmed in venv"

# 2) Initialize progress tracking
mkdir -p progress
if [ ! -f progress/lesson_progress.json ]; then
  echo '{"steps":[],"started_at":null,"completed_at":null}' > progress/lesson_progress.json
fi

# 3) Preflight check (informational only)
echo "[run] Running preflight check..."
python scripts/preflight.py || echo "[run] Preflight failed, continuing..."

# 4) Launch the guided flow (30-min micro-lesson)
echo "[run] Starting micro-lesson (see index.html for instructions)"
# Minimal golden path using --fakedata so it is self-contained
python -m piedge_edukit.train --fakedata --output-dir ./models || echo "[run] Training failed, continuing..."
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200 || echo "[run] Benchmark failed, continuing..."
python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 25 || echo "[run] Quantization failed, continuing..."
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata || echo "[run] Evaluation failed, continuing..."

# 5) Verify (auto-checks + JSON receipt) - ALWAYS run
echo "[run] Running verification..."
python verify.py
echo "[run] Done. See progress/receipt.json"
