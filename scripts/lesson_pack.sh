#!/usr/bin/env bash
# filename: scripts/lesson_pack.sh
set -euo pipefail

# Get current date for zip naming
DATE=$(date +%Y%m%d)
ZIP_NAME="VG_Ola_Blom_${DATE}.zip"

echo "[pack] Creating lesson package: ${ZIP_NAME}"

# Run verification to ensure everything is ready
echo "[pack] Running verification..."
if [ -f "verify.py" ]; then
    python verify.py || echo "[pack] Warning: Verification failed, but continuing with packaging"
else
    echo "[pack] Warning: verify.py not found, skipping verification"
fi

# Create the zip package with proper exclusions
echo "[pack] Creating zip package..."

# Pick the venv python on any OS
if [ -x ".venv/Scripts/python.exe" ]; then
  VENV_PY=".venv/Scripts/python.exe"   # Windows
elif [ -x ".venv/bin/python" ]; then
  VENV_PY=".venv/bin/python"           # Linux/macOS
else
  VENV_PY="python"
fi

if command -v zip >/dev/null 2>&1; then
  zip -r "${ZIP_NAME}" . \
      -x ".venv/*" \
      -x ".git/*" \
      -x "__pycache__/*" \
      -x "*/__pycache__/*" \
      -x "*.pyc" \
      -x "*.pyo" \
      -x "models/*.onnx" \
      -x "models/*.pth" \
      -x "models_synthetic/*" \
      -x "reports/*.png" \
      -x "reports/*.csv" \
      -x "reports/*.txt" \
      -x "reports/*.json" \
      -x "data/*" \
      -x ".mypy_cache/*" \
      -x ".pytest_cache/*" \
      -x "*.log" \
      -x "*.tmp" \
      -x "*.swp" \
      -x "*.DS_Store" \
      -x "Thumbs.db" \
      -x "*.egg-info/*" \
      -x "progress/receipt.json" \
      -x "*.ipynb_checkpoints/*"
else
  echo "[pack] 'zip' not found, using Python zipfile fallback"
  "$VENV_PY" - <<'PY'
import os, zipfile, sys
name = os.environ.get("ZIP_NAME", "VG_Ola_Blom_00000000.zip")
paths = [
  "index.html","run_lesson.sh","README.md","notebooks","src",
  "verify.py","scripts","requirements.txt","progress","LICENSE",
  "DATA_LICENSES.md",".env.example"
]
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    z.write(os.path.join(root,f))
        elif os.path.isfile(p):
            z.write(p)
        else:
            print(f"[pack] Skipping missing: {p}")
print(f"[pack] Created {name}")
PY
fi

echo "[pack] Package created: ${ZIP_NAME}"
echo "[pack] Size: $(du -h "${ZIP_NAME}" | cut -f1)"
echo ""
echo "To submit: Upload ${ZIP_NAME} to your assignment portal."
echo "Note: Remember to replace 'Firstname_Lastname' with your actual name!"
