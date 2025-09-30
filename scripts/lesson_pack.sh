#!/usr/bin/env bash
# filename: scripts/lesson_pack.sh
set -euo pipefail

# Get current date for zip naming
DATE=$(date +%Y%m%d)
ZIP_NAME="VG_Firstname_Lastname_${DATE}.zip"

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

echo "[pack] Package created: ${ZIP_NAME}"
echo "[pack] Size: $(du -h "${ZIP_NAME}" | cut -f1)"
echo ""
echo "To submit: Upload ${ZIP_NAME} to your assignment portal."
echo "Note: Remember to replace 'Firstname_Lastname' with your actual name!"
