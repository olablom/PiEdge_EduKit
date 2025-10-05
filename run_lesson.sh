#!/usr/bin/env bash
set -euo pipefail

echo "[run] PiEdge EduKit - One-click Bootstrap (macOS/Linux)"
echo "======================================================="

# Check if Python 3.12 is available
if ! python3 -V >/dev/null 2>&1; then
    echo "[ERROR] Python 3.12 required. Please install Python 3.12 and try again."
    exit 1
fi

# Run the main bootstrap script
echo "[run] Starting one-click setup..."
python3 main.py