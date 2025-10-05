#!/bin/bash
# demo_pretty.sh - Generate pretty training curves and confusion matrix
# Run from repo root: ./scripts/demo_pretty.sh

set -euo pipefail

echo "========================================"
echo "PiEdge EduKit - Pretty Demo Run"
echo "========================================"

echo
echo "[1/5] Cleaning old artifacts..."
rm -rf models reports progress
mkdir -p reports progress

echo
echo "[2/5] Training with 5 epochs for nice curves..."
python -m piedge_edukit.train --fakedata --no-pretrained --epochs 5 --batch-size 16 --output-dir ./models

echo
echo "[3/5] Benchmarking with Pretty Demo settings..."
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200 --providers CPUExecutionProvider

echo
echo "[4/5] Evaluating with 200 samples for stable confusion matrix..."
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 200

echo
echo "[5/5] Opening generated images..."
if command -v xdg-open >/dev/null 2>&1; then
    # Linux
    xdg-open reports/training_curves.png
    xdg-open reports/confusion_matrix.png
elif command -v open >/dev/null 2>&1; then
    # macOS
    open reports/training_curves.png
    open reports/confusion_matrix.png
else
    echo "Images saved to reports/ - open manually"
fi

echo
echo "========================================"
echo "Demo completed! Check the opened images."
echo "========================================"
