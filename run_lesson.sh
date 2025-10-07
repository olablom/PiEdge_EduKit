#!/bin/bash
# PiEdge EduKit - Lesson Runner (macOS/Linux)
# Sets up environment and runs the complete lesson

set -euo pipefail

echo "🎓 PiEdge EduKit - Lesson Runner"
echo "================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the PiEdge EduKit root directory"
    exit 1
fi

# Run setup
echo "🔧 Setting up environment..."
bash scripts/setup_venv.sh

# Activate virtual environment
source .venv/bin/activate

# Run the main script
echo "🚀 Starting lesson..."
python main.py
echo "🔎 Verifying lesson..."
python verify.py || true
echo "✔ Lesson finished. See progress/receipt.json"
