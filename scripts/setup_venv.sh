#!/bin/bash
# PiEdge EduKit - Virtual Environment Setup Script
# Works on Windows (Git Bash), macOS, and Linux

set -euo pipefail

echo "🚀 PiEdge EduKit - Setting up virtual environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Error: Python not found. Please install Python 3.8+ first."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Using Python: $PYTHON_CMD"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📋 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Determine activation script path
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash
    ACTIVATE_SCRIPT=".venv/Scripts/activate"
    PIP_CMD=".venv/Scripts/pip"
    PYTHON_CMD_VENV=".venv/Scripts/python"
else
    # macOS/Linux
    ACTIVATE_SCRIPT=".venv/bin/activate"
    PIP_CMD=".venv/bin/pip"
    PYTHON_CMD_VENV=".venv/bin/python"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

# Upgrade pip
echo "📚 Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
$PIP_CMD install -r requirements.txt

# Install package in editable mode
echo "🔧 Installing package in editable mode..."
$PIP_CMD install -e .

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo "To activate the virtual environment manually:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "To run the main script:"
echo "  $PYTHON_CMD_VENV main.py"
echo ""
echo "To verify installation:"
echo "  $PYTHON_CMD_VENV verify.py"
