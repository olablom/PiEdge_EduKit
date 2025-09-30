#!/bin/bash
# install_pi_requirements.sh - Install Pi-specific requirements
# Run this script on Raspberry Pi with 64-bit OS

set -euo pipefail

echo "Installing PiEdge EduKit requirements for Raspberry Pi..."

# Check if we're on a Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Warning: This script is designed for Raspberry Pi"
    echo "Continuing anyway..."
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher required, found $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev

# Create virtual environment (recommended)
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install onnxruntime FIRST to avoid NumPy ABI conflicts
echo "Installing ONNX Runtime (CPU only)..."
pip install onnxruntime

# Install other requirements in correct order
echo "Installing other requirements..."
pip install -r ../requirements-pi.txt

# GPIO library is handled by requirements-pi.txt with platform condition
echo "GPIO library installation handled by requirements file"

echo "Installation complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To add GPIO permissions:"
echo "  sudo usermod -aG gpio \$USER"
echo "  # Then logout and login again"

