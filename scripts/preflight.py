#!/usr/bin/env python3
# filename: scripts/preflight.py
"""
Preflight check for PiEdge EduKit micro-lesson.
Prints system info and available providers for quick diagnosis.
Never blocks verify.py - this is informational only.
"""

import sys
import platform
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor != 12:
        print(f"  WARNING: Expected Python 3.12, got {version.major}.{version.minor}")
    else:
        print(f"  OK: Python 3.12 confirmed")


def check_pip():
    """Check pip version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"pip: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("pip: NOT AVAILABLE")


def check_onnxruntime():
    """Check ONNX Runtime variant."""
    try:
        import onnxruntime as ort

        print(f"onnxruntime: {ort.__version__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {', '.join(providers)}")

        # Check if silicon variant
        if "onnxruntime-silicon" in str(ort.__file__):
            print("  OK: Apple Silicon variant detected")
        else:
            print("  INFO: Standard variant")

    except ImportError:
        print("onnxruntime: NOT INSTALLED")


def check_torch():
    """Check PyTorch installation."""
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("torch: NOT INSTALLED")


def check_system():
    """Check system info."""
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"OS: {platform.system()} {platform.release()}")


def main():
    """Run all preflight checks."""
    print("PiEdge EduKit - Preflight Check")
    print("=" * 40)

    check_system()
    print()
    check_python_version()
    check_pip()
    print()
    check_onnxruntime()
    print()
    check_torch()
    print()

    print("Preflight complete. This is informational only.")
    print("Run 'python verify.py' for actual lesson verification.")


if __name__ == "__main__":
    main()
