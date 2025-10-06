#!/usr/bin/env python3
"""
PiEdge EduKit - Verification script
Checks that all components are working correctly
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def main():
    print("🔍 PiEdge EduKit - Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check essential files
    print("\n📁 Essential files:")
    all_good &= check_file_exists("notebooks/00_run_everything.ipynb", "Main notebook")
    all_good &= check_file_exists("notebooks/01_training_and_export.ipynb", "Training notebook")
    all_good &= check_file_exists("notebooks/02_latency_benchmark.ipynb", "Benchmark notebook")
    all_good &= check_file_exists("notebooks/03_quantization.ipynb", "Quantization notebook")
    all_good &= check_file_exists("notebooks/04_evaluate_and_verify.ipynb", "Evaluation notebook")
    all_good &= check_file_exists("src/piedge_edukit/__init__.py", "Package init")
    all_good &= check_file_exists("requirements.txt", "Requirements file")
    all_good &= check_file_exists("pyproject.toml", "Project config")
    
    # Check Python packages
    print("\n🐍 Python packages:")
    all_good &= check_import("torch", "PyTorch")
    all_good &= check_import("torchvision", "TorchVision")
    all_good &= check_import("onnx", "ONNX")
    all_good &= check_import("onnxruntime", "ONNX Runtime")
    all_good &= check_import("jupyter", "Jupyter")
    all_good &= check_import("matplotlib", "Matplotlib")
    all_good &= check_import("numpy", "NumPy")
    all_good &= check_import("pandas", "Pandas")
    
    # Check package modules
    print("\n📦 Package modules:")
    all_good &= check_import("piedge_edukit", "Main package")
    all_good &= check_import("piedge_edukit.model", "Model module")
    all_good &= check_import("piedge_edukit.train", "Training module")
    all_good &= check_import("piedge_edukit.benchmark", "Benchmark module")
    all_good &= check_import("piedge_edukit.quantization", "Quantization module")
    
    # Check virtual environment
    print("\n🌐 Virtual environment:")
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment: .venv/")
    else:
        print("❌ Virtual environment: .venv/ - MISSING")
        all_good = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! PiEdge EduKit is ready to use.")
        print("📖 Start with: python main.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
