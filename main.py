#!/usr/bin/env python3
"""
PiEdge EduKit - One-click bootstrap script
Sets up virtual environment, installs dependencies, and launches Jupyter
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, shell=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            cmd, shell=shell, check=True, capture_output=True, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def main():
    print("🚀 PiEdge EduKit - One-click setup")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("notebooks").exists():
        print("❌ Error: Please run this script from the PiEdge EduKit root directory")
        sys.exit(1)

    # Check if .venv exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        success, output = run_command(f"{sys.executable} -m venv .venv")
        if not success:
            print(f"❌ Failed to create virtual environment: {output}")
            sys.exit(1)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")

    # Determine activation script path
    if platform.system() == "Windows":
        activate_script = ".venv\\Scripts\\activate.bat"
        pip_cmd = ".venv\\Scripts\\pip"
        python_cmd = ".venv\\Scripts\\python"
    else:
        activate_script = ".venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
        python_cmd = ".venv/bin/python"

    # Install requirements
    print("📚 Installing dependencies...")
    success, output = run_command(f"{pip_cmd} install --upgrade pip")
    if not success:
        print(f"❌ Failed to upgrade pip: {output}")
        sys.exit(1)

    success, output = run_command(f"{pip_cmd} install -r requirements.txt")
    if not success:
        print(f"❌ Failed to install requirements: {output}")
        sys.exit(1)
    print("✅ Dependencies installed")

    # Install package in editable mode
    print("🔧 Installing package in editable mode...")
    success, output = run_command(f"{pip_cmd} install -e .")
    if not success:
        print(f"❌ Failed to install package: {output}")
        sys.exit(1)
    print("✅ Package installed")
    
    # Verify package can be imported
    print("🔍 Verifying package installation...")
    success, output = run_command(f"{python_cmd} -c \"import piedge_edukit; print('Package import OK')\"")
    if not success:
        print(f"❌ Package import failed: {output}")
        sys.exit(1)
    print("✅ Package verification passed")

    # Launch Jupyter
    print("🎯 Launching Jupyter Lab...")
    print("=" * 50)
    print("📖 Opening web browser automatically...")
    print("🔗 Start with: 00_run_everything.ipynb")
    print("=" * 50)

    # Launch Jupyter Lab
    jupyter_cmd = f"{python_cmd} -m jupyter lab --ServerApp.open_browser=True --ip=127.0.0.1 --port=8888"
    print(f"Running: {jupyter_cmd}")
    print("Press Ctrl+C to stop Jupyter")

    try:
        subprocess.run(jupyter_cmd, shell=True)
    except KeyboardInterrupt:
        print("\n👋 Jupyter stopped. Goodbye!")


if __name__ == "__main__":
    main()
