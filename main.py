#!/usr/bin/env python3
"""
PiEdge EduKit - One-click bootstrap script
Creates .venv, installs requirements, registers kernel, and launches Jupyter.
"""

import os
import subprocess
import sys
import shutil
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
IS_WIN = platform.system() == "Windows"


def run(cmd, env=None):
    """Run command and print it."""
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, env=env or os.environ)


def python_exe():
    """Get Python executable path in venv."""
    return os.path.join(VENV, "Scripts" if IS_WIN else "bin", "python")


def ensure_venv():
    """Create .venv if it doesn't exist."""
    if not os.path.exists(VENV):
        print("[setup] Creating .venv ...")
        run([sys.executable, "-m", "venv", VENV])
    else:
        print("[setup] .venv exists")


def pip_install():
    """Install requirements and package."""
    print("[setup] Installing requirements ...")
    run([python_exe(), "-m", "pip", "install", "--upgrade", "pip"])
    run([python_exe(), "-m", "pip", "install", "-r", os.path.join(HERE, "requirements.txt")])

    # Install our package (editable preferred)
    try:
        run([python_exe(), "-m", "pip", "install", "-e", HERE])
        print("[setup] Package installed successfully")
    except subprocess.CalledProcessError:
        print("[setup] Editable install failed, continuing with source via PYTHONPATH")


def ensure_kernel():
    """Register Jupyter kernel 'piedge'."""
    print("[setup] Register Jupyter kernel 'piedge' ...")
    run(
        [
            python_exe(),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "piedge",
            "--display-name",
            "Python 3.12 (.venv piedge)",
        ]
    )


def launch_notebook():
    """Launch Jupyter with the main notebook."""
    nb = os.path.join(HERE, "notebooks", "00_run_everything.ipynb")
    if not os.path.exists(nb):
        raise SystemExit(f"Notebook not found: {nb}")

    print("[run] Launching Jupyter with the 'piedge' kernel ...")
    print("[run] The notebook will open automatically.")
    print("[run] Select 'Python 3.12 (.venv piedge)' as your kernel.")

    # Start Jupyter Notebook
    run([python_exe(), "-m", "jupyter", "notebook", nb, "--NotebookApp.default_kernel_name=piedge"])


def main():
    """Main bootstrap sequence."""
    print("PiEdge EduKit - One-click setup")
    print("=" * 50)

    try:
        ensure_venv()
        pip_install()
        ensure_kernel()
        launch_notebook()
    except KeyboardInterrupt:
        print("\n[setup] Interrupted by user")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[setup] Error: {e}")
        print("[setup] Try running manually:")
        print("  python -m venv .venv")
        print("  .venv/Scripts/activate  # Windows")
        print("  .venv/bin/activate      # macOS/Linux")
        print("  pip install -r requirements.txt")
        print("  pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    main()
