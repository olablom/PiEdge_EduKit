#!/usr/bin/env python3
# filename: utils/nb_helpers.py
"""
Notebook helper functions for PiEdge EduKit.
Provides run_module() and run_script() functions for cross-platform execution.
"""

import os
import sys
import shlex
import subprocess
from pathlib import Path

# Ensure we're in the repo root
ROOT = Path.cwd()
for d in ("models", "reports", "progress"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# Add repo root to Python path for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Also add the parent directory in case we're running from notebooks/
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


def _pretty(cmd_list):
    """Format command list for display."""
    return " ".join(shlex.quote(str(c)) for c in cmd_list)


def run_module(title: str, module: str, *args: str, cwd: Path | None = None):
    """
    Run `python -m <module> [args...]` and stream output; raise on non-zero.

    Args:
        title: Display title for the command
        module: Module name to run (e.g., 'piedge_edukit.train')
        *args: Additional arguments to pass to the module
        cwd: Working directory (defaults to repo root)

    Returns:
        CompletedProcess object

    Raises:
        RuntimeError: If the command fails (non-zero exit code)
    """
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n▶ {title}\n$ {_pretty(cmd)}\n")
    proc = subprocess.run(cmd, cwd=cwd or ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"{title} failed with code {proc.returncode}")
    return proc


def run_script(title: str, script_path: str, *args: str, cwd: Path | None = None):
    """
    Run `python path/to/script.py [args...]` and stream output; raise on non-zero.

    Args:
        title: Display title for the command
        script_path: Path to the script (relative to cwd or absolute)
        *args: Additional arguments to pass to the script
        cwd: Working directory (defaults to repo root)

    Returns:
        CompletedProcess object

    Raises:
        RuntimeError: If the command fails (non-zero exit code)
    """
    script = (
        str((cwd or ROOT) / script_path)
        if not script_path.startswith(".")
        else script_path
    )
    cmd = [sys.executable, script, *args]
    print(f"\n▶ {title}\n$ {_pretty(cmd)}\n")
    proc = subprocess.run(cmd, cwd=cwd or ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"{title} failed with code {proc.returncode}")
    return proc
