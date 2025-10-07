#!/usr/bin/env python3
# filename: main.py
import sys, subprocess, os
from pathlib import Path


def sh(*cmd):
    return subprocess.run(cmd, text=True, check=True)


def ensure_editable_install():
    try:
        __import__("piedge_edukit")
        return
    except ImportError:
        pass
    # install -e .
    sh(sys.executable, "-m", "pip", "install", "-e", ".")


def ensure_dirs():
    for d in ("models", "reports", "progress"):
        Path(d).mkdir(parents=True, exist_ok=True)


def open_jupyter():
    # open browser explicitly
    sh(
        sys.executable,
        "-m",
        "jupyter",
        "lab",
        "--ServerApp.open_browser=True",
        "--ServerApp.token=",
    )


if __name__ == "__main__":
    ensure_editable_install()
    ensure_dirs()
    open_jupyter()
