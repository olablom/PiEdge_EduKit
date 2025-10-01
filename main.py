#!/usr/bin/env python3
import os, sys, subprocess, pathlib, shutil
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent
LABS = ROOT / "labs"
KERNEL_NAME = "piedge-edukit-312"


def in_repo_venv() -> bool:
    repo = Path(__file__).resolve().parent
    venv_expected = (repo / ".venv").resolve()

    # Primary: use environment variable set by venv-activate
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        return Path(venv_env).resolve() == venv_expected

    # Fallback: compare sys.prefix (can point to Scripts/... on Windows)
    try:
        return Path(sys.prefix).resolve() == venv_expected
    except Exception:
        return False


def get_python_executable():
    """Get the correct Python executable, preferring venv if available."""
    # If we're in a venv, use that Python
    if in_repo_venv():
        venv_python = ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        # Windows fallback
        venv_python_win = ROOT / ".venv" / "Scripts" / "python.exe"
        if venv_python_win.exists():
            return str(venv_python_win)
    # Fallback to sys.executable
    return sys.executable

PY = get_python_executable()


def have(cmd):
    return shutil.which(cmd) is not None


def ensure_kernel():
    # Try to use existing kernelspec; otherwise install quickly.
    try:
        out = subprocess.check_output(
            [PY, "-m", "jupyter", "kernelspec", "list"], text=True
        )
        if KERNEL_NAME in out:
            return
    except Exception:
        pass
    print("→ Installing Jupyter kernel for this environment...")
    subprocess.check_call(
        [PY, "-m", "pip", "install", "-q", "ipykernel", "papermill==2.6.0"]
    )
    subprocess.check_call(
        [
            PY,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            "Python 3.12 (piedge)",
        ]
    )


def pick_notebook():
    nbs = sorted([p for p in LABS.glob("*.ipynb") if p.is_file()])
    if not nbs:
        print("No notebooks found in labs/.")
        sys.exit(1)
    print("\nChoose notebook:")
    for i, nb in enumerate(nbs, 1):
        print(f"  {i}) {nb.name}")
    while True:
        sel = input("Number: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(nbs):
            return nbs[int(sel) - 1]
        print("Invalid choice, try again.")


def open_in_jupyter(nb_path: pathlib.Path, lab=True):
    ensure_kernel()
    # Start Jupyter in same process until user closes (CTRL+C)
    if lab:
        cmd = [PY, "-m", "jupyter", "lab", str(nb_path)]
    else:
        cmd = [PY, "-m", "notebook", str(nb_path)]
    print(f"\nOpening {'Jupyter Lab' if lab else 'Jupyter Notebook'}: {nb_path.name}\n")
    subprocess.call(cmd)


def run_with_papermill(nb_path: pathlib.Path):
    ensure_kernel()
    out = ROOT / "reports" / f"{nb_path.stem}_out.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Run without parameters (notebook works as-is). Add 'parameters' cell for -p support.
    cmd = [PY, "-m", "papermill", str(nb_path), str(out), "-k", KERNEL_NAME]
    print(f"\nRunning headless with papermill → {out}")
    subprocess.check_call(cmd)
    print("Done! You can open the output notebook in Jupyter later.")


def run_full_pipeline():
    """Quick CLI run (terminal) with receipt."""
    cmds = [
        [
            PY,
            "-m",
            "piedge_edukit.train",
            "--fakedata",
            "--no-pretrained",
            "--epochs",
            "1",
            "--batch-size",
            "256",
            "--output-dir",
            "./models",
        ],
        [
            PY,
            "-m",
            "piedge_edukit.benchmark",
            "--fakedata",
            "--model-path",
            "./models/model.onnx",
            "--warmup",
            "1",
            "--runs",
            "3",
            "--providers",
            "CPUExecutionProvider",
        ],
        [
            PY,
            "scripts/evaluate_onnx.py",
            "--model",
            "./models/model.onnx",
            "--fakedata",
            "--limit",
            "16",
        ],
        [
            PY,
            "-m",
            "piedge_edukit.quantization",
            "--fakedata",
            "--model-path",
            "./models/model.onnx",
            "--calib-size",
            "32",
        ],
        [PY, "verify.py"],
    ]
    print("\nRunning full pipeline (terminal)...\n")
    for c in cmds:
        print(">", " ".join(c))
        subprocess.check_call(c)


def main():
    # Quick check: are we running in the right venv?
    if not in_repo_venv():
        print(
            "⚠️  You don't seem to be running the repo's .venv. Activate first:\n"
            "    Git Bash:   source .venv/Scripts/activate\n"
            "    PowerShell: .\\.venv\\Scripts\\Activate.ps1\n"
        )
    
    # Show which Python we're using
    print(f"Using Python: {PY}")
    
    # Menu
    print("\n=== PiEdge EduKit – Terminal Menu ===")
    print("1) Open notebook in Jupyter Lab")
    print("2) Open notebook in Jupyter Notebook")
    print("3) Run notebook headless (papermill) → reports/*_out.ipynb")
    print("4) Run FULL pipeline (terminal) + verify")
    print("5) Exit")
    choice = input("Choice: ").strip()

    if choice == "1":
        nb = pick_notebook()
        open_in_jupyter(nb, lab=True)
    elif choice == "2":
        nb = pick_notebook()
        open_in_jupyter(nb, lab=False)
    elif choice == "3":
        nb = pick_notebook()
        run_with_papermill(nb)
    elif choice == "4":
        run_full_pipeline()
    else:
        print("Goodbye!")


if __name__ == "__main__":
    # Ensure labs/ exists
    if not LABS.exists():
        print("Cannot find 'labs/' directory. Are you in the repo root?")
        sys.exit(1)
    # Minimal Jupyter check
    if not have("jupyter"):
        print(
            "⚠️  Jupyter doesn't seem to be on PATH. Install if you want to open UI:\n"
            "    python -m pip install jupyter"
        )
    main()
