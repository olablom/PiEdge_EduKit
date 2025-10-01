#!/usr/bin/env python3
import os, sys, subprocess, shutil

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 12
KERNEL_NAME = "piedge-edukit-312"
KERNEL_TITLE = "Python 3.12 (piedge)"
PRIMARY_NB = os.path.join("labs", "00_run_everything.ipynb")
FALLBACK_NB = os.path.join("labs", "01_training_and_export.ipynb")


def fail(msg: str, code: int = 2):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(code)


def ensure_py312():
    if not (
        sys.version_info.major == REQUIRED_MAJOR
        and sys.version_info.minor == REQUIRED_MINOR
    ):
        fail(
            f"This lesson requires Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}. "
            f"Found {sys.version.split()[0]}. See README for install instructions."
        )


def warn_if_not_repo_venv():
    # Nice warning only; proceed anyway so beginners aren't blocked.
    repo_venv = os.path.abspath(os.path.join(os.getcwd(), ".venv"))
    venv_env = os.environ.get("VIRTUAL_ENV", "")

    def samepath(a: str, b: str) -> bool:
        try:
            return os.path.samefile(a, b)
        except Exception:
            return os.path.abspath(a) == os.path.abspath(b)

    looks_ok = False
    if venv_env:
        looks_ok = samepath(venv_env, repo_venv)
    else:
        # Fallback: sys.prefix should live under .venv
        looks_ok = repo_venv in os.path.abspath(sys.prefix)

    if not looks_ok:
        print("WARNING: You don't seem to be using this repo's .venv.")
        print(
            "         Activate first for a clean run:\n"
            "           Git Bash:   source .venv/bin/activate\n"
            "           PowerShell: .\\.venv\\Scripts\\Activate.ps1\n"
        )


def ensure_editable_installed():
    """Ensure the project package is installed in editable mode for the current interpreter."""
    try:
        import piedge_edukit  # noqa: F401
    except Exception:
        print("Installing package in editable mode...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])


def pip_install_if_missing(module: str, pip_name: str | None = None):
    pip_name = pip_name or module
    try:
        __import__(module)
    except Exception:
        print(f"Installing {pip_name} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def ensure_kernel():
    # make sure jupyter + notebook + ipykernel exist
    pip_install_if_missing("jupyter")
    pip_install_if_missing("notebook", "notebook>=7")
    pip_install_if_missing("ipykernel")

    # list kernels
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "jupyter", "kernelspec", "list"],
            text=True,
            errors="ignore",
        )
        if KERNEL_NAME not in out:
            print(f"Installing Jupyter kernel '{KERNEL_NAME}' …")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "ipykernel",
                    "install",
                    "--user",
                    "--name",
                    KERNEL_NAME,
                    "--display-name",
                    KERNEL_TITLE,
                ]
            )
    except FileNotFoundError:
        fail("Could not find 'jupyter' on PATH. Activate .venv or install Jupyter.")


def resolve_target_notebook() -> str:
    if os.path.exists(PRIMARY_NB):
        return PRIMARY_NB
    if os.path.exists(FALLBACK_NB):
        return FALLBACK_NB
    # fallback to labs root
    if os.path.isdir("labs"):
        # open into labs folder
        return "labs/"
    fail("Could not find 'labs' folder. Are you running from the repo root?")


def launch_notebook(target: str):
    # Use Jupyter Server flags (Notebook 7+) and pass the file path directly
    web_target = target.replace(os.sep, "/")
    is_dir = web_target.endswith("/")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "notebook",
        "--ServerApp.open_browser=True",
        "--ServerApp.root_dir=.",
    ]

    if is_dir:
        # Folder view
        cmd.append(f"--ServerApp.default_url=/tree/{web_target}")
    else:
        # Open the exact file and set default_url for consistency
        cmd.append(f"--ServerApp.default_url=/notebooks/{web_target}")
        cmd.append(target)

    print("\nStarting Jupyter Notebook …")
    print(" ".join(cmd))
    
    # Show navigation instructions
    print("\n" + "="*60)
    print("🎓 PiEdge EduKit - Interactive Learning Path")
    print("="*60)
    print("📚 Lesson Sequence:")
    print("  00_run_everything.ipynb    - Quick demo & setup")
    print("  01_training_and_export.ipynb - CNN implementation & training")
    print("  02_latency_benchmark.ipynb   - Performance measurement")
    print("  03_quantization.ipynb        - Model compression")
    print("  04_evaluate_and_verify.ipynb - Evaluation & reflection")
    print("\n💡 Start with 00_run_everything.ipynb for a quick overview,")
    print("   then work through 01-04 for hands-on learning!")
    print("="*60)
    
    subprocess.call(cmd)


def main():
    ensure_py312()
    warn_if_not_repo_venv()
    ensure_editable_installed()
    ensure_kernel()
    target = resolve_target_notebook()

    # Auto-clear outputs and trust the primary lesson notebook before launch
    from pathlib import Path

    nb_path = (
        Path(target).resolve()
        if not target.endswith("/")
        else Path(PRIMARY_NB).resolve()
    )
    if nb_path.exists():
        # 1) Clear saved outputs
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--ClearOutputPreprocessor.enabled=True",
                "--inplace",
                str(nb_path),
            ],
            check=False,
        )
        # 2) Trust notebook
        subprocess.run(
            [sys.executable, "-m", "jupyter", "trust", str(nb_path)], check=False
        )
    launch_notebook(target)


if __name__ == "__main__":
    main()
