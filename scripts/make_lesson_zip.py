#!/usr/bin/env python3
# filename: scripts/make_lesson_zip.py
"""Create lesson package zip with proper naming and size reporting."""

from datetime import datetime
from pathlib import Path
import zipfile
import json


def ensure_dir(p: str) -> Path:
    """Ensure directory exists, create if needed."""
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    """Create lesson package zip."""
    root = Path(__file__).resolve().parents[1]
    date = datetime.now().strftime("%Y%m%d")
    zipname = root / f"Lesson_PiEdge_{date}.zip"

    include = [
        "README.md",
        "README.sv.md",
        "LICENSE",
        "DATA_LICENSES.md",
        "index.html",
        "index.sv.html",
        "requirements.txt",
        "pyproject.toml",
        "run_lesson.sh",
        "verify.py",
        "scripts/",
        "src/",
        "notebooks/",
    ]

    print(f"[pack] Creating lesson package: {zipname.name}")

    # Run verification to ensure everything is ready
    print("[pack] Running verification...")
    verify_script = root / "verify.py"
    if verify_script.exists():
        import subprocess

        try:
            result = subprocess.run(["python", str(verify_script)], capture_output=True, text=True, cwd=root)
            if result.returncode != 0:
                print("[pack] Warning: Verification failed, but continuing with packaging")
                print(f"[pack] Error: {result.stderr}")
            else:
                print("[pack] Verification passed")
        except Exception as e:
            print(f"[pack] Warning: Could not run verification: {e}")
    else:
        print("[pack] Warning: verify.py not found, skipping verification")

    # Create the zip package
    print("[pack] Creating zip package...")
    with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in include:
            p = root / item
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        # Skip certain files
                        if any(
                            skip in str(f)
                            for skip in [
                                "__pycache__",
                                ".ipynb_checkpoints",
                                ".git",
                                ".venv",
                                ".ruff_cache",
                                ".preview_html",
                            ]
                        ):
                            continue
                        z.write(f, f.relative_to(root))
            elif p.is_file():
                z.write(p, p.relative_to(root))

    size_mb = zipname.stat().st_size / 1024 / 1024
    print(f"[pack] Created {zipname.name} ({size_mb:.2f} MB)")
    print("")
    print(f"To submit: Upload {zipname.name} to your portal.")


if __name__ == "__main__":
    main()
