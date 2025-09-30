#!/usr/bin/env python3
# smoke_test.py - Smoke test for PiEdge EduKit

import subprocess
import sys
import os
from pathlib import Path
import json


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] SUCCESS")
        if result.stdout:
            print(
                "Output:",
                result.stdout[:500] + "..."
                if len(result.stdout) > 500
                else result.stdout,
            )
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] FAILED")
        print("Error:", e.stderr)
        return False


def check_artifacts(expected_files):
    """Check that expected artifacts exist."""
    print(f"\n{'=' * 60}")
    print("Checking artifacts...")
    print("=" * 60)

    all_exist = True
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[ERROR] {file_path} - MISSING")
            all_exist = False

    return all_exist


def smoke_test_pc():
    """Run smoke test on PC."""
    print("PiEdge EduKit - PC Smoke Test")
    print("=" * 60)

    # Test 1: Install package
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        "Install PiEdge EduKit package",
    ):
        return False

    # Test 2: Check imports
    if not run_command(
        [
            sys.executable,
            "-c",
            "import piedge_edukit; print('Package imported successfully')",
        ],
        "Test package import",
    ):
        return False

    # Test 3: Create sample data structure
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create sample class directories
    for class_name in ["cat", "dog", "bird"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        # Create empty placeholder files
        (class_dir / f"{class_name}_sample.txt").write_text("Sample data")

    # Test 4: Run training (with fakedata)
    if not run_command(
        [
            sys.executable,
            "-m",
            "piedge_edukit.train",
            "--fakedata",
            "--output-dir",
            "./models",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
        "Run training with FakeData (1 epoch)",
    ):
        return False

    # Test 5: Check training artifacts
    training_artifacts = [
        "models/model.onnx",
        "models/labels.json",
        "models/preprocess_config.json",
    ]
    if not check_artifacts(training_artifacts):
        return False

    # Test 6: Run benchmark
    if not run_command(
        [
            sys.executable,
            "-m",
            "piedge_edukit.benchmark",
            "--fakedata",
            "--model-path",
            "./models/model.onnx",
            "--runs",
            "10",
            "--warmup",
            "5",
        ],
        "Run benchmark with FakeData (10 runs)",
    ):
        return False

    # Test 7: Check benchmark artifacts
    benchmark_artifacts = [
        "reports/latency.csv",
        "reports/latency_summary.txt",
        "reports/latency_plot.png",
    ]
    if not check_artifacts(benchmark_artifacts):
        return False

    # Test 8: Run GPIO control (simulation)
    if not run_command(
        [
            sys.executable,
            "-m",
            "piedge_edukit.gpio_control",
            "--fakedata",
            "--model-path",
            "./models/model.onnx",
            "--simulate",
            "--target",
            "class1",
            "--interval",
            "0.1",
            "--duration",
            "2",
        ],
        "Run GPIO control with FakeData (simulation)",
    ):
        return False

    # Test 9: Check GPIO artifacts
    gpio_artifacts = ["reports/gpio_session.txt", "reports/gpio_history.png"]
    if not check_artifacts(gpio_artifacts):
        return False

    print(f"\n{'=' * 60}")
    print("[OK] PC Smoke Test PASSED")
    print("=" * 60)
    return True


def smoke_test_pi():
    """Run smoke test on Pi (minimal version)."""
    print("PiEdge EduKit - Pi Smoke Test")
    print("=" * 60)

    # Test 1: Check if we're on Pi
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpu_info = f.read()
            if "Raspberry Pi" not in cpu_info:
                print(
                    "[WARNING] Not running on Raspberry Pi - skipping Pi-specific tests"
                )
                return True
    except:
        print("[WARNING] Cannot detect Pi - skipping Pi-specific tests")
        return True

    # Test 2: Check Python version
    if not run_command([sys.executable, "--version"], "Check Python version"):
        return False

    # Test 3: Check ONNX Runtime
    if not run_command(
        [
            sys.executable,
            "-c",
            "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')",
        ],
        "Check ONNX Runtime",
    ):
        return False

    # Test 4: Run benchmark (if models exist)
    if Path("models/model.onnx").exists():
        if not run_command(
            [
                sys.executable,
                "-m",
                "piedge_edukit.benchmark",
                "--model-path",
                "./models/model.onnx",
                "--data-path",
                "./data",
                "--runs",
                "10",
                "--warmup",
                "5",
            ],
            "Run benchmark on Pi",
        ):
            return False

    print(f"\n{'=' * 60}")
    print("[OK] Pi Smoke Test PASSED")
    print("=" * 60)
    return True


def main():
    """Main smoke test function."""
    if len(sys.argv) > 1 and sys.argv[1] == "pi":
        success = smoke_test_pi()
    else:
        success = smoke_test_pc()

    if success:
        print("\n[OK] All smoke tests PASSED!")
        sys.exit(0)
    else:
        print("\n[ERROR] Smoke tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
