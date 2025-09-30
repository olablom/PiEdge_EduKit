@echo off
REM filename: run_lesson.bat
REM Windows batch version of run_lesson.sh

REM 1) Ensure venv
if not exist ".venv" (
    echo [run] .venv not found -^> running scripts/setup_venv.bat
    call scripts/setup_venv.bat
)

REM 2) Activate venv
call .venv\Scripts\activate.bat

REM Hård Python 3.12-kontroll efter aktivering
python -c "import sys; assert sys.version_info[:2]==(3,12), f'Python 3.12 required, got {sys.version_info.major}.{sys.version_info.minor}'"
if errorlevel 1 (
    echo ERROR: Python 3.12 required in venv
    exit /b 1
)
echo OK: Python 3.12 confirmed in venv

REM 3) Initialize progress tracking
if not exist "progress" mkdir progress
if not exist "progress\lesson_progress.json" (
    echo {"steps":[],"started_at":null,"completed_at":null} > progress\lesson_progress.json
)

REM 4) Preflight check (informational only)
echo [run] Running preflight check...
python scripts/preflight.py || echo [run] Preflight failed, continuing...

REM 5) Launch the guided flow (30-min micro-lesson)
echo [run] Starting micro-lesson (see index.html for instructions)
REM Minimal golden path using --fakedata so it is self-contained
python -m piedge_edukit.train --fakedata --output-dir ./models || echo [run] Training failed, continuing...
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200 || echo [run] Benchmark failed, continuing...
python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 25 || echo [run] Quantization failed, continuing...
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata || echo [run] Evaluation failed, continuing...

REM 6) Verify (auto-checks + JSON receipt) - ALWAYS run
echo [run] Running verification...
python verify.py
echo [run] Done. See progress/receipt.json
