@echo off
REM demo_pretty.bat - Generate pretty training curves and confusion matrix
REM Run from repo root: scripts\demo_pretty.bat

echo ========================================
echo PiEdge EduKit - Pretty Demo Run
echo ========================================

echo.
echo [1/5] Cleaning old artifacts...
if exist models rmdir /s /q models
if exist reports rmdir /s /q reports  
if exist progress rmdir /s /q progress
mkdir reports
mkdir progress

echo.
echo [2/5] Training with 5 epochs for nice curves...
python -m piedge_edukit.train --fakedata --no-pretrained --epochs 5 --batch-size 16 --output-dir ./models
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [3/5] Benchmarking with Pretty Demo settings...
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200 --providers CPUExecutionProvider
if errorlevel 1 (
    echo ERROR: Benchmarking failed!
    pause
    exit /b 1
)

echo.
echo [4/5] Evaluating with 200 samples for stable confusion matrix...
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 200
if errorlevel 1 (
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo [5/5] Opening generated images...
start "" "reports\training_curves.png"
start "" "reports\confusion_matrix.png"

echo.
echo ========================================
echo Demo completed! Check the opened images.
echo ========================================
pause
