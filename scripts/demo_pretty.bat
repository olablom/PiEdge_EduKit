@echo off
REM demo_pretty.bat - Generate pretty training curves and confusion matrix
REM Run from repo root: scripts\demo_pretty.bat

echo ========================================
echo PiEdge EduKit - Pretty Demo Run
echo ========================================

echo.
echo [1/4] Cleaning old artifacts...
if exist models rmdir /s /q models
if exist reports rmdir /s /q reports  
if exist progress rmdir /s /q progress
mkdir reports
mkdir progress

echo.
echo [2/4] Training with 5 epochs for nice curves...
python -m piedge_edukit.train --fakedata --no-pretrained --epochs 5 --batch-size 64 --output-dir ./models
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Evaluating with 200 samples for stable confusion matrix...
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --limit 200
if errorlevel 1 (
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Opening generated images...
start "" "reports\training_curves.png"
start "" "reports\confusion_matrix.png"

echo.
echo ========================================
echo Demo completed! Check the opened images.
echo ========================================
pause
