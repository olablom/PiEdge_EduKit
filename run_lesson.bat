@echo off
setlocal

echo [run] PiEdge EduKit - One-click Bootstrap (Windows)
echo ===================================================

REM Check if Python is available
python -c "import sys; print('Python version:', sys.version)" >NUL 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Please install Python 3.12 and try again.
    pause
    exit /b 1
)

REM Run the main bootstrap script
echo [run] Starting one-click setup...
python main.py

pause
