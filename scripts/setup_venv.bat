@echo off
REM filename: scripts/setup_venv.bat
REM Windows batch version of setup_venv.sh

set PYTHON_BIN=python3.12
if "%PYTHON_BIN%"=="" set PYTHON_BIN=python

echo [setup] Creating .venv with %PYTHON_BIN%

REM Hård Python 3.12-kontroll
%PYTHON_BIN% -c "import sys; assert sys.version_info[:2]==(3,12), f'Python 3.12 required, got {sys.version_info.major}.{sys.version_info.minor}'"
if errorlevel 1 (
    echo ERROR: Python 3.12 required
    echo Please install Python 3.12
    exit /b 1
)

echo OK: Python 3.12 confirmed
%PYTHON_BIN% -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

echo [setup] .venv ready. To activate: .venv\Scripts\activate.bat
