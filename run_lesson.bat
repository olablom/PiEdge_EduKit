@echo off
REM PiEdge EduKit - Lesson Runner (Windows)
REM Sets up environment and runs the complete lesson

echo 🎓 PiEdge EduKit - Lesson Runner
echo =================================

REM Check if we're in the right directory
if not exist "main.py" (
    echo ❌ Error: Please run this script from the PiEdge EduKit root directory
    exit /b 1
)

REM Run setup
echo 🔧 Setting up environment...
bash scripts/setup_venv.sh

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the main script
echo 🚀 Starting lesson...
python main.py
