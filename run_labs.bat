@echo off
REM PiEdge EduKit - Windows batch script for easy lab execution
REM Usage: run_labs.bat [lab1|lab2|lab2b|lab3|pc_all|pi_bench|clean|help]

if "%1"=="help" goto help
if "%1"=="lab1" goto lab1
if "%1"=="lab2" goto lab2
if "%1"=="lab2b" goto lab2b
if "%1"=="lab3" goto lab3
if "%1"=="pc_all" goto pc_all
if "%1"=="pi_bench" goto pi_bench
if "%1"=="clean" goto clean
if "%1"=="synthetic" goto synthetic
goto help

:help
echo PiEdge EduKit - Available commands:
echo   lab1     - Train model and export to ONNX
echo   lab2     - Benchmark latency
echo   lab2b    - Quantization benchmark (bonus)
echo   lab3     - GPIO control simulation
echo   pc_all   - Run all labs (lab1 + lab2 + lab2b + lab3)
echo   pi_bench - Benchmark on Raspberry Pi
echo   clean    - Clean reports directory
echo   synthetic - Create synthetic dataset
echo.
echo Examples:
echo   run_labs.bat lab1                    # Train with existing data
echo   run_labs.bat lab1 FAKEDATA=1        # Train with FakeData
echo   run_labs.bat pc_all                 # Run all labs
echo   run_labs.bat clean                  # Clean reports
goto end

:lab1
echo === Lab 1: Training and ONNX Export ===
if "%FAKEDATA%"=="1" (
    echo Using FakeData for training...
    python -m piedge_edukit.train --fakedata --output-dir ./models
) else (
    echo Using real data for training...
    python -m piedge_edukit.train --data-path ./data --output-dir ./models || python -m piedge_edukit.train --fakedata --output-dir ./models
)
echo Running evaluation...
python scripts/evaluate_onnx.py --model ./models/model.onnx --data ./data --outdir ./reports || python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --outdir ./reports
goto end

:lab2
echo === Lab 2: Latency Benchmark ===
if "%FAKEDATA%"=="1" (
    echo Using FakeData for benchmarking...
    python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200
) else (
    echo Using real data for benchmarking...
    python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200 || python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200
)
goto end

:lab2b
echo === Lab 2b: Quantization Benchmark (Bonus) ===
if "%FAKEDATA%"=="1" (
    echo Using FakeData for quantization...
    python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 50
) else (
    echo Using real data for quantization...
    python -m piedge_edukit.quantization --model-path ./models/model.onnx --data-path ./data --calib-size 50 || python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 50
)
goto end

:lab3
echo === Lab 3: GPIO Control Simulation ===
if "%FAKEDATA%"=="1" (
    echo Using FakeData for GPIO control...
    python -m piedge_edukit.gpio_control --fakedata --simulate --model-path ./models/model.onnx --target "class1" --duration 5
) else (
    echo Using real data for GPIO control...
    python -m piedge_edukit.gpio_control --simulate --model-path ./models/model.onnx --data-path ./data --target "class1" --duration 5 || python -m piedge_edukit.gpio_control --fakedata --simulate --model-path ./models/model.onnx --target "class1" --duration 5
)
goto end

:pc_all
call :lab1
call :lab2
call :lab2b
call :lab3
echo === All labs completed! ===
echo Check reports/ directory for results
echo Run 'streamlit run app.py' for dashboard view
goto end

:pi_bench
echo === Pi Benchmark ===
if "%FAKEDATA%"=="1" (
    echo Using FakeData for Pi benchmarking...
    python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200
) else (
    echo Using real data for Pi benchmarking...
    python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200 || python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200
)
goto end

:clean
echo Cleaning reports directory...
del /q reports\*.png reports\*.csv reports\*.txt reports\*.json 2>nul
echo Reports cleaned
goto end

:synthetic
echo Creating synthetic dataset...
python scripts/make_synthetic_dataset.py --root data --train-per-class 60 --val-per-class 20
echo Synthetic dataset created in data/
goto end

:end
