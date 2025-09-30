# PiEdge EduKit - Makefile for easy lab execution
.PHONY: lab1 lab2 lab2b lab3 pc_all pi_bench clean help

# Default target
help:
	@echo "PiEdge EduKit - Available targets:"
	@echo "  lab1     - Train model and export to ONNX"
	@echo "  lab2     - Benchmark latency"
	@echo "  lab2b    - Quantization benchmark (bonus)"
	@echo "  lab3     - GPIO control simulation"
	@echo "  pc_all   - Run all labs (lab1 + lab2 + lab2b + lab3)"
	@echo "  pi_bench - Benchmark on Raspberry Pi"
	@echo "  clean    - Clean reports directory"
	@echo ""
	@echo "Examples:"
	@echo "  make lab1                    # Train with existing data"
	@echo "  make lab1 FAKEDATA=1         # Train with FakeData"
	@echo "  make pc_all                  # Run all labs"
	@echo "  make clean                   # Clean reports"

# Lab 1: Training and ONNX export
lab1:
	@echo "=== Lab 1: Training and ONNX Export ==="
	@if [ "$(FAKEDATA)" = "1" ]; then \
		echo "Using FakeData for training..."; \
		python -m piedge_edukit.train --fakedata --output-dir ./models; \
	else \
		echo "Using real data for training..."; \
		python -m piedge_edukit.train --data-path ./data --output-dir ./models || \
		python -m piedge_edukit.train --fakedata --output-dir ./models; \
	fi
	@echo "Running evaluation..."
	@python scripts/evaluate_onnx.py --model ./models/model.onnx --data ./data --outdir ./reports || \
	python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata --outdir ./reports || true

# Lab 2: Latency benchmark
lab2:
	@echo "=== Lab 2: Latency Benchmark ==="
	@if [ "$(FAKEDATA)" = "1" ]; then \
		echo "Using FakeData for benchmarking..."; \
		python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200; \
	else \
		echo "Using real data for benchmarking..."; \
		python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200 || \
		python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200; \
	fi

# Lab 2b: Quantization benchmark (bonus)
lab2b:
	@echo "=== Lab 2b: Quantization Benchmark (Bonus) ==="
	@if [ "$(FAKEDATA)" = "1" ]; then \
		echo "Using FakeData for quantization..."; \
		python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 50; \
	else \
		echo "Using real data for quantization..."; \
		python -m piedge_edukit.quantization --model-path ./models/model.onnx --data-path ./data --calib-size 50 || \
		python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 50; \
	fi

# Lab 3: GPIO control simulation
lab3:
	@echo "=== Lab 3: GPIO Control Simulation ==="
	@if [ "$(FAKEDATA)" = "1" ]; then \
		echo "Using FakeData for GPIO control..."; \
		python -m piedge_edukit.gpio_control --fakedata --simulate --model-path ./models/model.onnx --target "class1" --duration 5; \
	else \
		echo "Using real data for GPIO control..."; \
		python -m piedge_edukit.gpio_control --simulate --model-path ./models/model.onnx --data-path ./data --target "class1" --duration 5 || \
		python -m piedge_edukit.gpio_control --fakedata --simulate --model-path ./models/model.onnx --target "class1" --duration 5; \
	fi

# Run all labs on PC
pc_all: lab1 lab2 lab2b lab3
	@echo "=== All labs completed! ==="
	@echo "Check reports/ directory for results"
	@echo "Run 'streamlit run app.py' for dashboard view"

# Benchmark on Raspberry Pi
pi_bench:
	@echo "=== Pi Benchmark ==="
	@if [ "$(FAKEDATA)" = "1" ]; then \
		echo "Using FakeData for Pi benchmarking..."; \
		python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200; \
	else \
		echo "Using real data for Pi benchmarking..."; \
		python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200 || \
		python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200; \
	fi

# Clean reports directory
clean:
	@echo "Cleaning reports directory..."
	@rm -f reports/*.png reports/*.csv reports/*.txt reports/*.json
	@echo "Reports cleaned"

# Create synthetic dataset
synthetic:
	@echo "Creating synthetic dataset..."
	@python scripts/make_synthetic_dataset.py --root data --train-per-class 60 --val-per-class 20
	@echo "Synthetic dataset created in data/"

# Install package
install:
	@echo "Installing PiEdge EduKit..."
	@pip install -e .
	@echo "Installation complete"
