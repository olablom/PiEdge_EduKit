# Source Code

This directory contains the core PiEdge EduKit source code.

## Package Structure

The main package is located in `piedge_edukit/` at the project root:

- `piedge_edukit/train.py` - Training script for MobileNetV2
- `piedge_edukit/benchmark.py` - Latency benchmarking
- `piedge_edukit/quantization.py` - INT8 quantization
- `piedge_edukit/gpio_control.py` - GPIO control with hysteresis
- `piedge_edukit/preprocess.py` - Data preprocessing
- `piedge_edukit/labels.py` - Label management
- `piedge_edukit/model.py` - ONNX export utilities

## CLI Entry Points

Command-line interfaces are available as:

- `piedge-train` / `python -m piedge_edukit.train`
- `piedge-benchmark` / `python -m piedge_edukit.benchmark`
- `piedge-gpio` / `python -m piedge_edukit.gpio_control`
- `piedge-quantize` / `python -m piedge_edukit.quantization`

## Installation

Install the package in development mode:

```bash
pip install -e .
```

This makes all CLI commands available system-wide.
