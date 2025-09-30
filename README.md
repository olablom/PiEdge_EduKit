# PiEdge EduKit

![CI](https://github.com/olablom/PiEdge_EduKit/actions/workflows/ci.yml/badge.svg)

**Start here → [`index.html`](index.html)** | Swedish: **[README.sv.md](README.sv.md)**

A **self-contained 30-minute micro-lesson** for edge ML: train a tiny image classifier → export to ONNX → benchmark latency → drive a GPIO LED with hysteresis.

> **Prerequisites (hard requirement)**  
> - **Python 3.12.x** (inte 3.11, inte 3.13)  
> - Git Bash (Windows) eller bash (macOS/Linux)
> - 3–4 GB ledigt disk-utrymme
>
> **Snabbinstallation**  
> **Windows (Git Bash):**
> ```bash
> winget install --id Python.Python.3.12 -e
> # öppna NY Git Bash efter installation
> ```
> **macOS (Homebrew):**
> ```bash
> brew install python@3.12
> echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```
> **Ubuntu:**
> ```bash
> sudo add-apt-repository ppa:deadsnakes/ppa -y
> sudo apt update && sudo apt install -y python3.12 python3.12-venv
> ```

## Quick start (Python 3.12 only)

```bash
# Create and activate venv
bash scripts/setup_venv.sh
source .venv/bin/activate      # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows

# Run the micro-lesson (self-contained, FakeData)
bash run_lesson.sh              # Linux/macOS
# or: run_labs.bat              # Windows

# Auto-verify (JSON receipt)
python verify.py
# See progress/receipt.json
```

## What you'll learn

- Deterministic preprocessing + ONNX export (opset=17)
- Latency benchmarking (p50/p95/mean/std)
- (Optional) INT8 static quantization with comparison
- GPIO inference with hysteresis + debounce (simulate/real)

## Data options

The pipeline accepts both layouts:

- **Flat structure:** `data/<class>/*.{jpg,png}`
- **Train/val structure:** `data/{train,val}/<class>/*.{jpg,png}`

**No images needed:** Use `--fakedata` flag for quick testing.

## CLI commands

```bash
# Training
python -m piedge_edukit.train --fakedata --output-dir ./models

# Benchmarking
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200

# Quantization (optional)
python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 25

# Evaluation
python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata
```

## Raspberry Pi (aarch64)

```bash
sudo bash pi_setup/install_pi_requirements.sh
python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200
python -m piedge_edukit.gpio_control --no-simulate --model-path ./models/model.onnx --data-path ./data --target class1 --duration 10
```

## Dashboard (optional)

```bash
streamlit run app.py
```

## Structure

```
index.html  run_lesson.sh  verify.py  scripts/  notebooks/  src/
requirements.txt  progress/  LICENSE  DATA_LICENSES.md  .env.example
piedge_edukit/  models/  reports/  pi_setup/  data/
```

## License

Apache-2.0. See `LICENSE`.
