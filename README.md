# PiEdge EduKit

![CI](https://github.com/olablom/PiEdge_EduKit/actions/workflows/ci.yml/badge.svg)

A **self-contained 30-minute micro-lesson** for edge ML: train a tiny image classifier → export to ONNX → benchmark latency → drive a GPIO LED with hysteresis.  
Primary language: **English**. Swedish mirror: **[README.sv.md](README.sv.md)**

## Quick start (Python 3.12 only)

```bash
# Create and activate venv
bash scripts/setup_venv.sh
source .venv/bin/activate      # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows

# Run the micro-lesson (self-contained, FakeData)
bash run_lesson.sh

# Auto-verify (JSON receipt)
python verify.py
# See progress/receipt.json
```

## What you'll learn

* Deterministic preprocessing + ONNX export (opset=17)
* Latency benchmarking (p50/p95/mean/std)
* (Optional) INT8 static quantization with comparison
* GPIO inference with hysteresis + debounce (simulate/real)

## Data options

* **No images needed:** `--fakedata`
* **Synthetic images:** `python scripts/make_synthetic_dataset.py --root data`
* Real images: `data/{class}/*.{jpg,png}` (see `DATA_LICENSES.md`)

## CLI (CPU is the golden path)

```bash
python -m piedge_edukit.train       --fakedata --output-dir ./models
python -m piedge_edukit.benchmark   --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200
python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx --calib-size 25
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