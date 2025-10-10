#!/usr/bin/env bash
set -euo pipefail

# 0) Miljö
bash scripts/setup_venv.sh
# shellcheck disable=SC1091
source .venv/Scripts/activate || source .venv/bin/activate || true
PYVER=$(python - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
[ "$PYVER" = "3.13" ] && { echo "Python 3.13 unsupported. Use 3.12."; exit 1; }
:
 "${MIN_SPEEDUP_PCT:=0}"
pip install -e .

mkdir -p artifacts reports progress

# 1) Exportera FP32 + test-split
python -m piedge_edukit.prepare_model --out-fp32 artifacts/model_fp32.onnx --calib-npz artifacts/calib.npz --seed 42

# 2) Kvantisering (NPZ-kalibrering) – QLinear (QOperator)
python - <<'PY'
import numpy as np, onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
class NpzBatchReader:
    def __init__(self, model_path, npz_path, key='X_te', samples=512, bs=32):
        self.input_name = onnx.load(model_path).graph.input[0].name
        X = np.load(npz_path)[key][:samples].astype('float32')
        if X.ndim == 1: X = X.reshape(-1,64)
        elif X.ndim > 2: X = X.reshape(X.shape[0], -1)
        self.it = iter([{self.input_name: X[i:i+bs]} for i in range(0, len(X), bs)])
    def get_next(self): return next(self.it, None)

dr = NpzBatchReader('artifacts/model_fp32.onnx', 'artifacts/calib.npz', 'input', 512, 32)
quantize_static(
    model_input='artifacts/model_fp32.onnx',
    model_output='artifacts/model_int8.onnx',
    calibration_data_reader=dr,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    quant_format=QuantFormat.QOperator,
)
print('✅ INT8 saved: artifacts/model_int8.onnx')
PY

# 3) Benchmark (fakedata)
python -m piedge_edukit.benchmark \
  --model-path artifacts/model_fp32.onnx \
  --fakedata --warmup 10 --runs 201 \
  --save-summary-as reports/latency_summary_fp32.txt

python -m piedge_edukit.benchmark \
  --model-path artifacts/model_int8.onnx \
  --fakedata --warmup 10 --runs 201 \
  --save-summary-as reports/latency_summary_int8.txt

# 4) Accuracy (paketets evaluator; Windows-vänlig, ingen jq)
python -m piedge_edukit.evaluate_accuracy \
  --fp32 artifacts/model_fp32.onnx \
  --int8 artifacts/model_int8.onnx \
  --input-npz artifacts/calib.npz \
  --npz-key input \
  --report reports/accuracy.json \
  --summary reports/accuracy_summary.txt \
  --max-mae "${MAX_MAE:-0.02}"

# 4b) Skriv verify-vänliga JSON-filer (robust mot CRLF och banners)
python - <<'PY'
import re, json
from pathlib import Path

def extract_mean_ms(txt: str) -> float:
    patterns = [
        r"(?im)Mean\s+latency\s*:\s*([0-9]*\.?[0-9]+)\s*ms\b",
        r"(?im)Mean\s+latency\s*\(ms\)\s*:\s*([0-9]*\.?[0-9]+)\b",
        r"(?im)^\s*P50\s*:\s*([0-9]*\.?[0-9]+)\s*$",
        r"(?im)\bmedian_ms\s*:\s*([0-9]*\.?[0-9]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            return float(m.group(1))
    raise SystemExit("Could not find latency value in summary text")

# Latency -> JSON
fp32_txt = Path("reports/latency_summary_fp32.txt").read_text(encoding="utf-8", errors="ignore").replace("\r","")
int8_txt = Path("reports/latency_summary_int8.txt").read_text(encoding="utf-8", errors="ignore").replace("\r","")
fp32_ms = extract_mean_ms(fp32_txt)
int8_ms = extract_mean_ms(int8_txt)
Path("reports/latency_summary_fp32.json").write_text(json.dumps({"mean_ms": fp32_ms}), encoding="utf-8")
Path("reports/latency_summary_int8.json").write_text(json.dumps({"mean_ms": int8_ms}), encoding="utf-8")

# Accuracy -> verify-vänlig JSON
acc_txt = Path("reports/accuracy_summary.txt").read_text(encoding="utf-8", errors="ignore").replace("\r","")
mae = float(re.search(r"^MAE:\s*([0-9.]+)", acc_txt, re.I|re.M).group(1))
thr = float(re.search(r"^Threshold.*:\s*([0-9.]+)", acc_txt, re.I|re.M).group(1))
passed = re.search(r"^RESULT:\s*(PASS|FAIL)", acc_txt, re.I|re.M).group(1).upper() == "PASS"

Path("reports/accuracy_for_verify.json").write_text(
    json.dumps({"mae": mae, "threshold": thr, "pass": passed}), encoding="utf-8"
)

print("Saved verify JSONs:",
      "\n - reports/latency_summary_fp32.json",
      "\n - reports/latency_summary_int8.json",
      "\n - reports/accuracy_for_verify.json")
PY

# 5) Verifiering – speedup-krav via env (default -100)
: "${MIN_SPEEDUP_PCT:=-100}"
python -m piedge_edukit.verify \
  --lat-fp32 reports/latency_summary_fp32.json \
  --lat-int8 reports/latency_summary_int8.json \
  --acc-json reports/accuracy_for_verify.json \
  --receipt progress/receipt.json \
  --progress progress/receipt.json \
  --min-speedup-pct "$MIN_SPEEDUP_PCT"

echo
echo "=========== RECEIPT (progress/receipt.json) ==========="
cat progress/receipt.json || true
echo "======================================================="
