# PiEdge EduKit v0.1.0 Release Notes

## 🎯 Overview
A **self-contained 30-minute micro-lesson** for edge ML: train a tiny image classifier → export to ONNX → benchmark latency → drive a GPIO LED with hysteresis.

## ✨ Key Features

### Core Learning Path
- **Deterministic preprocessing** + ONNX export (opset=17)
- **Latency benchmarking** with comprehensive metrics (p50/p95/mean/std)
- **INT8 static quantization** with performance comparison
- **GPIO inference** with hysteresis + debounce (simulate/real)

### Platform Support
- **Python 3.12 only** - enforced across all components
- **Cross-platform CI** - Ubuntu, Windows, macOS validation
- **Raspberry Pi ready** - aarch64 support with GPIO control

### Data Flexibility
- **No images needed** - `--fakedata` flag for quick testing
- **Synthetic datasets** - transparent data generation
- **Real image support** - flexible data layouts (`data/<class>/*` or `data/{train,val}/<class>/*`)

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd piedge_edukit
bash scripts/setup_venv.sh
source .venv/bin/activate

# Run complete lesson
bash run_lesson.sh

# Verify results
python verify.py
```

## 📊 Artifacts Generated

- `models/model.onnx` - Trained ONNX model
- `models/labels.json` - Class mapping
- `models/preprocess_config.json` - Preprocessing configuration
- `reports/latency.csv` - Detailed benchmark results
- `reports/latency_summary.txt` - Latency summary
- `reports/quantization_comparison.csv` - Quantization comparison
- `progress/receipt.json` - Automated verification receipt

## 🔧 Technical Highlights

- **Reproducible** - Fixed seeds, deterministic preprocessing
- **Self-contained** - No external data dependencies
- **CI/CD ready** - Automated testing across platforms
- **Documentation** - English primary with Swedish mirror
- **Quality gates** - Automated verification with detailed failure reasons

## 📋 Requirements

- **Python 3.12 only** (strict requirement)
- **PyTorch 2.3.1** + **ONNX Runtime 1.18.0**
- **Raspberry Pi**: 64-bit OS (aarch64) for GPIO control

## 🎓 Educational Value

Perfect for:
- **Edge ML introduction** - Complete pipeline in 30 minutes
- **ONNX workflow** - Training → export → deployment
- **Performance analysis** - Latency benchmarking methodology
- **Hardware integration** - GPIO control with hysteresis

## 🔗 Resources

- **Start here**: `index.html` - Interactive lesson guide
- **Swedish**: `README.sv.md` - Swedish documentation
- **Dashboard**: `streamlit run app.py` - Results visualization
- **CI Status**: ![CI](https://github.com/olablom/PiEdge_EduKit/actions/workflows/ci.yml/badge.svg)

---

**License**: Apache-2.0  
**Repository**: https://github.com/olablom/PiEdge_EduKit  
**Version**: v0.1.0  
**Release Date**: 2025-09-30
