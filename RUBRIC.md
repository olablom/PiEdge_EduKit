# Instructor Rubric (PiEdge EduKit Micro-Lesson)

**Scale**: 0 = Missing / incorrect • 1 = Partially correct • 2 = Correct & well-reasoned  
**Target score for "Pass"**: ≥ 12 / 16 (with no 0 on Safety/Reproducibility)

| Area                                    | What to check                                                                                                | Evidence (where)                                       | Score (0–2)    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | -------------- |
| **A1. Model implementation**            | TinyCNN implemented correctly (layers, shapes). Uses `eval()` for export.                                    | `01_training_and_export.ipynb` cells; ONNX export cell | 0 1 2          |
| **A2. Training loop**                   | Proper loop: zero_grad → forward → loss → backward → step; accuracy computed sensibly.                       | 01 notebook training cell output                       | 0 1 2          |
| **A3. ONNX export & validation**        | ONNX exported with opset ≥ 17, dynamic batch axis, and successfully loaded/checked.                          | 01 notebook export + check cell                        | 0 1 2          |
| **B1. Latency methodology**             | Warm-up rationale explained; latencies measured with steady-state; P50/P95 reported.                         | `02_latency_benchmark.ipynb` plots + text              | 0 1 2          |
| **B2. Analysis quality**                | Interprets P50 vs P95, batch-size trade-offs, and EP implications (CPU assumed).                             | 02 notebook analysis cells                             | 0 1 2          |
| **C1. Quantization workflow**           | Calibration data uses same preprocessing; runs CLI; handles INT8 fail gracefully (FP32 fallback acceptable). | `03_quantization.ipynb` + quant summary                | 0 1 2          |
| **C2. Trade-off reflection**            | Explains why INT8 may fail / regress and when to stick with FP32.                                            | 03 notebook reflection                                 | 0 1 2          |
| **D1. Evaluation & artifacts**          | Confusion matrix produced; verify passes; receipt/progress JSON present.                                     | `04_evaluate_and_verify.ipynb` + `progress/`           | 0 1 2          |
| **E1. Safety & reproducibility (gate)** | Python 3.12 venv used; commands reproducible; no hard-coded secrets.                                         | index.html steps + `run_lesson.sh` + `.env.example`    | **Gate** ✅/❌ |

## Scoring guidance

- **2** = Fully correct + clear reasoning / tidy outputs.
- **1** = Works but has gaps (minor mistakes or weak explanation).
- **0** = Missing, incorrect, or not executed.

## Common deductions (−1 each, cap −2 per area)

- Messy outputs not re-run (stale errors/warnings left in final save).
- Plots/metrics generated but never interpreted in text.
- Ignoring failing INT8 without noting fallback rationale.

## How to use during review

1. Open `00 → 04` in order; look for green cells and the final **PASS**.
2. Skim the **hints/solutions usage** — students should attempt tasks first; solutions are acceptable if they explain why.
3. Fill the rubric table; ensure **E1 gate** is ✅ (otherwise the submission is not compliant).

## Student feedback snippets (ready to paste)

- **Great methods**: "You justified warm-up and P95 impact clearly. Nice steady-state design!"
- **Needs clarity**: "You measured latencies, but please explain **why** P95 matters on-device."
- **Quantization tip**: "Good FP32 fallback. Next time, show a quick accuracy check to confirm no regression."

---

## Quick Reference

### Expected artifacts

- `models/model.onnx` (or `models/student_model.onnx`)
- `reports/latency_summary.txt` + `latency.csv`
- `reports/quantization_summary.txt` (or fallback note)
- `reports/eval_summary.txt` + `confusion_matrix.png`
- `progress/receipt.json` with `"pass": true`

### Key learning outcomes

- **CNN architecture**: Conv→ReLU→Pool pattern, proper shapes
- **Training**: Standard PyTorch loop with metrics
- **ONNX export**: Cross-platform deployment format
- **Benchmarking**: Warm-up, percentiles, batch size effects
- **Quantization**: FP32→INT8 trade-offs, calibration importance
- **Verification**: Automated quality gates for ML pipelines

### Red flags

- No Python 3.12 enforcement
- Missing `verify.py` run or FAIL status
- Hard-coded paths or secrets
- No explanation of methodology choices
