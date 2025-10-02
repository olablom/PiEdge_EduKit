#!/usr/bin/env python3
# filename: verify.py
"""
Auto-checks for the micro-lesson. Writes a JSON receipt to progress/receipt.json.
Checks:
- Required artifacts exist
- Benchmark CSV/summary parseable
- Quantization comparison CSV exists; if INT8 present, check reasonable speedup OR sensible fallback noted
- Evaluation (confusion_matrix.png + eval_summary.txt) exists
Emits: PASS/FAIL and structured metrics.
"""

import json
import re
import sys

# Python 3.12 gate
if sys.version_info[:2] != (3, 12):
    print("❌ Python 3.12 krävs. Du kör %d.%d." % sys.version_info[:2])
    print("Se README för installationssteg (Windows/macOS/Linux).")
    sys.exit(2)

from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROGRESS = ROOT / "progress"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# Verification criteria
CRITERIA = {
    "latency_p50_threshold_ms": 1.0,   # adjust as needed
    "onnx_required": True,
    "gpio_required": False,
}


def read_latency_summary(p: Path):
    """Read and parse latency summary."""
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    # naive parse
    m50 = re.search(r"p50\s*[:=]\s*([0-9.]+)", text, re.I)
    m95 = re.search(r"p95\s*[:=]\s*([0-9.]+)", text, re.I)
    mean = re.search(r"mean\s*[:=]\s*([0-9.]+)", text, re.I)
    std = re.search(r"std\s*[:=]\s*([0-9.]+)", text, re.I)
    return {
        "p50": float(m50.group(1)) if m50 else None,
        "p95": float(m95.group(1)) if m95 else None,
        "mean": float(mean.group(1)) if mean else None,
        "std": float(std.group(1)) if std else None,
    }


def read_quant_csv(p: Path):
    """Read quantization comparison CSV."""
    if not p.exists():
        return None
    rows = [
        line.strip().split(",")
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    header = [h.strip().lower() for h in rows[0]]
    data = rows[1:]
    return {"header": header, "rows": data}


def eval_exists():
    """Check if evaluation artifacts exist."""
    return (REPORTS / "confusion_matrix.png").exists() and (
        REPORTS / "eval_summary.txt"
    ).exists()


def main():
    """Main verification function."""
    PROGRESS.mkdir(exist_ok=True, parents=True)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [],
        "metrics": {},
        "pass": False,
    }

    # 1) Artifacts
    required_artifacts = [
        (MODELS / "model.onnx", "ONNX model"),
        (MODELS / "labels.json", "Labels configuration"),
        (MODELS / "preprocess_config.json", "Preprocessing configuration"),
        (REPORTS / "latency.csv", "Latency benchmark data"),
        (REPORTS / "latency_summary.txt", "Latency summary report"),
    ]

    missing_artifacts = [name for path, name in required_artifacts if not path.exists()]
    artifacts_ok = len(missing_artifacts) == 0

    result["checks"].append(
        {
            "name": "artifacts_exist",
            "ok": artifacts_ok,
            "reason": f"Missing: {', '.join(missing_artifacts)}"
            if missing_artifacts
            else "All required artifacts present",
        }
    )

    # 2) Latency summary parse
    lat = read_latency_summary(REPORTS / "latency_summary.txt")
    lat_ok = lat is not None and all(v is not None for v in lat.values())
    lat_reason = (
        "Could not parse latency metrics"
        if not lat_ok
        else "Successfully parsed latency metrics"
    )
    result["checks"].append(
        {"name": "latency_summary_parse", "ok": lat_ok, "reason": lat_reason}
    )
    if lat:
        result["metrics"]["latency"] = lat

    # 3) Quantization comparison
    quant = read_quant_csv(REPORTS / "quantization_comparison.csv")
    quant_ok = quant is not None
    quant_reason = (
        "Quantization report not found"
        if not quant_ok
        else "Quantization report exists"
    )
    result["checks"].append(
        {"name": "quantization_report_exists", "ok": quant_ok, "reason": quant_reason}
    )

    # Basic discriminative rule (if INT8 present):
    speedup_ok = None
    if quant_ok:
        # expect columns like: model, p50, p95, mean, size_mb, note
        header = quant["header"]
        rows = quant["rows"]
        try:
            idx_model = header.index("model")
            # Try different possible column names for P95
            idx_p95 = None
            for col_name in ["p95", "p95_latency_ms", "P95_Latency_ms"]:
                if col_name in header:
                    idx_p95 = header.index(col_name)
                    break
            if idx_p95 is None:
                raise ValueError("P95 column not found")
            p95 = {
                r[idx_model].strip().lower(): float(r[idx_p95])
                if r[idx_p95] != "N/A"
                else None
                for r in rows
            }
            if "fp32" in p95 and ("int8" in p95 or "INT8" in p95):
                int8_key = "int8" if "int8" in p95 else "INT8"
                if p95["fp32"] and p95[int8_key]:
                    speedup = p95["fp32"] / p95[int8_key]
                    result["metrics"]["quant_speedup_p95"] = speedup
                    speedup_ok = speedup >= 1.10  # ≥10% faster
                    speedup_reason = (
                        f"INT8 speedup: {speedup:.2f}x"
                        if speedup_ok
                        else f"INT8 speedup insufficient: {speedup:.2f}x"
                    )
                else:
                    speedup_ok = True  # fallback path accepted if noted
                    speedup_reason = "INT8 quantization failed - fallback accepted"
            else:
                speedup_ok = True  # fallback path accepted if noted
                speedup_reason = "No INT8 data found - fallback accepted"
        except Exception as e:
            speedup_ok = False
            speedup_reason = f"Error parsing quantization data: {str(e)}"
    else:
        speedup_reason = "No quantization data available"

    result["checks"].append(
        {
            "name": "quant_speedup_or_fallback",
            "ok": bool(speedup_ok),
            "reason": speedup_reason,
        }
    )

    # 4) Evaluation (confusion matrix & summary)
    eval_ok = eval_exists()
    eval_reason = (
        "Evaluation reports missing" if not eval_ok else "Evaluation reports present"
    )
    result["checks"].append(
        {"name": "evaluation_reports_exist", "ok": eval_ok, "reason": eval_reason}
    )

    # Add criteria and latency metrics
    result["criteria"] = CRITERIA
    if lat_ok and lat:
        result["metrics"]["latency_ms"] = {"p50": lat["p50"], "p95": lat["p95"]}
    
    # Baseline vs optimized comparison
    baseline = lat["p50"] if lat_ok and lat and lat["p50"] else None
    optimized = None
    int8_ok = False
    
    # Try to get INT8 latency from quantization data
    if quant_ok and quant:
        try:
            rows = quant["rows"]
            header = quant["header"]
            idx_model = next((i for i, h in enumerate(header) if "model" in h.lower()), None)
            idx_p50 = next((i for i, h in enumerate(header) if "p50" in h.lower()), None)
            
            if idx_model is not None and idx_p50 is not None:
                p50_data = {
                    r[idx_model].strip().lower(): float(r[idx_p50]) if r[idx_p50] != "N/A" else None
                    for r in rows
                }
                if "fp32" in p50_data and ("int8" in p50_data or "INT8" in p50_data):
                    int8_key = "int8" if "int8" in p50_data else "INT8"
                    if p50_data["fp32"] and p50_data[int8_key]:
                        optimized = p50_data[int8_key]
                        int8_ok = True
        except Exception:
            pass
    
    if baseline and optimized is None:
        optimized = baseline  # fallback to FP32
    
    if baseline and optimized:
        delta = optimized - baseline
        result["comparisons"] = [{
            "name": "baseline_vs_optimized_latency",
            "metric": "latency_ms_p50",
            "baseline": round(baseline, 3),
            "optimized": round(optimized, 3),
            "delta": round(delta, 3),
            "note": "INT8 fallback to FP32" if not int8_ok else "INT8 succeeded"
        }]
    
    # Discriminative thresholds
    fail_reasons = []
    if CRITERIA["onnx_required"] and not artifacts_ok:
        fail_reasons.append("ONNX export failed")
    if lat_ok and lat and lat["p50"] and lat["p50"] >= CRITERIA["latency_p50_threshold_ms"]:
        fail_reasons.append(
            f"p50 latency {lat['p50']:.3f} ms >= threshold {CRITERIA['latency_p50_threshold_ms']:.3f} ms"
        )
    
    result["fail_reasons"] = fail_reasons
    
    # PASS policy: all checks pass AND no threshold violations
    passed = all(ch["ok"] for ch in result["checks"]) and len(fail_reasons) == 0
    result["pass"] = bool(passed)

    # Write receipt & update lesson_progress
    (PROGRESS / "receipt.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    prog_path = PROGRESS / "lesson_progress.json"
    try:
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
    except Exception:
        prog = {"steps": [], "started_at": None, "completed_at": None}
    prog["steps"].append(
        {"ts": result["timestamp"], "event": "verify", "pass": result["pass"]}
    )
    if prog.get("started_at") is None:
        prog["started_at"] = result["timestamp"]
    if result["pass"]:
        prog["completed_at"] = result["timestamp"]
    prog_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")

    print("PASS" if result["pass"] else "FAIL")
    if not result["pass"]:
        for ch in result["checks"]:
            if not ch["ok"]:
                print(f"- FAILED: {ch['name']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
