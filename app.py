#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# app.py - Streamlit dashboard for PiEdge EduKit artifacts

import streamlit as st
import pandas as pd
from pathlib import Path
import json

st.set_page_config(page_title="PiEdge EduKit", layout="centered")
st.title("PiEdge EduKit — Results Dashboard")

reports = Path("reports")
models = Path("models")

# Sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page", ["Overview", "Training", "Evaluation", "Benchmark", "Quantization"]
)

if page == "Overview":
    st.header("Project Overview")

    st.subheader("Model Files")
    if (models / "model.onnx").exists():
        st.success("✅ model.onnx")
        st.write(
            f"Size: {(models / 'model.onnx').stat().st_size / (1024 * 1024):.1f} MB"
        )
    else:
        st.error("❌ model.onnx not found")

    if (models / "labels.json").exists():
        st.success("✅ labels.json")
        with open(models / "labels.json") as f:
            labels = json.load(f)
        st.write("Classes:", list(labels.values()))
    else:
        st.error("❌ labels.json not found")

    if (models / "preprocess_config.json").exists():
        st.success("✅ preprocess_config.json")
    else:
        st.error("❌ preprocess_config.json not found")

elif page == "Training":
    st.header("Training Results")

    # Training metrics
    tm = reports / "train_metrics.csv"
    if tm.exists():
        df = pd.read_csv(tm)
        st.subheader("Training Curves")

        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df[["epoch", "train_loss"]].set_index("epoch"))
            st.caption("Training Loss")

        with col2:
            if "val_loss" in df.columns:
                st.line_chart(df[["epoch", "val_loss"]].set_index("epoch"))
                st.caption("Validation Loss")

        if "val_acc" in df.columns:
            st.line_chart(df[["epoch", "val_acc"]].set_index("epoch"))
            st.caption("Validation Accuracy")
    else:
        st.info(
            "No train_metrics.csv found. Training metrics will be logged automatically."
        )

elif page == "Evaluation":
    st.header("Model Evaluation")

    # Confusion Matrix
    cm = reports / "confusion_matrix.png"
    if cm.exists():
        st.subheader("Confusion Matrix")
        st.image(str(cm))
    else:
        st.info("No confusion_matrix.png found. Run evaluation:")
        st.code(
            "python scripts/evaluate_onnx.py --model ./models/model.onnx --fakedata"
        )

    # Evaluation Summary
    es = reports / "eval_summary.txt"
    if es.exists():
        st.subheader("Evaluation Summary")
        st.code(es.read_text())
    else:
        st.info("No eval_summary.txt found.")

elif page == "Benchmark":
    st.header("Latency Benchmark")

    # Latency Summary
    ls = reports / "latency_summary.txt"
    if ls.exists():
        st.subheader("Latency Results")
        st.code(ls.read_text())
    else:
        st.info("No latency_summary.txt found. Run benchmark:")
        st.code(
            "python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx"
        )

    # Latency Plot
    lp = reports / "latency_plot.png"
    if lp.exists():
        st.subheader("Latency Distribution")
        st.image(str(lp))
    else:
        st.info("No latency_plot.png found.")

elif page == "Quantization":
    st.header("Quantization Results")

    # Quantization Comparison
    qc = reports / "quantization_comparison.csv"
    if qc.exists():
        st.subheader("Quantization Comparison")
        df = pd.read_csv(qc)
        st.dataframe(df)

        # Show size comparison
        if len(df) > 1:
            st.subheader("Size Comparison")
            size_data = df[["Model", "File_Size_MB"]].set_index("Model")
            st.bar_chart(size_data)
    else:
        st.info("No quantization comparison found. Run quantization:")
        st.code(
            "python -m piedge_edukit.quantization --fakedata --model-path ./models/model.onnx"
        )

    # Quantization Plot
    qp = reports / "quantization_comparison.png"
    if qp.exists():
        st.subheader("Quantization Analysis")
        st.image(str(qp))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Commands")
st.sidebar.code("""
# Generate synthetic data
python scripts/make_synthetic_dataset.py --root data

# Run all labs
make pc_all

# Or with FakeData
make pc_all FAKEDATA=1
""")

st.sidebar.markdown("### Tips")
st.sidebar.info("""
- Use `--fakedata` flags for quick testing without images
- Check `reports/` directory for all generated artifacts
- Run `make clean` to clear reports
""")
