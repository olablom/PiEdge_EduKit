#!/usr/bin/env python3
"""
PiEdge EduKit - Lab 01 Tests
Tests for the training and export functionality
"""

import pytest
import torch
import numpy as np
from pathlib import Path

def test_tinycnn_shape():
    """Test that TinyCNN produces correct output shape"""
    from piedge_edukit.model import TinyCNN
    
    model = TinyCNN(num_classes=2)
    model.eval()
    
    # Test input
    dummy_input = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

def test_onnx_exists_and_loads():
    """Test that ONNX model exists and can be loaded"""
    onnx_path = Path("models/model.onnx")
    
    if not onnx_path.exists():
        pytest.skip("ONNX model not found. Run training first.")
    
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    assert model is not None, "ONNX model should load successfully"

def test_onnxruntime_infer_shape_dtype():
    """Test that ONNX Runtime can run inference with correct shape/dtype"""
    onnx_path = Path("models/model.onnx")
    
    if not onnx_path.exists():
        pytest.skip("ONNX model not found. Run training first.")
    
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test input
    dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    
    # Run inference
    outputs = session.run([output_name], {input_name: dummy_input})
    output = outputs[0]
    
    assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
    assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"

if __name__ == "__main__":
    pytest.main([__file__])
