#!/usr/bin/env python3
"""
PiEdge EduKit - ONNX Model Evaluator
Evaluates ONNX models and compares with PyTorch models
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path

def evaluate_onnx_model(onnx_path, pytorch_path=None):
    """Evaluate ONNX model and optionally compare with PyTorch model"""
    print(f"🔍 Evaluating ONNX model: {onnx_path}")
    
    if not Path(onnx_path).exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return False
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"✅ ONNX model loaded successfully")
        print(f"   Input: {input_name} {session.get_inputs()[0].shape}")
        print(f"   Output: {output_name} {session.get_outputs()[0].shape}")
        
        # Test with dummy data
        dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
        
        # ONNX inference
        onnx_outputs = session.run([output_name], {input_name: dummy_input})
        onnx_output = onnx_outputs[0]
        
        print(f"✅ ONNX inference successful: {onnx_output.shape} {onnx_output.dtype}")
        
        # Compare with PyTorch if available
        if pytorch_path and Path(pytorch_path).exists():
            print(f"🔄 Comparing with PyTorch model: {pytorch_path}")
            
            # Load PyTorch model
            pytorch_model = torch.load(pytorch_path, map_location='cpu')
            pytorch_model.eval()
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_input = torch.from_numpy(dummy_input)
                pytorch_output = pytorch_model(pytorch_input).numpy()
            
            # Compare outputs
            diff = np.abs(onnx_output - pytorch_output).max()
            print(f"📊 Max difference: {diff:.6f}")
            
            if diff < 1e-5:
                print("✅ ONNX and PyTorch outputs match!")
            else:
                print("⚠️  ONNX and PyTorch outputs differ significantly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error evaluating ONNX model: {e}")
        return False

def main():
    print("🔍 PiEdge EduKit - ONNX Model Evaluator")
    print("=" * 50)
    
    # Check for models
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found. Run training first.")
        sys.exit(1)
    
    onnx_model = models_dir / "model.onnx"
    pytorch_model = models_dir / "model_best.pth"
    
    if not onnx_model.exists():
        print("❌ ONNX model not found. Run training first.")
        sys.exit(1)
    
    # Evaluate model
    if evaluate_onnx_model(onnx_model, pytorch_model):
        print("\n🎉 ONNX model evaluation completed!")
    else:
        print("\n⚠️  ONNX model evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
