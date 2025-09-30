#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# quantization_cli.py - Command line interface for quantization

from piedge_edukit.quantization import QuantizationBenchmark

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--data-path", help="Path to calibration data (not needed with --fakedata)"
    )
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument(
        "--fakedata", action="store_true", help="Use FakeData for calibration"
    )
    parser.add_argument(
        "--calib-size", type=int, default=100, help="Calibration dataset size"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.fakedata and not args.data_path:
        parser.error("--data-path is required unless --fakedata is used")

    # Create quantization benchmark
    benchmark = QuantizationBenchmark(
        model_path=args.model_path,
        data_dir=args.data_path,
        output_dir=args.output_dir,
        use_fakedata=args.fakedata,
    )

    # Update calibration size
    benchmark.calibration_size = args.calib_size

    # Run quantization benchmark
    benchmark.run_quantization_benchmark()

    print(f"[OK] Quantization completed!")
    print(f"Results saved to: {args.output_dir}")
