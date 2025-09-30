#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# gpio_control.py - GPIO control with hysteresis for PiEdge EduKit

import time
import threading
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import argparse

from .preprocess import PreprocessConfig, validate_preprocessing_consistency
from .labels import LabelManager, validate_labels_integrity


class MockGPIO:
    """Mock GPIO for testing without hardware."""

    def __init__(self):
        self.pin_states = {}
        self.callbacks = {}
        self.simulating = True

    def setup(self, pin: int, mode: str):
        """Setup pin mode."""
        self.pin_states[pin] = {"mode": mode, "value": 0}
        print(f"MockGPIO: Pin {pin} set to {mode}")

    def output(self, pin: int, value: int):
        """Set pin output value."""
        if pin in self.pin_states:
            self.pin_states[pin]["value"] = value
            print(f"MockGPIO: Pin {pin} = {value}")

    def input(self, pin: int) -> int:
        """Get pin input value."""
        return self.pin_states.get(pin, {}).get("value", 0)

    def cleanup(self):
        """Cleanup GPIO."""
        self.pin_states.clear()
        print("MockGPIO: Cleaned up")


class HysteresisController:
    """Hysteresis controller for GPIO control."""

    def __init__(
        self,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
        debounce_time: float = 0.5,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.debounce_time = debounce_time

        self.current_state = False
        self.last_change_time = 0
        self.state_history = deque(maxlen=1000)

    def update(self, confidence: float, timestamp: float) -> bool:
        """Update hysteresis state based on confidence."""

        # Debounce check
        if timestamp - self.last_change_time < self.debounce_time:
            return self.current_state

        # Hysteresis logic
        if not self.current_state and confidence >= self.high_threshold:
            self.current_state = True
            self.last_change_time = timestamp
        elif self.current_state and confidence <= self.low_threshold:
            self.current_state = False
            self.last_change_time = timestamp

        # Record state change
        self.state_history.append(
            {
                "timestamp": timestamp,
                "confidence": confidence,
                "state": self.current_state,
            }
        )

        return self.current_state

    def get_state(self) -> bool:
        """Get current state."""
        return self.current_state

    def get_history(self) -> List[Dict]:
        """Get state history."""
        return list(self.state_history)


class GPIOController:
    """GPIO controller with hysteresis and debouncing."""

    def __init__(
        self, pin: int = 17, simulate: bool = False, output_dir: str = "reports"
    ):
        self.pin = pin
        self.simulate = simulate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GPIO
        if simulate:
            self.gpio = MockGPIO()
        else:
            try:
                import RPi.GPIO as GPIO

                self.gpio = GPIO
                self.gpio.setmode(GPIO.BCM)
                self.gpio.setup(self.pin, GPIO.OUT)
                print(f"[OK] Real GPIO initialized on pin {self.pin}")
            except ImportError:
                print("Warning: RPi.GPIO not available, falling back to MockGPIO")
                self.gpio = MockGPIO()
                self.simulate = True

        # Initialize hysteresis controller
        self.hysteresis = HysteresisController()

        # Control state
        self.running = False
        self.control_thread = None

        # Session logging
        self.session_log = []
        self.session_start_time = None

    def start_session(self):
        """Start GPIO control session."""
        self.running = True
        self.session_start_time = datetime.now()
        self.session_log = []

        print(f"GPIO control session started on pin {self.pin}")
        print(f"Simulation mode: {self.simulate}")
        print(
            f"Hysteresis thresholds: high={self.hysteresis.high_threshold}, low={self.hysteresis.low_threshold}"
        )
        print("Press Ctrl+C to stop")

    def stop_session(self):
        """Stop GPIO control session."""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

        # Turn off LED
        self.gpio.output(self.pin, 0)

        # Save session log
        self._save_session_log()

        print("GPIO control session stopped")

    def _save_session_log(self):
        """Save session log to file."""
        if not self.session_log:
            return

        # Save detailed log
        log_data = {
            "session_info": {
                "start_time": self.session_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "pin": self.pin,
                "simulation": self.simulate,
                "total_events": len(self.session_log),
            },
            "hysteresis_config": {
                "high_threshold": self.hysteresis.high_threshold,
                "low_threshold": self.hysteresis.low_threshold,
                "debounce_time": self.hysteresis.debounce_time,
            },
            "events": self.session_log,
        }

        log_path = self.output_dir / "gpio_session.txt"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        # Create history plot
        self._create_history_plot()

        print(f"[OK] Session log saved to {log_path}")

    def _create_history_plot(self):
        """Create GPIO history plot."""
        if not self.session_log:
            return

        # Extract data
        timestamps = [event["timestamp"] for event in self.session_log]
        confidences = [event["confidence"] for event in self.session_log]
        states = [event["state"] for event in self.session_log]

        # Convert timestamps to relative time
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Confidence plot
        ax1.plot(relative_times, confidences, "b-", alpha=0.7, label="Confidence")
        ax1.axhline(
            y=self.hysteresis.high_threshold,
            color="r",
            linestyle="--",
            label=f"High threshold ({self.hysteresis.high_threshold})",
        )
        ax1.axhline(
            y=self.hysteresis.low_threshold,
            color="g",
            linestyle="--",
            label=f"Low threshold ({self.hysteresis.low_threshold})",
        )
        ax1.set_ylabel("Confidence")
        ax1.set_title("GPIO Control Session - Confidence Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # State plot
        ax2.plot(relative_times, states, "r-", linewidth=2, label="GPIO State")
        ax2.set_ylabel("GPIO State (0/1)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_title("GPIO State Over Time")
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "gpio_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] History plot saved to {plot_path}")

    def update_gpio(self, confidence: float, class_name: str):
        """Update GPIO based on confidence and class."""
        if not self.running:
            return

        timestamp = time.time()

        # Update hysteresis state
        gpio_state = self.hysteresis.update(confidence, timestamp)

        # Set GPIO output
        self.gpio.output(self.pin, 1 if gpio_state else 0)

        # Log event
        event = {
            "timestamp": timestamp,
            "confidence": confidence,
            "class_name": class_name,
            "gpio_state": gpio_state,
            "state": gpio_state,
        }
        self.session_log.append(event)

        # Print status
        status = "ON" if gpio_state else "OFF"
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {class_name}: {confidence:.3f} -> GPIO {status}"
        )


class EdgeMLController:
    """Edge ML controller with GPIO integration."""

    def __init__(
        self,
        model_path: str,
        data_dir: str = None,
        gpio_pin: int = 17,
        simulate: bool = False,
        output_dir: str = "reports",
        use_fakedata: bool = False,
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir) if data_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_fakedata = use_fakedata

        # Initialize components
        self.preprocess_config = PreprocessConfig()
        self.label_manager = LabelManager()
        self.gpio_controller = GPIOController(
            pin=gpio_pin, simulate=simulate, output_dir=output_dir
        )

        # Load model
        self.session = None
        self._load_model()

        # Control state
        self.running = False

    def _load_model(self):
        """Load ONNX model."""
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            print(f"[OK] Model loaded successfully")
            print(f"  Input shape: {self.session.get_inputs()[0].shape}")
            print(f"  Output shape: {self.session.get_outputs()[0].shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference."""
        transform = self.preprocess_config.get_transform(is_training=False)

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)

        return tensor.numpy()

    def _generate_fake_image(self) -> np.ndarray:
        """Generate a fake image for testing."""
        import torch
        from torchvision import datasets, transforms

        # Create transform
        transform = self.preprocess_config.get_transform(is_training=False)

        # Generate single fake image
        fake_data = datasets.FakeData(
            size=1,
            image_size=(3, 64, 64),
            num_classes=2,
            transform=transform,
        )

        image, _ = fake_data[0]
        return image.numpy()

    def _predict_image(self, image_path: str = None) -> tuple:
        """Predict image and return class and confidence."""
        if self.use_fakedata:
            # Use fake image
            img_array = self._generate_fake_image()
        else:
            # Preprocess real image
            img_array = self._preprocess_image(image_path)

        img_batch = img_array[np.newaxis, ...]  # Add batch dimension

        # Run inference
        outputs = self.session.run(None, {"input": img_batch})
        predictions = outputs[0][0]  # Remove batch dimension

        # Get prediction
        predicted_class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        class_name = self.label_manager.get_class_name(predicted_class_idx)

        return class_name, confidence

    def start_control_session(self):
        """Start GPIO control session."""

        # Validate components
        if not validate_preprocessing_consistency():
            raise RuntimeError("Preprocessing validation failed")

        if not validate_labels_integrity():
            raise RuntimeError("Labels validation failed")

        # Start GPIO controller
        self.gpio_controller.start_session()
        self.running = True

        print("Edge ML GPIO control started!")
        print("Monitoring images for GPIO control...")

    def stop_control_session(self):
        """Stop GPIO control session."""
        self.running = False
        self.gpio_controller.stop_session()
        print("Edge ML GPIO control stopped!")

    def process_image(self, image_path: str):
        """Process single image and update GPIO."""
        if not self.running:
            print("Control session not started")
            return

        try:
            class_name, confidence = self._predict_image(image_path)
            self.gpio_controller.update_gpio(confidence, class_name)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    def run_continuous_monitoring(
        self, target_class: str, interval: float = 1.0, duration: float = 10.0
    ):
        """Run continuous monitoring of images."""

        self.start_control_session()

        try:
            if self.use_fakedata:
                # Use fake data for monitoring
                print(f"[INFO] Running continuous monitoring with fake data")
                print(f"Target class: {target_class}")
                print(f"Duration: {duration} seconds")
                print(f"Interval: {interval} seconds")

                start_time = time.time()
                while time.time() - start_time < duration:
                    class_name, confidence = self._predict_image()
                    self.gpio_controller.update_gpio(confidence, class_name)
                    time.sleep(interval)
            else:
                # Find images to monitor
                image_files = []
                extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

                for img_path in Path(self.data_dir).rglob("*"):
                    if img_path.suffix.lower() in extensions:
                        image_files.append(img_path)

                if not image_files:
                    print(f"No images found in {self.data_dir}")
                    return

                print(f"Found {len(image_files)} images to monitor")
                print(f"Target class: {target_class}")
                print(f"Duration: {duration} seconds")

                # Continuous monitoring loop with timeout
                start_time = time.time()
                while self.running and (time.time() - start_time) < duration:
                    for img_path in image_files:
                        if not self.running or (time.time() - start_time) >= duration:
                            break

                        self.process_image(str(img_path))
                        time.sleep(interval)

                    if not self.running or (time.time() - start_time) >= duration:
                        break

                    print("Completed monitoring cycle, restarting...")
                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        finally:
            self.stop_control_session()


def main():
    parser = argparse.ArgumentParser(description="Edge ML GPIO Control")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--data-path", help="Path to test data (not needed with --fakedata)"
    )
    parser.add_argument("--gpio-pin", type=int, default=17, help="GPIO pin number")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Use MockGPIO (default: True)",
    )
    parser.add_argument(
        "--no-simulate",
        action="store_false",
        dest="simulate",
        help="Use real GPIO (requires RPi.GPIO)",
    )
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Monitoring interval"
    )
    parser.add_argument(
        "--high-threshold", type=float, default=0.7, help="High threshold"
    )
    parser.add_argument(
        "--low-threshold", type=float, default=0.3, help="Low threshold"
    )
    parser.add_argument(
        "--debounce-time", type=float, default=0.5, help="Debounce time"
    )
    parser.add_argument("--target", required=True, help="Target class name")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Monitoring duration in seconds"
    )
    parser.add_argument(
        "--fakedata", action="store_true", help="Use FakeData instead of real images"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.fakedata and not args.data_path:
        parser.error("--data-path is required unless --fakedata is used")

    # Create controller
    controller = EdgeMLController(
        model_path=args.model_path,
        data_dir=args.data_path,
        gpio_pin=args.gpio_pin,
        simulate=args.simulate,
        output_dir=args.output_dir,
        use_fakedata=args.fakedata,
    )

    # Update hysteresis parameters
    controller.gpio_controller.hysteresis.high_threshold = args.high_threshold
    controller.gpio_controller.hysteresis.low_threshold = args.low_threshold
    controller.gpio_controller.hysteresis.debounce_time = args.debounce_time

    # Run continuous monitoring
    controller.run_continuous_monitoring(args.target, args.interval, args.duration)


if __name__ == "__main__":
    main()
