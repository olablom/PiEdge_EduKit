#!/usr/bin/env python3
# create_test_images.py - Create dummy test images

from PIL import Image
import numpy as np
from pathlib import Path

# Create dummy images for testing
for class_name in ["cat", "dog", "bird"]:
    class_dir = Path(f"data/{class_name}")
    class_dir.mkdir(exist_ok=True)

    # Create 5 dummy images per class
    for i in range(5):
        # Create random RGB image
        img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(class_dir / f"{class_name}_{i:02d}.jpg")

print("Test images created successfully!")
