#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/make_synthetic_dataset.py
"""
Generate a tiny, deterministic 2-class image dataset (64x64) on disk:
- class0: filled circle
- class1: filled square
Creates data/train and data/val with N images per class.
"""

import argparse
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw


def gen_image(kind: str, size=64, margin=8, seed=0):
    """Generate a synthetic image (circle or square)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (rng.randint(0, 20),) * 3)
    draw = ImageDraw.Draw(img)
    x0, y0 = margin, margin
    x1, y1 = size - margin, size - margin
    color = (
        rng.randint(120, 255),
        rng.randint(120, 255),
        rng.randint(120, 255),
    )
    if kind == "class0":
        draw.ellipse([x0, y0, x1, y1], fill=color)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=color)
    return img


def main():
    """Main function to generate synthetic dataset."""
    ap = argparse.ArgumentParser(
        description="Generate synthetic image dataset for PiEdge EduKit"
    )
    ap.add_argument(
        "--root", default="data", help="root folder (contains train/ and val/)"
    )
    ap.add_argument(
        "--train-per-class", type=int, default=60, help="training images per class"
    )
    ap.add_argument(
        "--val-per-class", type=int, default=20, help="validation images per class"
    )
    ap.add_argument(
        "--classes", nargs="*", default=["class0", "class1"], help="class names"
    )
    ap.add_argument("--size", type=int, default=64, help="image size (square)")
    ap.add_argument("--seed", type=int, default=1337, help="random seed")
    args = ap.parse_args()

    root = Path(args.root)

    print(f"Generating synthetic dataset...")
    print(f"  Root: {root}")
    print(f"  Classes: {args.classes}")
    print(f"  Image size: {args.size}x{args.size}")
    print(f"  Train per class: {args.train_per_class}")
    print(f"  Val per class: {args.val_per_class}")
    print(f"  Seed: {args.seed}")

    for split, n in [("train", args.train_per_class), ("val", args.val_per_class)]:
        for ci, cname in enumerate(args.classes):
            out = root / split / cname
            out.mkdir(parents=True, exist_ok=True)
            # deterministic per image
            for i in range(n):
                img = gen_image(
                    cname,
                    size=args.size,
                    seed=hash((args.seed, split, ci, i)) & 0xFFFFFFFF,
                )
                img.save(out / f"{cname}_{i:04d}.png", format="PNG")

    print(f"[OK] Dataset generated successfully!")
    print(f"  Location: {root}/(train|val)/<class>/*.png")
    print(
        f"  Total images: {(args.train_per_class + args.val_per_class) * len(args.classes)}"
    )


if __name__ == "__main__":
    main()
