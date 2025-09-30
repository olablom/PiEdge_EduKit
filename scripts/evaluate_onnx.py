#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/evaluate_onnx.py
"""
Evaluate an ONNX image classifier on a folder dataset (or FakeData) and save:
- reports/confusion_matrix.png
- reports/eval_summary.txt
Assumes labels.json + preprocess_config.json exist in ./models.
"""

import argparse
import json
import os
import sys
import itertools
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt


def load_labels(labels_path: Path):
    """Load labels from JSON file."""
    with open(labels_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Handle PiEdge EduKit format
    if "idx_to_class" in d:
        return {int(k): v for k, v in d["idx_to_class"].items()}
    else:
        # Fallback for simple format
        return {int(k): v for k, v in d.items()}


def load_preprocess(cfg_path: Path):
    """Load preprocessing configuration."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    size = int(cfg.get("image_size", 64))
    mean = np.array(cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return size, mean, std


def img_to_tensor(img: Image.Image, size, mean, std):
    """Convert PIL image to tensor."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr[None, :, :, :]  # NCHW


def gather_images(root: Path):
    """Gather images from dataset directory."""
    # supports data/val/<class>/* as priority; else data/<class>/*
    candidates = []
    val_root = root / "val"
    base = val_root if val_root.exists() else root
    if not base.exists():
        return []
    for cls_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for p in cls_dir.glob("*.*"):
            candidates.append((cls_dir.name, p))
    return candidates


def fake_dataset(n=50):
    """Generate deterministic fake dataset."""
    # deterministic colored squares vs circles like your generator; no PIL needed for gen
    rng = np.random.default_rng(42)
    labels = []
    images = []
    for i in range(n):
        lab = i % 2
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        c = 220 if lab == 0 else 120
        arr[:, :, :] = c
        images.append(Image.fromarray(arr))
        labels.append(lab)
    classes = ["class0", "class1"]
    return images, labels, classes


def confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, class_names, outpath: Path):
    """Plot confusion matrix."""
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    """Main evaluation function."""
    ap = argparse.ArgumentParser(description="Evaluate ONNX model")
    ap.add_argument("--model", required=True, help="Path to ONNX model")
    ap.add_argument("--data", default="data", help="Root dataset folder")
    ap.add_argument(
        "--fakedata", action="store_true", help="Use synthetic data instead of files"
    )
    ap.add_argument("--outdir", default="reports", help="Output dir for reports")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path("models/labels.json"))
    size, mean, std = load_preprocess(Path("models/preprocess_config.json"))
    class_names = [labels[i] for i in sorted(labels.keys())]
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    if args.fakedata:
        imgs, y_true, classes = fake_dataset(n=50)
        # override class_names if mismatch
        if len(classes) == len(class_names):
            class_names = classes
    else:
        pairs = gather_images(Path(args.data))
        if len(pairs) == 0:
            print(
                "No images found; try --fakedata or create data/val/<class>/*",
                file=sys.stderr,
            )
            sys.exit(1)
        # map class index by label name order
        cname_to_idx = {c: i for i, c in enumerate(sorted({c for c, _ in pairs}))}
        y_true = [cname_to_idx[c] for c, _ in pairs]
        imgs = [Image.open(p) for _, p in pairs]
        class_names = [c for c, _ in sorted(cname_to_idx.items(), key=lambda x: x[1])]

    y_pred = []
    for im in imgs:
        x = img_to_tensor(im, size, mean, std)
        probs = sess.run(None, {iname: x})[0]
        y_pred.append(int(np.argmax(probs, axis=1)[0]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())

    cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    plot_confusion(cm, class_names, Path(outdir / "confusion_matrix.png"))

    with open(outdir / "eval_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Support per class:\n")
        for i, name in enumerate(class_names):
            f.write(f"- {name}: {int((y_true == i).sum())}\n")
    print("Wrote reports/confusion_matrix.png and reports/eval_summary.txt")


if __name__ == "__main__":
    main()
