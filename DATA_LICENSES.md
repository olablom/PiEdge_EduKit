# DATA_LICENSES

## Synthetic Data

- **FakeData mode**: Generated programmatically using `torchvision.datasets.FakeData` — no third-party license terms apply.
- **Synthetic shapes dataset** (`scripts/make_synthetic_dataset.py`): Programmatically generated geometric shapes (circles and squares) — no third-party assets.

## External Dependencies

- **PyTorch models**: MobileNetV2 pretrained weights are used under their respective licenses (typically BSD-style).
- **ONNX Runtime**: Licensed under MIT License.
- **Other dependencies**: See `requirements.txt` for pinned versions and their respective licenses.

## User Data

- **No external datasets bundled**: Users must provide their own images or use synthetic data.
- **User-generated content**: Any images provided by users remain under their ownership and licensing terms.

## Third-Party Licenses Summary

- **Apache 2.0**: This project (PiEdge EduKit)
- **MIT**: ONNX Runtime, matplotlib, numpy, Pillow
- **BSD**: PyTorch, torchvision
- **Python Software Foundation License**: Python standard library components

For detailed license information, refer to the individual package documentation and license files.
