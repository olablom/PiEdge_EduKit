#!/usr/bin/env python3
# setup.py - PiEdge EduKit package installation

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="piedge-edukit",
    version="0.1.0",
    author="PiEdge EduKit Team",
    description="Reproducerbart undervisningskit för edge-ML på Raspberry Pi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12,<3.13",
    entry_points={
        "console_scripts": [
            "piedge-train=piedge_edukit.train_cli:main",
            "piedge-benchmark=piedge_edukit.benchmark_cli:main",
            "piedge-gpio=piedge_edukit.gpio_cli:main",
            "piedge-quantize=piedge_edukit.quantization_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "piedge_edukit": ["*.json", "*.md"],
    },
)
