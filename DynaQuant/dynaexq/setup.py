"""
Setup script for DynaExQ
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="dynaexq",
    version="0.1.0",
    description="Dynamic Expert Quantization Runtime for MoE Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DynaQuant Team",
    author_email="support@dynaquant.ai",
    url="https://github.com/your-org/DynaQuant",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "full": [
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="moe quantization inference runtime gpu",
)
