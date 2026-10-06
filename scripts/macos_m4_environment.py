"""Reuse existing conversion dependencies read-only with current MLX/tokenizers.

Load NumPy 1.x, Torch and Core ML from the existing conversion environment;
then prefer the workspace MLX environment for current Hugging Face packages.
This avoids installing or changing packages in either reference environment.
"""
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.dont_write_bytecode=True
reference=Path.home()/"more-ane-transformers/.venv/lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"
if not reference.is_dir(): raise RuntimeError(f"Missing conversion environment: {reference}")
sys.path.insert(0,str(reference))
import numpy
import torch
import coremltools
# Keep the converter packages available, after the current environment.
sys.path=[p for p in sys.path if p!=str(reference)]+[str(reference)]
