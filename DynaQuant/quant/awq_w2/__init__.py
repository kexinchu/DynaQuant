"""
AWQ W2A16 Quantization Module
==============================
Weight-only 2-bit quantization with AWQ-style activation-aware calibration.

This module provides:
- 2-bit symmetric per-group quantization (4 weights per byte)
- AWQ activation-aware calibration
- Efficient packing/unpacking
- Runtime inference module (W2AWQLinear)
"""

from .pack import pack_2bit, unpack_2bit
from .quantize import quantize_weight_w2, symmetric_quantize, dequantize_weight
from .calib import collect_activations, search_scale_alpha, calibrate_layer
from .runtime import W2AWQLinear

__all__ = [
    'pack_2bit',
    'unpack_2bit',
    'quantize_weight_w2',
    'symmetric_quantize',
    'dequantize_weight',
    'collect_activations',
    'search_scale_alpha',
    'calibrate_layer',
    'W2AWQLinear',
]

__version__ = '1.0.0'
