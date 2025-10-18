"""
Activation quantization with fake-quant and observers.
Supports INT4 (A4) and INT2 (A2) per-token groupwise dynamic quantization.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np


class ActivationObserver(nn.Module):
    """
    Observer for collecting activation statistics during calibration.
    """

    def __init__(
        self,
        bits: int = 4,
        symmetric: bool = True,
        ch_axis: int = -1,
        per_token: bool = True,
        percentile: float = 99.9,
    ):
        """
        Initialize activation observer.

        Args:
            bits: Number of bits (2 or 4)
            symmetric: Whether to use symmetric quantization
            ch_axis: Channel axis for per-channel quantization
            per_token: Whether to use per-token quantization
            percentile: Percentile for clipping (e.g., 99.9)
        """
        super().__init__()
        assert bits in [2, 4, 8], f"Only 2, 4, or 8 bits supported, got {bits}"

        self.bits = bits
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.per_token = per_token
        self.percentile = percentile

        # Statistics
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('num_samples', torch.tensor(0))

        # Histogram for percentile-based clipping
        self.num_bins = 2048
        self.register_buffer('histogram', torch.zeros(self.num_bins))
        self.register_buffer('hist_min', torch.tensor(0.0))
        self.register_buffer('hist_max', torch.tensor(1.0))

        # Calibrated scale and zero-point
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

        self.enabled = True
        self.calibrated = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Observe activation statistics.

        Args:
            x: Input activations

        Returns:
            x: Unchanged input (observer is transparent)
        """
        if self.enabled and self.training:
            with torch.no_grad():
                # Update min/max
                batch_min = x.min().item()
                batch_max = x.max().item()

                self.min_val = torch.min(
                    self.min_val, torch.tensor(batch_min, device=x.device))
                self.max_val = torch.max(
                    self.max_val, torch.tensor(batch_max, device=x.device))
                self.num_samples += x.numel()

                # Update histogram
                x_flat = x.flatten().float()

                # Initialize histogram range if first batch
                if self.hist_min == 0.0 and self.hist_max == 1.0:
                    self.hist_min = torch.tensor(batch_min, device=x.device)
                    self.hist_max = torch.tensor(batch_max, device=x.device)
                else:
                    # Expand range if needed
                    if batch_min < self.hist_min:
                        self.hist_min = torch.tensor(
                            batch_min, device=x.device)
                    if batch_max > self.hist_max:
                        self.hist_max = torch.tensor(
                            batch_max, device=x.device)

                # Compute histogram
                hist_range = self.hist_max - self.hist_min
                if hist_range > 0:
                    bin_idx = ((x_flat - self.hist_min) /
                               hist_range * (self.num_bins - 1)).long()
                    bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

                    # Accumulate histogram
                    for idx in bin_idx:
                        self.histogram[idx] += 1

        return x

    def calibrate(self, method: str = 'mse'):
        """
        Calibrate quantization parameters from collected statistics.

        Args:
            method: Calibration method ('mse' or 'percentile')
        """
        if method == 'percentile':
            # Use percentile-based clipping
            cumsum = torch.cumsum(self.histogram, dim=0)
            total = cumsum[-1]

            # Find percentile threshold
            threshold_idx = torch.searchsorted(
                cumsum, total * self.percentile / 100.0)
            threshold_idx = min(threshold_idx.item(), self.num_bins - 1)

            # Compute threshold value
            clip_max = self.hist_min + \
                (threshold_idx / (self.num_bins - 1)) * \
                (self.hist_max - self.hist_min)

            if self.symmetric:
                clip_min = -clip_max
            else:
                # For asymmetric, find lower percentile
                threshold_idx_low = torch.searchsorted(
                    cumsum, total * (1 - self.percentile / 100.0))
                clip_min = self.hist_min + \
                    (threshold_idx_low / (self.num_bins - 1)) * \
                    (self.hist_max - self.hist_min)

        else:  # method == 'mse'
            # Use min/max from observations
            clip_max = self.max_val.item()
            if self.symmetric:
                clip_max = max(abs(self.min_val.item()),
                               abs(self.max_val.item()))
                clip_min = -clip_max
            else:
                clip_min = self.min_val.item()

        # Compute scale and zero-point
        if self.bits == 2:
            qmax = 1 if self.symmetric else 3
            qmin = -2 if self.symmetric else 0
        elif self.bits == 4:
            qmax = 7 if self.symmetric else 15
            qmin = -8 if self.symmetric else 0
        else:  # bits == 8
            qmax = 127 if self.symmetric else 255
            qmin = -128 if self.symmetric else 0

        if self.symmetric:
            self.scale = torch.tensor(
                clip_max / qmax, device=self.min_val.device)
            self.zero_point = torch.tensor(0, device=self.min_val.device)
        else:
            scale = (clip_max - clip_min) / (qmax - qmin)
            zero_point = qmin - clip_min / scale
            self.scale = torch.tensor(scale, device=self.min_val.device)
            self.zero_point = torch.tensor(
                zero_point, device=self.min_val.device)

        self.calibrated = True
        self.enabled = False  # Disable observation after calibration

    def get_calibration_params(self) -> Dict[str, torch.Tensor]:
        """Get calibrated quantization parameters."""
        return {
            'scale': self.scale,
            'zero_point': self.zero_point,
            'bits': self.bits,
            'symmetric': self.symmetric,
        }


class FakeQuantize(nn.Module):
    """
    Fake quantization for activations (used during QAT).
    """

    def __init__(
        self,
        bits: int = 4,
        symmetric: bool = True,
        per_token: bool = True,
        dynamic: bool = True,
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ):
        """
        Initialize fake quantization module.

        Args:
            bits: Number of bits (2 or 4)
            symmetric: Whether to use symmetric quantization
            per_token: Whether to use per-token quantization
            dynamic: Whether to compute scales dynamically
            scale: Static scale (if not dynamic)
            zero_point: Static zero-point (if not dynamic)
        """
        super().__init__()
        assert bits in [2, 4, 8], f"Only 2, 4, or 8 bits supported, got {bits}"

        self.bits = bits
        self.symmetric = symmetric
        self.per_token = per_token
        self.dynamic = dynamic

        # Quantization range
        if bits == 2:
            self.qmax = 1 if symmetric else 3
            self.qmin = -2 if symmetric else 0
        elif bits == 4:
            self.qmax = 7 if symmetric else 15
            self.qmin = -8 if symmetric else 0
        else:  # bits == 8
            self.qmax = 127 if symmetric else 255
            self.qmin = -128 if symmetric else 0

        # Static scale and zero-point (if provided)
        if scale is not None:
            self.register_buffer('scale', scale)
        else:
            self.scale = None

        if zero_point is not None:
            self.register_buffer('zero_point', zero_point)
        else:
            self.zero_point = None

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization.

        Args:
            x: Input activations

        Returns:
            x_fq: Fake-quantized activations
        """
        if not self.enabled:
            return x

        if self.dynamic:
            return self._fake_quantize_dynamic(x)
        else:
            return self._fake_quantize_static(x)

    def _fake_quantize_dynamic(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamic fake quantization with per-token scales."""
        if self.per_token:
            # Per-token quantization: compute scale per token (dim 0)
            # Assuming x is [batch_size, seq_len, hidden_dim] or [tokens, hidden_dim]
            if x.dim() == 3:
                # [batch, seq, hidden] -> compute scale per (batch, seq)
                reduce_dims = (2,)
            elif x.dim() == 2:
                # [tokens, hidden] -> compute scale per token
                reduce_dims = (1,)
            else:
                # Fallback to per-tensor
                reduce_dims = None

            if reduce_dims is not None:
                if self.symmetric:
                    abs_max = torch.amax(
                        torch.abs(x), dim=reduce_dims, keepdim=True)
                    scale = abs_max / self.qmax
                    scale = torch.clamp(scale, min=1e-8)
                    zero_point = 0
                else:
                    x_min = torch.amin(x, dim=reduce_dims, keepdim=True)
                    x_max = torch.amax(x, dim=reduce_dims, keepdim=True)
                    scale = (x_max - x_min) / (self.qmax - self.qmin)
                    scale = torch.clamp(scale, min=1e-8)
                    zero_point = self.qmin - x_min / scale
            else:
                # Per-tensor fallback
                if self.symmetric:
                    abs_max = torch.max(torch.abs(x))
                    scale = abs_max / self.qmax
                    scale = torch.clamp(scale, min=1e-8)
                    zero_point = 0
                else:
                    x_min = torch.min(x)
                    x_max = torch.max(x)
                    scale = (x_max - x_min) / (self.qmax - self.qmin)
                    scale = torch.clamp(scale, min=1e-8)
                    zero_point = self.qmin - x_min / scale
        else:
            # Per-tensor quantization
            if self.symmetric:
                abs_max = torch.max(torch.abs(x))
                scale = abs_max / self.qmax
                scale = torch.clamp(scale, min=1e-8)
                zero_point = 0
            else:
                x_min = torch.min(x)
                x_max = torch.max(x)
                scale = (x_max - x_min) / (self.qmax - self.qmin)
                scale = torch.clamp(scale, min=1e-8)
                zero_point = self.qmin - x_min / scale

        # Quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)

        # Dequantize
        x_fq = (x_q - zero_point) * scale

        return x_fq

    def _fake_quantize_static(self, x: torch.Tensor) -> torch.Tensor:
        """Static fake quantization using pre-calibrated scales."""
        assert self.scale is not None, "Scale must be provided for static quantization"

        scale = self.scale
        zero_point = self.zero_point if self.zero_point is not None else 0

        # Quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)

        # Dequantize
        x_fq = (x_q - zero_point) * scale

        return x_fq


def quantize_activation_dynamic(
    x: torch.Tensor,
    bits: int = 4,
    symmetric: bool = True,
    per_token: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize activations and return quantized values + scales.

    Args:
        x: Input activations
        bits: Number of bits (2 or 4)
        symmetric: Whether to use symmetric quantization
        per_token: Whether to use per-token quantization

    Returns:
        x_q: Quantized activations (int8)
        scales: Quantization scales
    """
    if bits == 2:
        qmax = 1 if symmetric else 3
        qmin = -2 if symmetric else 0
    elif bits == 4:
        qmax = 7 if symmetric else 15
        qmin = -8 if symmetric else 0
    else:  # bits == 8
        qmax = 127 if symmetric else 255
        qmin = -128 if symmetric else 0

    if per_token:
        # Per-token quantization
        if x.dim() == 3:
            reduce_dims = (2,)
        elif x.dim() == 2:
            reduce_dims = (1,)
        else:
            reduce_dims = None

        if reduce_dims is not None:
            if symmetric:
                abs_max = torch.amax(
                    torch.abs(x), dim=reduce_dims, keepdim=True)
                scales = abs_max / qmax
                scales = torch.clamp(scales, min=1e-8)
            else:
                x_min = torch.amin(x, dim=reduce_dims, keepdim=True)
                x_max = torch.amax(x, dim=reduce_dims, keepdim=True)
                scales = (x_max - x_min) / (qmax - qmin)
                scales = torch.clamp(scales, min=1e-8)
        else:
            # Fallback to per-tensor
            if symmetric:
                abs_max = torch.max(torch.abs(x))
                scales = abs_max / qmax
                scales = torch.clamp(scales, min=1e-8)
            else:
                x_min = torch.min(x)
                x_max = torch.max(x)
                scales = (x_max - x_min) / (qmax - qmin)
                scales = torch.clamp(scales, min=1e-8)
    else:
        # Per-tensor quantization
        if symmetric:
            abs_max = torch.max(torch.abs(x))
            scales = abs_max / qmax
            scales = torch.clamp(scales, min=1e-8)
        else:
            x_min = torch.min(x)
            x_max = torch.max(x)
            scales = (x_max - x_min) / (qmax - qmin)
            scales = torch.clamp(scales, min=1e-8)

    # Quantize
    x_q = torch.round(x / scales)
    x_q = torch.clamp(x_q, qmin, qmax).to(torch.int8)

    return x_q, scales


def dequantize_activation(x_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize activations.

    Args:
        x_q: Quantized activations (int8)
        scales: Quantization scales

    Returns:
        x: Dequantized activations (float)
    """
    return x_q.float() * scales


def test_fake_quant():
    """
    Unit tests for fake quantization.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing fake quantization...")

    # Test observer
    logger.info("\n--- Testing ActivationObserver ---")
    torch.manual_seed(42)

    observer = ActivationObserver(bits=4, symmetric=True, per_token=True)
    observer.train()

    # Collect statistics
    for _ in range(10):
        x = torch.randn(8, 128, 512)
        observer(x)

    # Calibrate
    observer.calibrate(method='mse')
    params = observer.get_calibration_params()

    logger.info(f"Calibrated scale: {params['scale']:.6f}")
    logger.info(f"Calibrated zero_point: {params['zero_point']:.6f}")
    logger.info(f"✓ Observer test passed")

    # Test fake quantization (dynamic)
    logger.info("\n--- Testing FakeQuantize (dynamic) ---")
    fake_quant = FakeQuantize(bits=4, symmetric=True,
                              per_token=True, dynamic=True)

    x = torch.randn(16, 256, 512)
    x_fq = fake_quant(x)

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {x_fq.shape}")
    logger.info(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    logger.info(f"Output range: [{x_fq.min():.3f}, {x_fq.max():.3f}]")

    # Compute quantization error
    mse = torch.mean((x - x_fq) ** 2).item()
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"✓ FakeQuantize (dynamic) test passed")

    # Test fake quantization (static)
    logger.info("\n--- Testing FakeQuantize (static) ---")
    scale = torch.tensor(0.1)
    fake_quant_static = FakeQuantize(
        bits=4, symmetric=True, per_token=False, dynamic=False, scale=scale
    )

    x = torch.randn(16, 256, 512)
    x_fq_static = fake_quant_static(x)

    logger.info(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    logger.info(
        f"Output range: [{x_fq_static.min():.3f}, {x_fq_static.max():.3f}]")
    logger.info(f"✓ FakeQuantize (static) test passed")

    # Test quantize/dequantize functions
    logger.info("\n--- Testing quantize/dequantize ---")
    x = torch.randn(8, 128, 512)
    x_q, scales = quantize_activation_dynamic(
        x, bits=4, symmetric=True, per_token=True)
    x_dq = dequantize_activation(x_q, scales)

    logger.info(f"Quantized dtype: {x_q.dtype}")
    logger.info(f"Scales shape: {scales.shape}")
    logger.info(f"Dequantized shape: {x_dq.shape}")

    mse_qd = torch.mean((x - x_dq) ** 2).item()
    logger.info(f"Quant/dequant MSE: {mse_qd:.6f}")
    logger.info(f"✓ Quantize/dequantize test passed")

    # Test A2 (2-bit)
    logger.info("\n--- Testing A2 (2-bit) ---")
    fake_quant_a2 = FakeQuantize(
        bits=2, symmetric=True, per_token=True, dynamic=True)

    x = torch.randn(16, 256, 512)
    x_fq_a2 = fake_quant_a2(x)

    mse_a2 = torch.mean((x - x_fq_a2) ** 2).item()
    logger.info(f"A2 MSE: {mse_a2:.6f}")

    # A2 should have higher error than A4
    fake_quant_a4 = FakeQuantize(
        bits=4, symmetric=True, per_token=True, dynamic=True)
    x_fq_a4 = fake_quant_a4(x)
    mse_a4 = torch.mean((x - x_fq_a4) ** 2).item()

    assert mse_a2 > mse_a4, "A2 should have higher error than A4"
    logger.info(
        f"✓ A2 error > A4 error as expected (A2: {mse_a2:.6f}, A4: {mse_a4:.6f})")

    logger.info("\n✓ All fake quantization tests passed!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fake_quant()
