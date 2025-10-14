"""
MoE-Quant: W2A2 Quantization with EBSS and AGQ

This module implements extreme low-bit quantization (W2A2) for Mixture-of-Experts models
with Expert-Balanced Self-Sampling (EBSS) and Affinity-Guided Quantization (AGQ).
"""

__version__ = "0.1.0"

from .quant.ebss import EBSSSampler
from .quant.agq import AGQuantizer
from .quant.quantizers import W2A2Quantizer
from .quant.router_guard_enhanced import EnhancedRouterGuard
from .models.load_moe import MoEModelLoader
from .runners.collect_calib import CalibrationCollector
from .runners.ptq_runner import PTQRunner
from .runners.eval_metrics import MetricsEvaluator
from .qat.train_qat import QATTrainer
from .losses.routing_losses import topk_consistency_loss, margin_loss

__all__ = [
    "EBSSSampler",
    "AGQuantizer",
    "W2A2Quantizer",
    "EnhancedRouterGuard",
    "MoEModelLoader",
    "CalibrationCollector",
    "PTQRunner",
    "MetricsEvaluator",
    "QATTrainer",
    "topk_consistency_loss",
    "margin_loss",
]
