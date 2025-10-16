"""
MoE-Quant: Mixed Precision Quantization with RouterRank

This module implements mixed precision quantization for Mixture-of-Experts models:
- Expert layers: W2A2/W4A4 with EBSS + AGQ
- Router layers: W8A8 with RouterRank optimization
- Non-expert layers: W8A8 standard quantization
- Distributed processing across multiple GPUs
"""

__version__ = "0.2.0"

from .quant.ebss import EBSSSampler
from .quant.agq import AGQuantizer
from .quant.quantizers import W2A2Quantizer
from .quant.router_guard_enhanced import EnhancedRouterGuard
from .quant.mixed_precision_quantizer import MixedPrecisionQuantizer, MixedPrecisionConfig
from .models.load_moe import MoEModelLoader
from .runners.collect_calib import CalibrationCollector
from .runners.ptq_runner import PTQRunner
from .runners.distributed_ptq_runner import DistributedPTQRunner, launch_distributed_ptq
from .runners.run_mixed_precision_ptq import MixedPrecisionPTQRunner
from .runners.eval_metrics import MetricsEvaluator
from .qat.train_qat import QATTrainer
from .losses.routing_losses import topk_consistency_loss, margin_loss
from .losses.router_rank_loss import RouterRankLoss, router_rank_loss

__all__ = [
    # Core quantizers
    "EBSSSampler",
    "AGQuantizer",
    "W2A2Quantizer",
    "EnhancedRouterGuard",
    "MixedPrecisionQuantizer",
    "MixedPrecisionConfig",

    # Model loading
    "MoEModelLoader",

    # Runners
    "CalibrationCollector",
    "PTQRunner",
    "DistributedPTQRunner",
    "MixedPrecisionPTQRunner",
    "MetricsEvaluator",
    "QATTrainer",

    # Distributed processing
    "launch_distributed_ptq",

    # Loss functions
    "topk_consistency_loss",
    "margin_loss",
    "RouterRankLoss",
    "router_rank_loss",
]
