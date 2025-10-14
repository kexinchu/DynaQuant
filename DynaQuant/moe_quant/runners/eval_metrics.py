"""
Metrics Evaluator for MoE Quantization

Evaluates:
- Top-k match rate (per layer, per token)
- Perplexity
- Task accuracy
- Throughput and latency
- Memory usage
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Top-k consistency
    overall_topk_match_rate: float
    per_layer_topk_match_rate: Dict[int, float]
    per_token_topk_match_rate: List[float]

    # Perplexity
    perplexity: float

    # Task metrics (optional)
    task_accuracy: Optional[float] = None
    task_metrics: Optional[Dict] = None

    # Performance
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0

    # Fallback statistics
    fallback_rate: float = 0.0
    boundary_sample_ratio: float = 0.0

    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save metrics to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str):
        """Load metrics from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return EvaluationMetrics(**data)


class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for quantized MoE models
    """

    def __init__(
        self,
        model_loader,
        reference_model_loader=None,
        device: str = "cuda"
    ):
        self.model_loader = model_loader
        self.reference_model_loader = reference_model_loader
        self.device = device

        # Statistics
        self.topk_matches = []
        self.layer_topk_matches = {}
        self.token_topk_matches = []

    def reset_stats(self):
        """Reset all statistics"""
        self.topk_matches = []
        self.layer_topk_matches = {}
        self.token_topk_matches = []

    def evaluate_topk_consistency(
        self,
        test_texts: List[str],
        batch_size: int = 1
    ) -> Dict:
        """
        Evaluate top-k routing consistency

        Args:
            test_texts: Test text samples
            batch_size: Batch size

        Returns:
            Dictionary with consistency metrics
        """
        if self.reference_model_loader is None:
            logger.warning(
                "No reference model provided, skipping top-k consistency eval")
            return {}

        logger.info("Evaluating top-k consistency...")

        self.reset_stats()

        model_quant = self.model_loader.model
        model_ref = self.reference_model_loader.model
        tokenizer = self.model_loader.tokenizer

        model_quant.eval()
        model_ref.eval()

        num_batches = (len(test_texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Evaluating consistency"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(test_texts))
                batch_texts = test_texts[start_idx:end_idx]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                # Forward through both models
                try:
                    # Get hidden states
                    x = model_quant.model.embed_tokens(inputs["input_ids"])
                    x_ref = model_ref.model.embed_tokens(inputs["input_ids"])

                    # Compare routing decisions for each MoE layer
                    for layer_idx in range(self.model_loader.get_num_moe_layers()):
                        # Get routing decisions from both models
                        logits_quant, ids_quant = self.model_loader.forward_router(
                            x, layer_idx)
                        logits_ref, ids_ref = self.reference_model_loader.forward_router(
                            x_ref, layer_idx)

                        # Compute match rate
                        matches = (ids_quant == ids_ref).all(dim=-1).float()
                        match_rate = matches.mean().item()

                        # Store statistics
                        if layer_idx not in self.layer_topk_matches:
                            self.layer_topk_matches[layer_idx] = []

                        self.layer_topk_matches[layer_idx].append(match_rate)
                        self.topk_matches.append(match_rate)

                        # Per-token statistics
                        self.token_topk_matches.extend(
                            matches.flatten().tolist())

                except Exception as e:
                    logger.warning(
                        f"Failed to evaluate batch {batch_idx}: {e}")
                    continue

        # Compute aggregate statistics
        overall_match_rate = np.mean(self.topk_matches)

        per_layer_match_rate = {
            layer_idx: np.mean(matches)
            for layer_idx, matches in self.layer_topk_matches.items()
        }

        return {
            "overall_topk_match_rate": overall_match_rate,
            "per_layer_topk_match_rate": per_layer_match_rate,
            "per_token_topk_match_rate": self.token_topk_matches,
        }

    def evaluate_perplexity(
        self,
        test_texts: List[str],
        batch_size: int = 1
    ) -> float:
        """
        Evaluate perplexity on test set

        Args:
            test_texts: Test text samples
            batch_size: Batch size

        Returns:
            Perplexity
        """
        logger.info("Evaluating perplexity...")

        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        num_batches = (len(test_texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing perplexity"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(test_texts))
                batch_texts = test_texts[start_idx:end_idx]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                # Forward
                try:
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                    # Accumulate
                    num_tokens = (inputs["attention_mask"] == 1).sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

                except Exception as e:
                    logger.warning(
                        f"Failed to compute PPL for batch {batch_idx}: {e}")
                    continue

        perplexity = np.exp(total_loss / total_tokens)

        return perplexity

    def evaluate_latency(
        self,
        test_texts: List[str],
        num_runs: int = 10
    ) -> Dict:
        """
        Evaluate latency and throughput

        Args:
            test_texts: Test text samples
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with latency metrics
        """
        logger.info("Evaluating latency...")

        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        model.eval()

        latencies = []
        throughputs = []

        with torch.no_grad():
            for run in range(num_runs):
                # Select random sample
                text = test_texts[run % len(test_texts)]

                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True
                ).to(self.device)

                num_tokens = inputs["input_ids"].size(1)

                # Warmup
                if run == 0:
                    _ = model(**inputs)

                # Measure
                torch.cuda.synchronize()
                start_time = time.time()

                _ = model(**inputs)

                torch.cuda.synchronize()
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                throughput = num_tokens / (end_time - start_time)

                latencies.append(latency_ms)
                throughputs.append(throughput)

        return {
            "latency_ms_mean": np.mean(latencies),
            "latency_ms_std": np.std(latencies),
            "latency_ms_p50": np.percentile(latencies, 50),
            "latency_ms_p90": np.percentile(latencies, 90),
            "latency_ms_p95": np.percentile(latencies, 95),
            "latency_ms_p99": np.percentile(latencies, 99),
            "throughput_tokens_per_sec_mean": np.mean(throughputs),
            "throughput_tokens_per_sec_std": np.std(throughputs),
        }

    def evaluate_memory(self) -> Dict:
        """
        Evaluate memory usage

        Returns:
            Dictionary with memory metrics
        """
        logger.info("Evaluating memory usage...")

        if not torch.cuda.is_available():
            return {"peak_memory_mb": 0.0}

        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()

        # Run dummy forward
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer

        dummy_text = "This is a test sentence for memory measurement."
        inputs = tokenizer(dummy_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            _ = model(**inputs)

        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)

        return {
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_gb": peak_memory_mb / 1024,
        }

    def evaluate_full(
        self,
        test_texts: List[str],
        output_path: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Run full evaluation suite

        Args:
            test_texts: Test text samples
            output_path: Optional path to save results

        Returns:
            EvaluationMetrics object
        """
        logger.info("Running full evaluation...")

        # Top-k consistency
        consistency_metrics = self.evaluate_topk_consistency(test_texts)

        # Perplexity
        perplexity = self.evaluate_perplexity(test_texts)

        # Latency
        latency_metrics = self.evaluate_latency(
            test_texts[:min(10, len(test_texts))])

        # Memory
        memory_metrics = self.evaluate_memory()

        # Package results
        metrics = EvaluationMetrics(
            overall_topk_match_rate=consistency_metrics.get(
                "overall_topk_match_rate", 0.0),
            per_layer_topk_match_rate=consistency_metrics.get(
                "per_layer_topk_match_rate", {}),
            per_token_topk_match_rate=consistency_metrics.get(
                "per_token_topk_match_rate", []),
            perplexity=perplexity,
            latency_ms=latency_metrics.get("latency_ms_mean", 0.0),
            throughput_tokens_per_sec=latency_metrics.get(
                "throughput_tokens_per_sec_mean", 0.0),
            peak_memory_mb=memory_metrics.get("peak_memory_mb", 0.0),
        )

        # Save if requested
        if output_path:
            metrics.save(output_path)
            logger.info(f"Saved evaluation results to {output_path}")

        # Log summary
        logger.info(f"Evaluation Summary:")
        logger.info(f"  Perplexity: {metrics.perplexity:.2f}")
        logger.info(
            f"  Top-k Match Rate: {metrics.overall_topk_match_rate:.2%}")
        logger.info(f"  Latency: {metrics.latency_ms:.2f} ms")
        logger.info(
            f"  Throughput: {metrics.throughput_tokens_per_sec:.2f} tokens/s")
        logger.info(f"  Peak Memory: {metrics.peak_memory_mb:.2f} MB")

        return metrics


def create_evaluator(
    model_name: str,
    reference_model_name: Optional[str] = None,
    device: str = "cuda"
) -> MetricsEvaluator:
    """
    Convenience function to create metrics evaluator

    Args:
        model_name: Quantized model name
        reference_model_name: Reference (FP16) model name
        device: Device

    Returns:
        MetricsEvaluator instance
    """
    from ..models.load_moe import load_moe_model

    model_loader = load_moe_model(model_name, device=device)

    reference_loader = None
    if reference_model_name:
        reference_loader = load_moe_model(reference_model_name, device=device)

    return MetricsEvaluator(model_loader, reference_loader, device)
