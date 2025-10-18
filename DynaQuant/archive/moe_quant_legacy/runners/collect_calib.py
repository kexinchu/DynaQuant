"""
Calibration Data Collector

Collects activations and gating affinities from MoE models for quantization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import pickle
import logging


logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Container for calibration data"""
    # Per-layer data
    layer_activations: Dict[int, torch.Tensor]  # layer_idx -> activations
    layer_affinities: Dict[int, torch.Tensor]   # layer_idx -> affinities

    # Per-expert data
    # (layer_idx, expert_id) -> activations
    expert_activations: Dict[Tuple[int, int], torch.Tensor]
    # (layer_idx, expert_id) -> affinities
    expert_affinities: Dict[Tuple[int, int], torch.Tensor]

    # Routing information
    routing_logits: Dict[int, torch.Tensor]     # layer_idx -> logits
    expert_ids: Dict[int, torch.Tensor]         # layer_idx -> expert_ids

    # Statistics
    # layer_idx -> {expert_id: count}
    expert_call_counts: Dict[int, Dict[int, int]]

    def save(self, path: str):
        """Save calibration data to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load calibration data from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class CalibrationCollector:
    """
    Collects calibration data from MoE model

    Hooks into forward pass to extract:
    - Input activations to each layer/expert
    - Gating affinities (router scores)
    - Routing decisions
    """

    def __init__(
        self,
        model_loader,
        num_samples: int = 128,
        max_seq_len: int = 512
    ):
        self.model_loader = model_loader
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len

        # Storage
        self.layer_activations = {}
        self.layer_affinities = {}
        self.expert_activations = {}
        self.expert_affinities = {}
        self.routing_logits = {}
        self.expert_ids = {}
        self.expert_call_counts = {}

        # Hooks
        self.hooks = []

    def reset(self):
        """Reset all collected data"""
        self.layer_activations = {}
        self.layer_affinities = {}
        self.expert_activations = {}
        self.expert_affinities = {}
        self.routing_logits = {}
        self.expert_ids = {}
        self.expert_call_counts = {}

    def _create_hook(self, layer_idx: int):
        """Create hook for a specific MoE layer"""

        def hook_fn(module, input, output):
            """Hook function to capture activations"""
            # Input is tuple, get first element
            x = input[0] if isinstance(input, tuple) else input

            # Store input activations
            if layer_idx not in self.layer_activations:
                self.layer_activations[layer_idx] = []

            self.layer_activations[layer_idx].append(x.detach().cpu())

            # Try to get routing information
            try:
                logits, expert_ids_selected = self.model_loader.forward_router(
                    x, layer_idx)

                # Store routing info
                if layer_idx not in self.routing_logits:
                    self.routing_logits[layer_idx] = []
                    self.expert_ids[layer_idx] = []

                self.routing_logits[layer_idx].append(logits.detach().cpu())
                self.expert_ids[layer_idx].append(
                    expert_ids_selected.detach().cpu())

                # Compute affinities (gating scores)
                weights = torch.softmax(logits, dim=-1)

                # Get affinities for selected experts
                selected_affinities = weights.gather(-1, expert_ids_selected)

                # Store affinities (sum over top-k for each token)
                token_affinities = selected_affinities.sum(
                    dim=-1)  # [batch, seq_len]

                if layer_idx not in self.layer_affinities:
                    self.layer_affinities[layer_idx] = []

                self.layer_affinities[layer_idx].append(
                    token_affinities.detach().cpu())

                # Per-expert statistics
                if layer_idx not in self.expert_call_counts:
                    self.expert_call_counts[layer_idx] = {}

                for eid in expert_ids_selected.flatten().tolist():
                    self.expert_call_counts[layer_idx][eid] = \
                        self.expert_call_counts[layer_idx].get(eid, 0) + 1

            except Exception as e:
                logger.warning(
                    f"Failed to get routing info for layer {layer_idx}: {e}")

        return hook_fn

    def register_hooks(self):
        """Register hooks on all MoE layers"""
        for layer_idx in range(self.model_loader.get_num_moe_layers()):
            moe_layer = self.model_loader.get_moe_layer(layer_idx)
            if moe_layer is not None:
                hook = moe_layer.register_forward_hook(
                    self._create_hook(layer_idx))
                self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def collect_from_dataset(
        self,
        dataset: List[str],
        batch_size: int = 1
    ) -> CalibrationData:
        """
        Collect calibration data from text dataset

        Args:
            dataset: List of text samples
            batch_size: Batch size for processing

        Returns:
            CalibrationData object
        """
        self.reset()
        self.register_hooks()

        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        model.eval()

        num_batches = (len(dataset) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Collecting calibration data"):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_texts = dataset[start_idx:end_idx]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len
                ).to(model.device)

                # Forward
                try:
                    outputs = model(**inputs)
                except Exception as e:
                    logger.warning(f"Failed to process batch {batch_idx}: {e}")
                    continue

        self.remove_hooks()

        # Concatenate collected data
        return self._package_calibration_data()

    def _package_calibration_data(self) -> CalibrationData:
        """Package collected data into CalibrationData object"""

        # Concatenate per-layer activations
        layer_acts = {}
        for layer_idx, acts_list in self.layer_activations.items():
            if acts_list:
                # Concatenate along batch dimension
                layer_acts[layer_idx] = torch.cat([
                    a.reshape(-1, a.size(-1)) for a in acts_list
                ], dim=0)

        # Concatenate per-layer affinities
        layer_affs = {}
        for layer_idx, affs_list in self.layer_affinities.items():
            if affs_list:
                layer_affs[layer_idx] = torch.cat([
                    a.reshape(-1) for a in affs_list
                ], dim=0)

        # Concatenate routing info
        routing_logits_cat = {}
        expert_ids_cat = {}
        for layer_idx in self.routing_logits:
            if self.routing_logits[layer_idx]:
                routing_logits_cat[layer_idx] = torch.cat(
                    self.routing_logits[layer_idx], dim=0
                )
            if self.expert_ids[layer_idx]:
                expert_ids_cat[layer_idx] = torch.cat(
                    self.expert_ids[layer_idx], dim=0
                )

        # Build per-expert data (group by expert ID)
        expert_acts = {}
        expert_affs = {}

        for layer_idx, acts in layer_acts.items():
            if layer_idx not in expert_ids_cat:
                continue

            expert_ids_layer = expert_ids_cat[layer_idx]  # [N, k]
            # [N * seq_len] or [N, seq_len]
            affinities_layer = layer_affs[layer_idx]

            # Flatten if needed
            if affinities_layer.dim() > 1:
                affinities_layer = affinities_layer.reshape(-1)

            # For each expert
            num_experts = self.model_loader.get_num_experts(layer_idx)
            for expert_id in range(num_experts):
                # Find tokens routed to this expert
                mask = (expert_ids_layer == expert_id).any(
                    dim=-1)  # [N, seq_len]

                if mask.sum() > 0:
                    # Select activations for this expert
                    expert_acts[(layer_idx, expert_id)] = acts[mask]
                    expert_affs[(layer_idx, expert_id)
                                ] = affinities_layer[mask]

        return CalibrationData(
            layer_activations=layer_acts,
            layer_affinities=layer_affs,
            expert_activations=expert_acts,
            expert_affinities=expert_affs,
            routing_logits=routing_logits_cat,
            expert_ids=expert_ids_cat,
            expert_call_counts=self.expert_call_counts
        )

    def collect_from_ebss(
        self,
        ebss_samples: List[str],
        batch_size: int = 1
    ) -> CalibrationData:
        """
        Collect calibration data from EBSS-generated samples

        Args:
            ebss_samples: EBSS-generated text samples
            batch_size: Batch size

        Returns:
            CalibrationData object
        """
        return self.collect_from_dataset(ebss_samples, batch_size)


def create_calibration_collector(
    model_loader,
    num_samples: int = 128,
    max_seq_len: int = 512
) -> CalibrationCollector:
    """Convenience function to create calibration collector"""
    return CalibrationCollector(model_loader, num_samples, max_seq_len)
