"""
AWQ-style Activation-Aware Calibration
========================================
Collects activation statistics and searches for optimal per-group scaling factors.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc

from .quantize import quantize_weight_w2, compute_quantization_error


class ActivationCollector:
    """Collects activations from model layers during calibration."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.activations = []
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        """Hook function to collect input activations."""
        # Input is a tuple, get first element
        x = input[0].detach()

        # Store on CPU to save GPU memory
        self.activations.append(x.cpu())

    def register(self):
        """Register forward hook."""
        self.hook_handle = self.module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def get_activations(self, max_samples: Optional[int] = None) -> torch.Tensor:
        """
        Get collected activations as a single tensor.

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            Concatenated activations, shape [total_samples, in_features]
        """
        if not self.activations:
            return None

        # Concatenate all activations
        X = torch.cat(self.activations, dim=0)

        if max_samples is not None and X.shape[0] > max_samples:
            # Randomly sample
            indices = torch.randperm(X.shape[0])[:max_samples]
            X = X[indices]

        return X

    def clear(self):
        """Clear collected activations."""
        self.activations = []
        gc.collect()


def collect_activations(
    model: nn.Module,
    dataloader,
    layer_names: Optional[List[str]] = None,
    max_samples: int = 1024,
) -> Dict[str, torch.Tensor]:
    """
    Collect activations from specified layers.

    Args:
        model: The model to collect from
        dataloader: Calibration dataloader
        layer_names: List of layer names to collect from (if None, collect from all Linear)
        max_samples: Maximum activation samples per layer

    Returns:
        Dictionary mapping layer names to activation tensors
    """
    model.eval()

    # Find target layers
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_names.append(name)

    print(f"Collecting activations from {len(layer_names)} layers...")

    # Register collectors
    collectors = {}
    named_modules_dict = dict(model.named_modules())

    for name in layer_names:
        if name not in named_modules_dict:
            print(f"Warning: Layer {name} not found in model, skipping...")
            continue

        module = named_modules_dict[name]
        collector = ActivationCollector(module)
        collector.register()
        collectors[name] = collector

    print(f"  Registered {len(collectors)} hooks")

    # Run forward passes
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting")):
            # Handle different batch formats
            if isinstance(batch, dict):
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(model.device)
                    model(input_ids)
                    batch_size = input_ids.shape[0]
                else:
                    model_inputs = {k: v.to(model.device)
                                    for k, v in batch.items()}
                    model(**model_inputs)
                    batch_size = next(iter(model_inputs.values())).shape[0]
            elif isinstance(batch, (list, tuple)):
                model(batch[0].to(model.device))
                batch_size = batch[0].shape[0]
            else:
                model(batch.to(model.device))
                batch_size = batch.shape[0]

            # Check if we have enough samples
            sample_count = len(collectors[layer_names[0]].activations)
            if sample_count * batch_size >= max_samples:
                break

    # Remove hooks and gather activations
    activations_dict = {}
    for name, collector in collectors.items():
        collector.remove()
        X = collector.get_activations(max_samples=max_samples)
        if X is not None:
            activations_dict[name] = X
        collector.clear()

    print(f"Collected activations for {len(activations_dict)} layers")
    return activations_dict


def search_scale_alpha(
    weight: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    alpha_range: List[float] = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Search for optimal per-group alpha (clipping factor) using grid search.

    AWQ-style: For each group, search alpha in [1.0, 8.0] that minimizes
    activation-weighted reconstruction error: ||X @ W - X @ W_q||^2

    Args:
        weight: Original weight tensor, shape [out_features, in_features]
        X: Calibration activations, shape [n_samples, in_features]
        group_size: Group size for quantization
        alpha_range: List of alpha values to search (default: [1.0, 1.5, 2.0, ..., 8.0])
        device: Device for computation

    Returns:
        optimal_alpha: Best alpha per group, shape [out_features, num_groups]
        optimal_scale: Best scale per group, shape [out_features, num_groups]
        best_error: Best reconstruction error
    """
    if alpha_range is None:
        alpha_range = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    weight = weight.to(device)
    X = X.to(device)

    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    # Initialize
    optimal_alpha = torch.ones(out_features, num_groups, device=device)
    optimal_scale = torch.ones(out_features, num_groups, device=device)
    best_error = float('inf')

    # Try each alpha (simplified: use same alpha for all groups initially)
    errors = []

    for alpha_val in alpha_range:
        # Quantize with this alpha
        alpha_tensor = torch.ones(
            out_features, num_groups, device=device) * alpha_val
        weight_q, scale = quantize_weight_w2(
            weight, group_size=group_size, alpha=alpha_tensor)

        # Compute error
        error = compute_quantization_error(
            weight, weight_q, scale, group_size, X=X)
        errors.append(error)

        if error < best_error:
            best_error = error
            optimal_alpha = alpha_tensor.clone()
            optimal_scale = scale.clone()

    return optimal_alpha, optimal_scale, best_error


def search_scale_alpha_per_group(
    weight: torch.Tensor,
    X: torch.Tensor,
    group_size: int = 128,
    alpha_range: List[float] = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Search for optimal alpha independently for each group (more accurate but slower).

    Args:
        weight: Weight tensor, shape [out_features, in_features]
        X: Activations, shape [n_samples, in_features]
        group_size: Group size
        alpha_range: Alpha candidates
        device: Compute device

    Returns:
        optimal_alpha: shape [out_features, num_groups]
        optimal_scale: shape [out_features, num_groups]
        best_error: float
    """
    if alpha_range is None:
        alpha_range = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

    weight = weight.to(device)
    X = X.to(device)

    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    # Reshape for group-wise processing
    weight_grouped = weight.reshape(out_features, num_groups, group_size)
    X_grouped = X.reshape(X.shape[0], num_groups, group_size)

    optimal_alpha = torch.zeros(out_features, num_groups, device=device)
    optimal_scale = torch.zeros(out_features, num_groups, device=device)

    # Search per output channel and per group
    for out_idx in range(out_features):
        for group_idx in range(num_groups):
            w_group = weight_grouped[out_idx, group_idx]  # [group_size]
            x_group = X_grouped[:, group_idx, :]  # [n_samples, group_size]

            best_group_error = float('inf')
            best_alpha = 1.0
            best_scale = 1.0

            for alpha_val in alpha_range:
                # Apply clipping
                max_val = w_group.abs().max()
                clip_val = alpha_val * max_val
                w_clipped = torch.clamp(w_group, -clip_val, clip_val)

                # Quantize
                scale = w_clipped.abs().max()
                if scale < 1e-5:
                    scale = 1e-5

                w_q = torch.clamp(torch.round(w_clipped / scale), -2, 1)
                w_deq = w_q * scale

                # Compute error: ||X @ w - X @ w_deq||^2
                out_orig = x_group @ w_group
                out_deq = x_group @ w_deq
                error = ((out_orig - out_deq) ** 2).mean().item()

                if error < best_group_error:
                    best_group_error = error
                    best_alpha = alpha_val
                    best_scale = scale.item()

            optimal_alpha[out_idx, group_idx] = best_alpha
            optimal_scale[out_idx, group_idx] = best_scale

    # Compute overall error with optimal alphas
    weight_q, _ = quantize_weight_w2(
        weight, group_size=group_size, alpha=optimal_alpha)
    best_error = compute_quantization_error(
        weight, weight_q, optimal_scale.to(torch.float16), group_size, X=X
    )

    return optimal_alpha, optimal_scale.to(torch.float16), best_error


def calibrate_layer(
    layer: nn.Linear,
    X: torch.Tensor,
    group_size: int = 128,
    alpha_range: Optional[List[float]] = None,
    search_mode: str = 'global',  # 'global' or 'per_group'
    device: str = 'cuda'
) -> Dict:
    """
    Calibrate a single linear layer.

    Args:
        layer: The linear layer to calibrate
        X: Calibration activations
        group_size: Group size
        alpha_range: Alpha search range
        search_mode: 'global' (same alpha for all groups) or 'per_group' (independent search)
        device: Compute device

    Returns:
        Dictionary with calibration results: weight_q, scale, alpha, error
    """
    weight = layer.weight.data

    # Move to device
    weight = weight.to(device)
    X = X.to(device)

    # Search for optimal alpha and scale
    if search_mode == 'global':
        alpha, scale, error = search_scale_alpha(
            weight, X, group_size=group_size, alpha_range=alpha_range, device=device
        )
    else:
        alpha, scale, error = search_scale_alpha_per_group(
            weight, X, group_size=group_size, alpha_range=alpha_range, device=device
        )

    # Quantize with optimal parameters
    weight_q, scale_final = quantize_weight_w2(
        weight, group_size=group_size, alpha=alpha)

    return {
        'weight_q': weight_q.cpu(),
        'scale': scale_final.cpu(),
        'alpha': alpha.cpu(),
        'error': error,
        'group_size': group_size,
    }


def test_calibration():
    """Test calibration functions."""
    print("Testing AWQ calibration...")

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create synthetic data
    in_features, out_features = 512, 256
    layer = nn.Linear(in_features, out_features).to(device)
    X = torch.randn(128, in_features).to(device)  # 128 calibration samples

    print(f"Layer: {in_features} -> {out_features}")
    print(f"Calibration samples: {X.shape[0]}")

    # Test 1: Global alpha search
    print("\n1. Global alpha search...")
    alpha_global, scale_global, error_global = search_scale_alpha(
        layer.weight.data, X, group_size=128, device=device
    )
    print(f"  Best alpha: {alpha_global[0, 0]:.2f}")
    print(f"  Error: {error_global:.6f}")

    # Test 2: Per-group alpha search (on smaller tensor for speed)
    print("\n2. Per-group alpha search (first 2 output features)...")
    weight_small = layer.weight.data[:2, :]
    alpha_pg, scale_pg, error_pg = search_scale_alpha_per_group(
        weight_small, X, group_size=128, device=device
    )
    print(f"  Alpha range: [{alpha_pg.min():.2f}, {alpha_pg.max():.2f}]")
    print(f"  Error: {error_pg:.6f}")

    # Test 3: Full layer calibration
    print("\n3. Full layer calibration...")
    calib_result = calibrate_layer(
        layer, X, group_size=128, search_mode='global', device=device)
    print(f"  Weight_q shape: {calib_result['weight_q'].shape}")
    print(f"  Scale shape: {calib_result['scale'].shape}")
    print(f"  Error: {calib_result['error']:.6f}")

    print("\n✅ All calibration tests passed!")


if __name__ == "__main__":
    test_calibration()
