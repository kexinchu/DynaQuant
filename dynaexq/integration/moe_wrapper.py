"""
MoE Wrapper: integrates DynaExq with model forward pass.

Wraps model forward to:
1. Observe router outputs
2. Update hotness tracker
3. Use expert handles from registry
4. Trigger scheduler updates
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from ..core import (
    ExpertKey,
    ExpertRegistry,
    HotnessTracker,
    PrecisionScheduler,
    RouterObserver,
    TransitionEngine,
    Tier,
)

logger = logging.getLogger(__name__)


class MoEWrapper:
    """
    Wraps MoE model forward pass with DynaExq integration.
    
    During forward:
    - Observes router outputs
    - Updates hotness tracker
    - Uses expert handles from registry
    - Scheduler/TransitionEngine run in background
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        router_observer: RouterObserver,
        hotness_tracker: HotnessTracker,
        scheduler: PrecisionScheduler,
        registry: ExpertRegistry,
        transition_engine: Optional[TransitionEngine] = None,
        num_layers: Optional[int] = None,
        experts_per_layer: Optional[int] = None,
    ):
        """
        Args:
            model: The MoE model to wrap
            router_observer: RouterObserver for extracting signals
            hotness_tracker: HotnessTracker for maintaining scores
            scheduler: PrecisionScheduler for planning transitions
            registry: ExpertRegistry for expert handles
            transition_engine: Optional TransitionEngine for executing transitions
            num_layers: Number of MoE layers (if not inferrable from model)
            experts_per_layer: Number of experts per layer (if not inferrable)
        """
        self.model = model
        self.router_observer = router_observer
        self.hotness_tracker = hotness_tracker
        self.scheduler = scheduler
        self.registry = registry
        self.transition_engine = transition_engine
        
        self._step = 0
        self._current_tiers: dict[ExpertKey, Tier] = {}  # Track current tiers
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        
        # Register forward hooks to capture router outputs
        self._hooks = []
        self._register_hooks()
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with DynaExq integration.
        
        Steps:
        1. Run model forward (hooks capture router outputs)
        2. Update hotness tracker from captured router outputs
        3. Check if scheduler should update
        4. Execute transitions if needed
        5. Use expert handles from registry (handled by hooks)
        """
        # Increment step
        self._step += 1
        
        try:
            # Run model forward (hooks will capture router outputs)
            outputs = self.model(*args, **kwargs)
            
            # Check if scheduler should update
            if self.scheduler.should_update(self._step):
                self._update_scheduler()
            
            return outputs
        except Exception as e:
            logger.error(f"Error in forward pass at step {self._step}: {e}")
            raise
    
    def _register_hooks(self):
        """Register forward hooks to capture router outputs."""
        # Track layer indices for MoE layers
        layer_idx = 0
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                try:
                    # Try to extract router outputs from MoE layer
                    # For Qwen3 and similar models, router logits may be in different places
                    router_logits = None
                    
                    # Check if output has router_logits attribute
                    if hasattr(output, 'router_logits'):
                        router_logits = output.router_logits
                    elif isinstance(output, tuple):
                        # Some models return (hidden_states, router_logits, ...)
                        for item in output:
                            if hasattr(item, 'router_logits'):
                                router_logits = item.router_logits
                                break
                            elif isinstance(item, dict) and 'router_logits' in item:
                                router_logits = item['router_logits']
                                break
                    elif isinstance(output, dict):
                        if 'router_logits' in output:
                            router_logits = output['router_logits']
                    
                    # Also check module attributes (some models store router state in module)
                    if router_logits is None and hasattr(module, 'router_logits'):
                        router_logits = module.router_logits
                    
                    if router_logits is not None:
                        self._observe_router_outputs(layer_idx, router_logits)
                except Exception as e:
                    logger.debug(f"Error in hook for layer {layer_idx}: {e}")
            return hook
        
        # Try to find and hook MoE layers
        # For Qwen3 models, MoE layers are typically in model.layers[i].mlp or similar
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            # Check for MoE-related patterns
            is_moe_layer = (
                'moe' in name_lower or 
                'expert' in name_lower or
                ('mlp' in name_lower and hasattr(module, 'experts')) or
                ('gate' in name_lower and 'router' in name_lower)
            )
            
            if is_moe_layer:
                try:
                    # Try to extract layer index from name (e.g., "model.layers.5.mlp")
                    import re
                    match = re.search(r'layers\.(\d+)', name)
                    if match:
                        layer_idx = int(match.group(1))
                    else:
                        layer_idx = len(self._hooks)
                    
                    hook = make_hook(layer_idx)
                    handle = module.register_forward_hook(hook)
                    self._hooks.append(handle)
                    logger.debug(f"Registered hook for {name} (layer {layer_idx})")
                except Exception as e:
                    logger.debug(f"Could not register hook for {name}: {e}")
    
    def _observe_router_outputs(self, layer: int, router_logits) -> None:
        """
        Observe router outputs for a specific layer and update hotness tracker.
        
        Args:
            layer: Layer index
            router_logits: Router logits or outputs from the layer
        """
        try:
            # Extract signal from router outputs
            # The exact format depends on the model architecture
            if isinstance(router_logits, torch.Tensor):
                # Assume shape: (batch, seq_len, num_experts) or (batch*seq_len, num_experts)
                logits_np = router_logits.detach().cpu().numpy()
                
                # Get top-k experts (simplified - would use actual top-k from model)
                # For now, we'll compute top-k from logits
                import numpy as np
                # Default topk - should come from model config
                topk = min(getattr(self, '_experts_per_layer', 8), logits_np.shape[-1])
                if logits_np.ndim == 2:
                    # (tokens, num_experts)
                    topk_indices = np.argsort(-logits_np, axis=-1)[:, :topk]
                elif logits_np.ndim == 3:
                    # (batch, seq_len, num_experts)
                    batch_size, seq_len, num_experts = logits_np.shape
                    logits_flat = logits_np.reshape(-1, num_experts)
                    topk_indices = np.argsort(-logits_flat, axis=-1)[:, :topk]
                else:
                    logger.warning(f"Unexpected router logits shape: {logits_np.shape}")
                    return
                
                # Extract signal
                signal = self.router_observer.extract_signal(
                    layer=layer,
                    topk_indices=topk_indices,
                    logits=logits_np,
                    topk=topk,
                )
                
                # Compute g values
                g_values = self.router_observer.compute_g_signal(signal)
                
                # Update tracker
                if g_values:
                    self.hotness_tracker.update(layer, g_values)
        except Exception as e:
            logger.debug(f"Error observing router outputs for layer {layer}: {e}")
    
    def _update_scheduler(self) -> None:
        """Update scheduler and execute transitions."""
        try:
            # Update current tier assignments from registry
            # (In practice, would track this more carefully)
            self._sync_tier_assignments()
            
            # Plan transitions
            requests = self.scheduler.plan(
                step=self._step,
                tracker=self.hotness_tracker,
                current_tiers=self._current_tiers,
            )
            
            if requests:
                logger.info(f"Step {self._step}: Planning {len(requests)} transitions")
            
            # Execute transitions if engine available
            if self.transition_engine is not None:
                for req in requests:
                    if self.transition_engine.enqueue(req):
                        logger.debug(f"Enqueued transition: {req.key} {req.src}->{req.dst}")
                    else:
                        logger.warning(f"Failed to enqueue transition: {req.key}")
            
            # Update current tiers
            for req in requests:
                self._current_tiers[req.key] = req.dst
        except Exception as e:
            logger.error(f"Error updating scheduler at step {self._step}: {e}")
    
    def _sync_tier_assignments(self) -> None:
        """Sync current tier assignments from registry."""
        # This is a simplified implementation
        # In practice, would query registry for all experts
        # For now, we'll just update from what we know
        pass
    
    def get_expert_handle(self, layer: int, expert: int):
        """Get expert handle for forward pass."""
        key = ExpertKey(layer=layer, expert=expert)
        return self.registry.get_handle(key)
    
    def reset(self) -> None:
        """Reset state."""
        self._step = 0
        self._current_tiers.clear()
        self.hotness_tracker.reset()
        logger.info("MoEWrapper state reset")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("Removed all forward hooks")
    
    def __del__(self):
        """Cleanup hooks on destruction."""
        try:
            self.remove_hooks()
        except Exception:
            pass

