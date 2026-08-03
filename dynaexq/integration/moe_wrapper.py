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
import math
import time
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
        topk: Optional[int] = None,
        scheduler_enabled: bool = True,
        routing_profile_enabled: bool = False,
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
            scheduler_enabled: Whether online precision reassignment is
                active. False freezes the deterministic bootstrap map for the
                static-precision ablation while retaining router observation.
            routing_profile_enabled: Accumulate exact selected-expert counts.
                This is disabled by default so ordinary latency runs do not
                pay for profiling synchronization.
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
        self._topk = topk
        self.scheduler_enabled = scheduler_enabled
        self.routing_profile_enabled = routing_profile_enabled
        self._router_observations = 0
        self._scheduler_update_samples_ms: list[float] = []
        self._routing_counts: dict[int, torch.Tensor] = {}
        
        self._attached_layers = self._attach_registry()
        # Register one hook per router to capture routing decisions.
        self._hooks = []
        self._router_layers = self._register_hooks()
    
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
            observations_before = self._router_observations
            # Run model forward (hooks capture router outputs).
            outputs = self.model(*args, **kwargs)
            if (
                self._router_layers > 0
                and self._router_observations == observations_before
            ):
                raise RuntimeError(
                    "No router observation was produced by this forward pass"
                )
            
            # Check if scheduler should update
            if (
                self.scheduler_enabled
                and self.scheduler.should_update(self._step)
            ):
                scheduler_started = time.perf_counter()
                self._update_scheduler()
                self._scheduler_update_samples_ms.append(
                    (time.perf_counter() - scheduler_started) * 1000.0
                )
            
            return outputs
        except Exception as e:
            logger.error(f"Error in forward pass at step {self._step}: {e}")
            raise

    def __call__(self, *args, **kwargs):
        """Make the wrapper usable by model-level evaluation functions."""
        return self.forward(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id=None,
        **kwargs,
    ) -> torch.Tensor:
        """Greedy cached generation that keeps every decode step wrapped.

        The evaluation harness only requests greedy tensor output. Unsupported
        sampling or structured-generation options fail explicitly instead of
        falling back to the unwrapped model's ``generate`` method.
        """
        if do_sample:
            raise NotImplementedError("MoEWrapper.generate supports greedy decoding only")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        unsupported = {
            name
            for name in (
                "return_dict_in_generate",
                "output_scores",
                "num_beams",
            )
            if name in kwargs and kwargs[name] not in (False, None, 1)
        }
        if unsupported:
            raise NotImplementedError(
                f"unsupported wrapped generation options: {sorted(unsupported)}"
            )
        if max_new_tokens == 0:
            return input_ids

        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        past = None
        finished = torch.zeros(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        configured_eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.model.generation_config, "eos_token_id", None)
        )
        eos_ids = (
            []
            if configured_eos is None
            else [configured_eos]
            if isinstance(configured_eos, int)
            else list(configured_eos)
        )
        replacement = (
            pad_token_id
            if pad_token_id is not None
            else (eos_ids[0] if eos_ids else 0)
        )

        for _ in range(max_new_tokens):
            if past is None:
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **kwargs,
                )
            else:
                prepared = self.model.prepare_inputs_for_generation(
                    generated,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                outputs = self.forward(**prepared)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            if finished.any():
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, replacement),
                    next_token,
                )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token[:, None])),
                dim=1,
            )
            past = outputs.past_key_values
            if eos_ids:
                reached_eos = torch.zeros_like(finished)
                for token_id in eos_ids:
                    reached_eos |= next_token == token_id
                finished |= reached_eos
                if finished.all():
                    break
        return generated
    
    def _register_hooks(self) -> int:
        """Register forward hooks to capture router outputs."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                try:
                    # Qwen-style routers return
                    # (full probabilities, top-k weights, top-k indices).
                    if (
                        isinstance(output, tuple)
                        and len(output) >= 3
                        and isinstance(output[-1], torch.Tensor)
                        and isinstance(output[-2], torch.Tensor)
                        and not output[-1].is_floating_point()
                    ):
                        self._observe_selected(
                            layer_idx, output[-1], output[-2]
                        )
                        return
                    if (
                        isinstance(output, tuple)
                        and len(output) == 2
                        and isinstance(output[1], torch.Tensor)
                        and output[1].is_floating_point()
                    ):
                        self._observe_router_outputs(layer_idx, output[1])
                        return
                    router_logits = (
                        output.router_logits
                        if hasattr(output, "router_logits")
                        else output if isinstance(output, torch.Tensor) else None
                    )
                    if router_logits is not None:
                        self._observe_router_outputs(layer_idx, router_logits)
                except Exception as e:
                    raise RuntimeError(
                        f"router hook failed for layer {layer_idx}"
                    ) from e
            return hook

        hooked_layers: set[int] = set()
        for name, module in self.model.named_modules():
            import re

            match = re.search(r"layers\.(\d+)", name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            if layer_idx in hooked_layers:
                continue
            class_name = type(module).__name__.lower()
            is_router = (
                "router" in class_name
                or hasattr(module, "attach_dynaexq")
                or (
                    name.endswith(".gate")
                    and (
                        hasattr(module, "top_k")
                        or hasattr(module, "num_experts")
                    )
                )
            )
            if not is_router:
                continue
            handle = module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)
            hooked_layers.add(layer_idx)
            logger.debug("Registered router hook for %s (layer %d)", name, layer_idx)
        return len(hooked_layers)

    def _attach_registry(self) -> int:
        """Attach the registry to supported model-native MoE adapters."""
        import re

        from .qwen3_next_adapter import attach_qwen3_next_experts

        attached: set[int] = set()
        for name, module in self.model.named_modules():
            match = re.search(r"layers\.(\d+)", name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            if layer_idx in attached:
                continue
            if hasattr(module, "attach_dynaexq"):
                module.attach_dynaexq(self.registry, layer_idx)
                attached.add(layer_idx)
                continue
            experts = getattr(module, "experts", None)
            if experts is not None and hasattr(experts, "_dynaexq_registry"):
                # Transformers may default fused Qwen experts to its
                # grouped-mm implementation, whose decorator bypasses the
                # model class's handle-aware forward method. Force the
                # original eager method whenever DynaExQ owns dispatch.
                experts_config = getattr(experts, "config", None)
                if experts_config is not None and hasattr(
                    experts_config,
                    "_experts_implementation",
                ):
                    experts_config._experts_implementation = "eager"
                experts._dynaexq_registry = self.registry
                experts._dynaexq_layer_idx = layer_idx
                attached.add(layer_idx)
                continue
            if experts is not None and attach_qwen3_next_experts(
                experts,
                self.registry,
                layer_idx,
            ):
                attached.add(layer_idx)
        return len(attached)

    def validate_integration(self) -> None:
        """Fail closed when a model cannot observe or consume runtime state."""
        if self._attached_layers == 0:
            raise RuntimeError(
                "No DynaExQ-capable MoE adapter was found; dynamic results "
                "would execute the original static expert weights."
            )
        if self._router_layers == 0:
            raise RuntimeError(
                "No router hook was registered; the scheduler would receive "
                "no workload signal."
            )

    def _observe_selected(
        self,
        layer: int,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        self._record_routing_counts(layer, topk_indices)
        signal = self.router_observer.extract_signal(
            layer=layer,
            topk_indices=topk_indices.detach(),
            topk=topk_indices.shape[-1],
            selected_weights=topk_weights.detach(),
        )
        self._router_observations += 1
        g_values = self.router_observer.compute_g_signal(signal)
        if g_values:
            self.hotness_tracker.update(layer, g_values)
    
    def _observe_router_outputs(self, layer: int, router_logits) -> None:
        """
        Observe router outputs for a specific layer and update hotness tracker.

        P3 optimisation: compute top-k entirely on the GPU via ``torch.topk``
        and only transfer the resulting *small* index/value tensors to CPU.
        This eliminates the per-layer implicit GPU→CPU sync that the old
        ``router_logits.cpu().numpy()`` path caused.
        """
        try:
            if not isinstance(router_logits, torch.Tensor):
                return

            # Flatten to 2D (tokens, num_experts) regardless of whether the
            # model returned (batch, seq, experts) or (tokens, experts).
            num_experts = router_logits.shape[-1]
            logits_2d = router_logits.detach().reshape(-1, num_experts)  # stays on GPU

            # Determine top-k from model config, clamped to actual expert count.
            topk = min(self._topk or 1, num_experts)

            # GPU-side top-k — no CPU transfer of the full logits tensor.
            topk_vals, topk_idx = torch.topk(
                logits_2d, k=topk, dim=-1, sorted=False
            )  # both: (num_tokens, topk), still on GPU
            topk_weights = torch.softmax(logits_2d, dim=-1).gather(1, topk_idx)
            self._record_routing_counts(layer, topk_idx)

            # Pass the small GPU tensors to extract_signal.
            # router_observer.extract_signal accepts torch.Tensor and calls
            # .cpu().numpy() internally — only tiny topk tensors cross PCIe.
            # Passing topk_vals as "logits" means the weight computation uses
            # a softmax over the top-k scores, which is a fine approximation
            # for the EMA hotness tracker (we don't need full-softmax precision).
            signal = self.router_observer.extract_signal(
                layer=layer,
                topk_indices=topk_idx,
                topk=topk,
                selected_weights=topk_weights,
            )
            self._router_observations += 1

            g_values = self.router_observer.compute_g_signal(signal)
            if g_values:
                self.hotness_tracker.update(layer, g_values)
        except Exception as e:
            raise RuntimeError(
                f"failed to observe router outputs for layer {layer}"
            ) from e

    def _record_routing_counts(
        self,
        layer: int,
        topk_indices: torch.Tensor,
    ) -> None:
        """Accumulate one count per selected token--expert dispatch."""
        if not self.routing_profile_enabled:
            return
        expert_count = self._experts_per_layer
        if expert_count is None or expert_count <= 0:
            raise RuntimeError(
                "routing profiling requires experts_per_layer"
            )
        flattened = topk_indices.detach().reshape(-1).to(torch.long)
        if flattened.numel() == 0:
            return
        if (
            int(flattened.min().item()) < 0
            or int(flattened.max().item()) >= expert_count
        ):
            raise RuntimeError("router selected an out-of-range expert")
        counts = torch.bincount(flattened, minlength=expert_count)
        previous = self._routing_counts.get(layer)
        if previous is None:
            self._routing_counts[layer] = counts
        elif previous.device == counts.device:
            previous.add_(counts)
        else:
            self._routing_counts[layer] = previous.cpu() + counts.cpu()
    
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
            # Do not optimistically update tiers for rejected or in-flight
            # transitions. The next scheduler tick snapshots published
            # registry state.
        except Exception as e:
            logger.error(f"Error updating scheduler at step {self._step}: {e}")
            raise
    
    def _sync_tier_assignments(self) -> None:
        """Sync current tier assignments from registry."""
        self._current_tiers = self.registry.tier_snapshot()
    
    def get_expert_handle(self, layer: int, expert: int):
        """Get expert handle for forward pass."""
        key = ExpertKey(layer=layer, expert=expert)
        return self.registry.get_handle(key)
    
    def reset(self) -> None:
        """Reset state."""
        self._step = 0
        self._router_observations = 0
        self._current_tiers.clear()
        self.hotness_tracker.reset()
        self._scheduler_update_samples_ms.clear()
        self.reset_routing_profile()
        logger.info("MoEWrapper state reset")

    def reset_routing_profile(self) -> None:
        """Clear dispatch counters without changing scheduler or EMA state."""
        self._routing_counts.clear()

    def get_routing_profile(self) -> dict[int, list[int]]:
        """Return a stable CPU snapshot of exact per-expert dispatch counts."""
        return {
            layer: [int(value) for value in counts.detach().cpu().tolist()]
            for layer, counts in sorted(self._routing_counts.items())
        }

    def get_stats(self) -> dict:
        """Return control-plane integration counters for result artifacts."""
        scheduler_samples = list(self._scheduler_update_samples_ms)
        ordered = sorted(scheduler_samples)
        p99_index = (
            max(0, math.ceil(0.99 * len(ordered)) - 1)
            if ordered
            else None
        )
        return {
            "forward_steps": self._step,
            "router_observations": self._router_observations,
            "attached_layers": self._attached_layers,
            "router_layers": self._router_layers,
            "scheduler_enabled": self.scheduler_enabled,
            "routing_profile_enabled": self.routing_profile_enabled,
            "scheduler_update_samples_ms": scheduler_samples,
            "scheduler_update_count": len(scheduler_samples),
            "scheduler_mean_ms": (
                sum(scheduler_samples) / len(scheduler_samples)
                if scheduler_samples
                else 0.0
            ),
            "scheduler_p99_ms": (
                ordered[p99_index] if p99_index is not None else 0.0
            ),
            "scheduler_max_ms": max(scheduler_samples, default=0.0),
        }
    
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
