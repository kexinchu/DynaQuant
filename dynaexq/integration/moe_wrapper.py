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
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..policy.runtime import RuntimeExchangePlanner, RuntimeResidencyController

from ..core import (
    ExpertKey,
    ExpertRegistry,
    HotnessTracker,
    PrecisionScheduler,
    RouterObserver,
    TransitionEngine,
    TransitionReq,
    Tier,
)
from .generation_utils import last_logit_only_kwargs, prepare_one_token_decode

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
        exchange_planner: Optional["RuntimeExchangePlanner"] = None,
        residency_controller: Optional["RuntimeResidencyController"] = None,
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
        self.exchange_planner = exchange_planner
        self.residency_controller = residency_controller
        
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
        logits_kwargs = last_logit_only_kwargs(self)
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
                    **logits_kwargs,
                    **kwargs,
                )
            else:
                prepared = prepare_one_token_decode(
                    self.model,
                    generated=generated,
                    next_token=generated[:, -1:],
                    past_key_values=past,
                    attention_mask=attention_mask,
                )
                prepared.update(logits_kwargs)
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
        self.finish_request()
        return generated

    def finish_request(self) -> None:
        """Complete request-boundary control work before latency stops."""
        self._repair_quality_risk_at_request_boundary()
    
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
                experts._dynaexq_residency_controller = self.residency_controller
                attached.add(layer_idx)
                continue
            if experts is not None and attach_qwen3_next_experts(
                experts,
                self.registry,
                layer_idx,
            ):
                attached.add(layer_idx)
                experts._dynaexq_residency_controller = self.residency_controller
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
        if self.residency_controller is not None:
            self.residency_controller.on_route(
                layer,
                topk_indices,
                issued_step=self._step,
            )
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
            if self.residency_controller is not None:
                self.residency_controller.on_route(
                    layer,
                    topk_idx,
                    issued_step=self._step,
                )
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
    
    def _update_scheduler(self, *, force: bool = False) -> None:
        """Update scheduler and execute transitions."""
        try:
            # Update current tier assignments from registry
            # (In practice, would track this more carefully)
            self._sync_tier_assignments()
            
            # Plan transitions
            plan_kwargs = {
                "step": self._step,
                "tracker": self.hotness_tracker,
                "current_tiers": self._current_tiers,
                "force": force,
            }
            if isinstance(self.scheduler, PrecisionScheduler):
                plan_kwargs["eligible_experts"] = {
                    layer: {
                        key.expert
                        for key in self._current_tiers
                        if key.layer == layer
                    }
                    for layer in range(self.scheduler.num_layers)
                }
            requests = self.scheduler.plan(**plan_kwargs)
            
            if requests:
                logger.info(f"Step {self._step}: Planning {len(requests)} transitions")
            
            # Execute transitions if engine available
            if self.transition_engine is not None:
                if self.exchange_planner is not None:
                    self._submit_precision_exchanges(requests)
                else:
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

    def _repair_quality_risk_at_request_boundary(self) -> None:
        """Restore the hard quality constraint before returning a request."""
        controller = self.residency_controller
        if (
            not self.scheduler_enabled
            or controller is None
            or controller.quality_risk is None
            or controller.quality_risk_feasible()
        ):
            return
        started = time.perf_counter()
        deadline = started + controller.wait_timeout_s
        while not controller.quality_risk_feasible():
            self._update_scheduler(force=True)
            if controller.quality_risk_feasible():
                break
            if self._repair_quality_risk_from_published_state():
                break
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
        self._scheduler_update_samples_ms.append((time.perf_counter() - started) * 1000.0)
        if not controller.quality_risk_feasible():
            snapshot = controller.snapshot().get("quality_risk", {})
            raise RuntimeError(
                "request ended outside the calibrated quality-risk limit "
                f"(current={snapshot.get('current')}, "
                f"limit={snapshot.get('limit')}, "
                f"tolerance={snapshot.get('tolerance')})"
            )

    def _repair_quality_risk_from_published_state(self) -> bool:
        """Synchronously repair risk from the registry snapshot.

        The scheduler normally produces the same repair as part of the slow
        loop.  At a request boundary, however, the hard constraint is checked
        against the published registry, not the scheduler's projected state.
        If the projected repair did not converge, promote the highest-risk
        resident LO experts directly until the published state is feasible or
        no byte-feasible repair remains.
        """
        controller = self.residency_controller
        if (
            controller is None
            or controller.quality_risk is None
            or self.exchange_planner is None
            or self.transition_engine is None
            or controller.quality_risk_feasible()
        ):
            return controller.quality_risk_feasible() if controller else True

        handles = self.registry.handle_snapshot()
        eligible = {
            key
            for key, handle in handles.items()
            if handle.memory_claim is not None
        }
        high = {
            key
            for key, handle in handles.items()
            if handle.tier == Tier.HI and handle.memory_claim is not None
        }
        weights = {
            ExpertKey(layer, expert): float(score)
            for layer in range(controller.num_layers)
            for expert, score in enumerate(
                self.hotness_tracker.get_layer_scores(layer)
            )
        }
        repair = controller.quality_risk.repair_order(
            weights,
            high,
            eligible,
        )
        progressed = False
        for key in repair:
            handle = self.registry.get_handle(key)
            if handle is None or handle.tier == Tier.HI:
                continue
            unit = [
                TransitionReq(
                    key=key,
                    src=Tier.LO,
                    dst=Tier.HI,
                    reason="repair_quality_risk",
                    issued_step=self._step,
                    score_gap=controller.quality_risk.contribution(
                        key,
                        weights.get(key, 0.0),
                    ),
                )
            ]
            extra_donors = controller.precision_extra_donors(unit)
            if not self._submit_precision_unit_with_repair_retry(
                unit,
                additional_donors=extra_donors,
                repair_mode=True,
            ):
                logger.warning(
                    "direct quality-risk repair was rejected for %s",
                    key,
                )
                continue
            if not self.transition_engine.wait_ready(
                key,
                timeout=controller.wait_timeout_s,
            ):
                raise TimeoutError(
                    f"direct quality-risk repair timed out for {key}"
                )
            published = self.registry.get_handle(key)
            if published is None or published.tier != Tier.HI:
                raise RuntimeError(
                    f"direct quality-risk repair did not publish HI for {key}"
                )
            progressed = True
            if controller.quality_risk_feasible():
                return True
        return controller.quality_risk_feasible() if progressed else False

    def _submit_precision_exchanges(self, requests) -> None:
        """Preserve scheduler swap pairs at the executor transaction boundary."""
        assert self.exchange_planner is not None
        repair_mode = (
            self.residency_controller is not None
            and self.residency_controller.quality_risk is not None
            and any(
                request.reason == "repair_quality_risk"
                for request in requests
            )
        )
        if repair_mode:
            assert self.transition_engine is not None
            # Risk is global, so repair exchanges must also be global. Pair
            # the largest-risk promotion with the smallest-risk demotion;
            # every accepted unit is byte-neutral and cannot increase risk.
            promotions = sorted(
                (item for item in requests if item.dst == Tier.HI),
                key=lambda item: (-item.score_gap, item.key.layer, item.key.expert),
            )
            demotions = sorted(
                (item for item in requests if item.dst == Tier.LO),
                key=lambda item: (item.score_gap, item.key.layer, item.key.expert),
            )
            paired = min(len(demotions), len(promotions))
            units = [
                [demotions[index], promotions[index]]
                for index in range(paired)
            ]
            units.extend([item] for item in promotions[paired:])
            units.extend([item] for item in demotions[paired:])
        else:
            units = []
            by_layer: dict[int, list] = {}
            for request in requests:
                by_layer.setdefault(request.key.layer, []).append(request)
            for layer_requests in by_layer.values():
                demotions = [item for item in layer_requests if item.dst == Tier.LO]
                promotions = [item for item in layer_requests if item.dst == Tier.HI]
                paired = min(len(demotions), len(promotions))
                units.extend(
                    [demotions[index], promotions[index]]
                    for index in range(paired)
                )
                units.extend([item] for item in demotions[paired:])
                units.extend([item] for item in promotions[paired:])
        for unit in units:
            if repair_mode:
                unit = self._complete_repair_unit_with_published_demotion(unit)
            if (
                not repair_mode
                and self.residency_controller is not None
                and not self.residency_controller.precision_unit_feasible(
                    unit
                )
            ):
                logger.debug(
                    "Rejected precision unit by quality-risk account: %s",
                    [item.key for item in unit],
                )
                continue
            extra_donors = (
                self.residency_controller.precision_extra_donors(unit)
                if self.residency_controller is not None
                else []
            )
            accepted = self._submit_precision_unit_with_repair_retry(
                unit,
                additional_donors=extra_donors,
                repair_mode=repair_mode,
            )
            if not accepted and repair_mode:
                logger.warning(
                    "quality-risk repair exchange was rejected for %s; "
                    "request-boundary repair will enforce the hard constraint",
                    [item.key for item in unit],
                )
                continue
            if not accepted:
                logger.warning(
                    "Rejected precision exchange for %s",
                    [item.key for item in unit],
                )
                continue
            if repair_mode:
                # The next unit is valued against published state. Waiting
                # also prevents a complete risk projection from overflowing
                # the executor's bounded in-flight queue.
                for request in unit:
                    if not self.transition_engine.wait_ready(
                        request.key,
                        timeout=self.residency_controller.wait_timeout_s,
                    ):
                        raise TimeoutError(
                            "quality-risk repair timed out for "
                            f"{request.key}"
                        )
                    handle = self.registry.get_handle(request.key)
                    if handle is None or handle.tier != request.dst:
                        raise RuntimeError(
                            "quality-risk repair published the wrong tier "
                            f"for {request.key}"
                        )
        if (
            repair_mode
            and not self.residency_controller.quality_risk_feasible()
        ):
            logger.warning(
                "quality-risk projection remains outside the calibrated limit "
                "after this scheduler update; request-boundary repair will "
                "enforce the hard constraint"
            )

    def _submit_precision_unit_with_repair_retry(
        self,
        unit,
        *,
        additional_donors: list[ExpertKey],
        repair_mode: bool,
    ) -> bool:
        """Submit a precision unit, retrying transient repair rejections.

        Quality-risk repair runs on the request boundary. A repair exchange can
        be rejected transiently when another exchange still holds a donor,
        when the target is already in flight, or when the bounded executor
        queue is momentarily full. Those cases should not abort the request;
        the repair is still required, so wait briefly and resubmit until the
        controller's configured timeout expires.
        """
        assert self.exchange_planner is not None
        if not repair_mode:
            return self.exchange_planner.submit_precision_unit(
                unit,
                additional_donors=additional_donors,
            )
        timeout_s = (
            self.residency_controller.wait_timeout_s
            if self.residency_controller is not None
            else 5.0
        )
        # ``wait_timeout_s`` is the request-boundary budget.  It is too large
        # for one executor admission attempt: a permanently infeasible repair
        # unit would otherwise spin for minutes before the caller can try the
        # next donor/promotion.  Keep a short retry window for true transient
        # races such as a just-published donor or a briefly full queue, then
        # return False so the boundary repair can select another unit.
        deadline = time.perf_counter() + min(timeout_s, 0.5)
        last_accepted = False
        while True:
            last_accepted = self.exchange_planner.submit_precision_unit(
                unit,
                additional_donors=additional_donors,
            )
            if last_accepted:
                return True
            if time.perf_counter() >= deadline:
                return False
            for request in unit:
                self.transition_engine.wait_ready(request.key, timeout=0.02)
            time.sleep(0.005)

    def _complete_repair_unit_with_published_demotion(self, unit):
        """Make a single repair promotion byte-neutral when possible.

        The scheduler plans from a snapshot, while the executor admits against
        the published registry. If a hard quality repair reaches the executor
        as a lone promotion, the exchange may need an explicit HI demotion
        chosen from the published state. Pick the coldest demotion that keeps
        the quality-risk account feasible or strictly improves it; otherwise
        leave the unit unchanged so the normal admission path reports the
        failure.
        """
        if (
            len(unit) != 1
            or unit[0].dst != Tier.HI
            or self.residency_controller is None
        ):
            return unit
        target = unit[0].key
        handles = self.registry.handle_snapshot()
        candidates = [
            key
            for key, handle in handles.items()
            if key != target
            and handle.tier == Tier.HI
            and handle.memory_claim is not None
            and handle.active_readers == 0
        ]
        candidates.sort(
            key=lambda key: (
                self.hotness_tracker.get_score(key.layer, key.expert),
                key.layer,
                key.expert,
            )
        )
        for key in candidates:
            demotion = TransitionReq(
                key,
                Tier.HI,
                Tier.LO,
                "release_fidelity_bytes",
                unit[0].issued_step,
                score_gap=self.hotness_tracker.get_score(key.layer, key.expert),
            )
            paired = [demotion, unit[0]]
            if self.residency_controller.precision_unit_feasible(paired):
                return paired
        return unit
    
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
            "exchange_planner": (
                self.exchange_planner.snapshot()
                if self.exchange_planner is not None
                else None
            ),
            "residency_controller": (
                self.residency_controller.snapshot()
                if self.residency_controller is not None
                else None
            ),
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
