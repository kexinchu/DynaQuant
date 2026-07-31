"""Handle-mode adapter for upstream Transformers Qwen3-Next experts.

Qwen3-Next stores every routed expert in two fused 3-D parameters:
``gate_up_proj[num_experts, ...]`` and
``down_proj[num_experts, ...]``.  The upstream module has no extension
point for replacing an individual expert, so DynaExQ installs an
instance-local forward method that resolves the two packed projections
through :class:`ExpertRegistry`.

The shared expert is owned by ``Qwen3NextSparseMoeBlock`` rather than this
container.  It deliberately remains on the native dense path and is charged
to the fixed-model reservation.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

import torch

from ..core.quant import fused_linear
from ..core.registry import ExpertKey

if TYPE_CHECKING:
    from ..core.registry import ExpertHandle, ExpertRegistry


_SUPPORTED_EXPERT_CLASSES = {"Qwen3NextExperts"}
_SUPPORTED_UNFUSED_EXPERT_CLASSES = {"Qwen3NextMLP"}


def _record_and_release(
    module: torch.nn.Module,
    handle: "ExpertHandle",
    output: torch.Tensor | None,
) -> None:
    event = None
    stream_id = None
    if output is not None and output.is_cuda:
        with torch.cuda.device(output.device):
            stream = torch.cuda.current_stream(output.device)
            event = torch.cuda.Event()
            event.record(stream)
            stream_id = int(stream.cuda_stream)
    module._dynaexq_registry.release_handle(handle, event, stream_id)


def _handle_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Execute routed experts exclusively through versioned handles."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index,
            num_classes=module.num_experts,
        ).permute(2, 1, 0)
        expert_hit = torch.greater(
            expert_mask.sum(dim=(-1, -2)),
            0,
        ).nonzero()

    for expert_index_tensor in expert_hit:
        expert_index = int(expert_index_tensor[0])
        key = ExpertKey(module._dynaexq_layer_idx, expert_index)
        handle = module._dynaexq_registry.acquire_handle(key)
        if handle is None:
            raise RuntimeError(
                f"Qwen3-Next routed expert {key} has no registered handle"
            )

        output_for_fence = None
        try:
            gate_up = handle.get_packed("gate_up_proj")
            down = handle.get_packed("down_proj")
            if gate_up is None or down is None:
                raise RuntimeError(
                    f"Qwen3-Next routed expert {key} is missing "
                    "gate_up_proj or down_proj"
                )

            top_k_position, token_index = torch.where(
                expert_mask[expert_index]
            )
            current_state = hidden_states[token_index]
            gate, up = fused_linear(current_state, gate_up).chunk(2, dim=-1)
            current_hidden_states = module.act_fn(gate) * up
            current_hidden_states = fused_linear(current_hidden_states, down)
            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_index, top_k_position, None]
            )
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )
            output_for_fence = final_hidden_states
        finally:
            _record_and_release(module, handle, output_for_fence)

    return final_hidden_states


def _unfused_handle_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Execute one AutoRound-unfused routed expert through its handle."""
    key = ExpertKey(
        module._dynaexq_layer_idx,
        module._dynaexq_expert_idx,
    )
    handle = module._dynaexq_registry.acquire_handle(key)
    if handle is None:
        raise RuntimeError(
            f"Qwen3-Next routed expert {key} has no registered handle"
        )

    output = None
    try:
        gate_up = handle.get_packed("gate_up_proj")
        down = handle.get_packed("down_proj")
        if gate_up is not None:
            if down is None:
                raise RuntimeError(
                    f"Qwen3-Next routed expert {key} is missing down_proj"
                )
            gate, up = fused_linear(hidden_states, gate_up).chunk(2, dim=-1)
        else:
            gate_proj = handle.get_packed("gate_proj")
            up_proj = handle.get_packed("up_proj")
            if gate_proj is None or up_proj is None or down is None:
                raise RuntimeError(
                    f"Qwen3-Next routed expert {key} has an incomplete "
                    "gate/up/down projection set"
                )
            gate = fused_linear(hidden_states, gate_proj)
            up = fused_linear(hidden_states, up_proj)
        output = fused_linear(module.act_fn(gate) * up, down)
        return output
    finally:
        _record_and_release(module, handle, output)


def _attach_unfused_experts(
    experts: torch.nn.ModuleList,
    registry: "ExpertRegistry",
    layer_index: int,
) -> bool:
    if not experts:
        raise RuntimeError("Qwen3-Next expert ModuleList is empty")
    if not all(
        type(expert).__name__ in _SUPPORTED_UNFUSED_EXPERT_CLASSES
        for expert in experts
    ):
        return False

    for expert_index, expert in enumerate(experts):
        projections = [
            getattr(expert, slot, None)
            for slot in ("gate_proj", "up_proj", "down_proj")
        ]
        valid = all(
            isinstance(projection, torch.nn.Module)
            and isinstance(getattr(projection, "qweight", None), torch.Tensor)
            and isinstance(getattr(projection, "qzeros", None), torch.Tensor)
            and isinstance(getattr(projection, "scales", None), torch.Tensor)
            for projection in projections
        )
        if not valid:
            raise RuntimeError(
                "unsupported unfused Qwen3NextMLP layout; expected three "
                "AutoGPTQ/AutoRound quantized projections"
            )
        if not hasattr(expert, "_dynaexq_original_forward"):
            expert._dynaexq_original_forward = expert.forward
            expert.forward = MethodType(_unfused_handle_forward, expert)
        expert._dynaexq_registry = registry
        expert._dynaexq_layer_idx = int(layer_index)
        expert._dynaexq_expert_idx = expert_index
    return True


def attach_qwen3_next_experts(
    experts: torch.nn.Module,
    registry: "ExpertRegistry",
    layer_index: int,
) -> bool:
    """Attach a supported upstream Qwen3-Next expert container.

    Returns ``False`` for unrelated module classes so callers can continue
    their adapter discovery.  A matching class with an incompatible layout
    raises immediately because silently accepting a changed Transformers
    implementation would invalidate both correctness and byte accounting.
    """
    if isinstance(experts, torch.nn.ModuleList):
        return _attach_unfused_experts(experts, registry, layer_index)
    if type(experts).__name__ not in _SUPPORTED_EXPERT_CLASSES:
        return False

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    num_experts = getattr(experts, "num_experts", None)
    parameters_exist = isinstance(gate_up, torch.nn.Parameter) and isinstance(
        down,
        torch.nn.Parameter,
    )
    native_layout = (
        parameters_exist
        and gate_up.dim() == 3
        and down.dim() == 3
        and isinstance(num_experts, int)
        and gate_up.shape[0] == num_experts
        and down.shape[0] == num_experts
    )
    released_layout = (
        parameters_exist
        and isinstance(num_experts, int)
        and gate_up.numel() == 0
        and down.numel() == 0
    )
    if not (native_layout or released_layout):
        raise RuntimeError(
            "unsupported Qwen3NextExperts layout; expected matching 3-D or "
            "fully released gate_up_proj/down_proj parameters"
        )

    if not hasattr(experts, "_dynaexq_original_forward"):
        experts._dynaexq_original_forward = experts.forward
        experts.forward = MethodType(_handle_forward, experts)
    experts._dynaexq_registry = registry
    experts._dynaexq_layer_idx = int(layer_index)
    return True
