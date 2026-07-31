"""
WeightStore: loads expert weights as ``PackedTensor`` (Plan A).

This is the Plan A rewrite of the original ``ModelWeightStore``. Key changes:

1. ``load_weights(key, tier)`` now returns a ``PackedTensor`` rather than a
   raw fp16 ``torch.Tensor``. HI tier returns ``PackedTensor(fmt=FP16)``;
   LO tier returns ``PackedTensor(fmt=INT4 or INT2)`` per the configured
   ``lo_format``. The runtime never has to branch on tier — only on
   ``packed.fmt`` — which removes the ``_wrap_as_packed`` bridge that
   ``TransitionEngine`` used during the Plan B phase.

2. ``get_byte_size(key, tier)`` delegates to
   ``quant.compute_packed_nbytes`` instead of doing its own
   ``params * bytes_per_param`` arithmetic. There is now exactly one
   source of truth for byte counting across the codebase, which is what
   the BudgetTracker reservation contract depends on.

3. Packed tensors are **memoized per (key, tier)** in host memory.
   Production experiments call ``preload_all`` before timing so first-use
   quantization cannot leak into a transition measurement.

Multi-linear expert support
---------------------------
Qwen3 and Phi-MoE experts are returned as a mapping from projection name
to ``PackedTensor`` and move through one transition lifecycle.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from .autogptq import packed_from_autogptq
from .config import Tier
from .quant import (
    DEFAULT_GROUP_SIZE,
    PackedTensor,
    QuantFormat,
    compute_packed_nbytes,
    pack,
)
from .registry import ExpertKey


def _parse_format(spec: str) -> QuantFormat:
    """Parse 'fp16' / 'int4' / 'int2' (case-insensitive) to QuantFormat."""
    spec_lower = spec.strip().lower()
    for fmt in QuantFormat:
        if fmt.value == spec_lower:
            return fmt
    valid = ", ".join(f.value for f in QuantFormat)
    raise ValueError(f"Unknown quant format {spec!r}; expected one of {{{valid}}}")


def _trim_released_host_memory() -> bool:
    """Return free temporary packing arenas to the OS when supported.

    Pinning a packed tensor necessarily creates a second allocation before
    the ordinary CPU tensor can be released. Glibc otherwise tends to retain
    those large temporary arenas, making peak RSS scale with the source model
    plus both the pinned cache and already-dead packing buffers. Trimming once
    per released layer keeps large-MoE initialization bounded without touching
    live pinned tensors.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return False


class ModelWeightStore:
    """
    Loads expert weights from a model and serves them as ``PackedTensor``.

    Args:
        model: The model containing expert weights. May be ``None`` for
            tests that pre-register packed tensors directly via
            :meth:`register_expert`.
        hi_format: Format for HI tier ("fp16" / "int4" / "int2").
        lo_format: Format for LO tier ("fp16" / "int4" / "int2").
        cache_dir: Optional directory for offline-packed weights produced
            by ``scripts/pack_experts.py``. Currently unused (in-process
            cache only); reserved for the offline path.
        backend: Quantization backend forwarded to ``quant.pack``. Default
            ``"reference"``; switch to ``"autoround"`` once the AutoRound
            calibration loop (Phase 2.1) is wired up.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        hi_format: str = "fp16",
        lo_format: str = "int4",
        cache_dir: Optional[str] = None,
        backend: str = "reference",
        pin_memory: bool = False,
        enable_int4_kernel_cache: bool = False,
        fused_pack_chunk_experts: int = 16,
    ):
        self.model = model
        self.hi_fmt = _parse_format(hi_format)
        self.lo_fmt = _parse_format(lo_format)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if backend not in {"reference", "autoround"}:
            raise ValueError(
                f"backend must be 'reference' or 'autoround', got {backend!r}"
            )
        self.backend = backend
        self.pin_memory = pin_memory
        self.enable_int4_kernel_cache = enable_int4_kernel_cache
        if fused_pack_chunk_experts <= 0:
            raise ValueError("fused_pack_chunk_experts must be positive")
        self.fused_pack_chunk_experts = fused_pack_chunk_experts

        # In-process cache: (key, tier) -> PackedTensor or dict.
        self._packed_cache: dict[
            tuple[ExpertKey, Tier], PackedTensor | dict[str, PackedTensor]
        ] = {}
        # Multi-linear cache: (key, tier) -> dict[str, PackedTensor]
        self._packed_multi_cache: dict[
            tuple[ExpertKey, Tier], dict[str, PackedTensor]
        ] = {}

        # For tests / pre-loaded weights: explicit raw fp16 source per
        # expert. Bypasses model lookup. ``register_expert`` populates this.
        self._raw_overrides: dict[ExpertKey, torch.Tensor] = {}
        # Multi-linear overrides: {key: {slot_name: tensor}}
        self._raw_multi_overrides: dict[
            ExpertKey, dict[str, torch.Tensor]
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_weights(
        self, key: ExpertKey, tier: Tier
    ) -> PackedTensor | dict[str, PackedTensor]:
        """Return every linear in an expert, packed for the requested tier."""
        cache_key = (key, tier)
        cached = self._packed_cache.get(cache_key)
        if cached is not None:
            return cached

        quantized_slots = self._fetch_autogptq_modules(key)
        if quantized_slots is not None:
            target_fmt = self._tier_fmt(tier)
            packed_slots = {
                slot: self._maybe_pin(
                    self._with_resident_footprint(
                        packed_from_autogptq(
                            module.qweight,
                            module.qzeros,
                            module.scales,
                            source_bits=int(module.bits),
                            source_group_size=int(module.group_size),
                            target_format=target_fmt,
                            g_idx=getattr(module, "g_idx", None),
                        )
                    )
                )
                for slot, module in quantized_slots.items()
            }
            self._packed_cache[cache_key] = packed_slots
            return packed_slots

        raw_slots = self._fetch_raw_slots(key)
        target_fmt = self._tier_fmt(tier)
        group_size = (
            None if target_fmt == QuantFormat.FP16 else DEFAULT_GROUP_SIZE[target_fmt]
        )
        packed_slots = {
            slot: self._maybe_pin(
                self._with_resident_footprint(pack(
                    raw.to(device="cpu", dtype=torch.float16),
                    target_fmt,
                    group_size=group_size,
                    backend=self.backend,
                ))
            )
            for slot, raw in raw_slots.items()
        }
        # Preserve the historical single-linear API for synthetic fixtures.
        packed: PackedTensor | dict[str, PackedTensor]
        if set(packed_slots) == {"weight"}:
            packed = packed_slots["weight"]
        else:
            packed = packed_slots
        self._packed_cache[cache_key] = packed
        return packed

    def preload_all(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
        tiers: tuple[Tier, ...] = (Tier.HI, Tier.LO),
    ) -> dict[str, int]:
        """Materialize every requested host representation before inference."""
        counts = (
            [experts_per_layer] * num_layers
            if isinstance(experts_per_layer, int)
            else list(experts_per_layer)
        )
        if len(counts) != num_layers:
            raise ValueError("experts_per_layer length must match num_layers")
        total_bytes = 0
        entries = 0
        for layer, count in enumerate(counts):
            for expert in range(count):
                key = ExpertKey(layer, expert)
                for tier in tiers:
                    packed = self.load_weights(key, tier)
                    total_bytes += self._packed_nbytes(packed)
                    entries += 1
        return {"entries": entries, "host_packed_bytes": total_bytes}

    def preload_and_release_all(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
        tiers: tuple[Tier, ...] = (Tier.HI, Tier.LO),
    ) -> dict[str, int]:
        """Pack and release one layer at a time to bound peak host memory."""
        if set(tiers) != {Tier.HI, Tier.LO}:
            raise ValueError("source release requires both HI and LO tiers")
        counts = (
            [experts_per_layer] * num_layers
            if isinstance(experts_per_layer, int)
            else list(experts_per_layer)
        )
        if len(counts) != num_layers:
            raise ValueError("experts_per_layer length must match num_layers")
        total_bytes = 0
        released_bytes = 0
        entries = 0
        allocator_trim_calls = 0
        layer_seconds: list[float] = []
        layer_packed_bytes: list[int] = []
        layer_released_bytes: list[int] = []
        for layer, count in enumerate(counts):
            layer_start = time.perf_counter()
            packed_before = total_bytes
            container = self._get_expert_container(ExpertKey(layer, 0))
            if container is None:
                raise RuntimeError(f"cannot locate expert container for layer {layer}")
            fused_slots = self._fused_source_slots(container, count)
            has_partial_cache = any(
                (ExpertKey(layer, expert), tier) in self._packed_cache
                for expert in range(count)
                for tier in tiers
            )
            if (
                fused_slots is not None
                and self.backend == "reference"
                and not has_partial_cache
            ):
                packed_bytes, packed_entries = self._preload_fused_layer(
                    layer,
                    count,
                    fused_slots,
                    tiers,
                )
                total_bytes += packed_bytes
                entries += packed_entries
            else:
                for expert in range(count):
                    key = ExpertKey(layer, expert)
                    for tier in tiers:
                        packed = self.load_weights(key, tier)
                        total_bytes += self._packed_nbytes(packed)
                        entries += 1
            released_this_layer = self._release_container(
                layer,
                count,
                container,
            )
            released_bytes += released_this_layer
            allocator_trim_calls += int(_trim_released_host_memory())
            layer_seconds.append(time.perf_counter() - layer_start)
            layer_packed_bytes.append(total_bytes - packed_before)
            layer_released_bytes.append(released_this_layer)
        self.model = None
        return {
            "entries": entries,
            "host_packed_bytes": total_bytes,
            "released_native_expert_bytes": released_bytes,
            "host_allocator_trim_calls": allocator_trim_calls,
            "host_allocator_trim_attempts": num_layers,
            "layer_pack_release_seconds": layer_seconds,
            "layer_host_packed_bytes": layer_packed_bytes,
            "layer_released_native_expert_bytes": layer_released_bytes,
            "preload_release_seconds": sum(layer_seconds),
        }

    @staticmethod
    def _fused_source_slots(
        container,
        count: int,
    ) -> dict[str, torch.Tensor] | None:
        slots: dict[str, torch.Tensor] = {}
        for name in ("gate_up_proj", "down_proj"):
            value = getattr(container, name, None)
            if isinstance(value, torch.nn.Parameter):
                value = value.data
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                slots[name] = value
        if not slots:
            return None
        if set(slots) != {"gate_up_proj", "down_proj"}:
            raise RuntimeError("incomplete fused expert source layout")
        if any(value.shape[0] != count for value in slots.values()):
            raise RuntimeError("fused expert count disagrees with model config")
        return slots

    def _preload_fused_layer(
        self,
        layer: int,
        count: int,
        source_slots: dict[str, torch.Tensor],
        tiers: tuple[Tier, ...],
    ) -> tuple[int, int]:
        """Pack row-independent fused experts in bounded expert chunks.

        Reference group-wise quantization is independent for every output row.
        Flattening a bounded ``[experts, out, in]`` chunk to
        ``[experts*out, in]`` is therefore bit-identical to packing experts
        separately, while removing tens of thousands of small Python/Torch
        calls for wide MoE layers.
        """
        total_bytes = 0
        entries = 0
        for tier in tiers:
            target_fmt = self._tier_fmt(tier)
            group_size = (
                None
                if target_fmt == QuantFormat.FP16
                else DEFAULT_GROUP_SIZE[target_fmt]
            )
            expert_slots: list[dict[str, PackedTensor]] = [
                {} for _ in range(count)
            ]
            for slot, source in source_slots.items():
                per_expert_out = int(source.shape[1])
                in_features = int(source.shape[2])
                for start in range(0, count, self.fused_pack_chunk_experts):
                    end = min(start + self.fused_pack_chunk_experts, count)
                    raw = (
                        source[start:end]
                        .detach()
                        .to(device="cpu", dtype=torch.float16)
                        .contiguous()
                        .view((end - start) * per_expert_out, in_features)
                    )
                    batch = self._maybe_pin(
                        pack(
                            raw,
                            target_fmt,
                            group_size=group_size,
                            backend=self.backend,
                        )
                    )
                    per_expert_nbytes = compute_packed_nbytes(
                        per_expert_out,
                        in_features,
                        target_fmt,
                        (
                            in_features
                            if target_fmt == QuantFormat.FP16
                            else int(group_size)
                        ),
                    )
                    for expert in range(start, end):
                        local = expert - start
                        rows = slice(
                            local * per_expert_out,
                            (local + 1) * per_expert_out,
                        )
                        item = replace(
                            batch,
                            qweight=batch.qweight[rows],
                            scales=(
                                None
                                if batch.scales is None
                                else batch.scales[rows]
                            ),
                            out_features=per_expert_out,
                            nbytes=per_expert_nbytes,
                            resident_nbytes=0,
                        )
                        expert_slots[expert][slot] = (
                            self._with_resident_footprint(item)
                        )
            for expert, packed_slots in enumerate(expert_slots):
                if set(packed_slots) != {"gate_up_proj", "down_proj"}:
                    raise RuntimeError(
                        f"incomplete packed fused expert {layer}:{expert}"
                    )
                key = ExpertKey(layer, expert)
                cache_key = (key, tier)
                if cache_key in self._packed_cache:
                    raise RuntimeError(
                        f"duplicate fused cache entry for {key} {tier}"
                    )
                self._packed_cache[cache_key] = packed_slots
                total_bytes += self._packed_nbytes(packed_slots)
                entries += 1
        return total_bytes, entries

    def release_model_expert_sources(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
    ) -> int:
        """Release native expert parameters after both tiers are cached.

        Only fused Qwen-style containers and ModuleList-style, bias-free
        experts are modified. The method fails before mutation if any tier is
        missing, preventing a later transition from depending on a removed
        source tensor.
        """
        counts = (
            [experts_per_layer] * num_layers
            if isinstance(experts_per_layer, int)
            else list(experts_per_layer)
        )
        if len(counts) != num_layers:
            raise ValueError("experts_per_layer length must match num_layers")
        for layer, count in enumerate(counts):
            for expert in range(count):
                key = ExpertKey(layer, expert)
                for tier in (Tier.HI, Tier.LO):
                    if (key, tier) not in self._packed_cache:
                        raise RuntimeError(
                            f"cannot release native source before caching {key} {tier}"
                        )

        released_bytes = 0
        for layer, count in enumerate(counts):
            container = self._get_expert_container(ExpertKey(layer, 0))
            if container is None:
                raise RuntimeError(f"cannot locate expert container for layer {layer}")
            released_bytes += self._release_container(layer, count, container)

        # All later loads must resolve from the complete host cache.
        self.model = None
        return released_bytes

    @staticmethod
    def _release_container(layer: int, count: int, container) -> int:
        released_bytes = 0
        fused_parameters = []
        for slot in ("gate_up_proj", "down_proj"):
            value = getattr(container, slot, None)
            if isinstance(value, torch.nn.Parameter) and value.dim() == 3:
                fused_parameters.append((slot, value))
        if fused_parameters:
            if {name for name, _ in fused_parameters} != {
                "gate_up_proj",
                "down_proj",
            }:
                raise RuntimeError(f"incomplete fused expert layout in layer {layer}")
            for slot, value in fused_parameters:
                released_bytes += value.numel() * value.element_size()
                setattr(
                    container,
                    slot,
                    torch.nn.Parameter(
                        torch.empty(0, dtype=value.dtype, device="cpu"),
                        requires_grad=False,
                    ),
                )
            return released_bytes

        if not isinstance(container, torch.nn.ModuleList):
            raise RuntimeError(
                f"unsupported expert container {type(container).__name__} "
                f"in layer {layer}"
            )
        if len(container) != count:
            raise RuntimeError(
                f"expert count mismatch in layer {layer}: "
                f"container={len(container)}, config={count}"
            )
        for expert_module in container:
            autogptq_linears = [
                getattr(expert_module, slot)
                for slot in ("gate_proj", "up_proj", "down_proj")
                if ModelWeightStore._is_autogptq_linear(
                    getattr(expert_module, slot, None)
                )
            ]
            if autogptq_linears:
                if len(autogptq_linears) != 3:
                    raise RuntimeError(
                        f"incomplete AutoGPTQ expert in layer {layer}"
                    )
                if any(
                    getattr(linear, "bias", None) is not None
                    for linear in autogptq_linears
                ):
                    raise RuntimeError(
                        "AutoGPTQ expert biases are not represented by PackedTensor"
                    )
                for linear in autogptq_linears:
                    for name in ("qweight", "qzeros", "scales", "g_idx"):
                        value = getattr(linear, name, None)
                        if not isinstance(value, torch.Tensor):
                            continue
                        released_bytes += value.numel() * value.element_size()
                        empty = torch.empty(0, dtype=value.dtype, device="cpu")
                        if isinstance(value, torch.nn.Parameter):
                            empty = torch.nn.Parameter(
                                empty,
                                requires_grad=False,
                            )
                        setattr(linear, name, empty)
                continue

            linears = [
                getattr(expert_module, slot)
                for slot in (
                    "w1",
                    "w2",
                    "w3",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )
                if isinstance(getattr(expert_module, slot, None), torch.nn.Linear)
            ]
            if not linears:
                weight = getattr(expert_module, "weight", None)
                if isinstance(weight, torch.nn.Parameter) and weight.dim() == 2:
                    released_bytes += weight.numel() * weight.element_size()
                    expert_module.weight = torch.nn.Parameter(
                        torch.empty(0, dtype=weight.dtype, device="cpu"),
                        requires_grad=False,
                    )
                    continue
                raise RuntimeError(
                    f"unsupported expert module {type(expert_module).__name__}"
                )
            if any(linear.bias is not None for linear in linears):
                raise RuntimeError(
                    "expert biases are not represented by PackedTensor"
                )
            for linear in linears:
                weight = linear.weight
                released_bytes += weight.numel() * weight.element_size()
                linear.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=weight.dtype, device="cpu"),
                    requires_grad=False,
                )
        return released_bytes

    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        """Return the on-device byte footprint of ``(key, tier)``.

        Canonical transfer bytes come from ``compute_packed_nbytes``. When
        enabled, the CUDA INT4 kernel-native footprint replaces that layout
        after transfer, so this method returns the larger resident size.
        Any drift from the actual pool-backed representation would break the
        HBM envelope.
        """
        cache_key = (key, tier)
        cached = self._packed_cache.get(cache_key)
        if cached is not None:
            if isinstance(cached, dict):
                return sum(item.resident_nbytes for item in cached.values())
            return cached.resident_nbytes

        # Avoid eager quantization for sizing: derive the exact packed size
        # of every expert projection from its source shape.
        target_fmt = self._tier_fmt(tier)
        quantized_slots = self._fetch_autogptq_modules(key)
        if quantized_slots is not None:
            total = 0
            for module in quantized_slots.values():
                out_features = int(module.outfeatures)
                in_features = int(module.infeatures)
                group_size = (
                    in_features
                    if target_fmt == QuantFormat.FP16
                    else DEFAULT_GROUP_SIZE[target_fmt]
                )
                total += compute_packed_nbytes(
                    out_features,
                    in_features,
                    target_fmt,
                    group_size,
                )
                if (
                    self.enable_int4_kernel_cache
                    and target_fmt == QuantFormat.INT4
                ):
                    total += (
                        out_features
                        * (in_features // group_size)
                        * 2
                    )
            return total

        total = 0
        for raw in self._fetch_raw_slots(key).values():
            if raw.dim() != 2:
                raise ValueError(
                    f"expert {key} weight has shape {tuple(raw.shape)}; expected 2D"
                )
            out_features, in_features = raw.shape
            group_size = (
                in_features
                if target_fmt == QuantFormat.FP16
                else DEFAULT_GROUP_SIZE[target_fmt]
            )
            total += compute_packed_nbytes(
                out_features, in_features, target_fmt, group_size
            )
            if self.enable_int4_kernel_cache and target_fmt == QuantFormat.INT4:
                qweight_bytes = out_features * (in_features // 2)
                scales_bytes = (
                    out_features * (in_features // group_size) * 2
                )
                # The device block keeps only the kernel-native qweight and
                # (scale, zero) pairs; canonical transfer bytes are overwritten.
                total += scales_bytes
        return total

    def load_weights_multi(
        self, key: ExpertKey, tier: Tier, slot_names: list[str]
    ) -> dict[str, PackedTensor]:
        """
        Return a ``dict[slot_name, PackedTensor]`` for a multi-linear expert.

        Args:
            key: Expert key.
            tier: Target tier.
            slot_names: List of linear names (e.g. ``["w1", "w2", "w3"]``
                for Phi-MoE, ``["gate_up_proj", "down_proj"]`` for Qwen3).

        Uses per-slot raw overrides from ``register_expert_multi``, or
        falls back to named attributes on the expert module.
        """
        cache_key = (key, tier)
        cached = self._packed_multi_cache.get(cache_key)
        if cached is not None:
            return cached

        target_fmt = self._tier_fmt(tier)
        group_size = (
            None if target_fmt == QuantFormat.FP16 else DEFAULT_GROUP_SIZE[target_fmt]
        )

        result: dict[str, PackedTensor] = {}
        for slot in slot_names:
            raw = self._fetch_raw_weight_slot(key, slot)
            result[slot] = self._with_resident_footprint(
                pack(
                    raw.to(device="cpu", dtype=torch.float16),
                    target_fmt,
                    group_size=group_size,
                    backend=self.backend,
                )
            )

        self._packed_multi_cache[cache_key] = result
        return result

    def get_byte_size_multi(
        self, key: ExpertKey, tier: Tier, slot_names: list[str]
    ) -> int:
        """Total byte footprint of a multi-linear expert across all slots."""
        return sum(
            self._get_single_byte_size(key, tier, slot) for slot in slot_names
        )

    def register_expert(self, key: ExpertKey, weight: torch.Tensor) -> None:
        """Pre-register a raw fp16 weight for ``key`` (single-linear)."""
        if weight.dim() != 2:
            raise ValueError(
                f"register_expert expects 2D weight, got shape {tuple(weight.shape)}"
            )
        self._raw_overrides[key] = weight.to(torch.float16).contiguous()
        for stale_key in [k for k in self._packed_cache if k[0] == key]:
            self._packed_cache.pop(stale_key, None)

    def register_expert_multi(
        self, key: ExpertKey, weights: dict[str, torch.Tensor]
    ) -> None:
        """Pre-register per-slot raw fp16 weights for a multi-linear expert."""
        validated: dict[str, torch.Tensor] = {}
        for slot, w in weights.items():
            if w.dim() != 2:
                raise ValueError(
                    f"register_expert_multi: slot {slot!r} expects 2D, "
                    f"got shape {tuple(w.shape)}"
                )
            validated[slot] = w.to(torch.float16).contiguous()
        self._raw_multi_overrides[key] = validated
        for stale_key in [k for k in self._packed_cache if k[0] == key]:
            self._packed_cache.pop(stale_key, None)
        for stale_key in [k for k in self._packed_multi_cache if k[0] == key]:
            self._packed_multi_cache.pop(stale_key, None)

    def clear_cache(self) -> None:
        """Drop all cached PackedTensors. Raw overrides are preserved."""
        self._packed_cache.clear()
        self._packed_multi_cache.clear()

    @staticmethod
    def _packed_nbytes(
        packed: PackedTensor | dict[str, PackedTensor]
    ) -> int:
        return (
            sum(item.nbytes for item in packed.values())
            if isinstance(packed, dict)
            else packed.nbytes
        )

    def _maybe_pin(self, packed: PackedTensor) -> PackedTensor:
        """Pin a CPU representation when requested by the experiment."""
        if not self.pin_memory:
            return packed
        if packed.qweight.device.type != "cpu":
            raise RuntimeError("host packed representations must reside on CPU")
        return replace(
            packed,
            qweight=packed.qweight.pin_memory(),
            scales=(
                packed.scales.pin_memory()
                if packed.scales is not None
                else None
            ),
        )

    def _with_resident_footprint(self, packed: PackedTensor) -> PackedTensor:
        if not (
            self.enable_int4_kernel_cache
            and packed.fmt == QuantFormat.INT4
        ):
            return packed
        qweight_bytes = packed.qweight.numel() * packed.qweight.element_size()
        assert packed.scales is not None
        scales_bytes = packed.scales.numel() * packed.scales.element_size()
        return replace(
            packed,
            resident_nbytes=qweight_bytes + 2 * scales_bytes,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tier_fmt(self, tier: Tier) -> QuantFormat:
        return self.hi_fmt if tier == Tier.HI else self.lo_fmt

    def _fetch_raw_weight(self, key: ExpertKey) -> torch.Tensor:
        """Get a raw fp16 2-D weight tensor for ``key``."""
        override = self._raw_overrides.get(key)
        if override is not None:
            return override

        module = self._get_expert_module(key)
        if module is None:
            raise ValueError(
                f"Expert {key} not found in model and no override registered"
            )

        # Pick the largest 2-D float parameter as a stand-in for the
        # expert's "primary" linear. This works for any single-linear
        # synthetic test fixture and degrades gracefully for multi-linear
        # real experts (it returns one of gate/up/down — Phase 3 will
        # replace this with a multi-linear loader).
        candidates: list[torch.Tensor] = []
        for param in module.parameters():
            if param.dim() == 2 and param.is_floating_point():
                candidates.append(param.data)
        if not candidates:
            raise ValueError(
                f"No 2-D float parameters found for expert {key}; cannot pack"
            )
        weight = max(candidates, key=lambda p: p.numel())
        return weight.detach().to(torch.float16).contiguous()

    def _fetch_raw_slots(self, key: ExpertKey) -> dict[str, torch.Tensor]:
        """Return all runtime-relevant 2-D projections for one expert."""
        multi_override = self._raw_multi_overrides.get(key)
        if multi_override is not None:
            return multi_override
        override = self._raw_overrides.get(key)
        if override is not None:
            return {"weight": override}

        container = self._get_expert_container(key)
        if container is None:
            raise ValueError(
                f"Expert {key} not found in model and no override registered"
            )

        # Fused Qwen-style container: one 3-D tensor holds all experts.
        fused_slots: dict[str, torch.Tensor] = {}
        for slot in ("gate_up_proj", "down_proj"):
            if not hasattr(container, slot):
                continue
            value = getattr(container, slot)
            if isinstance(value, torch.nn.Parameter):
                value = value.data
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                if key.expert >= value.shape[0]:
                    raise ValueError(f"expert index out of range for {slot}: {key}")
                fused_slots[slot] = (
                    value[key.expert].detach().to(torch.float16).contiguous()
                )
        if fused_slots:
            if set(fused_slots) != {"gate_up_proj", "down_proj"}:
                raise ValueError(
                    f"Incomplete fused expert {key}; found slots {sorted(fused_slots)}"
                )
            return fused_slots

        # ModuleList-style container: resolve the individual expert module.
        expert = None
        if isinstance(container, (torch.nn.ModuleList, list, tuple)):
            if key.expert < len(container):
                expert = container[key.expert]
        elif isinstance(container, torch.nn.Module):
            try:
                expert = container[key.expert]  # type: ignore[index]
            except (IndexError, KeyError, TypeError):
                expert = container
        if expert is None:
            raise ValueError(f"Expert {key} not found in expert container")

        known_layouts = (
            ("w1", "w2", "w3"),
            ("gate_up_proj", "down_proj"),
            ("gate_proj", "up_proj", "down_proj"),
            ("weight",),
        )
        for layout in known_layouts:
            slots: dict[str, torch.Tensor] = {}
            for slot in layout:
                value = getattr(expert, slot, None)
                if isinstance(value, torch.nn.Linear):
                    value = value.weight
                if isinstance(value, torch.nn.Parameter):
                    value = value.data
                if isinstance(value, torch.Tensor) and value.dim() == 2:
                    slots[slot] = value.detach().to(torch.float16).contiguous()
            if len(slots) == len(layout):
                if layout == ("gate_proj", "up_proj", "down_proj"):
                    return {
                        "gate_up_proj": torch.cat(
                            (slots["gate_proj"], slots["up_proj"]), dim=0
                        ).contiguous(),
                        "down_proj": slots["down_proj"],
                    }
                return slots

        # Last-resort single-linear fixture support.
        candidates = [
            parameter.detach().to(torch.float16).contiguous()
            for parameter in expert.parameters()
            if parameter.dim() == 2 and parameter.is_floating_point()
        ]
        if len(candidates) == 1:
            return {"weight": candidates[0]}
        raise ValueError(
            f"Cannot identify a complete expert projection layout for {key} "
            f"({type(expert).__name__})"
        )

    @staticmethod
    def _is_autogptq_linear(module: object) -> bool:
        """Whether ``module`` exposes the audited AutoGPTQ matrix contract."""
        return (
            isinstance(module, torch.nn.Module)
            and isinstance(getattr(module, "qweight", None), torch.Tensor)
            and isinstance(getattr(module, "qzeros", None), torch.Tensor)
            and isinstance(getattr(module, "scales", None), torch.Tensor)
            and isinstance(getattr(module, "bits", None), int)
            and isinstance(getattr(module, "group_size", None), int)
            and isinstance(getattr(module, "infeatures", None), int)
            and isinstance(getattr(module, "outfeatures", None), int)
        )

    def _fetch_autogptq_modules(
        self,
        key: ExpertKey,
    ) -> dict[str, torch.nn.Module] | None:
        """Return an unfused AutoRound expert's three quantized projections."""
        container = self._get_expert_container(key)
        if not isinstance(container, (torch.nn.ModuleList, list, tuple)):
            return None
        if key.expert >= len(container):
            raise ValueError(f"Expert {key} not found in expert container")
        expert = container[key.expert]
        slots = {
            slot: getattr(expert, slot, None)
            for slot in ("gate_proj", "up_proj", "down_proj")
        }
        recognized = {
            slot: module
            for slot, module in slots.items()
            if self._is_autogptq_linear(module)
        }
        if not recognized:
            return None
        if set(recognized) != set(slots):
            raise ValueError(
                f"Incomplete AutoGPTQ expert {key}; found slots "
                f"{sorted(recognized)}"
            )
        return recognized

    def _fetch_raw_weight_slot(self, key: ExpertKey, slot: str) -> torch.Tensor:
        """Get a raw fp16 2-D weight tensor for a specific slot of ``key``."""
        multi_override = self._raw_multi_overrides.get(key)
        if multi_override is not None:
            w = multi_override.get(slot)
            if w is not None:
                return w
            raise ValueError(
                f"Expert {key} multi-override has no slot {slot!r}; "
                f"available: {list(multi_override.keys())}"
            )

        module = self._get_expert_module(key)
        if module is None:
            raise ValueError(
                f"Expert {key} not found in model and no multi-override registered"
            )

        # Try named attribute (nn.Linear or nn.Parameter).
        if hasattr(module, slot):
            attr = getattr(module, slot)
            if isinstance(attr, torch.nn.Linear):
                return attr.weight.detach().to(torch.float16).contiguous()
            if isinstance(attr, torch.nn.Parameter):
                return attr.data.detach().to(torch.float16).contiguous()
            if isinstance(attr, torch.Tensor):
                return attr.detach().to(torch.float16).contiguous()

        raise ValueError(
            f"Expert {key} module {type(module).__name__} has no attribute "
            f"{slot!r}; cannot load multi-linear weight"
        )

    def _get_single_byte_size(
        self, key: ExpertKey, tier: Tier, slot: str
    ) -> int:
        """Byte size of one slot of a multi-linear expert."""
        raw = self._fetch_raw_weight_slot(key, slot)
        out_features, in_features = raw.shape
        target_fmt = self._tier_fmt(tier)
        group_size = (
            in_features
            if target_fmt == QuantFormat.FP16
            else DEFAULT_GROUP_SIZE[target_fmt]
        )
        base = compute_packed_nbytes(
            out_features, in_features, target_fmt, group_size
        )
        if self.enable_int4_kernel_cache and target_fmt == QuantFormat.INT4:
            qweight_bytes = out_features * (in_features // 2)
            scales_bytes = out_features * (in_features // group_size) * 2
            return qweight_bytes + 2 * scales_bytes
        return base

    def _get_expert_module(self, key: ExpertKey) -> Optional[torch.nn.Module]:
        """
        Walk ``self.model`` to find the expert module for ``key``.

        Tries the standard HF MoE layout (`layers.{layer}.experts[{expert}]`)
        first, then falls back to a name-based scan. Returns ``None`` if no
        match is found — the caller is responsible for raising.
        """
        if self.model is None:
            return None

        layer_name = f"layers.{key.layer}"
        try:
            layer = self.model.get_submodule(layer_name)
        except (AttributeError, KeyError):
            layer = None

        if layer is not None and hasattr(layer, "experts"):
            experts = layer.experts
            if isinstance(experts, torch.nn.ModuleList):
                if key.expert < len(experts):
                    return experts[key.expert]
            elif hasattr(experts, f"experts.{key.expert}"):
                return getattr(experts, f"experts.{key.expert}")

        # Fallback: name-based scan (slow, kept for legacy fixtures).
        for name, module in self.model.named_modules():
            n = name.lower()
            if f"layer.{key.layer}" in n and f"expert.{key.expert}" in n:
                return module
        return None

    def _get_expert_container(self, key: ExpertKey):
        """Locate the layer's expert container across supported HF layouts."""
        if self.model is None:
            return None
        layer = None
        for path in (
            f"model.layers.{key.layer}",
            f"layers.{key.layer}",
            f"transformer.layers.{key.layer}",
        ):
            try:
                layer = self.model.get_submodule(path)
                break
            except (AttributeError, KeyError):
                continue
        if layer is None:
            return None
        for path in (
            "mlp.experts",
            "block_sparse_moe.experts",
            "moe.experts",
            "experts",
        ):
            try:
                return layer.get_submodule(path)
            except (AttributeError, KeyError):
                # ``experts`` may be a Parameter-owning module that older
                # PyTorch get_submodule implementations do not resolve.
                current = layer
                try:
                    for part in path.split("."):
                        current = getattr(current, part)
                    return current
                except AttributeError:
                    continue
        return None
