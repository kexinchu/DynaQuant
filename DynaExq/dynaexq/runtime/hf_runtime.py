from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import (
    Bitwidth,
    DualPrecisionWeights,
    ExpertID,
    ExpertMonitor,
    MemoryManager,
    PrecisionController,
    SwapConfig,
    SwapEngine,
)
from .memmgr import PoolConfig
from .types import Residency
from .weights import InMemoryWeightStore, SSDWeightStore


LOGGER = logging.getLogger("dynaexq.runtime.hf")


@dataclass
class RuntimeStats:
    upgrades: int = 0
    downgrades: int = 0
    skipped_upgrades: int = 0
    skipped_downgrades: int = 0
    monitor_updates: int = 0
    controller_calls: int = 0


@dataclass
class HFRuntimeConfig:
    layer_hot_limits: Dict[int, int]
    per_layer_hot_bytes: Dict[int, int]
    per_layer_cold_bytes: Dict[int, int]
    layer_total_counts: Dict[int, int]
    total_available_hot_bytes: int
    tau_hot: float = 0.65
    tau_cold: float = 0.45
    epoch_interval_s: float = 600.0


TaskType = Tuple[ExpertID, Bitwidth, Optional[threading.Event]]


class ExpertSwapWorker:
    """Background worker that serializes expert swap operations."""

    def __init__(
        self,
        swap_engine: SwapEngine,
        on_success: Callable[[ExpertID, Bitwidth, Residency], None],
        on_failure: Callable[[ExpertID, Bitwidth, Exception], None],
    ) -> None:
        self._swap_engine = swap_engine
        self._on_success = on_success
        self._on_failure = on_failure
        self._queue: "queue.Queue[Optional[TaskType]]" = queue.Queue()
        self._pending: Dict[ExpertID, threading.Event] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._thread.start()

    def submit(
        self,
        expert: ExpertID,
        target: Bitwidth,
        ready_event: Optional[threading.Event] = None,
    ) -> threading.Event:
        with self._lock:
            event = self._pending.get(expert)
            if event is None:
                event = threading.Event()
                self._pending[expert] = event
            else:
                return event
        self._queue.put((expert, target, ready_event))
        return event

    def flush(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break

            expert, target, ready_event = task
            try:
                if ready_event is not None:
                    ready_event.wait()

                if target is Bitwidth.W4:
                    self._swap_engine.upgrade(expert)
                else:
                    self._swap_engine.downgrade(expert)
                residency = self._swap_engine.wait_ready(expert)
            except Exception as exc:  # pylint: disable=broad-except
                self._on_failure(expert, target, exc)
            else:
                assert isinstance(residency, Residency)
                self._on_success(expert, target, residency)
            finally:
                with self._lock:
                    event = self._pending.pop(expert, None)
                if event is not None:
                    event.set()
                self._queue.task_done()


class ExpertPrefetchWorker:
    """Dedicated worker to stage expert payloads from SSD into DRAM."""

    def __init__(self, store) -> None:
        self._store = store
        self._enabled = getattr(store, "supports_prefetch", False)
        self._queue: "queue.Queue[Optional[Tuple[ExpertID, Bitwidth]]]" = queue.Queue(
        )
        self._pending: Dict[Tuple[int, int, str], threading.Event] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._thread.start()

    def schedule(self, expert: ExpertID, bitwidth: Bitwidth) -> threading.Event:
        event = threading.Event()
        if not self._enabled:
            event.set()
            return event

        key = (expert.layer, expert.idx, bitwidth.value)
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None:
                return existing
            self._pending[key] = event
        self._queue.put((expert, bitwidth))
        return event

    def flush(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break

            expert, bitwidth = task
            key = (expert.layer, expert.idx, bitwidth.value)
            try:
                self._store.prefetch(expert, bitwidth)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "Prefetch for expert %s (%s) failed: %s",
                    expert,
                    bitwidth.value,
                    exc,
                )
            finally:
                with self._lock:
                    event = self._pending.pop(key, None)
                if event is not None:
                    event.set()
                self._queue.task_done()


class HuggingFaceDynaExQ:
    """Attach DynaExQ runtime management to a HuggingFace MoE model."""

    def __init__(
        self,
        *,
        weight_store,
        all_experts: Sequence[ExpertID],
        device: torch.device,
        dtype: torch.dtype,
        config: HFRuntimeConfig,
    ) -> None:
        self._store = weight_store
        self._all_experts = list(all_experts)
        self._device = device
        self._dtype = dtype
        self._config = config

        self._model: Optional[torch.nn.Module] = None
        self._parameter_cache: Dict[str, torch.nn.Parameter] = {}
        self._buffer_cache: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._scheduler_stop = threading.Event()
        self._epoch_interval = max(self._config.epoch_interval_s, 60.0)
        self._scheduler_poll = min(max(self._epoch_interval / 20.0, 1.0), 15.0)
        self._next_epoch_deadline = time.time() + self._epoch_interval
        self._epoch_id = 0

        self._monitor = ExpertMonitor()
        total_hot_slots = max(
            1, sum(max(limit, 0) for limit in config.layer_hot_limits.values())
        )
        self._controller = PrecisionController(
            tau_h=config.tau_hot,
            tau_c=config.tau_cold,
            max_w4_slots=total_hot_slots,
            layer_cap=dict(config.layer_hot_limits),
        )

        pool_config = self._build_pool_config()
        self._memory = MemoryManager(pool_config)
        self._swap_engine = SwapEngine(
            self._memory,
            self._store,
            SwapConfig(max_workers=1),
        )

        self._current_bitwidth: Dict[ExpertID, Bitwidth] = {}
        self._initial_low_precision: Set[ExpertID] = set()
        self._stats = RuntimeStats()
        self._inflight: Set[ExpertID] = set()

        self._prefetch_worker: Optional[ExpertPrefetchWorker] = (
            ExpertPrefetchWorker(self._store)
            if getattr(self._store, "supports_prefetch", False)
            else None
        )
        self._swap_worker = ExpertSwapWorker(
            self._swap_engine,
            self._on_swap_success,
            self._on_swap_failure,
        )
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self._scheduler_thread.start()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> RuntimeStats:
        return RuntimeStats(
            upgrades=self._stats.upgrades,
            downgrades=self._stats.downgrades,
            skipped_upgrades=self._stats.skipped_upgrades,
            skipped_downgrades=self._stats.skipped_downgrades,
            monitor_updates=self._stats.monitor_updates,
            controller_calls=self._stats.controller_calls,
        )

    @property
    def initial_low_precision(self) -> Set[ExpertID]:
        return set(self._initial_low_precision)

    def export_stats(self) -> Dict[str, int]:
        snapshot = self.stats
        return {
            "upgrades": snapshot.upgrades,
            "downgrades": snapshot.downgrades,
            "skipped_upgrades": snapshot.skipped_upgrades,
            "skipped_downgrades": snapshot.skipped_downgrades,
            "monitor_updates": snapshot.monitor_updates,
            "controller_plan_calls": snapshot.controller_calls,
        }

    def attach(self, model: torch.nn.Module) -> None:
        self._model = model
        self._parameter_cache.clear()
        self._buffer_cache.clear()
        for name, param in model.named_parameters():
            self._parameter_cache[name] = param
        for name, buffer in model.named_buffers():
            self._buffer_cache[name] = buffer
        self._register_router_hooks(model)

    def set_initial_precisions(self, downgraded: Iterable[ExpertID]) -> None:
        downgraded_set = set(downgraded)
        self._initial_low_precision = downgraded_set
        for expert in self._all_experts:
            self._current_bitwidth.setdefault(expert, Bitwidth.W4)
        for expert in downgraded_set:
            self._current_bitwidth[expert] = Bitwidth.W2

    def shutdown(self) -> None:
        self._scheduler_stop.set()
        self._scheduler_thread.join()
        if self._prefetch_worker is not None:
            self._prefetch_worker.flush()
            self._prefetch_worker.stop()
        self._swap_worker.flush()
        self._swap_worker.stop()
        self._swap_engine.close()

    def epoch_tick(self) -> None:
        self._monitor.epoch_tick()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_pool_config(self) -> PoolConfig:
        hot_slot_size = max(
            [self._config.per_layer_hot_bytes.get(
                layer, 1) for layer in self._config.layer_hot_limits]
            or [self._config.total_available_hot_bytes, 1]
        )
        total_hot_slots = max(
            sum(self._config.layer_hot_limits.values()), 0) + 1
        hot_capacity = max(
            self._config.total_available_hot_bytes,
            total_hot_slots * hot_slot_size,
        )

        cold_slot_size = max(
            [self._config.per_layer_cold_bytes.get(
                layer, hot_slot_size) for layer in self._config.layer_total_counts]
            or [hot_slot_size]
        )
        total_cold_slots = max(
            sum(self._config.layer_total_counts.values()), 0) + 1
        cold_capacity = max(total_cold_slots * cold_slot_size,
                            hot_capacity * 2)

        transient_capacity = max(hot_slot_size, cold_slot_size)

        return PoolConfig(
            hot_capacity_bytes=int(hot_capacity),
            cold_capacity_bytes=int(cold_capacity),
            transient_capacity_bytes=int(transient_capacity),
            hot_slots=int(total_hot_slots),
            hot_slot_bytes=int(hot_slot_size),
            cold_slots=int(total_cold_slots),
            cold_slot_bytes=int(cold_slot_size),
        )

    def _register_router_hooks(self, model: torch.nn.Module) -> None:
        layers = self._locate_moe_layers(model)
        if not layers:
            LOGGER.warning(
                "Unable to locate MoE router modules; DynaExQ runtime hooks are disabled."
            )
            return

        for layer_idx, gate_module, top_k in layers:
            gate_module.register_forward_hook(
                self._make_router_hook(layer_idx, top_k))
            LOGGER.debug(
                "Attached router hook to layer %d using %s (top_k=%d)",
                layer_idx,
                gate_module.__class__.__name__,
                top_k,
            )

    def _locate_moe_layers(
        self, model: torch.nn.Module
    ) -> List[Tuple[int, torch.nn.Module, int]]:
        result: List[Tuple[int, torch.nn.Module, int]] = []
        layer_sequence: Optional[Sequence[torch.nn.Module]] = None

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer_sequence = model.model.layers  # type: ignore[attr-defined]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            # type: ignore[attr-defined]
            layer_sequence = model.transformer.layers

        if layer_sequence is None:
            return result

        for layer_idx, layer in enumerate(layer_sequence):
            mlp = getattr(layer, "mlp", layer)
            gate = None
            for attr_name in ("gate", "router", "routing", "gating"):
                gate = getattr(mlp, attr_name, None)
                if gate is not None:
                    break

            if gate is None:
                continue

            fallback_topk = 1
            raw_topk = getattr(gate, "top_k", fallback_topk)
            try:
                top_k = int(raw_topk)
            except (TypeError, ValueError):
                top_k = fallback_topk
            if top_k <= 0:
                top_k = fallback_topk

            result.append((layer_idx, gate, top_k))

        return result

    def _make_router_hook(self, layer_idx: int, top_k: int):
        def _hook(module, inputs, outputs):
            del module, inputs  # unused
            with torch.no_grad():
                logits, indices = self._extract_router_outputs(outputs, top_k)
                if logits is None or indices is None:
                    return outputs
                self._handle_router_event(layer_idx, logits, indices)
            return outputs

        return _hook

    def _extract_router_outputs(
        self,
        outputs,
        top_k: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        logits: Optional[torch.Tensor] = None
        indices: Optional[torch.Tensor] = None

        if isinstance(outputs, dict):
            logits = outputs.get("logits") or outputs.get("router_logits")
            indices = outputs.get("indices") or outputs.get("topk_indices")
        elif isinstance(outputs, (tuple, list)):
            if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
                logits = outputs[0]
                indices = outputs[1]
            elif outputs:
                logits = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            logits = outputs

        if logits is None:
            return None, None

        if indices is None:
            k = min(top_k, logits.shape[-1])
            values, new_indices = torch.topk(logits, k=k, dim=-1)
            logits = values
            indices = new_indices

        return logits, indices

    def _handle_router_event(
        self,
        layer: int,
        logits: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        logits_cpu = logits.detach().to("cpu", dtype=torch.float32)
        indices_cpu = indices.detach().to("cpu", dtype=torch.int32)

        if logits_cpu.ndim == 1:
            logits_cpu = logits_cpu.unsqueeze(0)
        if indices_cpu.ndim == 1:
            indices_cpu = indices_cpu.unsqueeze(0)

        np_logits = logits_cpu.numpy()
        np_indices = indices_cpu.numpy()

        active_experts = [
            ExpertID(layer=layer, idx=int(idx))
            for idx in np.unique(np_indices)
        ]
        if not active_experts:
            return

        self._stats.monitor_updates += 1
        self._monitor.update_batch(layer, np_indices, np_logits)

    def _synchronize_experts(self, targets: Dict[ExpertID, Bitwidth]) -> None:
        upgrades: List[ExpertID] = []
        downgrades: List[ExpertID] = []

        with self._lock:
            for expert, desired in targets.items():
                current = self._current_bitwidth.get(expert, Bitwidth.W4)
                if desired is Bitwidth.W4 and current is not Bitwidth.W4:
                    if expert not in self._inflight:
                        self._inflight.add(expert)
                        upgrades.append(expert)
                elif desired is Bitwidth.W2 and current is not Bitwidth.W2:
                    if expert not in self._inflight:
                        self._inflight.add(expert)
                        downgrades.append(expert)

        if not upgrades and not downgrades:
            return

        upgrades.sort(key=lambda e: (e.layer, e.idx))
        downgrades.sort(key=lambda e: (e.layer, e.idx))

        sequence: List[Tuple[ExpertID, Bitwidth]] = []
        up_idx = 0
        down_idx = 0
        last_op: Optional[str] = None

        while up_idx < len(upgrades) or down_idx < len(downgrades):
            take_upgrade = (
                (last_op != "upgrade" and up_idx < len(upgrades))
                or down_idx >= len(downgrades)
            )
            if take_upgrade:
                sequence.append((upgrades[up_idx], Bitwidth.W4))
                up_idx += 1
                last_op = "upgrade"
            else:
                sequence.append((downgrades[down_idx], Bitwidth.W2))
                down_idx += 1
                last_op = "downgrade"

        ready_events: Dict[ExpertID, threading.Event] = {}
        if self._prefetch_worker is not None:
            for expert, target in sequence:
                ready_events[expert] = self._prefetch_worker.schedule(
                    expert, target
                )
        else:
            for expert, _ in sequence:
                event = threading.Event()
                event.set()
                ready_events[expert] = event

        for expert, target in sequence:
            self._swap_worker.submit(
                expert,
                target,
                ready_events.get(expert),
            )

    def _apply_residency(self, expert: ExpertID, residency: Residency) -> None:
        bundle = residency.tensor_bundle
        if bundle is None:
            return

        tensors = bundle.materialize(device=self._device)
        with torch.no_grad():
            for name, tensor in tensors.items():
                target_param = self._parameter_cache.get(name)
                if target_param is not None:
                    cast_tensor = tensor.to(
                        self._device, dtype=target_param.dtype, copy=False)
                    target_param.data.copy_(cast_tensor)
                    continue
                target_buffer = self._buffer_cache.get(name)
                if target_buffer is not None:
                    cast_tensor = tensor.to(
                        self._device, dtype=target_buffer.dtype, copy=False)
                    target_buffer.data.copy_(cast_tensor)
        self._current_bitwidth[expert] = residency.bitwidth

    def _on_swap_success(
        self,
        expert: ExpertID,
        target: Bitwidth,
        residency: Residency,
    ) -> None:
        self._apply_residency(expert, residency)
        if target is Bitwidth.W4:
            self._stats.upgrades += 1
        else:
            self._stats.downgrades += 1
        with self._lock:
            self._inflight.discard(expert)

    def _on_swap_failure(
        self,
        expert: ExpertID,
        target: Bitwidth,
        error: Exception,
    ) -> None:
        if target is Bitwidth.W4:
            self._stats.skipped_upgrades += 1
        else:
            self._stats.skipped_downgrades += 1
        LOGGER.warning("Swap operation for %s (%s) failed: %s",
                       expert, target.value, error)
        with self._lock:
            self._inflight.discard(expert)

    def _run_epoch_cycle(self) -> None:
        with self._lock:
            expert_snapshot = list(self._all_experts)
            self._controller.reset()
        if not expert_snapshot:
            return
        self._stats.controller_calls += 1
        targets = self._controller.plan(expert_snapshot, self._monitor)
        self._synchronize_experts(targets)
        self._monitor.reset_all()

    def _scheduler_loop(self) -> None:
        while not self._scheduler_stop.wait(self._scheduler_poll):
            now = time.time()
            if now < self._next_epoch_deadline:
                continue
            self._next_epoch_deadline = now + self._epoch_interval
            self._epoch_id += 1
            self._run_epoch_cycle()


def _cast_floating_parameters(module: torch.nn.Module, dtype: torch.dtype) -> None:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=dtype)
    for buffer in module.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=dtype)


def _tensor_collection_nbytes(tensors: Iterable[torch.Tensor]) -> int:
    return sum(int(t.element_size() * t.numel()) for t in tensors)


def _collect_layer_metadata(
    weights: DualPrecisionWeights,
) -> Tuple[Dict[int, Dict[str, int]], List[ExpertID]]:
    layer_counts: Dict[int, int] = {}
    layer_hot_bytes: Dict[int, int] = {}
    layer_cold_bytes: Dict[int, int] = {}
    expert_map: Dict[Tuple[int, int], ExpertID] = {}

    for index in weights.indices():
        expert = index.expert
        if expert is None:
            continue

        key = (expert.layer, expert.idx)
        if key not in expert_map:
            expert_map[key] = expert
            layer_counts[expert.layer] = layer_counts.get(expert.layer, 0) + 1

        if index.bitwidth is Bitwidth.W4 and expert.layer not in layer_hot_bytes:
            bundle = weights.expert_bundle(expert, Bitwidth.W4)
            layer_hot_bytes[expert.layer] = bundle.nbytes

        if index.bitwidth is Bitwidth.W2 and expert.layer not in layer_cold_bytes:
            try:
                bundle = weights.expert_bundle(expert, Bitwidth.W2)
            except KeyError:
                continue
            layer_cold_bytes[expert.layer] = bundle.nbytes

    metadata: Dict[int, Dict[str, int]] = {}
    for layer, count in layer_counts.items():
        hot_bytes = layer_hot_bytes.get(layer)
        cold_bytes = layer_cold_bytes.get(layer)
        if hot_bytes is None and cold_bytes is not None:
            hot_bytes = cold_bytes
        if cold_bytes is None and hot_bytes is not None:
            cold_bytes = hot_bytes
        if hot_bytes is None:
            hot_bytes = 1
        if cold_bytes is None:
            cold_bytes = hot_bytes
        metadata[layer] = {
            "count": int(count),
            "hot_bytes": int(max(hot_bytes, 1)),
            "cold_bytes": int(max(cold_bytes, 1)),
        }

    all_experts = sorted(expert_map.values(), key=lambda e: (e.layer, e.idx))
    return metadata, all_experts


def _distribute_hot_slots(
    metadata: Dict[int, Dict[str, int]],
    available_bytes: int,
) -> Dict[int, int]:
    if not metadata:
        return {}

    layer_limits = {layer: 0 for layer in metadata}
    ordered = sorted(metadata.items(), key=lambda item: item[1]["hot_bytes"])
    remaining = max(int(available_bytes), 0)

    if remaining <= 0:
        smallest_layer, info = ordered[0]
        if info["count"] > 0:
            layer_limits[smallest_layer] = 1
        return layer_limits

    while remaining > 0:
        progress = False
        for layer, info in ordered:
            if layer_limits[layer] >= info["count"]:
                continue
            needed = max(info["hot_bytes"], 1)
            if remaining < needed:
                continue
            layer_limits[layer] += 1
            remaining -= needed
            progress = True
        if not progress:
            break

    if all(limit == 0 for limit in layer_limits.values()):
        smallest_layer, info = ordered[0]
        if info["count"] > 0:
            layer_limits[smallest_layer] = 1

    return layer_limits


def _build_ssd_repository(
    weights: DualPrecisionWeights,
    experts: Sequence[ExpertID],
    base_dir: Path,
) -> SSDWeightStore:
    base_dir.mkdir(parents=True, exist_ok=True)
    data_path = base_dir / "experts.bin"
    index: Dict[str, Dict[str, int]] = {}
    offset = 0

    with data_path.open("wb") as handle:
        for bitwidth in (Bitwidth.W4, Bitwidth.W2):
            for expert in experts:
                try:
                    bundle = weights.expert_bundle(expert, bitwidth)
                except KeyError:
                    continue
                buffer = io.BytesIO()
                torch.save(bundle.tensors, buffer)
                blob = buffer.getvalue()
                size = len(blob)
                if size <= 0:
                    continue
                handle.write(blob)
                key = f"{expert.layer}:{expert.idx}:{bitwidth.value}"
                index[key] = {"offset": offset, "size": size}
                offset += size

    index_path = base_dir / "experts_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle)

    return SSDWeightStore(data_path, index)


def infer_runtime_config(
    metadata: Dict[int, Dict[str, int]],
    non_expert_bytes: int,
    device: torch.device,
) -> HFRuntimeConfig:
    total_memory = 0
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_memory = int(props.total_memory)

    reserve = int(total_memory * 0.2)
    usable = max(total_memory - reserve, 0)
    if usable > 0:
        available_hot_bytes = max(usable - non_expert_bytes, 0)
    else:
        available_hot_bytes = sum(info["hot_bytes"]
                                  for info in metadata.values())

    if available_hot_bytes <= 0:
        available_hot_bytes = sum(info["hot_bytes"]
                                  for info in metadata.values())

    layer_limits = _distribute_hot_slots(metadata, available_hot_bytes)
    per_layer_hot_bytes = {
        layer: info["hot_bytes"] for layer, info in metadata.items()}
    per_layer_cold_bytes = {
        layer: info["cold_bytes"] for layer, info in metadata.items()}
    layer_total_counts = {
        layer: info["count"] for layer, info in metadata.items()}

    return HFRuntimeConfig(
        layer_hot_limits=layer_limits,
        per_layer_hot_bytes=per_layer_hot_bytes,
        per_layer_cold_bytes=per_layer_cold_bytes,
        layer_total_counts=layer_total_counts,
        total_available_hot_bytes=int(available_hot_bytes),
    )


def load_model_with_dynaexq(
    *,
    fp16_path: str,
    int4_path: str,
    requested_low_precision: Iterable[ExpertID],
    device: torch.device,
    dtype: torch.dtype,
    use_ssd: bool = False,
    ssd_directory: Optional[str] = None,
    trust_remote_code: bool = False,
) -> tuple[
    AutoModelForCausalLM,
    AutoTokenizer,
    HuggingFaceDynaExQ,
    Set[ExpertID],
    HFRuntimeConfig,
]:
    LOGGER.info("Building dual-precision repository (W4: %s, W2: %s)",
                fp16_path, int4_path)
    weights = DualPrecisionWeights.from_files(
        fp16_path,
        int4_path,
        map_location="cpu",
    )

    metadata, all_experts = _collect_layer_metadata(weights)
    non_expert_bytes = _tensor_collection_nbytes(
        weights.non_expert_state().values())
    runtime_config = infer_runtime_config(metadata, non_expert_bytes, device)

    requested_set = set(requested_low_precision)
    available_low_precision: Set[ExpertID] = set()
    for expert in sorted(requested_set, key=lambda e: (e.layer, e.idx)):
        try:
            weights.expert_bundle(expert, Bitwidth.W2)
        except KeyError:
            LOGGER.warning(
                "Low-precision weights not found for expert %s; keeping W4.",
                expert,
            )
        else:
            available_low_precision.add(expert)

    if use_ssd:
        cache_dir = Path(ssd_directory) if ssd_directory else Path(
            int4_path) / "dynaexq_ssd_cache"
        store = _build_ssd_repository(weights, all_experts, cache_dir)
    else:
        store = InMemoryWeightStore(weights, Bitwidth.W4)

    LOGGER.info("Materializing mixed-precision state_dict in memory")

    def selector(expert_id: ExpertID) -> Bitwidth:
        return Bitwidth.W2 if expert_id in available_low_precision else Bitwidth.W4

    state_dict = weights.materialize_state_dict(
        expert_precision=selector,
    )

    LOGGER.info("Loading model config from %s", fp16_path)
    model_config = AutoConfig.from_pretrained(
        fp16_path,
        trust_remote_code=trust_remote_code,
    )
    LOGGER.info("Instantiating model from config")
    model = AutoModelForCausalLM.from_config(
        model_config,
        trust_remote_code=trust_remote_code,
    )

    LOGGER.info("Applying mixed-precision weights (%d tensors)",
                len(state_dict))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing parameters during load: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected parameters during load: %s", unexpected)
    del state_dict

    tokenizer = AutoTokenizer.from_pretrained(
        fp16_path, trust_remote_code=trust_remote_code)

    model.to(device=device)
    _cast_floating_parameters(model, dtype)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_ssd:
        weights.release_expert_storage()

    runtime = HuggingFaceDynaExQ(
        weight_store=store,
        all_experts=all_experts,
        device=device,
        dtype=dtype,
        config=runtime_config,
    )
    runtime.attach(model)
    runtime.set_initial_precisions(available_low_precision)

    return model, tokenizer, runtime, available_low_precision, runtime_config


__all__ = [
    "HFRuntimeConfig",
    "HuggingFaceDynaExQ",
    "RuntimeStats",
    "infer_runtime_config",
    "load_model_with_dynaexq",
]
