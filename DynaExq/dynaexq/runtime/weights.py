from __future__ import annotations

import io
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import torch

from .types import Bitwidth, ExpertID

try:  # pragma: no cover - optional dependency
    from safetensors.torch import load_file as safetensors_load
    _HAVE_SAFETENSORS = True
except Exception:  # pragma: no cover - fallback when safetensors missing
    safetensors_load = None
    _HAVE_SAFETENSORS = False


@dataclass(frozen=True, slots=True)
class ParameterIndex:
    """Unique identifier for a parameter view stored in memory."""

    name: str
    bitwidth: Bitwidth
    expert: Optional[ExpertID] = None


@dataclass
class ExpertTensorBundle:
    """Container holding all parameter tensors for one expert."""

    tensors: Dict[str, torch.Tensor]

    @property
    def nbytes(self) -> int:
        return sum(t.element_size() * t.numel() for t in self.tensors.values())

    def materialize(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        if device is None:
            return self.tensors
        return {name: tensor.to(device) for name, tensor in self.tensors.items()}


def _default_expert_patterns() -> Tuple[re.Pattern[str], ...]:
    patterns = [
        r"(?:^|\.)(?P<layer>\d+)\.experts\.(?P<expert>\d+)",
        r"experts(?:\.layers)?\.(?P<layer>\d+)\.(?P<expert>\d+)",
    ]
    return tuple(re.compile(expr) for expr in patterns)


def _compile_patterns(patterns: Optional[Iterable[str]]) -> Tuple[re.Pattern[str], ...]:
    if patterns is None:
        return _default_expert_patterns()
    return tuple(re.compile(expr) for expr in patterns)


def _parse_expert(name: str, patterns: Tuple[re.Pattern[str], ...]) -> Optional[ExpertID]:
    for pattern in patterns:
        match = pattern.search(name)
        if match:
            layer = int(match.group("layer"))
            expert_idx = int(match.group("expert"))
            return ExpertID(layer=layer, idx=expert_idx)
    return None


class DualPrecisionWeights:
    """In-memory repository for dual-precision (W4/W2) model parameters."""

    def __init__(
        self,
        *,
        expert_patterns: Optional[Iterable[str]] = None,
        prefer_non_expert: Bitwidth = Bitwidth.W4,
    ) -> None:
        self._expert_patterns = _compile_patterns(expert_patterns)
        self._prefer_non_expert = prefer_non_expert
        self._non_expert: Dict[str, Dict[Bitwidth, torch.Tensor]] = {}
        self._experts: Dict[Bitwidth, Dict[ExpertID, Dict[str, torch.Tensor]]] = {
            Bitwidth.W4: {},
            Bitwidth.W2: {},
        }

    # ------------------------------------------------------------------ Factory
    @classmethod
    def from_state_dicts(
        cls,
        w4_state: Mapping[str, torch.Tensor],
        w2_state: Optional[Mapping[str, torch.Tensor]] = None,
        *,
        expert_patterns: Optional[Iterable[str]] = None,
        prefer_non_expert: Bitwidth = Bitwidth.W4,
    ) -> "DualPrecisionWeights":
        repo = cls(expert_patterns=expert_patterns,
                   prefer_non_expert=prefer_non_expert)
        for name, tensor in w4_state.items():
            repo._register(name, tensor, Bitwidth.W4)
        if w2_state is not None:
            for name, tensor in w2_state.items():
                repo._register(name, tensor, Bitwidth.W2)
        return repo

    @classmethod
    def from_files(
        cls,
        w4_path: str | Path,
        w2_path: str | Path | None = None,
        *,
        expert_patterns: Optional[Iterable[str]] = None,
        prefer_non_expert: Bitwidth = Bitwidth.W4,
        map_location: str | torch.device = "cpu",
    ) -> "DualPrecisionWeights":
        w4_state = cls._load_state_dict_any(w4_path, map_location)
        w2_state = cls._load_state_dict_any(
            w2_path, map_location) if w2_path else None
        return cls.from_state_dicts(
            w4_state,
            w2_state,
            expert_patterns=expert_patterns,
            prefer_non_expert=prefer_non_expert,
        )

    # ------------------------------------------------------------- Registration
    def _register(self, name: str, tensor: torch.Tensor, bitwidth: Bitwidth) -> None:
        tensor_cpu = tensor.detach().cpu().contiguous()
        expert_id = _parse_expert(name, self._expert_patterns)
        if expert_id is None:
            bucket = self._non_expert.setdefault(name, {})
            if bitwidth in bucket and bitwidth is not self._prefer_non_expert:
                return
            bucket[bitwidth] = tensor_cpu
            return

        expert_store = self._experts.setdefault(
            bitwidth, {}).setdefault(expert_id, {})
        expert_store[name] = tensor_cpu

    # ----------------------------------------------------------- Public API
    def indices(self) -> Iterator[ParameterIndex]:
        for name, variants in self._non_expert.items():
            for bitwidth in variants:
                yield ParameterIndex(name=name, bitwidth=bitwidth, expert=None)
        for bitwidth, bundle in self._experts.items():
            for expert_id, params in bundle.items():
                for name in params:
                    yield ParameterIndex(name=name, bitwidth=bitwidth, expert=expert_id)

    def get_tensor(
        self,
        index: ParameterIndex,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if index.expert is None:
            tensor_map = self._non_expert.get(index.name)
            if tensor_map is None:
                raise KeyError(f"Unknown parameter {index.name!r}")
            tensor = tensor_map.get(index.bitwidth)
            if tensor is None:
                if index.bitwidth is not self._prefer_non_expert:
                    tensor = tensor_map.get(self._prefer_non_expert)
            if tensor is None:
                available = ", ".join(bit.value for bit in tensor_map)
                raise KeyError(
                    f"Parameter {index.name!r} missing bitwidth {index.bitwidth.value}. "
                    f"Available: {available}"
                )
        else:
            expert_params = self._experts.get(
                index.bitwidth, {}).get(index.expert)
            if expert_params is None:
                raise KeyError(
                    f"Expert {index.expert} missing precision {index.bitwidth.value}"
                )
            try:
                tensor = expert_params[index.name]
            except KeyError as exc:
                raise KeyError(
                    f"Parameter {index.name!r} not found for expert {index.expert}"
                ) from exc

        return tensor if device is None else tensor.to(device)

    def expert_bundle(self, expert: ExpertID, bitwidth: Bitwidth) -> ExpertTensorBundle:
        params = self._experts.get(bitwidth, {}).get(expert)
        if not params:
            raise KeyError(
                f"No weights for expert {expert} at precision {bitwidth.value}")
        return ExpertTensorBundle(dict(params))

    def non_expert_state(self, *, bitwidth: Optional[Bitwidth] = None) -> Dict[str, torch.Tensor]:
        selected = bitwidth or self._prefer_non_expert
        state: Dict[str, torch.Tensor] = {}
        for name, variants in self._non_expert.items():
            tensor = variants.get(selected)
            if tensor is None:
                tensor = variants.get(self._prefer_non_expert)
            if tensor is None:
                tensor = next(iter(variants.values()))
            state[name] = tensor
        return state

    def materialize_state_dict(
        self,
        *,
        expert_precision: Optional[Callable[[ExpertID], Bitwidth]] = None,
    ) -> Dict[str, torch.Tensor]:
        state = self.non_expert_state()
        visited: set[ExpertID] = set()
        for experts in self._experts.values():
            for expert_id in experts:
                if expert_id in visited:
                    continue
                visited.add(expert_id)
                target = expert_precision(
                    expert_id) if expert_precision else Bitwidth.W4
                chosen = self._select_expert_precision(expert_id, target)
                for name, tensor in chosen.items():
                    state[name] = tensor
        return state

    def _select_expert_precision(
        self, expert_id: ExpertID, preferred: Bitwidth
    ) -> Dict[str, torch.Tensor]:
        order = [preferred, Bitwidth.W4, Bitwidth.W2]
        for precision in order:
            params = self._experts.get(precision, {}).get(expert_id)
            if params:
                return params
        return {}

    def release_expert_storage(self) -> None:
        """Drop all expert tensors to reclaim memory (non-expert parameters remain)."""
        self._experts = {
            Bitwidth.W4: {},
            Bitwidth.W2: {},
        }

    @classmethod
    def _load_state_dict_any(
        cls, path: str | Path, map_location: str | torch.device
    ) -> Dict[str, torch.Tensor]:
        if path is None:
            return {}
        resolved = Path(path)
        if resolved.is_dir():
            return cls._load_state_dict_from_dir(resolved, map_location)
        if resolved.exists():
            return cls._load_state_dict_from_file(resolved, map_location)
        raise FileNotFoundError(f"Weight path {resolved} does not exist")

    @classmethod
    def _load_state_dict_from_dir(
        cls, directory: Path, map_location: str | torch.device
    ) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        candidates: list[Path] = []
        for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
            candidates.extend(sorted(directory.glob(pattern)))
        if not candidates:
            raise FileNotFoundError(
                f"No weight shards found under {directory}")
        for shard in candidates:
            shard_state = cls._load_state_dict_from_file(shard, map_location)
            aggregated.update(shard_state)
        return aggregated

    @staticmethod
    def _load_state_dict_from_file(
        path: Path, map_location: str | torch.device
    ) -> Dict[str, torch.Tensor]:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            if not _HAVE_SAFETENSORS:
                raise ImportError(
                    "safetensors is required to load .safetensors files. Install safetensors first."
                )
            state = safetensors_load(path, device=str(
                map_location))  # type: ignore[arg-type]
        else:
            state = torch.load(path, map_location=map_location)

        if isinstance(state, Mapping):
            if "state_dict" in state and isinstance(state["state_dict"], Mapping):
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], Mapping):
                state = state["model"]

        if not isinstance(state, Mapping):
            raise TypeError(f"Expected mapping in {path}, got {type(state)!r}")

        return {str(name): tensor for name, tensor in state.items() if isinstance(tensor, torch.Tensor)}


def random_precision_selector(prob_low_precision: float) -> Callable[[ExpertID], Bitwidth]:
    import random

    def selector(_: ExpertID) -> Bitwidth:
        return Bitwidth.W2 if random.random() < prob_low_precision else Bitwidth.W4

    return selector


class InMemoryWeightStore:
    """WeightStore abstraction backed by DualPrecisionWeights."""

    def __init__(self, weights: DualPrecisionWeights, bitwidth: Bitwidth) -> None:
        self._weights = weights
        self._bitwidth = bitwidth

    def byte_size(self, expert: ExpertID, bitwidth: Bitwidth) -> int:
        bundle = self._weights.expert_bundle(expert, bitwidth)
        return bundle.nbytes

    def load(self, expert: ExpertID, bitwidth: Bitwidth) -> ExpertTensorBundle:
        return self._weights.expert_bundle(expert, bitwidth)

    def persist(self, expert: ExpertID, bitwidth: Bitwidth, bundle: ExpertTensorBundle) -> None:
        store = self._weights._experts.setdefault(
            bitwidth, {}).setdefault(expert, {})
        store.update(bundle.tensors)

    @property
    def supports_prefetch(self) -> bool:
        return False


class SSDWeightStore:
    """WeightStore backed by an SSD payload with a DRAM cache."""

    def __init__(
        self,
        data_path: Path | str,
        index: Dict[str, Dict[str, int]],
        cache_limit_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        self._data_path = Path(data_path)
        self._index = dict(index)
        self._cache: "OrderedDict[str, ExpertTensorBundle]" = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = max(int(cache_limit_bytes), 0)
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(expert: ExpertID, bitwidth: Bitwidth) -> str:
        return f"{expert.layer}:{expert.idx}:{bitwidth.value}"

    def byte_size(self, expert: ExpertID, bitwidth: Bitwidth) -> int:
        entry = self._index.get(self._make_key(expert, bitwidth))
        if entry is None:
            raise KeyError(
                f"Expert {expert} missing precision {bitwidth.value}")
        return int(entry["size"])

    def load(self, expert: ExpertID, bitwidth: Bitwidth) -> ExpertTensorBundle:
        key = self._make_key(expert, bitwidth)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return ExpertTensorBundle(dict(cached.tensors))
        bundle = self.prefetch(expert, bitwidth)
        return ExpertTensorBundle(dict(bundle.tensors))

    def prefetch(self, expert: ExpertID, bitwidth: Bitwidth) -> ExpertTensorBundle:
        key = self._make_key(expert, bitwidth)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        entry = self._index.get(key)
        if entry is None:
            raise KeyError(
                f"Expert {expert} missing precision {bitwidth.value}")

        offset = int(entry["offset"])
        size = int(entry["size"])
        if size <= 0:
            raise ValueError(
                f"Invalid payload size {size} for expert {expert}")

        with self._data_path.open("rb") as handle:
            handle.seek(offset)
            blob = handle.read(size)

        buffer = io.BytesIO(blob)
        tensors = torch.load(buffer, map_location="cpu")
        bundle = ExpertTensorBundle(
            {str(name): tensor for name, tensor in tensors.items()}
        )
        self._insert_cache(key, bundle)
        return bundle

    def persist(self, expert: ExpertID, bitwidth: Bitwidth, bundle: ExpertTensorBundle) -> None:
        key = self._make_key(expert, bitwidth)
        self._insert_cache(key, bundle)

    def _insert_cache(self, key: str, bundle: ExpertTensorBundle) -> None:
        nbytes = bundle.nbytes
        with self._lock:
            self._cache[key] = bundle
            self._cache.move_to_end(key)
            self._cache_bytes += nbytes

            while self._cache_limit and self._cache_bytes > self._cache_limit and self._cache:
                _, old_bundle = self._cache.popitem(last=False)
                self._cache_bytes -= old_bundle.nbytes

    @property
    def supports_prefetch(self) -> bool:
        return True


__all__ = [
    "DualPrecisionWeights",
    "ExpertTensorBundle",
    "InMemoryWeightStore",
    "SSDWeightStore",
    "ParameterIndex",
]
