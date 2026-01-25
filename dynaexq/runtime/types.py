from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .weights import ExpertTensorBundle


class Bitwidth(str, Enum):
    """Supported expert weight precisions."""

    W4 = "W4"
    W2 = "W2"


class ResidencyLocation(str, Enum):
    """Where an expert currently resides."""

    HBM = "HBM"
    DRAM = "DRAM"
    SSD = "SSD"


@dataclass(frozen=True, slots=True)
class ExpertID:
    """Unique identifier for an expert within a model."""

    layer: int
    idx: int

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ExpertID(layer={self.layer}, idx={self.idx})"


@dataclass(slots=True)
class Residency:
    """Metadata describing where an expert's weights currently live."""

    bitwidth: Bitwidth
    location: ResidencyLocation
    hbm_ptr: Optional[int] = None
    bytes: int = 0
    last_used_ts: float = 0.0
    pending: bool = False
    tags: dict[str, str] = field(default_factory=dict)
    tensor_bundle: Optional["ExpertTensorBundle"] = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            "Residency("
            f"bitwidth={self.bitwidth}, location={self.location}, "
            f"bytes={self.bytes}, last_used_ts={self.last_used_ts}, "
            f"pending={self.pending})"
        )


__all__ = [
    "Bitwidth",
    "ExpertID",
    "Residency",
    "ResidencyLocation",
]
