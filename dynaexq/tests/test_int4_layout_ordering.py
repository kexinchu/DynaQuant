from __future__ import annotations

from dataclasses import dataclass

import torch

from dynaexq.core.quant import QuantFormat
from dynaexq.core.transition_engine import TransitionEngine
import dynaexq.core.transition_engine as transition_module


@dataclass
class _FakeQWeight:
    region: torch.Tensor
    is_cuda: bool = True


@dataclass
class _FakePacked:
    qweight: _FakeQWeight
    nbytes: int = 4
    resident_nbytes: int = 6
    fmt: QuantFormat = QuantFormat.INT4
    int4pack_weight: torch.Tensor | None = None
    int4pack_scales_and_zeros: torch.Tensor | None = None
    canonical_valid: bool = True


def test_all_multimatrix_int4_slots_are_prepared_before_pool_overwrite(
    monkeypatch,
):
    storage = torch.tensor(
        [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
        dtype=torch.uint8,
    )
    resident = {
        "first": _FakePacked(_FakeQWeight(storage[0:4])),
        "second": _FakePacked(_FakeQWeight(storage[4:8])),
    }
    source = {
        name: _FakePacked(_FakeQWeight(torch.empty(0, dtype=torch.uint8)))
        for name in resident
    }
    observed_canonical_regions: list[list[int]] = []

    def fake_prepare(item):
        observed_canonical_regions.append(item.qweight.region.tolist())
        marker = len(observed_canonical_regions) + 10
        return (
            torch.full((4,), marker, dtype=torch.uint8),
            torch.full((2,), marker, dtype=torch.uint8),
        )

    monkeypatch.setattr(transition_module, "_INT4PACK_MM_AVAILABLE", True)
    monkeypatch.setattr(transition_module, "_prepare_int4pack_mm", fake_prepare)

    TransitionEngine._materialize_kernel_caches(
        source,
        resident,
        storage,
    )

    # Slot 0's six-byte native layout overlaps bytes 4:6, where slot 1's
    # canonical payload began. Slot 1 must therefore have been converted
    # before any native write occurred.
    assert observed_canonical_regions == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ]
    assert storage.tolist() == [11] * 6 + [12] * 6
